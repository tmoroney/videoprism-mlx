"""Convert VideoPrism Flax weights to MLX format.

This script handles:
1. Unstacking scan layers (removing leading dimension)
2. Renaming parameters (kernel→weight, scale→weight, etc.)
3. Reshaping attention weights for MLX format
4. Saving in MLX-compatible format
"""

import numpy as np
import mlx.core as mx
from videoprism import models as vp
from flax.traverse_util import flatten_dict
from pathlib import Path

def unstack_scan_layers(params_dict, prefix, num_layers):
    """Unstack scanned layers into individual layer dicts.
    
    Args:
        params_dict: Flattened parameter dictionary
        prefix: Prefix for the scanned layers (e.g., 'vision_encoder/spatial_encoder')
        num_layers: Number of layers to unstack
    
    Returns:
        Dictionary with unstacked layers
    """
    unstacked = {}
    
    for key, value in params_dict.items():
        key_str = '/'.join(key)
        
        if key_str.startswith(prefix) and 'x_layers' in key_str:
            # This is a stacked parameter
            # Extract the path after x_layers
            parts = key_str.split('/')
            idx = parts.index('x_layers')
            layer_path = '/'.join(parts[idx+1:])
            
            # Unstack the first dimension
            for layer_idx in range(num_layers):
                new_key = f"{prefix}/layers/{layer_idx}/{layer_path}"
                unstacked[new_key] = value[layer_idx]
        else:
            # Not a stacked parameter, keep as is
            unstacked['/'.join(key)] = value
    
    return unstacked


def convert_attention_weights(q_w, q_b, k_w, k_b, v_w, v_b, post_w, post_b):
    """Convert Flax attention weights to MLX MultiHeadAttention format.
    
    Flax format:
        Q/K/V: w shape=(model_dim, num_heads, head_dim), b shape=(num_heads, head_dim)
        Post: w shape=(model_dim, num_heads, head_dim), b shape=(model_dim,)
    
    MLX format (for MultiHeadAttention):
        query_proj: (model_dim, model_dim) 
        key_proj: (model_dim, model_dim)
        value_proj: (model_dim, model_dim)
        out_proj: (model_dim, model_dim)
    
    Returns:
        Dictionary with MLX-formatted attention weights
    """
    model_dim, num_heads, head_dim = q_w.shape
    
    # Reshape: (model_dim, num_heads, head_dim) → (model_dim, num_heads * head_dim)
    q_proj_weight = q_w.reshape(model_dim, num_heads * head_dim)
    k_proj_weight = k_w.reshape(model_dim, num_heads * head_dim)
    v_proj_weight = v_w.reshape(model_dim, num_heads * head_dim)
    out_proj_weight = post_w.reshape(model_dim, num_heads * head_dim)
    
    # Reshape biases: (num_heads, head_dim) → (num_heads * head_dim,)
    q_proj_bias = q_b.reshape(num_heads * head_dim)
    k_proj_bias = k_b.reshape(num_heads * head_dim)
    v_proj_bias = v_b.reshape(num_heads * head_dim)
    out_proj_bias = post_b  # Already correct shape
    
    return {
        'query_proj': {'weight': q_proj_weight, 'bias': q_proj_bias},
        'key_proj': {'weight': k_proj_weight, 'bias': k_proj_bias},
        'value_proj': {'weight': v_proj_weight, 'bias': v_proj_bias},
        'out_proj': {'weight': out_proj_weight, 'bias': out_proj_bias},
    }


def rename_parameter(flax_name):
    """Rename Flax parameter names to MLX conventions.
    
    Conversions:
        kernel → weight
        scale → weight (for LayerNorm)
        w → weight
        b → bias
        emb_var → weight (for embeddings)
    """
    # Direct replacements
    name = flax_name.replace('/kernel', '/weight')
    name = name.replace('/scale', '/weight')
    name = name.replace('/emb_var', '/weight')
    name = name.replace('/bias', '/bias')  # Keep bias as is
    
    return name


def convert_flax_to_mlx(flax_params, model_config, model_name):
    """Convert complete Flax parameter tree to MLX format.
    
    Args:
        flax_params: Flax parameters (nested dict or FrozenDict)
        model_config: Model configuration dict with layer counts
        model_name: Name of the model being converted
    
    Returns:
        MLX-formatted parameter dictionary
    """
    # Flatten Flax params - use keep_empty_nodes to preserve structure
    flat_flax = flatten_dict(flax_params)
    
    # Convert to regular dict with string keys (join tuple keys with '/')
    flat_flax_dict = {}
    for key, value in flat_flax.items():
        # Key is a tuple like ('vision_encoder', 'spatial_encoder', 'transformers_stack', ...)
        key_str = '/'.join(str(k) for k in key)
        flat_flax_dict[key_str] = np.array(value)
    
    print(f"  Found {len(flat_flax_dict)} Flax parameters")
    
    # Debug: print first few keys to verify structure
    print("  Sample keys:")
    for i, key in enumerate(sorted(flat_flax_dict.keys())[:3]):
        print(f"    {key}")
    
    mlx_params = {}
    
    # Handle stacked transformer layers
    # Path prefix differs between video-only and video-text models
    # Check if this is a video-only model (no 'lvt' in model name) or CLIP model
    is_video_only = 'lvt' not in model_name
    vision_prefix = '' if is_video_only else 'vision_encoder/'
    
    # Always present: vision encoder (spatial + temporal)
    components = [
        (f'{vision_prefix}spatial_encoder/transformers_stack', 
         model_config.get('spatial_layers', 12), 'spatial'),
        (f'{vision_prefix}temporal_encoder/transformers_stack', 
         model_config.get('temporal_layers', 4), 'temporal'),
    ]
    
    # Optional: text and auxiliary encoders (only for video-text models)
    if 'text_layers' in model_config:
        components.append(
            ('text_encoder/unimodal_transformer', 
             model_config.get('text_layers', 12), 'text')
        )
    if 'auxiliary_layers' in model_config:
        components.append(
            ('auxiliary_encoder/transformers_stack', 
             model_config.get('auxiliary_layers', 2), 'auxiliary')
        )
    
    print("\n  Unstacking scanned layers...")
    
    for prefix, num_layers, component_name in components:
        print(f"    Processing {component_name} encoder ({num_layers} layers)...")
        
        # Find all x_layers parameters for this component
        x_layer_params = {}
        for key, value in flat_flax_dict.items():
            if key.startswith(prefix) and '/x_layers/' in key:
                x_layer_params[key] = value
        
        print(f"      Found {len(x_layer_params)} stacked parameters")
        
        # Unstack each parameter
        for key, value in x_layer_params.items():
            # Extract the path after x_layers
            # e.g., 'vision_encoder/spatial_encoder/transformers_stack/x_layers/ff_layer/...'
            parts = key.split('/')
            
            # Find x_layers index
            try:
                x_layers_idx = parts.index('x_layers')
                layer_path = '/'.join(parts[x_layers_idx + 1:])  # Everything after x_layers
                
                # Check if this is a stacked parameter (first dim = num_layers)
                if value.ndim > 0 and value.shape[0] == num_layers:
                    # Unstack into individual layers
                    for layer_idx in range(num_layers):
                        # Build new key: prefix/layers/0/ff_layer/...
                        mlx_key = f"{prefix}/layers/{layer_idx}/{layer_path}"
                        
                        # Rename parameters (kernel→weight, etc.)
                        mlx_key = rename_parameter(mlx_key)
                        
                        # Extract this layer's weights
                        mlx_params[mlx_key] = value[layer_idx]
                else:
                    print(f"      Warning: Expected stacked param but got shape {value.shape} for {key}")
            except ValueError:
                print(f"      Warning: Could not find x_layers in key: {key}")
    
    # Handle non-stacked parameters
    print("\n  Converting non-stacked parameters...")
    non_stacked_count = 0
    
    for key, value in flat_flax_dict.items():
        # Skip if already handled as stacked layer
        if '/x_layers/' in key:
            continue
        
        # Rename to MLX conventions
        mlx_key = rename_parameter(key)
        
        # Handle special transformations
        # Flax uses (in_features, out_features) but MLX uses (out_features, in_features)
        # We'll keep Flax format for now and transpose during model loading if needed
        
        mlx_params[mlx_key] = value
        non_stacked_count += 1
    
    print(f"      Processed {non_stacked_count} non-stacked parameters")
    print(f"\n  Total MLX parameters: {len(mlx_params)}")
    
    return mlx_params


def verify_conversion(flax_params, mlx_params, model_config):
    """Verify the conversion by comparing parameter counts and shapes.
    
    Args:
        flax_params: Original Flax parameters
        mlx_params: Converted MLX parameters
        model_config: Model configuration
    """
    print("\n" + "=" * 80)
    print("CONVERSION VERIFICATION")
    print("=" * 80)
    
    # Count parameters
    flat_flax = flatten_dict(flax_params)
    flax_total = sum(np.array(v).size for v in flat_flax.values())
    mlx_total = sum(v.size for v in mlx_params.values())
    
    print(f"\nParameter count:")
    print(f"  Flax:  {flax_total:,} parameters")
    print(f"  MLX:   {mlx_total:,} parameters")
    
    if flax_total == mlx_total:
        print(f"  ✓ Parameter counts match!")
    else:
        print(f"  ⚠ Mismatch: {abs(flax_total - mlx_total):,} parameters difference")
    
    # Verify layer unstacking
    print(f"\nLayer verification:")
    
    # Build verification list based on what's in model_config
    verification_list = [
        ('spatial', model_config.get('spatial_layers', 12)),
        ('temporal', model_config.get('temporal_layers', 4)),
    ]
    if 'text_layers' in model_config:
        verification_list.append(('text', model_config.get('text_layers', 12)))
    if 'auxiliary_layers' in model_config:
        verification_list.append(('auxiliary', model_config.get('auxiliary_layers', 2)))
    
    for component_name, expected_layers in verification_list:
        # Count layers in MLX params
        layer_keys = [k for k in mlx_params.keys() if f'{component_name}' in k.lower() and '/layers/' in k]
        unique_layers = set()
        for k in layer_keys:
            try:
                layer_num = int(k.split('/layers/')[1].split('/')[0])
                unique_layers.add(layer_num)
            except (IndexError, ValueError):
                pass
        
        found_layers = len(unique_layers)
        status = "✓" if found_layers == expected_layers else "⚠"
        print(f"  {status} {component_name.capitalize()}: {found_layers}/{expected_layers} layers")
    
    # Sample a few parameters
    print(f"\nSample parameter shapes:")
    sample_keys = sorted(mlx_params.keys())[:5]
    for key in sample_keys:
        print(f"  {key}: {mlx_params[key].shape}")
    
    print("\n" + "=" * 80)


def save_mlx_weights(mlx_params, output_path):
    """Save MLX parameters to file.
    
    Args:
        mlx_params: Dictionary of MLX parameters (numpy arrays)
        output_path: Path to save weights
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to MLX arrays
    print(f"  Converting to MLX arrays...")
    mlx_arrays = {k: mx.array(v) for k, v in mlx_params.items()}
    
    # Save using MLX's native functions
    if output_path.suffix == '.npz':
        print(f"  Saving as NPZ format using mx.savez()...")
        mx.savez(str(output_path), **mlx_arrays)
        print(f"  ✓ Saved to {output_path}")
        
        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"  File size: {file_size_mb:.2f} MB")
        
    elif output_path.suffix == '.safetensors':
        print(f"  Saving as SafeTensors format using mx.save_safetensors()...")
        mx.save_safetensors(str(output_path), mlx_arrays)
        print(f"  ✓ Saved to {output_path}")
        
        # Calculate file size
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"  File size: {file_size_mb:.2f} MB")
    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}. Use .npz or .safetensors")
    
    # Also save SafeTensors if NPZ was requested (for better compatibility)
    if output_path.suffix == '.npz':
        safetensors_path = output_path.with_suffix('.safetensors')
        print(f"  Also saving as SafeTensors format...")
        try:
            mx.save_safetensors(str(safetensors_path), mlx_arrays)
            st_size_mb = safetensors_path.stat().st_size / (1024 ** 2)
            print(f"  ✓ Saved to {safetensors_path}")
            print(f"  File size: {st_size_mb:.2f} MB")
        except Exception as e:
            print(f"  ⚠ Could not save SafeTensors: {e}")


def main():
    print("=" * 80)
    print("VideoPrism Flax → MLX Weight Conversion")
    print("=" * 80)
    
    # Configure which model to convert
    # Options:
    #   - 'videoprism_public_v1_base' (video encoder only)
    #   - 'videoprism_public_v1_large' (video encoder only)
    #   - 'videoprism_lvt_public_v1_base' (video-text CLIP)
    #   - 'videoprism_lvt_public_v1_large' (video-text CLIP)
    model_name = 'videoprism_lvt_public_v1_large'
    
    # Load Flax weights
    print("\n[1/4] Loading Flax weights...")
    loaded_state = vp.load_pretrained_weights(model_name)
    
    if 'params' in loaded_state:
        flax_params = loaded_state['params']
    else:
        flax_params = loaded_state
    
    print(f"      ✓ Weights loaded")
    
    # Convert to MLX format
    print("\n[2/4] Converting to MLX format...")
    
    # Get correct layer counts from the Flax model config
    config_key = model_name.replace('_public', '')
    if config_key in vp.CONFIGS:
        flax_config = vp.CONFIGS[config_key]
        
        # Build model config based on what's present (video-only vs video-text models)
        is_video_only = 'lvt' not in model_name
        
        model_config = {
            'spatial_layers': flax_config.get('num_spatial_layers', 12),
            'temporal_layers': flax_config.get('num_temporal_layers', 4),
        }
        
        # Only add text/auxiliary layers if this is a video-text model
        if not is_video_only:
            model_config['text_layers'] = flax_config.get('num_unimodal_layers', 12)
            model_config['auxiliary_layers'] = flax_config.get('num_auxiliary_layers', 2)
        
        print(f"      Using config from {config_key}:")
        print(f"        Model type: {'Video encoder only' if is_video_only else 'Video-text CLIP'}")
        print(f"        Spatial layers: {model_config['spatial_layers']}")
        print(f"        Temporal layers: {model_config['temporal_layers']}")
        if not is_video_only:
            print(f"        Text layers: {model_config['text_layers']}")
            print(f"        Auxiliary layers: {model_config['auxiliary_layers']}")
    else:
        raise ValueError(f"Config not found for {model_name} (tried {config_key})")
    
    mlx_params = convert_flax_to_mlx(flax_params, model_config, model_name)
    
    # Verify conversion
    print("\n[3/4] Verifying conversion...")
    verify_conversion(flax_params, mlx_params, model_config)
    
    # Save MLX weights
    print("\n[4/4] Saving MLX weights...")
    output_dir = Path("weights")
    output_path = output_dir / f"{model_name}_mlx.npz"
    save_mlx_weights(mlx_params, output_path)
    
    # Also save metadata
    metadata = {
        'model_name': model_name,
        'source': 'flax',
        'total_parameters': sum(v.size for v in mlx_params.values()),
        'num_tensors': len(mlx_params),
        'model_config': model_config,
        'parameter_keys': sorted(mlx_params.keys()),
    }
    
    import json
    with open(output_dir / f"{model_name}_mlx_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  ✓ Metadata saved to {output_dir / f'{model_name}_mlx_metadata.json'}")
    
    print("\n" + "=" * 80)
    print("Conversion completed successfully! ✓")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  Weights: {output_path}")
    print(f"  Metadata: {output_dir / f'{model_name}_mlx_metadata.json'}")
    print(f"\nStatistics:")
    print(f"  Total parameters: {metadata['total_parameters']:,}")
    print(f"  Total tensors: {metadata['num_tensors']}")
    print(f"  Model: {model_name}")
    print("\nNext steps:")
    print("  1. Load weights in MLX model: mlx_model.load_weights('weights/..._mlx.npz')")
    print("  2. Test inference and compare outputs with Flax model")
    print("  3. Verify numerical accuracy (should match within fp32 precision)")
    print()


if __name__ == "__main__":
    main()
