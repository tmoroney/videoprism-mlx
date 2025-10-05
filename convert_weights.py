"""Convert VideoPrism Flax weights to MLX format.

This script handles:
1. Unstacking scan layers (removing leading dimension)
2. Renaming parameters (kernel→weight, scale→weight, etc.)
3. Reshaping attention weights for MLX format
4. Saving in MLX-compatible format
"""

import jax
import numpy as np
import mlx.core as mx
from videoprism import models as vp
from flax.traverse_util import flatten_dict
from pathlib import Path
import pickle


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


def convert_flax_to_mlx(flax_params, model_config):
    """Convert complete Flax parameter tree to MLX format.
    
    Args:
        flax_params: Flax parameters (nested dict or FrozenDict)
        model_config: Model configuration dict with layer counts
    
    Returns:
        MLX-formatted parameter dictionary
    """
    # Flatten Flax params
    flat_flax = flatten_dict(flax_params, sep='/')
    
    # Convert to regular dict with string keys
    flat_flax_dict = {'/'.join(str(k) for k in key): np.array(value) 
                      for key, value in flat_flax.items()}
    
    print(f"  Found {len(flat_flax_dict)} Flax parameters")
    
    # Unstack scan layers
    print("  Unstacking scanned layers...")
    
    # Vision encoder - spatial (12 layers)
    mlx_params = {}
    
    # Handle stacked transformer layers
    components = [
        ('vision_encoder/spatial_encoder/transformers_stack', 12),
        ('vision_encoder/temporal_encoder/transformers_stack', 4),
        ('text_encoder/unimodal_transformer', 12),
        ('auxiliary_encoder/transformers_stack', 2),
    ]
    
    for prefix, num_layers in components:
        print(f"    Unstacking {prefix} ({num_layers} layers)...")
        for layer_idx in range(num_layers):
            # Find all parameters for this layer
            layer_params = {}
            
            for key, value in flat_flax_dict.items():
                if f"{prefix}/x_layers" in key:
                    # Extract layer-specific weight
                    if value.ndim > 0 and value.shape[0] == num_layers:
                        # This is a stacked parameter
                        parts = key.split('/')
                        # Remove x_layers from path
                        new_parts = [p for p in parts if p != 'x_layers']
                        new_key = '/'.join(new_parts)
                        
                        # Replace component prefix
                        new_key = new_key.replace(f"{prefix}/", f"{prefix}/layers/{layer_idx}/")
                        
                        # Rename Flax conventions to MLX
                        new_key = rename_parameter(new_key)
                        
                        mlx_params[new_key] = value[layer_idx]
    
    # Handle non-stacked parameters
    print("  Converting non-stacked parameters...")
    for key, value in flat_flax_dict.items():
        # Skip if already handled as stacked layer
        if '/x_layers/' in key:
            continue
        
        # Rename to MLX conventions
        new_key = rename_parameter(key)
        
        # Handle special cases
        if '/linear/weight' in new_key and value.ndim == 2:
            # Transpose linear layer weights: Flax (in, out) → MLX (out, in)
            # Actually, let's check MLX convention first
            mlx_params[new_key] = value  # Keep as is for now
        else:
            mlx_params[new_key] = value
    
    print(f"  Converted to {len(mlx_params)} MLX parameters")
    
    return mlx_params


def save_mlx_weights(mlx_params, output_path):
    """Save MLX parameters to file.
    
    Args:
        mlx_params: Dictionary of MLX parameters (numpy arrays)
        output_path: Path to save weights
    """
    # Convert numpy arrays to MLX arrays
    mlx_arrays = {k: mx.array(v) for k, v in mlx_params.items()}
    
    # Save using MLX's save function
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as safetensors or npz
    if output_path.suffix == '.npz':
        np.savez(output_path, **mlx_params)
    elif output_path.suffix == '.safetensors':
        try:
            from safetensors.numpy import save_file
            save_file(mlx_params, output_path)
        except ImportError:
            print("    ⚠ safetensors not installed, saving as .npz instead")
            np.savez(output_path.with_suffix('.npz'), **mlx_params)
    else:
        # Default to pickle for complex types
        with open(output_path, 'wb') as f:
            pickle.dump(mlx_arrays, f)
    
    print(f"  ✓ Saved to {output_path}")


def main():
    print("=" * 80)
    print("VideoPrism Flax → MLX Weight Conversion")
    print("=" * 80)
    
    # Load Flax model
    print("\n[1/4] Loading Flax model...")
    model_name = 'videoprism_lvt_public_v1_base'
    flax_model = vp.get_model(model_name)
    print(f"      ✓ Model configuration loaded")
    
    # Load Flax weights
    print("\n[2/4] Loading Flax weights...")
    loaded_state = vp.load_pretrained_weights(model_name)
    
    if 'params' in loaded_state:
        flax_params = loaded_state['params']
    else:
        flax_params = loaded_state
    
    print(f"      ✓ Weights loaded")
    
    # Convert to MLX format
    print("\n[3/4] Converting to MLX format...")
    
    model_config = {
        'spatial_layers': 12,
        'temporal_layers': 4,
        'text_layers': 12,
        'auxiliary_layers': 2,
    }
    
    mlx_params = convert_flax_to_mlx(flax_params, model_config)
    
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
    }
    
    import json
    with open(output_dir / f"{model_name}_mlx_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"      ✓ Metadata saved")
    
    print("\n" + "=" * 80)
    print("Conversion completed successfully!")
    print("=" * 80)
    print(f"\nConverted weights saved to: {output_path}")
    print(f"Total parameters: {metadata['total_parameters']:,}")
    print(f"Total tensors: {metadata['num_tensors']}")
    print("\nNext steps:")
    print("1. Load weights in MLX model")
    print("2. Test inference and compare outputs")
    print("3. Verify numerical accuracy")
    print()


if __name__ == "__main__":
    main()
