"""Weight loading utilities for VideoPrism MLX models.

Handles conversion from Flax weight format to MLX model structure.
"""

import mlx.core as mx
import numpy as np
from typing import Dict


def convert_weight_key(flax_key: str) -> str:
    """Convert Flax weight key to MLX module path.
    
    Args:
        flax_key: Flax-style key with '/' separators
    
    Returns:
        MLX-style key with '.' separators
    
    Example:
        >>> convert_weight_key("vision_encoder/spatial_ln/bias")
        "vision_encoder.spatial_ln.bias"
    """
    # Convert separators
    key = flax_key.replace('/', '.')
    
    # Handle specific parameter name mappings
    # TrainablePositionalEmbedding: weight -> emb_var
    if 'pos_emb.weight' in key:
        key = key.replace('pos_emb.weight', 'pos_emb.emb_var')
    
    # Embedding: weight -> emb_var
    if 'token_emb.weight' in key:
        key = key.replace('token_emb.weight', 'token_emb.emb_var')
    
    # AttentionPooling: The learnable query parameter (not part of attention sublayer)
    if 'pooling_attention_query' in key:
        key = key.replace('contrastive_vision_pooler.pooling_attention_query', 'contrastive_vision_pooler.query')
    
    # AttentionPooling attention sublayer: pooling_attention → attention
    if 'contrastive_vision_pooler.pooling_attention.' in key:
        key = key.replace('contrastive_vision_pooler.pooling_attention.', 'contrastive_vision_pooler.attention.')
    
    # AttentionPooling layer norm
    if 'pooling_attention_layer_norm' in key:
        key = key.replace('pooling_attention_layer_norm', 'layer_norm')
    
    # LayerNorm: Map to MLX's internal structure (all LayerNorm modules have .norm submodule)
    # Match any _ln suffix (like unimodal_ln, spatial_ln, temporal_ln)
    if (key.endswith('.bias') or key.endswith('.weight')) and ('_ln.' in key or key.endswith('_ln.bias') or key.endswith('_ln.weight') or '.ln1.' in key or '.ln2.' in key or '.layer_norm.' in key):
        if key.endswith('.bias'):
            key = key.replace('.bias', '.norm.bias')
        if key.endswith('.weight'):
            key = key.replace('.weight', '.norm.weight')
    
    # Patch projection: Flax has patch_projection/linear/*, MLX has patch_projection.*
    if 'patch_projection.linear.' in key:
        key = key.replace('patch_projection.linear.', 'patch_projection.')
    
    # Transformer layer naming: Flax uses different names than MLX
    # Flax: ff_layer, self_attention, layer_norm
    # MLX: ffn, attention, ln1/ln2
    if '.self_attention.' in key:
        key = key.replace('.self_attention.', '.attention.')
    
    # FFN renaming - but we need to be careful about nested layer_norm
    # Pattern: layers/N/ff_layer/...  →  layers/N/ffn/...
    # Also: ffn_layer1 → ffn.fc1, ffn_layer2 → ffn.fc2
    # layer_norm inside ff_layer → goes up to layer level as ln2
    if '.ff_layer.' in key:
        parts = key.split('.')
        new_parts = []
        in_ff_layer = False
        skip_to_layer_level = False
        
        for i, part in enumerate(parts):
            if part == 'ff_layer':
                in_ff_layer = True
                # Don't add ff_layer yet, we'll decide based on what comes next
                continue
            elif part == 'ffn_layer1' and in_ff_layer:
                # TransformerFeedForward has nested FeedForward
                new_parts.extend(['ffn', 'ffn', 'fc1'])
            elif part == 'ffn_layer2' and in_ff_layer:
                new_parts.extend(['ffn', 'ffn', 'fc2'])
            elif part == 'layer_norm' and in_ff_layer:
                # This layer_norm should be at the layer level, not inside ffn
                # Find the layer index and insert ln2 there
                # Parts before ff_layer stay, then we add ln2
                new_parts.append('ln2')
                skip_to_layer_level = True
            elif part == 'linear' and in_ff_layer:
                # Skip 'linear' as fc1/fc2 are directly Linear layers
                continue
            else:
                new_parts.append(part)
        key = '.'.join(new_parts)
    
    # First layer norm (not in ff_layer) should be ln1
    # Pattern: layers/N/layer_norm/...  →  layers/N/ln1/...
    # But NOT for contrastive_vision_pooler which uses layer_norm directly
    if '.layer_norm.' in key and '.ffn.' not in key and 'contrastive_vision_pooler' not in key:
        key = key.replace('.layer_norm.', '.ln1.')
    
    return key


def reshape_attention_weights(weights: Dict[str, mx.array], prefix: str) -> Dict[str, mx.array]:
    """Reshape Flax attention weights to MLX MultiHeadAttention format.
    
    Flax format (per head):
        query/w: (model_dim, num_heads, head_dim)
        query/b: (num_heads, head_dim)
        key/w: (model_dim, num_heads, head_dim)
        key/b: (num_heads, head_dim)
        value/w: (model_dim, num_heads, head_dim)
        value/b: (num_heads, head_dim)
        post/w: (model_dim, num_heads, head_dim)
        post/b: (model_dim,)
    
    MLX format:
        query_proj.weight: (model_dim, model_dim)
        query_proj.bias: (model_dim,)
        key_proj.weight: (model_dim, model_dim)
        key_proj.bias: (model_dim,)
        value_proj.weight: (model_dim, model_dim)
        value_proj.bias: (model_dim,)
        out_proj.weight: (model_dim, model_dim)
        out_proj.bias: (model_dim,)
    
    Args:
        weights: Dictionary of all weights
        prefix: Prefix for the attention layer (e.g., "vision_encoder.spatial_encoder...")
    
    Returns:
        Dictionary with reshaped attention weights
    """
    reshaped = {}
    
    # Extract attention weights (using MLX naming: .attention. not .self_attention.)
    q_w_key = f"{prefix}.attention.query.w"
    q_b_key = f"{prefix}.attention.query.b"
    k_w_key = f"{prefix}.attention.key.w"
    k_b_key = f"{prefix}.attention.key.b"
    v_w_key = f"{prefix}.attention.value.w"
    v_b_key = f"{prefix}.attention.value.b"
    post_w_key = f"{prefix}.attention.post.w"
    post_b_key = f"{prefix}.attention.post.b"
    
    if q_w_key not in weights:
        return {}
    
    q_w = weights[q_w_key]  # (model_dim, num_heads, head_dim)
    q_b = weights[q_b_key]  # (num_heads, head_dim)
    k_w = weights[k_w_key]
    k_b = weights[k_b_key]
    v_w = weights[v_w_key]
    v_b = weights[v_b_key]
    post_w = weights[post_w_key]
    post_b = weights[post_b_key]  # (model_dim,)
    
    model_dim, num_heads, head_dim = q_w.shape
    
    # Reshape: (model_dim, num_heads, head_dim) → (model_dim, total_dim)
    # Flax attention weights are ALREADY in MLX Linear format (out_features, in_features) after reshape
    # No transpose needed for attention weights!
    # DotProductAttention uses q_proj, k_proj, v_proj
    # nn.MultiHeadAttention (pooling) uses query_proj, key_proj, value_proj
    total_dim = num_heads * head_dim
    is_pooling = 'contrastive_vision_pooler' in prefix
    
    if is_pooling:
        # Pooling attention: Q/K/V transpose (768→3072), out_proj no transpose (3072→768)
        reshaped[f"{prefix}.attention.query_proj.weight"] = q_w.reshape(model_dim, total_dim).T
        reshaped[f"{prefix}.attention.key_proj.weight"] = k_w.reshape(model_dim, total_dim).T
        reshaped[f"{prefix}.attention.value_proj.weight"] = v_w.reshape(model_dim, total_dim).T
        reshaped[f"{prefix}.attention.out_proj.weight"] = post_w.reshape(model_dim, total_dim)  # No transpose!
        
        reshaped[f"{prefix}.attention.query_proj.bias"] = q_b.reshape(total_dim)
        reshaped[f"{prefix}.attention.key_proj.bias"] = k_b.reshape(total_dim)
        reshaped[f"{prefix}.attention.value_proj.bias"] = v_b.reshape(total_dim)
        reshaped[f"{prefix}.attention.out_proj.bias"] = post_b
    else:
        # Use short names for DotProductAttention
        reshaped[f"{prefix}.attention.q_proj.weight"] = q_w.reshape(model_dim, total_dim)
        reshaped[f"{prefix}.attention.k_proj.weight"] = k_w.reshape(model_dim, total_dim)
        reshaped[f"{prefix}.attention.v_proj.weight"] = v_w.reshape(model_dim, total_dim)
        reshaped[f"{prefix}.attention.out_proj.weight"] = post_w.reshape(model_dim, total_dim)
        
        reshaped[f"{prefix}.attention.q_proj.bias"] = q_b.reshape(total_dim)
        reshaped[f"{prefix}.attention.k_proj.bias"] = k_b.reshape(total_dim)
        reshaped[f"{prefix}.attention.v_proj.bias"] = v_b.reshape(total_dim)
        reshaped[f"{prefix}.attention.out_proj.bias"] = post_b
    
    return reshaped


def unflatten_dict(flat_dict: Dict[str, mx.array]) -> Dict:
    """Convert flat dictionary with dot notation to nested dictionary.
    
    Handles both dict and list structures (e.g., layers.0.weight becomes layers[0]['weight']).
    
    Args:
        flat_dict: Dictionary with keys like "vision_encoder.spatial_ln.norm.bias"
    
    Returns:
        Nested dictionary structure
    """
    nested = {}
    
    for key, value in flat_dict.items():
        parts = key.split('.')
        current = nested
        
        for i, part in enumerate(parts[:-1]):
            # Check if this part is a numeric index (indicating a list)
            if part.isdigit():
                # Parent should be a list
                parent_key = parts[i-1]
                idx = int(part)
                
                # Ensure parent is a list
                if not isinstance(current, list):
                    # This shouldn't happen with correct structure
                    raise ValueError(f"Expected list for {parts[:i]}, got {type(current)}")
                
                # Extend list if needed
                while len(current) <= idx:
                    current.append({})
                
                current = current[idx]
            else:
                # Regular dict key
                if part not in current:
                    # Check if next part is numeric (indicating we need a list)
                    if i + 1 < len(parts) - 1 and parts[i + 1].isdigit():
                        current[part] = []
                    else:
                        current[part] = {}
                current = current[part]
        
        # Set the final value
        final_key = parts[-1]
        if isinstance(current, dict):
            current[final_key] = value
        else:
            raise ValueError(f"Cannot set {final_key} in {type(current)}")
    
    return nested


def load_and_convert_weights(weights_dict: Dict[str, mx.array]) -> Dict:
    """Convert Flax-format weights to MLX model format.
    
    Args:
        weights_dict: Dictionary of Flax weights (with '/' separators and Flax structure)
    
    Returns:
        Nested dictionary of MLX weights ready for model.update()
    """
    mlx_weights_flat = {}
    
    # Create temporary dict with all keys converted to MLX format
    temp_weights = {convert_weight_key(k): v for k, v in weights_dict.items()}
    
    # Find all unique attention layer prefixes (after conversion)
    attention_prefixes = set()
    for mlx_key in temp_weights.keys():
        # Look for query.w as indicator of attention layer (after conversion to .attention.)
        if '.attention.query.w' in mlx_key:
            # Extract prefix (everything before .attention)
            prefix = mlx_key.split('.attention.')[0]
            attention_prefixes.add(prefix)
    
    # Process attention layers: reshape Q/K/V/Post weights
    for prefix in attention_prefixes:
        reshaped = reshape_attention_weights(temp_weights, prefix)
        mlx_weights_flat.update(reshaped)
    
    # Add all non-attention weights
    for mlx_key, value in temp_weights.items():
        # Skip raw attention weights (q/k/v/post) as they've been reshaped
        if ('.attention.query.' in mlx_key or '.attention.key.' in mlx_key or 
            '.attention.value.' in mlx_key or '.attention.post.' in mlx_key):
            continue
        # Skip per_dim_scale (Flax-specific, not used in MLX)
        if '.per_dim_scale' in mlx_key:
            continue
        
        # Transpose linear layer weights: Flax uses (in, out), MLX uses (out, in)
        if mlx_key.endswith('.weight') and len(value.shape) == 2:
            # This is a 2D weight matrix (linear layer)
            value = value.T
        
        mlx_weights_flat[mlx_key] = value
    
    print(f"Converted {len(mlx_weights_flat)} weight tensors for MLX model")
    
    # Convert flat dictionary to nested structure for model.update()
    mlx_weights_nested = unflatten_dict(mlx_weights_flat)
    
    return mlx_weights_nested


def print_weight_summary(weights_dict: Dict[str, mx.array], max_keys: int = 10):
    """Print summary of weights for debugging.
    
    Args:
        weights_dict: Dictionary of weights
        max_keys: Maximum number of keys to print
    """
    print(f"\nWeight Dictionary Summary:")
    print(f"  Total keys: {len(weights_dict)}")
    print(f"  Sample keys (first {max_keys}):")
    for i, (key, value) in enumerate(sorted(weights_dict.items())[:max_keys]):
        shape_str = f"{value.shape}" if hasattr(value, 'shape') else "no shape"
        print(f"    {key}: {shape_str}")
    if len(weights_dict) > max_keys:
        print(f"    ... and {len(weights_dict) - max_keys} more")
