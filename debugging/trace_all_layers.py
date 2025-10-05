"""Trace through all transformer layers to find where divergence accumulates."""

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
import math
from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video


def compare(flax_arr, mlx_arr, name: str):
    """Compare arrays and return correlation."""
    flax_np = np.array(flax_arr)
    mlx_np = np.array(mlx_arr)
    
    if flax_np.shape != mlx_np.shape:
        print(f"  ❌ {name:50s} SHAPE MISMATCH: {flax_np.shape} vs {mlx_np.shape}")
        return 0.0
    
    corr = np.corrcoef(flax_np.flatten(), mlx_np.flatten())[0, 1]
    max_diff = np.max(np.abs(flax_np - mlx_np))
    
    flax_mean = np.mean(flax_np)
    flax_std = np.std(flax_np)
    mlx_mean = np.mean(mlx_np)
    mlx_std = np.std(mlx_np)
    
    status = "✓✓" if corr > 0.99 else "✓ " if corr > 0.9 else "⚠️ " if corr > 0.7 else "❌"
    
    print(f"  {status} {name:50s} corr={corr:6.4f} | max_diff={max_diff:8.4f} | Flax: μ={flax_mean:7.3f} σ={flax_std:6.3f} | MLX: μ={mlx_mean:7.3f} σ={mlx_std:6.3f}")
    
    return corr


def trace_spatial_layers_mlx(model, patches):
    """Trace through all 12 spatial transformer layers in MLX."""
    outputs = {}
    vision = model.vision_encoder
    
    x = patches
    for i in range(12):
        layer = vision.spatial_encoder.transformers_stack.layers[i]
        x = layer(x, paddings=None, atten_mask=None)
        outputs[f'spatial_layer_{i}'] = x
    
    return outputs


def trace_spatial_layers_flax_manual(state, patches):
    """Manually trace through Flax spatial transformer layers."""
    outputs = {}
    
    layer_params = state['params']['vision_encoder']['spatial_encoder']['transformers_stack']['x_layers']
    
    x = patches
    for layer_idx in range(12):
        # Pre-LN
        ln_scale = layer_params['layer_norm']['scale'][layer_idx]
        ln_bias = layer_params['layer_norm']['bias'][layer_idx]
        
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        ln_out = (x - mean) / jnp.sqrt(var + 1e-6)
        ln_out = ln_out * (ln_scale + 1.0) + ln_bias
        
        # Attention
        q_w = layer_params['self_attention']['query']['w'][layer_idx]
        q_b = layer_params['self_attention']['query']['b'][layer_idx]
        k_w = layer_params['self_attention']['key']['w'][layer_idx]
        k_b = layer_params['self_attention']['key']['b'][layer_idx]
        v_w = layer_params['self_attention']['value']['w'][layer_idx]
        v_b = layer_params['self_attention']['value']['b'][layer_idx]
        
        q = jnp.einsum('...d,dhf->...hf', ln_out, q_w) + q_b
        k = jnp.einsum('...d,dhf->...hf', ln_out, k_w) + k_b
        v = jnp.einsum('...d,dhf->...hf', ln_out, v_w) + v_b
        
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        scale = 1.0 / jnp.sqrt(64.0)
        logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        logits = 50.0 * jnp.tanh(logits / 50.0)
        attn_weights = jax.nn.softmax(logits, axis=-1)
        attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))
        
        # Output projection
        out_w = layer_params['self_attention']['post']['w'][layer_idx]
        out_b = layer_params['self_attention']['post']['b'][layer_idx]
        attn_out = jnp.einsum('...hf,dhf->...d', attn_out, out_w) + out_b
        
        # Residual
        x = x + attn_out
        
        # FFN
        ffn_ln_scale = layer_params['ff_layer']['layer_norm']['scale'][layer_idx]
        ffn_ln_bias = layer_params['ff_layer']['layer_norm']['bias'][layer_idx]
        
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        ln_out = (x - mean) / jnp.sqrt(var + 1e-6)
        ln_out = ln_out * (ffn_ln_scale + 1.0) + ffn_ln_bias
        
        # FFN layers
        ffn1_w = layer_params['ff_layer']['ffn_layer1']['linear']['kernel'][layer_idx]
        ffn1_b = layer_params['ff_layer']['ffn_layer1']['linear']['bias'][layer_idx]
        ffn2_w = layer_params['ff_layer']['ffn_layer2']['linear']['kernel'][layer_idx]
        ffn2_b = layer_params['ff_layer']['ffn_layer2']['linear']['bias'][layer_idx]
        
        ffn_out = ln_out @ ffn1_w + ffn1_b
        ffn_out = jax.nn.gelu(ffn_out, approximate=False)
        ffn_out = ffn_out @ ffn2_w + ffn2_b
        
        # Residual
        x = x + ffn_out
        
        outputs[f'spatial_layer_{layer_idx}'] = x
    
    return outputs


def main():
    print("=" * 80)
    print("TRACING ALL SPATIAL TRANSFORMER LAYERS")
    print("=" * 80)
    
    # Load models
    print("\n[1/4] Loading models...")
    flax_state = vp.load_pretrained_weights('videoprism_lvt_public_v1_base')
    mlx_model = models_mlx.load_model('videoprism_lvt_public_v1_base')
    print("✓ Models loaded")
    
    # Load video
    print("\n[2/4] Loading video...")
    video_np = load_video('videoprism/assets/water_bottle_drumming.mp4', num_frames=16, target_size=288)
    video_flax = jnp.array(video_np[None, ...])
    video_mlx = mx.array(video_np[None, ...])
    
    # Prepare patches
    print("\n[3/4] Preparing patches...")
    b, t, h, w, c = video_flax.shape
    bt = b * t
    
    # Flax patches
    inputs_flax = video_flax.reshape(bt, h, w, c)
    patch_size = 18
    num_patches = (h // patch_size) * (w // patch_size)
    
    patches_flax = inputs_flax.reshape(bt, h//patch_size, patch_size, w//patch_size, patch_size, c)
    patches_flax = jnp.transpose(patches_flax, (0, 1, 3, 2, 4, 5))
    patches_flax = patches_flax.reshape(bt, num_patches, patch_size * patch_size * c)
    
    patch_w = flax_state['params']['vision_encoder']['patch_projection']['linear']['kernel']
    patch_b = flax_state['params']['vision_encoder']['patch_projection']['linear']['bias']
    patches_flax = patches_flax @ patch_w + patch_b
    
    spatial_pos = flax_state['params']['vision_encoder']['spatial_pos_emb']['emb_var'][:num_patches]
    position = jnp.arange(num_patches, dtype=jnp.int32)[jnp.newaxis, :]
    one_hot = jax.nn.one_hot(position, num_patches)
    spatial_pos_emb = jnp.einsum('...y,yz->...z', one_hot, spatial_pos)
    patches_flax = patches_flax + spatial_pos_emb
    
    # MLX patches
    vision = mlx_model.vision_encoder
    inputs_mlx = video_mlx.reshape(bt, h, w, c)
    patches_mlx = inputs_mlx.reshape(bt, h//patch_size, patch_size, w//patch_size, patch_size, c)
    patches_mlx = patches_mlx.transpose(0, 1, 3, 2, 4, 5)
    patches_mlx = patches_mlx.reshape(bt, num_patches, patch_size * patch_size * c)
    patches_mlx = vision.patch_projection(patches_mlx)
    spatial_pos_emb_mlx = vision.spatial_pos_emb(num_patches)
    patches_mlx = patches_mlx + spatial_pos_emb_mlx
    
    print(f"✓ Patches prepared: {patches_flax.shape}")
    
    # Trace through layers
    print("\n[4/4] Tracing through spatial transformer layers...")
    print("\n" + "=" * 80)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 80)
    
    flax_outputs = trace_spatial_layers_flax_manual(flax_state, patches_flax)
    mlx_outputs = trace_spatial_layers_mlx(mlx_model, patches_mlx)
    
    print()
    for i in range(12):
        key = f'spatial_layer_{i}'
        corr = compare(flax_outputs[key], mlx_outputs[key], f"Layer {i} output")
        
        if corr < 0.95:
            print(f"\n⚠️  DIVERGENCE DETECTED AT LAYER {i}!")
            if i > 0:
                prev_key = f'spatial_layer_{i-1}'
                prev_corr = compare(flax_outputs[prev_key], mlx_outputs[prev_key], f"Layer {i-1} (previous)")
                print(f"   Correlation dropped from {prev_corr:.4f} to {corr:.4f}")
            break
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
