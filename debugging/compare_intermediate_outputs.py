"""Compare intermediate outputs between Flax and MLX models step-by-step.

This script instruments both models to capture activations at the same checkpoints
and identifies exactly where they diverge.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import mlx.core as mx
import numpy as np
import math
from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video
from typing import Dict, List, Tuple


def compare_arrays(flax_arr, mlx_arr, name: str, verbose=True) -> Dict:
    """Compare two arrays and return detailed metrics."""
    flax_np = np.array(flax_arr)
    mlx_np = np.array(mlx_arr)
    
    if flax_np.shape != mlx_np.shape:
        if verbose:
            print(f"  ❌ {name:50s} SHAPE MISMATCH: {flax_np.shape} vs {mlx_np.shape}")
        return {'status': 'shape_mismatch', 'correlation': 0.0}
    
    flax_flat = flax_np.flatten()
    mlx_flat = mlx_np.flatten()
    
    max_diff = np.max(np.abs(flax_flat - mlx_flat))
    mean_diff = np.mean(np.abs(flax_flat - mlx_flat))
    corr = np.corrcoef(flax_flat, mlx_flat)[0, 1]
    
    flax_mean = np.mean(flax_flat)
    flax_std = np.std(flax_flat)
    mlx_mean = np.mean(mlx_flat)
    mlx_std = np.std(mlx_flat)
    
    if verbose:
        if corr > 0.99:
            status = "✓✓"
        elif corr > 0.9:
            status = "✓ "
        elif corr > 0.7:
            status = "⚠️ "
        else:
            status = "❌"
        
        print(f"  {status} {name:50s} corr={corr:6.4f} | max_diff={max_diff:8.4f} | Flax: μ={flax_mean:7.3f} σ={flax_std:6.3f} | MLX: μ={mlx_mean:7.3f} σ={mlx_std:6.3f}")
    
    return {
        'status': 'ok' if corr > 0.9 else 'diverged',
        'correlation': corr,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'flax_mean': flax_mean,
        'flax_std': flax_std,
        'mlx_mean': mlx_mean,
        'mlx_std': mlx_std,
    }


def extract_flax_vision_intermediates(model, state, video):
    """Extract intermediate activations from Flax vision encoder.
    
    This manually replicates the forward pass to capture intermediate values.
    """
    print("\n" + "=" * 80)
    print("EXTRACTING FLAX INTERMEDIATE OUTPUTS")
    print("=" * 80)
    
    intermediates = {}
    
    b, t, h, w, c = video.shape
    bt = b * t
    
    # 1. Input
    intermediates['00_input'] = video
    print(f"\n✓ Input: {video.shape}")
    
    # 2. Reshape for patches
    inputs = video.reshape(bt, h, w, c)
    intermediates['01_reshaped'] = inputs
    
    # 3. Create patches
    patch_size = 18
    num_patches = (h // patch_size) * (w // patch_size)
    
    patches = inputs.reshape(bt, h//patch_size, patch_size, w//patch_size, patch_size, c)
    patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
    patches = patches.reshape(bt, num_patches, patch_size * patch_size * c)
    intermediates['02_patches'] = patches
    print(f"✓ Patches: {patches.shape}")
    
    # 4. Patch projection
    patch_w = state['params']['vision_encoder']['patch_projection']['linear']['kernel']
    patch_b = state['params']['vision_encoder']['patch_projection']['linear']['bias']
    patches = patches @ patch_w + patch_b
    intermediates['03_patch_projected'] = patches
    print(f"✓ Patch projection: mean={jnp.mean(patches):.6f}, std={jnp.std(patches):.6f}")
    
    # 5. Add spatial positional embeddings  
    spatial_pos = state['params']['vision_encoder']['spatial_pos_emb']['emb_var']
    spatial_pos = spatial_pos[:num_patches]
    
    # Flax uses one-hot matmul for embedding lookup
    position = jnp.arange(num_patches, dtype=jnp.int32)[jnp.newaxis, :]
    one_hot = jax.nn.one_hot(position, num_patches)
    spatial_pos_emb = jnp.einsum('...y,yz->...z', one_hot, spatial_pos)
    
    patches = patches + spatial_pos_emb
    intermediates['04_with_spatial_pos'] = patches
    print(f"✓ With spatial pos: mean={jnp.mean(patches):.6f}, std={jnp.std(patches):.6f}")
    
    # 6. Apply first spatial transformer layer manually
    print(f"\n✓ Applying spatial transformer layer 0...")
    
    layer_idx = 0
    layer_params = state['params']['vision_encoder']['spatial_encoder']['transformers_stack']['x_layers']
    
    # Flax uses scan=True, so parameters are stacked with shape (num_layers, ...)
    # Pre-LN - slice to get layer 0
    ln_scale = layer_params['layer_norm']['scale'][layer_idx]  # (768,)
    ln_bias = layer_params['layer_norm']['bias'][layer_idx]     # (768,)
    
    # LayerNorm with +1.0 on scale (Flax convention)
    mean = jnp.mean(patches, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(patches - mean), axis=-1, keepdims=True)
    ln_out = (patches - mean) / jnp.sqrt(var + 1e-6)
    ln_out = ln_out * (ln_scale + 1.0) + ln_bias
    intermediates['05_layer0_ln1'] = ln_out
    print(f"  LN1: mean={jnp.mean(ln_out):.6f}, std={jnp.std(ln_out):.6f}")
    
    # Attention Q, K, V projections - slice to get layer 0
    q_w = layer_params['self_attention']['query']['w'][layer_idx]  # (768, 12, 64)
    q_b = layer_params['self_attention']['query']['b'][layer_idx]  # (12, 64)
    k_w = layer_params['self_attention']['key']['w'][layer_idx]
    k_b = layer_params['self_attention']['key']['b'][layer_idx]
    v_w = layer_params['self_attention']['value']['w'][layer_idx]
    v_b = layer_params['self_attention']['value']['b'][layer_idx]
    
    # Project Q, K, V
    # Flax format: (model_dim, num_heads, head_dim)
    # Need to reshape for matmul: (model_dim, num_heads * head_dim)
    q = jnp.einsum('...d,dhf->...hf', ln_out, q_w) + q_b  # (..., 12, 64)
    k = jnp.einsum('...d,dhf->...hf', ln_out, k_w) + k_b
    v = jnp.einsum('...d,dhf->...hf', ln_out, v_w) + v_b
    
    print(f"  Q: shape={q.shape}, mean={jnp.mean(q):.6f}, std={jnp.std(q):.6f}")
    
    intermediates['06_layer0_q'] = q
    intermediates['07_layer0_k'] = k
    intermediates['08_layer0_v'] = v
    
    # Compute attention
    # q, k, v are (batch, seq_len, num_heads, head_dim)
    # Need to transpose to (batch, num_heads, seq_len, head_dim)
    q = jnp.transpose(q, (0, 2, 1, 3))  # (bt, 12, 256, 64)
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    
    # Attention logits
    scale = 1.0 / jnp.sqrt(64.0)  # No per_dim_scale for VisionTransformer
    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
    
    intermediates['09_layer0_logits'] = logits
    print(f"  Logits: mean={jnp.mean(logits):.6f}, std={jnp.std(logits):.6f}, max={jnp.max(logits):.3f}")
    
    # Softmax
    attn_weights = jax.nn.softmax(logits, axis=-1)
    intermediates['10_layer0_attn_weights'] = attn_weights
    print(f"  Attn weights: mean={jnp.mean(attn_weights):.6f}, std={jnp.std(attn_weights):.6f}")
    
    # Apply attention to values
    attn_out = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
    attn_out = jnp.transpose(attn_out, (0, 2, 1, 3))  # (bt, 256, 12, 64)
    
    intermediates['11_layer0_attn_out_before_proj'] = attn_out
    print(f"  Attn out (before proj): mean={jnp.mean(attn_out):.6f}, std={jnp.std(attn_out):.6f}")
    
    # Output projection - slice for layer 0
    # Weight shape: (768, 12, 64), takes input (..., 12, 64) -> (..., 768)
    out_w = layer_params['self_attention']['post']['w'][layer_idx]  # (768, 12, 64)
    out_b = layer_params['self_attention']['post']['b'][layer_idx]  # (768,)
    
    # Apply projection using einsum: (..., num_heads, head_dim) @ (out_dim, num_heads, head_dim) -> (..., out_dim)
    attn_out = jnp.einsum('...hf,dhf->...d', attn_out, out_w) + out_b  # (bt, 256, 768)
    
    intermediates['12_layer0_attn_out'] = attn_out
    print(f"  Attn out (after proj): mean={jnp.mean(attn_out):.6f}, std={jnp.std(attn_out):.6f}")
    
    # Residual
    x = patches + attn_out
    intermediates['13_layer0_after_residual1'] = x
    print(f"  After residual 1: mean={jnp.mean(x):.6f}, std={jnp.std(x):.6f}")
    
    # Now we'll get the full model output for comparison
    @jax.jit
    def full_forward(inputs):
        return model.apply(state, inputs, None, None, train=False, normalize=False)
    
    full_video_emb, _, _ = full_forward(video)
    intermediates['99_final_output'] = full_video_emb
    print(f"\n✓ Full model output: mean={jnp.mean(full_video_emb):.6f}, std={jnp.std(full_video_emb):.6f}")
    
    return intermediates


def extract_mlx_vision_intermediates(model, video):
    """Extract intermediate activations from MLX vision encoder."""
    print("\n" + "=" * 80)
    print("EXTRACTING MLX INTERMEDIATE OUTPUTS")
    print("=" * 80)
    
    intermediates = {}
    
    vision = model.vision_encoder
    b, t, h, w, c = video.shape
    bt = b * t
    
    # 1. Input
    intermediates['00_input'] = video
    print(f"\n✓ Input: {video.shape}")
    
    # 2. Reshape
    inputs = video.reshape(bt, h, w, c)
    intermediates['01_reshaped'] = inputs
    
    # 3. Create patches
    patch_size = vision.patch_size
    num_patches = (h // patch_size) * (w // patch_size)
    
    patches = inputs.reshape(bt, h//patch_size, patch_size, w//patch_size, patch_size, c)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(bt, num_patches, patch_size * patch_size * c)
    intermediates['02_patches'] = patches
    print(f"✓ Patches: {patches.shape}")
    
    # 4. Patch projection
    patches = vision.patch_projection(patches)
    intermediates['03_patch_projected'] = patches
    print(f"✓ Patch projection: mean={float(mx.mean(patches)):.6f}, std={float(mx.std(patches)):.6f}")
    
    # 5. Add spatial positional embeddings
    spatial_pos_emb = vision.spatial_pos_emb(num_patches)
    patches = patches + spatial_pos_emb
    intermediates['04_with_spatial_pos'] = patches
    print(f"✓ With spatial pos: mean={float(mx.mean(patches)):.6f}, std={float(mx.std(patches)):.6f}")
    
    # 6. Apply first spatial transformer layer manually
    print(f"\n✓ Applying spatial transformer layer 0...")
    
    layer0 = vision.spatial_encoder.transformers_stack.layers[0]
    
    # Pre-LN
    ln_out = layer0.ln1(patches)
    intermediates['05_layer0_ln1'] = ln_out
    print(f"  LN1: mean={float(mx.mean(ln_out)):.6f}, std={float(mx.std(ln_out)):.6f}")
    
    # Attention - we need to manually extract Q, K, V
    q = layer0.attention.q_proj(ln_out)
    k = layer0.attention.k_proj(ln_out)
    v = layer0.attention.v_proj(ln_out)
    
    print(f"  Q: shape={q.shape}, mean={float(mx.mean(q)):.6f}, std={float(mx.std(q)):.6f}")
    
    # Reshape for multi-head: (B, T, D) -> (B, T, H, Dh) -> (B, H, T, Dh)
    B, T, _ = q.shape
    num_heads = layer0.attention.num_heads
    dim_per_head = layer0.attention.dim_per_head
    
    q = q.reshape(B, T, num_heads, dim_per_head).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, num_heads, dim_per_head).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, num_heads, dim_per_head).transpose(0, 2, 1, 3)
    
    intermediates['06_layer0_q'] = q.transpose(0, 2, 1, 3)  # Back to (B, T, H, Dh) for comparison
    intermediates['07_layer0_k'] = k.transpose(0, 2, 1, 3)
    intermediates['08_layer0_v'] = v.transpose(0, 2, 1, 3)
    
    # Compute attention logits
    scale = 1.0 / math.sqrt(dim_per_head)
    logits = (q @ k.transpose(0, 1, 3, 2)) * scale
    
    intermediates['09_layer0_logits'] = logits
    print(f"  Logits: mean={float(mx.mean(logits)):.6f}, std={float(mx.std(logits)):.6f}, max={float(mx.max(logits)):.3f}")
    
    # Softmax
    attn_weights = mx.softmax(logits, axis=-1)
    intermediates['10_layer0_attn_weights'] = attn_weights
    print(f"  Attn weights: mean={float(mx.mean(attn_weights)):.6f}, std={float(mx.std(attn_weights)):.6f}")
    
    # Apply attention
    attn_out = attn_weights @ v  # (B, H, T, Dh)
    attn_out = attn_out.transpose(0, 2, 1, 3)  # (B, T, H, Dh)
    
    intermediates['11_layer0_attn_out_before_proj'] = attn_out
    print(f"  Attn out (before proj): mean={float(mx.mean(attn_out)):.6f}, std={float(mx.std(attn_out)):.6f}")
    
    # Output projection - reshape to (B, T, H*Dh) first
    attn_out_reshaped = attn_out.reshape(B, T, num_heads * dim_per_head)
    attn_out = layer0.attention.out_proj(attn_out_reshaped)
    intermediates['12_layer0_attn_out'] = attn_out
    print(f"  Attn out (after proj): mean={float(mx.mean(attn_out)):.6f}, std={float(mx.std(attn_out)):.6f}")
    
    # Residual
    x = patches + attn_out
    intermediates['13_layer0_after_residual1'] = x
    print(f"  After residual 1: mean={float(mx.mean(x)):.6f}, std={float(mx.std(x)):.6f}")
    
    # Full model output
    full_video_emb, _, _ = model(video, None, normalize=False)
    intermediates['99_final_output'] = full_video_emb
    print(f"\n✓ Full model output: mean={float(mx.mean(full_video_emb)):.6f}, std={float(mx.std(full_video_emb)):.6f}")
    
    return intermediates


def main():
    print("=" * 80)
    print("INTERMEDIATE OUTPUT COMPARISON - FLAX VS MLX")
    print("=" * 80)
    
    # Load models
    print("\n[1/4] Loading models...")
    flax_model = vp.get_model('videoprism_lvt_public_v1_base')
    flax_state = vp.load_pretrained_weights('videoprism_lvt_public_v1_base')
    mlx_model = models_mlx.load_model('videoprism_lvt_public_v1_base')
    print("✓ Models loaded")
    
    # Load video
    print("\n[2/4] Loading video...")
    video_np = load_video('videoprism/assets/water_bottle_drumming.mp4', num_frames=16, target_size=288)
    video_flax = jnp.array(video_np[None, ...])
    video_mlx = mx.array(video_np[None, ...])
    print(f"✓ Video loaded: {video_np.shape}")
    
    # Extract intermediates
    print("\n[3/4] Extracting intermediate outputs...")
    flax_intermediates = extract_flax_vision_intermediates(flax_model, flax_state, video_flax)
    mlx_intermediates = extract_mlx_vision_intermediates(mlx_model, video_mlx)
    
    # Compare
    print("\n" + "=" * 80)
    print("[4/4] COMPARISON RESULTS")
    print("=" * 80)
    
    # Find common keys
    common_keys = sorted(set(flax_intermediates.keys()) & set(mlx_intermediates.keys()))
    
    print(f"\nComparing {len(common_keys)} checkpoints:\n")
    
    results = []
    first_divergence = None
    last_good = None
    
    for key in common_keys:
        result = compare_arrays(flax_intermediates[key], mlx_intermediates[key], key)
        results.append((key, result))
        
        if result['status'] == 'ok' or result['correlation'] > 0.95:
            last_good = key
        elif first_divergence is None and result['correlation'] < 0.8:
            first_divergence = key
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if last_good:
        print(f"✓ Last checkpoint with good correlation (>0.95): {last_good}")
    if first_divergence:
        print(f"❌ First checkpoint with poor correlation (<0.8): {first_divergence}")
        print(f"\n🔍 DIVERGENCE POINT IDENTIFIED: The issue occurs at {first_divergence}")
        
        # Show what happens between
        if last_good and first_divergence:
            last_idx = [k for k, _ in results].index(last_good)
            first_idx = [k for k, _ in results].index(first_divergence)
            
            if first_idx > last_idx:
                print(f"\n📊 Correlation progression from {last_good} to {first_divergence}:")
                for i in range(last_idx, min(first_idx + 1, len(results))):
                    k, r = results[i]
                    if 'correlation' in r:
                        print(f"   {k}: corr={r['correlation']:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import math
    main()
