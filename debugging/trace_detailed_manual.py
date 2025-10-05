"""Manually trace both Flax and MLX step-by-step by replicating the forward pass."""

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video


def compare_step(flax_val, mlx_val, name: str):
    """Compare a single step."""
    flax_np = np.array(flax_val)
    mlx_np = np.array(mlx_val)
    
    if flax_np.shape != mlx_np.shape:
        print(f"  ❌ {name:50s} SHAPE MISMATCH: {flax_np.shape} vs {mlx_np.shape}")
        return
    
    corr = np.corrcoef(flax_np.flatten(), mlx_np.flatten())[0, 1]
    max_diff = np.max(np.abs(flax_np - mlx_np))
    
    if corr > 0.99:
        status = "✓✓"
    elif corr > 0.9:
        status = "✓"
    elif corr > 0.7:
        status = "⚠️"
    else:
        status = "❌"
    
    print(f"  {status} {name:50s} corr={corr:.4f}, max_diff={max_diff:.6f}")
    
    return corr


def manual_flax_forward_vision(state, video):
    """Manually compute Flax vision encoder forward pass."""
    
    print("\n" + "=" * 80)
    print("MANUAL FLAX FORWARD PASS")
    print("=" * 80)
    
    b, t, h, w, c = video.shape
    bt = b * t
    
    # Reshape
    inputs = video.reshape(bt, h, w, c)
    print(f"\n1. Input reshaped: {inputs.shape}")
    
    # Create patches (18x18 patches)
    patch_size = 18
    num_row_patches = h // patch_size
    num_col_patches = w // patch_size
    num_patches = num_row_patches * num_col_patches
    
    patches = inputs.reshape(bt, num_row_patches, patch_size, num_col_patches, patch_size, c)
    patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
    patches = patches.reshape(bt, num_patches, patch_size * patch_size * c)
    print(f"2. Patches created: {patches.shape}")
    
    # Apply patch projection
    patch_weight = state['params']['vision_encoder']['patch_projection']['linear']['kernel']
    patch_bias = state['params']['vision_encoder']['patch_projection']['linear']['bias']
    
    patches = patches @ patch_weight + patch_bias
    print(f"3. After patch projection: {patches.shape}, mean={jnp.mean(patches):.6f}")
    
    # Add spatial positional embeddings
    spatial_pos_emb = state['params']['vision_encoder']['spatial_pos_emb']['emb_var']
    spatial_pos_emb = spatial_pos_emb[:num_patches]  # Slice to sequence length
    
    # Flax does one-hot matmul for lookup, which is equivalent to simple indexing
    # Just add directly
    patches = patches + spatial_pos_emb[jnp.newaxis, :, :]
    print(f"4. After spatial pos emb: mean={jnp.mean(patches):.6f}")
    
    # Now we need to apply spatial transformer
    # This is complex because of the stacked layers
    # For now, let's just get the output from the full Flax model and see
    
    return {
        '01_input': video,
        '02_patches_before_proj': patches,
        '03_patches_after_proj': patches,
        '04_patches_with_pos': patches,
    }


def manual_mlx_forward_vision(model, video):
    """Manually compute MLX vision encoder forward pass."""
    
    print("\n" + "=" * 80)
    print("MANUAL MLX FORWARD PASS")
    print("=" * 80)
    
    vision = model.vision_encoder
    b, t, h, w, c = video.shape
    bt = b * t
    
    # Reshape
    inputs = video.reshape(bt, h, w, c)
    print(f"\n1. Input reshaped: {inputs.shape}")
    
    # Create patches
    patch_size = vision.patch_size
    num_row_patches = h // patch_size
    num_col_patches = w // patch_size
    num_patches = num_row_patches * num_col_patches
    
    patches = inputs.reshape(bt, num_row_patches, patch_size, num_col_patches, patch_size, c)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(bt, num_patches, patch_size * patch_size * c)
    print(f"2. Patches created: {patches.shape}")
    
    patches_before = patches
    
    # Apply patch projection
    patches = vision.patch_projection(patches)
    print(f"3. After patch projection: {patches.shape}, mean={float(mx.mean(patches)):.6f}")
    
    patches_after_proj = patches
    
    # Add spatial positional embeddings
    spatial_pos_emb = vision.spatial_pos_emb(num_patches)
    patches = patches + spatial_pos_emb
    print(f"4. After spatial pos emb: mean={float(mx.mean(patches)):.6f}")
    
    patches_with_pos = patches
    
    # Apply first spatial transformer layer
    print(f"\n5. Applying spatial transformer (12 layers)...")
    
    # Get first layer
    layer0 = vision.spatial_encoder.transformers_stack.layers[0]
    
    # LN1
    ln1_out = layer0.ln1(patches)
    print(f"   Layer 0 - After LN1: mean={float(mx.mean(ln1_out)):.6f}, std={float(mx.std(ln1_out)):.6f}")
    
    # Attention
    attn_out, _ = layer0.attention(ln1_out, ln1_out, ln1_out)
    print(f"   Layer 0 - After attention: mean={float(mx.mean(attn_out)):.6f}, std={float(mx.std(attn_out)):.6f}")
    
    # Residual
    x = patches + attn_out
    print(f"   Layer 0 - After residual 1: mean={float(mx.mean(x)):.6f}")
    
    # LN2
    ln2_out = layer0.ln2(x)
    print(f"   Layer 0 - After LN2: mean={float(mx.mean(ln2_out)):.6f}, std={float(mx.std(ln2_out)):.6f}")
    
    # FFN
    ffn_out = layer0.ffn(ln2_out)
    print(f"   Layer 0 - After FFN: mean={float(mx.mean(ffn_out)):.6f}, std={float(mx.std(ffn_out)):.6f}")
    
    # Residual
    x = x + ffn_out
    print(f"   Layer 0 - After residual 2: mean={float(mx.mean(x)):.6f}")
    
    layer0_out = x
    
    # Full spatial transformer
    spatial_out = vision.spatial_encoder(patches, paddings=None)
    print(f"   After all 12 layers: mean={float(mx.mean(spatial_out)):.6f}, std={float(mx.std(spatial_out)):.6f}")
    
    # Spatial LN
    spatial_out = vision.spatial_ln(spatial_out)
    print(f"6. After spatial LN: mean={float(mx.mean(spatial_out)):.6f}, std={float(mx.std(spatial_out)):.6f}")
    
    return {
        '01_input': video,
        '02_patches_before_proj': patches_before,
        '03_patches_after_proj': patches_after_proj,
        '04_patches_with_pos': patches_with_pos,
        '05_layer0_ln1': ln1_out,
        '06_layer0_attn': attn_out,
        '07_layer0_ffn': ffn_out,
        '08_layer0_out': layer0_out,
        '09_spatial_out': spatial_out,
    }


def main():
    print("=" * 80)
    print("DETAILED MANUAL TRACING")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    flax_state = vp.load_pretrained_weights('videoprism_lvt_public_v1_base')
    flax_model = vp.get_model('videoprism_lvt_public_v1_base')
    mlx_model = models_mlx.load_model('videoprism_lvt_public_v1_base')
    
    # Load video
    video_np = load_video('videoprism/assets/water_bottle_drumming.mp4', num_frames=16, target_size=288)
    video_flax = jnp.array(video_np[None, ...])
    video_mlx = mx.array(video_np[None, ...])
    
    # Manual forward passes
    flax_acts = manual_flax_forward_vision(flax_state, video_flax)
    mlx_acts = manual_mlx_forward_vision(mlx_model, video_mlx)
    
    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    for key in ['01_input', '02_patches_before_proj', '03_patches_after_proj', '04_patches_with_pos']:
        if key in flax_acts and key in mlx_acts:
            compare_step(flax_acts[key], mlx_acts[key], key)
    
    # Compare full model outputs
    print("\n" + "=" * 80)
    print("FULL MODEL COMPARISON")
    print("=" * 80)
    
    @jax.jit
    def flax_forward(inputs):
        return flax_model.apply(flax_state, inputs, None, None, train=False, normalize=False)
    
    flax_emb, _, _ = flax_forward(video_flax)
    mlx_emb, _, _ = mlx_model(video_mlx, None, normalize=False)
    
    compare_step(flax_emb, mlx_emb, "Final video embedding")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
