"""Deep debugging of vision encoder to find exact divergence point."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import mlx.core as mx
import numpy as np
from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video


def compare(flax_arr, mlx_arr, name: str, verbose=True):
    """Compare arrays and return metrics."""
    flax_np = np.array(flax_arr)
    mlx_np = np.array(mlx_arr)
    
    if flax_np.shape != mlx_np.shape:
        print(f"  ❌ {name}: SHAPE MISMATCH - Flax {flax_np.shape} vs MLX {mlx_np.shape}")
        return None
    
    max_diff = np.max(np.abs(flax_np - mlx_np))
    mean_diff = np.mean(np.abs(flax_np - mlx_np))
    rel_diff = mean_diff / (np.abs(np.mean(flax_np)) + 1e-10)
    
    corr = np.corrcoef(flax_np.flatten(), mlx_np.flatten())[0, 1]
    
    if verbose:
        status = "✓" if max_diff < 0.01 else "❌"
        print(f"  {status} {name:40s} corr={corr:.4f}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rel_diff': rel_diff,
        'correlation': corr,
        'flax_mean': np.mean(flax_np),
        'mlx_mean': np.mean(mlx_np),
        'flax_std': np.std(flax_np),
        'mlx_std': np.std(mlx_np),
    }


def test_patch_projection():
    """Test patch projection in isolation."""
    print("\n" + "=" * 80)
    print("TESTING: Patch Projection")
    print("=" * 80)
    
    # Load models
    flax_model = vp.get_model('videoprism_lvt_public_v1_base')
    flax_state = vp.load_pretrained_weights('videoprism_lvt_public_v1_base')
    mlx_model = models_mlx.load_model('videoprism_lvt_public_v1_base')
    
    # Create test input (single frame)
    np.random.seed(42)
    test_img = np.random.randn(1, 288, 288, 3).astype(np.float32)
    
    # Extract patches manually (16x16 patches from 288x288)
    patch_size = 18
    num_patches = (288 // patch_size) ** 2
    
    # Reshape to patches
    h, w = 288, 288
    patches = test_img.reshape(1, h//patch_size, patch_size, w//patch_size, patch_size, 3)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)  # [1, 16, 16, 18, 18, 3]
    patches = patches.reshape(1, num_patches, patch_size * patch_size * 3)  # [1, 256, 972]
    
    print(f"\nInput: {test_img.shape}")
    print(f"Patches: {patches.shape}")
    
    # Apply Flax patch projection
    # Need to extract the weights and apply manually
    flax_weight = np.array(flax_state['params']['vision_encoder']['patch_projection']['linear']['kernel'])
    flax_bias = np.array(flax_state['params']['vision_encoder']['patch_projection']['linear']['bias'])
    
    flax_output = patches @ flax_weight + flax_bias
    print(f"Flax output: {flax_output.shape}, mean={np.mean(flax_output):.6f}, std={np.std(flax_output):.6f}")
    
    # Apply MLX patch projection
    mlx_patches = mx.array(patches)
    mlx_output = mlx_model.vision_encoder.patch_projection(mlx_patches)
    print(f"MLX output: {mlx_output.shape}, mean={float(mx.mean(mlx_output)):.6f}, std={float(mx.std(mlx_output)):.6f}")
    
    # Compare
    compare(flax_output, mlx_output, "Patch projection output")
    
    # Also check weights
    print("\nWeight comparison:")
    mlx_weight_loaded = np.array(mlx_model.vision_encoder.patch_projection.weight)
    mlx_bias_loaded = np.array(mlx_model.vision_encoder.patch_projection.bias)
    
    # Flax stores as (in, out), MLX stores as (out, in)
    flax_weight_t = flax_weight.T
    
    print(f"  Flax weight shape: {flax_weight.shape} (in, out)")
    print(f"  MLX weight shape: {mlx_weight_loaded.shape} (out, in)")
    print(f"  Weight match after transpose: {np.allclose(flax_weight_t, mlx_weight_loaded)}")
    print(f"  Bias match: {np.allclose(flax_bias, mlx_bias_loaded)}")
    
    if not np.allclose(flax_weight_t, mlx_weight_loaded):
        diff = np.abs(flax_weight_t - mlx_weight_loaded)
        print(f"  Weight diff: max={np.max(diff):.6e}, mean={np.mean(diff):.6e}")


def test_layer_norm():
    """Test LayerNorm implementation in isolation."""
    print("\n" + "=" * 80)
    print("TESTING: LayerNorm")
    print("=" * 80)
    
    # Load models
    flax_model = vp.get_model('videoprism_lvt_public_v1_base')
    flax_state = vp.load_pretrained_weights('videoprism_lvt_public_v1_base')
    mlx_model = models_mlx.load_model('videoprism_lvt_public_v1_base')
    
    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(1, 256, 768).astype(np.float32)
    
    print(f"Test input: mean={np.mean(test_input):.6f}, std={np.std(test_input):.6f}")
    
    # Test spatial_ln
    print("\n1. Spatial LayerNorm")
    print("-" * 40)
    
    # Flax LayerNorm
    flax_scale = np.array(flax_state['params']['vision_encoder']['spatial_ln']['scale'])
    flax_bias = np.array(flax_state['params']['vision_encoder']['spatial_ln']['bias'])
    
    # Manual Flax LayerNorm computation
    mean = np.mean(test_input, axis=-1, keepdims=True)
    var = np.mean(np.square(test_input - mean), axis=-1, keepdims=True)
    flax_normed = (test_input - mean) / np.sqrt(var + 1e-6)
    flax_output = flax_normed * (flax_scale + 1.0) + flax_bias  # +1.0 is critical!
    
    print(f"  Flax scale: mean={np.mean(flax_scale):.6f}, min={np.min(flax_scale):.6f}, max={np.max(flax_scale):.6f}")
    print(f"  Flax scale+1: mean={np.mean(flax_scale + 1.0):.6f}")
    print(f"  Flax output: mean={np.mean(flax_output):.6f}, std={np.std(flax_output):.6f}")
    
    # MLX LayerNorm
    mlx_input = mx.array(test_input)
    mlx_output = mlx_model.vision_encoder.spatial_ln(mlx_input)
    
    mlx_weight = np.array(mlx_model.vision_encoder.spatial_ln.weight)
    mlx_bias_loaded = np.array(mlx_model.vision_encoder.spatial_ln.bias)
    
    print(f"  MLX weight: mean={np.mean(mlx_weight):.6f}, min={np.min(mlx_weight):.6f}, max={np.max(mlx_weight):.6f}")
    print(f"  MLX output: mean={float(mx.mean(mlx_output)):.6f}, std={float(mx.std(mlx_output)):.6f}")
    
    # Compare
    compare(flax_output, mlx_output, "LayerNorm output")
    
    # Verify weights loaded correctly
    print("\n  Weight verification:")
    print(f"    Flax scale == MLX weight: {np.allclose(flax_scale, mlx_weight)}")
    print(f"    Flax bias == MLX bias: {np.allclose(flax_bias, mlx_bias_loaded)}")


def test_positional_embeddings():
    """Test positional embeddings."""
    print("\n" + "=" * 80)
    print("TESTING: Positional Embeddings")
    print("=" * 80)
    
    # Load models
    flax_state = vp.load_pretrained_weights('videoprism_lvt_public_v1_base')
    mlx_model = models_mlx.load_model('videoprism_lvt_public_v1_base')
    
    # Test spatial positional embedding
    print("\n1. Spatial Positional Embedding")
    print("-" * 40)
    
    flax_spatial_pos_weights = np.array(flax_state['params']['vision_encoder']['spatial_pos_emb']['emb_var'])
    mlx_spatial_pos_weights = np.array(mlx_model.vision_encoder.spatial_pos_emb.emb_var)
    
    print(f"  Flax weights shape: {flax_spatial_pos_weights.shape}")
    print(f"  MLX weights shape: {mlx_spatial_pos_weights.shape}")
    
    compare(flax_spatial_pos_weights, mlx_spatial_pos_weights, "Spatial pos emb weights")
    
    # Test temporal positional embedding
    print("\n2. Temporal Positional Embedding")
    print("-" * 40)
    
    flax_temporal_pos_weights = np.array(flax_state['params']['vision_encoder']['temporal_pos_emb']['emb_var'])
    mlx_temporal_pos_weights = np.array(mlx_model.vision_encoder.temporal_pos_emb.emb_var)
    
    print(f"  Flax weights shape: {flax_temporal_pos_weights.shape}")
    print(f"  MLX weights shape: {mlx_temporal_pos_weights.shape}")
    
    compare(flax_temporal_pos_weights, mlx_temporal_pos_weights, "Temporal pos emb weights")


def test_attention_layer():
    """Test attention layer in isolation."""
    print("\n" + "=" * 80)
    print("TESTING: Attention Layer")
    print("=" * 80)
    
    # Load models
    flax_state = vp.load_pretrained_weights('videoprism_lvt_public_v1_base')
    mlx_model = models_mlx.load_model('videoprism_lvt_public_v1_base')
    
    # Create test input
    np.random.seed(42)
    test_input = np.random.randn(1, 256, 768).astype(np.float32)
    
    # Test first spatial transformer layer attention
    print("\nFirst Spatial Transformer Attention:")
    print("-" * 40)
    
    # Check attention weights
    layer_idx = 0
    
    # Check if attention weights are loaded correctly
    print("\nChecking attention weight loading:")
    
    # Q projection  
    flax_q_w = np.array(flax_state['params']['vision_encoder']['spatial_encoder']['transformers_stack']['x_layers'][str(layer_idx)]['self_attention']['query']['w'])
    flax_q_b = np.array(flax_state['params']['vision_encoder']['spatial_encoder']['transformers_stack']['x_layers'][str(layer_idx)]['self_attention']['query']['b'])
    
    mlx_q_w = np.array(mlx_model.vision_encoder.spatial_encoder.transformers_stack.layers[0].attention.q_proj.weight)
    mlx_q_b = np.array(mlx_model.vision_encoder.spatial_encoder.transformers_stack.layers[0].attention.q_proj.bias)
    
    print(f"  Flax Q weight shape: {flax_q_w.shape}")  # (768, 12, 64)
    print(f"  MLX Q weight shape: {mlx_q_w.shape}")    # Should be (768, 768) after reshape
    
    # Flax format: (model_dim, num_heads, head_dim) = (768, 12, 64)
    # Should reshape to (768, 768)
    flax_q_reshaped = flax_q_w.reshape(768, 768)
    
    print(f"  Q weight match: {np.allclose(flax_q_reshaped, mlx_q_w, atol=1e-5)}")
    if not np.allclose(flax_q_reshaped, mlx_q_w, atol=1e-5):
        diff = np.abs(flax_q_reshaped - mlx_q_w)
        print(f"  Q weight diff: max={np.max(diff):.6e}, mean={np.mean(diff):.6e}")
    
    # Check bias
    flax_q_b_reshaped = flax_q_b.reshape(-1)
    print(f"  Q bias match: {np.allclose(flax_q_b_reshaped, mlx_q_b, atol=1e-5)}")


def main():
    print("=" * 80)
    print("DEEP VISION ENCODER DEBUGGING")
    print("=" * 80)
    
    # Test each component in isolation
    test_patch_projection()
    test_positional_embeddings()
    test_layer_norm()
    test_attention_layer()
    
    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
