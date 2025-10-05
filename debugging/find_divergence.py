"""Find the exact layer where Flax and MLX outputs diverge."""

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video


def compare_tensors(flax_arr, mlx_arr, name: str, threshold=0.01):
    """Compare two tensors and return if they match."""
    flax_np = np.array(flax_arr)
    mlx_np = np.array(mlx_arr)
    
    if flax_np.shape != mlx_np.shape:
        print(f"  ❌ {name}: SHAPE MISMATCH - Flax {flax_np.shape} vs MLX {mlx_np.shape}")
        return False
    
    max_diff = np.max(np.abs(flax_np - mlx_np))
    mean_diff = np.mean(np.abs(flax_np - mlx_np))
    
    is_close = np.allclose(flax_np, mlx_np, rtol=threshold, atol=threshold)
    
    status = "✓" if is_close else "❌"
    print(f"  {status} {name:50s} max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    return is_close


def test_vision_encoder_detailed(flax_model, flax_state, mlx_model, video_flax, video_mlx):
    """Test vision encoder at key checkpoints."""
    print("\n" + "=" * 80)
    print("VISION ENCODER - KEY CHECKPOINTS")
    print("=" * 80)
    
    print("\n1. Input Video")
    print("-" * 80)
    compare_tensors(video_flax, video_mlx, "Input video")
    
    # We'll use monkey-patching to capture intermediate outputs
    # For now, just run full pipeline and check output
    
    print("\n2. Full Vision Encoder Output")
    print("-" * 80)
    
    @jax.jit
    def flax_forward(inputs):
        return flax_model.apply(flax_state, inputs, None, None, train=False, normalize=False)
    
    flax_video_emb, _, _ = flax_forward(video_flax)
    mlx_video_emb, _, _ = mlx_model(video_mlx, None, normalize=False)
    
    compare_tensors(flax_video_emb, mlx_video_emb, "Final video embedding", threshold=0.1)
    
    print(f"\n  Flax: mean={np.mean(flax_video_emb):.6f}, std={np.std(flax_video_emb):.6f}, min={np.min(flax_video_emb):.6f}, max={np.max(flax_video_emb):.6f}")
    print(f"  MLX:  mean={float(mx.mean(mlx_video_emb)):.6f}, std={float(mx.std(mlx_video_emb)):.6f}, min={float(mx.min(mlx_video_emb)):.6f}, max={float(mx.max(mlx_video_emb)):.6f}")
    
    # Check correlation
    flax_flat = np.array(flax_video_emb).flatten()
    mlx_flat = np.array(mlx_video_emb).flatten()
    correlation = np.corrcoef(flax_flat, mlx_flat)[0, 1]
    print(f"  Correlation: {correlation:.6f}")
    
    # Check if it's a scaling issue
    flax_norm = np.linalg.norm(flax_flat)
    mlx_norm = np.linalg.norm(mlx_flat)
    print(f"  Norm ratio (MLX/Flax): {mlx_norm/flax_norm:.6f}")


def test_individual_transformer_layers(flax_model, flax_state, mlx_model, video_flax, video_mlx):
    """Test each transformer layer individually."""
    print("\n" + "=" * 80)
    print("TRANSFORMER LAYERS - INDIVIDUAL TESTING")
    print("=" * 80)
    
    # This would require manually tracing through each layer
    # For now, let's test the first spatial transformer layer
    
    print("\nTesting first spatial transformer layer...")
    print("-" * 80)
    
    # Get MLX components
    mlx_vision = mlx_model.vision_encoder
    mlx_layer0 = mlx_vision.spatial_encoder.transformers_stack.layers[0]
    
    # Create a simple test input
    test_input = mx.random.normal((1, 256, 768))
    
    # Test LayerNorm
    ln1_out = mlx_layer0.ln1(test_input)
    print(f"  LN1 output: mean={float(mx.mean(ln1_out)):.6f}, std={float(mx.std(ln1_out)):.6f}")
    
    # Test Attention
    attn_out = mlx_layer0.attention(ln1_out, ln1_out, ln1_out)
    print(f"  Attention output: mean={float(mx.mean(attn_out)):.6f}, std={float(mx.std(attn_out)):.6f}")
    
    # Test FFN
    ln2_out = mlx_layer0.ln2(test_input + attn_out)
    ffn_out = mlx_layer0.ffn(ln2_out)
    print(f"  FFN output: mean={float(mx.mean(ffn_out)):.6f}, std={float(mx.std(ffn_out)):.6f}")


def test_text_encoder_detailed(flax_model, flax_state, mlx_model, text_ids_flax, text_paddings_flax, text_ids_mlx, text_paddings_mlx):
    """Test text encoder."""
    print("\n" + "=" * 80)
    print("TEXT ENCODER - KEY CHECKPOINTS")
    print("=" * 80)
    
    print("\n1. Full Text Encoder Output")
    print("-" * 80)
    
    @jax.jit
    def flax_forward(text_ids, text_paddings):
        return flax_model.apply(flax_state, None, text_ids, text_paddings, train=False, normalize=False)
    
    _, flax_text_emb, _ = flax_forward(text_ids_flax, text_paddings_flax)
    _, mlx_text_emb, _ = mlx_model(None, text_ids_mlx, text_paddings=text_paddings_mlx, normalize=False)
    
    compare_tensors(flax_text_emb, mlx_text_emb, "Final text embedding", threshold=0.1)
    
    print(f"\n  Flax: mean={np.mean(flax_text_emb):.6f}, std={np.std(flax_text_emb):.6f}, min={np.min(flax_text_emb):.6f}, max={np.max(flax_text_emb):.6f}")
    print(f"  MLX:  mean={float(mx.mean(mlx_text_emb)):.6f}, std={float(mx.std(mlx_text_emb)):.6f}, min={float(mx.min(mlx_text_emb)):.6f}, max={float(mx.max(mlx_text_emb)):.6f}")
    
    # Check correlation
    flax_flat = np.array(flax_text_emb).flatten()
    mlx_flat = np.array(mlx_text_emb).flatten()
    correlation = np.corrcoef(flax_flat, mlx_flat)[0, 1]
    print(f"  Correlation: {correlation:.6f}")
    
    # Check if it's a scaling issue
    flax_norm = np.linalg.norm(flax_flat)
    mlx_norm = np.linalg.norm(mlx_flat)
    print(f"  Norm ratio (MLX/Flax): {mlx_norm/flax_norm:.6f}")


def main():
    print("=" * 80)
    print("FINDING EXACT DIVERGENCE POINT")
    print("=" * 80)
    
    # Load models
    print("\n[1/3] Loading models...")
    model_name = 'videoprism_lvt_public_v1_base'
    
    flax_model = vp.get_model(model_name)
    flax_state = vp.load_pretrained_weights(model_name)
    print("  ✓ Flax model loaded")
    
    mlx_model = models_mlx.load_model(model_name)
    print("  ✓ MLX model loaded")
    
    # Load data
    print("\n[2/3] Loading test data...")
    video_path = "videoprism/assets/water_bottle_drumming.mp4"
    video_np = load_video(video_path, num_frames=16, target_size=288)
    
    text_tokenizer = vp.load_text_tokenizer('c4_en')
    text_queries = ["a person walking"]
    text_ids_np, text_paddings_np = vp.tokenize_texts(text_tokenizer, text_queries)
    
    video_flax = jnp.array(video_np[None, ...])
    text_ids_flax = jnp.array(text_ids_np)
    text_paddings_flax = jnp.array(text_paddings_np)
    
    video_mlx = mx.array(video_np[None, ...])
    text_ids_mlx = mx.array(text_ids_np)
    text_paddings_mlx = mx.array(text_paddings_np)
    
    # Run detailed tests
    print("\n[3/3] Running detailed layer tests...")
    
    test_vision_encoder_detailed(flax_model, flax_state, mlx_model, video_flax, video_mlx)
    test_text_encoder_detailed(flax_model, flax_state, mlx_model, text_ids_flax, text_paddings_flax, text_ids_mlx, text_paddings_mlx)
    test_individual_transformer_layers(flax_model, flax_state, mlx_model, video_flax, video_mlx)
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
