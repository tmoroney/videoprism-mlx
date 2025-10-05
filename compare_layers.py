"""Deep layer-by-layer comparison to find where outputs diverge."""

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video


def compare_arrays(flax_arr, mlx_arr, name: str):
    """Quick comparison of arrays."""
    flax_np = np.array(flax_arr)
    mlx_np = np.array(mlx_arr)
    
    if flax_np.shape != mlx_np.shape:
        print(f"❌ {name}: Shape mismatch! Flax {flax_np.shape} vs MLX {mlx_np.shape}")
        return
    
    abs_diff = np.abs(flax_np - mlx_np)
    rel_diff = abs_diff / (np.abs(flax_np) + 1e-10)
    
    is_close = np.allclose(flax_np, mlx_np, rtol=1e-3, atol=1e-4)
    status = "✓" if is_close else "❌"
    
    print(f"{status} {name:40s} | Flax: std={np.std(flax_np):.4f} | MLX: std={np.std(mlx_np):.4f} | Max diff: {np.max(abs_diff):.4f}")


def test_vision_encoder(flax_model, flax_state, mlx_model, video_flax, video_mlx):
    """Test vision encoder layer by layer."""
    print("\n" + "=" * 80)
    print("VISION ENCODER COMPARISON")
    print("=" * 80)
    
    print("\nTesting full vision encoding...")
    print("-" * 80)
    
    # Flax full vision encoding
    @jax.jit
    def flax_vision_forward(inputs):
        return flax_model.apply(
            flax_state,
            inputs,
            None,  # text_token_ids
            None,  # text_paddings
            train=False,
            normalize=False,
        )
    
    video_emb_flax, _, _ = flax_vision_forward(video_flax)
    
    # MLX full vision encoding  
    video_emb_mlx, _, _ = mlx_model(
        video_mlx,
        None,  # text_ids
        text_paddings=None,
        normalize=False,
    )
    
    compare_arrays(video_emb_flax, video_emb_mlx, "Vision Encoder Output")


def test_text_encoder(flax_model, flax_state, mlx_model, text_ids_flax, text_paddings_flax, text_ids_mlx, text_paddings_mlx):
    """Test text encoder."""
    print("\n" + "=" * 80)
    print("TEXT ENCODER COMPARISON")
    print("=" * 80)
    
    # Flax text encoding
    @jax.jit
    def flax_text_forward(text_ids, text_paddings):
        return flax_model.apply(
            flax_state,
            None,  # inputs
            text_ids,
            text_paddings,
            train=False,
            normalize=False,
        )
    
    _, text_emb_flax, _ = flax_text_forward(text_ids_flax, text_paddings_flax)
    
    # MLX text encoding
    _, text_emb_mlx, _ = mlx_model(
        None,  # video
        text_ids_mlx,
        text_paddings=text_paddings_mlx,
        normalize=False,
    )
    
    compare_arrays(text_emb_flax, text_emb_mlx, "Text Encoder Output")


def test_specific_weights(flax_state, mlx_model):
    """Compare specific weight values between models."""
    print("\n" + "=" * 80)
    print("WEIGHT COMPARISON")
    print("=" * 80)
    
    # Load raw checkpoint weights (before conversion)
    mlx_weights_raw = dict(mx.load('weights/videoprism_lvt_public_v1_base_mlx.safetensors'))
    
    # Check a few critical weights from the checkpoint
    weights_to_check = [
        'vision_encoder/spatial_ln/bias',
        'vision_encoder/spatial_ln/weight',
        'text_encoder/unimodal_ln/bias',
        'text_encoder/unimodal_ln/weight',
    ]
    
    # Helper to navigate nested Flax state
    def get_nested_param(state, path):
        keys = path.split('/')
        result = state['params']
        for key in keys:
            result = result[key]
        return result
    
    for key in weights_to_check:
        try:
            flax_val = get_nested_param(flax_state, key)
            mlx_val = mlx_weights_raw[key]
            
            flax_np = np.array(flax_val)
            mlx_np = np.array(mlx_val)
            
            match = np.allclose(flax_np, mlx_np)
            status = "✓" if match else "❌"
            
            print(f"\n{status} {key}:")
            print(f"  Flax: mean={np.mean(flax_np):.6f}, std={np.std(flax_np):.6f}, min={np.min(flax_np):.6f}, max={np.max(flax_np):.6f}")
            print(f"  MLX:  mean={np.mean(mlx_np):.6f}, std={np.std(mlx_np):.6f}, min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}")
            
            if not match:
                diff = np.abs(flax_np - mlx_np)
                print(f"  Max diff: {np.max(diff):.6e}")
        except Exception as e:
            print(f"\n❌ {key}: Error - {e}")


def main():
    print("=" * 80)
    print("Deep Layer-by-Layer Comparison")
    print("=" * 80)
    
    # Load models
    print("\n[1/4] Loading models...")
    model_name = 'videoprism_lvt_public_v1_base'
    
    flax_model = vp.get_model(model_name)
    flax_state = vp.load_pretrained_weights(model_name)
    print("  ✓ Flax model loaded")
    
    mlx_model = models_mlx.load_model(model_name)
    print("  ✓ MLX model loaded")
    
    # Load data
    print("\n[2/4] Loading test data...")
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
    
    # Compare weights first
    print("\n[3/4] Comparing weights...")
    test_specific_weights(flax_state, mlx_model)
    
    # Compare encoders
    print("\n[4/4] Comparing encoders...")
    test_vision_encoder(flax_model, flax_state, mlx_model, video_flax, video_mlx)
    test_text_encoder(flax_model, flax_state, mlx_model, text_ids_flax, text_paddings_flax, text_ids_mlx, text_paddings_mlx)
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
