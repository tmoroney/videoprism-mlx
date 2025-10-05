"""Compare intermediate activations between Flax and MLX models.

This script runs the same input through both models and compares activations
at various points to identify where outputs diverge.
"""

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from typing import Dict, Any
from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video


def compare_arrays(flax_arr, mlx_arr, name: str, rtol=1e-4, atol=1e-5):
    """Compare Flax and MLX arrays and print statistics."""
    flax_np = np.array(flax_arr)
    mlx_np = np.array(mlx_arr)
    
    # Check shape
    if flax_np.shape != mlx_np.shape:
        print(f"❌ {name}: Shape mismatch! Flax {flax_np.shape} vs MLX {mlx_np.shape}")
        return False
    
    # Compute differences
    abs_diff = np.abs(flax_np - mlx_np)
    rel_diff = abs_diff / (np.abs(flax_np) + 1e-10)
    
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # Check if close
    is_close = np.allclose(flax_np, mlx_np, rtol=rtol, atol=atol)
    
    # Statistics
    flax_stats = f"mean={np.mean(flax_np):.6f}, std={np.std(flax_np):.6f}, min={np.min(flax_np):.6f}, max={np.max(flax_np):.6f}"
    mlx_stats = f"mean={np.mean(mlx_np):.6f}, std={np.std(mlx_np):.6f}, min={np.min(mlx_np):.6f}, max={np.max(mlx_np):.6f}"
    
    status = "✓" if is_close else "❌"
    print(f"{status} {name}:")
    print(f"  Shape: {flax_np.shape}")
    print(f"  Flax:  {flax_stats}")
    print(f"  MLX:   {mlx_stats}")
    print(f"  Diff:  max_abs={max_abs_diff:.6e}, mean_abs={mean_abs_diff:.6e}, max_rel={max_rel_diff:.6f}, mean_rel={mean_rel_diff:.6f}")
    
    if not is_close:
        # Find where differences are largest
        max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"  Max diff at index {max_idx}: Flax={flax_np[max_idx]:.6f}, MLX={mlx_np[max_idx]:.6f}")
    
    print()
    return is_close


def extract_flax_activations(model, state, video, text_ids, text_paddings):
    """Extract intermediate activations from Flax model."""
    activations = {}
    
    # Get final outputs with normalization
    video_emb, text_emb, outputs = model.apply(
        state,
        video,
        text_ids,
        text_paddings,
        train=False,
        normalize=True,
        return_intermediate=False,
    )
    
    activations['video_embedding'] = video_emb
    activations['text_embedding'] = text_emb
    activations['outputs'] = outputs
    
    # Also get without normalization to check intermediate values
    video_emb_unnorm, text_emb_unnorm, _ = model.apply(
        state,
        video,
        text_ids,
        text_paddings,
        train=False,
        normalize=False,
        return_intermediate=False,
    )
    
    activations['video_embedding_unnormalized'] = video_emb_unnorm
    activations['text_embedding_unnormalized'] = text_emb_unnorm
    
    return activations


def extract_mlx_activations(model, video, text_ids, text_paddings):
    """Extract intermediate activations from MLX model."""
    activations = {}
    
    # Get final outputs with normalization
    video_emb, text_emb, outputs = model(
        video,
        text_ids,
        text_paddings=text_paddings,
        normalize=True,
        return_intermediate=False,
    )
    
    activations['video_embedding'] = video_emb
    activations['text_embedding'] = text_emb
    activations['outputs'] = outputs
    
    # Also get without normalization
    video_emb_unnorm, text_emb_unnorm, _ = model(
        video,
        text_ids,
        text_paddings=text_paddings,
        normalize=False,
        return_intermediate=False,
    )
    
    activations['video_embedding_unnormalized'] = video_emb_unnorm
    activations['text_embedding_unnormalized'] = text_emb_unnorm
    
    return activations


def main():
    print("=" * 80)
    print("Comparing Flax and MLX Model Activations")
    print("=" * 80)
    print()
    
    # Load models
    print("[1/5] Loading models...")
    model_name = 'videoprism_lvt_public_v1_base'
    
    # Flax
    flax_model = vp.get_model(model_name)
    flax_state = vp.load_pretrained_weights(model_name)
    print("  ✓ Flax model loaded")
    
    # MLX
    mlx_model = models_mlx.load_model(model_name)
    print("  ✓ MLX model loaded")
    print()
    
    # Load data
    print("[2/5] Loading test data...")
    video_path = "videoprism/assets/water_bottle_drumming.mp4"
    video_np = load_video(video_path, num_frames=16, target_size=288)
    
    text_tokenizer = vp.load_text_tokenizer('c4_en')
    text_queries = ["a person walking", "a car driving", "child with 2 sticks"]
    text_ids_np, text_paddings_np = vp.tokenize_texts(text_tokenizer, text_queries)
    
    # Prepare inputs for both models
    video_flax = jnp.array(video_np[None, ...])  # Add batch dim
    text_ids_flax = jnp.array(text_ids_np)
    text_paddings_flax = jnp.array(text_paddings_np)
    
    video_mlx = mx.array(video_np[None, ...])
    text_ids_mlx = mx.array(text_ids_np)
    text_paddings_mlx = mx.array(text_paddings_np)
    
    print(f"  Video shape: {video_np.shape}")
    print(f"  Text IDs shape: {text_ids_np.shape}")
    print()
    
    # Extract activations
    print("[3/5] Running Flax model...")
    flax_acts = extract_flax_activations(
        flax_model, flax_state, video_flax, text_ids_flax, text_paddings_flax
    )
    print("  ✓ Flax activations extracted")
    print()
    
    print("[4/5] Running MLX model...")
    mlx_acts = extract_mlx_activations(
        mlx_model, video_mlx, text_ids_mlx, text_paddings_mlx
    )
    print("  ✓ MLX activations extracted")
    print()
    
    # Compare activations
    print("[5/5] Comparing activations...")
    print("=" * 80)
    print()
    
    # First check unnormalized embeddings
    print("UNNORMALIZED EMBEDDINGS (before L2 normalization):")
    print("-" * 80)
    compare_arrays(
        flax_acts['video_embedding_unnormalized'],
        mlx_acts['video_embedding_unnormalized'],
        "Video Embedding (unnormalized)"
    )
    
    compare_arrays(
        flax_acts['text_embedding_unnormalized'],
        mlx_acts['text_embedding_unnormalized'],
        "Text Embedding (unnormalized)"
    )
    
    # Then check normalized embeddings
    print("NORMALIZED EMBEDDINGS (after L2 normalization):")
    print("-" * 80)
    compare_arrays(
        flax_acts['video_embedding'],
        mlx_acts['video_embedding'],
        "Video Embedding (normalized)"
    )
    
    compare_arrays(
        flax_acts['text_embedding'],
        mlx_acts['text_embedding'],
        "Text Embedding (normalized)"
    )
    
    # Compute similarities
    print("COSINE SIMILARITIES:")
    print("-" * 80)
    
    video_emb_flax = np.array(flax_acts['video_embedding'][0])
    video_emb_mlx = np.array(mlx_acts['video_embedding'][0])
    
    for i, text in enumerate(text_queries):
        text_emb_flax = np.array(flax_acts['text_embedding'][i])
        text_emb_mlx = np.array(mlx_acts['text_embedding'][i])
        
        sim_flax = np.dot(video_emb_flax, text_emb_flax)
        sim_mlx = np.dot(video_emb_mlx, text_emb_mlx)
        
        print(f"{text:30s} Flax: {sim_flax:.4f}  MLX: {sim_mlx:.4f}  Diff: {abs(sim_flax - sim_mlx):.4f}")
    
    print("-" * 80)
    
    print("=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
