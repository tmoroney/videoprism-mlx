#!/usr/bin/env python3
"""Verify CLIP models match Flax outputs."""

import time
import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np

from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video

print("=" * 80)
print("Verifying CLIP Models: Flax vs MLX")
print("=" * 80)

model_names = [
    'videoprism_lvt_public_v1_base',
    'videoprism_lvt_public_v1_large',
]

for model_name in model_names:
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name}")
    print(f"{'=' * 80}")
    
    # Load models
    print("\n[1/3] Loading models...")
    flax_model = vp.get_model(model_name)
    flax_state = vp.load_pretrained_weights(model_name)
    mlx_model = models_mlx.load_model(model_name)
    
    # Load tokenizer and video
    print("[2/3] Loading video and text...")
    video = load_video("videoprism/assets/water_bottle_drumming.mp4", num_frames=16, target_size=288)
    tokenizer = vp.load_text_tokenizer('c4_en')
    text_queries = ["child drumming on water bottles"]
    
    # Tokenize
    text_ids, text_paddings = vp.tokenize_texts(tokenizer, text_queries)
    
    # Prepare inputs
    video_flax = jnp.array(video)[None, ...]
    video_mlx = mx.array(video)[None, ...]
    text_ids_flax = jnp.array(text_ids)
    text_paddings_flax = jnp.array(text_paddings)
    text_ids_mlx = mx.array(text_ids)
    text_paddings_mlx = mx.array(text_paddings)
    
    # Run inference
    print("[3/3] Running inference...")
    
    @jax.jit
    def flax_forward(video, text_ids, text_paddings):
        return flax_model.apply(flax_state, video, text_ids, text_paddings, train=False)
    
    video_emb_flax, text_emb_flax, _ = flax_forward(video_flax, text_ids_flax, text_paddings_flax)
    video_emb_mlx, text_emb_mlx, _ = mlx_model(video_mlx, text_ids_mlx, text_paddings_mlx)
    
    # Compare
    print(f"\n  Comparison Results:")
    print(f"  {'-' * 76}")
    
    # Video embeddings
    video_flax_np = np.array(video_emb_flax)
    video_mlx_np = np.array(video_emb_mlx)
    video_diff = np.abs(video_flax_np - video_mlx_np)
    
    print(f"  Video embeddings:")
    print(f"    Max diff:  {np.max(video_diff):.6e}")
    print(f"    Mean diff: {np.mean(video_diff):.6e}")
    
    # Text embeddings
    text_flax_np = np.array(text_emb_flax)
    text_mlx_np = np.array(text_emb_mlx)
    text_diff = np.abs(text_flax_np - text_mlx_np)
    
    print(f"  Text embeddings:")
    print(f"    Max diff:  {np.max(text_diff):.6e}")
    print(f"    Mean diff: {np.mean(text_diff):.6e}")
    
    # Similarity comparison
    sim_flax = float((video_emb_flax @ text_emb_flax.T)[0, 0])
    sim_mlx = float((video_emb_mlx @ text_emb_mlx.T)[0, 0])
    
    print(f"  Cosine similarity:")
    print(f"    Flax: {sim_flax:.6f}")
    print(f"    MLX:  {sim_mlx:.6f}")
    print(f"    Diff: {abs(sim_flax - sim_mlx):.6e}")
    
    if np.max(video_diff) < 1e-3 and np.max(text_diff) < 1e-3 and abs(sim_flax - sim_mlx) < 1e-3:
        print(f"  ✓✓✓ PASS - Outputs match within tolerance ✓✓✓")
    else:
        print(f"  ⚠ Differences detected")

print("\n" + "=" * 80)
print("Verification Complete")
print("=" * 80)
