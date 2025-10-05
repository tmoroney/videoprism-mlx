#!/usr/bin/env python3
"""Test script for VideoPrism video encoder backbone (video-only, no text)."""

import time
import mlx.core as mx
from videoprism import models_mlx
from videoprism.video_utils import load_video

print("=" * 80)
print("VideoPrism Video Encoder Test (MLX)")
print("=" * 80)

# ============================================================================
# Step 1: Load Model
# ============================================================================
print("\n[1/3] Loading video encoder...")
start = time.time()

model_name = 'videoprism_public_v1_base'
try:
    model = models_mlx.load_video_encoder(model_name)
    print(f"      ✓ Model loaded in {time.time() - start:.2f}s")
except FileNotFoundError as e:
    print(f"      ⚠ {e}")
    print(f"      Run: python convert_weights.py")
    print(f"      (Make sure to set model_name = '{model_name}' in convert_weights.py)")
    exit(1)

# ============================================================================
# Step 2: Load Video
# ============================================================================
print("\n[2/3] Loading and preprocessing video...")
start = time.time()

video_path = "videoprism/assets/water_bottle_drumming.mp4"
video = load_video(video_path, num_frames=16, target_size=288)

# Convert to MLX array and add batch dimension
video_input = mx.array(video)[None, ...]  # Shape: [1, T, H, W, 3]

print(f"      ✓ Video loaded in {time.time() - start:.2f}s")
print(f"      Video shape: {video_input.shape}")

# ============================================================================
# Step 3: Run Inference
# ============================================================================
print("\n[3/3] Running video encoder...")
start = time.time()

# Get video features
features, outputs = model(video_input, return_intermediate=True)

# Force computation
mx.eval(features)

print(f"      ✓ Inference completed in {time.time() - start:.2f}s")

# ============================================================================
# Results
# ============================================================================
print("\n" + "=" * 80)
print("Output Analysis")
print("=" * 80)

print(f"\nVideo features shape: {features.shape}")
batch, tokens, dim = features.shape
frames = 8 if 'large' in model_name else 16
spatial = 16
print(f"  Expected: [batch={batch}, tokens={tokens}, model_dim={dim}]")

print(f"\nFeature statistics:")
print(f"  Mean: {float(mx.mean(features)):.6f}")
print(f"  Std:  {float(mx.std(features)):.6f}")
print(f"  Min:  {float(mx.min(features)):.6f}")
print(f"  Max:  {float(mx.max(features)):.6f}")

if outputs:
    print(f"\nIntermediate outputs available:")
    for key, value in outputs.items():
        print(f"  {key}: shape={value.shape}")

print("\n" + "=" * 80)
print("Test completed successfully! ✓")
print("=" * 80)

print("\nUsage notes:")
print("  - Video features can be reshaped to (B, T, H, W, D) for spatiotemporal analysis")
print(f"  - Current shape {features.shape} can be reshaped to ({batch}, {frames}, {spatial}, {spatial}, {dim})")
print("  - Use these features for downstream tasks like classification, retrieval, etc.")
