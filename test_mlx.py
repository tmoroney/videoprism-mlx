"""VideoPrism MLX inference test script.

This script demonstrates how to use the MLX-converted VideoPrism model for:
1. Video-text similarity computation
2. Video embedding extraction
3. Text embedding extraction
"""

import time
import numpy as np
import mlx.core as mx
from pathlib import Path

# Import MLX model utilities
from videoprism import models_mlx
from videoprism import models as vp
from videoprism.video_utils import load_video

print("=" * 80)
print("Starting VideoPrism MLX inference pipeline")
print("=" * 80)

# ============================================================================
# Step 1 & 2: Load MLX Model and Weights
# ============================================================================
start = time.time()
print("\n[1/6] Loading MLX model and weights...")

model_name = 'videoprism_lvt_public_v1_base'
try:
    model = models_mlx.load_model(model_name)
    print(f"      ✓ Model loaded in {time.time() - start:.2f}s")
except FileNotFoundError as e:
    print(f"      ⚠ {e}")
    print("      Please run: python convert_weights.py")
    exit(1)

# ============================================================================
# Step 2: Load Text Tokenizer
# ============================================================================
start = time.time()
print("\n[2/5] Loading text tokenizer...")
text_tokenizer = vp.load_text_tokenizer('c4_en')
print(f"      ✓ Tokenizer loaded in {time.time() - start:.2f}s")

# ============================================================================
# Step 3: Load and Preprocess Video
# ============================================================================
start = time.time()
print("\n[3/5] Loading and preprocessing video...")
video_path = "videoprism/assets/water_bottle_drumming.mp4"
video = load_video(video_path, num_frames=16, target_size=288)

# Convert to MLX array and add batch dimension
video_input = mx.array(video)[None, ...]  # Shape: [1, T, H, W, 3]
print(f"      ✓ Video loaded in {time.time() - start:.2f}s")
print(f"      Video shape: {video_input.shape}")

# ============================================================================
# Step 4: Tokenize Text Queries
# ============================================================================
start = time.time()
print("\n[4/5] Tokenizing text queries...")
text_queries = ["a person walking", "a car driving", "child with 2 sticks"]
text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, text_queries)

# Convert to MLX arrays
text_ids_mlx = mx.array(text_ids)
text_paddings_mlx = mx.array(text_paddings)
print(f"      ✓ Text tokenized in {time.time() - start:.2f}s")
print(f"      Text shape: {text_ids_mlx.shape}")

# ============================================================================
# Step 5: Run MLX Inference
# ============================================================================
start = time.time()
print("\n[5/5] Running MLX model inference...")

# Forward pass
video_embeddings, text_embeddings, outputs = model(
    inputs=video_input,
    text_token_ids=text_ids_mlx,
    text_paddings=text_paddings_mlx,
    normalize=True,
    return_intermediate=False,
)

# Evaluate (MLX is lazy, so we need to force computation)
mx.eval(video_embeddings, text_embeddings)

print(f"      ✓ Inference completed in {time.time() - start:.2f}s")
print(f"\nVideo embeddings shape: {video_embeddings.shape}")
print(f"Text embeddings shape: {text_embeddings.shape}")
print()

# ============================================================================
# Compute Similarities
# ============================================================================
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

# Compute similarity between video and each text query
video_emb = video_embeddings[0]  # Shape: (768,)
print("Cosine similarities between video and text queries:")
print("-" * 80)

similarities = []
for i, text_query in enumerate(text_queries):
    text_emb = text_embeddings[i]  # Shape: (768,)
    similarity = cosine_similarity(video_emb, text_emb)
    similarities.append(similarity)
    print(f"{text_query:30s} -> {similarity:.4f}")

print("-" * 80)

# Find best matching text
best_idx = np.argmax(similarities)
print(f"\nBest match: '{text_queries[best_idx]}' (similarity: {similarities[best_idx]:.4f})")

print("\n" + "=" * 80)
print("MLX Pipeline completed successfully! ✓")
print("=" * 80)
print("\nModel Statistics:")
config = models_mlx.get_model_config(model_name)
print(f"  Model dimension: {config['model_dim']}")
print(f"  Spatial layers: {config['num_spatial_layers']}")
print(f"  Temporal layers: {config['num_temporal_layers']}")
print(f"  Vocabulary size: {config['vocabulary_size']:,}")
print()
