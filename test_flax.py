import jax
import numpy as np
import time
from videoprism import models as vp
from videoprism.video_utils import load_video, load_video_batch

print("=" * 60)
print("Starting VideoPrism inference pipeline")
print("=" * 60)

# Video-text encoders.
start = time.time()
print("\n[1/6] Loading model configuration...")
model_name = 'videoprism_lvt_public_v1_base'  # configuration name
flax_model = vp.get_model(model_name)
print(f"      ✓ Model loaded in {time.time() - start:.2f}s")

start = time.time()
print("[2/6] Loading pretrained weights...")
loaded_state = vp.load_pretrained_weights(model_name)
print(f"      ✓ Weights loaded in {time.time() - start:.2f}s")

start = time.time()
print("[3/6] Loading text tokenizer...")
text_tokenizer = vp.load_text_tokenizer('c4_en')
print(f"      ✓ Tokenizer loaded in {time.time() - start:.2f}s")

@jax.jit
def forward_fn(inputs, text_token_ids, text_token_paddings, train=False):
  return flax_model.apply(
      loaded_state,
      inputs,
      text_token_ids,
      text_token_paddings,
      train=train,
  )

# Load a single video
start = time.time()
print("[4/6] Loading and preprocessing video...")
video_path = "videoprism/assets/water_bottle_drumming.mp4"
video = load_video(video_path, num_frames=16, target_size=288)
video_inputs = np.expand_dims(video, axis=0)  # Add batch dimension
print(f"      ✓ Video loaded in {time.time() - start:.2f}s")

# Or load multiple videos as a batch
# video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
# video_inputs = load_video_batch(video_paths, num_frames=16, target_size=288)

# Prepare text queries
start = time.time()
print("[5/6] Tokenizing text queries...")
text_queries = ["a person walking", "drumming on water bottles", "a car driving"]
text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, text_queries)
print(f"      ✓ Text tokenized in {time.time() - start:.2f}s")

# Run inference
start = time.time()
print("[6/6] Running model inference...")
video_embeddings, text_embeddings, _ = forward_fn(video_inputs, text_ids, text_paddings)
print(f"      ✓ Inference completed in {time.time() - start:.2f}s")
print(f"\nVideo embeddings shape: {video_embeddings.shape}")
print(f"Text embeddings shape: {text_embeddings.shape}")
print()

# Calculate cosine similarities
def cosine_similarity(a, b):
  """Compute cosine similarity between two vectors."""
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compute similarity between video and each text query
video_emb = video_embeddings[0]  # Shape: (768,)
print("Cosine similarities between video and text queries:")
print("-" * 60)
for i, text_query in enumerate(text_queries):
  text_emb = text_embeddings[i]  # Shape: (768,)
  similarity = cosine_similarity(video_emb, text_emb)
  print(f"{text_query:30s} -> {similarity:.4f}")
print("-" * 60)

# Find best matching text
best_idx = np.argmax([cosine_similarity(video_emb, text_embeddings[i]) 
                       for i in range(len(text_queries))])
print(f"\nBest match: '{text_queries[best_idx]}'")
print("\n" + "=" * 60)
print("Pipeline completed successfully")
print("=" * 60)