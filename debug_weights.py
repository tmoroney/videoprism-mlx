"""Debug script to verify weight loading.

This tests weight conversion without running full inference.
"""

import mlx.core as mx
from videoprism import models_mlx
from pathlib import Path

print("=" * 80)
print("Weight Loading Debug Script")
print("=" * 80)

# Load weights
weights_path = "weights/videoprism_lvt_public_v1_base_mlx.safetensors"
if not Path(weights_path).exists():
    weights_path = "weights/videoprism_lvt_public_v1_base_mlx.npz"

print(f"\n1. Loading weights from {weights_path}...")
if str(weights_path).endswith('.safetensors'):
    weights = mx.load(str(weights_path))
else:
    weights = dict(mx.load(str(weights_path)))

print(f"   Loaded {len(weights)} weight tensors")

# Sample some keys
print(f"\n2. Sample weight keys (original Flax format):")
for i, key in enumerate(sorted(weights.keys())[:5]):
    print(f"   {key}: {weights[key].shape}")

# Convert weights
print(f"\n3. Converting to MLX format...")
from videoprism.weight_utils import load_and_convert_weights
mlx_weights = load_and_convert_weights(weights)

print(f"\n4. MLX weights structure (nested dict):")
print(f"   Top-level keys: {list(mlx_weights.keys())}")
print(f"   Vision encoder keys: {list(mlx_weights['vision_encoder'].keys())}")

# Check attention weights were converted
print(f"\n5. Checking attention weight conversion...")
attention_keys = [k for k in mlx_weights.keys() if 'query_proj' in k or 'key_proj' in k]
if attention_keys:
    print(f"   ✓ Found {len(attention_keys)} converted attention weights")
    print(f"   Sample: {attention_keys[0]}")
else:
    print(f"   ⚠ No converted attention weights found!")

# Try loading into model
print(f"\n6. Testing model update...")
try:
    config = models_mlx.get_model_config('videoprism_lvt_public_v1_base')
    from videoprism.encoders_mlx import FactorizedVideoCLIP
    model = FactorizedVideoCLIP(**config)
    print(f"   Model initialized")
    
    # Try updating gradually to find the issue
    for top_key in mlx_weights.keys():
        try:
            model.update({top_key: mlx_weights[top_key]})
            print(f"   ✓ Updated {top_key}")
        except Exception as e:
            print(f"   ⚠ Error updating {top_key}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"   ✓ Model.update() succeeded!")
except Exception as e:
    print(f"   ⚠ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug complete!")
print("=" * 80)
