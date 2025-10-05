"""List all MLX model parameters to compare with weight keys."""

from videoprism import models_mlx
from videoprism.encoders_mlx import FactorizedVideoCLIP

print("=" * 80)
print("MLX Model Parameter Structure")
print("=" * 80)

# Initialize model
config = models_mlx.get_model_config('videoprism_lvt_public_v1_base')
model = FactorizedVideoCLIP(**config)

# Get all parameters (nested structure)
params = model.parameters()

# Flatten the nested dictionary
def flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key))
        else:
            items.append((new_key, v))
    return items

flat_params = dict(flatten_dict(params))

print(f"\nTotal parameters: {len(flat_params)}")
print(f"\nFirst 20 parameter keys:")
for i, (key, value) in enumerate(sorted(flat_params.items())[:20]):
    shape = value.shape if hasattr(value, 'shape') else 'no shape'
    print(f"  {key}: {shape}")

if len(flat_params) > 20:
    print(f"\n... and {len(flat_params) - 20} more")

# Check for specific patterns
print(f"\nParameter name patterns:")
pos_emb_params = [k for k in flat_params.keys() if 'pos_emb' in k][:3]
print(f"  pos_emb parameters: {pos_emb_params}")
token_emb_params = [k for k in flat_params.keys() if 'token_emb' in k][:3]
print(f"  token_emb parameters: {token_emb_params}")
attention_params = [k for k in flat_params.keys() if 'attention' in k and 'query' in k][:3]
print(f"  attention parameters: {attention_params}")
pooling_params = [k for k in flat_params.keys() if 'pooling' in k][:5]
print(f"  pooling parameters: {pooling_params}")

print("\n" + "=" * 80)
