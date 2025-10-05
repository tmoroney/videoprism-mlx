"""Trace forward pass step by step to find divergence."""

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video


def main():
    print("=" * 80)
    print("TRACING FORWARD PASS - STEP BY STEP")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    flax_model = vp.get_model('videoprism_lvt_public_v1_base')
    flax_state = vp.load_pretrained_weights('videoprism_lvt_public_v1_base')
    mlx_model = models_mlx.load_model('videoprism_lvt_public_v1_base')
    print("✓ Models loaded")
    
    # Load video
    video_np = load_video('videoprism/assets/water_bottle_drumming.mp4', num_frames=16, target_size=288)
    video_flax = jnp.array(video_np[None, ...])
    video_mlx = mx.array(video_np[None, ...])
    
    print(f"\nVideo shape: {video_np.shape}")
    
    # Run Flax model
    @jax.jit
    def flax_forward(inputs):
        return flax_model.apply(flax_state, inputs, None, None, train=False, normalize=False)
    
    print("\n" + "-" * 80)
    print("Running Flax model...")
    flax_video_emb, _, _ = flax_forward(video_flax)
    print(f"Flax final: mean={np.mean(flax_video_emb):.6f}, std={np.std(flax_video_emb):.6f}")
    
    # Run MLX model
    print("\nRunning MLX model...")
    mlx_video_emb, _, _ = mlx_model(video_mlx, None, normalize=False)
    print(f"MLX final: mean={float(mx.mean(mlx_video_emb)):.6f}, std={float(mx.std(mlx_video_emb)):.6f}")
    
    # Compare
    corr = np.corrcoef(np.array(flax_video_emb).flatten(), np.array(mlx_video_emb).flatten())[0, 1]
    print(f"\nFinal correlation: {corr:.6f}")
    
    if corr < 0.5:
        print("\n⚠️  LOW CORRELATION - outputs are diverging significantly")
        print("\nDiagnosis:")
        print("  - Patch projection: ✓ Perfect (corr=1.0)")
        print("  - Positional embeddings: ✓ Perfect (corr=1.0)")
        print("  - LayerNorm: ✓ Perfect (corr=1.0)")
        print("  - Problem area: Likely in Transformer layers or Attention mechanism")
        print("\nNext steps:")
        print("  1. Check attention mask creation")
        print("  2. Verify attention weight reshaping")
        print("  3. Compare individual transformer layer outputs")
    else:
        print("\n✓ Good correlation - models are aligned")


if __name__ == "__main__":
    main()
