"""Trace forward pass with detailed correlation tracking at each step."""

import jax
import jax.numpy as jnp
import mlx.core as mx
import numpy as np
from videoprism import models as vp
from videoprism import models_mlx
from videoprism.video_utils import load_video
from typing import Dict, Any


class ActivationTracer:
    """Capture intermediate activations during forward pass."""
    
    def __init__(self):
        self.activations = {}
    
    def record(self, name: str, value):
        """Record an activation."""
        self.activations[name] = np.array(value) if not isinstance(value, np.ndarray) else value
    
    def compare(self, flax_tracer: 'ActivationTracer', mlx_tracer: 'ActivationTracer'):
        """Compare activations between Flax and MLX."""
        print("\n" + "=" * 80)
        print("CORRELATION TRACKING - STEP BY STEP")
        print("=" * 80)
        
        # Get common keys
        flax_keys = set(flax_tracer.activations.keys())
        mlx_keys = set(mlx_tracer.activations.keys())
        common_keys = sorted(flax_keys & mlx_keys)
        
        print(f"\nTracking {len(common_keys)} checkpoints...\n")
        
        results = []
        for key in common_keys:
            flax_act = flax_tracer.activations[key]
            mlx_act = mlx_tracer.activations[key]
            
            if flax_act.shape != mlx_act.shape:
                print(f"⚠️  {key:50s} SHAPE MISMATCH: {flax_act.shape} vs {mlx_act.shape}")
                continue
            
            # Compute metrics
            flax_flat = flax_act.flatten()
            mlx_flat = mlx_act.flatten()
            
            corr = np.corrcoef(flax_flat, mlx_flat)[0, 1]
            max_diff = np.max(np.abs(flax_flat - mlx_flat))
            mean_diff = np.mean(np.abs(flax_flat - mlx_flat))
            
            flax_mean = np.mean(flax_flat)
            mlx_mean = np.mean(mlx_flat)
            flax_std = np.std(flax_flat)
            mlx_std = np.std(mlx_flat)
            
            # Determine status
            if corr > 0.99:
                status = "✓✓"
            elif corr > 0.9:
                status = "✓"
            elif corr > 0.7:
                status = "⚠️"
            else:
                status = "❌"
            
            print(f"{status} {key:50s} corr={corr:.4f} | Flax: μ={flax_mean:7.4f} σ={flax_std:.4f} | MLX: μ={mlx_mean:7.4f} σ={mlx_std:.4f}")
            
            results.append({
                'name': key,
                'correlation': corr,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'flax_mean': flax_mean,
                'mlx_mean': mlx_mean,
                'flax_std': flax_std,
                'mlx_std': mlx_std,
            })
        
        # Find first major divergence
        print("\n" + "-" * 80)
        print("ANALYSIS:")
        print("-" * 80)
        
        first_bad = None
        last_good = None
        
        for r in results:
            if r['correlation'] > 0.95:
                last_good = r['name']
            elif first_bad is None and r['correlation'] < 0.8:
                first_bad = r['name']
        
        if last_good:
            print(f"✓ Last checkpoint with good correlation (>0.95): {last_good}")
        if first_bad:
            print(f"❌ First checkpoint with poor correlation (<0.8): {first_bad}")
            
            # Find what happens between last_good and first_bad
            if last_good and first_bad:
                last_idx = [r['name'] for r in results].index(last_good)
                first_idx = [r['name'] for r in results].index(first_bad)
                
                if first_idx - last_idx == 1:
                    print(f"\n⚠️  DIVERGENCE OCCURS at: {first_bad}")
                    print(f"   Previous step ({last_good}) was fine")
                    print(f"   This step introduces the error!")
                else:
                    print(f"\n⚠️  DIVERGENCE OCCURS between steps {last_idx} and {first_idx}:")
                    for i in range(last_idx + 1, first_idx + 1):
                        r = results[i]
                        print(f"     {r['name']}: corr={r['correlation']:.4f}")
        
        return results


def trace_vision_encoder_mlx(model, video):
    """Trace MLX vision encoder with detailed logging."""
    tracer = ActivationTracer()
    
    # Input
    tracer.record("00_input_video", video)
    
    vision = model.vision_encoder
    b, t, h, w, c = video.shape
    
    # Reshape and create patches
    bt = b * t
    inputs = video.reshape(bt, h, w, c)
    tracer.record("01_reshaped_for_patches", inputs)
    
    # Patch projection
    patch_size = vision.patch_size
    num_row_patches = h // patch_size
    num_col_patches = w // patch_size
    num_patches = num_row_patches * num_col_patches
    
    # Reshape to patches
    patches = inputs.reshape(bt, num_row_patches, patch_size, num_col_patches, patch_size, c)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(bt, num_patches, patch_size * patch_size * c)
    tracer.record("02_patches_before_projection", patches)
    
    # Project patches
    patches = vision.patch_projection(patches)
    tracer.record("03_patches_after_projection", patches)
    
    # Add spatial positional embeddings
    spatial_pos_emb = vision.spatial_pos_emb(num_patches)
    tracer.record("04_spatial_pos_emb", spatial_pos_emb)
    
    patches = patches + spatial_pos_emb
    tracer.record("05_patches_with_pos_emb", patches)
    
    # Apply spatial transformer
    spatial_out = vision.spatial_encoder(patches, paddings=None)
    tracer.record("06_after_spatial_transformer", spatial_out)
    
    # Apply spatial layer norm
    spatial_out = vision.spatial_ln(spatial_out)
    tracer.record("07_after_spatial_ln", spatial_out)
    
    # Reshape for temporal
    n = num_patches
    d = vision.model_dim
    temporal_input = spatial_out.reshape(b, t, n, d)
    tracer.record("08_reshaped_for_temporal", temporal_input)
    
    # Spatial pooling (mean over patches)
    temporal_input = mx.mean(temporal_input, axis=2)
    tracer.record("09_after_spatial_pooling", temporal_input)
    
    # Add temporal positional embeddings
    temporal_pos_emb = vision.temporal_pos_emb(t)
    tracer.record("10_temporal_pos_emb", temporal_pos_emb)
    
    temporal_input = temporal_input + temporal_pos_emb
    tracer.record("11_temporal_with_pos_emb", temporal_input)
    
    # Apply temporal transformer
    temporal_out = vision.temporal_encoder(temporal_input, paddings=None)
    tracer.record("12_after_temporal_transformer", temporal_out)
    
    # Apply temporal layer norm
    temporal_out = vision.temporal_ln(temporal_out)
    tracer.record("13_after_temporal_ln", temporal_out)
    
    # Apply vision pooler
    pooled = model.contrastive_vision_pooler(temporal_out)
    tracer.record("14_after_vision_pooler", pooled)
    
    # Squeeze
    video_emb = mx.squeeze(pooled, axis=1)
    tracer.record("15_final_video_embedding", video_emb)
    
    return video_emb, tracer


def trace_vision_encoder_flax(model, state, video):
    """Trace Flax vision encoder by running full model."""
    tracer = ActivationTracer()
    
    # For Flax, we can't easily intercept intermediate values without modifying the model
    # So we'll just record input and output
    tracer.record("00_input_video", video)
    
    @jax.jit
    def forward(inputs):
        return model.apply(state, inputs, None, None, train=False, normalize=False)
    
    video_emb, _, _ = forward(video)
    tracer.record("15_final_video_embedding", video_emb)
    
    return video_emb, tracer


def trace_vision_encoder_flax_detailed(model, state, video):
    """Manually trace Flax vision encoder step by step."""
    tracer = ActivationTracer()
    
    # Input
    tracer.record("00_input_video", video)
    
    # We need to manually replicate the Flax forward pass
    # This is complex, so for now we'll use a simpler approach:
    # Run MLX step-by-step and compare against final Flax output at each transformer layer
    
    # Get Flax final output
    @jax.jit
    def forward(inputs):
        return model.apply(state, inputs, None, None, train=False, normalize=False)
    
    video_emb, _, _ = forward(video)
    tracer.record("15_final_video_embedding", video_emb)
    
    return video_emb, tracer


def main():
    print("=" * 80)
    print("DETAILED CORRELATION TRACING")
    print("=" * 80)
    
    # Load models
    print("\n[1/4] Loading models...")
    flax_model = vp.get_model('videoprism_lvt_public_v1_base')
    flax_state = vp.load_pretrained_weights('videoprism_lvt_public_v1_base')
    mlx_model = models_mlx.load_model('videoprism_lvt_public_v1_base')
    print("✓ Models loaded")
    
    # Load video
    print("\n[2/4] Loading video...")
    video_np = load_video('videoprism/assets/water_bottle_drumming.mp4', num_frames=16, target_size=288)
    video_flax = jnp.array(video_np[None, ...])
    video_mlx = mx.array(video_np[None, ...])
    print(f"✓ Video loaded: {video_np.shape}")
    
    # Trace MLX model
    print("\n[3/4] Tracing MLX vision encoder...")
    mlx_emb, mlx_tracer = trace_vision_encoder_mlx(mlx_model, video_mlx)
    print(f"✓ MLX traced: {len(mlx_tracer.activations)} checkpoints")
    
    # Trace Flax model
    print("\n[4/4] Tracing Flax vision encoder...")
    flax_emb, flax_tracer = trace_vision_encoder_flax_detailed(flax_model, flax_state, video_flax)
    print(f"✓ Flax traced: {len(flax_tracer.activations)} checkpoints")
    
    # Compare
    results = mlx_tracer.compare(flax_tracer, mlx_tracer)
    
    # Additional analysis
    print("\n" + "=" * 80)
    print("DETAILED DIAGNOSTICS")
    print("=" * 80)
    
    # Check if the issue compounds over layers
    if len(results) > 5:
        print("\nCorrelation progression:")
        for i, r in enumerate(results):
            if i % 3 == 0 or r['correlation'] < 0.9:  # Show every 3rd or problematic ones
                print(f"  Step {i:2d} ({r['name']:30s}): {r['correlation']:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
