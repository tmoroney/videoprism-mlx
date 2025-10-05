# Converting VideoPrism from Flax/JAX to MLX: A Technical Guide

This document explains the process of converting Google's VideoPrism model from Flax/JAX to Apple's MLX framework, detailing the technical challenges encountered and solutions implemented.

## Table of Contents
1. [Model Overview](#model-overview)
2. [Conversion Strategy](#conversion-strategy)
3. [Key Technical Challenges](#key-technical-challenges)
4. [Weight Conversion](#weight-conversion)
5. [Numerical Stability Fixes](#numerical-stability-fixes)
6. [Validation](#validation)

---

## Model Overview

### VideoPrism Architecture
VideoPrism is a video-language foundation model that encodes videos and text into a shared embedding space. Two model variants have been converted:

**1. FactorizedVideoCLIP (video-text contrastive learning)**
```
FactorizedVideoCLIP
├── Vision Encoder (FactorizedEncoder)
│   ├── Patch Projection (18×18 patches)
│   ├── Spatial Positional Embeddings
│   ├── Spatial Transformer Stack (12 layers)
│   ├── Temporal Positional Embeddings
│   ├── Temporal Transformer Stack (4 layers)
│   └── Attention Pooling Layer
├── Text Encoder
│   ├── Token Embeddings (vocab_size=32000)
│   ├── Positional Embeddings
│   └── Unimodal Transformer (12 layers)
└── Auxiliary Encoder (2 layers, optional)
```

**2. FactorizedVideoClassifier (video classification)**
```
FactorizedVideoClassifier
├── Vision Encoder (FactorizedEncoder)
│   ├── Patch Projection (18×18 patches)
│   ├── Spatial Positional Embeddings
│   ├── Spatial Transformer Stack (12 layers)
│   ├── Temporal Positional Embeddings
│   └── Temporal Transformer Stack (4 layers)
├── Attention Pooling Layer
└── Classification Head (Linear projection)
```

### Model Specifications
- **Model Dimension**: 768
- **Attention Heads**: 12
- **MLP Hidden Dimension**: 3072
- **Attention Logit Cap**: 50.0 (tanh soft-capping)
- **Normalization**: Pre-LN (LayerNorm before attention/FFN)
- **Activation**: GELU (exact, not approximate)
---

## Conversion Strategy
Because there are no automated Flax → MLX conversion utilities, I reimplemented the model directly in MLX.

### Layer-by-Layer Equivalence
Each Flax component was mapped to an equivalent MLX implementation:

| Flax Component | MLX Component | Key Differences |
|----------------|---------------|-----------------|
| `flax.linen.LayerNorm` | Custom `LayerNorm` | Scale convention (+1.0) |
| `DotProductAttention` | Custom `DotProductAttention` | Manual softmax, logit capping |
| `nn.gelu` | Custom `gelu_exact` | Exact ERF-based implementation |
| `MultiHeadAttention` | Manual Q/K/V projections | Flax-style weight layout |

### Weight Conversion Pipeline
```python
Flax Checkpoint (.msgpack)
    ↓
Extract and flatten weights
    ↓
Rename keys (Flax → MLX conventions)
    ↓
Transpose linear layers (in,out) → (out,in)
    ↓
Reshape attention weights (model_dim, heads, head_dim) → (total_dim, model_dim)
    ↓
Save as MLX checkpoint (.safetensors)
```

---

## Key Technical Challenges

### Challenge 1: LayerNorm Scale Convention

**Problem**: Flax uses `direct_scale=False` by default, meaning:
- Scale parameter initialized to 0.0
- During forward pass: `output = (input - mean) / sqrt(var + eps) * (scale + 1.0) + bias`
- MLX's `nn.LayerNorm` applies scale directly without the +1.0

**Solution**: Custom LayerNorm implementation
```python
class LayerNorm(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.var(x, axis=-1, keepdims=True)
        x_normalized = (x - mean) / mx.sqrt(variance + self.eps)
        
        # Flax direct_scale=False behavior
        scale = self.weight + 1.0  # Add 1.0 during forward pass
        x_normalized = x_normalized * scale
        
        if self.use_bias:
            x_normalized = x_normalized + self.bias
        
        return x_normalized
```

### Challenge 2: GELU Activation Precision

**Problem**: 
- Flax uses exact GELU: `0.5 * x * (1 + erf(x / sqrt(2)))`
- MLX's `nn.gelu` uses an approximation
- Through 12 transformer layers, approximation errors compound

**Solution**: Implement exact GELU
```python
def gelu_exact(x: mx.array) -> mx.array:
    """Exact GELU activation (matches Flax's approximate=False)."""
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))
```

**Impact**: Using exact GELU eliminates 0.1-0.2% error per layer that was causing cumulative drift.

### Challenge 3: Attention Weight Reshaping

**Problem**: Flax stores Q/K/V weights as `(model_dim, num_heads, head_dim)` but MLX Linear layers expect `(out_features, in_features)`.

**Solution**: Careful transpose and reshape
```python
# Flax format: (model_dim_in=768, num_heads=12, head_dim=64)
q_w_flax = checkpoint['q.w']  # Shape: (768, 12, 64)

# MLX Linear expects: (out_features=768, in_features=768)
q_w_mlx = q_w_flax.transpose(1, 2, 0).reshape(768, 768)
#          Step 1: (12, 64, 768) ← reorder dimensions
#          Step 2: (768, 768)    ← flatten heads×head_dim

# Output projection is simpler (already in right order):
out_w_mlx = out_w_flax.reshape(768, 768)
```

### Challenge 4: Residual Connection Stability

**Problem**: With 12 spatial layers + 4 temporal layers, residual connections are applied 32 times. Small numerical differences in addition order caused drift.

**Solution**: Explicit evaluation and addition
```python
# Before (implicit lazy evaluation):
inputs = inputs + atten_out

# After (explicit evaluation order):
mx.eval(atten_out)  # Force evaluation before addition
inputs = mx.add(inputs, atten_out)  # Explicit add operation
```

**Why it works**: `mx.eval()` forces computation immediately, preventing MLX's graph optimization from reordering operations in ways that introduce numerical differences.

### Challenge 5: Attention Softmax Numerical Stability

**Problem**: Standard softmax can overflow/underflow with large logits, and attention masks need careful handling.

**Solution**: Numerically stable softmax with proper masking
```python
def _manual_attention_with_logit_cap(q, k, v, scale, logit_cap, mask):
    # Compute logits
    logits = (q @ k.transpose(0, 1, 3, 2)) * scale
    
    # Apply tanh soft-capping
    if logit_cap > 0.0:
        logits = logit_cap * mx.tanh(logits / logit_cap)
    
    # Stable softmax with masking
    logits32 = logits.astype(mx.float32)
    
    # Apply mask before softmax
    if mask is not None:
        mask32 = mask.astype(mx.float32)
        min_value = mx.finfo(logits32.dtype).min
        mask_selector = mask32 >= (min_value * 0.5)
        logits32 = mx.where(mask_selector, logits32, min_value)
    
    # Numerically stable softmax (subtract max)
    max_logits = mx.max(logits32, axis=-1, keepdims=True)
    stabilized = logits32 - max_logits
    exp_logits = mx.exp(stabilized)
    
    # Apply mask to probabilities
    if mask_selector is not None:
        exp_logits = exp_logits * mask_selector.astype(exp_logits.dtype)
    
    # Normalize
    denom = mx.sum(exp_logits, axis=-1, keepdims=True)
    denom = mx.maximum(denom, mx.array(1e-9, dtype=denom.dtype))
    probs = exp_logits / denom
    
    return probs @ v
```

### Challenge 6: Attention Pooling Layer

**Problem**: Vision pooler uses cross-attention with learnable queries. MLX's `nn.MultiHeadAttention` has subtle differences from Flax's implementation.

**Solution**: Manual implementation using Flax-style weight layout
```python
class AttentionPoolingLayer(nn.Module):
    def __init__(self, ...):
        # Store weights in Flax format: (dim, num_heads, head_dim)
        self.q_proj_w = mx.zeros((query_dim, num_heads, head_dim))
        self.k_proj_w = mx.zeros((input_dim, num_heads, head_dim))
        self.v_proj_w = mx.zeros((input_dim, num_heads, head_dim))
        self.out_proj_w = mx.zeros((query_dim, num_heads, head_dim))
        
        # Per-dimension scaling (if enabled)
        self.per_dim_scale = PerDimScale(...) if internal_enable_per_dim_scale else None
    
    def __call__(self, tokens, paddings=None):
        # Manual Q/K/V projections using einsum
        q = mx.einsum('bqd,dhf->bqhf', query, self.q_proj_w)
        k = mx.einsum('bsd,dhf->bshf', tokens, self.k_proj_w)
        v = mx.einsum('bsd,dhf->bshf', tokens, self.v_proj_w)
        
        # Apply per-dim scaling to queries
        q = q.transpose(0, 2, 1, 3)  # (B, H, Q, D)
        if self.per_dim_scale is not None:
            q = self.per_dim_scale(q)
        
        # Attention computation
        scale = 1.0 / math.sqrt(head_dim)
        logits = mx.einsum('bhqd,bhkd->bhqk', q, k) * scale
        attn = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True)
        
        # Output projection
        attn_out = mx.einsum('bhqk,bhkd->bhqd', attn, v)
        output = mx.einsum('bqhd,mhd->bqm', attn_out.transpose(0, 2, 1, 3), self.out_proj_w)
        
        return output
```

**Key insight**: Keeping weights in Flax's `(dim, heads, head_dim)` format and using `einsum` for projections ensures exact numerical equivalence.

---

## Weight Conversion

### Key Renaming Rules

```python
# 1. Attention projections
'self_attention/query/w' → 'attention.q_proj.weight'
'self_attention/key/w'   → 'attention.k_proj.weight'
'self_attention/value/w' → 'attention.v_proj.weight'
'self_attention/post/w'  → 'attention.out_proj.weight'

# 2. FFN layers
'ff_layer/ffn_layer1/linear/kernel' → 'ffn.ffn.fc1.weight'
'ff_layer/ffn_layer2/linear/kernel' → 'ffn.ffn.fc2.weight'

# 3. LayerNorm (keep Flax naming for scale/bias)
'layer_norm/scale' → 'ln1.weight'  # Will add +1.0 in forward pass
'layer_norm/bias'  → 'ln1.bias'

# 4. Embeddings
'token_emb/embedding' → 'token_emb.emb_var'
'pos_emb/emb_var'     → 'pos_emb.emb_var'
```

### Transpose Rules

```python
# Linear layers: Flax uses (in, out), MLX uses (out, in)
if key.endswith('.weight') and len(value.shape) == 2:
    value = value.T

# Attention Q/K/V: (model_dim, heads, head_dim) → (total_dim, model_dim)
q_w_mlx = q_w_flax.transpose(1, 2, 0).reshape(num_heads * head_dim, model_dim)

# Output projection: (model_dim, heads, head_dim) → (model_dim, total_dim)
out_w_mlx = out_w_flax.reshape(model_dim, num_heads * head_dim)

# Pooling attention: Keep original (dim, heads, head_dim) format
# No reshaping needed - handled in runtime with einsum
```

---

## Numerical Stability Fixes

### Summary of Critical Fixes

| Issue | Impact | Solution | File:Line |
|-------|--------|----------|-----------|
| Approximate GELU | 0.2% error/layer | Exact ERF implementation | `layers_mlx.py:36` |
| LayerNorm scale | Immediate divergence | Add +1.0 in forward pass | `layers_mlx.py:122` |
| Residual drift | 0.1% error/layer | Explicit `mx.eval` + `mx.add` | `layers_mlx.py:436` |
| Attention overflow | NaN in deep layers | Stable softmax with max subtraction | `layers_mlx.py:215` |
| Pooling mismatch | Final embedding error | Flax-style manual projections | `layers_mlx.py:595` |
| Variance computation | Subtle drift | Use `mx.var` instead of manual | `layers_mlx.py:117` |

### Debugging Strategy Used

1. **Text encoder first**: Simpler architecture (no spatial/temporal split), got this working perfectly first
2. **Layer 0 verification**: Ensured first transformer layer matched exactly before testing full stack
3. **Layer-by-layer correlation**: Tracked correlation drop through 12 layers to identify accumulation points
4. **Component isolation**: Tested individual components (LayerNorm, attention, FFN) separately
5. **Numerical comparison**: Compared intermediate activations (mean, std, max, correlation) at each step

---

## Validation

### Test Results

**Final Comparison** (after all fixes):
```
Video Embedding (unnormalized):
  Max absolute difference: 2.24e-4
  Mean absolute difference: 3.8e-5
  Correlation: 0.9999

Video Embedding (normalized):  
  Max absolute difference: 5.9e-6
  Mean absolute difference: 8.2e-7
  Correlation: 1.0000

Text Embedding:
  Max absolute difference: 1.75e-7
  Mean absolute difference: 2.5e-8
  Correlation: 1.0000

Cosine Similarities (video-text):
  "child with 2 sticks": Flax=0.1514, MLX=0.1514 ✓
  "a person walking":    Flax=0.0852, MLX=0.0852 ✓
  "a car driving":       Flax=0.0469, MLX=0.0469 ✓
```

**Layer-by-Layer Spatial Transformer Correlation**:
```
Layer  0: corr=1.0000
Layer  1: corr=1.0000
Layer  2: corr=1.0000
Layer  3: corr=1.0000
Layer  4: corr=1.0000
Layer  5: corr=1.0000
Layer  6: corr=1.0000
Layer  7: corr=1.0000
Layer  8: corr=1.0000
Layer  9: corr=1.0000
Layer 10: corr=1.0000
Layer 11: corr=1.0000
```

### Validation Scripts

```bash
# Run weight conversion
python convert_weights.py

# Compare full model outputs
python compare_activations.py

# Trace layer-by-layer correlation
python trace_all_layers.py

# Test individual components
python debugging/find_divergence.py
```

---

## Lessons Learned

### Key Takeaways

1. **Framework conventions matter**: Small differences (LayerNorm scale, GELU approximation) compound in deep networks
2. **Explicit is better than implicit**: Force evaluation order with `mx.eval()` for reproducibility
3. **Test incrementally**: Verify each component before moving to full model
4. **Keep reference format**: For complex operations (pooling attention), keeping Flax's weight layout simplified debugging
5. **Numerical stability is critical**: Use float32 for sensitive operations (softmax), subtract max before exp

### Common Pitfalls

- **Don't assume API equivalence**: `nn.gelu` ≠ Flax's exact GELU
- **Weight transpose is tricky**: Attention weights need careful dimension reordering, not just transpose
- **Lazy evaluation**: MLX's graph optimization can reorder ops; use `mx.eval()` when order matters
- **Mask handling**: Apply masks before softmax for stability, not after

---

## Performance Notes

Note that the performance numbers are specific to Apple Silicon hardware and may not be representative of other hardware configurations.

### Speed Comparison (16-frame clip, single batch)

Measured on this machine with the provided inference scripts. Each timing
excludes model/tokenizer loading and averages multiple warm-started forward
passes.

```
Flax/JAX (CPU, 10 runs):   4.54 s ± 0.15 s per pass
MLX (GPU, 20 runs):        1.42 s ± 0.04 s per pass
Observed speedup:          ≈3.2× faster
```

### Memory Usage (peak resident set size)

Measured with `resource.getrusage(RUSAGE_SELF).ru_maxrss` after one warm
forward pass in a fresh process.

```
Flax/JAX:  5.26 GB
MLX:       1.17 GB
```

#### Benchmark methodology

- Hardware: 14" MacBook Pro (Apple M3 Pro; unified-memory capacity per local configuration).
- Commands:
  - `python benchmark_performance.py --framework flax --runs 10 --warmup 2`
  - `python benchmark_performance.py --framework mlx --runs 20 --warmup 3`
- Video clip: `videoprism/assets/water_bottle_drumming.mp4`, 16 frames @ 288×288.
- Tokenizer: `c4_en`; text prompts: `"a person walking"`, `"drumming on water bottles"`, `"a car driving"` (default in the script).
- Results above are the script output (mean/std/min/max) with embeddings left unnormalized during timing to avoid unnecessary work.

The helper script `benchmark_performance.py` is checked into the repository so
that new measurements can be reproduced or captured on different hardware. Run
`python benchmark_performance.py --help` for the full set of options (batch
size, clip length, number of runs, etc.).

MLX's unified memory architecture on Apple Silicon provides both speed and efficiency benefits compared to Flax/JAX (CPU only).

---

## Future Work

### Potential Improvements

1. **Quantization**: Implement int8/int4 quantization for faster inference
2. **Batch processing**: Optimize for larger batch sizes
3. **Compilation**: Investigate `@mx.compile` for frequently-called functions
4. **Mixed precision**: Explore bfloat16 for non-critical operations

---

## References

- **VideoPrism Paper**: [Video Foundation Models via Multi-Task Learning](https://arxiv.org/abs/2312.03640)
- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **Flax Documentation**: https://flax.readthedocs.io/
- **Original Flax Implementation**: `videoprism/layers.py`, `videoprism/encoders.py`

---

## Acknowledgments

This conversion was made possible by:
- Google's open release of VideoPrism weights
- Apple's MLX framework and documentation
- The JAX/Flax community for reference implementations

**License**: Same as original VideoPrism (Apache 2.0)

**Maintainer**: See repository for contact information
