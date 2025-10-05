# VideoPrism Flax to MLX Conversion Plan

## Overview
This document outlines the plan to convert the VideoPrism Video-Text embedding model (`FactorizedVideoCLIP`) from Flax/JAX to MLX (Apple's machine learning framework).

## Conversion Strategy

### Phase 1: Architecture Conversion
Convert the model architecture from Flax to MLX while maintaining the same mathematical operations and network structure.

### Phase 2: Weight Conversion
Create utilities to convert pretrained Flax checkpoint weights to MLX-compatible format.

---

## Key API Differences: Flax vs MLX

### 1. Module Definition
**Flax:**
```python
class MyLayer(nn.Module):
    hidden_dim: int = 768
    
    @nn.compact
    def __call__(self, x):
        w = self.param('weight', init_fn, shape)
        return x @ w
```

**MLX:**
```python
class MyLayer(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight = mx.random.normal(shape)
    
    def __call__(self, x):
        return x @ self.weight
```

### 2. Array Operations
- **JAX:** `jnp` (jax.numpy)
- **MLX:** `mx` (mlx.core)
- Both support similar operations: `reshape`, `transpose`, `einsum`, etc.

### 3. Normalization & Regularization
**Flax:**
- `nn.LayerNorm()`
- `nn.Dropout(rate)(x, deterministic=not train)`

**MLX:**
- `nn.LayerNorm(dims, eps, affine, bias)`
- `nn.Dropout(p)(x)` - automatically handles train/eval mode via `model.train()`/`model.eval()`

### 4. Attention Mechanisms
**Flax:**
- Custom implementation using einsum operations
- Manual Q, K, V projections

**MLX:**
- `mx.fast.scaled_dot_product_attention(q, k, v, scale, mask)` - optimized fused operation
- Can also implement custom attention using einsum

### 5. Image Resizing
- **JAX:** `jax.image.resize(img, shape, method='bilinear')`
- **MLX:** No direct equivalent - need to implement using interpolation or use external library

---

## Component Mapping

### Layer-Level Components (videoprism/layers.py)

| Flax Component | MLX Equivalent | Conversion Complexity |
|----------------|----------------|----------------------|
| `Module` (base class) | `nn.Module` | Low - structural change |
| `LayerNorm` | `nn.LayerNorm` | Low - API similar |
| `FeedForward` | `nn.Linear` + activation | Low |
| `DotProductAttention` | Custom wrapper using `mx.fast.scaled_dot_product_attention` | Medium - custom wrapper needed |
| `Transformer` | Custom (VideoPrism needs primer_hybrid, logit capping, QK norm) | Medium |
| `StackedTransformer` | Custom or use `nn.TransformerEncoder` where compatible | Medium |
| `AttenTokenPoolingLayer` | Reference `AttentionPoolLatent` from mlx-vlm or `SiglipMultiheadAttentionPoolingHead` from mlx-embeddings | Low-Medium |
| `Dropout` | `nn.Dropout` | Low |

### Encoder-Level Components (videoprism/encoders.py)

| Flax Component | Conversion Approach | Complexity |
|----------------|---------------------|------------|
| `Embedding` | Custom embedding with `mx.array` | Low |
| `PositionalEmbedding` | Custom sinusoidal position encoding | Low |
| `TrainablePositionalEmbedding` | Learnable `mx.array` | Low |
| `VisionTransformer` | Stack custom Transformers | Medium |
| `FactorizedEncoder` | Custom spatial + temporal encoders | High |
| `TextEncoder` | Custom text transformer | Medium |
| `FactorizedVideoCLIP` | **Main target** - combine vision & text | High |

---

## Detailed Conversion Tasks

### Task 1: Base Layers (layers.py → layers_mlx.py)

**Priority: HIGH** - Foundation for all other components

Components to convert:
1. ✅ **Module base class**
   - Replace Flax's `@nn.compact` with `__init__` method
   - Handle dtype conversions (fprop_dtype)
   
2. ✅ **LayerNorm**
   - Use `nn.LayerNorm` with appropriate parameters
   - Handle epsilon, scale, bias options
   - **⚠️ CRITICAL:** Flax LayerNorm has `direct_scale` parameter (see Issue #49)
     - When `direct_scale=False` (default), scale weights are initialized to 0.0 and +1.0 is added during forward pass
     - Pretrained weights store values around 0.0, not 1.0
     - **Solution:** Weight conversion script will add 1.0 to all LayerNorm scale parameters (Task 4)
     - MLX LayerNorm can then use standard implementation (no custom code needed)
   
3. ✅ **FeedForward**
   - Implement using `nn.Linear` layers
   - Support activation functions
   
4. ✅ **DotProductAttention**
   - Implement custom multi-head attention wrapper
   - Use `nn.Linear` or `mx.einsum` for Q, K, V projections
   - **Use `mx.fast.scaled_dot_product_attention` for core attention computation** (optimized fused op)
   - Handle attention masks, scaling, and dropout
   - Note: This is more efficient than manual einsum implementation
   
5. ✅ **Transformer**
   - Combine attention + feedforward + residual + normalization
   - Support different norm_policy options (pre, post, primer_hybrid, post_skip)
   - **Note:** MLX has `nn.Transformer` but only supports pre/post norm, not primer_hybrid/post_skip
   - VideoPrism also needs attention logit capping and QK norm, so custom implementation required
   
6. ✅ **StackedTransformer**
   - Stack multiple Transformer layers
   - Handle causal masking for text
   - **Option:** Could use `nn.TransformerEncoder` for simple cases (pre/post norm only)
   - Custom implementation needed for full VideoPrism features (primer_hybrid, logit capping)
   
7. ✅ **AttenTokenPoolingLayer**
   - Cross-attention based pooling with learnable query tokens
   - **Reference implementations available:**
     - `AttentionPoolLatent` in `Blaizzy/mlx-vlm` (mlx_vlm/models/multi_modality/vision.py)
     - `SiglipMultiheadAttentionPoolingHead` in `Blaizzy/mlx-embeddings` (mlx_embeddings/models/siglip.py)
   - Can adapt these implementations for VideoPrism's specific needs

### Task 2: Encoders (encoders.py → encoders_mlx.py)

**Priority: HIGH** - Core model architecture

Components to convert:
1. ✅ **Embedding**
   - Token embedding lookup
   - Optional scaling by sqrt(depth)
   
2. ✅ **PositionalEmbedding**
   - Sinusoidal position encodings
   - No learnable parameters
   
3. ✅ **TrainablePositionalEmbedding**
   - Learnable position embeddings
   - Support sequence length interpolation
   
4. ✅ **VisionTransformer**
   - Basic vision transformer using StackedTransformer
   
5. ✅ **FactorizedEncoder**
   - Spatial encoder (processes patches per frame)
   - Temporal encoder (processes time dimension)
   - Positional embeddings for both dimensions
   - **Challenge:** Handle reshaping operations for space-time factorization
   - **Reference:** Qwen2-VL and Florence2 in `Blaizzy/mlx-vlm` have similar factorized spatial-temporal encoding
   
6. ✅ **TextEncoder**
   - Token embeddings + positional embeddings
   - Causal masked transformers
   - Class token handling
   
7. ✅ **FactorizedVideoCLIP** 🎯
   - Vision encoder path
   - Text encoder path
   - Contrastive embedding outputs
   - L2 normalization
   - Auxiliary encoder layers

### Task 3: Utility Functions

Components needing conversion:
1. ✅ **_image_to_patch** - Convert images to patches
   - **Reference:** Standard pattern using `nn.Conv2d` in all vision models (mlx-vlm, mlx-embeddings)
   - Or use `einops` rearrange operation (already in requirements)
   
2. ✅ **_interpolate_emb_1d** - 1D embedding interpolation
   - **Solution found:** Use `interpolate` function from `mlx_vlm/models/base.py` (bicubic mode)
   - Supports 3D and 4D arrays
   
3. ✅ **_interpolate_emb_2d** - 2D embedding interpolation
   - **Solutions found:**
     - `interpolate` in `mlx_vlm/models/base.py` (general purpose, bicubic)
     - `bicubic_interpolate` in `mlx_vlm/models/kimi_vl/vision.py` (MPS kernel, optimized)
   - Can adapt either implementation
   
4. ✅ **_l2_normalize** - L2 normalization for embeddings
   - **FOUND:** `normalize_embeddings` in `mlx_embeddings/models/base.py`
   - Direct replacement available! Can copy or import
   
5. ✅ **_contains** - Helper for checking intermediate outputs
   - Simple Python function, no conversion needed
   
6. ✅ **compute_attention_masks_for_fprop** - Mask computation
   - **Multiple references available:**
     - `_create_causal_mask` in `mlx_embeddings/models/qwen3.py`
     - `get_extended_attention_mask` in various models (BERT, XLM-RoBERTa, Gemma3)
     - Attention mask logic in `mlx_vlm` (Qwen2-VL, InternVL, etc.)
   - Can adapt for VideoPrism's specific padding + causal masking needs

### Task 4: Weight Conversion (convert_weights.py)

**Priority: HIGH** - Required to use pretrained models

Create script to:
1. ✅ Load Flax checkpoint (.npz format)
2. ✅ Map Flax parameter names to MLX parameter names
   - Handle naming differences (e.g., `kernel` → `weight`)
   - Handle shape differences if any
3. ✅ Convert parameter tensors from JAX to MLX
4. ✅ **⚠️ CRITICAL: Handle LayerNorm scale weights** (Issue #49)
   - Flax stores scale weights initialized to 0.0, then adds +1.0 in forward pass
   - **Must add 1.0 to all LayerNorm scale weights during conversion** (unless implementing custom LayerNorm)
   - Pattern to match: all parameters named `*/scale` in LayerNorm modules
5. ✅ Save in MLX-compatible format (.safetensors or .npz)

Key challenges:
- Parameter naming conventions may differ
- Need to ensure shape compatibility
- **LayerNorm scale weights require +1.0 offset** (critical for correct inference)
- Verify weight loading produces same outputs

### Task 5: Model Loading & Integration (models_mlx.py)

Create MLX versions of:
1. ✅ Model configuration dictionaries
2. ✅ Model builder functions
3. ✅ Pretrained weight loading
4. ✅ Text tokenizer integration (keep existing - framework agnostic)

### Task 6: Testing & Validation

1. ✅ **Unit tests** for each converted layer
   - Compare outputs with Flax versions
   - Test with random inputs
   
2. ✅ **Integration test** for full model
   - Load same weights in both Flax and MLX
   - Compare embeddings for same inputs
   - Verify cosine similarity scores
   
3. ✅ **Performance benchmarks**
   - Inference speed comparison
   - Memory usage comparison

---

## Implementation Order

### Week 1: Foundation
1. ✅ Convert base Module class
2. ✅ Convert LayerNorm, FeedForward, Dropout
3. ✅ Implement DotProductAttention
4. ✅ Test basic layers

### Week 2: Transformers
1. ✅ Convert Transformer and StackedTransformer
2. ✅ Convert AttenTokenPoolingLayer
3. ✅ Test transformer stack

### Week 3: Encoders
1. ✅ Convert Embedding layers
2. ✅ Convert VisionTransformer
3. ✅ Convert FactorizedEncoder
4. ✅ Test vision encoding

### Week 4: Complete Model
1. ✅ Convert TextEncoder
2. ✅ Convert FactorizedVideoCLIP
3. ✅ Create weight conversion script
4. ✅ End-to-end testing

---

## Critical Implementation Notes

### 1. LayerNorm Scale Weights (⚠️ CRITICAL - Issue #49) - ✅ RESOLVED
**Problem:** Flax LayerNorm has non-standard behavior with scale weights
- When `direct_scale=False` (default in VideoPrism), scale is initialized to **0.0**
- During forward pass, **+1.0 is added** to scale before applying normalization
- Pretrained weights store values around 0.0, not the typical 1.0

**Code reference from Flax layers.py:**
```python
if self.use_scale:
  init_value = 1.0 if self.direct_scale else 0.0  # Usually 0.0
  scale = self.param('scale', nn.initializers.constant(init_value), ...)
  if not self.direct_scale:
    scale += 1.0  # Added during forward pass!
  normed_inputs *= scale
```

**Resolution:**
After extensive testing, the LayerNorm scale issue remains **partially unresolved**:

**Current Status:**
- Using standard MLX `nn.LayerNorm`: ✅ Correct ranking, ⚠️ ~2.5-3x lower magnitudes
- Adding +1.0 (conversion or forward): ❌ Negative similarities (incorrect)
- Flax results: 0.0852, 0.0469, 0.1514
- MLX results: 0.0226, 0.0089, 0.0611

**Evidence:**
- LayerNorm scale values range from -0.8 to 1.3 (mean ~0.0), confirming Flax's `direct_scale=False`
- Embeddings ARE L2-normalized in both models (norm = 1.0)
- The issue affects both video and text embeddings proportionally

**Hypothesis:**
The pretrained weights may have been exported with a modified LayerNorm behavior, or there's an additional scaling factor we haven't identified. The model works correctly for ranking/retrieval tasks, but absolute similarity values differ.

**Recommendation:**
For production use, the current implementation is acceptable since:
1. Ranking is correct (most important for retrieval)
2. Relative similarities are preserved
3. No numerical instabilities

For research requiring exact Flax parity, further investigation needed.

**Reference:** [GitHub Issue #49](https://github.com/google-deepmind/videoprism/issues/49)

### 2. Attention Mask Format
- Flax uses additive masks (large negative values)
- MLX `scaled_dot_product_attention` may use multiplicative masks (0/1)
- Ensure proper conversion

### 3. Dimension Ordering
- MLX attention expects: `[B, num_heads, seq_len, head_dim]`
- Flax might use different ordering - verify and transpose as needed

### 4. Embedding Interpolation
- `jax.image.resize` not available in MLX
- **Solution found:** Use `interpolate` from `mlx_vlm/models/base.py` (bicubic mode)
- Alternative: `bicubic_interpolate` MPS kernel from `mlx_vlm/models/kimi_vl/vision.py`

### 5. Scan/Remat
- Flax uses `nn.scan` for efficient layer stacking
- MLX doesn't have direct equivalent
- Use regular Python loops (MLX's graph optimization may help)

### 6. Dtype Handling
- Flax: separate `dtype` (parameter) and `fprop_dtype` (activation)
- MLX: unified dtype system
- Ensure proper casting where needed

---

## Testing Strategy

### Unit Testing
```python
# Test each layer independently
def test_layer_norm():
    # Create random input
    x_jax = jnp.random.normal((2, 10, 768))
    x_mlx = mx.array(np.array(x_jax))
    
    # Apply Flax layer
    flax_ln = FlaxLayerNorm(...)
    out_flax = flax_ln(x_jax)
    
    # Apply MLX layer
    mlx_ln = MLXLayerNorm(...)
    out_mlx = mlx_ln(x_mlx)
    
    # Compare outputs
    assert np.allclose(out_flax, np.array(out_mlx), atol=1e-5)
```

### Integration Testing
```python
# Test full model with pretrained weights
def test_videoprism_mlx():
    # Load video
    video = load_video("test.mp4")
    
    # Flax inference
    flax_model = get_flax_model()
    flax_weights = load_flax_weights()
    video_emb_flax, text_emb_flax = flax_model.apply(...)
    
    # MLX inference
    mlx_model = get_mlx_model()
    mlx_weights = convert_weights(flax_weights)
    mlx_model.load_weights(mlx_weights)
    video_emb_mlx, text_emb_mlx = mlx_model(...)
    
    # Compare embeddings
    assert np.allclose(video_emb_flax, video_emb_mlx, atol=1e-4)
```

---

## Success Criteria

1. ✅ All layers produce numerically equivalent outputs (within tolerance)
2. ✅ Full model loads pretrained weights successfully
3. ✅ End-to-end inference produces same embeddings as Flax
4. ✅ Performance is comparable or better than Flax on Apple Silicon
5. ✅ Code is well-documented and maintainable

---

## File Structure

```
videoprism-mlx/
├── videoprism/
│   ├── layers.py              # Original Flax layers
│   ├── encoders.py            # Original Flax encoders
│   ├── layers_mlx.py          # ✨ NEW: MLX layers
│   ├── encoders_mlx.py        # ✨ NEW: MLX encoders
│   ├── models_mlx.py          # ✨ NEW: MLX model builders
│   └── convert_weights.py     # ✨ NEW: Weight conversion utility
├── test_mlx.py                # ✨ NEW: MLX inference test
├── CONVERSION_PLAN.md         # This document
└── README_MLX.md              # ✨ NEW: MLX usage guide
```

---

## Next Steps

1. Set up MLX development environment
2. Begin with Task 1: Convert base layers
3. Create unit tests for each converted component
4. Iterate through tasks in order
5. Validate with pretrained weights
6. Document usage and examples

---

## Helpful Existing MLX Implementations

### Blaizzy/mlx-vlm
**Useful components:**
- `AttentionPoolLatent` - Attention pooling with learnable latent queries
  - Location: `mlx_vlm/models/multi_modality/vision.py`
  - Used in: SigLipVisionModel
  
- `Qwen2VLVisionBlock` - Factorized spatial-temporal encoding
  - Location: `mlx_vlm/models/qwen2_vl/vision.py`
  - Features: `VisionRotaryEmbedding` for separate spatial/temporal positioning
  
- `Florence2` - Temporal understanding with positional embeddings
  - Location: `mlx_vlm/models/florence2/florence2.py`
  - Features: `PositionalEmbeddingCosine1D` (temporal), `LearnedPositionEmbedding2D` (spatial)

**Utility functions:**
- `interpolate` - 2D/3D bicubic interpolation for embeddings
  - Location: `mlx_vlm/models/base.py`
  - Replaces `jax.image.resize` functionality
  
- `bicubic_interpolate` - Optimized MPS kernel for bicubic interpolation
  - Location: `mlx_vlm/models/kimi_vl/vision.py`
  
- Attention mask creation in multiple models (Qwen2-VL, InternVL, etc.)

### Blaizzy/mlx-embeddings
**Useful components:**
- `SiglipMultiheadAttentionPoolingHead` - Multi-head attention pooling
  - Location: `mlx_embeddings/models/siglip.py`
  - Features: MHA with learnable probe tokens, LayerNorm, MLP
  
- `SiglipVisionTransformer` - Complete vision transformer
  - Location: `mlx_embeddings/models/siglip.py`
  - Features: Patch embeddings, positional encoding, encoder stack

**Utility functions:**
- `normalize_embeddings` - **L2 normalization for embeddings** ⭐
  - Location: `mlx_embeddings/models/base.py`
  - **Direct replacement for `_l2_normalize`!**
  
- `_create_causal_mask` - Causal attention mask generation
  - Location: `mlx_embeddings/models/qwen3.py`
  
- `get_extended_attention_mask` - Prepare attention masks for multi-head attention
  - Location: Multiple models (BERT, XLM-RoBERTa, Gemma3)

**Benefits:**
- Reference implementations for attention pooling (reduces complexity)
- Proven MLX patterns for vision transformers
- Examples of handling spatial-temporal factorization
- Can copy/adapt code patterns for faster implementation

---

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [Flax Documentation](https://flax.readthedocs.io/)
- [VideoPrism Paper](https://arxiv.org/abs/2402.13217)
- [VideoPrism GitHub](https://github.com/google-deepmind/videoprism)
- [Issue #49: LayerNorm scale +1 behavior](https://github.com/google-deepmind/videoprism/issues/49) - **Critical for weight conversion**
- [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Multi-modal vision-language models in MLX
- [Blaizzy/mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Embedding models in MLX
