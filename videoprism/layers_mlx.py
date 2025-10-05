# Copyright 2025 VideoPrism Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLX implementation of VideoPrism layers.

Converted from Flax implementation in layers.py.
Key differences:
- Uses MLX nn.Module instead of Flax nn.Module
- Uses __init__ instead of @nn.compact decorator
- LayerNorm scale weights will be handled during weight conversion (+1.0 offset)
- Uses mx.fast.scaled_dot_product_attention for efficient attention
"""

from typing import Callable, Optional
import math

import mlx.core as mx
import mlx.nn as nn


# Activation function type alias
ActivationFunc = Callable[[mx.array], mx.array]


def _convert_paddings_to_mask(paddings: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Converts padding indicators to additive attention mask.
    
    Args:
        paddings: Padding indicators of shape [batch, seq_len].
                  1.0 indicates padding, 0.0 indicates valid token.
        dtype: Data type for the mask.
    
    Returns:
        Additive attention mask of shape [batch, 1, 1, seq_len].
        0.0 for valid positions, -inf for masked positions.
    """
    # paddings: 1 where padded, 0 where valid
    pad_bool = paddings.astype(mx.bool_)
    # Shape: (B, 1, 1, L) — broadcast over heads and query length
    pad_mask = mx.where(
        pad_bool[:, None, None, :],
        mx.finfo(dtype).min,
        mx.array(0.0, dtype=dtype),
    )
    return pad_mask


def compute_attention_masks_for_fprop(
    inputs: mx.array,
    paddings: mx.array,
    causal_attention: bool = False,
) -> mx.array:
    """Computes additive attention masks for forward propagation.

    Returns an additive mask broadcastable to (B, H, Tq, Tk) with 0.0 for
    valid positions and -inf for masked positions.
    """
    seq_len = inputs.shape[1]
    dtype = inputs.dtype

    # Get padding mask: (B, 1, 1, L)
    pad_mask = _convert_paddings_to_mask(paddings, dtype=dtype)

    if causal_attention:
        # Shape: (1, 1, L, L)
        causal = nn.MultiHeadAttention.create_additive_causal_mask(seq_len, dtype=dtype)
        return pad_mask + causal

    return pad_mask


class LayerNorm(nn.Module):
    """Layer normalization wrapper."""
    
    def __init__(self, dims: int, eps: float = 1e-6, affine: bool = True, bias: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dims, eps=eps, affine=affine, bias=bias)
    
    def __call__(self, x: mx.array) -> mx.array:
        return self.norm(x)


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        activation_fn: ActivationFunc = nn.relu,
        use_bias: bool = True,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.activation_fn = activation_fn
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim, bias=use_bias)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else None
    
    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.activation_fn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc2(x)
        return x


class PerDimScale(nn.Module):
    """Per-dimension scaling layer."""
    
    def __init__(self, dim: int, init_scale: float = 1.0):
        super().__init__()
        self.scale = mx.full((dim,), init_scale)
    
    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale


@mx.compile
def _manual_attention_with_logit_cap(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    logit_cap: float,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Compiled manual attention with logit soft-capping.
    
    This function is compiled to fuse operations into optimized kernels.
    """
    # Compute attention logits
    logits = (q @ k.transpose(0, 1, 3, 2)) * scale
    
    # Apply tanh soft-capping if enabled
    if logit_cap > 0.0:
        logits = logit_cap * mx.tanh(logits / logit_cap)
    
    # Apply additive mask after soft-capping
    if mask is not None:
        logits = logits + mask
    
    # Compute attention weights and output
    attn_weights = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True).astype(logits.dtype)
    return attn_weights @ v


class DotProductAttention(nn.Module):
    """Multi-head dot-product attention with optional logit soft-capping.

    Notes
    -----
    * Supports additive masks (0.0 keep, -inf mask) broadcastable to (B, H, Tq, Tk).
    * If `atten_logit_cap > 0`, we apply: logits = cap * tanh(logits / cap) **before** softmax,
      and we add the mask **after** soft-capping so masked positions remain -inf.
    * If `atten_logit_cap <= 0` and `dropout_prob == 0`, we use the fast SDPA path for speed.
    * Manual path uses mx.compile for operation fusion and optimization.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dim_per_head: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        use_bias: bool = True,
        dropout_prob: float = 0.0,
        internal_enable_per_dim_scale: bool = True,  # kept for API compatibility
        atten_logit_cap: float = 0.0,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head if dim_per_head is not None else model_dim // num_heads
        self.total_dim = self.num_heads * self.dim_per_head
        self.hidden_dim = hidden_dim if hidden_dim is not None else model_dim
        self.use_qk_norm = use_qk_norm
        self.atten_logit_cap = float(atten_logit_cap) if atten_logit_cap is not None else 0.0

        # Projections
        self.q_proj = nn.Linear(model_dim, self.total_dim, bias=use_bias)
        self.k_proj = nn.Linear(model_dim, self.total_dim, bias=use_bias)
        self.v_proj = nn.Linear(model_dim, self.total_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.total_dim, self.hidden_dim, bias=use_bias)

        if use_qk_norm:
            # Normalize the last dim of per-head features
            self.q_norm = nn.RMSNorm(self.dim_per_head)
            self.k_norm = nn.RMSNorm(self.dim_per_head)

        # Dropout on attention *probabilities* (classic attention dropout)
        self.attn_dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else None

        # Optional per-dim scaling on Q (kept for parity with prior code)
        self.per_dim_scale = (
            PerDimScale(self.dim_per_head, init_scale=1.0 / math.sqrt(self.dim_per_head))
            if internal_enable_per_dim_scale else None
        )

    def _shape_qkv(self, q: mx.array, k: mx.array, v: mx.array):
        # (B, T, D) -> (B, H, T, Dh)
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        q = q.reshape(B, Tq, self.num_heads, self.dim_per_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, Tk, self.num_heads, self.dim_per_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, Tk, self.num_heads, self.dim_per_head).transpose(0, 2, 1, 3)
        return q, k, v

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        atten_mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, Optional[mx.array]]:
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        dtype = query.dtype

        # Projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q, k, v = self._shape_qkv(q, k, v)

        # Optional RMSNorm on Q/K per head
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Optional per-dim scale on Q
        if self.per_dim_scale is not None:
            q = self.per_dim_scale(q)

        # Compute scale factor: only scale if per_dim_scale was NOT applied
        # (to avoid double-scaling by 1/dim_per_head)
        scale = 1.0 if self.per_dim_scale is not None else 1.0 / math.sqrt(self.dim_per_head)

        # Fast path when no logit cap and no attn dropout
        if self.atten_logit_cap <= 0.0 and self.attn_dropout is None:
            # Ensure mask dtype promotes correctly (bf16/fp16 safety)
            if atten_mask is not None and atten_mask.dtype != dtype:
                atten_mask = atten_mask.astype(dtype)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=atten_mask)
            out = out.transpose(0, 2, 1, 3).reshape(B, Tq, self.total_dim)
            out = self.out_proj(out)
            return out, None

        # Manual path with logit capping
        # Ensure mask dtype matches
        if atten_mask is not None and atten_mask.dtype != dtype:
            atten_mask = atten_mask.astype(dtype)
        
        # Use compiled function when no dropout (for operation fusion and speed)
        if self.attn_dropout is None:
            # Compiled path: fuses matmul, tanh, mask, softmax, matmul into optimized kernels
            out = _manual_attention_with_logit_cap(q, k, v, scale, self.atten_logit_cap, atten_mask)
        else:
            # Uncompiled path with dropout (dropout must be applied to attention weights)
            logits = (q @ k.transpose(0, 1, 3, 2)) * scale
            if self.atten_logit_cap > 0.0:
                logits = self.atten_logit_cap * mx.tanh(logits / self.atten_logit_cap)
            if atten_mask is not None:
                logits = logits + atten_mask
            attn_weights = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True).astype(logits.dtype)
            attn_weights = self.attn_dropout(attn_weights)  # Dropout on attention weights
            out = attn_weights @ v
        
        # Reshape and project output
        out = out.transpose(0, 2, 1, 3).reshape(B, Tq, self.total_dim)
        out = self.out_proj(out)
        return out, None


class TransformerFeedForward(nn.Module):
    """Transformer feed-forward with residual dropout."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation_fn: ActivationFunc = nn.relu,
        use_bias: bool = True,
        relu_dropout_prob: float = 0.0,
        residual_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.ffn = FeedForward(input_dim, hidden_dim, input_dim, activation_fn, use_bias, relu_dropout_prob)
        self.residual_dropout = nn.Dropout(residual_dropout_prob) if residual_dropout_prob > 0.0 else None
    
    def __call__(self, x: mx.array) -> mx.array:
        output = self.ffn(x)
        if self.residual_dropout is not None:
            output = self.residual_dropout(output)
        return output


class Transformer(nn.Module):
    """Transformer layer with attention and feed-forward."""
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        hidden_dim: int,
        dim_per_head: Optional[int] = None,
        atten_dropout_prob: float = 0.0,
        residual_dropout_prob: float = 0.0,
        relu_dropout_prob: float = 0.0,
        norm_policy: str = 'pre',
        use_bias: bool = True,
        activation_fn: ActivationFunc = nn.relu,
        internal_enable_per_dim_scale: bool = True,
        atten_logit_cap: float = 0.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.norm_policy = norm_policy
        
        self.attention = DotProductAttention(
            model_dim, num_heads, dim_per_head, model_dim, use_bias,
            atten_dropout_prob, internal_enable_per_dim_scale, atten_logit_cap
        )
        self.ffn = TransformerFeedForward(
            model_dim, hidden_dim, activation_fn, use_bias,
            relu_dropout_prob, residual_dropout_prob
        )
        
        self.ln1 = LayerNorm(model_dim)
        self.ln2 = LayerNorm(model_dim)
        
        if norm_policy == 'primer_hybrid':
            self.ln1_post = LayerNorm(model_dim)
            self.ln2_post = LayerNorm(model_dim)
        
        self.residual_dropout = nn.Dropout(residual_dropout_prob) if residual_dropout_prob > 0.0 else None
    
    def __call__(self, inputs: mx.array, paddings: mx.array, atten_mask: mx.array) -> mx.array:
        # Self-attention block
        if self.norm_policy == 'pre':
            normed = self.ln1(inputs)
            atten_out, _ = self.attention(normed, normed, normed, atten_mask)
            if self.residual_dropout is not None:
                atten_out = self.residual_dropout(atten_out)
            inputs = inputs + atten_out
        elif self.norm_policy == 'post':
            atten_out, _ = self.attention(inputs, inputs, inputs, atten_mask)
            if self.residual_dropout is not None:
                atten_out = self.residual_dropout(atten_out)
            inputs = self.ln1(inputs + atten_out)
        elif self.norm_policy == 'primer_hybrid':
            normed = self.ln1(inputs)
            atten_out, _ = self.attention(normed, normed, normed, atten_mask)
            atten_out = self.ln1_post(atten_out)
            if self.residual_dropout is not None:
                atten_out = self.residual_dropout(atten_out)
            inputs = inputs + atten_out
        elif self.norm_policy == 'post_skip':
            atten_out, _ = self.attention(inputs, inputs, inputs, atten_mask)
            atten_out = self.ln1(atten_out)
            if self.residual_dropout is not None:
                atten_out = self.residual_dropout(atten_out)
            inputs = inputs + atten_out
        
        # Feed-forward block
        if self.norm_policy == 'pre':
            normed = self.ln2(inputs)
            ffn_out = self.ffn(normed)
            inputs = inputs + ffn_out
        elif self.norm_policy == 'post':
            ffn_out = self.ffn(inputs)
            inputs = self.ln2(inputs + ffn_out)
        elif self.norm_policy == 'primer_hybrid':
            normed = self.ln2(inputs)
            ffn_out = self.ffn(normed)
            ffn_out = self.ln2_post(ffn_out)
            inputs = inputs + ffn_out
        elif self.norm_policy == 'post_skip':
            ffn_out = self.ffn(inputs)
            ffn_out = self.ln2(ffn_out)
            inputs = inputs + ffn_out
        
        return inputs


class StackedTransformer(nn.Module):
    """Stack of transformer layers."""
    
    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        hidden_dim: int,
        dim_per_head: Optional[int] = None,
        dropout_prob: float = 0.0,
        atten_dropout_prob: Optional[float] = None,
        residual_dropout_prob: Optional[float] = None,
        relu_dropout_prob: Optional[float] = None,
        input_dropout_prob: float = 0.0,
        norm_policy: str = 'pre',
        use_bias: bool = True,
        activation_fn: ActivationFunc = nn.relu,
        internal_enable_per_dim_scale: bool = True,
        atten_logit_cap: float = 0.0,
        enable_causal_atten: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.enable_causal_atten = enable_causal_atten
        
        atten_dropout_prob = atten_dropout_prob or dropout_prob
        residual_dropout_prob = residual_dropout_prob or dropout_prob
        relu_dropout_prob = relu_dropout_prob or dropout_prob
        
        self.input_dropout = nn.Dropout(input_dropout_prob) if input_dropout_prob > 0.0 else None
        
        self.layers = []
        for _ in range(num_layers):
            layer = Transformer(
                model_dim, num_heads, hidden_dim, dim_per_head,
                atten_dropout_prob, residual_dropout_prob, relu_dropout_prob,
                norm_policy, use_bias, activation_fn,
                internal_enable_per_dim_scale, atten_logit_cap
            )
            self.layers.append(layer)
    
    def __call__(self, inputs: mx.array, paddings: mx.array) -> mx.array:
        atten_mask = compute_attention_masks_for_fprop(inputs, paddings, self.enable_causal_atten)
        outputs = inputs
        if self.input_dropout is not None:
            outputs = self.input_dropout(outputs)
        for layer in self.layers:
            outputs = layer(outputs, paddings, atten_mask)
        return outputs


class AttentionPoolingLayer(nn.Module):
    """Attention-based pooling with learnable queries.
    
    Reference: AttentionPoolLatent from mlx-vlm.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_queries: int = 1,
        num_heads: int = 1,
        hidden_dim: Optional[int] = None,
        query_dim: Optional[int] = None,
        use_bias: bool = True,
        dropout_prob: float = 0.0,
        add_layer_norm: bool = True,
        internal_enable_per_dim_scale: bool = True,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        self.num_queries = num_queries
        query_dim = query_dim or input_dim
        hidden_dim = hidden_dim if hidden_dim and hidden_dim > 0 else 4 * input_dim
        
        self.query = mx.random.normal((num_queries, query_dim))
        
        # Match Flax: all projections go to hidden_dim, then out projects back
        self.attention = nn.MultiHeadAttention(
            dims=hidden_dim,  # All projections output hidden_dim
            num_heads=num_heads,
            query_input_dims=query_dim,
            key_input_dims=input_dim,
            value_input_dims=input_dim,
            value_dims=hidden_dim,  # Explicitly set value dims
            bias=use_bias,
        )
        
        self.layer_norm = LayerNorm(hidden_dim) if add_layer_norm else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else None
    
    def __call__(self, tokens: mx.array, paddings: Optional[mx.array] = None) -> mx.array:
        batch_size, seq_length = tokens.shape[0], tokens.shape[1]
        
        query = mx.broadcast_to(self.query[None, :, :], (batch_size, self.num_queries, self.query.shape[-1]))
        
        if paddings is None:
            paddings = mx.zeros((batch_size, seq_length), dtype=tokens.dtype)
        
        atten_mask = _convert_paddings_to_mask(paddings, dtype=tokens.dtype)
        outputs = self.attention(query, tokens, tokens, mask=atten_mask)
        
        if self.layer_norm is not None:
            outputs = self.layer_norm(outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        
        return outputs
