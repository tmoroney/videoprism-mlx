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


def _get_large_negative_number(dtype: mx.Dtype = mx.float32) -> float:
    """Returns a large negative value for creating attention masks."""
    return -1e9


def _apply_mask_to_logits(logits: mx.array, mask: mx.array) -> mx.array:
    """Applies attention mask to logits."""
    large_negative = _get_large_negative_number(logits.dtype)
    return mx.where(mask, logits, large_negative)


def _convert_paddings_to_mask(paddings: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Converts padding indicators to attention mask."""
    attention_mask = 1.0 - paddings
    attention_mask = attention_mask[:, None, None, :]
    return attention_mask.astype(dtype)


def _causal_mask(seq_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Creates causal attention mask."""
    mask = mx.tril(mx.ones((seq_len, seq_len), dtype=dtype))
    return mask[None, None, :, :]


def _merge_masks(mask1: Optional[mx.array], mask2: Optional[mx.array]) -> Optional[mx.array]:
    """Merges two attention masks using logical AND."""
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1
    return mask1 * mask2


def compute_attention_masks_for_fprop(
    inputs: mx.array,
    paddings: mx.array,
    causal_attention: bool = False,
) -> mx.array:
    """Computes attention masks for forward propagation."""
    batch_size, seq_len = inputs.shape[0], inputs.shape[1]
    attention_mask = _convert_paddings_to_mask(paddings, dtype=inputs.dtype)
    
    if causal_attention:
        causal = _causal_mask(seq_len, dtype=inputs.dtype)
        padding_mask_expanded = mx.broadcast_to(attention_mask, (batch_size, 1, seq_len, seq_len))
        attention_mask = _merge_masks(padding_mask_expanded, causal)
    
    return attention_mask


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


class DotProductAttention(nn.Module):
    """Multi-head dot-product attention using mx.fast.scaled_dot_product_attention."""
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dim_per_head: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        use_bias: bool = True,
        dropout_prob: float = 0.0,
        internal_enable_per_dim_scale: bool = True,
        atten_logit_cap: float = 0.0,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head if dim_per_head is not None else model_dim // num_heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else model_dim
        self.atten_logit_cap = atten_logit_cap
        self.use_qk_norm = use_qk_norm
        self.total_dim = self.num_heads * self.dim_per_head
        
        self.q_proj = nn.Linear(model_dim, self.total_dim, bias=use_bias)
        self.k_proj = nn.Linear(model_dim, self.total_dim, bias=use_bias)
        self.v_proj = nn.Linear(model_dim, self.total_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.total_dim, self.hidden_dim, bias=use_bias)
        
        if internal_enable_per_dim_scale:
            self.per_dim_scale = PerDimScale(self.dim_per_head, init_scale=1.0 / math.sqrt(self.dim_per_head))
        else:
            self.per_dim_scale = None
        
        if use_qk_norm:
            self.q_norm = nn.RMSNorm(self.dim_per_head)
            self.k_norm = nn.RMSNorm(self.dim_per_head)
        
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else None
    
    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        atten_mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, Optional[mx.array]]:
        batch_size, query_len = query.shape[0], query.shape[1]
        kv_len = key.shape[1]
        
        q = self.q_proj(query).reshape(batch_size, query_len, self.num_heads, self.dim_per_head).transpose(0, 2, 1, 3)
        k = self.k_proj(key).reshape(batch_size, kv_len, self.num_heads, self.dim_per_head).transpose(0, 2, 1, 3)
        v = self.v_proj(value).reshape(batch_size, kv_len, self.num_heads, self.dim_per_head).transpose(0, 2, 1, 3)
        
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        if self.per_dim_scale is not None:
            q = self.per_dim_scale(q)
        
        scale = 1.0 / math.sqrt(self.dim_per_head)
        
        if self.atten_logit_cap > 0.0:
            attn_logits = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
            attn_logits = self.atten_logit_cap * mx.tanh(attn_logits / self.atten_logit_cap)
            if atten_mask is not None:
                attn_logits = _apply_mask_to_logits(attn_logits, atten_mask)
            attn_weights = mx.softmax(attn_logits, axis=-1)
            if self.dropout is not None:
                attn_weights = self.dropout(attn_weights)
            output = mx.matmul(attn_weights, v)
        else:
            if atten_mask is not None:
                additive_mask = _apply_mask_to_logits(mx.zeros_like(atten_mask), atten_mask)
            else:
                additive_mask = None
            output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=additive_mask)
            attn_weights = None
        
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_len, self.total_dim)
        output = self.out_proj(output)
        return output, attn_weights


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
        
        self.attention = DotProductAttention(
            model_dim=query_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            use_bias=use_bias,
            dropout_prob=0.0,
            internal_enable_per_dim_scale=internal_enable_per_dim_scale,
            use_qk_norm=use_qk_norm,
        )
        
        self.layer_norm = LayerNorm(hidden_dim) if add_layer_norm else None
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else None
    
    def __call__(self, tokens: mx.array, paddings: Optional[mx.array] = None) -> mx.array:
        batch_size, seq_length = tokens.shape[0], tokens.shape[1]
        
        query = mx.broadcast_to(self.query[None, :, :], (batch_size, self.num_queries, self.query.shape[-1]))
        
        if paddings is None:
            paddings = mx.zeros((batch_size, seq_length), dtype=tokens.dtype)
        
        atten_mask = _convert_paddings_to_mask(paddings, dtype=tokens.dtype)
        outputs, _ = self.attention(query, tokens, tokens, atten_mask)
        
        if self.layer_norm is not None:
            outputs = self.layer_norm(outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        
        return outputs
