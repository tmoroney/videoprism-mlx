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

"""MLX implementation of VideoPrism encoders.

Converted from Flax implementation in encoders.py.
Key differences:
- Uses MLX nn.Module instead of Flax nn.Module
- Uses __init__ instead of @nn.compact decorator
- No scan/remat (not needed in MLX)
- Simplified einops operations using MLX reshape/transpose
"""

from collections.abc import Collection, Sequence
from typing import Optional
import math

import mlx.core as mx
import mlx.nn as nn

from videoprism import layers_mlx as layers


def _contains(collection: Collection[str] | bool, key: str) -> bool:
    """Checks if a collection contains a key.
    
    Args:
        collection: A collection of strings or a boolean value.
        key: A string key to check.
    
    Returns:
        True if the collection contains the key, or if the collection is a True
        boolean. False otherwise.
    """
    return collection if isinstance(collection, bool) else key in collection


def _l2_normalize(
    x: mx.array, axis: int | Sequence[int] = -1, epsilon: float = 1e-12
) -> mx.array:
    """L2-normalizes an array along certain dimension.
    
    Args:
        x: An input array.
        axis: An integer or a sequence of integers for the axis to normalize.
        epsilon: A small constant for numerical stability.
    
    Returns:
        Normalized array.
    """
    x_dtype = x.dtype
    # Always convert to float32 for precision
    x = x.astype(mx.float32)
    norm = mx.sqrt(mx.sum(x * x, axis=axis, keepdims=True) + epsilon)
    return (x / norm).astype(x_dtype)


def _image_to_patch(inputs: mx.array, patch_size: int) -> mx.array:
    """Converts an image to patches.
    
    Args:
        inputs: An array of shape [B, H, W, C].
        patch_size: An integer for dimension of a square patch.
    
    Returns:
        batched_patches: [B, (H * W / P^2), P^2 * C].
    """
    if len(inputs.shape) < 4:
        raise ValueError(
            f'Image should be formatted as 4D [B, H, W, C], Shape: {inputs.shape}'
        )
    batch_size, height, width, channels = inputs.shape
    
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f'Image height ({height}) and width ({width}) should be multiples '
            f'of patch_size ({patch_size}).'
        )
    
    row_blocks = height // patch_size
    column_blocks = width // patch_size
    
    # Reshape: [B, H, W, C] -> [B, row_blocks, patch_size, column_blocks, patch_size, C]
    x = inputs.reshape(batch_size, row_blocks, patch_size, column_blocks, patch_size, channels)
    # Transpose to: [B, row_blocks, column_blocks, patch_size, patch_size, C]
    x = x.transpose(0, 1, 3, 2, 4, 5)
    # Reshape to: [B, num_patches, patch_size^2 * C]
    patches = x.reshape(batch_size, row_blocks * column_blocks, patch_size * patch_size * channels)
    
    return patches


def _interpolate_emb_1d(emb: mx.array, target_emb_length: int) -> mx.array:
    """Interpolates a 1D positional embedding to a new shape.
    
    Args:
        emb: An array of shape [1, L, D] or [L, D].
        target_emb_length: Target length for interpolation.
    
    Returns:
        target_emb: An array of shape [1, target_emb_length, D] or
                   [target_emb_length, D].
    """
    # Handle both [1, L, D] and [L, D] shapes
    if emb.ndim == 3:
        emb = emb[0]  # [L, D]
        had_batch = True
    else:
        had_batch = False
    
    source_length, emb_dim = emb.shape
    
    if source_length == target_emb_length:
        return emb[None, :, :] if had_batch else emb
    
    # Linear interpolation
    # Create interpolation indices
    source_indices = mx.arange(source_length, dtype=mx.float32)
    target_indices = mx.arange(target_emb_length, dtype=mx.float32)
    target_indices = target_indices * (source_length - 1) / (target_emb_length - 1)
    
    # Find lower and upper indices
    lower_indices = mx.floor(target_indices).astype(mx.int32)
    upper_indices = mx.clip(lower_indices + 1, 0, source_length - 1)
    lower_indices = mx.clip(lower_indices, 0, source_length - 1)
    
    # Compute interpolation weights
    weights = (target_indices - lower_indices.astype(mx.float32))[:, None]
    
    # Interpolate
    target_emb = emb[lower_indices] * (1 - weights) + emb[upper_indices] * weights
    
    return target_emb[None, :, :] if had_batch else target_emb


def _interpolate_emb_2d(
    emb: mx.array,
    source_emb_shape: tuple[int, int],
    target_emb_shape: tuple[int, int],
) -> mx.array:
    """Interpolates a 2D positional embedding to a new shape.
    
    Args:
        emb: An array of shape [1, source_h * source_w, D] or [source_h * source_w, D].
        source_emb_shape: Source (height, width) tuple.
        target_emb_shape: Target (height, width) tuple.
    
    Returns:
        target_emb: An array of shape [1, target_h * target_w, D] or
                   [target_h * target_w, D].
    """
    # Handle both [1, L, D] and [L, D] shapes
    if emb.ndim == 3:
        emb = emb[0]  # [L, D]
        had_batch = True
    else:
        had_batch = False
    
    source_h, source_w = source_emb_shape
    target_h, target_w = target_emb_shape
    emb_dim = emb.shape[-1]
    
    if (source_h, source_w) == (target_h, target_w):
        return emb[None, :, :] if had_batch else emb
    
    # Reshape to 2D grid: [source_h, source_w, D]
    emb_2d = emb.reshape(source_h, source_w, emb_dim)
    
    # Add batch dim for interpolation: [1, source_h, source_w, D]
    emb_2d = emb_2d[None, :, :, :]
    
    # For now, use simple bilinear interpolation via reshape
    # This is a simplified version - full bilinear would require more work
    # For exact match with Flax, we'd use jax.image.resize with 'linear' method
    # For MLX, we'll do a simpler approach that works for most cases
    
    # Transpose to [1, D, source_h, source_w] for processing
    emb_2d = emb_2d.transpose(0, 3, 1, 2)
    
    # Use linear interpolation for each dimension separately
    # First interpolate height
    if source_h != target_h:
        indices_h = mx.arange(target_h, dtype=mx.float32) * (source_h - 1) / (target_h - 1)
        lower_h = mx.floor(indices_h).astype(mx.int32)
        upper_h = mx.clip(lower_h + 1, 0, source_h - 1)
        weight_h = (indices_h - lower_h.astype(mx.float32))
        
        emb_lower = emb_2d[:, :, lower_h, :]
        emb_upper = emb_2d[:, :, upper_h, :]
        emb_2d = emb_lower * (1 - weight_h)[None, None, :, None] + emb_upper * weight_h[None, None, :, None]
    
    # Then interpolate width
    if source_w != target_w:
        indices_w = mx.arange(target_w, dtype=mx.float32) * (source_w - 1) / (target_w - 1)
        lower_w = mx.floor(indices_w).astype(mx.int32)
        upper_w = mx.clip(lower_w + 1, 0, source_w - 1)
        weight_w = (indices_w - lower_w.astype(mx.float32))
        
        emb_lower = emb_2d[:, :, :, lower_w]
        emb_upper = emb_2d[:, :, :, upper_w]
        emb_2d = emb_lower * (1 - weight_w)[None, None, None, :] + emb_upper * weight_w[None, None, None, :]
    
    # Transpose back: [1, target_h, target_w, D]
    emb_2d = emb_2d.transpose(0, 2, 3, 1)
    
    # Reshape to [target_h * target_w, D]
    target_emb = emb_2d.reshape(target_h * target_w, emb_dim)
    
    return target_emb[None, :, :] if had_batch else target_emb


class Embedding(nn.Module):
    """A simple embedding layer that performs embedding lookups from ids.
    
    Attributes:
        num_classes: Number of tokens in the vocabulary.
        input_dim: Depth of the embedding output.
        lookup_style: Style of lookup, one of 'index' or 'matmul'.
        scale_sqrt_depth: If True, activations are scaled with sqrt(embedding_dim).
        set_nan_for_oob_id: If True, out-of-boundary ids will be set to NaN.
    """
    
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        lookup_style: str = 'index',
        scale_sqrt_depth: bool = False,
        set_nan_for_oob_id: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.lookup_style = lookup_style
        self.scale_sqrt_depth = scale_sqrt_depth
        self.set_nan_for_oob_id = set_nan_for_oob_id
        
        # Initialize embedding table
        stddev = 1.0 / math.sqrt(input_dim)
        self.emb_var = mx.random.normal((num_classes, input_dim)) * stddev
    
    def __call__(self, ids: mx.array) -> mx.array:
        """Generates embedding lookup result.
        
        Args:
            ids: Indexes of shape [...] for embedding lookup.
        
        Returns:
            An array of shape [..., input_dim].
        """
        if self.lookup_style == 'index':
            embs = self.emb_var[ids]
        elif self.lookup_style == 'matmul':
            one_hot_ids = mx.one_hot(ids, self.num_classes)
            # [..., num_classes] @ [num_classes, input_dim] -> [..., input_dim]
            embs = one_hot_ids @ self.emb_var
        else:
            raise ValueError(f'Unknown lookup style: `{self.lookup_style}`.')
        
        # Map out-of-boundary ids to NaN
        if self.set_nan_for_oob_id:
            valid_mask = ids < self.num_classes
            embs = mx.where(valid_mask[..., None], embs, mx.nan)
        
        if self.scale_sqrt_depth:
            embs = embs * (self.input_dim ** 0.5)
        
        return embs


class PositionalEmbedding(nn.Module):
    """Generates sinusoidal position embedding for a given 1-d sequence.
    
    Uses MLX's built-in SinusoidalPositionalEncoding with parameter mapping:
    - min_freq = 1 / max_timescale
    - max_freq = 1 / min_timescale
    
    Attributes:
        embedding_dim: Dimension of the embedding to be generated.
        min_timescale: Start of the geometric index (default 1).
        max_timescale: End of the geometric index (default 10,000).
    """
    
    def __init__(
        self,
        embedding_dim: int,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Convert timescales to frequencies: freq = 1 / timescale
        min_freq = 1.0 / max_timescale
        max_freq = 1.0 / min_timescale
        
        # Use MLX's built-in sinusoidal positional encoding
        # scale=1.0 to match original behavior (no additional scaling)
        # cos_first=False means [sin; cos] order (matches original)
        self.pos_encoding = nn.SinusoidalPositionalEncoding(
            dims=embedding_dim,
            min_freq=min_freq,
            max_freq=max_freq,
            scale=1.0,
            cos_first=False,
        )
    
    def __call__(self, seq_length: int) -> mx.array:
        """Generates sinusoidal positional embeddings.
        
        Args:
            seq_length: Sequence length of the embeddings to be generated.
        
        Returns:
            An array of shape [1, seq_length, embedding_dim].
        """
        # Create position indices
        positions = mx.arange(seq_length)
        # Get embeddings: [seq_length, embedding_dim]
        embs = self.pos_encoding(positions)
        # Add batch dimension: [1, seq_length, embedding_dim]
        return embs[None, :, :]


class TrainablePositionalEmbedding(nn.Module):
    """Generates trainable position embedding for a given 1-d sequence.
    
    Attributes:
        embedding_dim: Dimension of the embedding to be generated.
        max_seq_length: Max sequence length.
        lookup_style: Style of lookup, currently only 'matmul' supported.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        max_seq_length: int = 10_240,
        lookup_style: str = 'matmul',
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.lookup_style = lookup_style
        
        # Initialize trainable positional embeddings
        # Using Lecun normal initialization
        stddev = 1.0 / math.sqrt(embedding_dim)
        self.emb_var = mx.random.normal((max_seq_length, embedding_dim)) * stddev
    
    def __call__(self, seq_length: int) -> mx.array:
        """Generates trainable positional embeddings.
        
        Args:
            seq_length: Sequence length of the embeddings to be generated.
        
        Returns:
            Position embeddings of shape [1, seq_length, embedding_dim].
        """
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds max_seq_length {self.max_seq_length}"
            )
        
        # Simple indexing to get position embeddings
        embs = self.emb_var[:seq_length]  # [seq_length, embedding_dim]
        
        # Add batch dimension
        return embs[None, :, :]  # [1, seq_length, embedding_dim]


class VisionTransformer(nn.Module):
    """Vision transformer model.
    
    This class follows a minimalistic design pattern. It's simply a wrapper
    around StackedTransformer for sequences of patches or embeddings.
    
    Attributes:
        num_tfm_layers: Number of transformer layers.
        mlp_dim: Hidden dimension of FFN in Transformer layers.
        num_heads: Number of attention heads.
        xformer_has_bias: Whether to use bias in transformer layers.
        xformer_dropout_prob: Dropout probability.
        xformer_atten_dropout_prob: Attention dropout probability.
        xformer_residual_dropout_prob: Residual dropout probability.
        xformer_relu_dropout_prob: FFN dropout probability.
        atten_logit_cap: Attention logit capping value.
        norm_policy: Normalization policy ('pre', 'post', 'primer_hybrid', 'post_skip').
    """
    
    def __init__(
        self,
        model_dim: int,
        num_tfm_layers: int = 12,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        xformer_has_bias: bool = True,
        xformer_dropout_prob: float = 0.0,
        xformer_atten_dropout_prob: Optional[float] = None,
        xformer_residual_dropout_prob: Optional[float] = None,
        xformer_relu_dropout_prob: Optional[float] = None,
        atten_logit_cap: float = 0.0,
        norm_policy: str = 'pre',
    ):
        super().__init__()
        self.model_dim = model_dim
        self.transformers_stack = layers.StackedTransformer(
            num_layers=num_tfm_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            hidden_dim=mlp_dim,
            dropout_prob=xformer_dropout_prob,
            atten_dropout_prob=xformer_atten_dropout_prob,
            residual_dropout_prob=xformer_residual_dropout_prob,
            relu_dropout_prob=xformer_relu_dropout_prob,
            use_bias=xformer_has_bias,
            atten_logit_cap=atten_logit_cap,
            norm_policy=norm_policy,
            internal_enable_per_dim_scale=False,
            activation_fn=nn.gelu,
            enable_causal_atten=False,
        )
    
    def __call__(
        self,
        inputs: mx.array,
        paddings: Optional[mx.array] = None,
    ) -> mx.array:
        """Applies the ViT model to the inputs.
        
        Args:
            inputs: Input tensor of shape [B, N, D], sequences of embeddings or patches.
            paddings: Optional [B, N] padding field (1 = padding, 0 = valid).
        
        Returns:
            Output tensor of shape [B, N, D].
        """
        if paddings is None:
            paddings = mx.zeros(inputs.shape[:-1], dtype=inputs.dtype)
        
        features = self.transformers_stack(inputs, paddings)
        return features


class FactorizedEncoder(nn.Module):
    """Factorized encoder from ViViT: A Video Vision Transformer.
    
    This implements model-2 from the paper - a factorized space-time encoder
    that first processes spatial information per frame, then temporal information
    across frames.
    
    Reference: https://arxiv.org/abs/2103.15691
    
    Attributes:
        patch_size: Size of square patches to extract from frames.
        pos_emb_shape: Shape of positional embeddings (T, H_patches, W_patches).
        model_dim: Model dimensionality.
        num_spatial_layers: Number of transformer layers for spatial encoding.
        num_temporal_layers: Number of transformer layers for temporal encoding.
        num_heads: Number of attention heads.
        mlp_dim: Hidden dimension of FFN in transformers.
        atten_logit_cap: Attention logit capping value.
        norm_policy: Normalization policy for transformers.
    """
    
    def __init__(
        self,
        patch_size: int = 18,
        pos_emb_shape: tuple[int, int, int] = (16, 16, 16),
        model_dim: int = 768,
        num_spatial_layers: int = 12,
        num_temporal_layers: int = 4,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        atten_logit_cap: float = 0.0,
        norm_policy: str = 'pre',
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pos_emb_shape = pos_emb_shape
        self.model_dim = model_dim
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers
        
        # Patch projection (simple linear layer)
        self.patch_projection = nn.Linear(
            patch_size * patch_size * 3,  # input dim
            model_dim,  # output dim  
            bias=True,
        )
        
        # Spatial positional embeddings
        spatial_pos_emb_shape = pos_emb_shape[-2:]
        spatial_seq_length = spatial_pos_emb_shape[0] * spatial_pos_emb_shape[1]
        self.spatial_pos_emb = TrainablePositionalEmbedding(
            embedding_dim=model_dim,
            max_seq_length=spatial_seq_length,
        )
        self.spatial_pos_emb_shape = spatial_pos_emb_shape
        
        # Spatial encoder
        self.spatial_encoder = VisionTransformer(
            model_dim=model_dim,
            num_tfm_layers=num_spatial_layers,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            atten_logit_cap=atten_logit_cap,
            norm_policy=norm_policy,
        )
        self.spatial_ln = layers.LayerNorm(model_dim)
        
        # Temporal positional embeddings
        temporal_seq_length = pos_emb_shape[0]
        self.temporal_pos_emb = TrainablePositionalEmbedding(
            embedding_dim=model_dim,
            max_seq_length=temporal_seq_length,
        )
        self.temporal_seq_length = temporal_seq_length
        
        # Temporal encoder
        self.temporal_encoder = VisionTransformer(
            model_dim=model_dim,
            num_tfm_layers=num_temporal_layers,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            atten_logit_cap=atten_logit_cap,
            norm_policy=norm_policy,
        )
        self.temporal_ln = layers.LayerNorm(model_dim)
    
    def __call__(
        self,
        inputs: mx.array,
        return_intermediate: bool | Collection[str] = False,
        frame_paddings: Optional[mx.array] = None,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Computes predictions for batched video inputs.
        
        Args:
            inputs: Input video tensor of shape [B, T, H, W, 3] (H == W).
            return_intermediate: Boolean or collection of intermediate feature names.
            frame_paddings: Optional binary tensor [B, T] (1 = padding frame).
        
        Returns:
            embeddings: Output tensor [B, T * N, D] where N is number of patches.
            outputs: Dictionary of intermediate features if requested.
        """
        b, t, h, w, c = inputs.shape
        assert h == w, "Height and width must be equal"
        
        # Reshape to process all frames: [B, T, H, W, C] -> [B*T, H, W, C]
        reshaped_inputs = inputs.reshape(b * t, h, w, c)
        
        # Tokenization: convert images to patches
        patches = _image_to_patch(reshaped_inputs, self.patch_size)  # [B*T, N, P²*C]
        
        # Handle frame paddings
        patches_paddings = None
        if frame_paddings is not None:
            assert frame_paddings.shape == (b, t)
            reshaped_frame_paddings = frame_paddings.reshape(b * t)  # [B*T]
            num_patches = patches.shape[1]
            # Repeat padding for each patch: [B*T] -> [B*T, N]
            patches_paddings = mx.repeat(reshaped_frame_paddings[:, None], num_patches, axis=1)
        
        embeddings, outputs = self.encode_with_patches(
            patches=patches,
            image_shape=(t, h, w),
            patches_paddings=patches_paddings,
            return_intermediate=return_intermediate,
        )
        return embeddings, outputs
    
    def encode_with_patches(
        self,
        patches: mx.array,
        image_shape: tuple[int, int, int],
        patches_paddings: Optional[mx.array] = None,
        return_intermediate: bool | Collection[str] = False,
    ) -> tuple[mx.array, dict[str, mx.array]]:
        """Computes predictions from patches.
        
        Args:
            patches: Input patches [B*T, N, P²*C] where N = (H*W)/P².
            image_shape: Original shape (T, H, W).
            patches_paddings: Optional binary tensor [B*T, N] (1 = padding).
            return_intermediate: Boolean or collection of feature names.
        
        Returns:
            embeddings: Output tensor [B, T*N, D].
            outputs: Dictionary of intermediate features.
        """
        t, h, w = image_shape
        bt, n, _ = patches.shape
        b = bt // t
        
        # Project patches to model dimension: [B*T, N, P²*C] -> [B*T, N, D]
        patches = self.patch_projection(patches)
        
        # Add spatial positional encoding
        num_row_patches = h // self.patch_size
        num_col_patches = w // self.patch_size
        spatial_seq_length = self.spatial_pos_emb_shape[0] * self.spatial_pos_emb_shape[1]
        spatial_pos_emb = self.spatial_pos_emb(spatial_seq_length)  # [1, L, D]
        
        # Interpolate if needed
        if self.spatial_pos_emb_shape != (num_row_patches, num_col_patches):
            spatial_pos_emb = _interpolate_emb_2d(
                spatial_pos_emb,
                self.spatial_pos_emb_shape,
                (num_row_patches, num_col_patches),
            )
        
        patches = patches + spatial_pos_emb  # [B*T, N, D]
        
        # Spatial encoding: process each frame independently
        features = self.spatial_encoder(patches, patches_paddings)  # [B*T, N, D]
        features = self.spatial_ln(features)
        spatial_features = features
        
        # Reshape for temporal processing: [B*T, N, D] -> [B*N, T, D]
        features = features.reshape(b, t, n, self.model_dim)  # [B, T, N, D]
        features = features.transpose(0, 2, 1, 3)  # [B, N, T, D]
        features = features.reshape(b * n, t, self.model_dim)  # [B*N, T, D]
        
        # Handle temporal paddings
        temporal_paddings = None
        if patches_paddings is not None:
            # [B*T, N] -> [B, T, N] -> [B, N, T] -> [B*N, T]
            temporal_paddings = patches_paddings.reshape(b, t, n)
            temporal_paddings = temporal_paddings.transpose(0, 2, 1)
            temporal_paddings = temporal_paddings.reshape(b * n, t)
        
        # Add temporal positional encoding
        temporal_pos_emb = self.temporal_pos_emb(self.temporal_seq_length)  # [1, L, D]
        if self.temporal_seq_length != t:
            temporal_pos_emb = _interpolate_emb_1d(temporal_pos_emb, t)
        features = features + temporal_pos_emb  # [B*N, T, D]
        
        # Temporal encoding: process temporal dimension
        features = self.temporal_encoder(features, temporal_paddings)  # [B*N, T, D]
        features = self.temporal_ln(features)
        
        # Reshape back: [B*N, T, D] -> [B, T*N, D]
        features = features.reshape(b, n, t, self.model_dim)  # [B, N, T, D]
        features = features.transpose(0, 2, 1, 3)  # [B, T, N, D]
        features = features.reshape(b, t * n, self.model_dim)  # [B, T*N, D]
        
        embeddings, outputs = features, {}
        if _contains(return_intermediate, 'spatial_features'):
            # Reshape spatial features: [B*T, N, D] -> [B, T*N, D]
            spatial_out = spatial_features.reshape(b, t, n, self.model_dim)
            spatial_out = spatial_out.reshape(b, t * n, self.model_dim)
            outputs['spatial_features'] = spatial_out
        
        return embeddings, outputs


class TextEncoder(nn.Module):
    """CoCa-style text encoder.
    
    Reference: https://arxiv.org/abs/2205.01917
    
    Attributes:
        vocabulary_size: Vocabulary size of text tokens.
        num_class_tokens: Number of class tokens to append.
        enable_causal_atten: Whether to enable causal attention.
        model_dim: Model dimensionality.
        num_layers: Number of transformer layers.
        mlp_dim: Hidden dimension of FFN.
        num_heads: Number of attention heads.
        atten_logit_cap: Attention logit capping value.
        norm_policy: Normalization policy.
        enable_per_dim_scale: Whether to enable per-dim scaling in attention.
    """
    
    def __init__(
        self,
        vocabulary_size: int = 128,
        num_class_tokens: int = 0,
        enable_causal_atten: bool = True,
        model_dim: int = 768,
        num_layers: int = 12,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        atten_logit_cap: float = 0.0,
        norm_policy: str = 'pre',
        enable_per_dim_scale: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_class_tokens = num_class_tokens
        
        # Positional embeddings (sinusoidal)
        self.pos_emb = PositionalEmbedding(embedding_dim=model_dim)
        
        # Token embeddings
        self.token_emb = Embedding(
            num_classes=vocabulary_size,
            input_dim=model_dim,
            scale_sqrt_depth=True,
        )
        
        # Class token embeddings if needed
        if num_class_tokens > 0:
            stddev = 1.0 / math.sqrt(model_dim)
            self.cls_emb = mx.random.normal((1, num_class_tokens, model_dim)) * stddev
        
        # Transformer stack
        self.unimodal_transformer = layers.StackedTransformer(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            hidden_dim=mlp_dim,
            atten_logit_cap=atten_logit_cap,
            norm_policy=norm_policy,
            internal_enable_per_dim_scale=enable_per_dim_scale,
            activation_fn=nn.relu,
            enable_causal_atten=enable_causal_atten,
        )
        self.unimodal_ln = layers.LayerNorm(model_dim)
    
    def __call__(self, inputs: mx.array, paddings: mx.array) -> mx.array:
        """Applies text encoder to token inputs.
        
        Args:
            inputs: Input tensor [B, N] with token ids.
            paddings: Padding tensor [B, N] (1 = padding, 0 = valid).
        
        Returns:
            Output tensor [B, N + num_class_tokens, D].
        """
        batch_size, seq_length = inputs.shape
        
        # Get positional and token embeddings
        pos_emb = self.pos_emb(seq_length)  # [1, N, D]
        input_emb = self.token_emb(inputs)  # [B, N, D]
        features = input_emb + pos_emb
        
        # Add class tokens if needed
        if self.num_class_tokens > 0:
            cls_emb = mx.broadcast_to(self.cls_emb, (batch_size, self.num_class_tokens, self.model_dim))
            cls_emb = cls_emb * (self.model_dim ** 0.5)
            features = mx.concatenate([features, cls_emb], axis=1)
            
            # Extend paddings for class tokens (class tokens are never padded)
            cls_paddings = mx.zeros((batch_size, self.num_class_tokens), dtype=paddings.dtype)
            paddings = mx.concatenate([paddings, cls_paddings], axis=1)
        
        # Apply transformer
        features = self.unimodal_transformer(features, paddings)
        features = self.unimodal_ln(features)
        
        return features


class FactorizedVideoCLIP(nn.Module):
    """Video CLIP model with factorized vision encoder.
    
    This is the main VideoPrism model that combines a factorized video encoder
    with a text encoder for contrastive learning.
    
    Attributes:
        patch_size: Patch size for vision encoder.
        pos_emb_shape: Positional embedding shape (T, H, W).
        num_spatial_layers: Number of spatial transformer layers.
        num_temporal_layers: Number of temporal transformer layers.
        mlp_dim: MLP hidden dimension.
        num_auxiliary_layers: Number of auxiliary vision layers (optional).
        vocabulary_size: Text vocabulary size.
        enable_causal_atten: Whether text encoder uses causal attention.
        num_unimodal_layers: Number of text transformer layers.
        norm_policy: Normalization policy.
        model_dim: Model dimensionality (shared between vision and text).
        num_heads: Number of attention heads.
        atten_logit_cap: Attention logit capping value.
    
    Example:
        >>> model = FactorizedVideoCLIP(model_dim=768, num_spatial_layers=12)
        >>> weights = mx.load("weights/model.safetensors")
        >>> model.load_weights(list(weights.items()))
        >>> video_emb, text_emb, _ = model(video_input, text_ids, text_paddings)
    """
    
    def __init__(
        self,
        # Vision parameters
        patch_size: int = 18,
        pos_emb_shape: tuple[int, int, int] = (16, 16, 16),
        num_spatial_layers: int = 12,
        num_temporal_layers: int = 4,
        mlp_dim: int = 3072,
        num_auxiliary_layers: int = 0,
        # Text parameters
        vocabulary_size: int = 128,
        enable_causal_atten: bool = True,
        num_unimodal_layers: int = 12,
        norm_policy: str = 'pre',
        # Shared parameters
        model_dim: int = 768,
        num_heads: int = 12,
        atten_logit_cap: float = 0.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_auxiliary_layers = num_auxiliary_layers
        
        # Vision encoder
        self.vision_encoder = FactorizedEncoder(
            patch_size=patch_size,
            pos_emb_shape=pos_emb_shape,
            model_dim=model_dim,
            num_spatial_layers=num_spatial_layers,
            num_temporal_layers=num_temporal_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            atten_logit_cap=atten_logit_cap,
            norm_policy='pre',  # Vision always uses 'pre'
        )
        
        # Auxiliary encoder (optional)
        if num_auxiliary_layers > 0:
            self.auxiliary_encoder = VisionTransformer(
                model_dim=model_dim,
                num_tfm_layers=num_auxiliary_layers,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                atten_logit_cap=atten_logit_cap,
                norm_policy='pre',
            )
        
        # Vision pooling layer
        self.contrastive_vision_pooler = layers.AttentionPoolingLayer(
            input_dim=model_dim,
            hidden_dim=model_dim * 4,
            num_heads=num_heads,
            num_queries=1,
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocabulary_size=vocabulary_size,
            num_class_tokens=1,
            enable_causal_atten=enable_causal_atten,
            model_dim=model_dim,
            num_layers=num_unimodal_layers,
            mlp_dim=model_dim * 4,
            num_heads=num_heads,
            atten_logit_cap=atten_logit_cap,
            norm_policy=norm_policy,
        )
    
    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        text_token_ids: Optional[mx.array] = None,
        text_paddings: Optional[mx.array] = None,
        normalize: bool = True,
        return_intermediate: bool | Collection[str] = False,
        frame_paddings: Optional[mx.array] = None,
    ) -> tuple[Optional[mx.array], Optional[mx.array], dict[str, mx.array]]:
        """Computes predictions for input batch.
        
        Args:
            inputs: Input video tensor [B, T, H, W, 3] (H == W).
            text_token_ids: Input text token ids [B, L].
            text_paddings: Text paddings [B, L]. Required if text_token_ids is not None.
            normalize: Whether to L2-normalize output embeddings.
            return_intermediate: Boolean or collection of intermediate feature names.
            frame_paddings: Optional binary tensor [B, T] (1 = padding frame).
        
        Returns:
            video_embeddings: Contrastive video embeddings [B, D]. None if inputs is None.
            text_embeddings: Contrastive text embeddings [B, D]. None if text_token_ids is None.
            outputs: Dictionary of intermediate outputs (spatial_features, etc.).
        """
        video_embeddings, text_embeddings, outputs = None, None, {}
        
        if inputs is not None:
            num_frames = inputs.shape[1]
            
            # Vision encoding
            vision_features, vision_outputs = self.vision_encoder(
                inputs,
                return_intermediate=return_intermediate,
                frame_paddings=frame_paddings,
            )
            outputs.update(vision_outputs)
            
            if _contains(return_intermediate, 'spatiotemporal_features'):
                outputs['spatiotemporal_features'] = vision_features
            
            # Auxiliary encoding (optional)
            if self.num_auxiliary_layers > 0:
                vision_features = self.auxiliary_encoder(vision_features)
            
            # Pool to get video embeddings
            video_embeddings = self.contrastive_vision_pooler(vision_features)
            
            # Squeeze query dimension: [B, 1, D] -> [B, D]
            video_embeddings = mx.squeeze(video_embeddings, axis=1)
            
            if normalize:
                video_embeddings = _l2_normalize(video_embeddings, axis=-1)
            
            # Frame embeddings (optional)
            if _contains(return_intermediate, 'frame_embeddings'):
                # Reshape: [B, T*N, D] -> [B*T, N, D]
                bt = vision_features.shape[0] // num_frames
                n = vision_features.shape[1] // num_frames
                frame_features = vision_features.reshape(bt * num_frames, n, self.model_dim)
                
                # Pool each frame
                frame_embeddings = self.contrastive_vision_pooler(frame_features)
                frame_embeddings = mx.squeeze(frame_embeddings, axis=1)  # [B*T, D]
                frame_embeddings = frame_embeddings.reshape(bt, num_frames, self.model_dim)  # [B, T, D]
                
                if normalize:
                    frame_embeddings = _l2_normalize(frame_embeddings, axis=-1)
                outputs['frame_embeddings'] = frame_embeddings
        
        if text_token_ids is not None:
            assert text_paddings is not None, 'Text paddings are required.'
            
            # Text encoding
            text_features = self.text_encoder(text_token_ids, text_paddings)
            
            # Take the last token (class token) as text embedding
            text_embeddings = text_features[:, -1]  # [B, D]
            
            if normalize:
                text_embeddings = _l2_normalize(text_embeddings, axis=-1)
        
        return video_embeddings, text_embeddings, outputs
