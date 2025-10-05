"""MLX model loading utilities for VideoPrism.

This module provides convenient functions to load pre-trained VideoPrism models
in MLX format.
"""

import mlx.core as mx
from pathlib import Path
from .encoders_mlx import FactorizedVideoCLIP
from .weight_utils import load_and_convert_weights


# Model configurations
MODEL_CONFIGS = {
    'videoprism_lvt_public_v1_base': {
        'patch_size': 18,
        'pos_emb_shape': (16, 16, 16),
        'num_spatial_layers': 12,
        'num_temporal_layers': 4,
        'mlp_dim': 3072,
        'num_auxiliary_layers': 2,
        'vocabulary_size': 32000,
        'enable_causal_atten': True,
        'num_unimodal_layers': 12,
        'norm_policy': 'pre',
        'model_dim': 768,
        'num_heads': 12,
        'atten_logit_cap': 50.0,
    },
}


def get_model_config(model_name: str) -> dict:
    """Get model configuration by name.
    
    Args:
        model_name: Name of the model (e.g., 'videoprism_lvt_public_v1_base')
    
    Returns:
        Dictionary with model configuration parameters
    
    Raises:
        ValueError: If model name is not found
    """
    if model_name not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    return MODEL_CONFIGS[model_name].copy()


def load_model(model_name: str, weights_path: str = None) -> FactorizedVideoCLIP:
    """Load a pre-trained VideoPrism model in MLX format.
    
    Args:
        model_name: Name of the model configuration
        weights_path: Path to weights file (.safetensors or .npz).
                     If None, looks in default location: weights/{model_name}_mlx.safetensors
    
    Returns:
        Initialized model with loaded weights
    
    Example:
        >>> model = load_model('videoprism_lvt_public_v1_base')
        >>> video_emb, text_emb, _ = model(video, text_ids, text_paddings)
    """
    # Get model configuration
    config = get_model_config(model_name)
    
    # Initialize model
    print(f"Initializing {model_name}...")
    model = FactorizedVideoCLIP(**config)
    
    # Determine weights path
    if weights_path is None:
        weights_dir = Path("weights")
        # Try safetensors first, then npz
        weights_path = weights_dir / f"{model_name}_mlx.safetensors"
        if not weights_path.exists():
            weights_path = weights_dir / f"{model_name}_mlx.npz"
    else:
        weights_path = Path(weights_path)
    
    # Load weights
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. "
            f"Please run: python convert_weights.py"
        )
    
    print(f"Loading weights from {weights_path}...")
    if str(weights_path).endswith('.safetensors'):
        weights = mx.load(str(weights_path))
    else:
        weights = dict(mx.load(str(weights_path)))
    
    print(f"Converting weights to MLX format...")
    mlx_weights = load_and_convert_weights(weights)
    
    # Load into model
    model.update(mlx_weights)
    print(f"✓ Model loaded successfully")
    
    return model


def load_weights_from_file(filepath: str) -> dict:
    """Load weights from a file.
    
    Args:
        filepath: Path to weights file (.safetensors or .npz)
    
    Returns:
        Dictionary of weights (parameter name -> mx.array)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if filepath.suffix == '.safetensors':
        return mx.load(str(filepath))
    elif filepath.suffix == '.npz':
        return dict(mx.load(str(filepath)))
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
