"""MLX model loading utilities for VideoPrism.

This module provides convenient functions to load pre-trained VideoPrism models
in MLX format.
"""

import mlx.core as mx
from pathlib import Path
from .encoders_mlx import FactorizedVideoCLIP, FactorizedVideoClassifier
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


def load_classifier(
    model_name: str, 
    num_classes: int, 
    weights_path: str = None
) -> FactorizedVideoClassifier:
    """Load a VideoPrism video classifier in MLX format.
    
    This loads the factorized encoder backbone with attention pooling and a 
    classification head. Useful for fine-tuning on video classification tasks.
    
    Args:
        model_name: Name of the model configuration (e.g., 'videoprism_lvt_public_v1_base')
        num_classes: Number of output classes for the classifier
        weights_path: Path to weights file (.safetensors or .npz).
                     If None, looks in default location: weights/{model_name}_mlx.safetensors
                     Note: Pre-trained weights only include encoder, not classification head.
    
    Returns:
        Initialized classifier model with encoder weights loaded
    
    Example:
        >>> # For fine-tuning on a custom dataset with 10 classes
        >>> model = load_classifier('videoprism_lvt_public_v1_base', num_classes=10)
        >>> logits, intermediate = model(video, return_intermediate=True)
    """
    # Get model configuration
    config = get_model_config(model_name)
    
    # Extract encoder parameters (remove CLIP-specific params)
    encoder_params = {
        'patch_size': config['patch_size'],
        'pos_emb_shape': config['pos_emb_shape'],
        'model_dim': config['model_dim'],
        'num_spatial_layers': config['num_spatial_layers'],
        'num_temporal_layers': config['num_temporal_layers'],
        'num_heads': config['num_heads'],
        'mlp_dim': config['mlp_dim'],
        'atten_logit_cap': config['atten_logit_cap'],
        'norm_policy': config['norm_policy'],
    }
    
    # Initialize classifier
    print(f"Initializing {model_name} classifier with {num_classes} classes...")
    model = FactorizedVideoClassifier(**encoder_params, num_classes=num_classes)
    
    # Determine weights path
    if weights_path is None:
        weights_dir = Path("weights")
        # Try safetensors first, then npz
        weights_path = weights_dir / f"{model_name}_mlx.safetensors"
        if not weights_path.exists():
            weights_path = weights_dir / f"{model_name}_mlx.npz"
    else:
        weights_path = Path(weights_path)
    
    # Load weights (optional - for fine-tuning from pre-trained encoder)
    if weights_path.exists():
        print(f"Loading encoder weights from {weights_path}...")
        if str(weights_path).endswith('.safetensors'):
            weights = mx.load(str(weights_path))
        else:
            weights = dict(mx.load(str(weights_path)))
        
        print(f"Converting weights to MLX format...")
        mlx_weights = load_and_convert_weights(weights)
        
        # Filter to only encoder weights (ignore text encoder, auxiliary, etc.)
        encoder_weights = {}
        for key, value in mlx_weights.items():
            # Map vision_encoder weights to encoder
            if key.startswith('vision_encoder.'):
                new_key = key.replace('vision_encoder.', 'encoder.')
                encoder_weights[new_key] = value
        
        # Load encoder weights (classification head will remain randomly initialized)
        model.update(encoder_weights)
        print(f"✓ Encoder weights loaded. Classification head initialized randomly.")
    else:
        print(f"⚠ Weights not found at {weights_path}. Model initialized with random weights.")
        print(f"  To use pre-trained encoder, run: python convert_weights.py")
    
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
