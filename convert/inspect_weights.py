"""Inspect Flax model weights structure and save to text file.

This script loads the VideoPrism Flax model weights and outputs:
1. Complete weight tree structure with shapes and dtypes
2. Hierarchical organization of parameters
3. Parameter counts and memory usage

Output is saved to 'flax_weights_structure.txt' for easy reference
during MLX conversion.
"""

import jax
import numpy as np
from videoprism import models as vp
from flax.traverse_util import flatten_dict, unflatten_dict
import json


def format_shape_dtype(arr):
    """Format array shape and dtype for display."""
    if hasattr(arr, 'shape') and hasattr(arr, 'dtype'):
        return f"shape={arr.shape}, dtype={arr.dtype}"
    return f"type={type(arr)}"


def count_parameters(params_dict):
    """Count total number of parameters."""
    total = 0
    for key, value in params_dict.items():
        if hasattr(value, 'size'):
            total += value.size
    return total


def get_memory_size(params_dict):
    """Calculate total memory size in MB."""
    total_bytes = 0
    for key, value in params_dict.items():
        if hasattr(value, 'nbytes'):
            total_bytes += value.nbytes
    return total_bytes / (1024 ** 2)  # Convert to MB


def print_nested_dict(d, output_file, indent=0, prefix=""):
    """Recursively print nested dictionary structure."""
    if isinstance(d, dict):
        for key, value in sorted(d.items()):
            full_key = f"{prefix}.{key}" if prefix else key
            indent_str = "  " * indent
            
            if isinstance(value, dict):
                # It's a nested dict
                output_file.write(f"{indent_str}📁 {key}/\n")
                print_nested_dict(value, output_file, indent + 1, full_key)
            else:
                # It's a leaf (actual parameter)
                info = format_shape_dtype(value)
                output_file.write(f"{indent_str}📄 {key}: {info}\n")
    else:
        # Non-dict value (shouldn't happen at top level)
        info = format_shape_dtype(d)
        output_file.write(f"{'  ' * indent}• {prefix}: {info}\n")


def inspect_flat_structure(flat_params, output_file):
    """Print flattened parameter structure."""
    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write("FLATTENED PARAMETER PATHS (for mapping reference)\n")
    output_file.write("=" * 80 + "\n\n")
    
    for key_path, value in sorted(flat_params.items()):
        key_str = " / ".join(str(k) for k in key_path)
        info = format_shape_dtype(value)
        output_file.write(f"{key_str}\n  → {info}\n\n")


def group_by_component(flat_params):
    """Group parameters by component (vision_encoder, text_encoder, etc.)."""
    groups = {}
    for key_path, value in flat_params.items():
        if len(key_path) > 0:
            component = str(key_path[0])
            if component not in groups:
                groups[component] = []
            groups[component].append((key_path, value))
    return groups


def print_component_summary(groups, output_file):
    """Print summary of parameters by component."""
    output_file.write("\n" + "=" * 80 + "\n")
    output_file.write("COMPONENT-WISE PARAMETER SUMMARY\n")
    output_file.write("=" * 80 + "\n\n")
    
    for component, params in sorted(groups.items()):
        total_params = sum(value.size for _, value in params if hasattr(value, 'size'))
        total_mb = sum(value.nbytes for _, value in params if hasattr(value, 'nbytes')) / (1024 ** 2)
        output_file.write(f"📦 {component}\n")
        output_file.write(f"   Parameters: {total_params:,}\n")
        output_file.write(f"   Memory: {total_mb:.2f} MB\n")
        output_file.write(f"   Layers: {len(params)}\n\n")


def main():
    print("=" * 80)
    print("VideoPrism Flax Weight Structure Inspector")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading model configuration...")
    model_name = 'videoprism_lvt_public_v1_base'
    flax_model = vp.get_model(model_name)
    print(f"      ✓ Model loaded: {model_name}")
    
    # Load weights
    print("\n[2/3] Loading pretrained weights...")
    loaded_state = vp.load_pretrained_weights(model_name)
    print("      ✓ Weights loaded")
    
    # Extract params
    if 'params' in loaded_state:
        params = loaded_state['params']
    else:
        params = loaded_state
    
    # Flatten for analysis
    flat_params = flatten_dict(params, sep='/')
    
    # Calculate statistics
    total_params = count_parameters(flat_params)
    total_memory = get_memory_size(flat_params)
    
    print(f"\n      Total parameters: {total_params:,}")
    print(f"      Total memory: {total_memory:.2f} MB")
    print(f"      Number of weight tensors: {len(flat_params)}")
    
    # Save to file
    print("\n[3/3] Writing structure to file...")
    output_path = "flax_weights_structure.txt"
    
    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("VideoPrism Flax Model Weight Structure\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Total Memory: {total_memory:.2f} MB\n")
        f.write(f"Number of Tensors: {len(flat_params)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Nested structure
        f.write("HIERARCHICAL PARAMETER STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        print_nested_dict(params, f)
        
        # Component summary
        groups = group_by_component(flat_params)
        print_component_summary(groups, f)
        
        # Flat structure for mapping
        inspect_flat_structure(flat_params, f)
        
        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF WEIGHT STRUCTURE\n")
        f.write("=" * 80 + "\n")
    
    print(f"      ✓ Structure saved to: {output_path}")
    
    # Also save as JSON for programmatic access
    json_path = "flax_weights_structure.json"
    flat_params_serializable = {
        '/'.join(str(k) for k in key): {
            'shape': list(value.shape) if hasattr(value, 'shape') else None,
            'dtype': str(value.dtype) if hasattr(value, 'dtype') else None,
        }
        for key, value in flat_params.items()
    }
    
    with open(json_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'total_parameters': int(total_params),
            'total_memory_mb': float(total_memory),
            'num_tensors': len(flat_params),
            'parameters': flat_params_serializable
        }, f, indent=2)
    
    print(f"      ✓ JSON structure saved to: {json_path}")
    
    print("\n" + "=" * 80)
    print("Inspection completed successfully!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Review '{output_path}' to understand weight structure")
    print(f"2. Create weight mapping from Flax names to MLX names")
    print(f"3. Implement conversion script to transform weights")
    print()


if __name__ == "__main__":
    main()
