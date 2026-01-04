"""Extract text encoder weights from a full VideoPrism MLX weights file.

This is useful if you want to estimate (or actually create) a smaller checkpoint
containing only the text encoder parameters.

Example:
  python scripts/extract_text_encoder_weights.py \
    --input weights/videoprism_lvt_public_v1_base_mlx.safetensors

  python scripts/extract_text_encoder_weights.py \
    --input weights/videoprism_lvt_public_v1_base_mlx.npz \
    --output weights/videoprism_lvt_public_v1_base_text_encoder_mlx.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import mlx.core as mx


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KiB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MiB"
    return f"{num_bytes / (1024**3):.2f} GiB"


def _prod(shape) -> int:
    if shape is None:
        return 0
    return int(math.prod(int(x) for x in shape))


def _dtype_nbytes(dtype) -> int | None:
    # Best-effort dtype size mapping without importing numpy.
    # MLX dtype string values are typically like: 'float16', 'float32', 'bfloat16'.
    name = str(dtype)
    table = {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 2,
        "uint32": 4,
        "int32": 4,
        "uint64": 8,
        "int64": 8,
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
    }
    return table.get(name)


def _estimate_tensor_bytes(x: mx.array) -> int | None:
    try:
        # Some MLX builds expose nbytes.
        return int(x.nbytes)  # type: ignore[attr-defined]
    except Exception:
        pass

    itemsize = _dtype_nbytes(x.dtype)
    if itemsize is None:
        return None
    return _prod(x.shape) * itemsize


def _load_weights(path: Path) -> Dict[str, mx.array]:
    if path.suffix == ".safetensors":
        return dict(mx.load(str(path)))
    if path.suffix == ".npz":
        return dict(mx.load(str(path)))
    raise ValueError(f"Unsupported file format: {path.suffix} (expected .safetensors or .npz)")


def _save_weights(path: Path, weights: Dict[str, mx.array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".safetensors":
        mx.save_safetensors(str(path), weights)
        return

    if path.suffix == ".npz":
        mx.savez(str(path), **weights)
        return

    raise ValueError(f"Unsupported output format: {path.suffix} (expected .safetensors or .npz)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text encoder weights (text_encoder/*) from a full VideoPrism MLX checkpoint."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input MLX weights (.safetensors or .npz)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path to output weights file. If omitted, derived from input path as "
            "<stem>_text_encoder<suffix>."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="text_encoder/",
        help="Key prefix to extract (default: text_encoder/)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input weights not found: {input_path}")

    if args.output is None:
        output_path = input_path.with_name(f"{input_path.stem}_text_encoder{input_path.suffix}")
    else:
        output_path = Path(args.output)

    print(f"Loading weights: {input_path}")
    weights = _load_weights(input_path)

    prefix = str(args.prefix)
    extracted = {k: v for k, v in weights.items() if k.startswith(prefix)}

    if not extracted:
        sample = "\n".join(sorted(list(weights.keys()))[:25])
        raise ValueError(
            f"No weights matched prefix '{prefix}'. "
            f"First keys in checkpoint:\n{sample}"
        )

    # Stats
    input_size = input_path.stat().st_size
    extracted_param_count = sum(_prod(v.shape) for v in extracted.values())

    estimated_bytes = 0
    estimated_bytes_missing = 0
    for v in extracted.values():
        est = _estimate_tensor_bytes(v)
        if est is None:
            estimated_bytes_missing += 1
        else:
            estimated_bytes += est

    print("Summary:")
    print(f"  input_tensors: {len(weights):,}")
    print(f"  extracted_tensors: {len(extracted):,}")
    print(f"  extracted_parameters: {extracted_param_count:,}")
    print(f"  input_file_size: {_format_bytes(input_size)}")
    if estimated_bytes_missing == 0:
        print(f"  extracted_tensor_bytes_est: {_format_bytes(estimated_bytes)}")
    else:
        print(
            "  extracted_tensor_bytes_est: "
            f"{_format_bytes(estimated_bytes)} (missing {estimated_bytes_missing} dtype estimates)"
        )

    print(f"Saving extracted weights: {output_path}")
    _save_weights(output_path, extracted)

    output_size = output_path.stat().st_size
    print("Done:")
    print(f"  output_file_size: {_format_bytes(output_size)}")
    print(f"  output_path: {output_path}")


if __name__ == "__main__":
    main()
