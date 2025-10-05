"""Benchmark Flax vs MLX VideoPrism forward-pass performance.

Usage examples:

    python benchmark_performance.py --framework both --runs 20 --warmup 3
    python benchmark_performance.py --framework flax --device cpu --runs 5

The script reports per-run wall-clock timings (after warm-up) and peak
resident-set size observed during the process lifetime.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
import resource

import numpy as np


DEFAULT_MODEL_NAME = "videoprism_lvt_public_v1_base"
DEFAULT_VIDEO_PATH = Path("videoprism/assets/water_bottle_drumming.mp4")


def _load_video(path: Path, num_frames: int, target_size: int):
    from videoprism.video_utils import load_video

    return load_video(str(path), num_frames=num_frames, target_size=target_size)


def _format_stats(times: list[float]) -> str:
    if not times:
        return "(no samples)"
    mean = statistics.mean(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return (
        f"mean={mean:.4f}s  std={std:.4f}s  min={min(times):.4f}s  max={max(times):.4f}s"
    )


def _rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS returns bytes, Linux returns KiB. Detect via heuristic.
    rss = usage.ru_maxrss
    if rss > 1 << 32:  # already bytes
        return rss / (1024**3)
    return rss / (1024**2) / 1024.0


def benchmark_flax(args: argparse.Namespace):
    import jax
    import jax.numpy as jnp
    from videoprism import models as vp

    print("\n=== Flax/JAX benchmark ===")
    model = vp.get_model(args.model_name)
    state = vp.load_pretrained_weights(args.model_name)
    tokenizer = vp.load_text_tokenizer(args.text_tokenizer)

    video = _load_video(args.video_path, args.num_frames, args.target_size)
    video_inputs = jnp.array(video[None, ...])

    text_queries = args.text_queries.split("||")
    text_ids, text_paddings = vp.tokenize_texts(tokenizer, text_queries)
    text_ids = jnp.array(text_ids)
    text_paddings = jnp.array(text_paddings)

    @jax.jit
    def forward_fn(inputs, text_token_ids, text_token_paddings):
        return model.apply(
            state,
            inputs,
            text_token_ids,
            text_token_paddings,
            train=False,
            normalize=args.normalize,
        )

    def run_once():
        out = forward_fn(video_inputs, text_ids, text_paddings)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)

    # Warm-up
    for _ in range(args.warmup):
        run_once()

    durations = []
    for _ in range(args.runs):
        start = time.perf_counter()
        run_once()
        durations.append(time.perf_counter() - start)

    print("runs:", args.runs, " warmup:", args.warmup)
    print("timings:", [f"{t:.4f}" for t in durations])
    print("stats:", _format_stats(durations))
    print(f"ru_maxrss: {_rss_gb():.3f} GB")


def benchmark_mlx(args: argparse.Namespace):
    import mlx.core as mx
    from videoprism import models_mlx
    from videoprism import models as vp

    print("\n=== MLX benchmark ===")
    model = models_mlx.load_model(args.model_name)
    tokenizer = vp.load_text_tokenizer(args.text_tokenizer)

    video = _load_video(args.video_path, args.num_frames, args.target_size)
    video_inputs = mx.array(video)[None, ...]

    text_queries = args.text_queries.split("||")
    text_ids, text_paddings = vp.tokenize_texts(tokenizer, text_queries)
    text_ids = mx.array(text_ids)
    text_paddings = mx.array(text_paddings)

    def run_once():
        outputs = model(
            inputs=video_inputs,
            text_token_ids=text_ids,
            text_paddings=text_paddings,
            normalize=args.normalize,
        )
        mx.eval(*outputs)

    for _ in range(args.warmup):
        run_once()

    durations = []
    for _ in range(args.runs):
        start = time.perf_counter()
        run_once()
        durations.append(time.perf_counter() - start)

    print("runs:", args.runs, " warmup:", args.warmup)
    print("timings:", [f"{t:.4f}" for t in durations])
    print("stats:", _format_stats(durations))
    print(f"ru_maxrss: {_rss_gb():.3f} GB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--framework",
        choices=["flax", "mlx", "both"],
        default="both",
        help="Which implementation(s) to benchmark.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--target-size", type=int, default=288)
    parser.add_argument("--text-tokenizer", default="c4_en")
    parser.add_argument(
        "--text-queries",
        default="a person walking||drumming on water bottles||a car driving",
        help="Pipe-delimited list of text prompts used during benchmarking.",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--normalize", action="store_true", help="Return normalized embeddings.")

    args = parser.parse_args()

    if args.framework in {"flax", "both"}:
        benchmark_flax(args)
    if args.framework in {"mlx", "both"}:
        benchmark_mlx(args)


if __name__ == "__main__":
    main()
