"""Entry point for demo 02: GPU k-means clustering."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU k-means clustering demo")
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from src.utils.device import check_cuda_available
    except ImportError as exc:
        print(f"ERROR: src/utils not available: {exc}")
        sys.exit(1)

    if not check_cuda_available():
        print(
            "ERROR: No CUDA GPU detected or cuda-python not installed.\n"
            "Install CUDA Toolkit 12.x and run: pip install cuda-python"
        )
        sys.exit(1)

    print("=" * 60)
    print("Demo 02: GPU K-Means Clustering")
    print("=" * 60)
    print(f"n_samples={args.n_samples}, n_features={args.n_features}, k={args.k}, seed={args.seed}")

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n_samples, args.n_features)).astype(np.float32)

    print("\nRunning CPU baseline (Lloyd's algorithm)...")
    from src.utils.timing import BenchmarkRunner

    runner = BenchmarkRunner(n_repeats=3, warmup=1)

    from .cpu_kmeans import kmeans_cpu

    centroids_cpu, labels_cpu, inertia_cpu = kmeans_cpu(X, k=args.k, seed=args.seed)
    cpu_time = runner.time_cpu(kmeans_cpu, X, k=args.k, seed=args.seed)
    print(f"CPU inertia : {inertia_cpu:.2f}  time: {cpu_time:.3f}s")

    print("\nRunning GPU version...")
    from .gpu_kmeans import kmeans_gpu

    centroids_gpu, labels_gpu, inertia_gpu = kmeans_gpu(X, k=args.k, seed=args.seed)

    try:
        from src.utils.device import get_device

        device = get_device(0)
        stream = device.create_stream()
        from src.utils.timing import BenchmarkRunner as BR

        gpu_runner = BR(n_repeats=3, warmup=1)
        gpu_time = gpu_runner.time_gpu(
            lambda: kmeans_gpu(X, k=args.k, seed=args.seed), stream
        )
        stream.sync()
        stream.close()
    except Exception:
        import time

        t0 = time.perf_counter()
        kmeans_gpu(X, k=args.k, seed=args.seed)
        gpu_time = time.perf_counter() - t0

    print(f"GPU inertia : {inertia_gpu:.2f}  time: {gpu_time:.3f}s")

    max_centroid_err = float(
        np.max(np.abs(np.sort(centroids_cpu, axis=0) - np.sort(centroids_gpu, axis=0)))
    )
    correct = max_centroid_err < 1e-3
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    from src.utils.timing import BenchmarkResult

    result = BenchmarkResult(
        demo_name="kmeans",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=correct,
        max_abs_error=max_centroid_err,
    )
    print("\n" + result.summary_line())
    print(f"Max centroid error   : {max_centroid_err:.4f}")
    print(f"CPU cluster inertia  : {inertia_cpu:.2f}")
    print(f"GPU cluster inertia  : {inertia_gpu:.2f}")


if __name__ == "__main__":
    main()
