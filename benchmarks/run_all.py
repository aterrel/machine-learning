"""CLI benchmark runner: runs all available demos and prints a summary table."""

from __future__ import annotations

import argparse
import importlib
import sys


def _fmt_row(name: str, cpu_s: float, gpu_s: float, speedup: float, correct: bool) -> str:
    return (
        f"  {name:<20} CPU: {cpu_s:7.3f}s  GPU: {gpu_s:7.3f}s"
        f"  Speedup: {speedup:5.1f}x  Correct: {correct}"
    )


def run_01_core_apis(n_samples: int) -> list:
    results = []
    try:
        va_mod = importlib.import_module("demos.01_core_apis.vector_add")
        r = va_mod.run_vector_add(n=n_samples)
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  01_core_apis/vector_add FAILED: {exc}")
    return results


def run_02_kmeans(n_samples: int) -> list:
    results = []
    try:
        import numpy as np

        from src.utils.timing import BenchmarkResult, BenchmarkRunner

        cpu_mod = importlib.import_module("demos.02_kmeans.cpu_kmeans")
        gpu_mod = importlib.import_module("demos.02_kmeans.gpu_kmeans")

        rng = np.random.default_rng(42)
        X = rng.standard_normal((n_samples, 32)).astype(np.float32)
        runner = BenchmarkRunner(n_repeats=3, warmup=1)

        centroids_cpu, _, _ = cpu_mod.kmeans_cpu(X, k=8, seed=42)
        cpu_time = runner.time_cpu(cpu_mod.kmeans_cpu, X, k=8, seed=42)

        centroids_gpu, _, _ = gpu_mod.kmeans_gpu(X, k=8, seed=42)
        gpu_time = runner.time_cpu(gpu_mod.kmeans_gpu, X, k=8, seed=42)

        max_err = float(
            np.max(
                np.abs(np.sort(centroids_cpu, axis=0) - np.sort(centroids_gpu, axis=0))
            )
        )
        correct = max_err < 1e-3
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        r = BenchmarkResult(
            demo_name="kmeans",
            cpu_time_mean_s=cpu_time,
            gpu_time_mean_s=gpu_time,
            speedup=speedup,
            correct=correct,
            max_abs_error=max_err,
        )
        results.append(r)
        print(_fmt_row(r.demo_name, r.cpu_time_mean_s, r.gpu_time_mean_s, r.speedup, r.correct))
    except Exception as exc:
        print(f"  02_kmeans FAILED: {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all CUDA Python ML demo benchmarks")
    parser.add_argument(
        "--demo",
        choices=["01_core_apis", "02_kmeans", "all"],
        default="all",
    )
    parser.add_argument("--n-samples", type=int, default=10_000)
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

    print("=" * 70)
    print("CUDA Python ML Demos — Benchmark Summary")
    print("=" * 70)

    all_results = []

    if args.demo in ("01_core_apis", "all"):
        print("\n[01_core_apis]")
        all_results.extend(run_01_core_apis(args.n_samples))

    if args.demo in ("02_kmeans", "all"):
        print("\n[02_kmeans]")
        all_results.extend(run_02_kmeans(args.n_samples))

    print("\n" + "=" * 70)
    print(f"Total demos run: {len(all_results)}")
    all_correct = all(r.correct for r in all_results)
    print(f"All correct    : {all_correct}")
    print("=" * 70)


if __name__ == "__main__":
    main()
