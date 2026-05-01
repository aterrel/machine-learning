"""Linear regression demo: GPU normal equation vs NumPy baseline."""

from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU linear regression demo via normal equation"
    )
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of data samples")
    parser.add_argument("--n-features", type=int, default=32, help="Number of features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        from src.utils.device import check_cuda_available
        from src.utils.timing import BenchmarkResult, BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: src/utils not available: {exc}")
        sys.exit(1)

    if not check_cuda_available():
        print(
            "ERROR: No CUDA GPU detected or cuda-python not installed.\n"
            "Install CUDA Toolkit 12.x and run: pip install cuda-python"
        )
        sys.exit(1)

    cpu_mod = importlib.import_module("demos.04_linear_model.cpu_linear")
    gpu_mod = importlib.import_module("demos.04_linear_model.gpu_linear")
    linear_regression_cpu = cpu_mod.linear_regression_cpu
    linear_regression_gpu = gpu_mod.linear_regression_gpu

    rng = np.random.default_rng(args.seed)
    true_w = rng.standard_normal(args.n_features).astype(np.float32)
    X = rng.standard_normal((args.n_samples, args.n_features)).astype(np.float32)
    noise = rng.standard_normal(args.n_samples).astype(np.float32) * 0.1
    y = (X @ true_w + noise).astype(np.float32)

    runner = BenchmarkRunner(n_repeats=5, warmup=1)

    # CPU baseline
    cpu_w, cpu_r2, cpu_mse = linear_regression_cpu(X, y)
    cpu_time = runner.time_cpu(linear_regression_cpu, X, y)

    # GPU version
    gpu_w, gpu_r2, gpu_mse = linear_regression_gpu(X, y)
    gpu_time = runner.time_cpu(linear_regression_gpu, X, y)

    # Compare weights within 1e-4
    max_err = float(np.max(np.abs(cpu_w - gpu_w)))
    correct = max_err < 1e-4

    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    result = BenchmarkResult(
        demo_name="linear_regression",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=correct,
        max_abs_error=max_err,
    )

    print("=" * 60)
    print("Linear Regression Demo — GPU Normal Equation")
    print("=" * 60)
    print(f"  n_samples  : {args.n_samples}")
    print(f"  n_features : {args.n_features}")
    print(f"  CPU R^2    : {cpu_r2:.6f}  MSE: {cpu_mse:.6f}")
    print(f"  GPU R^2    : {gpu_r2:.6f}  MSE: {gpu_mse:.6f}")
    print(f"  Max weight error : {max_err:.2e}")
    print(f"  Correct    : {correct}")
    print(f"\n{result.summary_line()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
