"""PCA demo: GPU covariance computation vs NumPy baseline."""

from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU PCA demo via covariance matrix")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of data samples")
    parser.add_argument("--n-features", type=int, default=16, help="Number of features")
    parser.add_argument("--n-components", type=int, default=2, help="Number of PCA components")
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

    cpu_mod = importlib.import_module("demos.03_pca.cpu_pca")
    gpu_mod = importlib.import_module("demos.03_pca.gpu_pca")
    pca_cpu = cpu_mod.pca_cpu
    pca_gpu = gpu_mod.pca_gpu

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n_samples, args.n_features)).astype(np.float32)

    runner = BenchmarkRunner(n_repeats=5, warmup=1)

    # CPU baseline
    cpu_comps, cpu_var, cpu_proj = pca_cpu(X, n_components=args.n_components)
    cpu_time = runner.time_cpu(pca_cpu, X, n_components=args.n_components)

    # GPU version
    gpu_comps, gpu_var, gpu_proj = pca_gpu(X, n_components=args.n_components)
    gpu_time = runner.time_cpu(pca_gpu, X, n_components=args.n_components)

    # Compare eigenvectors up to sign flip:
    # |diag(cpu_comps @ gpu_comps.T)| should be close to 1
    dot_diag = np.abs(np.diag(cpu_comps @ gpu_comps.T))
    correct = bool(np.all(dot_diag > 1.0 - 1e-3))
    max_err = float(np.max(np.abs(1.0 - dot_diag)))

    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    result = BenchmarkResult(
        demo_name="pca",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=correct,
        max_abs_error=max_err,
    )

    print("=" * 60)
    print("PCA Demo — GPU Covariance Matrix")
    print("=" * 60)
    print(f"  n_samples    : {args.n_samples}")
    print(f"  n_features   : {args.n_features}")
    print(f"  n_components : {args.n_components}")
    print(f"  Eigenvector alignment (|dot|): {dot_diag}")
    print(f"  Correct      : {correct}")
    print(f"  Max error    : {max_err:.2e}")
    print(f"\n{result.summary_line()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
