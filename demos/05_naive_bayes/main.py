"""Entry point for demo 05: GPU Gaussian Naive Bayes classifier."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _generate_data(
    n_train: int,
    n_test: int,
    n_features: int,
    n_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic multi-class Gaussian blob data.

    Each class has a cluster center drawn from N(0, 3.0*I) and points drawn
    from N(center, 0.5*I).  Samples per class are distributed as evenly as
    possible (last class absorbs remainder).
    """
    rng = np.random.default_rng(seed)

    # Draw n_classes cluster centers, spread by 3.0
    centers = rng.standard_normal((n_classes, n_features)) * 3.0

    def _make_split(n_total: int) -> tuple[np.ndarray, np.ndarray]:
        X_parts = []
        y_parts = []
        base = n_total // n_classes
        for c in range(n_classes):
            count = base if c < n_classes - 1 else n_total - base * (n_classes - 1)
            pts = centers[c] + rng.standard_normal((count, n_features)) * 0.5
            X_parts.append(pts.astype(np.float32))
            y_parts.append(np.full(count, c, dtype=np.int32))
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        # Shuffle
        perm = rng.permutation(len(y))
        return X[perm], y[perm]

    X_train, y_train = _make_split(n_train)
    X_test, y_test = _make_split(n_test)
    return X_train, y_train, X_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Gaussian Naive Bayes demo")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-classes", type=int, default=4)
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
    print("Demo 05: GPU Gaussian Naive Bayes Classifier")
    print("=" * 60)
    print(
        f"n_train={args.n_train}, n_test={args.n_test}, "
        f"n_features={args.n_features}, n_classes={args.n_classes}, seed={args.seed}"
    )

    X_train, y_train, X_test, y_test = _generate_data(
        args.n_train, args.n_test, args.n_features, args.n_classes, args.seed
    )

    # ------------------------------------------------------------------
    # CPU baseline
    # ------------------------------------------------------------------
    print("\nRunning CPU baseline (Gaussian Naive Bayes)...")
    from .cpu_nb import gaussian_nb_cpu

    t0 = time.perf_counter()
    cpu_preds, cpu_means, cpu_vars, cpu_log_priors = gaussian_nb_cpu(
        X_train, y_train, X_test, args.n_classes
    )
    cpu_time = time.perf_counter() - t0

    cpu_accuracy = float(np.mean(cpu_preds == y_test))
    print(f"CPU accuracy : {cpu_accuracy:.4f}  time: {cpu_time:.4f}s")

    # ------------------------------------------------------------------
    # GPU version
    # ------------------------------------------------------------------
    print("\nRunning GPU version (fit on CPU, predict on GPU)...")
    from .gpu_nb import gaussian_nb_gpu

    try:
        t0 = time.perf_counter()
        gpu_preds, gpu_means, gpu_vars, gpu_log_priors = gaussian_nb_gpu(
            X_train, y_train, X_test, args.n_classes
        )
        gpu_time = time.perf_counter() - t0
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: GPU inference failed: {exc}")
        sys.exit(1)

    gpu_accuracy = float(np.mean(gpu_preds == y_test))
    print(f"GPU accuracy : {gpu_accuracy:.4f}  time: {gpu_time:.4f}s")

    # ------------------------------------------------------------------
    # Compare CPU vs GPU predictions — should match exactly
    # ------------------------------------------------------------------
    n_agree = int(np.sum(cpu_preds == gpu_preds))
    agreement = n_agree / len(cpu_preds)
    predictions_match = n_agree == len(cpu_preds)
    print(f"\nCPU vs GPU prediction agreement: {n_agree}/{len(cpu_preds)} ({agreement:.4f})")
    if not predictions_match:
        print("WARNING: CPU and GPU predictions differ on some samples.")

    # ------------------------------------------------------------------
    # BenchmarkResult summary
    # ------------------------------------------------------------------
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    max_abs_error = float(np.max(np.abs(cpu_preds - gpu_preds)))

    from src.utils.timing import BenchmarkResult

    result = BenchmarkResult(
        demo_name="naive_bayes",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=predictions_match,
        max_abs_error=max_abs_error,
    )
    print("\n" + result.summary_line())
    print(f"CPU accuracy : {cpu_accuracy:.4f}")
    print(f"GPU accuracy : {gpu_accuracy:.4f}")
    print(f"Predictions match : {predictions_match}")


if __name__ == "__main__":
    main()
