"""CuPy linear regression using cuBLAS matrix multiply and cuSOLVER linear solve."""

from __future__ import annotations

import sys

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def linear_regression_cupy(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """CuPy linear regression via normal equation: (X^T X) w = X^T y.

    Uses cp.linalg.solve (cuSOLVER) for the solve step — stays on GPU.
    Returns (weights, r2_score, mse) matching linear_regression_gpu interface.
    """
    if not _CUPY_AVAILABLE:
        raise ImportError("cupy is required. Install with: pip install cupy-cuda12x")

    X_gpu = cp.asarray(X, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.float64)
    n_samples = X_gpu.shape[0]

    XtX = X_gpu.T @ X_gpu           # (n_features, n_features)
    Xty = X_gpu.T @ y_gpu           # (n_features,)
    weights_gpu = cp.linalg.solve(XtX, Xty)

    y_pred = X_gpu @ weights_gpu
    ss_res = float(cp.sum((y_gpu - y_pred) ** 2))
    ss_tot = float(cp.sum((y_gpu - y_gpu.mean()) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    mse = ss_res / n_samples

    cp.cuda.Stream.null.synchronize()
    weights = cp.asnumpy(weights_gpu).astype(np.float32)
    return weights, float(r2_score), float(mse)


if __name__ == "__main__":
    try:
        from src.utils.timing import BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    rng = np.random.default_rng(42)
    n_features = 16
    X = rng.standard_normal((2000, n_features)).astype(np.float32)
    w_true = rng.standard_normal(n_features).astype(np.float32)
    y = X @ w_true + 0.01 * rng.standard_normal(2000).astype(np.float32)
    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    gpu_time = runner.time_cpu(linear_regression_cupy, X, y)
    weights, r2, mse = linear_regression_cupy(X, y)
    print(f"linear_regression_cupy: {gpu_time:.3f}s | R²={r2:.4f} | MSE={mse:.6f}")
