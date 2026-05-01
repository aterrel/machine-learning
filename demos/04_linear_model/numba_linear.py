"""Numba-CUDA linear regression: X^T X and X^T y on GPU, solve on CPU."""

from __future__ import annotations

import sys

import numpy as np

try:
    from numba import cuda as nb_cuda

    _NUMBA_AVAILABLE = True

    @nb_cuda.jit
    def _xtx_xty(X, y, XtX, Xty):
        """Compute X^T X and X^T y. 1D flat grid over n_features * n_features."""
        tid = nb_cuda.grid(1)
        n_features = X.shape[1]
        n_samples = X.shape[0]
        total = n_features * n_features
        if tid >= total:
            return
        i = tid // n_features
        j = tid % n_features
        s = 0.0
        for k in range(n_samples):
            s += float(X[k, i]) * float(X[k, j])
        XtX[i, j] = s
        # Compute Xty[i] once per row (when j == 0 to avoid races)
        if j == 0:
            t = 0.0
            for k in range(n_samples):
                t += float(X[k, i]) * float(y[k])
            Xty[i] = t

except ImportError:
    _NUMBA_AVAILABLE = False

BLOCK_SIZE = 256


def linear_regression_numba(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Numba-CUDA linear regression via normal equation.

    X^T X and X^T y computed on GPU; np.linalg.solve used on CPU.
    Returns (weights, r2_score, mse) matching linear_regression_gpu interface.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba is required. Install with: pip install numba")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n_samples, n_features = X.shape

    d_X = nb_cuda.to_device(X)
    d_y = nb_cuda.to_device(y)
    d_XtX = nb_cuda.to_device(np.zeros((n_features, n_features), dtype=np.float32))
    d_Xty = nb_cuda.to_device(np.zeros(n_features, dtype=np.float32))

    total = n_features * n_features
    grid = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    _xtx_xty[grid, BLOCK_SIZE](d_X, d_y, d_XtX, d_Xty)
    nb_cuda.synchronize()

    XtX = d_XtX.copy_to_host()
    Xty = d_Xty.copy_to_host()

    weights = np.linalg.solve(XtX.astype(np.float64), Xty.astype(np.float64)).astype(np.float32)

    y_pred = X @ weights.astype(np.float64)
    y_f64 = y.astype(np.float64)
    ss_res = float(np.sum((y_f64 - y_pred) ** 2))
    ss_tot = float(np.sum((y_f64 - y_f64.mean()) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    mse = ss_res / n_samples

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
    gpu_time = runner.time_cpu(linear_regression_numba, X, y)
    weights, r2, mse = linear_regression_numba(X, y)
    print(f"linear_regression_numba: {gpu_time:.3f}s | R²={r2:.4f} | MSE={mse:.6f}")
