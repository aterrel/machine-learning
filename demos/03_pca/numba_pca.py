"""Numba-CUDA PCA: mean-centering and covariance on GPU, eigendecomposition on CPU."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

try:
    from numba import cuda as nb_cuda

    _NUMBA_AVAILABLE = True

    @nb_cuda.jit
    def _mean_center(X, means):
        """Subtract per-feature mean from each element. 1D flat grid over n_samples*n_features."""
        i = nb_cuda.grid(1)
        n_features = X.shape[1]
        total = X.shape[0] * n_features
        if i >= total:
            return
        row = i // n_features
        col = i % n_features
        X[row, col] -= means[col]

    @nb_cuda.jit
    def _compute_cov(X_c, C):
        """Compute symmetric covariance matrix. 1D flat grid over n_features*n_features."""
        tid = nb_cuda.grid(1)
        n_features = X_c.shape[1]
        n_samples = X_c.shape[0]
        total = n_features * n_features
        if tid >= total:
            return
        i = tid // n_features
        j = tid % n_features
        s = 0.0
        for k in range(n_samples):
            s += float(X_c[k, i]) * float(X_c[k, j])
        C[i, j] = s / (n_samples - 1)

except ImportError:
    _NUMBA_AVAILABLE = False

BLOCK_SIZE = 256


def pca_numba(
    X: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba-CUDA PCA. Returns (components, explained_variance, X_transformed).

    Mean-centering and covariance computed on GPU via @cuda.jit kernels.
    Eigendecomposition performed on CPU (same as cuda-python variant).
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba is required. Install with: pip install numba")

    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape

    means = X.mean(axis=0).astype(np.float32)
    X_c = X.copy()

    d_X_c = nb_cuda.to_device(X_c)
    d_means = nb_cuda.to_device(means)
    d_C = nb_cuda.to_device(np.zeros((n_features, n_features), dtype=np.float32))

    n_total = n_samples * n_features
    grid_1d = (n_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    _mean_center[grid_1d, BLOCK_SIZE](d_X_c, d_means)
    nb_cuda.synchronize()

    grid_cov = (n_features * n_features + BLOCK_SIZE - 1) // BLOCK_SIZE
    _compute_cov[grid_cov, BLOCK_SIZE](d_X_c, d_C)
    nb_cuda.synchronize()

    X_c_host = d_X_c.copy_to_host()
    C_host = d_C.copy_to_host()

    eigenvalues, eigenvectors = np.linalg.eigh(C_host.astype(np.float64))
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].astype(np.float32)
    eigenvectors = eigenvectors[:, idx].astype(np.float32)

    components = eigenvectors[:, :n_components].T
    explained_variance = eigenvalues[:n_components]
    X_transformed = X_c_host @ eigenvectors[:, :n_components]

    return components, explained_variance, X_transformed


if __name__ == "__main__":
    try:
        from src.utils.timing import BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 16)).astype(np.float32)
    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    gpu_time = runner.time_cpu(pca_numba, X, n_components=2)
    comps, ev, _ = pca_numba(X, n_components=2)
    print(f"pca_numba: {gpu_time:.3f}s | components={comps.shape} | expl_var={ev}")
