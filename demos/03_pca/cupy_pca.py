"""CuPy PCA: mean-centering and covariance via CuPy array ops, eigendecomposition on GPU."""

from __future__ import annotations

import sys

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def pca_cupy(
    X: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CuPy PCA using cp.linalg.eigh for full GPU eigendecomposition.

    Unlike the cuda-python variant (which calls np.linalg.eigh on CPU),
    this version keeps the eigendecomposition on the GPU via cuSOLVER.

    Returns (components, explained_variance, X_transformed).
    """
    if not _CUPY_AVAILABLE:
        raise ImportError("cupy is required. Install with: pip install cupy-cuda12x")

    X_gpu = cp.asarray(X, dtype=cp.float32)
    n_samples = X_gpu.shape[0]

    # Mean-center
    means = X_gpu.mean(axis=0)
    X_c = X_gpu - means

    # Covariance matrix: X_c^T @ X_c / (n-1)
    C = (X_c.T @ X_c) / (n_samples - 1)

    # Eigendecomposition (cuSOLVER via CuPy) — stays on GPU
    eigenvalues, eigenvectors = cp.linalg.eigh(C)

    # Sort descending
    idx = cp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    components = eigenvectors[:, :n_components].T
    explained_variance = eigenvalues[:n_components]
    X_transformed = X_c @ eigenvectors[:, :n_components]

    cp.cuda.Stream.null.synchronize()
    return (
        cp.asnumpy(components).astype(np.float32),
        cp.asnumpy(explained_variance).astype(np.float32),
        cp.asnumpy(X_transformed).astype(np.float32),
    )


if __name__ == "__main__":
    try:
        from src.utils.timing import BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((2000, 16)).astype(np.float32)
    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    gpu_time = runner.time_cpu(pca_cupy, X, n_components=2)
    comps, ev, _ = pca_cupy(X, n_components=2)
    print(f"pca_cupy: {gpu_time:.3f}s | components={comps.shape} | expl_var={ev}")
