"""CuPy k-means clustering (Lloyd's algorithm) using CuPy array operations."""

from __future__ import annotations

import sys

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def kmeans_cupy(
    X: np.ndarray,
    k: int = 8,
    max_iter: int = 100,
    seed: int = 42,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """CuPy k-means using broadcast distance computation and cp.argmin.

    Returns (centroids, labels, inertia) matching the kmeans_gpu interface.
    Distance formula: ||x - c||^2 = ||x||^2 - 2*x@c^T + ||c||^2
    avoids the (n_samples, k, n_features) intermediate tensor.
    """
    if not _CUPY_AVAILABLE:
        raise ImportError("cupy is required. Install with: pip install cupy-cuda12x")

    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)
    rng_empty = np.random.default_rng(seed + 1)

    idx = rng.choice(n_samples, size=k, replace=False)
    centroids = cp.asarray(X[idx].copy())
    X_gpu = cp.asarray(X)

    # Precompute squared norms of X rows: (n_samples,)
    X_sq = cp.sum(X_gpu ** 2, axis=1, keepdims=True)  # (n_samples, 1)

    labels = cp.zeros(n_samples, dtype=cp.int32)

    for _ in range(max_iter):
        # Squared distance: ||x - c||^2 = ||x||^2 - 2*x@c^T + ||c||^2
        c_sq = cp.sum(centroids ** 2, axis=1)           # (k,)
        cross = X_gpu @ centroids.T                      # (n_samples, k)
        dists = X_sq - 2.0 * cross + c_sq[cp.newaxis, :]  # (n_samples, k)
        new_labels = cp.argmin(dists, axis=1).astype(cp.int32)

        # Update centroids: compute mean per cluster
        new_centroids = cp.zeros((k, n_features), dtype=cp.float32)
        counts = cp.zeros(k, dtype=cp.float32)
        for j in range(k):
            mask = new_labels == j
            count = int(cp.sum(mask))
            if count > 0:
                new_centroids[j] = cp.mean(X_gpu[mask], axis=0)
                counts[j] = count
            else:
                # Empty cluster: reinitialise to random sample (CPU RNG for reproducibility)
                new_centroids[j] = X_gpu[rng_empty.integers(n_samples)]
                counts[j] = 1

        shift = float(cp.max(cp.linalg.norm(new_centroids - centroids, axis=1)))
        centroids = new_centroids
        labels = new_labels
        if shift < tol:
            break

    cp.cuda.Stream.null.synchronize()
    centroids_host = cp.asnumpy(centroids)
    labels_host = cp.asnumpy(labels).astype(np.int32)

    dists_final = np.sum((X - centroids_host[labels_host]) ** 2, axis=1)
    inertia = float(dists_final.sum())
    return centroids_host, labels_host, inertia


if __name__ == "__main__":
    try:
        from src.utils.timing import BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((5000, 16)).astype(np.float32)
    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    gpu_time = runner.time_cpu(kmeans_cupy, X, k=4, seed=42)
    centroids, labels, inertia = kmeans_cupy(X, k=4, seed=42)
    print(f"kmeans_cupy: {gpu_time:.3f}s | inertia={inertia:.1f} | centroids={centroids.shape}")
