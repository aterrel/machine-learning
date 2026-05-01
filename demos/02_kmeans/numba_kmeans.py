"""Numba-CUDA k-means clustering (Lloyd's algorithm) using @cuda.jit kernels."""

from __future__ import annotations

import sys

import numpy as np

try:
    from numba import cuda as nb_cuda

    _NUMBA_AVAILABLE = True

    @nb_cuda.jit
    def _assign_labels(X, centroids, labels):
        """Assign each sample to the nearest centroid (1D grid over samples)."""
        i = nb_cuda.grid(1)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        k = centroids.shape[0]
        if i >= n_samples:
            return
        best_dist = 1e38
        best_label = 0
        for c in range(k):
            dist = 0.0
            for f in range(n_features):
                diff = X[i, f] - centroids[c, f]
                dist += diff * diff
            if dist < best_dist:
                best_dist = dist
                best_label = c
        labels[i] = best_label

    @nb_cuda.jit
    def _accumulate_centroids(X, labels, new_centroids, counts):
        """Accumulate per-class sums for centroid update using atomics."""
        i = nb_cuda.grid(1)
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if i >= n_samples:
            return
        c = labels[i]
        nb_cuda.atomic.add(counts, c, 1)
        for f in range(n_features):
            nb_cuda.atomic.add(new_centroids, (c, f), X[i, f])

except ImportError:
    _NUMBA_AVAILABLE = False

BLOCK_SIZE = 256


def kmeans_numba(
    X: np.ndarray,
    k: int = 8,
    max_iter: int = 100,
    seed: int = 42,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Numba-CUDA k-means. Returns (centroids, labels, inertia) matching kmeans_gpu interface."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba is required. Install with: pip install numba")

    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape
    rng = np.random.default_rng(seed)
    rng_empty = np.random.default_rng(seed + 1)

    idx = rng.choice(n_samples, size=k, replace=False)
    centroids = X[idx].copy()

    d_X = nb_cuda.to_device(X)
    d_labels = nb_cuda.device_array(n_samples, dtype=np.int32)

    grid = (n_samples + BLOCK_SIZE - 1) // BLOCK_SIZE

    for _ in range(max_iter):
        d_centroids = nb_cuda.to_device(centroids)
        _assign_labels[grid, BLOCK_SIZE](d_X, d_centroids, d_labels)
        nb_cuda.synchronize()

        new_centroids = np.zeros((k, n_features), dtype=np.float32)
        counts = np.zeros(k, dtype=np.int32)
        d_new_centroids = nb_cuda.to_device(new_centroids)
        d_counts = nb_cuda.to_device(counts)

        _accumulate_centroids[grid, BLOCK_SIZE](d_X, d_labels, d_new_centroids, d_counts)
        nb_cuda.synchronize()

        new_centroids = d_new_centroids.copy_to_host()
        counts = d_counts.copy_to_host()

        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
            else:
                new_centroids[j] = X[rng_empty.integers(n_samples)]

        shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
        centroids = new_centroids
        if shift < tol:
            break

    labels = d_labels.copy_to_host()
    dists_final = np.sum((X - centroids[labels]) ** 2, axis=1)
    inertia = float(dists_final.sum())
    return centroids, labels, inertia


if __name__ == "__main__":
    try:
        from src.utils.timing import BenchmarkResult, BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((5000, 16)).astype(np.float32)
    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    gpu_time = runner.time_cpu(kmeans_numba, X, k=4, seed=42)
    centroids, labels, inertia = kmeans_numba(X, k=4, seed=42)
    print(f"kmeans_numba: {gpu_time:.3f}s | inertia={inertia:.1f} | centroids={centroids.shape}")
