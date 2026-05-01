"""NumPy k-means (Lloyd's algorithm) reference implementation."""

from __future__ import annotations

import numpy as np


def kmeans_cpu(
    X: np.ndarray,
    k: int = 8,
    max_iter: int = 100,
    seed: int = 42,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lloyd's k-means on CPU using NumPy.

    Returns (centroids (k, n_features), labels (n_samples,), inertia).
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    # Random centroid initialisation (same selection logic as GPU variant)
    idx = rng.choice(n_samples, size=k, replace=False)
    centroids = X[idx].copy().astype(np.float32)
    X = X.astype(np.float32)

    labels = np.empty(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        # Assignment step: squared Euclidean distance from each sample to each centroid
        # dists shape: (n_samples, k)
        dists = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)

        # Update step
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.float32)
        for j in range(k):
            mask = new_labels == j
            if mask.any():
                new_centroids[j] = X[mask].mean(axis=0)
                counts[j] = mask.sum()
            else:
                # Empty cluster: reinitialise to a random sample
                new_centroids[j] = X[rng.integers(n_samples)]
                counts[j] = 1

        shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids = new_centroids
        labels = new_labels

        if shift < tol:
            break

    # Compute inertia
    dists_final = np.sum((X - centroids[labels]) ** 2, axis=1)
    inertia = float(dists_final.sum())

    return centroids, labels, inertia
