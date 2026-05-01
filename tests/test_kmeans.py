"""Tests for the k-means clustering implementation.

CPU-only tests cover kmeans_cpu: output shapes, label validity, inertia,
determinism, and convergence on well-separated synthetic data.

GPU tests cover correctness parity between kmeans_gpu and kmeans_cpu.
All GPU tests use the gpu_device fixture and are marked @pytest.mark.gpu.

Imports for the demo modules use importlib because the directory names
start with digits, which are not valid Python identifiers.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module-level import via importlib (avoids numeric-prefix limitation)
# ---------------------------------------------------------------------------

_cpu_kmeans_mod = importlib.import_module("demos.02_kmeans.cpu_kmeans")
kmeans_cpu = _cpu_kmeans_mod.kmeans_cpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blobs(seed: int = 0) -> np.ndarray:
    """Return 1000 samples × 4 features, random float32."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((1000, 4)).astype(np.float32)


def _make_separated_clusters(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, true_centers) for 3 tight clusters far apart.

    Centers are at [0,0], [10,0], [0,10] with std=0.1.
    100 points per cluster → 300 total, 2 features.
    """
    rng = np.random.default_rng(seed)
    true_centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    parts = [rng.normal(c, 0.1, (100, 2)).astype(np.float32) for c in true_centers]
    X = np.vstack(parts)
    return X, true_centers


# ---------------------------------------------------------------------------
# CPU-only tests — no GPU required
# ---------------------------------------------------------------------------


def test_kmeans_cpu_shape():
    """kmeans_cpu returns (centroids, labels, inertia) with correct shapes."""
    X = _make_blobs(seed=1)
    centroids, labels, inertia = kmeans_cpu(X, k=3, seed=42)

    assert centroids.shape == (3, 4), f"Expected centroids shape (3, 4), got {centroids.shape}"
    assert labels.shape == (1000,), f"Expected labels shape (1000,), got {labels.shape}"
    assert inertia > 0, f"Expected inertia > 0, got {inertia}"


def test_kmeans_cpu_labels_in_range():
    """All label values must be in [0, k)."""
    X = _make_blobs(seed=2)
    k = 5
    _, labels, _ = kmeans_cpu(X, k=k, seed=42)

    assert labels.min() >= 0, f"Found negative label: {labels.min()}"
    assert labels.max() < k, f"Label {labels.max()} is out of range for k={k}"
    assert len(np.unique(labels)) <= k


def test_kmeans_cpu_inertia_positive():
    """Inertia must be strictly positive for non-trivial data."""
    X = _make_blobs(seed=3)
    _, _, inertia = kmeans_cpu(X, k=4, seed=42)
    assert inertia > 0.0, f"Inertia should be > 0, got {inertia}"


def test_kmeans_cpu_deterministic():
    """Same seed must produce identical centroids on two independent calls."""
    X = _make_blobs(seed=4)
    centroids_a, _, _ = kmeans_cpu(X, k=3, seed=99)
    centroids_b, _, _ = kmeans_cpu(X, k=3, seed=99)

    np.testing.assert_array_equal(
        centroids_a,
        centroids_b,
        err_msg="kmeans_cpu is not deterministic: same seed produced different centroids",
    )


def test_kmeans_cpu_convergence():
    """On well-separated clusters, each computed centroid must be near its true center."""
    X, true_centers = _make_separated_clusters(seed=0)
    centroids, _, _ = kmeans_cpu(X, k=3, seed=42)

    # Sort both by row-sum to match up cluster indices (order is arbitrary)
    computed_sorted = centroids[np.argsort(centroids.sum(axis=1))]
    true_sorted = true_centers[np.argsort(true_centers.sum(axis=1))]

    max_deviation = np.max(np.linalg.norm(computed_sorted - true_sorted, axis=1))
    assert max_deviation < 0.5, (
        f"Centroid deviation from true center is {max_deviation:.4f}, expected < 0.5"
    )


# ---------------------------------------------------------------------------
# GPU tests — require NVIDIA GPU (auto-skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_kmeans_gpu_matches_cpu(gpu_device):  # noqa: ARG001  (fixture ensures GPU is ready)
    """GPU k-means centroids must match CPU centroids within 1e-3 tolerance."""
    gpu_kmeans_mod = importlib.import_module("demos.02_kmeans.gpu_kmeans")
    kmeans_gpu = gpu_kmeans_mod.kmeans_gpu

    rng = np.random.default_rng(7)
    X = rng.standard_normal((500, 4)).astype(np.float32)
    k = 3
    seed = 42

    cpu_centroids, _, _ = kmeans_cpu(X, k=k, seed=seed)
    gpu_centroids, _, _ = kmeans_gpu(X, k=k, seed=seed)

    # Sort both by row-sum to handle arbitrary cluster-index permutations
    cpu_sorted = cpu_centroids[np.argsort(cpu_centroids.sum(axis=1))]
    gpu_sorted = gpu_centroids[np.argsort(gpu_centroids.sum(axis=1))]

    max_diff = float(np.max(np.abs(cpu_sorted - gpu_sorted)))
    assert max_diff < 1e-3, (
        f"GPU vs CPU centroid max difference is {max_diff:.6f}, expected < 1e-3"
    )
