"""Tests for the PCA implementation.

CPU-only tests cover pca_cpu: output shapes, component orthogonality,
explained variance ordering and positivity, and determinism.

GPU tests cover correctness parity between pca_gpu and pca_cpu (sign-flip tolerant).
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

_cpu_pca_mod = importlib.import_module("demos.03_pca.cpu_pca")
pca_cpu = _cpu_pca_mod.pca_cpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n_samples: int = 200, n_features: int = 8, seed: int = 0) -> np.ndarray:
    """Return random float32 data of shape (n_samples, n_features)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features)).astype(np.float32)


# ---------------------------------------------------------------------------
# CPU-only tests — no GPU required
# ---------------------------------------------------------------------------


def test_pca_cpu_shape():
    """pca_cpu returns (components, explained_variance, X_transformed) with correct shapes."""
    X = _make_data(n_samples=200, n_features=8, seed=1)
    components, explained_variance, X_transformed = pca_cpu(X, n_components=2)

    assert components.shape == (2, 8), (
        f"Expected components shape (2, 8), got {components.shape}"
    )
    assert explained_variance.shape == (2,), (
        f"Expected explained_variance shape (2,), got {explained_variance.shape}"
    )
    assert X_transformed.shape == (200, 2), (
        f"Expected X_transformed shape (200, 2), got {X_transformed.shape}"
    )


def test_pca_cpu_components_orthogonal():
    """Principal components must be orthonormal: components @ components.T ≈ identity."""
    X = _make_data(n_samples=200, n_features=8, seed=2)
    components, _, _ = pca_cpu(X, n_components=2)

    gram = components @ components.T  # should be identity (2, 2)
    identity = np.eye(2, dtype=np.float32)

    np.testing.assert_allclose(
        gram,
        identity,
        atol=1e-6,
        err_msg="Principal components are not orthonormal: components @ components.T != I",
    )


def test_pca_cpu_explained_variance_positive():
    """All explained variance values (eigenvalues) must be strictly positive."""
    X = _make_data(n_samples=200, n_features=8, seed=3)
    _, explained_variance, _ = pca_cpu(X, n_components=4)

    assert np.all(explained_variance > 0), (
        f"Expected all explained variances > 0, got {explained_variance}"
    )


def test_pca_cpu_explained_variance_ordered():
    """Explained variances must be in non-increasing order."""
    X = _make_data(n_samples=200, n_features=8, seed=4)
    _, explained_variance, _ = pca_cpu(X, n_components=4)

    for i in range(len(explained_variance) - 1):
        assert explained_variance[i] >= explained_variance[i + 1], (
            f"Explained variance not ordered: {explained_variance[i]:.6f} < {explained_variance[i+1]:.6f}"
            f" at index {i}"
        )


def test_pca_cpu_deterministic():
    """Same input data must produce identical components on two independent calls."""
    X = _make_data(n_samples=200, n_features=8, seed=5)

    components_a, _, _ = pca_cpu(X, n_components=2)
    components_b, _, _ = pca_cpu(X, n_components=2)

    np.testing.assert_array_equal(
        components_a,
        components_b,
        err_msg="pca_cpu is not deterministic: same input produced different components",
    )


# ---------------------------------------------------------------------------
# GPU tests — require NVIDIA GPU (auto-skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_pca_gpu_matches_cpu(gpu_device):  # noqa: ARG001
    """GPU PCA components must match CPU components within sign-flip tolerance."""
    _gpu_pca_mod = importlib.import_module("demos.03_pca.gpu_pca")
    pca_gpu = _gpu_pca_mod.pca_gpu

    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 6)).astype(np.float32)

    cpu_components, _, _ = pca_cpu(X, n_components=3)
    gpu_components, _, _ = pca_gpu(X, n_components=3)

    # Compute diagonal of cpu_components @ gpu_components.T
    # Each entry should be close to +1 or -1 (sign-flip tolerant)
    dot_diag = np.abs(np.diag(cpu_components @ gpu_components.T))
    min_agreement = float(dot_diag.min())

    assert min_agreement > 1.0 - 1e-3, (
        f"GPU PCA components do not match CPU within sign-flip tolerance. "
        f"Min |diag(cpu @ gpu.T)| = {min_agreement:.6f}, expected > {1.0 - 1e-3:.6f}"
    )
