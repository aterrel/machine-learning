"""CPU-safe tests for Numba-CUDA variants (demos/*/numba_*.py).

All tests run without a GPU. They verify:
- Module imports correctly even when Numba is absent
- Functions raise ImportError (not crash) when Numba is absent
- Shape and dtype contracts match the corresponding CPU baselines
- Output values agree with CPU reference implementations

GPU tests (marked @pytest.mark.gpu) actually launch Numba kernels.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Availability sentinels
# ---------------------------------------------------------------------------

try:
    import numba  # noqa: F401
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

_skip_no_numba = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blobs(n_per_class=100, n_features=8, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, n_features)) * 5.0
    X_parts, y_parts = [], []
    for c in range(n_classes):
        X_parts.append(rng.standard_normal((n_per_class, n_features)) * 0.5 + centers[c])
        y_parts.append(np.full(n_per_class, c, dtype=np.int32))
    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Import tests — modules must be importable even without Numba
# ---------------------------------------------------------------------------

def test_numba_vector_add_importable():
    mod = importlib.import_module("demos.01_core_apis.numba_vector_add")
    assert hasattr(mod, "vector_add_numba")
    assert hasattr(mod, "run_vector_add_numba")


def test_numba_kmeans_importable():
    mod = importlib.import_module("demos.02_kmeans.numba_kmeans")
    assert hasattr(mod, "kmeans_numba")


def test_numba_pca_importable():
    mod = importlib.import_module("demos.03_pca.numba_pca")
    assert hasattr(mod, "pca_numba")


def test_numba_linear_importable():
    mod = importlib.import_module("demos.04_linear_model.numba_linear")
    assert hasattr(mod, "linear_regression_numba")


def test_numba_nb_importable():
    mod = importlib.import_module("demos.05_naive_bayes.numba_nb")
    assert hasattr(mod, "gaussian_nb_numba")


def test_numba_kernels_importable():
    mod = importlib.import_module("demos.05_kernels.numba_kernels")
    assert hasattr(mod, "gemm_numba")
    assert hasattr(mod, "relu_numba")


# ---------------------------------------------------------------------------
# ImportError raised gracefully when Numba absent
# ---------------------------------------------------------------------------

def test_numba_vector_add_raises_when_absent():
    mod = importlib.import_module("demos.01_core_apis.numba_vector_add")
    orig = mod._NUMBA_AVAILABLE
    try:
        mod._NUMBA_AVAILABLE = False
        a = np.ones(16, dtype=np.float32)
        with pytest.raises(ImportError):
            mod.vector_add_numba(a, a)
    finally:
        mod._NUMBA_AVAILABLE = orig


def test_numba_kmeans_raises_when_absent():
    mod = importlib.import_module("demos.02_kmeans.numba_kmeans")
    orig = mod._NUMBA_AVAILABLE
    try:
        mod._NUMBA_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        with pytest.raises(ImportError):
            mod.kmeans_numba(X, k=2)
    finally:
        mod._NUMBA_AVAILABLE = orig


def test_numba_pca_raises_when_absent():
    mod = importlib.import_module("demos.03_pca.numba_pca")
    orig = mod._NUMBA_AVAILABLE
    try:
        mod._NUMBA_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        with pytest.raises(ImportError):
            mod.pca_numba(X, n_components=2)
    finally:
        mod._NUMBA_AVAILABLE = orig


def test_numba_linear_raises_when_absent():
    mod = importlib.import_module("demos.04_linear_model.numba_linear")
    orig = mod._NUMBA_AVAILABLE
    try:
        mod._NUMBA_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.float32)
        with pytest.raises(ImportError):
            mod.linear_regression_numba(X, y)
    finally:
        mod._NUMBA_AVAILABLE = orig


def test_numba_nb_raises_when_absent():
    mod = importlib.import_module("demos.05_naive_bayes.numba_nb")
    orig = mod._NUMBA_AVAILABLE
    try:
        mod._NUMBA_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.int32)
        with pytest.raises(ImportError):
            mod.gaussian_nb_numba(X, y, X, n_classes=2)
    finally:
        mod._NUMBA_AVAILABLE = orig


# ---------------------------------------------------------------------------
# GPU tests — actually launch Numba kernels
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@_skip_no_numba
def test_numba_vector_add_correct(gpu_device):
    mod = importlib.import_module("demos.01_core_apis.numba_vector_add")
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1024).astype(np.float32)
    b = rng.standard_normal(1024).astype(np.float32)
    result = mod.vector_add_numba(a, b)
    assert result.shape == (1024,)
    assert np.allclose(result, a + b, atol=1e-5)


@pytest.mark.gpu
@_skip_no_numba
def test_numba_kmeans_shape_and_type(gpu_device):
    mod = importlib.import_module("demos.02_kmeans.numba_kmeans")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    centroids, labels, inertia = mod.kmeans_numba(X, k=4, seed=0)
    assert centroids.shape == (4, 8)
    assert labels.shape == (200,)
    assert np.all(labels >= 0) and np.all(labels < 4)
    assert inertia >= 0.0


@pytest.mark.gpu
@_skip_no_numba
def test_numba_pca_shape(gpu_device):
    mod = importlib.import_module("demos.03_pca.numba_pca")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    comps, ev, X_t = mod.pca_numba(X, n_components=2)
    assert comps.shape == (2, 8)
    assert ev.shape == (2,)
    assert X_t.shape == (200, 2)


@pytest.mark.gpu
@_skip_no_numba
def test_numba_linear_correct(gpu_device):
    mod = importlib.import_module("demos.04_linear_model.numba_linear")
    cpu_mod = importlib.import_module("demos.04_linear_model.cpu_linear")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8)).astype(np.float32)
    w_true = rng.standard_normal(8).astype(np.float32)
    y = X @ w_true + 0.01 * rng.standard_normal(500).astype(np.float32)
    w_cpu, _, _ = cpu_mod.linear_regression_cpu(X, y)
    w_numba, _, _ = mod.linear_regression_numba(X, y)
    assert np.max(np.abs(w_cpu - w_numba)) < 1e-3


@pytest.mark.gpu
@_skip_no_numba
def test_numba_nb_predictions_shape(gpu_device):
    mod = importlib.import_module("demos.05_naive_bayes.numba_nb")
    X, y = _make_blobs(n_per_class=100, n_features=8, n_classes=3)
    split = int(0.8 * len(X))
    preds, means, vars_, lp = mod.gaussian_nb_numba(X[:split], y[:split], X[split:], n_classes=3)
    assert preds.shape == (len(X[split:]),)
    assert means.shape == (3, 8)
    assert vars_.shape == (3, 8)
    assert lp.shape == (3,)
