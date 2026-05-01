"""CPU-safe tests for CuPy variants (demos/*/cupy_*.py).

All tests run without a GPU. They verify:
- Module imports correctly even when CuPy is absent
- Functions raise ImportError (not crash) when CuPy is absent
- Shape and dtype contracts match the corresponding CPU baselines
- Output values agree with CPU reference implementations

GPU tests (marked @pytest.mark.gpu) actually run CuPy operations.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Availability sentinels
# ---------------------------------------------------------------------------

try:
    import cupy  # noqa: F401
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

_skip_no_cupy = pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy not installed")


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
# Import tests — modules must be importable even without CuPy
# ---------------------------------------------------------------------------

def test_cupy_vector_add_importable():
    mod = importlib.import_module("demos.01_core_apis.cupy_vector_add")
    assert hasattr(mod, "vector_add_cupy")
    assert hasattr(mod, "vector_add_cupy_raw")
    assert hasattr(mod, "run_vector_add_cupy")


def test_cupy_kmeans_importable():
    mod = importlib.import_module("demos.02_kmeans.cupy_kmeans")
    assert hasattr(mod, "kmeans_cupy")


def test_cupy_pca_importable():
    mod = importlib.import_module("demos.03_pca.cupy_pca")
    assert hasattr(mod, "pca_cupy")


def test_cupy_linear_importable():
    mod = importlib.import_module("demos.04_linear_model.cupy_linear")
    assert hasattr(mod, "linear_regression_cupy")


def test_cupy_nb_importable():
    mod = importlib.import_module("demos.05_naive_bayes.cupy_nb")
    assert hasattr(mod, "gaussian_nb_cupy")


def test_cupy_kernels_importable():
    mod = importlib.import_module("demos.05_kernels.cupy_kernels")
    assert hasattr(mod, "gemm_cupy")
    assert hasattr(mod, "relu_cupy")


# ---------------------------------------------------------------------------
# ImportError raised gracefully when CuPy absent
# ---------------------------------------------------------------------------

def test_cupy_vector_add_raises_when_absent():
    mod = importlib.import_module("demos.01_core_apis.cupy_vector_add")
    orig = mod._CUPY_AVAILABLE
    try:
        mod._CUPY_AVAILABLE = False
        a = np.ones(16, dtype=np.float32)
        with pytest.raises(ImportError):
            mod.vector_add_cupy(a, a)
    finally:
        mod._CUPY_AVAILABLE = orig


def test_cupy_kmeans_raises_when_absent():
    mod = importlib.import_module("demos.02_kmeans.cupy_kmeans")
    orig = mod._CUPY_AVAILABLE
    try:
        mod._CUPY_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        with pytest.raises(ImportError):
            mod.kmeans_cupy(X, k=2)
    finally:
        mod._CUPY_AVAILABLE = orig


def test_cupy_pca_raises_when_absent():
    mod = importlib.import_module("demos.03_pca.cupy_pca")
    orig = mod._CUPY_AVAILABLE
    try:
        mod._CUPY_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        with pytest.raises(ImportError):
            mod.pca_cupy(X, n_components=2)
    finally:
        mod._CUPY_AVAILABLE = orig


def test_cupy_linear_raises_when_absent():
    mod = importlib.import_module("demos.04_linear_model.cupy_linear")
    orig = mod._CUPY_AVAILABLE
    try:
        mod._CUPY_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.float32)
        with pytest.raises(ImportError):
            mod.linear_regression_cupy(X, y)
    finally:
        mod._CUPY_AVAILABLE = orig


def test_cupy_nb_raises_when_absent():
    mod = importlib.import_module("demos.05_naive_bayes.cupy_nb")
    orig = mod._CUPY_AVAILABLE
    try:
        mod._CUPY_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.int32)
        with pytest.raises(ImportError):
            mod.gaussian_nb_cupy(X, y, X, n_classes=2)
    finally:
        mod._CUPY_AVAILABLE = orig


# ---------------------------------------------------------------------------
# GPU tests — actually run CuPy operations
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@_skip_no_cupy
def test_cupy_vector_add_correct(gpu_device):
    mod = importlib.import_module("demos.01_core_apis.cupy_vector_add")
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1024).astype(np.float32)
    b = rng.standard_normal(1024).astype(np.float32)
    result = mod.vector_add_cupy(a, b)
    assert result.shape == (1024,)
    assert np.allclose(result, a + b, atol=1e-5)


@pytest.mark.gpu
@_skip_no_cupy
def test_cupy_kmeans_shape_and_type(gpu_device):
    mod = importlib.import_module("demos.02_kmeans.cupy_kmeans")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    centroids, labels, inertia = mod.kmeans_cupy(X, k=4, seed=0)
    assert centroids.shape == (4, 8)
    assert labels.shape == (200,)
    assert np.all(labels >= 0) and np.all(labels < 4)
    assert inertia >= 0.0


@pytest.mark.gpu
@_skip_no_cupy
def test_cupy_pca_shape(gpu_device):
    mod = importlib.import_module("demos.03_pca.cupy_pca")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    comps, ev, X_t = mod.pca_cupy(X, n_components=2)
    assert comps.shape == (2, 8)
    assert ev.shape == (2,)
    assert X_t.shape == (200, 2)


@pytest.mark.gpu
@_skip_no_cupy
def test_cupy_linear_correct(gpu_device):
    mod = importlib.import_module("demos.04_linear_model.cupy_linear")
    cpu_mod = importlib.import_module("demos.04_linear_model.cpu_linear")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 8)).astype(np.float32)
    w_true = rng.standard_normal(8).astype(np.float32)
    y = X @ w_true + 0.01 * rng.standard_normal(500).astype(np.float32)
    w_cpu, _, _ = cpu_mod.linear_regression_cpu(X, y)
    w_cupy, r2, _ = mod.linear_regression_cupy(X, y)
    assert np.max(np.abs(w_cpu - w_cupy)) < 1e-3
    assert r2 > 0.99


@pytest.mark.gpu
@_skip_no_cupy
def test_cupy_nb_predictions_shape(gpu_device):
    mod = importlib.import_module("demos.05_naive_bayes.cupy_nb")
    X, y = _make_blobs(n_per_class=100, n_features=8, n_classes=3)
    split = int(0.8 * len(X))
    preds, means, vars_, lp = mod.gaussian_nb_cupy(X[:split], y[:split], X[split:], n_classes=3)
    assert preds.shape == (len(X[split:]),)
    assert means.shape == (3, 8)
    assert vars_.shape == (3, 8)
    assert lp.shape == (3,)


@pytest.mark.gpu
@_skip_no_cupy
def test_cupy_nb_agrees_with_cpu(gpu_device):
    cupy_mod = importlib.import_module("demos.05_naive_bayes.cupy_nb")
    cpu_mod = importlib.import_module("demos.05_naive_bayes.cpu_nb")
    X, y = _make_blobs(n_per_class=200, n_features=8, n_classes=3, seed=7)
    split = int(0.8 * len(X))
    cpu_preds, *_ = cpu_mod.gaussian_nb_cpu(X[:split], y[:split], X[split:], n_classes=3)
    gpu_preds, *_ = cupy_mod.gaussian_nb_cupy(X[:split], y[:split], X[split:], n_classes=3)
    agreement = float((cpu_preds == gpu_preds).mean())
    assert agreement > 0.99, f"CuPy/CPU agreement {agreement:.2%} < 99%"
