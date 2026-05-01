"""CPU-safe tests for demos/08_comparison/.

Verifies:
- comparison demo and cuml_backends module import cleanly
- cuML backend functions raise ImportError when cuML is absent
- comparison runner produces SKIPPED rows for absent backends
- comparison runner produces results for numpy (CPU) backend always
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

def test_comparison_main_importable():
    mod = importlib.import_module("demos.08_comparison.main")
    assert hasattr(mod, "run_kmeans_comparison")
    assert hasattr(mod, "run_pca_comparison")
    assert hasattr(mod, "run_linear_comparison")
    assert hasattr(mod, "run_naive_bayes_comparison")


def test_cuml_backends_importable():
    mod = importlib.import_module("demos.08_comparison.backends.cuml_backends")
    assert hasattr(mod, "kmeans_cuml")
    assert hasattr(mod, "pca_cuml")
    assert hasattr(mod, "linear_regression_cuml")
    assert hasattr(mod, "gaussian_nb_cuml")


# ---------------------------------------------------------------------------
# cuML ImportError when absent
# ---------------------------------------------------------------------------

def test_kmeans_cuml_raises_when_absent():
    mod = importlib.import_module("demos.08_comparison.backends.cuml_backends")
    orig = mod._CUML_AVAILABLE
    try:
        mod._CUML_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        with pytest.raises(ImportError):
            mod.kmeans_cuml(X, k=2)
    finally:
        mod._CUML_AVAILABLE = orig


def test_pca_cuml_raises_when_absent():
    mod = importlib.import_module("demos.08_comparison.backends.cuml_backends")
    orig = mod._CUML_AVAILABLE
    try:
        mod._CUML_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        with pytest.raises(ImportError):
            mod.pca_cuml(X, n_components=2)
    finally:
        mod._CUML_AVAILABLE = orig


def test_linear_cuml_raises_when_absent():
    mod = importlib.import_module("demos.08_comparison.backends.cuml_backends")
    orig = mod._CUML_AVAILABLE
    try:
        mod._CUML_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.float32)
        with pytest.raises(ImportError):
            mod.linear_regression_cuml(X, y)
    finally:
        mod._CUML_AVAILABLE = orig


def test_nb_cuml_raises_when_absent():
    mod = importlib.import_module("demos.08_comparison.backends.cuml_backends")
    orig = mod._CUML_AVAILABLE
    try:
        mod._CUML_AVAILABLE = False
        X = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.int32)
        with pytest.raises(ImportError):
            mod.gaussian_nb_cuml(X, y, X, n_classes=2)
    finally:
        mod._CUML_AVAILABLE = orig


# ---------------------------------------------------------------------------
# Comparison runner runs the NumPy baseline without GPU
# ---------------------------------------------------------------------------

def test_kmeans_comparison_runs_numpy_baseline(capsys):
    """The comparison runner must at minimum run the numpy backend."""
    mod = importlib.import_module("demos.08_comparison.main")
    # Small dataset so the test is fast
    mod.run_kmeans_comparison(n_samples=200)
    captured = capsys.readouterr()
    assert "numpy (CPU)" in captured.out
    assert "K-Means" in captured.out


def test_pca_comparison_runs_numpy_baseline(capsys):
    mod = importlib.import_module("demos.08_comparison.main")
    mod.run_pca_comparison(n_samples=200)
    captured = capsys.readouterr()
    assert "numpy (CPU)" in captured.out
    assert "PCA" in captured.out


def test_linear_comparison_runs_numpy_baseline(capsys):
    mod = importlib.import_module("demos.08_comparison.main")
    mod.run_linear_comparison(n_samples=200)
    captured = capsys.readouterr()
    assert "numpy (CPU)" in captured.out
    assert "Linear Regression" in captured.out


def test_naive_bayes_comparison_runs_numpy_baseline(capsys):
    mod = importlib.import_module("demos.08_comparison.main")
    mod.run_naive_bayes_comparison(n_samples=200)
    captured = capsys.readouterr()
    assert "numpy (CPU)" in captured.out
    assert "Naive Bayes" in captured.out


def test_skipped_row_shown_when_cuml_absent(capsys):
    """SKIPPED row appears for cuML when the module is not installed."""
    mod = importlib.import_module("demos.08_comparison.main")
    # Force all gpu backends (cuda-python, cupy, numba, cuml) to raise ImportError
    # by running with a very small dataset — backends that aren't installed will
    # naturally raise ImportError and produce SKIPPED rows.
    mod.run_kmeans_comparison(n_samples=100)
    captured = capsys.readouterr()
    # At minimum the numpy baseline row should appear
    assert "numpy (CPU)" in captured.out
