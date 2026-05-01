"""Tests for the linear regression implementation.

CPU-only tests cover linear_regression_cpu: output shapes, weight recovery,
R² score, and MSE positivity.

GPU tests cover correctness parity between linear_regression_gpu and
linear_regression_cpu (weights within 1e-3 tolerance).
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

_cpu_linear_mod = importlib.import_module("demos.04_linear_model.cpu_linear")
linear_regression_cpu = _cpu_linear_mod.linear_regression_cpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regression(
    n_samples: int = 500,
    n_features: int = 4,
    noise_std: float = 0.01,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, w_true) for a synthetic linear regression problem.

    y = X @ w_true + noise, where noise ~ N(0, noise_std).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    w_true = rng.standard_normal(n_features).astype(np.float32)
    noise = rng.normal(0, noise_std, n_samples).astype(np.float32)
    y = (X @ w_true + noise).astype(np.float32)
    return X, y, w_true


# ---------------------------------------------------------------------------
# CPU-only tests — no GPU required
# ---------------------------------------------------------------------------


def test_linear_cpu_shape():
    """linear_regression_cpu returns weights with shape (n_features,)."""
    X, y, _ = _make_regression(n_samples=500, n_features=4, seed=1)
    weights, r2, mse = linear_regression_cpu(X, y)

    assert weights.shape == (4,), f"Expected weights shape (4,), got {weights.shape}"
    assert isinstance(r2, float), f"Expected r2 to be float, got {type(r2)}"
    assert isinstance(mse, float), f"Expected mse to be float, got {type(mse)}"


def test_linear_cpu_recovers_true_weights():
    """linear_regression_cpu recovers true weights within 0.1 for low-noise data."""
    X, y, w_true = _make_regression(n_samples=500, n_features=4, noise_std=0.01, seed=2)
    weights, _, _ = linear_regression_cpu(X, y)

    max_deviation = float(np.max(np.abs(weights - w_true)))
    assert max_deviation < 0.1, (
        f"Weight recovery error {max_deviation:.6f} exceeds threshold 0.1. "
        f"true={w_true}, estimated={weights}"
    )


def test_linear_cpu_r2_close_to_one():
    """R² must be > 0.99 for a low-noise linear regression problem."""
    X, y, _ = _make_regression(n_samples=500, n_features=4, noise_std=0.01, seed=3)
    _, r2, _ = linear_regression_cpu(X, y)

    assert r2 > 0.99, f"Expected R² > 0.99 for low-noise data, got R² = {r2:.6f}"


def test_linear_cpu_mse_positive():
    """MSE must be strictly positive for data with noise."""
    X, y, _ = _make_regression(n_samples=500, n_features=4, noise_std=0.01, seed=4)
    _, _, mse = linear_regression_cpu(X, y)

    assert mse > 0, f"Expected MSE > 0, got {mse}"


# ---------------------------------------------------------------------------
# GPU tests — require NVIDIA GPU (auto-skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_linear_gpu_matches_cpu(gpu_device):  # noqa: ARG001
    """GPU linear regression weights must match CPU weights within 1e-3 tolerance."""
    _gpu_linear_mod = importlib.import_module("demos.04_linear_model.gpu_linear")
    linear_regression_gpu = _gpu_linear_mod.linear_regression_gpu

    rng = np.random.default_rng(7)
    X = rng.standard_normal((400, 4)).astype(np.float32)
    w_true = rng.standard_normal(4).astype(np.float32)
    y = (X @ w_true + rng.normal(0, 0.01, 400)).astype(np.float32)

    cpu_weights, _, _ = linear_regression_cpu(X, y)
    gpu_weights, _, _ = linear_regression_gpu(X, y)

    max_diff = float(np.max(np.abs(cpu_weights - gpu_weights)))
    assert max_diff < 1e-3, (
        f"GPU vs CPU weight max difference is {max_diff:.6f}, expected < 1e-3"
    )
