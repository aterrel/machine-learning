"""CPU-only tests for Gaussian Naive Bayes (demos/05_naive_bayes/)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

nb_mod = importlib.import_module("demos.05_naive_bayes.cpu_nb")
gaussian_nb_cpu = nb_mod.gaussian_nb_cpu


def _make_blobs(n_per_class: int = 200, n_features: int = 8, n_classes: int = 3, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, n_features)) * 5.0
    X_parts, y_parts = [], []
    for c in range(n_classes):
        X_parts.append(rng.standard_normal((n_per_class, n_features)) * 0.5 + centers[c])
        y_parts.append(np.full(n_per_class, c, dtype=np.int32))
    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


@pytest.fixture
def blobs_3class():
    X, y = _make_blobs(n_per_class=300, n_features=8, n_classes=3)
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


def test_gaussian_nb_cpu_prediction_shape(blobs_3class):
    X_tr, y_tr, X_te, _ = blobs_3class
    preds, means, vars_, priors = gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=3)
    assert preds.shape == (len(X_te),)


def test_gaussian_nb_cpu_labels_in_range(blobs_3class):
    X_tr, y_tr, X_te, _ = blobs_3class
    preds, *_ = gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=3)
    assert np.all(preds >= 0) and np.all(preds < 3)


def test_gaussian_nb_cpu_statistics_shapes(blobs_3class):
    X_tr, y_tr, X_te, _ = blobs_3class
    n_features = X_tr.shape[1]
    _, means, vars_, priors = gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=3)
    assert means.shape == (3, n_features)
    assert vars_.shape == (3, n_features)
    assert priors.shape == (3,)


def test_gaussian_nb_cpu_variances_positive(blobs_3class):
    X_tr, y_tr, X_te, _ = blobs_3class
    _, _, vars_, _ = gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=3)
    assert np.all(vars_ > 0)


def test_gaussian_nb_cpu_priors_sum_to_one(blobs_3class):
    X_tr, y_tr, X_te, _ = blobs_3class
    _, _, _, log_priors = gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=3)
    assert abs(np.exp(log_priors).sum() - 1.0) < 1e-6


def test_gaussian_nb_cpu_high_accuracy_on_separated_blobs(blobs_3class):
    X_tr, y_tr, X_te, y_te = blobs_3class
    preds, *_ = gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=3)
    accuracy = float((preds == y_te).mean())
    assert accuracy > 0.90, f"Expected >90% accuracy on well-separated blobs, got {accuracy:.2%}"


def test_gaussian_nb_cpu_deterministic(blobs_3class):
    X_tr, y_tr, X_te, _ = blobs_3class
    p1, *_ = gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=3)
    p2, *_ = gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=3)
    assert np.array_equal(p1, p2)


@pytest.mark.gpu
def test_gaussian_nb_gpu_matches_cpu(blobs_3class, gpu_device):
    X_tr, y_tr, X_te, _ = blobs_3class
    gpu_mod = importlib.import_module("demos.05_naive_bayes.gpu_nb")
    cpu_preds, *_ = gaussian_nb_cpu(X_tr, y_tr, X_te, n_classes=3)
    gpu_preds, *_ = gpu_mod.gaussian_nb_gpu(X_tr, y_tr, X_te, n_classes=3)
    agreement = float((cpu_preds == gpu_preds).mean())
    assert agreement > 0.99, f"GPU/CPU prediction agreement {agreement:.2%} < 99%"
