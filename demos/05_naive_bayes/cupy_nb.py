"""CuPy Gaussian Naive Bayes: vectorized log-likelihood using CuPy broadcasting."""

from __future__ import annotations

import sys

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


def gaussian_nb_cupy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CuPy Gaussian Naive Bayes: fit on CPU, vectorized inference on GPU.

    Log-likelihood computed via broadcasting:
      log_probs[i, c] = log_prior[c] + sum_f(-0.5*log(2*pi*var[c,f]) - 0.5*(x[i,f]-mu[c,f])^2/var[c,f])

    Returns (predictions, means, vars_, log_priors) matching gaussian_nb_cpu interface.
    """
    if not _CUPY_AVAILABLE:
        raise ImportError("cupy is required. Install with: pip install cupy-cuda12x")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    n_train, n_features = X_train.shape

    # Fit on CPU (same as other variants)
    means = np.zeros((n_classes, n_features), dtype=np.float64)
    vars_ = np.zeros((n_classes, n_features), dtype=np.float64)
    log_priors = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask = y_train == c
        Xc = X_train[mask].astype(np.float64)
        means[c] = Xc.mean(axis=0)
        vars_[c] = Xc.var(axis=0) + 1e-9
        log_priors[c] = np.log(mask.sum() / n_train)

    means_f32 = means.astype(np.float32)
    vars_f32 = vars_.astype(np.float32)
    log_priors_f32 = log_priors.astype(np.float32)

    # Inference on GPU via broadcasting
    # X_test: (n_test, n_features), means: (n_classes, n_features), vars_: (n_classes, n_features)
    X_gpu = cp.asarray(X_test)                   # (n_test, n_features)
    mu = cp.asarray(means_f32)                   # (n_classes, n_features)
    var = cp.asarray(vars_f32)                   # (n_classes, n_features)
    lp = cp.asarray(log_priors_f32)              # (n_classes,)

    # Expand for broadcasting: (n_test, 1, n_features) vs (1, n_classes, n_features)
    X_exp = X_gpu[:, cp.newaxis, :]             # (n_test, 1, n_features)
    diff_sq = (X_exp - mu[cp.newaxis, :, :]) ** 2  # (n_test, n_classes, n_features)

    log_gaussian = -0.5 * cp.log(2.0 * cp.pi * var) - 0.5 * diff_sq / var
    log_probs = lp[cp.newaxis, :] + cp.sum(log_gaussian, axis=2)  # (n_test, n_classes)

    predictions_gpu = cp.argmax(log_probs, axis=1).astype(cp.int32)

    cp.cuda.Stream.null.synchronize()
    predictions = cp.asnumpy(predictions_gpu)
    return predictions, means_f32, vars_f32, log_priors_f32


if __name__ == "__main__":
    try:
        from src.utils.timing import BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    rng = np.random.default_rng(42)
    n_classes, n_features = 3, 8
    centers = rng.standard_normal((n_classes, n_features)) * 5.0
    X_parts, y_parts = [], []
    for c in range(n_classes):
        X_parts.append(rng.standard_normal((200, n_features)) * 0.5 + centers[c])
        y_parts.append(np.full(200, c, dtype=np.int32))
    X_tr = np.concatenate(X_parts).astype(np.float32)
    y_tr = np.concatenate(y_parts)
    X_te = rng.standard_normal((60, n_features)).astype(np.float32)
    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    gpu_time = runner.time_cpu(gaussian_nb_cupy, X_tr, y_tr, X_te, n_classes)
    preds, *_ = gaussian_nb_cupy(X_tr, y_tr, X_te, n_classes)
    print(f"gaussian_nb_cupy: {gpu_time:.3f}s | preds shape={preds.shape}")
