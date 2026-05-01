"""Numba-CUDA Gaussian Naive Bayes: fit on CPU, infer on GPU via @cuda.jit."""

from __future__ import annotations

import sys

import numpy as np

try:
    import math as _math

    from numba import cuda as nb_cuda

    _NUMBA_AVAILABLE = True

    @nb_cuda.jit
    def _compute_log_likelihood(X_test, class_means, class_vars, log_priors, log_probs):
        """Log-likelihood for each (sample, class) pair. 1D flat grid."""
        tid = nb_cuda.grid(1)
        n_test = X_test.shape[0]
        n_features = X_test.shape[1]
        n_classes = class_means.shape[0]
        total = n_test * n_classes
        if tid >= total:
            return
        sample = tid // n_classes
        cls = tid % n_classes
        ll = log_priors[cls]
        for f in range(n_features):
            x = X_test[sample, f]
            mu = class_means[cls, f]
            var = class_vars[cls, f]
            ll += -0.5 * _math.log(2.0 * _math.pi * var) - 0.5 * (x - mu) * (x - mu) / var
        log_probs[sample, cls] = ll

    @nb_cuda.jit
    def _argmax_predictions(log_probs, predictions):
        """Argmax over classes for each test sample."""
        i = nb_cuda.grid(1)
        if i >= log_probs.shape[0]:
            return
        n_classes = log_probs.shape[1]
        best = log_probs[i, 0]
        best_cls = 0
        for c in range(1, n_classes):
            v = log_probs[i, c]
            if v > best:
                best = v
                best_cls = c
        predictions[i] = best_cls

except ImportError:
    _NUMBA_AVAILABLE = False

BLOCK_SIZE = 256


def gaussian_nb_numba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-CUDA Gaussian Naive Bayes: fit on CPU, predict on GPU.

    Returns (predictions, means, vars_, log_priors) matching gaussian_nb_cpu interface.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba is required. Install with: pip install numba")

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]

    # Fit on CPU (same as cuda-python variant)
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

    d_X_test = nb_cuda.to_device(X_test)
    d_means = nb_cuda.to_device(means_f32)
    d_vars = nb_cuda.to_device(vars_f32)
    d_log_priors = nb_cuda.to_device(log_priors_f32)
    d_log_probs = nb_cuda.to_device(np.zeros((n_test, n_classes), dtype=np.float32))
    d_predictions = nb_cuda.to_device(np.zeros(n_test, dtype=np.int32))

    total_ll = n_test * n_classes
    grid_ll = (total_ll + BLOCK_SIZE - 1) // BLOCK_SIZE
    _compute_log_likelihood[grid_ll, BLOCK_SIZE](
        d_X_test, d_means, d_vars, d_log_priors, d_log_probs
    )
    nb_cuda.synchronize()

    grid_am = (n_test + BLOCK_SIZE - 1) // BLOCK_SIZE
    _argmax_predictions[grid_am, BLOCK_SIZE](d_log_probs, d_predictions)
    nb_cuda.synchronize()

    predictions = d_predictions.copy_to_host()
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
    gpu_time = runner.time_cpu(gaussian_nb_numba, X_tr, y_tr, X_te, n_classes)
    preds, *_ = gaussian_nb_numba(X_tr, y_tr, X_te, n_classes)
    print(f"gaussian_nb_numba: {gpu_time:.3f}s | preds shape={preds.shape}")
