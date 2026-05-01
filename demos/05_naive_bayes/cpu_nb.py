"""NumPy CPU baseline for Gaussian Naive Bayes classification."""

from __future__ import annotations

import numpy as np


def gaussian_nb_cpu(
    X_train: np.ndarray,  # (n_train, n_features) float32
    y_train: np.ndarray,  # (n_train,) int32 class labels [0..k-1]
    X_test: np.ndarray,   # (n_test, n_features) float32
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian Naive Bayes: fit on train, predict on test.

    For each class c:
      prior[c] = count(y_train == c) / n_train
      mean[c]  = X_train[y_train == c].mean(axis=0)
      var[c]   = X_train[y_train == c].var(axis=0) + 1e-9  (epsilon for stability)

    For each test sample compute log-likelihood for each class:
      log_p = log(prior[c]) + sum(-0.5 * log(2*pi*var[c]) - 0.5 * (x - mean[c])^2 / var[c])

    Prediction is argmax over classes.

    Returns
    -------
    predictions:
        Integer array of shape (n_test,) with predicted class labels.
    accuracy:
        Fraction of test samples matching y_test (caller supplies y_test
        separately; here accuracy is computed against y_test if passed in via
        the accuracy helper below, or skipped).
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)

    n_train = X_train.shape[0]

    # Per-class statistics
    class_means = np.zeros((n_classes, X_train.shape[1]), dtype=np.float64)
    class_vars = np.zeros((n_classes, X_train.shape[1]), dtype=np.float64)
    log_priors = np.zeros(n_classes, dtype=np.float64)

    for c in range(n_classes):
        mask = y_train == c
        count = mask.sum()
        if count == 0:
            # Degenerate: no samples for this class — assign uniform prior, zero mean
            log_priors[c] = np.log(1.0 / n_classes)
            class_vars[c] = 1e-9
        else:
            X_c = X_train[mask].astype(np.float64)
            log_priors[c] = np.log(count / n_train)
            class_means[c] = X_c.mean(axis=0)
            class_vars[c] = X_c.var(axis=0) + 1e-9  # epsilon for numerical stability

    # Log-likelihood computation for all test samples
    # Shape: (n_test, n_classes)
    X_test_d = X_test.astype(np.float64)
    log_probs = np.empty((X_test.shape[0], n_classes), dtype=np.float64)

    for c in range(n_classes):
        # Broadcast: (n_test, n_features)
        diff = X_test_d - class_means[c]
        log_likelihood = (
            -0.5 * np.sum(np.log(2.0 * np.pi * class_vars[c]), axis=-1)
            - 0.5 * np.sum(diff ** 2 / class_vars[c], axis=-1)
        )
        log_probs[:, c] = log_priors[c] + log_likelihood

    predictions = np.argmax(log_probs, axis=1).astype(np.int32)
    return predictions, class_means, class_vars, log_priors
