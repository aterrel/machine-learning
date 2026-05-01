"""NumPy CPU baseline for linear regression via normal equation (lstsq)."""

from __future__ import annotations

import numpy as np


def linear_regression_cpu(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """NumPy normal equation linear regression.

    Parameters
    ----------
    X:
        Design matrix of shape (n_samples, n_features).
    y:
        Target vector of shape (n_samples,).

    Returns
    -------
    weights:
        Coefficient vector, shape (n_features,).
    r2_score:
        Coefficient of determination R^2.
    mse:
        Mean squared error on training data.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Solve via least squares (handles rank-deficient cases gracefully)
    weights, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ weights
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    mse = ss_res / len(y)

    return weights.astype(np.float32), float(r2_score), float(mse)
