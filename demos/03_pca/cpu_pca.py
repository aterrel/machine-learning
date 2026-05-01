"""NumPy CPU baseline for PCA via covariance matrix + eigendecomposition."""

from __future__ import annotations

import numpy as np


def pca_cpu(
    X: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy PCA via covariance matrix eigendecomposition.

    Parameters
    ----------
    X:
        Input data of shape (n_samples, n_features). Converted to float32 internally.
    n_components:
        Number of principal components to return.

    Returns
    -------
    components:
        Top-k eigenvectors as rows, shape (n_components, n_features).
    explained_variance:
        Corresponding eigenvalues, shape (n_components,).
    X_transformed:
        X projected onto components, shape (n_samples, n_components).
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape

    # Center X by subtracting per-feature mean
    means = X.mean(axis=0)
    X_c = X - means

    # Covariance matrix: C = X_c^T @ X_c / (n_samples - 1)
    C = (X_c.T @ X_c) / (n_samples - 1)

    # Eigendecomposition (eigh returns eigenvalues in ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  # columns are eigenvectors

    # Top-k components (rows)
    components = eigenvectors[:, :n_components].T  # (n_components, n_features)
    explained_variance = eigenvalues[:n_components]  # (n_components,)

    # Project X onto components
    X_transformed = X_c @ eigenvectors[:, :n_components]  # (n_samples, n_components)

    return components, explained_variance, X_transformed
