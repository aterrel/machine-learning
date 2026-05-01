"""cuML backend wrappers for the unified comparison demo.

All functions accept numpy arrays and return numpy arrays.
All are guarded by ``try: import cuml`` so this module is always importable.
If cuML is not installed, calling any function raises ImportError.

Install cuML via RAPIDS:
  conda install -c rapidsai cuml  (recommended)
  See https://rapids.ai/install for the version matrix.
"""

from __future__ import annotations

import numpy as np

try:
    import cuml

    _CUML_AVAILABLE = True
except ImportError:
    _CUML_AVAILABLE = False


def _to_numpy(arr) -> np.ndarray:
    """Convert cuDF Series / CuPy array / numpy array to numpy."""
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy()
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def kmeans_cuml(
    X: np.ndarray,
    k: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """cuML k-means. Returns (centroids, labels, inertia) matching kmeans_cpu interface."""
    if not _CUML_AVAILABLE:
        raise ImportError("cuML not installed. See https://rapids.ai/install")
    import cuml.cluster

    model = cuml.cluster.KMeans(n_clusters=k, random_state=seed, output_type="numpy")
    model.fit(X.astype(np.float32))
    centroids = _to_numpy(model.cluster_centers_).astype(np.float32)
    labels = _to_numpy(model.labels_).astype(np.int32)
    inertia = float(_to_numpy(model.inertia_)) if hasattr(model, "inertia_") else 0.0
    return centroids, labels, inertia


def pca_cuml(
    X: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cuML PCA. Returns (components, explained_variance, X_transformed)."""
    if not _CUML_AVAILABLE:
        raise ImportError("cuML not installed. See https://rapids.ai/install")
    import cuml.decomposition

    model = cuml.decomposition.PCA(n_components=n_components, output_type="numpy")
    X_transformed = model.fit_transform(X.astype(np.float32))
    components = _to_numpy(model.components_).astype(np.float32)
    explained_variance = _to_numpy(model.explained_variance_).astype(np.float32)
    return components, explained_variance, _to_numpy(X_transformed).astype(np.float32)


def linear_regression_cuml(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """cuML linear regression. Returns (weights, r2_score, mse)."""
    if not _CUML_AVAILABLE:
        raise ImportError("cuML not installed. See https://rapids.ai/install")
    import cuml.linear_model

    model = cuml.linear_model.LinearRegression(output_type="numpy")
    model.fit(X.astype(np.float32), y.astype(np.float32))
    weights = _to_numpy(model.coef_).astype(np.float32)

    y_pred = X.astype(np.float64) @ weights.astype(np.float64)
    y_f64 = y.astype(np.float64)
    ss_res = float(np.sum((y_f64 - y_pred) ** 2))
    ss_tot = float(np.sum((y_f64 - y_f64.mean()) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    mse = ss_res / len(y)
    return weights, float(r2_score), float(mse)


def gaussian_nb_cuml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """cuML Gaussian Naive Bayes. Returns (predictions, means, vars_, log_priors)."""
    if not _CUML_AVAILABLE:
        raise ImportError("cuML not installed. See https://rapids.ai/install")
    import cuml.naive_bayes

    model = cuml.naive_bayes.GaussianNB()
    model.fit(X_train.astype(np.float32), y_train.astype(np.int32))
    predictions = _to_numpy(model.predict(X_test.astype(np.float32))).astype(np.int32)

    means = _to_numpy(model.theta_).astype(np.float32)     # (n_classes, n_features)
    vars_ = _to_numpy(model.sigma_).astype(np.float32)     # (n_classes, n_features)
    log_priors = _to_numpy(model.class_log_prior_).astype(np.float32)
    return predictions, means, vars_, log_priors
