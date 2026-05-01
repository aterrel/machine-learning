"""GPU PCA: mean-centering and covariance computation on GPU, eigh on CPU."""

from __future__ import annotations

import sys

import numpy as np

# Kernel 1: subtract per-feature mean from each element (in-place)
MEAN_CENTER_SRC = r"""
extern "C" __global__
void mean_center(float* X, const float* means, int n_samples, int n_features) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;  // flat index into X
    if (i < n_samples * n_features) {
        int f = i % n_features;
        X[i] -= means[f];
    }
}
"""

# Kernel 2: compute symmetric covariance matrix (upper + lower triangle)
# Launch as 2D grid: (n_features, n_features); each thread computes C[i,j]
COVARIANCE_SRC = r"""
extern "C" __global__
void compute_cov(const float* X_c, float* C, int n_samples, int n_features) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= n_features || j >= n_features || j < i) return;  // upper triangle only
    float s = 0.0f;
    for (int k = 0; k < n_samples; k++) {
        s += X_c[k * n_features + i] * X_c[k * n_features + j];
    }
    s /= (n_samples - 1);
    C[i * n_features + j] = s;
    C[j * n_features + i] = s;  // mirror to lower triangle
}
"""

BLOCK_SIZE = 256


def pca_gpu(
    X: np.ndarray,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU PCA: mean-center and covariance on GPU, eigh on CPU.

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
    try:
        from cuda.bindings import cudart

        from src.kernels.compiler import KernelCompiler
        from src.utils.device import get_device
        from src.utils.memory import DeviceBuffer
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        sys.exit(1)

    try:
        device = get_device(0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape

    # Compute per-feature means on CPU to pass to GPU
    means = X.mean(axis=0).astype(np.float32)

    stream = device.create_stream()
    compiler = KernelCompiler(device)

    center_kernel = compiler.compile(
        MEAN_CENTER_SRC, "mean_center", cache_key="pca_mean_center", stream=stream
    )
    cov_kernel = compiler.compile(
        COVARIANCE_SRC, "compute_cov", cache_key="pca_compute_cov", stream=stream
    )

    n_total = n_samples * n_features
    x_bytes = X.nbytes
    means_bytes = means.nbytes
    cov_bytes = n_features * n_features * np.dtype(np.float32).itemsize

    def _h2d(dst_handle: int, src_arr: np.ndarray, nbytes: int) -> None:
        (err,) = cudart.cudaMemcpy(
            dst_handle,
            src_arr.ctypes.data,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if err.value != 0:
            raise RuntimeError(f"H2D copy failed: {err.value}")

    def _d2h(dst_arr: np.ndarray, src_handle: int, nbytes: int) -> None:
        (err,) = cudart.cudaMemcpy(
            dst_arr.ctypes.data,
            src_handle,
            nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if err.value != 0:
            raise RuntimeError(f"D2H copy failed: {err.value}")

    X_c_host = np.empty_like(X)
    C_host = np.empty((n_features, n_features), dtype=np.float32)

    with (
        DeviceBuffer(x_bytes, stream=stream, device=device) as d_X,
        DeviceBuffer(means_bytes, stream=stream, device=device) as d_means,
        DeviceBuffer(cov_bytes, stream=stream, device=device) as d_cov,
    ):
        # Upload X and means
        _h2d(d_X.handle, X, x_bytes)
        _h2d(d_means.handle, means, means_bytes)

        # Zero covariance buffer
        (err,) = cudart.cudaMemset(d_cov.handle, 0, cov_bytes)
        if err.value != 0:
            raise RuntimeError(f"cudaMemset cov failed: {err.value}")

        # Launch mean_center kernel (1D, covers all n_samples * n_features elements)
        grid_1d = center_kernel.compute_grid_1d(n_total, BLOCK_SIZE)
        block_1d = (BLOCK_SIZE, 1, 1)
        center_kernel.launch(
            grid_1d,
            block_1d,
            [d_X.handle, d_means.handle, np.int32(n_samples), np.int32(n_features)],
            stream=stream,
        )

        # Launch compute_cov kernel (2D grid: n_features x n_features)
        grid_2d = (n_features, n_features, 1)
        block_2d = (1, 1, 1)
        cov_kernel.launch(
            grid_2d,
            block_2d,
            [d_X.handle, d_cov.handle, np.int32(n_samples), np.int32(n_features)],
            stream=stream,
        )
        stream.sync()

        # Copy centered X and covariance matrix back to host
        _d2h(X_c_host, d_X.handle, x_bytes)
        _d2h(C_host, d_cov.handle, cov_bytes)

    stream.sync()
    stream.close()

    # Eigendecomposition on CPU (cuSOLVER out of scope)
    eigenvalues, eigenvectors = np.linalg.eigh(C_host.astype(np.float64))

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].astype(np.float32)
    eigenvectors = eigenvectors[:, idx].astype(np.float32)

    components = eigenvectors[:, :n_components].T  # (n_components, n_features)
    explained_variance = eigenvalues[:n_components]  # (n_components,)
    X_transformed = X_c_host @ eigenvectors[:, :n_components]  # (n_samples, n_components)

    return components, explained_variance, X_transformed
