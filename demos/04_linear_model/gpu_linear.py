"""GPU linear regression: X^T X and X^T y on GPU, solve on CPU."""

from __future__ import annotations

import sys

import numpy as np

# Kernel: compute X^T X (n_features x n_features) and X^T y (n_features,)
# Launch as 2D grid: blockIdx.x = row i (0..n_features-1), blockIdx.y = col j
# Each thread computes XtX[i,j] and (when j==0) Xty[i]
XTX_XTY_SRC = r"""
extern "C" __global__
void xtx_xty(const float* X, const float* y, float* XtX, float* Xty,
             int n_samples, int n_features) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= n_features || j >= n_features) return;
    float s = 0.0f;
    for (int k = 0; k < n_samples; k++) {
        s += X[k * n_features + i] * X[k * n_features + j];
    }
    XtX[i * n_features + j] = s;
    if (j == 0) {  // compute Xty only once per row (when j==0)
        float t = 0.0f;
        for (int k = 0; k < n_samples; k++) t += X[k * n_features + i] * y[k];
        Xty[i] = t;
    }
}
"""


def linear_regression_gpu(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """GPU linear regression: X^T X and X^T y on GPU, solve on CPU.

    Parameters
    ----------
    X:
        Design matrix of shape (n_samples, n_features). Converted to float32 internally.
    y:
        Target vector of shape (n_samples,). Converted to float32 internally.

    Returns
    -------
    weights:
        Coefficient vector, shape (n_features,).
    r2_score:
        Coefficient of determination R^2.
    mse:
        Mean squared error on training data.
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
    y = np.asarray(y, dtype=np.float32)
    n_samples, n_features = X.shape

    stream = device.create_stream()
    compiler = KernelCompiler(device)

    kernel = compiler.compile(
        XTX_XTY_SRC, "xtx_xty", cache_key="linear_xtx_xty", stream=stream
    )

    x_bytes = X.nbytes
    y_bytes = y.nbytes
    xtx_bytes = n_features * n_features * np.dtype(np.float32).itemsize
    xty_bytes = n_features * np.dtype(np.float32).itemsize

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

    XtX_host = np.empty((n_features, n_features), dtype=np.float32)
    Xty_host = np.empty(n_features, dtype=np.float32)

    with (
        DeviceBuffer(x_bytes, stream=stream, device=device) as d_X,
        DeviceBuffer(y_bytes, stream=stream, device=device) as d_y,
        DeviceBuffer(xtx_bytes, stream=stream, device=device) as d_XtX,
        DeviceBuffer(xty_bytes, stream=stream, device=device) as d_Xty,
    ):
        _h2d(d_X.handle, X, x_bytes)
        _h2d(d_y.handle, y, y_bytes)

        # Zero output buffers
        (err,) = cudart.cudaMemset(d_XtX.handle, 0, xtx_bytes)
        if err.value != 0:
            raise RuntimeError(f"cudaMemset XtX failed: {err.value}")
        (err,) = cudart.cudaMemset(d_Xty.handle, 0, xty_bytes)
        if err.value != 0:
            raise RuntimeError(f"cudaMemset Xty failed: {err.value}")

        # 2D grid: (n_features, n_features), each thread handles one (i,j) pair
        grid_2d = (n_features, n_features, 1)
        block_2d = (1, 1, 1)
        kernel.launch(
            grid_2d,
            block_2d,
            [
                d_X.handle,
                d_y.handle,
                d_XtX.handle,
                d_Xty.handle,
                np.int32(n_samples),
                np.int32(n_features),
            ],
            stream=stream,
        )
        stream.sync()

        _d2h(XtX_host, d_XtX.handle, xtx_bytes)
        _d2h(Xty_host, d_Xty.handle, xty_bytes)

    stream.sync()
    stream.close()

    # Solve normal equation on CPU: XtX @ w = Xty
    weights = np.linalg.solve(XtX_host.astype(np.float64), Xty_host.astype(np.float64))
    weights = weights.astype(np.float32)

    # Compute metrics on CPU
    y_pred = X @ weights.astype(np.float64)
    y_f64 = y.astype(np.float64)
    ss_res = float(np.sum((y_f64 - y_pred) ** 2))
    ss_tot = float(np.sum((y_f64 - y_f64.mean()) ** 2))
    r2_score = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    mse = ss_res / n_samples

    return weights, float(r2_score), float(mse)
