"""GPU Gaussian Naive Bayes using custom CUDA kernels for inference."""

from __future__ import annotations

import sys

import numpy as np

# ---------------------------------------------------------------------------
# Kernel 1: compute log-likelihood for each (test_sample, class) pair
# ---------------------------------------------------------------------------

COMPUTE_LOG_LIKELIHOOD_SRC = r"""
extern "C" __global__
void compute_log_likelihood(
    const float* X_test,       // (n_test, n_features)
    const float* class_means,  // (n_classes, n_features)
    const float* class_vars,   // (n_classes, n_features), pre-added epsilon
    const float* log_priors,   // (n_classes,)
    float* log_probs,          // (n_test, n_classes) — output
    int n_test, int n_features, int n_classes
) {
    // Each thread handles one (test_sample, class) pair
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    int cls = blockIdx.y;
    if (sample >= n_test || cls >= n_classes) return;

    float ll = log_priors[cls];
    for (int f = 0; f < n_features; f++) {
        float x   = X_test[sample * n_features + f];
        float mu  = class_means[cls * n_features + f];
        float var = class_vars[cls * n_features + f];
        // log N(x; mu, var) = -0.5 * log(2*pi*var) - 0.5 * (x-mu)^2/var
        ll += -0.5f * logf(2.0f * 3.14159265f * var) - 0.5f * (x - mu) * (x - mu) / var;
    }
    log_probs[sample * n_classes + cls] = ll;
}
"""

# ---------------------------------------------------------------------------
# Kernel 2: argmax over classes for each test sample
# ---------------------------------------------------------------------------

ARGMAX_PREDICTIONS_SRC = r"""
extern "C" __global__
void argmax_predictions(
    const float* log_probs,  // (n_test, n_classes)
    int* predictions,         // (n_test,)
    int n_test, int n_classes
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_test) return;
    float best = log_probs[i * n_classes];
    int best_cls = 0;
    for (int c = 1; c < n_classes; c++) {
        float v = log_probs[i * n_classes + c];
        if (v > best) { best = v; best_cls = c; }
    }
    predictions[i] = best_cls;
}
"""

BLOCK_SIZE = 256


def gaussian_nb_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """GPU Gaussian Naive Bayes: fit on CPU, predict on GPU.

    Fitting (computing class means, variances, and priors) is done on CPU since
    it operates over the training set once.  The compute-intensive inference
    step (log-likelihood evaluation + argmax over all test samples and classes)
    is offloaded to two custom CUDA kernels.

    Parameters
    ----------
    X_train:
        Training features, shape (n_train, n_features), float32.
    y_train:
        Training labels, shape (n_train,), int32, values in [0, n_classes).
    X_test:
        Test features, shape (n_test, n_features), float32.
    n_classes:
        Number of distinct classes.

    Returns
    -------
    predictions:
        Integer array of shape (n_test,) with predicted class labels.
    class_means:
        Per-class feature means, shape (n_classes, n_features).
    class_vars:
        Per-class feature variances (+ epsilon), shape (n_classes, n_features).
    log_priors:
        Log prior probabilities, shape (n_classes,).
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

    # ------------------------------------------------------------------
    # 1. Fit on CPU — compute class means, variances, and log-priors
    # ------------------------------------------------------------------
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    X_test = np.asarray(X_test, dtype=np.float32)

    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]

    class_means = np.zeros((n_classes, n_features), dtype=np.float32)
    class_vars = np.zeros((n_classes, n_features), dtype=np.float32)
    log_priors = np.zeros(n_classes, dtype=np.float32)

    for c in range(n_classes):
        mask = y_train == c
        count = int(mask.sum())
        if count == 0:
            log_priors[c] = np.float32(np.log(1.0 / n_classes))
            class_vars[c] = np.float32(1e-9)
        else:
            X_c = X_train[mask].astype(np.float64)
            log_priors[c] = np.float32(np.log(count / n_train))
            class_means[c] = X_c.mean(axis=0).astype(np.float32)
            class_vars[c] = (X_c.var(axis=0) + 1e-9).astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Compile kernels
    # ------------------------------------------------------------------
    stream = device.create_stream()
    compiler = KernelCompiler(device)

    try:
        ll_kernel = compiler.compile(
            COMPUTE_LOG_LIKELIHOOD_SRC,
            "compute_log_likelihood",
            cache_key="nb_compute_log_likelihood",
            stream=stream,
        )
        argmax_kernel = compiler.compile(
            ARGMAX_PREDICTIONS_SRC,
            "argmax_predictions",
            cache_key="nb_argmax_predictions",
            stream=stream,
        )
    except Exception as exc:
        print(f"ERROR: Kernel compilation failed: {exc}")
        stream.close()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Allocate GPU buffers
    # ------------------------------------------------------------------
    x_test_bytes = X_test.nbytes                                  # (n_test, n_features) float32
    means_bytes = class_means.nbytes                               # (n_classes, n_features) float32
    vars_bytes = class_vars.nbytes                                 # (n_classes, n_features) float32
    priors_bytes = log_priors.nbytes                               # (n_classes,) float32
    log_probs_bytes = n_test * n_classes * np.dtype(np.float32).itemsize  # (n_test, n_classes)
    preds_bytes = n_test * np.dtype(np.int32).itemsize             # (n_test,)

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

    with (
        DeviceBuffer(x_test_bytes, stream=stream, device=device) as d_X_test,
        DeviceBuffer(means_bytes, stream=stream, device=device) as d_means,
        DeviceBuffer(vars_bytes, stream=stream, device=device) as d_vars,
        DeviceBuffer(priors_bytes, stream=stream, device=device) as d_priors,
        DeviceBuffer(log_probs_bytes, stream=stream, device=device) as d_log_probs,
        DeviceBuffer(preds_bytes, stream=stream, device=device) as d_preds,
    ):
        # ------------------------------------------------------------------
        # 4. H2D copy
        # ------------------------------------------------------------------
        _h2d(d_X_test.handle, X_test, x_test_bytes)
        _h2d(d_means.handle, class_means, means_bytes)
        _h2d(d_vars.handle, class_vars, vars_bytes)
        _h2d(d_priors.handle, log_priors, priors_bytes)

        # ------------------------------------------------------------------
        # 5. Launch compute_log_likelihood kernel
        #    Grid: x-dim covers test samples, y-dim covers classes
        # ------------------------------------------------------------------
        block = (BLOCK_SIZE, 1, 1)
        grid_x = (n_test + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid_ll = (grid_x, n_classes, 1)

        ll_kernel.launch(
            grid_ll,
            block,
            [
                d_X_test.handle,
                d_means.handle,
                d_vars.handle,
                d_priors.handle,
                d_log_probs.handle,
                np.int32(n_test),
                np.int32(n_features),
                np.int32(n_classes),
            ],
            stream=stream,
        )

        # ------------------------------------------------------------------
        # 6. Launch argmax_predictions kernel
        # ------------------------------------------------------------------
        grid_argmax = (grid_x, 1, 1)
        argmax_kernel.launch(
            grid_argmax,
            block,
            [
                d_log_probs.handle,
                d_preds.handle,
                np.int32(n_test),
                np.int32(n_classes),
            ],
            stream=stream,
        )
        stream.sync()

        # ------------------------------------------------------------------
        # 7. D2H copy predictions
        # ------------------------------------------------------------------
        predictions_host = np.empty(n_test, dtype=np.int32)
        _d2h(predictions_host, d_preds.handle, preds_bytes)

    stream.sync()
    stream.close()

    return predictions_host, class_means, class_vars, log_priors
