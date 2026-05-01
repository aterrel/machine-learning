"""ReLU and softmax CUDA kernel demos."""

from __future__ import annotations

import sys

import numpy as np

# ReLU in-place: x[i] = max(0, x[i])
RELU_SRC = r"""
extern "C" __global__
void relu_inplace(float* x, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}
"""

# Softmax per row (single-threaded per row for educational simplicity)
SOFTMAX_SRC = r"""
extern "C" __global__
void softmax_row(float* x, int n_rows, int n_cols) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    float* row_ptr = x + row * n_cols;
    // All work done in thread 0 for clarity (non-parallel, educational)
    if (threadIdx.x == 0) {
        float mx = row_ptr[0];
        for (int j = 1; j < n_cols; j++) if (row_ptr[j] > mx) mx = row_ptr[j];
        float s = 0.0f;
        for (int j = 0; j < n_cols; j++) { row_ptr[j] = expf(row_ptr[j] - mx); s += row_ptr[j]; }
        for (int j = 0; j < n_cols; j++) row_ptr[j] /= s;
    }
}
"""

BLOCK_SIZE = 256


def run_activations_demo(n: int = 1_000_000) -> "BenchmarkResult":
    """Run ReLU and softmax kernel demos.

    Parameters
    ----------
    n:
        Number of elements for the ReLU demo (main timing target).
        Softmax uses a fixed (1000 x 1000) matrix.

    Returns
    -------
    BenchmarkResult for the ReLU kernel (main timing target).
    """
    try:
        from cuda.bindings import cudart

        from src.kernels.compiler import KernelCompiler
        from src.utils.device import get_device
        from src.utils.memory import DeviceBuffer
        from src.utils.timing import BenchmarkResult, BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        sys.exit(1)

    try:
        device = get_device(0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    rng = np.random.default_rng(42)

    stream = device.create_stream()
    compiler = KernelCompiler(device)

    relu_kernel = compiler.compile(
        RELU_SRC, "relu_inplace", cache_key="relu_inplace", stream=stream
    )
    softmax_kernel = compiler.compile(
        SOFTMAX_SRC, "softmax_row", cache_key="softmax_row", stream=stream
    )

    # ------------------------------------------------------------------ ReLU
    x_cpu = rng.standard_normal(n).astype(np.float32)
    x_ref = np.maximum(0.0, x_cpu)
    n_bytes = x_cpu.nbytes

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

    runner = BenchmarkRunner(n_repeats=5, warmup=1)
    cpu_time = runner.time_cpu(lambda: np.maximum(0.0, x_cpu))

    x_gpu = np.empty(n, dtype=np.float32)
    relu_correct = False
    relu_max_err = float("inf")

    grid_relu = relu_kernel.compute_grid_1d(n, BLOCK_SIZE)
    block_1d = (BLOCK_SIZE, 1, 1)

    with DeviceBuffer(n_bytes, stream=stream, device=device) as d_x:
        # Upload once before timing; ReLU is idempotent so repeated launches on
        # the modified buffer are valid for throughput measurement.
        _h2d(d_x.handle, x_cpu, n_bytes)

        def relu_fn() -> None:
            relu_kernel.launch(
                grid_relu, block_1d, [d_x.handle, np.int32(n)], stream=stream
            )

        gpu_time = runner.time_gpu(relu_fn, stream)

        # Final run to capture result
        _h2d(d_x.handle, x_cpu, n_bytes)
        relu_kernel.launch(grid_relu, block_1d, [d_x.handle, np.int32(n)], stream=stream)
        stream.sync()
        _d2h(x_gpu, d_x.handle, n_bytes)

    relu_max_err = float(np.max(np.abs(x_gpu - x_ref)))
    relu_correct = relu_max_err < 1e-6

    # --------------------------------------------------------------- Softmax
    n_rows, n_cols = 1000, 1000
    sm_cpu = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    sm_bytes = sm_cpu.nbytes
    sm_gpu = np.empty_like(sm_cpu)

    with DeviceBuffer(sm_bytes, stream=stream, device=device) as d_sm:
        _h2d(d_sm.handle, sm_cpu, sm_bytes)
        # One block per row, thread 0 does all work
        softmax_kernel.launch(
            (n_rows, 1, 1),
            (1, 1, 1),
            [d_sm.handle, np.int32(n_rows), np.int32(n_cols)],
            stream=stream,
        )
        stream.sync()
        _d2h(sm_gpu, d_sm.handle, sm_bytes)

    row_sums = sm_gpu.sum(axis=1)
    softmax_correct = bool(np.all(np.abs(row_sums - 1.0) < 1e-5))
    print(f"  Softmax row-sum max deviation: {float(np.max(np.abs(row_sums - 1.0))):.2e}")
    print(f"  Softmax correct: {softmax_correct}")

    stream.sync()
    stream.close()

    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    from src.utils.timing import BenchmarkResult

    return BenchmarkResult(
        demo_name="relu",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=relu_correct,
        max_abs_error=relu_max_err,
    )
