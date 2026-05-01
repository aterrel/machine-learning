"""Naive (non-tiled) GEMM kernel demo for educational clarity.

Tiled GEMM with shared memory is deferred to REQ-0003-F7 (Sprint 3).
"""

from __future__ import annotations

import sys

import numpy as np

# Naive GEMM: C = A @ B
# A is (M, K), B is (K, N), C is (M, N)
# Each thread computes one output element C[row, col]
NAIVE_GEMM_SRC = r"""
extern "C" __global__
void naive_gemm(const float* A, const float* B, float* C,
                int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
"""

BLOCK_DIM = 16  # 16x16 thread block


def run_gemm_demo(M: int = 512, N: int = 512, K: int = 512) -> "BenchmarkResult":
    """Run naive GEMM kernel demo.

    Parameters
    ----------
    M, N, K:
        Matrix dimensions: A is (M, K), B is (K, N), C is (M, N).

    Returns
    -------
    BenchmarkResult comparing NumPy matmul (CPU) vs naive CUDA GEMM (GPU).
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
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)

    # CPU reference
    C_ref = A @ B

    stream = device.create_stream()
    compiler = KernelCompiler(device)
    kernel = compiler.compile(
        NAIVE_GEMM_SRC, "naive_gemm", cache_key="naive_gemm", stream=stream
    )

    a_bytes = A.nbytes
    b_bytes = B.nbytes
    c_bytes = M * N * np.dtype(np.float32).itemsize

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

    # 2D grid: ceil(N/BLOCK_DIM) x ceil(M/BLOCK_DIM) blocks of (BLOCK_DIM x BLOCK_DIM) threads
    grid_x = (N + BLOCK_DIM - 1) // BLOCK_DIM
    grid_y = (M + BLOCK_DIM - 1) // BLOCK_DIM
    grid = (grid_x, grid_y, 1)
    block = (BLOCK_DIM, BLOCK_DIM, 1)

    C_gpu = np.empty((M, N), dtype=np.float32)

    runner = BenchmarkRunner(n_repeats=5, warmup=1)
    cpu_time = runner.time_cpu(lambda: A @ B)

    with (
        DeviceBuffer(a_bytes, stream=stream, device=device) as d_A,
        DeviceBuffer(b_bytes, stream=stream, device=device) as d_B,
        DeviceBuffer(c_bytes, stream=stream, device=device) as d_C,
    ):
        _h2d(d_A.handle, A, a_bytes)
        _h2d(d_B.handle, B, b_bytes)

        def gpu_fn() -> None:
            kernel.launch(
                grid,
                block,
                [d_A.handle, d_B.handle, d_C.handle, np.int32(M), np.int32(N), np.int32(K)],
                stream=stream,
            )

        gpu_time = runner.time_gpu(gpu_fn, stream)

        stream.sync()
        _d2h(C_gpu, d_C.handle, c_bytes)

    stream.sync()
    stream.close()

    max_err = float(np.max(np.abs(C_gpu - C_ref)))
    correct = max_err < 1e-3
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    from src.utils.timing import BenchmarkResult

    return BenchmarkResult(
        demo_name="naive_gemm",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=correct,
        max_abs_error=max_err,
    )
