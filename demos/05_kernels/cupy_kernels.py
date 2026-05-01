"""CuPy kernel demos: GEMM via cp.matmul (cuBLAS) and ReLU via cp.RawKernel."""

from __future__ import annotations

import sys

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True

    # ReLU as RawKernel — shows CUDA C kernel compilation path in CuPy
    _RELU_SRC = r"""
extern "C" __global__
void relu_inplace(float* x, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}
"""
    _relu_kernel = cp.RawKernel(_RELU_SRC, "relu_inplace")

except ImportError:
    _CUPY_AVAILABLE = False

BLOCK_SIZE = 256


def gemm_cupy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """GEMM via cp.matmul — dispatches to cuBLAS under the hood."""
    if not _CUPY_AVAILABLE:
        raise ImportError("cupy is required. Install with: pip install cupy-cuda12x")
    A_gpu = cp.asarray(A, dtype=cp.float32)
    B_gpu = cp.asarray(B, dtype=cp.float32)
    C_gpu = cp.matmul(A_gpu, B_gpu)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(C_gpu)


def relu_cupy(x: np.ndarray) -> np.ndarray:
    """ReLU via cp.RawKernel — CUDA C kernel compiled via CuPy's NVRTC wrapper."""
    if not _CUPY_AVAILABLE:
        raise ImportError("cupy is required. Install with: pip install cupy-cuda12x")
    x_gpu = cp.asarray(x, dtype=cp.float32)
    n = x_gpu.size
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    _relu_kernel((grid,), (BLOCK_SIZE,), (x_gpu, np.int32(n)))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(x_gpu)


def run_gemm_cupy(M: int = 512, N: int = 512, K: int = 512, seed: int = 42):
    """Benchmark GEMM with CuPy backend (cuBLAS). Returns BenchmarkResult."""
    try:
        from src.utils.timing import BenchmarkResult, BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not _CUPY_AVAILABLE:
        print("CuPy not available — install with: pip install cupy-cuda12x")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ref = A @ B

    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    cpu_time = runner.time_cpu(np.matmul, A, B)
    gpu_time = runner.time_cpu(gemm_cupy, A, B)
    result = gemm_cupy(A, B)

    max_err = float(np.max(np.abs(result - ref)))
    correct = max_err < 1e-2
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    return BenchmarkResult(
        demo_name="gemm_cupy",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=correct,
        max_abs_error=max_err,
    )


if __name__ == "__main__":
    r = run_gemm_cupy()
    print(r.summary_line())
