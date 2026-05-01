"""CuPy vector addition — using cp.RawKernel (CUDA C via CuPy) and cp.add."""

from __future__ import annotations

import sys

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True

    # RawKernel: same CUDA C source as cuda-python demo, compiled via CuPy's NVRTC wrapper
    _VECTOR_ADD_SRC = r"""
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""
    _raw_kernel = cp.RawKernel(_VECTOR_ADD_SRC, "vector_add")

except ImportError:
    _CUPY_AVAILABLE = False

BLOCK_SIZE = 256


def vector_add_cupy_raw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector addition using cp.RawKernel — shows CUDA C kernel style via CuPy."""
    if not _CUPY_AVAILABLE:
        raise ImportError("cupy is required. Install with: pip install cupy-cuda12x")
    a_gpu = cp.asarray(a, dtype=cp.float32)
    b_gpu = cp.asarray(b, dtype=cp.float32)
    c_gpu = cp.empty_like(a_gpu)
    n = a_gpu.shape[0]
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    _raw_kernel((grid,), (BLOCK_SIZE,), (a_gpu, b_gpu, c_gpu, np.int32(n)))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(c_gpu)


def vector_add_cupy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector addition using cp.add — the idiomatic NumPy-style CuPy approach."""
    if not _CUPY_AVAILABLE:
        raise ImportError("cupy is required. Install with: pip install cupy-cuda12x")
    a_gpu = cp.asarray(a, dtype=cp.float32)
    b_gpu = cp.asarray(b, dtype=cp.float32)
    c_gpu = cp.add(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(c_gpu)


def run_vector_add_cupy(n: int = 1_000_000, seed: int = 42):
    """Benchmark vector-add with the CuPy backend. Returns BenchmarkResult."""
    try:
        from src.utils.timing import BenchmarkResult, BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not _CUPY_AVAILABLE:
        print("CuPy not available — install with: pip install cupy-cuda12x")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    a = rng.random(n).astype(np.float32)
    b = rng.random(n).astype(np.float32)
    ref = a + b

    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    cpu_time = runner.time_cpu(np.add, a, b)

    # Use cp.add (idiomatic); warmup=1 handles RawKernel JIT compile
    gpu_time = runner.time_cpu(vector_add_cupy, a, b)
    result = vector_add_cupy(a, b)

    max_err = float(np.max(np.abs(result - ref)))
    correct = max_err < 1e-5
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    return BenchmarkResult(
        demo_name="vector_add_cupy",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=correct,
        max_abs_error=max_err,
    )


if __name__ == "__main__":
    r = run_vector_add_cupy()
    print(r.summary_line())
