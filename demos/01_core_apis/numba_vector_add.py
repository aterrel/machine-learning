"""Numba-CUDA vector addition — same operation as vector_add.py using @cuda.jit."""

from __future__ import annotations

import sys

import numpy as np

try:
    from numba import cuda as nb_cuda

    _NUMBA_AVAILABLE = True

    @nb_cuda.jit
    def _vector_add(a, b, c):
        i = nb_cuda.grid(1)
        if i < c.shape[0]:
            c[i] = a[i] + b[i]

except ImportError:
    _NUMBA_AVAILABLE = False

BLOCK_SIZE = 256


def vector_add_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Numba-CUDA vector addition. Accepts and returns numpy arrays.

    The @cuda.jit kernel compiles to PTX via LLVM on first call (JIT).
    Subsequent calls run the cached compiled kernel.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba is required. Install with: pip install numba")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = a.shape[0]
    d_a = nb_cuda.to_device(a)
    d_b = nb_cuda.to_device(b)
    d_c = nb_cuda.device_array(n, dtype=np.float32)
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    _vector_add[grid, BLOCK_SIZE](d_a, d_b, d_c)
    nb_cuda.synchronize()
    return d_c.copy_to_host()


def run_vector_add_numba(n: int = 1_000_000, seed: int = 42):
    """Benchmark vector-add with the Numba-CUDA backend. Returns BenchmarkResult."""
    try:
        from src.utils.timing import BenchmarkResult, BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not _NUMBA_AVAILABLE:
        print("Numba not available — install with: pip install numba")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    a = rng.random(n).astype(np.float32)
    b = rng.random(n).astype(np.float32)
    ref = a + b

    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    cpu_time = runner.time_cpu(np.add, a, b)

    # warmup=1 inside runner handles JIT compile on first call
    gpu_time = runner.time_cpu(vector_add_numba, a, b)
    result = vector_add_numba(a, b)

    max_err = float(np.max(np.abs(result - ref)))
    correct = max_err < 1e-5
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    return BenchmarkResult(
        demo_name="vector_add_numba",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=correct,
        max_abs_error=max_err,
    )


if __name__ == "__main__":
    r = run_vector_add_numba()
    print(r.summary_line())
