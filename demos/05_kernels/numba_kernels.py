"""Numba-CUDA kernel demos: GEMM and ReLU using @cuda.jit."""

from __future__ import annotations

import sys

import numpy as np

try:
    from numba import cuda as nb_cuda

    _NUMBA_AVAILABLE = True

    @nb_cuda.jit
    def _naive_gemm(A, B, C, M, N, K):
        """Naive GEMM: C[i,j] = sum_k A[i,k]*B[k,j]. One thread per output element."""
        tid = nb_cuda.grid(1)
        total = M * N
        if tid >= total:
            return
        i = tid // N
        j = tid % N
        s = 0.0
        for k in range(K):
            s += float(A[i, k]) * float(B[k, j])
        C[i, j] = s

    @nb_cuda.jit
    def _relu_inplace(x):
        """ReLU activation in-place."""
        i = nb_cuda.grid(1)
        if i < x.shape[0]:
            if x[i] < 0.0:
                x[i] = 0.0

except ImportError:
    _NUMBA_AVAILABLE = False

BLOCK_SIZE = 256


def gemm_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Numba-CUDA matrix multiply. A: (M,K), B: (K,N) → C: (M,N)."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba is required. Install with: pip install numba")
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    M, K = A.shape
    N = B.shape[1]
    d_A = nb_cuda.to_device(A)
    d_B = nb_cuda.to_device(B)
    d_C = nb_cuda.to_device(np.zeros((M, N), dtype=np.float32))
    total = M * N
    grid = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    _naive_gemm[grid, BLOCK_SIZE](d_A, d_B, d_C, M, N, K)
    nb_cuda.synchronize()
    return d_C.copy_to_host()


def relu_numba(x: np.ndarray) -> np.ndarray:
    """Numba-CUDA ReLU: applies max(0, x) in-place on GPU."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba is required. Install with: pip install numba")
    x = np.asarray(x, dtype=np.float32)
    d_x = nb_cuda.to_device(x.copy())
    n = x.size
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    _relu_inplace[grid, BLOCK_SIZE](d_x)
    nb_cuda.synchronize()
    return d_x.copy_to_host()


def run_gemm_numba(M: int = 512, N: int = 512, K: int = 512, seed: int = 42):
    """Benchmark GEMM with Numba-CUDA backend. Returns BenchmarkResult."""
    try:
        from src.utils.timing import BenchmarkResult, BenchmarkRunner
    except ImportError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not _NUMBA_AVAILABLE:
        print("Numba not available — install with: pip install numba")
        sys.exit(1)

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    ref = A @ B

    runner = BenchmarkRunner(n_repeats=3, warmup=1)
    cpu_time = runner.time_cpu(np.matmul, A, B)
    gpu_time = runner.time_cpu(gemm_numba, A, B)
    result = gemm_numba(A, B)

    max_err = float(np.max(np.abs(result - ref)))
    correct = max_err < 1e-2
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    return BenchmarkResult(
        demo_name="gemm_numba",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=correct,
        max_abs_error=max_err,
    )


if __name__ == "__main__":
    r = run_gemm_numba()
    print(r.summary_line())
