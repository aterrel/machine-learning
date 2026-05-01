"""Vector addition GPU demo: compile CUDA C kernel, launch, verify vs NumPy."""

from __future__ import annotations

import ctypes
import sys

import numpy as np

VECTOR_ADD_SRC = r"""
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""

BLOCK_SIZE = 256


def run_vector_add(n: int = 1_000_000):
    """Compile and launch vector_add kernel; verify against NumPy; return BenchmarkResult."""
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
    a_host = rng.random(n, dtype=np.float32)
    b_host = rng.random(n, dtype=np.float32)
    c_ref = a_host + b_host

    n_bytes = n * np.dtype(np.float32).itemsize

    stream = device.create_stream()

    with DeviceBuffer(n_bytes, stream=stream, device=device) as d_a, DeviceBuffer(
        n_bytes, stream=stream, device=device
    ) as d_b, DeviceBuffer(n_bytes, stream=stream, device=device) as d_c:
        # H2D copies
        (err,) = cudart.cudaMemcpy(
            d_a.handle,
            a_host.ctypes.data,
            n_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if err.value != 0:
            raise RuntimeError(f"H2D copy a failed: {err.value}")

        (err,) = cudart.cudaMemcpy(
            d_b.handle,
            b_host.ctypes.data,
            n_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if err.value != 0:
            raise RuntimeError(f"H2D copy b failed: {err.value}")

        compiler = KernelCompiler(device)
        kernel = compiler.compile(VECTOR_ADD_SRC, "vector_add", cache_key="vector_add", stream=stream)

        grid = kernel.compute_grid_1d(n, BLOCK_SIZE)
        block = (BLOCK_SIZE, 1, 1)

        runner = BenchmarkRunner(n_repeats=5, warmup=1)

        def gpu_fn():
            kernel.launch(
                grid,
                block,
                [d_a.handle, d_b.handle, d_c.handle, np.int32(n)],
                stream=stream,
            )

        def cpu_fn():
            return a_host + b_host

        cpu_time = runner.time_cpu(cpu_fn)
        gpu_time = runner.time_gpu(gpu_fn, stream)

        # D2H copy result
        c_gpu = np.empty(n, dtype=np.float32)
        stream.sync()
        (err,) = cudart.cudaMemcpy(
            c_gpu.ctypes.data,
            d_c.handle,
            n_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if err.value != 0:
            raise RuntimeError(f"D2H copy c failed: {err.value}")

    stream.sync()
    stream.close()

    max_err = float(np.max(np.abs(c_gpu - c_ref)))
    correct = max_err < 1e-5
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

    from src.utils.timing import BenchmarkResult

    result = BenchmarkResult(
        demo_name="vector_add",
        cpu_time_mean_s=cpu_time,
        gpu_time_mean_s=gpu_time,
        speedup=speedup,
        correct=correct,
        max_abs_error=max_err,
    )
    return result
