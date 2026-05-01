"""CUDA Python buffer <-> CuPy interop via __cuda_array_interface__ (zero-copy)."""

from __future__ import annotations

import sys

import numpy as np


# ReLU kernel used to modify device memory in-place before wrapping as CuPy array
_RELU_SRC = r"""
extern "C" __global__
void relu_inplace(float* x, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}
"""

_SCALE_SRC = r"""
extern "C" __global__
void scale_inplace(float* x, float factor, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
"""

BLOCK_SIZE = 256


def demo_cuda_python_to_cupy(n: int = 1_000_000) -> None:
    """Show how to wrap a CUDA Python DeviceBuffer as a CuPy array (zero-copy).

    The __cuda_array_interface__ protocol lets CuPy consume a raw device pointer
    without any data copy — both sides see the same GPU memory.
    """
    try:
        import cupy as cp
    except ImportError:
        print("CuPy not installed — skipping CuPy interop demo. Install with: pip install cupy-cuda12x")
        return

    try:
        from cuda.bindings import cudart

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

    rng = np.random.default_rng(42)
    x_host = rng.standard_normal(n).astype(np.float32)
    n_bytes = x_host.nbytes

    stream = device.create_stream()

    with DeviceBuffer(n_bytes, stream=stream, device=device) as buf:
        # H2D: copy numpy array to device buffer
        (err,) = cudart.cudaMemcpy(
            buf.handle,
            x_host.ctypes.data,
            n_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if err.value != 0:
            raise RuntimeError(f"H2D copy failed: {err.value}")

        stream.sync()

        # Zero-copy: wrap the raw device pointer as a CuPy array via UnownedMemory.
        # CuPy sees the same physical GPU memory; no data is moved.
        mem = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(buf.handle, n_bytes, buf),
            0,
        )
        cp_arr = cp.ndarray(shape=(n,), dtype=cp.float32, memptr=mem)

        # Perform a CuPy operation on the wrapped array (still on the same GPU memory)
        gpu_sum = float(cp.sum(cp_arr))
        gpu_mean = float(cp.mean(cp_arr))

        # Reference on CPU
        cpu_sum = float(x_host.sum())
        cpu_mean = float(x_host.mean())

        # Show what the __cuda_array_interface__ protocol dict looks like.
        # This is the standard cross-framework descriptor; libraries that accept
        # __cuda_array_interface__ (e.g., Numba) can use it directly.
        interface = cp_arr.__cuda_array_interface__
        print(f"  demo_cuda_python_to_cupy (n={n:,})")
        print(f"    NumPy  sum={cpu_sum:.4f}  mean={cpu_mean:.6f}")
        print(f"    CuPy   sum={gpu_sum:.4f}  mean={gpu_mean:.6f}")
        print(f"    Match: {abs(gpu_sum - cpu_sum) < 1.0}")
        print(f"    __cuda_array_interface__ shape: {interface['shape']}, typestr: {interface['typestr']}, ptr: {interface['data'][0]:#x}")

    stream.sync()
    stream.close()


def demo_cupy_to_cuda_python(n: int = 1_000_000) -> None:
    """Show how to pass a CuPy array pointer to a CUDA Python kernel.

    The raw pointer (arr.data.ptr) is extracted and passed directly to the
    CUDA Python kernel launch — no copy required.
    """
    try:
        import cupy as cp
    except ImportError:
        print("CuPy not installed — skipping CuPy interop demo. Install with: pip install cupy-cuda12x")
        return

    try:
        from src.kernels.compiler import KernelCompiler
        from src.utils.device import get_device
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        sys.exit(1)

    try:
        device = get_device(0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    rng = np.random.default_rng(42)
    x_host = rng.standard_normal(n).astype(np.float32)

    stream = device.create_stream()
    compiler = KernelCompiler(device)
    scale_kernel = compiler.compile(_SCALE_SRC, "scale_inplace", cache_key="scale_inplace", stream=stream)

    # Allocate CuPy array and fill it
    cp_arr = cp.array(x_host)

    # Extract the raw device pointer — zero-copy handoff to CUDA Python
    raw_ptr = cp_arr.data.ptr  # integer device pointer
    factor = np.float32(2.0)

    grid = scale_kernel.compute_grid_1d(n, BLOCK_SIZE)
    scale_kernel.launch(grid, (BLOCK_SIZE, 1, 1), [raw_ptr, factor, np.int32(n)], stream=stream)
    stream.sync()

    # D2H for verification
    x_scaled = cp.asnumpy(cp_arr)
    expected = x_host * 2.0
    max_err = float(np.max(np.abs(x_scaled - expected)))

    print(f"  demo_cupy_to_cuda_python (n={n:,})")
    print(f"    Scale factor: {float(factor):.1f}")
    print(f"    Max error vs numpy reference: {max_err:.2e}")
    print(f"    Correct: {max_err < 1e-5}")

    stream.sync()
    stream.close()
