"""CUDA Python buffer <-> PyTorch CUDA tensor interop."""

from __future__ import annotations

import sys

import numpy as np


_SCALE_SRC = r"""
extern "C" __global__
void scale_inplace(float* x, float factor, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}
"""

BLOCK_SIZE = 256


def demo_cuda_python_to_torch(n: int = 1_000_000) -> None:
    """Show how to wrap a CUDA Python DeviceBuffer as a PyTorch CUDA tensor.

    Strategy: cuda-python allocates device memory, we describe it via
    __cuda_array_interface__, then route through CuPy (if available) to reach
    torch.as_tensor().  If CuPy is absent, we print what the interface looks
    like and explain the bridging options.
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not installed — skipping torch interop demo. Install with: pip install torch")
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
        # H2D: copy host data into CUDA Python managed buffer
        (err,) = cudart.cudaMemcpy(
            buf.handle,
            x_host.ctypes.data,
            n_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if err.value != 0:
            raise RuntimeError(f"H2D copy failed: {err.value}")
        stream.sync()

        print(f"  demo_cuda_python_to_torch (n={n:,})")
        print(f"    Device buffer pointer: {buf.handle:#x}")

        # Bridge via CuPy if available (cuda-python → CuPy → PyTorch).
        # Step 1: CuPy wraps the raw pointer via UnownedMemory (zero-copy).
        # Step 2: torch.as_tensor() reads CuPy's __cuda_array_interface__ (zero-copy).
        try:
            import cupy as cp

            # Zero-copy: CuPy owns no memory — it references buf.handle directly.
            mem = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(buf.handle, n_bytes, buf),
                0,
            )
            cp_arr = cp.ndarray(shape=(n,), dtype=cp.float32, memptr=mem)

            # Zero-copy: PyTorch consumes CuPy's __cuda_array_interface__
            t = torch.as_tensor(cp_arr, device="cuda")

            gpu_sum = float(t.sum())
            cpu_sum = float(x_host.sum())
            print(f"    Bridged via CuPy: torch sum={gpu_sum:.4f}, numpy sum={cpu_sum:.4f}")
            print(f"    Match: {abs(gpu_sum - cpu_sum) < 1.0}")

        except ImportError:
            # Without CuPy: explain the available routes
            print(
                "    CuPy not installed — zero-copy bridge unavailable.\n"
                "    Alternative routes:\n"
                "      1. Install CuPy: pip install cupy-cuda12x\n"
                "      2. D2H to numpy → torch.from_numpy() → .cuda() (copies data)\n"
                "      3. Implement __dlpack__ on DeviceBuffer for direct torch.from_dlpack()"
            )
            # Fallback: D2H → CPU tensor → move to GPU (copies data, not zero-copy)
            x_back = np.empty(n, dtype=np.float32)
            (err,) = cudart.cudaMemcpy(
                x_back.ctypes.data,
                buf.handle,
                n_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
            stream.sync()
            t = torch.from_numpy(x_back).cuda()  # involves a data copy
            gpu_sum = float(t.sum())
            print(f"    Fallback (D2H→CPU→GPU): torch sum={gpu_sum:.4f}")

    stream.sync()
    stream.close()


def demo_torch_to_cuda_python(n: int = 1_000_000) -> None:
    """Show how to pass a PyTorch CUDA tensor pointer to a CUDA Python kernel.

    tensor.data_ptr() returns the raw device pointer as an integer, which can
    be passed directly to a CUDA Python kernel launch.
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not installed — skipping torch interop demo. Install with: pip install torch")
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
    scale_kernel = compiler.compile(_SCALE_SRC, "scale_inplace", cache_key="scale_inplace_torch", stream=stream)

    # Create PyTorch CUDA tensor
    t = torch.from_numpy(x_host.copy()).cuda()

    # Extract raw device pointer — zero-copy handoff to CUDA Python kernel
    raw_ptr = t.data_ptr()  # integer device pointer
    factor = np.float32(3.0)

    grid = scale_kernel.compute_grid_1d(n, BLOCK_SIZE)
    scale_kernel.launch(grid, (BLOCK_SIZE, 1, 1), [raw_ptr, factor, np.int32(n)], stream=stream)
    stream.sync()

    # Verify by reading back through PyTorch
    x_scaled = t.cpu().numpy()
    expected = x_host * 3.0
    max_err = float(np.max(np.abs(x_scaled - expected)))

    print(f"  demo_torch_to_cuda_python (n={n:,})")
    print(f"    Scale factor: {float(factor):.1f}")
    print(f"    data_ptr(): {raw_ptr:#x}")
    print(f"    Max error vs numpy reference: {max_err:.2e}")
    print(f"    Correct: {max_err < 1e-5}")

    stream.sync()
    stream.close()
