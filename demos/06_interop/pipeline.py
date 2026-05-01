"""End-to-end pipeline: CUDA Python kernel → optional CuPy → optional PyTorch."""

from __future__ import annotations

import sys

import numpy as np


# ReLU kernel: applied to the raw device buffer produced by CUDA Python
_RELU_SRC = r"""
extern "C" __global__
void relu_inplace(float* x, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}
"""

BLOCK_SIZE = 256


def run_end_to_end_pipeline(n: int = 100_000) -> dict:
    """CUDA Python custom kernel → (optional) CuPy operation → result.

    Returns dict with operation results and whether each library was available.
    Steps:
      1. Allocate device buffer via CUDA Python; H2D copy a numpy array.
      2. Run custom ReLU kernel (in-place on device buffer).
      3. If CuPy is available: wrap result as CuPy array, compute L2 norm.
      4. If PyTorch is available: wrap/transfer to tensor, compute sum.
      5. D2H copy back to verify correctness.

    Always prints which steps were executed.
    """
    results: dict = {
        "n": n,
        "cupy_available": False,
        "torch_available": False,
        "relu_correct": False,
        "cupy_norm": None,
        "torch_sum": None,
    }

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

    rng = np.random.default_rng(42)
    x_host = rng.standard_normal(n).astype(np.float32)  # mix of positive and negative values
    n_bytes = x_host.nbytes
    x_ref = np.maximum(0.0, x_host)  # expected result after ReLU

    stream = device.create_stream()
    compiler = KernelCompiler(device)
    relu_kernel = compiler.compile(_RELU_SRC, "relu_inplace", cache_key="relu_pipeline", stream=stream)

    print(f"  run_end_to_end_pipeline (n={n:,})")

    with DeviceBuffer(n_bytes, stream=stream, device=device) as buf:
        # Step 1: H2D copy numpy array into CUDA Python managed buffer
        (err,) = cudart.cudaMemcpy(
            buf.handle,
            x_host.ctypes.data,
            n_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        if err.value != 0:
            raise RuntimeError(f"H2D copy failed: {err.value}")
        stream.sync()
        print("  [1] H2D copy complete")

        # Step 2: Run custom ReLU kernel on the device buffer
        grid = relu_kernel.compute_grid_1d(n, BLOCK_SIZE)
        relu_kernel.launch(grid, (BLOCK_SIZE, 1, 1), [buf.handle, np.int32(n)], stream=stream)
        stream.sync()
        print("  [2] ReLU kernel launched and synced")

        # Step 3: Optional CuPy — wrap the modified buffer as a CuPy array (zero-copy)
        try:
            import cupy as cp

            results["cupy_available"] = True
            # Zero-copy: CuPy wraps the existing device pointer
            mem = cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(buf.handle, n_bytes, buf),
                0,
            )
            cp_arr = cp.ndarray(shape=(n,), dtype=cp.float32, memptr=mem)
            norm = float(cp.linalg.norm(cp_arr))
            results["cupy_norm"] = norm
            print(f"  [3] CuPy L2 norm of ReLU output: {norm:.4f}")

        except ImportError:
            print("  [3] CuPy not installed — skipping norm computation")

        # Step 4: Optional PyTorch
        try:
            import torch

            results["torch_available"] = True
            if results["cupy_available"]:
                # Zero-copy: PyTorch wraps CuPy array (which references the same device memory)
                t = torch.as_tensor(cp_arr, device="cuda")
            else:
                # Fallback: D2H then transfer to GPU (involves a data copy)
                x_back_tmp = np.empty(n, dtype=np.float32)
                (err,) = cudart.cudaMemcpy(
                    x_back_tmp.ctypes.data,
                    buf.handle,
                    n_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
                stream.sync()
                t = torch.from_numpy(x_back_tmp).cuda()

            tensor_sum = float(t.sum())
            results["torch_sum"] = tensor_sum
            print(f"  [4] PyTorch sum of ReLU output: {tensor_sum:.4f}")

        except ImportError:
            print("  [4] PyTorch not installed — skipping tensor sum")

        # Step 5: D2H copy back for correctness verification
        x_back = np.empty(n, dtype=np.float32)
        (err,) = cudart.cudaMemcpy(
            x_back.ctypes.data,
            buf.handle,
            n_bytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if err.value != 0:
            raise RuntimeError(f"D2H copy failed: {err.value}")
        stream.sync()

    max_err = float(np.max(np.abs(x_back - x_ref)))
    results["relu_correct"] = max_err < 1e-6
    print(f"  [5] ReLU correctness check: max_err={max_err:.2e}, correct={results['relu_correct']}")

    stream.sync()
    stream.close()

    return results
