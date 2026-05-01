# End-to-End Interop Pipeline

The pipeline demo shows how cuda-python, CuPy, and PyTorch can be chained in a single computation without ever moving data back to the host between steps. All three libraries operate on the same physical GPU memory allocation throughout.

The five-step sequence: (1) allocate with cuda-python and H2D copy input, (2) run a custom ReLU kernel in-place, (3) wrap the result as a CuPy array (zero-copy) and compute L2 norm, (4) wrap as a PyTorch tensor (zero-copy via CuPy) and compute sum, (5) D2H copy for correctness verification. Steps 2–4 all reference the same device pointer.

```python
with DeviceBuffer(n_bytes, stream=stream, device=device) as buf:
    # [1] H2D copy
    cudart.cudaMemcpy(buf.handle, x_host.ctypes.data, n_bytes, H2D)
    stream.sync()

    # [2] Custom ReLU kernel on the buffer (in-place)
    relu_kernel.launch(grid, (BLOCK_SIZE, 1, 1),
                       [buf.handle, np.int32(n)], stream=stream)
    stream.sync()

    # [3] CuPy wraps the same GPU memory (zero-copy)
    mem    = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(buf.handle, n_bytes, buf), 0)
    cp_arr = cp.ndarray(shape=(n,), dtype=cp.float32, memptr=mem)
    norm   = float(cp.linalg.norm(cp_arr))

    # [4] PyTorch wraps CuPy (zero-copy chain: cuda-python → CuPy → torch)
    t   = torch.as_tensor(cp_arr, device="cuda")
    s   = float(t.sum())
```

**Source:** `demos/06_interop/pipeline.py:71–131`
