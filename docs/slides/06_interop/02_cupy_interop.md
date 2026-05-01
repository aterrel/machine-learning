# CuPy Interop: Zero-Copy Wrapping

There are two directions: cuda-python → CuPy and CuPy → cuda-python. In the first direction, `cp.cuda.UnownedMemory` wraps the raw device pointer (an integer from `buf.handle`) without taking ownership of it. `cp.ndarray(..., memptr=mem)` then presents that memory as a full CuPy array, enabling all CuPy operations on the existing GPU allocation with no data copy.

In the second direction, `cp_arr.data.ptr` extracts the integer device pointer from a CuPy array. This pointer can be passed directly to a CUDA Python kernel launch as an integer argument — the kernel sees the same physical GPU memory the CuPy array holds.

```python
# Direction 1: cuda-python DeviceBuffer → CuPy array (zero-copy)
mem = cp.cuda.MemoryPointer(
    cp.cuda.UnownedMemory(buf.handle, n_bytes, buf),  # buf: keep-alive reference
    0,
)
cp_arr = cp.ndarray(shape=(n,), dtype=cp.float32, memptr=mem)
gpu_sum = float(cp.sum(cp_arr))   # operates on the same GPU memory

# Direction 2: CuPy array → cuda-python kernel (zero-copy)
cp_arr  = cp.array(x_host)        # allocates on GPU
raw_ptr = cp_arr.data.ptr         # raw integer device pointer
scale_kernel.launch(grid, (BLOCK_SIZE, 1, 1),
                    [raw_ptr, factor, np.int32(n)], stream=stream)
```

**Source:** `demos/06_interop/cupy_interop.py:76–83`, `demos/06_interop/cupy_interop.py:139–146`
