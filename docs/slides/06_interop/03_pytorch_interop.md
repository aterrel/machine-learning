# PyTorch Interop

PyTorch CUDA tensors expose their device pointer via `tensor.data_ptr()`, an integer that can be passed directly to a CUDA Python kernel launch. This is the CuPy → cuda-python pattern applied to PyTorch: no copy, no conversion, just the raw device address.

In the other direction (cuda-python → PyTorch), the cleanest zero-copy path routes through CuPy: cuda-python buffer → CuPy `UnownedMemory` → `torch.as_tensor(cp_arr, device="cuda")`. PyTorch reads CuPy's `__cuda_array_interface__` and references the same physical memory. Without CuPy, the fallback is D2H → `torch.from_numpy()` → `.cuda()`, which involves a data copy.

```python
# PyTorch → cuda-python: extract raw pointer, pass to kernel
t       = torch.from_numpy(x_host.copy()).cuda()
raw_ptr = t.data_ptr()   # integer device pointer
scale_kernel.launch(grid, (BLOCK_SIZE, 1, 1),
                    [raw_ptr, np.float32(3.0), np.int32(n)], stream=stream)
# Result is visible in `t` immediately — same GPU memory

# cuda-python → PyTorch via CuPy (zero-copy chain)
mem    = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(buf.handle, n_bytes, buf), 0)
cp_arr = cp.ndarray(shape=(n,), dtype=cp.float32, memptr=mem)
t      = torch.as_tensor(cp_arr, device="cuda")  # reads __cuda_array_interface__
```

**Source:** `demos/06_interop/torch_interop.py:151–158`, `demos/06_interop/torch_interop.py:79–85`
