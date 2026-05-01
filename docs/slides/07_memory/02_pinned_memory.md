# Pinned Memory: Transfer Speed

Pinned (page-locked) host memory enables the GPU's DMA engine to read data directly from CPU memory without OS involvement. The benchmark in `demo_pinned_vs_pageable` transfers the same array using `cudaHostAlloc`-allocated memory vs. ordinary heap memory and measures the throughput difference. On PCIe 4.0 x16 links, pinned transfers are typically 1.5–3× faster.

The tradeoff: pinned memory is a scarce system resource. Over-allocating it can degrade overall system performance. Use it for large, frequently-transferred arrays (training batches, model weights) but not for small or one-time copies.

```python
from src.utils.memory import alloc_pinned, free_pinned

# --- Pinned: allocated with cudaHostAlloc ---
ptr, pinned_arr = alloc_pinned(n_bytes)
pinned_arr[:] = data                          # fill via numpy view

t0 = time.perf_counter()
cudart.cudaMemcpy(d_buf.handle, ptr, n_bytes, H2D)
stream.sync()
pinned_time = time.perf_counter() - t0

free_pinned(ptr)

# --- Pageable: ordinary numpy array ---
pageable_arr = np.random.random(n_floats).astype(np.float32)
t0 = time.perf_counter()
cudart.cudaMemcpy(d_buf.handle, pageable_arr.ctypes.data, n_bytes, H2D)
stream.sync()
pageable_time = time.perf_counter() - t0
```

**Source:** `demos/01_core_apis/pinned_memory.py:37–78`
