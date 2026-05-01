# Pinned (Page-Locked) Memory

Pinned memory is host memory that the OS guarantees will never be swapped to disk. Because the GPU's DMA engine can read pinned memory directly without OS involvement, Host-to-Device (H2D) transfers from pinned buffers are significantly faster than from ordinary pageable heap allocations — typically 1.5–3× faster on PCIe 4.0 links.

`alloc_pinned(n_bytes)` calls `cudaHostAlloc` and returns both a raw integer pointer (for `cudaMemcpy`) and a numpy array view (for filling data on the host). Always pair with `free_pinned(ptr)` when done; pinned allocations are a scarce system resource.

```python
from src.utils.memory import alloc_pinned, free_pinned

n_bytes  = n_mb * 1024 * 1024
ptr, arr = alloc_pinned(n_bytes)   # arr is a numpy float32 view

arr[:] = data  # fill via numpy

(err,) = cudart.cudaMemcpy(
    d_buf.handle,
    ptr,                                          # raw pinned pointer
    n_bytes,
    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
)
stream.sync()
free_pinned(ptr)
```

**Source:** `src/utils/memory.py:53–67`, `demos/01_core_apis/pinned_memory.py:37–57`
