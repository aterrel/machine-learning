# Device Memory Allocation

GPU memory must be allocated explicitly before copying data to the device. `device.allocate(n_bytes)` returns a device buffer whose lifetime you are responsible for. `DeviceBuffer` wraps this in a context manager: `__exit__` calls `buf.close()` which frees the allocation even if an exception occurs mid-kernel. The raw integer device pointer is accessed via `buf.handle` and passed directly to kernel launches or `cudaMemcpy` calls.

GPU memory is not garbage-collected — any buffer not explicitly freed will leak for the lifetime of the Python process. Context managers are the safest pattern, especially in loops or error-prone code paths.

```python
from src.utils.memory import DeviceBuffer

n_bytes = n * 4  # float32: 4 bytes per element

with DeviceBuffer(n_bytes, stream=stream, device=device) as d_a, \
     DeviceBuffer(n_bytes, stream=stream, device=device) as d_b, \
     DeviceBuffer(n_bytes, stream=stream, device=device) as d_c:

    # d_a.handle is the raw integer device pointer
    cudart.cudaMemcpy(d_a.handle, a_host.ctypes.data, n_bytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    # ... kernel launch ...
# buffers freed automatically here
```

**Source:** `src/utils/memory.py:10–46`, `demos/01_core_apis/vector_add.py:49–51`
