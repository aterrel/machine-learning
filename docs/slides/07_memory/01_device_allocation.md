# Device Memory: Allocation and Queries

The `DeviceBuffer` context manager wraps `device.allocate(n_bytes)` and guarantees `buf.close()` is called on exit — even if an exception is raised inside the `with` block. This prevents GPU memory leaks in error-prone code paths. `query_device_memory(device)` calls `cudaMemGetInfo` to report current free and total GPU memory, useful for confirming allocations and releases.

Unlike CPU heap memory, GPU memory is not garbage-collected by Python's reference counting. A DeviceBuffer not freed explicitly will hold its allocation for the lifetime of the process. Context managers are the primary safeguard.

```python
from src.utils.memory import DeviceBuffer, query_device_memory

n_bytes = 256 * 1024 * 1024  # 256 MB

mem_before = query_device_memory(device)
print(f"Before: free={mem_before['free_gb']:.3f} GB")

with DeviceBuffer(n_bytes, device=device) as buf:
    mem_during = query_device_memory(device)
    print(f"During: free={mem_during['free_gb']:.3f} GB  (handle={buf.handle:#x})")
    # ... kernel launches using buf.handle ...

# buf.close() called automatically by __exit__
mem_after = query_device_memory(device)
print(f"After:  free={mem_after['free_gb']:.3f} GB  (restored)")
```

**Source:** `demos/07_memory/main.py:23–54`
