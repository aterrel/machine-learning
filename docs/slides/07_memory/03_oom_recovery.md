# OOM Recovery

When a `DeviceBuffer` allocation fails because the requested size exceeds available GPU memory, the CUDA runtime raises an error which propagates as a Python exception. The `demo_oom_recovery` demo deliberately requests more than the total GPU memory to trigger this, catches the exception, and confirms that (a) execution continues normally, (b) device memory is unchanged (no partial allocation), and (c) subsequent small allocations succeed.

Catching OOM errors is important in production code: model training loops that increase batch size dynamically should catch OOM, halve the batch size, and retry rather than crashing. Always verify that memory is truly released (via `query_device_memory`) before concluding recovery is clean.

```python
# Attempt to allocate more than total device memory (guaranteed OOM)
oom_bytes = int((total_gb + 1.0) * (1024 ** 3))

try:
    buf = DeviceBuffer(oom_bytes, device=device)
    buf.close()  # should not reach here
except Exception as exc:
    print(f"OOM caught: {type(exc).__name__}: {exc}")
    # device memory is unchanged — no partial allocation occurred

# Confirm device memory is intact
mem_after = query_device_memory(device)
print(f"After OOM: free={mem_after['free_gb']:.3f} GB  (unchanged)")

# Small allocation after OOM still works
with DeviceBuffer(1024 * 1024, device=device) as small_buf:
    print(f"Small alloc OK: handle={small_buf.handle:#x}")
```

**Source:** `demos/07_memory/main.py:131–179`
