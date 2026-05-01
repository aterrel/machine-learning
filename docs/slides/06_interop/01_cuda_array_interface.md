# The CUDA Array Interface

The `__cuda_array_interface__` protocol is a Python-level contract (not a C API) that lets GPU array libraries share device memory without copying data. Any object that implements this attribute exposes a dictionary describing its device pointer, shape, dtype, and strides. Libraries that consume the protocol — CuPy, Numba, PyTorch (via DLPack) — can wrap a raw pointer as a first-class array without any H2D/D2H transfer.

This is the key interop mechanism in the CUDA Python ecosystem: cuda-python allocates the raw memory, describes it via the interface, and downstream libraries treat it as their own array. The GPU data never leaves the device.

```python
import cupy as cp

# After wrapping a raw device pointer as a CuPy array:
cp_arr = cp.ndarray(shape=(n,), dtype=cp.float32, memptr=mem)

# The interface dictionary that downstream libraries read:
interface = cp_arr.__cuda_array_interface__
# {
#   'shape':   (1000000,),
#   'typestr': '<f4',
#   'data':    (140234567, False),  # (device_ptr, read_only)
#   'version': 3,
# }
print(f"shape:   {interface['shape']}")
print(f"typestr: {interface['typestr']}")
print(f"ptr:     {interface['data'][0]:#x}")
```

**Source:** `demos/06_interop/cupy_interop.py:92–100`
