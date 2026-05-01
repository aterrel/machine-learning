# Backend Comparison: Vector Addition

All three GPU Python backends perform the same vector addition, but at different abstraction levels. cuda-python requires you to write the CUDA C kernel, compile it, and manage device memory manually — most code, most control. Numba lets you write the kernel in Python with `@cuda.jit`, eliminating CUDA C but still exposing thread indexing. CuPy's `cp.add` has no kernel code at all; it dispatches to cuBLAS/cuDNN internally.

Choose the backend that matches your need: cuda-python for novel kernels or exact control, Numba for iterative prototyping with CUDA semantics, CuPy when an existing operator already does what you need.

```python
# cuda-python — write CUDA C, compile, launch
kernel.launch(grid, block, [d_a.handle, d_b.handle, d_c.handle, np.int32(n)], stream=stream)

# Numba — Python kernel with @cuda.jit
@nb_cuda.jit
def _vector_add(a, b, c):
    i = nb_cuda.grid(1)
    if i < c.shape[0]:
        c[i] = a[i] + b[i]
_vector_add[grid, BLOCK_SIZE](d_a, d_b, d_c)

# CuPy — no kernel code; uses cuBLAS/cuDNN internally
c_gpu = cp.add(a_gpu, b_gpu)
```

**Source:** `demos/01_core_apis/vector_add.py:79–86`, `demos/01_core_apis/numba_vector_add.py:15–19`, `demos/01_core_apis/cupy_vector_add.py`
