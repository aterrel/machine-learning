# Kernels: Backend Comparison

GEMM and ReLU show the widest abstraction gap across backends. cuda-python and Numba both require you to write the kernel; CuPy dispatches `cp.matmul` to cuBLAS (which uses tensor cores on Ampere+) and has a built-in `cp.maximum` for ReLU. The cuda-python naive GEMM will always be slower than cuBLAS — the comparison illustrates the cost of correctness-over-performance education code versus a hand-tuned library.

For custom activation functions not in cuBLAS/cuDNN, the cuda-python or Numba path is the right choice. For standard GEMM in a real model, use `cp.matmul` or PyTorch.

```python
# cuda-python: naive GEMM kernel (educational, not cuBLAS-fast)
kernel.launch((grid_x, grid_y, 1), (BLOCK_DIM, BLOCK_DIM, 1),
              [d_A.handle, d_B.handle, d_C.handle,
               np.int32(M), np.int32(N), np.int32(K)], stream=stream)

# CuPy GEMM: dispatches to cuBLAS (uses tensor cores on Ampere+)
C_gpu = cp.matmul(A_gpu, B_gpu)

# Numba ReLU kernel
@nb_cuda.jit
def _relu_inplace(x):
    i = nb_cuda.grid(1)
    if i < x.shape[0] and x[i] < 0.0:
        x[i] = 0.0

# CuPy ReLU: built-in ufunc
x_relu = cp.maximum(x_gpu, 0.0)
```

**Source:** `demos/05_kernels/gemm.py:116–123`, `demos/05_kernels/cupy_kernels.py`, `demos/05_kernels/numba_kernels.py`
