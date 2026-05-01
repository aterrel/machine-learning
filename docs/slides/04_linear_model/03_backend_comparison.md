# Linear Regression: Backend Comparison

All backends implement the same interface: `(weights, r2, mse) = linear_regression_*(X, y)`. The differences are how `XᵀX` and `Xᵀy` are computed.

cuda-python writes a custom 2D grid kernel that fuses both computations. CuPy uses `cp.linalg.solve(X.T @ X, X.T @ y)` — three lines, all dispatched to cuBLAS and cuSOLVER with no kernel code. Numba-CUDA writes a Python-syntax kernel with the same 2D thread structure as the cuda-python version. In all three cases the final `solve` or equivalent runs in `np.linalg.solve` on the CPU, since `n_features` is small.

```python
# cuda-python: custom fused XtX + Xty kernel
grid_2d = (n_features, n_features, 1)
kernel.launch(grid_2d, (1, 1, 1),
    [d_X.handle, d_y.handle, d_XtX.handle, d_Xty.handle,
     np.int32(n_samples), np.int32(n_features)], stream=stream)
weights = np.linalg.solve(XtX_host, Xty_host)

# CuPy: three lines, cuBLAS + cuSOLVER
X_gpu, y_gpu = cp.asarray(X), cp.asarray(y)
weights = cp.asnumpy(cp.linalg.solve(X_gpu.T @ X_gpu, X_gpu.T @ y_gpu))

# Numba: Python kernel, same 2D grid pattern
@nb_cuda.jit
def _xtx_xty(X, y, XtX, Xty):
    i = nb_cuda.grid(1) // n_features
    j = nb_cuda.grid(1) % n_features
    ...
```

**Source:** `demos/04_linear_model/gpu_linear.py:127–153`, `demos/04_linear_model/cupy_linear.py`
