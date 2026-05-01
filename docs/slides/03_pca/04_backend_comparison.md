# PCA: Backend Comparison

Three backends implement the same PCA interface `(components, explained_variance, X_transformed) = pca_*(X, n_components)`. The key difference is where the covariance and eigendecomposition happen.

cuda-python gives the finest control: you write the mean-centering and covariance kernels yourself, decide exactly how the 2D grid maps to the feature space, and copy results back explicitly. CuPy's version is five lines and stays fully on-GPU including the eigensolve. Numba-CUDA occupies the middle: you write a 1D covariance kernel in Python, but the eigendecomposition still falls back to NumPy on the CPU.

```python
# cuda-python: explicit 2D kernel launch for covariance
grid_2d = (n_features, n_features, 1)
cov_kernel.launch(grid_2d, (1, 1, 1),
    [d_X.handle, d_cov.handle, np.int32(n_samples), np.int32(n_features)],
    stream=stream)
eigenvalues, eigenvectors = np.linalg.eigh(C_host)  # CPU fallback

# CuPy: two lines, eigensolve stays on GPU via cuSOLVER
C = (X_c.T @ X_c) / (n_samples - 1)
eigenvalues, eigenvectors = cp.linalg.eigh(C)        # GPU cuSOLVER

# Numba: 1D kernel, n_features^2 threads, each computes C[i,j]
@nb_cuda.jit
def _compute_cov(X_c, C):
    tid = nb_cuda.grid(1); i = tid // n_features; j = tid % n_features
    ...
```

**Source:** `demos/03_pca/gpu_pca.py:148–156`, `demos/03_pca/cupy_pca.py:39–42`, `demos/03_pca/numba_pca.py`
