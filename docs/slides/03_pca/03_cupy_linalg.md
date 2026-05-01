# PCA: CuPy with GPU Eigendecomposition

The CuPy variant expresses all three PCA steps as high-level array operations. Mean-centering is a broadcast subtraction, covariance is a matrix multiplication, and eigendecomposition uses `cp.linalg.eigh` which calls NVIDIA's cuSOLVER library under the hood — keeping the entire computation on the GPU. No D2H copy is needed until the final result extraction.

Contrast with the cuda-python variant, which copies the covariance matrix back to the host and calls `np.linalg.eigh`. For small feature counts (≤ a few hundred) the overhead of the D2H copy is negligible, but for large feature spaces the CuPy path avoids it entirely.

```python
X_gpu = cp.asarray(X, dtype=cp.float32)
n_samples = X_gpu.shape[0]

# Step 1: mean-center (broadcast)
means = X_gpu.mean(axis=0)
X_c   = X_gpu - means

# Step 2: covariance (matmul, stays on GPU)
C = (X_c.T @ X_c) / (n_samples - 1)

# Step 3: eigendecomposition via cuSOLVER — stays on GPU
eigenvalues, eigenvectors = cp.linalg.eigh(C)

# Sort descending and project
idx = cp.argsort(eigenvalues)[::-1]
components    = eigenvectors[:, idx][:, :n_components].T
X_transformed = X_c @ eigenvectors[:, idx][:, :n_components]
```

**Source:** `demos/03_pca/cupy_pca.py:31–51`
