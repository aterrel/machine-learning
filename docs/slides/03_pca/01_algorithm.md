# PCA: Algorithm

Principal Component Analysis finds the directions of maximum variance in the data. The standard approach: (1) mean-center each feature, (2) compute the covariance matrix `C = XᶜᵀXᶜ / (n-1)`, (3) eigendecompose `C` to get eigenvectors (principal components) and eigenvalues (explained variance). Projecting the data onto the top-k eigenvectors gives the low-dimensional representation.

The GPU accelerates steps 1 and 2 — mean-centering and covariance — using custom kernels, while step 3 (eigendecomposition) runs on the CPU via `np.linalg.eigh` in the cuda-python variant, or on the GPU via `cp.linalg.eigh` (which calls cuSOLVER) in the CuPy variant.

```
Input:  X (n_samples, n_features), n_components

Steps:
  1. Xᶜ = X - mean(X, axis=0)          # mean-center (GPU kernel)
  2. C  = Xᶜᵀ @ Xᶜ / (n - 1)          # covariance  (GPU kernel)
  3. eigenvalues, eigenvectors = eigh(C) # eigendecompose
  4. Sort descending; keep top k
  5. X_transformed = Xᶜ @ eigenvectors[:, :k]

Output: components (k, n_features), explained_variance (k,), X_transformed (n_samples, k)
```

**Source:** `demos/03_pca/gpu_pca.py:42–72` (function signature and docstring)
