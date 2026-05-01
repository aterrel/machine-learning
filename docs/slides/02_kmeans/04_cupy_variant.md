# K-Means: CuPy Variant

The CuPy variant avoids writing any CUDA C by expressing the assignment step as a matrix operation. The key identity `||x - c||² = ||x||² - 2·x·cᵀ + ||c||²` decomposes pairwise distances into three terms, all computable as matrix multiplications and reductions — no `(n_samples, k, n_features)` intermediate tensor required.

`X_sq` is precomputed once outside the loop (doesn't change across iterations). `X_gpu @ centroids.T` is a `(n_samples, k)` matmul dispatched to cuBLAS. `cp.argmin` runs a GPU reduction. All data stays on the GPU; only the convergence check scalar is transferred to the host.

```python
# Precompute squared norms of X rows once (unchanged across iterations)
X_sq = cp.sum(X_gpu ** 2, axis=1, keepdims=True)  # (n_samples, 1)

for _ in range(max_iter):
    # Distance identity: ||x-c||^2 = ||x||^2 - 2*x@c^T + ||c||^2
    c_sq  = cp.sum(centroids ** 2, axis=1)             # (k,)
    cross = X_gpu @ centroids.T                         # (n_samples, k)
    dists = X_sq - 2.0 * cross + c_sq[cp.newaxis, :]   # (n_samples, k)

    new_labels = cp.argmin(dists, axis=1).astype(cp.int32)
```

**Source:** `demos/02_kmeans/cupy_kmeans.py:43–52`
