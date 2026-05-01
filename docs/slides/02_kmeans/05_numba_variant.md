# K-Means: Numba-CUDA Variant

Numba-CUDA lets you write the kernel in Python with `@cuda.jit`. The kernel body looks like CUDA C but uses Python syntax — `nb_cuda.grid(1)` replaces `blockDim.x * blockIdx.x + threadIdx.x`, and `nb_cuda.atomic.add` provides the atomic accumulation. Arrays are transferred with `nb_cuda.to_device()` and retrieved with `.copy_to_host()`.

The `[grid, BLOCK_SIZE]` launch syntax is Numba's shorthand for the grid/block configuration. At first call the kernel is JIT-compiled to PTX; subsequent calls use the cached binary. The `BenchmarkRunner(warmup=1)` in the benchmark runner ensures this compilation cost is excluded from timing.

```python
@nb_cuda.jit
def _assign_labels(X, centroids, labels):
    i = nb_cuda.grid(1)
    if i >= X.shape[0]:
        return
    best_dist, best_label = 1e38, 0
    for c in range(centroids.shape[0]):
        dist = 0.0
        for f in range(X.shape[1]):
            diff = X[i, f] - centroids[c, f]
            dist += diff * diff
        if dist < best_dist:
            best_dist = dist; best_label = c
    labels[i] = best_label

d_X = nb_cuda.to_device(X)
_assign_labels[grid, BLOCK_SIZE](d_X, d_centroids, d_labels)
nb_cuda.synchronize()
```

**Source:** `demos/02_kmeans/numba_kmeans.py:14–46`
