# K-Means: Custom CUDA Kernels

The GPU implementation uses two kernels. `assign_labels` runs one thread per sample: each thread loops over all `k` centroids, computes squared distances, and writes the best label. `accumulate_centroids` uses `atomicAdd` to safely accumulate per-class sums from multiple threads simultaneously — without a lock or serial reduction.

`atomicAdd` is the key primitive: it performs a read-modify-write on a global memory location in one indivisible operation, so any number of threads can accumulate into the same centroid slot without race conditions.

```cuda
// Kernel 1: assign labels — one thread per sample
__global__ void assign_labels(const float* X, const float* centroids,
                               int* labels, int n_samples, int n_features, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_samples) return;
    float best_dist = 1e38f; int best_label = 0;
    for (int c = 0; c < k; c++) {
        float dist = 0.0f;
        for (int f = 0; f < n_features; f++) {
            float diff = X[i * n_features + f] - centroids[c * n_features + f];
            dist += diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best_label = c; }
    }
    labels[i] = best_label;
}

// Kernel 2: accumulate — atomicAdd prevents race conditions
    atomicAdd(&counts[c], 1);
    atomicAdd(&new_centroids[c * n_features + f], X[i * n_features + f]);
```

**Source:** `demos/02_kmeans/gpu_kmeans.py:10–60`
