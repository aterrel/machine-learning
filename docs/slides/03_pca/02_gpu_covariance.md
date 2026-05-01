# PCA: GPU Covariance Kernel

The covariance kernel launches a 2D grid of `(n_features × n_features)` blocks, one per `(i, j)` pair. Each thread computes one element of the covariance matrix by looping over all `n_samples` rows of the mean-centered data. Because the matrix is symmetric, the kernel computes the upper triangle and mirrors it to the lower triangle in the same thread — halving effective work.

The mean-centering kernel runs first on a flat 1D grid over all `n_samples × n_features` elements. Each thread recovers the feature index `f = i % n_features` and subtracts the precomputed per-feature mean. Both kernels operate in-place on the device buffer.

```cuda
// Mean-centering kernel (1D grid, in-place)
__global__ void mean_center(float* X, const float* means,
                             int n_samples, int n_features) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_samples * n_features) {
        int f = i % n_features;
        X[i] -= means[f];
    }
}

// Covariance kernel (2D grid: n_features × n_features)
__global__ void compute_cov(const float* X_c, float* C,
                             int n_samples, int n_features) {
    int i = blockIdx.x; int j = blockIdx.y;
    if (i >= n_features || j >= n_features || j < i) return;  // upper triangle
    float s = 0.0f;
    for (int k = 0; k < n_samples; k++)
        s += X_c[k * n_features + i] * X_c[k * n_features + j];
    s /= (n_samples - 1);
    C[i * n_features + j] = s;
    C[j * n_features + i] = s;  // mirror
}
```

**Source:** `demos/03_pca/gpu_pca.py:10–37`
