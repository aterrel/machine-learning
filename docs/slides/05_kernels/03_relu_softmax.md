# ReLU and Softmax Kernels

ReLU is the simplest GPU kernel: one thread per element, one operation (`fmaxf(0, x[i])`), in-place. It serves as the reference point for memory-bandwidth-bound kernel performance — ReLU does almost no arithmetic, so runtime is entirely determined by how fast the GPU can read and write global memory.

Softmax is more complex: it requires a two-pass reduction (max for numerical stability, then sum for normalization). The demo implements a single-threaded-per-row educational variant for clarity. A production softmax uses warp-level shuffle reductions and cooperative groups.

```cuda
// ReLU: one thread per element, in-place
__global__ void relu_inplace(float* x, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}

// Softmax: one block per row, thread 0 does all work (educational)
__global__ void softmax_row(float* x, int n_rows, int n_cols) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    float* row_ptr = x + row * n_cols;
    if (threadIdx.x == 0) {
        float mx = row_ptr[0];
        for (int j = 1; j < n_cols; j++) if (row_ptr[j] > mx) mx = row_ptr[j];
        float s = 0.0f;
        for (int j = 0; j < n_cols; j++) { row_ptr[j] = expf(row_ptr[j]-mx); s += row_ptr[j]; }
        for (int j = 0; j < n_cols; j++) row_ptr[j] /= s;
    }
}
```

**Source:** `demos/05_kernels/activations.py:10–33`
