# GEMM: The Kernel

The naive GEMM kernel uses a 2D thread block `(BLOCK_DIM × BLOCK_DIM)` and a 2D grid sized to cover the output matrix. Each thread computes `row = blockDim.y * blockIdx.y + threadIdx.y` and `col = blockDim.x * blockIdx.x + threadIdx.x`, then accumulates the dot product over the `K` inner dimension. The bounds check `if (row >= M || col >= N) return` handles non-square matrices cleanly.

At `BLOCK_DIM = 16` each block has 256 threads, which is a common occupancy sweet spot for compute-bound kernels. A production kernel would add shared-memory tiling to reuse A and B tiles across threads in the same block — reducing global memory bandwidth by `BLOCK_DIM×`.

```cuda
extern "C" __global__
void naive_gemm(const float* A, const float* B, float* C,
                int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// Launch: 2D grid of 16x16 blocks
int grid_x = (N + BLOCK_DIM - 1) / BLOCK_DIM;
int grid_y = (M + BLOCK_DIM - 1) / BLOCK_DIM;
kernel.launch((grid_x, grid_y, 1), (BLOCK_DIM, BLOCK_DIM, 1), [...])
```

**Source:** `demos/05_kernels/gemm.py:15–26`, `demos/05_kernels/gemm.py:97–101`
