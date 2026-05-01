# GEMM: Overview

GEMM (General Matrix-Matrix Multiply) computes `C = A @ B` where A is `(M, K)`, B is `(K, N)`, and C is `(M, N)`. It is the most important kernel in deep learning — every dense layer, convolution (im2col), and attention score is ultimately a GEMM. Understanding how to write and optimize a GEMM kernel teaches the core GPU programming concepts: 2D thread indexing, shared memory tiling, memory coalescing, and occupancy.

This demo implements a naive (un-tiled) GEMM where each thread computes one output element by iterating over the K inner dimension. It is not production-fast — cuBLAS is orders of magnitude faster — but it is simple enough to understand every line, which is the goal.

```
Naive GEMM: each thread computes C[row, col]

Thread (row, col):
  sum = 0
  for k in range(K):
      sum += A[row, k] * B[k, col]
  C[row, col] = sum

Grid: ceil(N / BLOCK_DIM) × ceil(M / BLOCK_DIM) blocks
Block: (BLOCK_DIM, BLOCK_DIM) threads   [BLOCK_DIM = 16]
```

**Source:** `demos/05_kernels/gemm.py:1–28`
