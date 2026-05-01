# CUDA Python ML Demos — Slide Index

One concept per slide. Each slide includes a code snippet with a file:line reference to the actual source.

Run the demos: `python benchmarks/run_all.py` (requires NVIDIA GPU + CUDA 12.x)

---

## 00. Introduction

| # | File | Concept |
|---|------|---------|
| 1 | [01_what_is_cuda_python.md](00_introduction/01_what_is_cuda_python.md) | What CUDA Python is and how it differs from CuPy / Numba |
| 2 | [02_project_overview.md](00_introduction/02_project_overview.md) | Repository structure and how the demos are organized |

## 01. Core APIs

| # | File | Concept |
|---|------|---------|
| 3 | [01_device_setup.md](01_core_apis/01_device_setup.md) | Selecting a GPU device and querying its properties |
| 4 | [02_memory_allocation.md](01_core_apis/02_memory_allocation.md) | Allocating device memory with `DeviceBuffer` context manager |
| 5 | [03_kernel_compilation.md](01_core_apis/03_kernel_compilation.md) | Compiling CUDA C strings at runtime via NVRTC |
| 6 | [04_kernel_launch.md](01_core_apis/04_kernel_launch.md) | Grid/block sizing and launching a compiled kernel |
| 7 | [05_pinned_memory.md](01_core_apis/05_pinned_memory.md) | Page-locked host memory for faster H2D transfers |
| 8 | [06_backend_comparison.md](01_core_apis/06_backend_comparison.md) | Vector add in cuda-python, Numba, and CuPy side-by-side |

## 02. K-Means Clustering

| # | File | Concept |
|---|------|---------|
| 9  | [01_algorithm.md](02_kmeans/01_algorithm.md) | Lloyd's algorithm: assign + accumulate + normalize |
| 10 | [02_cpu_baseline.md](02_kmeans/02_cpu_baseline.md) | NumPy broadcast distance computation |
| 11 | [03_gpu_kernel.md](02_kmeans/03_gpu_kernel.md) | Custom CUDA kernels with `atomicAdd` for centroid accumulation |
| 12 | [04_cupy_variant.md](02_kmeans/04_cupy_variant.md) | CuPy distance identity: `‖x-c‖² = ‖x‖² - 2xcᵀ + ‖c‖²` |
| 13 | [05_numba_variant.md](02_kmeans/05_numba_variant.md) | Numba `@cuda.jit` kernels and `nb_cuda.atomic.add` |

## 03. PCA

| # | File | Concept |
|---|------|---------|
| 14 | [01_algorithm.md](03_pca/01_algorithm.md) | Mean-center → covariance → eigendecompose → project |
| 15 | [02_gpu_covariance.md](03_pca/02_gpu_covariance.md) | 2D kernel grid for symmetric covariance computation |
| 16 | [03_cupy_linalg.md](03_pca/03_cupy_linalg.md) | `cp.linalg.eigh` keeps eigendecomposition on GPU (cuSOLVER) |
| 17 | [04_backend_comparison.md](03_pca/04_backend_comparison.md) | Where eigendecomposition runs: CPU vs GPU per backend |

## 04. Linear Regression

| # | File | Concept |
|---|------|---------|
| 18 | [01_normal_equations.md](04_linear_model/01_normal_equations.md) | Closed-form solution via XᵀX · w = Xᵀy |
| 19 | [02_gpu_xtx_kernel.md](04_linear_model/02_gpu_xtx_kernel.md) | Fused XᵀX + Xᵀy kernel with 2D grid |
| 20 | [03_backend_comparison.md](04_linear_model/03_backend_comparison.md) | cuda-python custom kernel vs CuPy `linalg.solve` |

## 05. Custom Kernels (GEMM / ReLU / Softmax)

| # | File | Concept |
|---|------|---------|
| 21 | [01_gemm_overview.md](05_kernels/01_gemm_overview.md) | Why GEMM matters and the naive thread-per-output-element approach |
| 22 | [02_gemm_kernel.md](05_kernels/02_gemm_kernel.md) | 2D thread block, bounds checking, inner-K loop |
| 23 | [03_relu_softmax.md](05_kernels/03_relu_softmax.md) | ReLU (bandwidth-bound) and educational softmax kernel |
| 24 | [04_backend_comparison.md](05_kernels/04_backend_comparison.md) | Naive GEMM vs cuBLAS (`cp.matmul`) vs Numba ReLU |

## 05. Gaussian Naive Bayes

| # | File | Concept |
|---|------|---------|
| 25 | [01_algorithm.md](05_naive_bayes/01_algorithm.md) | Generative classifier: fit on CPU, infer on GPU |
| 26 | [02_gpu_log_likelihood.md](05_naive_bayes/02_gpu_log_likelihood.md) | 2D kernel grid over (test samples × classes) |
| 27 | [03_cupy_broadcasting.md](05_naive_bayes/03_cupy_broadcasting.md) | Vectorized log-likelihood via 3D broadcasting |
| 28 | [04_backend_comparison.md](05_naive_bayes/04_backend_comparison.md) | 2D kernel vs broadcasting; `import math as _math` in Numba |

## 06. Interop

| # | File | Concept |
|---|------|---------|
| 29 | [01_cuda_array_interface.md](06_interop/01_cuda_array_interface.md) | `__cuda_array_interface__` protocol for zero-copy sharing |
| 30 | [02_cupy_interop.md](06_interop/02_cupy_interop.md) | `UnownedMemory` wrapping and `data.ptr` extraction |
| 31 | [03_pytorch_interop.md](06_interop/03_pytorch_interop.md) | `tensor.data_ptr()` and `torch.as_tensor(cp_arr)` |
| 32 | [04_pipeline.md](06_interop/04_pipeline.md) | End-to-end: cuda-python kernel → CuPy norm → PyTorch sum |

## 07. Memory Management

| # | File | Concept |
|---|------|---------|
| 33 | [01_device_allocation.md](07_memory/01_device_allocation.md) | `DeviceBuffer` context manager and `query_device_memory` |
| 34 | [02_pinned_memory.md](07_memory/02_pinned_memory.md) | Pinned vs pageable transfer throughput benchmark |
| 35 | [03_oom_recovery.md](07_memory/03_oom_recovery.md) | Catching and recovering from CUDA OOM errors |

## 08. Multi-Backend Comparison

| # | File | Concept |
|---|------|---------|
| 36 | [01_abstraction_levels.md](08_comparison/01_abstraction_levels.md) | The abstraction ladder: cuda-python → Numba → CuPy → cuML |
| 37 | [02_reading_the_table.md](08_comparison/02_reading_the_table.md) | How to interpret speedup, correctness, and SKIPPED rows |

---

*37 slides total across 10 sections. All code snippets verified against source files.*
