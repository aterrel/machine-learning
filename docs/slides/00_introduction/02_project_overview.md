# Project Overview

This repository is a collection of hands-on GPU demonstrations for ML practitioners who want low-level GPU control without leaving Python. Each demo covers one algorithm (k-means, PCA, linear regression, naive Bayes, GEMM/ReLU) and shows three implementations side-by-side: a NumPy CPU baseline, a cuda-python custom-kernel version, and one or more high-level variants (CuPy, Numba-CUDA, cuML).

The project is structured so every demo is self-contained — you can read one directory and understand it completely without cross-demo imports. The `src/` library provides shared utilities (device management, memory, timing) that each demo imports. Tests in `tests/` run without a GPU; GPU tests are marked `@pytest.mark.gpu`.

```
cuda-python-ml-demos/
├── src/
│   ├── kernels/      # NVRTC compiler + CompiledKernel wrapper
│   └── utils/        # device, memory, timing utilities
├── demos/
│   ├── 01_core_apis/ # Device setup, vector add, pinned memory
│   ├── 02_kmeans/    # CPU / GPU / CuPy / Numba k-means
│   ├── 03_pca/       # CPU / GPU / CuPy PCA
│   ├── 04_linear_model/
│   ├── 05_kernels/   # GEMM, ReLU, softmax
│   ├── 05_naive_bayes/
│   ├── 06_interop/   # CuPy and PyTorch interop
│   ├── 07_memory/    # Alloc, pinned, OOM recovery
│   └── 08_comparison/  # 5-backend comparison table
├── benchmarks/       # run_all.py with --backend flag
└── tests/            # 72 CPU tests, 26 GPU tests
```

**Source:** `CLAUDE.md` (project structure section)
