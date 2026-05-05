# CUDA Python ML Demos

[![CI](https://github.com/aterrel/agentic-project-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/aterrel/agentic-project-starter/actions/workflows/ci.yml)

Hands-on demonstrations showing how to use [CUDA Python](https://github.com/nvidia/cuda-python) to accelerate common machine learning algorithms directly on NVIDIA GPUs.

---

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA 12.x drivers (for GPU demos)
- [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) installed system-wide
- `cuda-python` — NVIDIA's official Python bindings
- `numpy` — CPU baseline reference implementations

> **CPU-only mode**: Demos `09_kernel_model` and `10_ptx_tracer` run without a GPU and are safe to use on any machine.

---

## Installation

```bash
pip install cuda-python numpy pytest ruff jupyter
# Optional (required for interop demos 06, 08):
pip install cupy-cuda12x torch
```

---

## Quickstart

### CPU-Only (no GPU required)

Run the kernel occupancy + roofline model demo — no NVIDIA hardware needed:

```bash
python -m demos.09_kernel_model.main
```

Run the PTX instruction tracer — also CPU-only:

```bash
python -m demos.10_ptx_tracer.main
```

### Full GPU Path

```bash
# Run a single demo
python -m demos.01_core_apis.main

# Run all demos
python benchmarks/run_all.py
```

---

## Demo Overview

| # | Title | Description | Needs GPU |
|---|-------|-------------|-----------|
| 01 | Core CUDA Python APIs | Device management, kernel compile/launch, memory | Yes |
| 02 | GPU K-Means Clustering | Lloyd's algorithm on the GPU | Yes |
| 03 | GPU PCA | PCA via covariance computation | Yes |
| 04 | GPU Linear Regression | Normal equation on the GPU | Yes |
| 05a | Custom CUDA Kernels | Naive GEMM, ReLU, Softmax kernels | Yes |
| 05b | GPU Gaussian Naive Bayes | Gaussian Naive Bayes classifier | Yes |
| 06 | NumPy / CuPy / PyTorch Interop | Interoperability with popular array frameworks | Yes |
| 07 | GPU Memory Management | Pinned, pooled, and unified memory patterns | Yes |
| 08 | Multi-Backend Comparison | Unified comparison: cuda-python vs Numba vs CuPy | Yes |
| 09 | Kernel Occupancy + Roofline Model | CPU-only occupancy analysis and roofline modeling | **No** |
| 10 | PTX Instruction Tracer | PTX tracer for Ampere/Ada/Hopper/Blackwell | **No** |

---

## Project Structure

```
cuda-python-ml-demos/
├── src/                    # Reusable CUDA kernel library
│   ├── kernels/            # .cu source strings and kernel wrappers
│   └── utils/              # Device management, timing, memory utilities
├── demos/                  # Self-contained ML algorithm demos
│   ├── 01_core_apis/
│   ├── 02_kmeans/
│   ├── 03_pca/
│   ├── 04_linear_model/
│   ├── 05_kernels/
│   ├── 05_naive_bayes/
│   ├── 06_interop/
│   ├── 07_memory/
│   ├── 08_comparison/
│   ├── 09_kernel_model/
│   └── 10_ptx_tracer/
├── notebooks/              # Jupyter versions of select demos
├── docs/slides/            # Slide decks (37 slides across 10 demo directories)
├── benchmarks/             # CPU vs GPU timing comparisons
└── tests/                  # Unit and integration tests
```

---

## Jupyter Notebooks

Interactive notebook versions of select demos are available in [`notebooks/`](notebooks/):

- [`notebooks/01_core_apis.ipynb`](notebooks/01_core_apis.ipynb) — Core CUDA Python API walkthrough
- [`notebooks/02_kmeans.ipynb`](notebooks/02_kmeans.ipynb) — GPU k-means clustering

Launch with:

```bash
jupyter notebook notebooks/
```

---

## Slides

Slide decks for all 10 demos (37 slides total) are in [`docs/slides/`](docs/slides/).

---

## Running Tests

```bash
# All tests (requires NVIDIA GPU for GPU tests)
pytest tests/ -v

# CPU-safe tests only (no GPU required — 96 tests)
pytest tests/ -v -m "not gpu"

# GPU tests only (26 tests, requires NVIDIA GPU + CUDA 12.x)
pytest tests/ -v -m "gpu"
```

---

## Development

```bash
# Lint
ruff check src/ demos/ tests/

# Format
ruff format src/ demos/ tests/
```

---

## Key Concepts

- **`cuda.core` vs `cuda.bindings`**: Prefer `cuda.core` (high-level) for new code; use `cuda.bindings` only when direct C API access is necessary.
- **Memory discipline**: Always free device allocations — unfreed GPU memory persists across Python objects.
- **Correctness first**: Every GPU implementation is validated against its NumPy baseline before reporting speedups.
