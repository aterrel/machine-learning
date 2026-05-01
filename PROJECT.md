# PROJECT.md — Fill This In Before Running /bootstrap

This is the **only file you need to fill in** to initialize a multi-agent project.
Once complete, run `/bootstrap` in Claude Code and the system will generate everything else.

---

## Project Name

CUDA Python ML Demos

---

## Vision

A collection of hands-on demonstrations showing how to use [CUDA Python](https://github.com/nvidia/cuda-python) to accelerate common machine learning algorithms directly on NVIDIA GPUs. Targeted at ML practitioners and researchers who want low-level GPU control without leaving Python, bridging the gap between high-level frameworks and raw CUDA programming.

---

## High-Level Objectives

1. Demonstrate core CUDA Python APIs (cuda.core, cuda.bindings) with working ML examples
2. Implement and benchmark GPU-accelerated versions of foundational ML algorithms (e.g., k-means, PCA, linear regression, naive Bayes)
3. Show how to write and launch custom CUDA kernels from Python for ML operations (matrix multiply, softmax, ReLU, etc.)
4. Provide side-by-side comparisons of NumPy/CPU vs CUDA Python/GPU implementations with timing and speedup metrics
5. Include end-to-end examples that integrate CUDA Python with popular ML libraries (NumPy, PyTorch, CuPy)
6. Document memory management patterns (device allocation, transfers, streams) relevant to ML workloads

---

## Tech Stack

- **Backend language/framework**: Python 3.11+
- **Frontend**: CLI only / Jupyter notebooks for interactive demos
- **Database**: none
- **Background jobs**: none
- **Deployment target**: local only (requires NVIDIA GPU)
- **Special infrastructure**: CUDA Python (cuda-python), CUDA Toolkit 12.x, NumPy, CuPy (optional interop), PyTorch (optional interop)

---

## Constraints

- [x] Requires an NVIDIA GPU with CUDA 12.x support
- [x] All core demos must use cuda-python directly — not just PyTorch/CuPy wrappers
- [x] Each demo must be self-contained and runnable independently
- [x] Must work on a single machine (no distributed/multi-node setup)
- [x] Avoid heavy ML framework dependencies in core demos — keep them close to the metal

---

## Success Metrics

- Each demo runs end-to-end without errors on a CUDA 12.x GPU
- GPU implementations achieve measurable speedup over CPU baselines (documented per demo)
- A new user familiar with Python and basic ML can understand and run any demo in under 15 minutes
- All CUDA Python API usage reflects current cuda-python best practices (cuda.core / cuda.bindings)

---

## Out of Scope

- Training large neural networks (focus is algorithms, not DNN frameworks)
- Multi-GPU or distributed training
- Windows support (Linux only for now)
- Replacing or competing with CuPy, PyTorch, or JAX — this is educational, not a library

---

## Additional Context

- CUDA Python repo: https://github.com/nvidia/cuda-python
- cuda-python exposes two layers: `cuda.bindings` (raw CUDA C API bindings) and `cuda.core` (higher-level Pythonic wrappers) — demos should use `cuda.core` where practical and drop to `cuda.bindings` when needed for low-level control
- Target audience: ML engineers comfortable with Python and NumPy who want to learn GPU programming without C++
- Reference for algorithm selection: classic ML algorithms that have well-understood GPU parallelization patterns (e.g., embarrassingly parallel distance computations in k-means, matrix ops in linear models)
