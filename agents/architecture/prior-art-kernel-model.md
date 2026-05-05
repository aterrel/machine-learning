# Prior Art Survey: GPU Kernel Performance Modeling in Python

**Related to**: ARCH-004, REQ-0010
**Date**: 2026-05-05
**Purpose**: Survey of existing tools and libraries for analytical GPU kernel occupancy and roofline modeling, to inform the design of `src/kernel_model/`.

---

## Summary

No publicly available pure-Python library combines both occupancy modeling and roofline bound computation in a single, dependency-light package targeting CUDA GPUs analytically. `src/kernel_model/` fills a real gap. The closest prior art (Nsight Compute's `ncu_occupancy`) is not pip-installable and ships only with Nsight Compute.

---

## Projects Surveyed

### 1. NVIDIA Nsight Compute — `ncu_occupancy` Python Interface

**Location**: Ships with Nsight Compute in `extras/python/`. Not a public GitHub repo.  
**Docs**: https://docs.nvidia.com/nsight-compute/OccupancyCalculatorPythonInterface/index.html

**What it does**: The closest thing to an official NVIDIA occupancy library for Python. Exposes:
- `OccupancyCalculator` class
- `OccupancyParameters` dataclass (threads-per-block, registers-per-thread, shared memory per block, block barriers)
- `get_sm_occupancy()`, `get_occupancy_limiters()`, `get_optimal_occupancy()`
- `get_gpu_data(major, minor)` — looks up device specs by compute capability, no GPU required

Works fully offline; explicitly supports Python 3.7+. Tracks Nsight Compute releases.

**Relevance**: This is the ground-truth reference for validating our occupancy formulas. The `OccupancyLimiter` enum returning the binding constraint (registers, shared memory, blocks, or barriers) matches exactly the `limiting_resource` field in our `OccupancyResult`. The fourth limiter — **block barriers** (cooperative groups) — is present here but absent from our initial design; worth noting as a future extension.

Note: This library was flagged as missing from `cuda-python` itself in [cuda-python issue #504](https://github.com/NVIDIA/cuda-python/issues/504), confirming there is no pip-installable equivalent today.

---

### 2. jefby — Jetson Xavier Occupancy Gist

**Location**: https://gist.github.com/jefby/06d94b7802854f227aa13916b13e1e9c

**What it does**: A short pure-Python script that implements the occupancy formula from scratch — no CUDA runtime, no GPU, no dependencies. Accepts block size, registers per thread, and shared memory per block. Returns occupancy fraction and the binding limiter (warps, registers, or shared memory). Hardcoded for Jetson Xavier device specs.

**Relevance**: The cleanest available standalone reference for the occupancy formula. Confirms register allocation granularity is **256 registers per warp** on Volta+ architectures — consistent with ARCH-004. The generalization from hardcoded specs to a configurable `DeviceSpec` is the core extension our library makes.

---

### 3. ekondis/gpuroofperf-toolkit

**Repo**: https://github.com/ekondis/gpuroofperf-toolkit  
**Stars**: 19 | **License**: MIT | **Last release**: March 2019

**What it does**: A GPU performance prediction toolkit combining a C/CUDA benchmarking component (to measure device capabilities empirically) with a Python offline prediction tool (`gpuroofperf-tool`) that classifies kernels as compute-bound or memory-bound using instruction mix analysis. The Python component runs offline if you supply device specs via a CSV.

**Relevance**: Most architecturally similar to `src/kernel_model/`. The separation between "benchmark to populate device params" and "offline predict from params" is the same design pattern as `DeviceSpec.from_device()` vs `DeviceSpec.from_name()`. Unmaintained since 2019 but the design is sound. MIT license means code is reusable directly.

---

### 4. Giotyp/GPU-Roofline-Python

**Repo**: https://github.com/Giotyp/GPU-Roofline-Python  
**Stars**: 15

**What it does**: A hierarchical roofline visualization module based on Ding & Williams (2019). Plots roofline ceilings for L1/L2/HBM memory tiers at warp-level instruction granularity for V100S and A100. Ingests profiling data from `nvprof` or `ncu` output files and produces roofline charts offline.

**Relevance**: Implements the hierarchical roofline variant with multiple memory tiers rather than the single-DRAM roofline in our initial design. Useful reference if we later extend `RooflineModel` to support L1/L2/HBM tiered bandwidth. Uses instruction intensity (instructions / memory transactions) rather than classic FLOP/byte — a more accurate measure for GPU workloads.

---

### 5. giopaglia/rooflini

**Repo**: https://github.com/giopaglia/rooflini  
**Stars**: 17 | **License**: GPL-3.0 | **Last commit**: October 2024

**What it does**: A minimal pure-Python matplotlib script for drawing Intel Advisor-style roofline plots. Takes manually specified peak FLOP rate and bandwidth plus data points (arithmetic intensity, achieved performance) and renders the roofline chart. No GPU, no profiler.

**Relevance**: Clearest minimal implementation of the roofline formula and the two-line intersection: `performance = min(peak_flops, arithmetic_intensity * peak_bandwidth)`. Actively maintained. GPL-3.0 — use as reference only, do not copy code.

---

### 6. RRZE-HPC/kerncraft

**Repo**: https://github.com/RRZE-HPC/kerncraft  
**Stars**: 96 | **License**: AGPLv3 | **Status**: Actively developed (1,290+ commits)

**What it does**: A loop kernel analysis and performance modeling toolkit applying the Roofline model (and the more advanced ECM model) through static code analysis. Pure Python, targets CPU architectures, integrates with `likwid` for hardware counter benchmarking.

**Relevance**: The most mature pure-Python performance modeling library in this space. CPU-only — no CUDA support — but the architecture is directly adaptable: machine model files describe peak FLOP rate and memory bandwidth tiers, exactly analogous to `DeviceSpec`. The cleanest code structure reference. AGPLv3 — use as design reference only.

---

### 7. maestro-project/FRAME

**Repo**: https://github.com/maestro-project/frame  
**Stars**: 39 | **Status**: Active

**What it does**: Fast Roofline Analytical Modeling and Estimation for DNN accelerators. Pure-Python Jupyter-based tool computing layer-wise latency and memory usage for CNN/MLP/Transformer workloads given a configurable PE array, on-chip/off-chip bandwidth, and compute capacity.

**Relevance**: The roofline bound formula (`latency = max(compute_cycles, memory_cycles)`) is cleanly structured and generalizable. Targets DNN accelerators, not general GPU kernels, but the core math is the same. Good reference for multi-tier bandwidth handling (SRAM vs DRAM).

---

### 8. ProjectPhysX/PTXprofiler

**Repo**: https://github.com/ProjectPhysX/PTXprofiler  
**Stars**: 57 | **Last commit**: January 2023

**What it does**: A C++ tool that statically analyzes PTX assembly to count instruction categories (FLOPs, integer ops, memory ops) per kernel — enabling construction of arithmetic intensity inputs for a roofline plot without runtime profiling.

**Relevance**: The only tool that derives arithmetic intensity directly from PTX static analysis rather than hardware counters. C++ only, not a Python library. Useful conceptual reference if `src/kernel_model/` is ever extended to accept PTX source and compute theoretical instruction counts analytically.

---

## Comparison Table

| Project | Stars | Pure Python | No GPU | Occupancy | Roofline | CUDA-specific | Maintained |
|---------|-------|------------|--------|-----------|----------|--------------|------------|
| Nsight Compute `ncu_occupancy` | — | Yes | Yes | Full API | No | Yes | Yes |
| jefby gist | — | Yes | Yes | Formula | No | Yes | No |
| ekondis/gpuroofperf-toolkit | 19 | Partial | Partial | No | Yes | Yes | No (2019) |
| Giotyp/GPU-Roofline-Python | 15 | Partial | Partial | No | Yes (hierarchical) | Yes | Unknown |
| giopaglia/rooflini | 17 | Yes | Yes | No | Yes (plot) | No | Yes (2024) |
| RRZE-HPC/kerncraft | 96 | Yes | Yes | No | Yes | No (CPU) | Yes |
| maestro-project/FRAME | 39 | Yes | Yes | No | Yes | No (DNN accel) | Yes |
| ProjectPhysX/PTXprofiler | 57 | No (C++) | Partial | No | Partial | Yes | No (2023) |
| **src/kernel_model/** | — | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | — |

---

## Key Takeaways for Implementation

**On occupancy:**
- No pip-installable pure-Python CUDA occupancy library exists. The formulas are well-understood but not packaged standalone.
- Register allocation granularity is **256 registers per warp** on Volta+ — confirmed by both the jefby gist and NVIDIA documentation. This is already accounted for in ARCH-004.
- The Nsight Compute model includes a fourth limiter (block barriers for cooperative groups). Our V1 omits this; add as a `block_barriers: int = 0` parameter in a future release.

**On roofline:**
- The core formula is simple: `min(peak_flops, arithmetic_intensity * peak_bandwidth)`.
- The main gap in existing tools is the absence of a maintainable GPU SKU spec table. Most tools either require a live GPU measurement or hardcode a single architecture.
- Hierarchical roofline (L1/L2/HBM tiers) adds accuracy for memory-bound kernels with significant cache reuse — worthwhile future extension once the single-tier model is validated.

**On the gap:**
- No existing library combines occupancy + roofline + a maintained GPU spec table in a single dependency-light Python package. `src/kernel_model/` fills this gap.
- The `ekondis/gpuroofperf-toolkit` (MIT) and jefby gist are the most directly reusable references for the implementation.

---

## References

- [Nsight Compute Occupancy Calculator Python Interface](https://docs.nvidia.com/nsight-compute/OccupancyCalculatorPythonInterface/index.html)
- [cuda-python issue #504 — Occupancy Calculator APIs](https://github.com/NVIDIA/cuda-python/issues/504)
- [ekondis/gpuroofperf-toolkit](https://github.com/ekondis/gpuroofperf-toolkit)
- [Giotyp/GPU-Roofline-Python](https://github.com/Giotyp/GPU-Roofline-Python)
- [giopaglia/rooflini](https://github.com/giopaglia/rooflini)
- [RRZE-HPC/kerncraft](https://github.com/RRZE-HPC/kerncraft)
- [maestro-project/FRAME](https://github.com/maestro-project/frame)
- [ProjectPhysX/PTXprofiler](https://github.com/ProjectPhysX/PTXprofiler)
- [jefby — GPU occupancy for Jetson Xavier](https://gist.github.com/jefby/06d94b7802854f227aa13916b13e1e9c)
- Williams, Waterman, Patterson (2009) — "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
- Ding & Williams (2019) — "Hierarchical Roofline Analysis for GPUs"
