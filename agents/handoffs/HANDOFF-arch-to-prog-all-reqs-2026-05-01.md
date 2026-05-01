# HANDOFF: Software Architect → Programmer
# Topic: All REQs — Sprint 1 Implementation
# Date: 2026-05-01
# Status: APPROVED — Programmer is unblocked (2026-05-01)

---

## What Was Completed

- ARCH-001: Overall System Architecture — module structure, key classes, data flow, design decisions
- ARCH-002: CUDA Kernel Compilation and Launch Pipeline — KernelCompiler, CompiledKernel, NVRTC usage patterns
- REQ-0001 through REQ-0006 — requirements documents for all 6 project objectives
- Project directory structure scaffolded (`src/`, `demos/`, `benchmarks/`, `tests/`, `notebooks/`)

---

## What You Need to Do

### Task 1: Scaffold `src/utils/`

Create the three utility modules that everything else depends on. Read ARCH-001 for interface signatures.

- `src/utils/device.py` — `check_cuda_available()`, `get_device(index)`, `get_device_props(device)`
- `src/utils/memory.py` — `DeviceBuffer` context manager with `__cuda_array_interface__`, `alloc_pinned()`, `query_device_memory()`
- `src/utils/timing.py` — `BenchmarkResult` dataclass, `BenchmarkRunner` (CUDA event timing, N repeats, warmup run)

### Task 2: Scaffold `src/kernels/`

- `src/kernels/compiler.py` — `KernelCompiler` wrapping `cuda.core.experimental.Program`
- `src/kernels/compiled_kernel.py` — `CompiledKernel.launch()` with `LaunchConfig`, `compute_grid_1d()` helper

Read ARCH-002 carefully before implementing — it has the argument-passing gotchas and the SM arch flag pattern.

### Task 3: Implement `demos/01_core_apis/` (REQ-0001, Sprint 1 P0)

This is the integration smoke test. If this runs end-to-end, the entire src/ layer works.

- `device_info.py` — print device name, compute capability, total VRAM
- `vector_add.py` — compile vector-add kernel, launch, verify against NumPy, return BenchmarkResult
- `main.py` — call both, print results

### Task 4: Implement `demos/02_kmeans/` (REQ-0002, Sprint 1 P0)

GPU k-means with Lloyd's algorithm. The distance computation (n_samples × n_centroids) is the key parallel step.

- `cpu_kmeans.py` — NumPy reference implementation
- `gpu_kmeans.py` — GPU implementation using custom CUDA kernels (distance kernel + argmin + centroid update)
- `main.py` — run both, validate centroids within 1e-3, print BenchmarkResult

### Task 5: Create `pyproject.toml`

Set up ruff (linting + formatting) and pytest (with GPU mark). See ARCH-001 testing strategy section.

---

## Relevant Files

| File | Status | Notes |
|------|--------|-------|
| `agents/architecture/ARCH-001.md` | Draft | Read before implementing — full module structure and class signatures |
| `agents/architecture/ARCH-002.md` | Draft | Read before implementing `src/kernels/` — critical for NVRTC usage |
| `agents/requirements/REQ-0001.md` | Draft | Core API demo acceptance criteria |
| `agents/requirements/REQ-0002.md` | Draft | ML algorithm demo acceptance criteria |
| `CLAUDE.md` | Created | Development commands, code quality standards |
| `src/` | Scaffolded (empty) | Your implementation target |
| `demos/` | Scaffolded (empty) | Your implementation target |
| `tests/` | Scaffolded (empty) | Implement conftest.py alongside src/ |

---

## Open Questions / Decisions Needed

- [ ] Which cuda-python minimum version? Confirm `cuda.core.experimental` is available (requires >= 12.3)
- [ ] For k-means: use cuBLAS for distance GEMM or pure custom kernel? Custom kernel preferred for educational value
- [ ] For k-means: handle k-means++ initialization or start with random centroids? Random is fine for v1

---

## Acceptance Criteria (from REQ documents)

The Sprint 1 work is complete when:

- [ ] AC-REQ0001-1: `python demos/01_core_apis/main.py` prints device name, compute capability, VRAM
- [ ] AC-REQ0001-3: Vector-add kernel output matches NumPy within 1e-5
- [ ] AC-REQ0001-4: Running without GPU prints clear error and exits with code 1
- [ ] AC-REQ0002-1: `python demos/02_kmeans/main.py` shows cluster assignments, inertia, GPU vs CPU timing
- [ ] AC-REQ0002-2: K-means GPU centroids match CPU within 1e-3 tolerance on same seed
- [ ] AC-REQ0002-7: Both demos print: `GPU: X.XXs | CPU: X.XXs | Speedup: X.Xx | Correct: True`

---

## Context / Notes

**Critical CUDA Python gotchas** (from ARCH-002):

1. All kernel functions must use `extern "C"` — NVRTC C++ mangles names without it
2. `device.set_current()` must be called before any memory allocation on that device
3. Integer scalar args to kernels need `ctypes.c_int(n)`, not plain Python `int`
4. Device pointers pass as raw `int` (use `.ptr` attribute from DeviceBuffer)
5. Always call `stream.sync()` before reading D2H results — kernel is async

**For k-means kernel design**:
- Distance kernel: each thread computes distance from one sample to one centroid → output is (n_samples × k) matrix
- Argmin kernel: each thread finds the nearest centroid for one sample
- Centroid update: atomic adds or reduction across samples assigned to each centroid
- Convergence check can be done on CPU after each iteration (copy centroid delta back)

---

*From: Software Architect*
*To: Programmer*
*Next step: Start with `src/utils/device.py` and `src/utils/memory.py`. Then implement `demos/01_core_apis/` as an end-to-end smoke test before touching k-means.*
