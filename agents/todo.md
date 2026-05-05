# agents/todo.md — CUDA Python ML Demos Task Tracker

## Status: SPRINT 7 TL REVIEW COMPLETE — Conditional Approval — Minor issues tracked (2026-05-05)

Sprint 7 goal: `src/kernel_model/` — pure-Python occupancy + roofline library. Programmer deliverables complete.
REQ: REQ-0010 | ARCH: ARCH-004 (Conditional Approval) | TL verdict: GO

---

## Sprint 7 — Kernel Performance Model (REQ-0010)

### P0 — Programmer

- [x] [Prog] Create `src/kernel_model/__init__.py` — re-exports DeviceSpec, OccupancyModel, OccupancyResult, RooflineModel, RooflineResult
- [x] [Prog] Create `src/kernel_model/device_spec.py` — DeviceSpec dataclass (incl. `sm_version: str`), GPU SKU table (V100/A100-40/A100-80/H100/B100/RTX3090/RTX4090/RTX5090), `from_name()`, `from_device()`
- [x] [Prog] Create `src/kernel_model/occupancy.py` — OccupancyResult dataclass + OccupancyModel with `compute()` + `sweep_block_sizes()`
- [x] [Prog] Create `src/kernel_model/roofline.py` — RooflineResult dataclass + RooflineModel with `compute()` + `sweep_intensities()`
- [x] [Prog] Create `demos/09_kernel_model/__init__.py` — empty package init
- [x] [Prog] Create `demos/09_kernel_model/main.py` — CLI demo: occupancy table + roofline summary for vector-add kernel on A100
- [x] [Prog] Create `tests/test_kernel_model.py` — 11 CPU-safe unit tests + 1 `@pytest.mark.gpu` test

### P1 — Tech Lead Review

- [x] [TL] Sprint 7 code review — verify occupancy math, roofline math, SKU table accuracy, test coverage — Conditional Approval (2026-05-05)

---

## Sprint 6 — Slide-Based Demo Documentation (REQ-0009)

### P0 — Programmer

- [x] [Prog] Create `docs/slides/README.md` index
- [x] [Prog] Create `docs/slides/00_introduction/` (2 slides: what_is_cuda_python, project_overview)
- [x] [Prog] Create `docs/slides/01_core_apis/` (6 slides: device_setup, memory_allocation, kernel_compilation, kernel_launch, pinned_memory, backend_comparison)
- [x] [Prog] Create `docs/slides/02_kmeans/` (5 slides: algorithm, cpu_baseline, gpu_kernel, cupy_variant, numba_variant)
- [x] [Prog] Create `docs/slides/03_pca/` (4 slides: algorithm, gpu_covariance, cupy_linalg, backend_comparison)
- [x] [Prog] Create `docs/slides/04_linear_model/` (3 slides: normal_equations, gpu_xtx_kernel, backend_comparison)
- [x] [Prog] Create `docs/slides/05_kernels/` (4 slides: gemm_overview, gemm_kernel, relu_softmax, backend_comparison)
- [x] [Prog] Create `docs/slides/05_naive_bayes/` (4 slides: algorithm, gpu_log_likelihood, cupy_broadcasting, backend_comparison)
- [x] [Prog] Create `docs/slides/06_interop/` (4 slides: cuda_array_interface, cupy_interop, pytorch_interop, pipeline)
- [x] [Prog] Create `docs/slides/07_memory/` (3 slides: device_allocation, pinned_memory, oom_recovery)
- [x] [Prog] Create `docs/slides/08_comparison/` (2 slides: abstraction_levels, reading_the_table)

### P1 — Tech Lead Review

- [x] [TL] Sprint 6 code review — verify slide content accuracy and code snippet correctness

---

## Remaining Backlog (not blocking)

- [ ] [Prog] Generate Jupyter notebook versions of demos 01 and 02
- [ ] README.md user-facing documentation
- [ ] CI/CD pipeline (GitHub Actions)

---

## Completed

### Sprint 0 / Bootstrap
- [x] [Claude-Mgr] Full project initialization (PROJECT.md, CLAUDE.md, REQ-0001–0006, ARCH-001–002)

### Sprint 1 — Core GPU Loop + K-Means Demo (CLOSED — Conditional Approval)
- [x] src/utils/ (device, memory, timing), src/kernels/ (compiler, compiled_kernel)
- [x] demos/01_core_apis/, demos/02_kmeans/, benchmarks/run_all.py
- [x] tests: conftest, test_device, test_kernels, test_kmeans (11 CPU tests)

### Sprint 2 — Remaining Algorithm Demos (CLOSED — Conditional Approval)
- [x] demos/03_pca/, demos/04_linear_model/, demos/05_kernels/
- [x] tests: test_memory, test_pca, test_linear (22 CPU tests)

### Sprint 3 — Interop, Memory (CLOSED — Conditional Approval)
- [x] demos/06_interop/ (CuPy + PyTorch interop), demos/07_memory/
- [x] tests: test_interop (32 CPU tests)

### Sprint 4 — Polish (CLOSED — Approved)
- [x] demos/05_naive_bayes/ — GPU Gaussian Naive Bayes
- [x] tests: test_naive_bayes (39 CPU tests)
- [x] All TL nit fixes applied; benchmark runner complete (all 8 demos)
- [x] Tech Lead final review — Approved

### Sprint 5 — Multi-Backend Comparison (CLOSED — Conditional Approval)
- [x] [Arch] ARCH-003 multi-backend design pattern
- [x] [Prog] demos/01_core_apis/numba_vector_add.py, cupy_vector_add.py
- [x] [Prog] demos/02_kmeans/numba_kmeans.py, cupy_kmeans.py
- [x] [Prog] demos/03_pca/numba_pca.py, cupy_pca.py
- [x] [Prog] demos/04_linear_model/numba_linear.py, cupy_linear.py
- [x] [Prog] demos/05_naive_bayes/numba_nb.py, cupy_nb.py
- [x] [Prog] demos/05_kernels/numba_kernels.py, cupy_kernels.py (P1)
- [x] [Prog] benchmarks/run_all.py --backend flag + 3-way comparison table
- [x] [Prog] demos/08_comparison/main.py + backends/cuml_backends.py (REQ-0008)
- [x] [QA]  tests/test_numba_variants.py, test_cupy_variants.py, test_comparison.py (33 CPU tests)
- [x] [TL]  TL-S5-001/003 nit fixes applied; Tech Lead Conditional Approval
