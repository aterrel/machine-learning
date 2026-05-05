# agents/todo.md — CUDA Python ML Demos Task Tracker

## Status: SPRINT 10 OPEN — README + Physical GPU Validation (2026-05-05)

Sprint 10 goal: `README.md` covering all 10 demos with CI badge + Physical GPU validation of all 26 GPU tests.
REQ: REQ-0014 | Depends on: Sprint 9 CLOSED

---

## Sprint 10 — README + Physical GPU Validation (REQ-0014)

### P0 — Programmer

- [ ] [Prog] Create `README.md` — top-level docs: quickstart, demo table (01–10), prerequisites, CI badge (REQ-0014-F1–F8)
- [ ] [Prog] Fix m-2: add `cupy` and `torch` to `gpu-ci.yml` pip install step (Sprint 9 TL finding)
- [ ] [Prog] Fix m-1: add CI badge to README (REQ-0012-F5) — can be done as part of README task above

### P1 — GPU Validation (requires NVIDIA hardware)

- [ ] [Prog] Run `pytest tests/ -v` on physical NVIDIA GPU; document pass/fail for all 26 GPU tests
- [ ] [Prog] Fix any correctness issues found during GPU validation

### P1 — Tech Lead Review

- [ ] [TL] Sprint 10 code review

---

## Remaining Backlog (not blocking)

- [ ] Physical GPU validation — all 26 GPU tests (Sprint 10 P1, requires NVIDIA hardware)
- [ ] Jupyter notebooks for demos 03–08 (post-Sprint 10, P2)
- [ ] FP16/BF16/TF32 performance modeling extensions (P2)
- [ ] cuobjdump PTX extraction in PTXTracer (P2)
- [ ] Code coverage reporting in CI (P2)

---

## Completed

### Sprint 9 — CI/CD + Jupyter Notebooks (CLOSED — Conditional Approval)
- [x] [Prog] Create `.github/workflows/ci.yml` — ruff lint + CPU-safe pytest on push/PR
- [x] [Prog] Create `.github/workflows/gpu-ci.yml` — full GPU test suite, manual trigger, self-hosted runner
- [x] [Prog] Create `notebooks/01_core_apis.ipynb` — interactive CUDA Python API walkthrough
- [x] [Prog] Create `notebooks/02_kmeans.ipynb` — interactive GPU k-means notebook
- [x] [TL] Sprint 9 code review — Conditional Approval (2026-05-05) — 0 critical, 2 major (accepted deviations), 4 minor findings

### Sprint 8 — PTX Kernel Execution Tracer (CLOSED — Approved)
- [x] [Prog] Create `src/kernel_model/_taxonomy.py` — `_INSTRUCTION_TAXONOMY` dict: PTX mnemonic prefix → InstructionRecord
- [x] [Prog] Create `src/kernel_model/_arch_table.py` — `ArchSpec` dataclass + `_ARCH_TABLE` for sm_70/80/86/89/90/100 + `_MMA_LATENCY` values from ptx-tracer-research.md
- [x] [Prog] Create `src/kernel_model/ptx_tracer.py` — PTXTracer, TracerResult, InstructionRecord with trace(), trace_file(), bottleneck()
- [x] [Prog] Update `src/kernel_model/__init__.py` — add PTXTracer, TracerResult to re-exports
- [x] [Prog] Create `demos/10_ptx_tracer/__init__.py` — empty package init
- [x] [Prog] Create `demos/10_ptx_tracer/ptx_fixtures/vector_add.ptx` — handwritten minimal PTX for vector-add
- [x] [Prog] Create `demos/10_ptx_tracer/ptx_fixtures/gemm_mma.ptx` — handwritten minimal PTX with mma.sync instructions
- [x] [Prog] Create `demos/10_ptx_tracer/main.py` — CLI demo: trace both fixtures vs A100 and H100 side-by-side
- [x] [Prog] Create `tests/test_ptx_tracer.py` — 13 CPU-safe test cases per ARCH-005
- [x] [TL] Sprint 8 code review — Approved (2026-05-05) — 0 critical, 0 major, 4 minor findings

### Sprint 7 — Kernel Performance Model (CLOSED — Conditional Approval)
- [x] [Prog] Create `src/kernel_model/__init__.py` — re-exports DeviceSpec, OccupancyModel, OccupancyResult, RooflineModel, RooflineResult
- [x] [Prog] Create `src/kernel_model/device_spec.py` — DeviceSpec dataclass (incl. `sm_version: str`), GPU SKU table (V100/A100-40/A100-80/H100/B100/RTX3090/RTX4090/RTX5090), `from_name()`, `from_device()`
- [x] [Prog] Create `src/kernel_model/occupancy.py` — OccupancyResult dataclass + OccupancyModel with `compute()` + `sweep_block_sizes()`
- [x] [Prog] Create `src/kernel_model/roofline.py` — RooflineResult dataclass + RooflineModel with `compute()` + `sweep_intensities()`
- [x] [Prog] Create `demos/09_kernel_model/__init__.py` — empty package init
- [x] [Prog] Create `demos/09_kernel_model/main.py` — CLI demo: occupancy table + roofline summary for vector-add kernel on A100; M-1 (dead `mem_bw_util`) fixed post-review
- [x] [Prog] Create `tests/test_kernel_model.py` — 11 CPU-safe unit tests + 1 `@pytest.mark.gpu` test
- [x] [TL] Sprint 7 code review — Conditional Approval (2026-05-05) — 0 critical, 0 major, 7 minor findings; M-1 fixed

### Sprint 6 — Slide-Based Demo Documentation (CLOSED — Approved)
- [x] [Prog] docs/slides/ — 37 slides across 10 directories (REQ-0009)
- [x] [TL] Sprint 6 code review — Approved

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
