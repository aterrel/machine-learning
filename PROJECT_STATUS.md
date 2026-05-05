# PROJECT_STATUS.md — CUDA Python ML Demos

**Last Updated**: 2026-05-05
**Current Sprint**: Sprint 8 — OPEN (PTX Kernel Execution Tracer)
**All Sprints**: 1–7 CLOSED

---

## Project Health

| Area | Status | Notes |
|------|--------|-------|
| Requirements | 🟡 Yellow | REQ-0001–0010 implemented; REQ-0011 open (Sprint 8); REQ-0012–0014 planned (Sprints 9–10) |
| Architecture | 🟡 Yellow | ARCH-001–004 Approved/Conditional Approval (Sprint 7 done); ARCH-005 Conditional Approval (Sprint 8 active); ARCH-006/007 Draft |
| Implementation | 🟢 Green | 8 core demos + 12 backend variants + comparison demo delivered |
| Tests | 🟢 Green | 72 CPU tests pass; 26 GPU tests ready for hardware |
| Documentation | 🟢 Green | 37 slides in docs/slides/ + README index (REQ-0009 complete) |
| CI/Build | 🔴 Red | pyproject.toml in place; no CI pipeline (Sprint 9) |

---

## Architecture Document Status

| Doc | Title | REQ | Status | Verdict |
|-----|-------|-----|--------|---------|
| ARCH-001 | Core src/ Library | REQ-0001–0004 | Approved | — |
| ARCH-002 | Interop Layer | REQ-0005 | Approved | — |
| ARCH-003 | Memory Management | REQ-0006 | Approved | — |
| ARCH-004 | Kernel Performance Model | REQ-0010 | **Conditional Approval** | Add `sm_version` field to DeviceSpec; align `from_device()` exception to `ValueError` |
| ARCH-005 | PTX Kernel Execution Tracer | REQ-0011 | **Conditional Approval** | Define `_MMA_LATENCY` dict values from ptx-tracer-research.md; fill sm_86 ArchSpec placeholders |
| ARCH-006 | CI/CD Pipeline (GitHub Actions) | REQ-0012 | Draft | — |
| ARCH-007 | Jupyter Notebook Coverage | REQ-0013 | Draft | — |

---

## Sprint 7 — CLOSED: Kernel Performance Model

**Goal**: Implement `src/kernel_model/` — a pure-Python library for GPU kernel occupancy and roofline analysis, plus `demos/09_kernel_model/`. Fully functional without a GPU.

**ARCH ref**: ARCH-004 (Conditional Approval)
**REQ ref**: REQ-0010
**Tech Lead verdict**: Conditional Approval — 0 critical, 0 major, 7 minor findings. M-1 (dead `mem_bw_util` variable) fixed before close. M-2 through M-7 tracked as follow-on cleanup (P2, non-blocking).

**TL Findings Summary**:
- M-1 (CLOSED): Dead code `mem_bw_util` in `demos/09_kernel_model/main.py` — removed.
- M-2 (open): Misleading "Memory bandwidth use" label for compute-bound case.
- M-3 (open): RTX 5090 `sm_version` may be sm_120, not sm_100 — needs datasheet verification.
- M-4 (open): Block size validation uses SM capacity (2048) rather than CUDA block limit (1024).
- M-5 (open): `noqa: F401` comment on deferred `Device` import could be clearer.
- M-6 (open): Unnecessary `* 1.0` in `sweep_intensities`.
- M-7 (open): Ruff not installed in venv; run `ruff check` before Sprint 8.

### Deliverables (in dependency order)

| # | File | Description | REQ | Status |
|---|------|-------------|-----|--------|
| 1 | `src/kernel_model/__init__.py` | Package scaffold; re-exports DeviceSpec, OccupancyModel, RooflineModel | REQ-0010 | Done |
| 2 | `src/kernel_model/device_spec.py` | DeviceSpec dataclass + `sm_version` field + GPU SKU table (V100/A100/H100/B100/RTX3090/4090/5090) + `from_name()` + `from_device()` | REQ-0010-F5–F8 | Done |
| 3 | `src/kernel_model/occupancy.py` | OccupancyResult dataclass + OccupancyModel with `compute()` + `sweep_block_sizes()` | REQ-0010-F1/F2/F9 | Done |
| 4 | `src/kernel_model/roofline.py` | RooflineResult dataclass + RooflineModel with `compute()` + `sweep_intensities()` | REQ-0010-F3/F4/F10 | Done |
| 5 | `demos/09_kernel_model/__init__.py` | Empty package init | REQ-0010 | Done |
| 6 | `demos/09_kernel_model/main.py` | CLI demo: occupancy table + roofline summary for vector-add kernel on A100 | REQ-0010-F11/F12 | Done |
| 7 | `tests/test_kernel_model.py` | 11 CPU-safe unit tests + 1 `@pytest.mark.gpu` test | REQ-0010 | Done |

### GPU SKUs to support

V100 (16GB), A100 (40GB), A100 (80GB), H100 (80GB SXM), B100 (80GB SXM), RTX 3090, RTX 4090, RTX 5090.

### Definition of Done

- [x] `from src.kernel_model import OccupancyModel, RooflineModel, DeviceSpec` succeeds on a machine with no GPU
- [x] `DeviceSpec.from_name("A100")` returns a spec with `sm_version == "sm_80"`
- [x] `DeviceSpec.from_name("RTX 9999")` raises `KeyError` listing valid names
- [x] `OccupancyModel.compute(block_size=256, shared_mem_per_block=0, registers_per_thread=32)` returns occupancy between 0 and 1
- [x] `RooflineModel.compute(flops=1e9, bytes_accessed=1e8)` returns `bound` of `"compute"` or `"memory"`
- [x] `python demos/09_kernel_model/main.py` prints occupancy table + roofline summary without a GPU
- [x] All 11 CPU tests pass; 1 GPU test marked appropriately
- [x] Tech Lead sprint review: Approved or Conditional Approval

---

## Sprint 8 — OPEN (PTX Kernel Execution Tracer)

**Goal**: Implement `src/kernel_model/ptx_tracer.py` — a pure-Python PTX instruction scanner classifying instruction mix by category (smem, tmem, mma_warp, mma_warpgroup, mma_cta, async_copy, async_tma, fused_fp, global_mem) with architecture-specific annotations for Ampere, Ada, Hopper, and Blackwell.

**ARCH ref**: ARCH-005 (Conditional Approval)
**REQ ref**: REQ-0011
**Depends on**: Sprint 7 CLOSED — DeviceSpec (including `sm_version`) delivered
**Tech Lead verdict**: GO — `_MMA_LATENCY` values must be sourced from `agents/architecture/ptx-tracer-research.md`; sm_86 ArchSpec placeholders must be filled

### Deliverables (in dependency order)

| # | File | Description | REQ | Status |
|---|------|-------------|-----|--------|
| 1 | `src/kernel_model/_taxonomy.py` | `_INSTRUCTION_TAXONOMY` dict: PTX mnemonic prefix → InstructionRecord (category + arch_introduced + latency + throughput note) | REQ-0011-F7 | Not started |
| 2 | `src/kernel_model/_arch_table.py` | `ArchSpec` dataclass + `_ARCH_TABLE` dict for sm_80/86/89/90/100; includes `_MMA_LATENCY` values | REQ-0011-F6 | Not started |
| 3 | `src/kernel_model/ptx_tracer.py` | `InstructionRecord`, `TracerResult`, `PTXTracer` class with `trace()`, `trace_file()`, `bottleneck()` | REQ-0011-F1–F9 | Not started |
| 4 | `src/kernel_model/__init__.py` | Update re-exports: add PTXTracer, TracerResult | REQ-0011 | Not started |
| 5 | `demos/10_ptx_tracer/__init__.py` | Empty package init | REQ-0011 | Not started |
| 6 | `demos/10_ptx_tracer/ptx_fixtures/vector_add.ptx` | Handwritten minimal PTX: vector-add kernel | REQ-0011-F13 | Not started |
| 7 | `demos/10_ptx_tracer/ptx_fixtures/gemm_mma.ptx` | Handwritten minimal PTX: GEMM with mma.sync instructions | REQ-0011-F13 | Not started |
| 8 | `demos/10_ptx_tracer/main.py` | CLI demo: trace both fixtures vs A100 and H100; side-by-side comparison | REQ-0011-F13/F14 | Not started |
| 9 | `tests/test_ptx_tracer.py` | 13 CPU-safe test cases per ARCH-005 testing strategy | REQ-0011 | Not started |

### Architectures covered

sm_80 (A100), sm_86 (GA10x/RTX 3090), sm_89 (Ada/L40S/RTX 4090), sm_90 (H100), sm_100 (B100/B200/RTX 5090).

### Definition of Done

- [ ] `from src.kernel_model import PTXTracer` succeeds on a machine with no GPU
- [ ] `mma.sync` PTX traced against A100: `mma_warp_count == 1`, `mma_warpgroup_count == 0`
- [ ] `wgmma.mma_async` PTX traced against H100: `mma_warpgroup_count == 1`
- [ ] `tcgen05.mma` PTX traced against RTX 5090: `mma_cta_count == 1`
- [ ] `tcgen05.mma` PTX traced against A100: `unsupported_instructions` non-empty; note in `TracerResult.notes`
- [ ] `python demos/10_ptx_tracer/main.py` prints tracer report for both fixtures vs A100 and H100 without a GPU
- [ ] All 13 CPU tests pass
- [ ] Tech Lead sprint review: Approved or Conditional Approval

---

## Sprint 9 — PLANNED: CI/CD Pipeline + Jupyter Notebooks

**Goal**: Establish automated code quality validation via GitHub Actions and deliver interactive Jupyter notebooks for the two most foundational demos.

**REQ refs**: REQ-0012 (CI/CD), REQ-0013 (Jupyter notebooks)
**ARCH refs**: ARCH-006 (CI/CD), ARCH-007 (Notebooks)
**Depends on**: Sprint 8 complete — PTX Tracer (demos/10_ptx_tracer) must be delivered so CI pipeline covers all 10 demos

### Deliverables

| # | File | Description | REQ | Status |
|---|------|-------------|-----|--------|
| 1 | `.github/workflows/ci.yml` | Lint + CPU-safe tests on every push and PR | REQ-0012-F1/F2/F6/F7 | Not started |
| 2 | `.github/workflows/gpu-ci.yml` | Full GPU test suite; manual trigger; self-hosted runner config | REQ-0012-F3/F4 | Not started |
| 3 | `notebooks/01_core_apis.ipynb` | Interactive CUDA Python API walkthrough; GPU-guarded cells | REQ-0013-F1/F3/F4 | Not started |
| 4 | `notebooks/02_kmeans.ipynb` | Interactive GPU k-means clustering notebook; GPU-guarded cells | REQ-0013-F2/F3/F4 | Not started |

### Definition of Done

- [ ] Pushing to any branch triggers `CI` workflow automatically on GitHub Actions
- [ ] `CI` workflow passes: ruff clean + 72 CPU tests (+ any new tests from Sprints 7+8) green
- [ ] `jupyter nbconvert --execute notebooks/01_core_apis.ipynb` succeeds on CPU-only machine
- [ ] `jupyter nbconvert --execute notebooks/02_kmeans.ipynb` succeeds on CPU-only machine
- [ ] Both notebooks committed with cleared output
- [ ] Tech Lead sprint review: Approved or Conditional Approval

**Priority breakdown**: P0 = CI workflow; P1 = notebooks (both have user-visible value, CI unblocks badge in README)

---

## Sprint 10 — PLANNED: README + Physical GPU Validation

**Goal**: Deliver polished user-facing `README.md` covering all 10 demos and validate all 26 GPU tests against physical NVIDIA hardware.

**REQ refs**: REQ-0014 (README)
**Depends on**: Sprint 9 complete (CI badge available for README; notebooks exist to reference)

### Deliverables

| # | File | Description | REQ | Status |
|---|------|-------------|-----|--------|
| 1 | `README.md` | Top-level user-facing documentation: quickstart, demo table (01-10), prerequisites, test instructions, CI badge | REQ-0014-F1–F8 | Not started |
| 2 | GPU validation report | Run `pytest tests/ -v` on NVIDIA GPU hardware; document pass/fail per test | — | Not started |
| 3 | Bug fixes (if GPU validation reveals issues) | Fix any CPU/GPU correctness divergences found during hardware validation | — | Not started |

### Definition of Done

- [ ] `README.md` exists at project root and renders correctly on GitHub
- [ ] Demo Overview table lists all 10 demos (01-10) with correct one-line descriptions
- [ ] All relative links in README resolve correctly
- [ ] CI badge present in README
- [ ] All 26 GPU tests validated against physical hardware (pass or documented known failure with tracking issue)
- [ ] Tech Lead sprint review: Approved or Conditional Approval

---

## Sprint History

| Sprint | Status | Verdict | Key Deliverables |
|--------|--------|---------|-----------------|
| Sprint 0 | CLOSED | — | Bootstrap, all documents |
| Sprint 1 | CLOSED | Conditional Approval | src/ library, demos 01+02, 11 CPU tests |
| Sprint 2 | CLOSED | Conditional Approval | demos 03+04+05_kernels, 22 CPU tests |
| Sprint 3 | CLOSED | Conditional Approval | demos 06+07, 32 CPU tests |
| Sprint 4 | CLOSED | **Approved** | demos 05_naive_bayes, 39 CPU tests |
| Sprint 5 | CLOSED | Conditional Approval | 12 backend variants + comparison demo, 72 CPU tests |
| Sprint 6 | CLOSED | **Approved** | docs/slides/ — 37 slides covering all 8 demo directories |
| Sprint 7 | CLOSED | Conditional Approval | src/kernel_model/ — occupancy + roofline model library; 11 CPU tests; M-1 fixed |
| Sprint 8 | OPEN | — | src/kernel_model/ptx_tracer.py — PTX instruction tracer (Ampere→Blackwell) |
| Sprint 9 | PLANNED | — | GitHub Actions CI + Jupyter notebooks (demos 01+02) |
| Sprint 10 | PLANNED | — | README.md + physical GPU validation |

---

## Sprint 5 Complete — Multi-Backend Comparison

12 backend variant files + unified comparison demo + 33 new CPU tests. TL Conditional Approval issued, all fixes applied.

| Demo | REQ | Sprint |
|------|-----|--------|
| demos/01_core_apis/numba_vector_add.py, cupy_vector_add.py | REQ-0007 | Sprint 5 |
| demos/02_kmeans/numba_kmeans.py, cupy_kmeans.py | REQ-0007 | Sprint 5 |
| demos/03_pca/numba_pca.py, cupy_pca.py | REQ-0007 | Sprint 5 |
| demos/04_linear_model/numba_linear.py, cupy_linear.py | REQ-0007 | Sprint 5 |
| demos/05_naive_bayes/numba_nb.py, cupy_nb.py | REQ-0007 | Sprint 5 |
| demos/05_kernels/numba_kernels.py, cupy_kernels.py | REQ-0007 | Sprint 5 |
| demos/08_comparison/ (cuML comparison table) | REQ-0008 | Sprint 5 |
| benchmarks/run_all.py --backend flag | REQ-0007 | Sprint 5 |

---

## Sprint 4 Complete — Summary

All 6 requirements (REQ-0001 through REQ-0006) delivered across 4 sprints. Tech Lead issued **Approved** on Sprint 4 final review.

### Demos Delivered

| Demo | REQ | Sprint |
|------|-----|--------|
| demos/01_core_apis/ | REQ-0001 | Sprint 1 |
| demos/02_kmeans/ | REQ-0002 | Sprint 1 |
| demos/03_pca/ | REQ-0002 | Sprint 2 |
| demos/04_linear_model/ | REQ-0002 | Sprint 2 |
| demos/05_kernels/ (GEMM, ReLU, softmax) | REQ-0003 | Sprint 2 |
| demos/05_naive_bayes/ | REQ-0002 | Sprint 4 |
| demos/06_interop/ | REQ-0005 | Sprint 3 |
| demos/07_memory/ | REQ-0006 | Sprint 3 |
| benchmarks/run_all.py | REQ-0004 | Sprints 1–4 |

### Test Suite (Sprint 4 baseline)

| File | CPU Tests | GPU Tests |
|------|-----------|-----------|
| test_device.py | 2 | 3 |
| test_kernels.py | 4 | 1 |
| test_kmeans.py | 5 | 1 |
| test_memory.py | 2 | 2 |
| test_pca.py | 5 | 1 |
| test_linear.py | 4 | 1 |
| test_interop.py | 12 | 5 |
| test_naive_bayes.py | 7 | 1 |
| **Total (Sprint 4)** | **41** | **15** |

Sprint 5 added 31 CPU tests → **72 CPU total, 26 GPU total** (including 11 from multi-backend tests).

Run: `pytest tests/ -m "not gpu"` — 72 pass

---

## Sprint 6 Complete — Slide-Based Documentation

37 slides + README index across 10 subdirectories. Tech Lead Approved.

| Section | Slides | Status |
|---------|--------|--------|
| docs/slides/00_introduction/ | 2 | Done |
| docs/slides/01_core_apis/ | 6 | Done |
| docs/slides/02_kmeans/ | 5 | Done |
| docs/slides/03_pca/ | 4 | Done |
| docs/slides/04_linear_model/ | 3 | Done |
| docs/slides/05_kernels/ | 4 | Done |
| docs/slides/05_naive_bayes/ | 4 | Done |
| docs/slides/06_interop/ | 4 | Done |
| docs/slides/07_memory/ | 3 | Done |
| docs/slides/08_comparison/ | 2 | Done |

---

## Remaining Backlog

| Item | REQ | ARCH | Priority | Sprint |
|------|-----|------|----------|--------|
| Jupyter notebooks (demos 01+02) | REQ-0013 | ARCH-007 | P1 | Sprint 9 |
| CI/CD pipeline (GitHub Actions) | REQ-0012 | ARCH-006 | P1 | Sprint 9 |
| README.md user-facing documentation | REQ-0014 | — | P1 | Sprint 10 |
| Physical GPU validation (all 26 GPU tests) | — | — | P1 | Sprint 10 |
| Jupyter notebooks (demos 03-08) | — | — | P2 | Backlog |
| FP16/BF16/TF32 performance modeling extensions | — | — | P2 | Backlog |
| cuobjdump PTX extraction in PTXTracer | — | — | P2 | Backlog |
| Code coverage reporting in CI | — | — | P2 | Backlog |

---

## Key File Locations

| Document | Path |
|----------|------|
| Requirements (active) | `agents/requirements/REQ-0010.md` – `REQ-0014.md` |
| Requirements (implemented) | `agents/requirements/REQ-0001.md` – `REQ-0009.md` |
| Architecture | `agents/architecture/ARCH-001.md` – `ARCH-007.md` |
| TL reviews | `agents/reviews/code/TL-review-sprint{1-7}*.md` |
| Sprint retros | `agents/session-logs/proj-mgr/sprint-{1-6}-retro-2026-05-01.md`, `sprint-7-retro-2026-05-05.md` |
| PTX tracer research | `agents/architecture/ptx-tracer-research.md` |
| Prior art survey | `agents/architecture/prior-art-kernel-model.md` |
| Run all demos | `python benchmarks/run_all.py` (requires GPU) |
| Run CPU tests | `pytest tests/ -m "not gpu"` |
| Run GPU tests | `pytest tests/ -v -m gpu` (requires GPU) |
