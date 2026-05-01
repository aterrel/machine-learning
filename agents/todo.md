# agents/todo.md — CUDA Python ML Demos Task Tracker

## Status: PROJECT COMPLETE — SPRINT 5 CLOSED (2026-05-01)

All 8 requirements delivered (REQ-0001–0008). 72 CPU tests pass. 26 GPU tests ready for hardware validation.

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
