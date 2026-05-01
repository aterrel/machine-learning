# agents/todo.md — CUDA Python ML Demos Task Tracker

## Status: SPRINT 5 IN PROGRESS (2026-05-01)

REQ-0007 + REQ-0008 created. Sprint 5 goal: multi-backend comparison (Numba + CuPy + cuML).
39 CPU tests pass. 15 GPU tests ready for hardware validation.

---

## Sprint 5 — Multi-Backend Comparison (IN PROGRESS)

### REQ-0007: Numba-CUDA + CuPy variants for all algorithm demos

- [ ] [Arch] Create ARCH-003 for multi-backend design pattern
- [ ] [Prog] demos/01_core_apis/numba_vector_add.py
- [ ] [Prog] demos/01_core_apis/cupy_vector_add.py
- [ ] [Prog] demos/02_kmeans/numba_kmeans.py
- [ ] [Prog] demos/02_kmeans/cupy_kmeans.py
- [ ] [Prog] demos/03_pca/numba_pca.py
- [ ] [Prog] demos/03_pca/cupy_pca.py
- [ ] [Prog] demos/04_linear_model/numba_linear.py
- [ ] [Prog] demos/04_linear_model/cupy_linear.py
- [ ] [Prog] demos/05_naive_bayes/numba_nb.py
- [ ] [Prog] demos/05_naive_bayes/cupy_nb.py
- [ ] [Prog] demos/05_kernels/numba_kernels.py (P1)
- [ ] [Prog] demos/05_kernels/cupy_kernels.py (P1)
- [ ] [Prog] Update benchmarks/run_all.py with --backend flag and multi-column table
- [ ] [QA]  tests/test_numba_variants.py (CPU-safe: shape + skip-if-absent)
- [ ] [QA]  tests/test_cupy_variants.py (CPU-safe: shape + skip-if-absent)

### REQ-0008: Unified comparison demo (demos/08_comparison/) + cuML

- [ ] [Prog] demos/08_comparison/__init__.py
- [ ] [Prog] demos/08_comparison/main.py (CLI runner; table output)
- [ ] [Prog] demos/08_comparison/backends/__init__.py
- [ ] [Prog] demos/08_comparison/backends/cuml_backends.py (guarded import)
- [ ] [QA]  tests/test_comparison.py (CPU-safe: SKIPPED rows when backends absent)

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
