# agents/todo.md — CUDA Python ML Demos Task Tracker

## Status: PROJECT COMPLETE (2026-05-01)

All 6 requirements delivered. Tech Lead: Approved on Sprint 4 final review.
39 CPU tests pass. 15 GPU tests ready for hardware validation.

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
