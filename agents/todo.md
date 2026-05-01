# agents/todo.md — CUDA Python ML Demos Task Tracker

## Sprint 2 — Remaining Algorithm Demos (2026-05-01 → 2026-05-15)

### Claude Manager
- [x] [Claude-Mgr] Close Sprint 1, open Sprint 2

### Product Manager
- [ ] [Prod-Mgr] Review and finalize REQ-0003: Custom CUDA Kernels Library
- [ ] [Prod-Mgr] Review and finalize REQ-0004: Benchmarking Infrastructure

### Programmer
- [ ] [Prog] Fix TL-004: Implement `tests/test_memory.py`
- [x] [Prog] Fix TL-005: Remove duplicate import in `demos/01_core_apis/vector_add.py`
- [x] [Prog] Fix TL-006: Use BenchmarkRunner in `benchmarks/run_all.py`
- [x] [Prog] Fix TL-007: Document `cache_key=None` behavior in `src/kernels/compiler.py`
- [x] [Prog] Implement `demos/03_pca/` — GPU PCA via covariance matrix (REQ-0002)
- [x] [Prog] Implement `demos/04_linear_model/` — GPU linear regression normal equation (REQ-0002)
- [ ] [Prog] Implement `demos/05_naive_bayes/` — GPU Gaussian naive Bayes (REQ-0002) [deferred Sprint 3]
- [x] [Prog] Implement `demos/05_kernels/` — GEMM + ReLU + softmax kernels (REQ-0003)

### QA Agent
- [ ] [QA] Write test plans for REQ-0003, REQ-0004
- [ ] [QA] Implement `tests/test_pca.py`
- [ ] [QA] Implement `tests/test_linear.py`

### Tech Lead
- [!] [Tech-Lead] Sprint 2 code review (triggers at sprint end)

---

## Backlog (Future Sprints)

### Sprint 3 — Interop, Memory, Notebooks
- [ ] [Prog] Implement `demos/06_interop/` — NumPy/CuPy/PyTorch interop (REQ-0005)
- [ ] [Prog] Implement `demos/07_memory/` — memory management patterns (REQ-0006)
- [ ] [Prog] Generate Jupyter notebook versions of all demos
- [ ] [QA] Test plans for REQ-0005, REQ-0006

### Sprint 4 — Polish and Documentation
- [ ] [Prod-Mgr] Review all REQ documents against completed implementation
- [ ] [Tech-Lead] Final architecture review
- [ ] README.md user-facing documentation

---

## Completed

### Sprint 0 / Bootstrap
- [x] [Claude-Mgr] PROJECT.md filled in and validated
- [x] [Claude-Mgr] Bootstrap initialization complete (2026-05-01)

### Sprint 1 — Core GPU Loop + K-Means Demo (CLOSED 2026-05-01)
- [x] [Claude-Mgr] CLAUDE.md generated from PROJECT.md
- [x] [Claude-Mgr] PROJECT_STATUS.md created (Sprint 1 defined)
- [x] [Claude-Mgr] REQ-0001 through REQ-0006 created
- [x] [Claude-Mgr] ARCH-001: Overall System Architecture created
- [x] [Claude-Mgr] ARCH-002: CUDA Kernel Pipeline created
- [x] [Claude-Mgr] Arch → Programmer handoff created
- [x] [Prod-Mgr] Review and finalize REQ-0001, REQ-0002 — marked Active
- [x] [Arch] ARCH-001, ARCH-002 approved
- [x] [Arch] pyproject.toml created (ruff + pytest config)
- [x] [Prog] Scaffold `src/utils/device.py`, `memory.py`, `timing.py`
- [x] [Prog] Implement `src/kernels/compiler.py`, `compiled_kernel.py`
- [x] [Prog] Implement `demos/01_core_apis/` (device_info, vector_add, pinned_memory, main)
- [x] [Prog] Implement `demos/02_kmeans/` (cpu_kmeans, gpu_kmeans, main)
- [x] [Prog] Implement `benchmarks/run_all.py`
- [x] [Prog] Fix TL-001 (pinned mem leak), TL-003 (RNG loop reset)
- [x] [QA] TEST-PLAN-REQ-0001.md, TEST-PLAN-REQ-0002.md
- [x] [QA] tests/conftest.py, test_device.py, test_kernels.py, test_kmeans.py (11 CPU tests pass)
- [x] [Tech-Lead] Sprint 1 review — Conditional Approval (2026-05-01)
