# agents/todo.md — CUDA Python ML Demos Task Tracker

## Sprint 1 — Core GPU Loop + K-Means Demo (2026-05-01 → 2026-05-15)

### Claude Manager
- [x] [Claude-Mgr] Bootstrap initialization
- [x] [Claude-Mgr] CLAUDE.md generated from PROJECT.md
- [x] [Claude-Mgr] PROJECT_STATUS.md created (Sprint 1 defined)
- [x] [Claude-Mgr] REQ-0001 through REQ-0006 created
- [x] [Claude-Mgr] ARCH-001: Overall System Architecture created
- [x] [Claude-Mgr] ARCH-002: CUDA Kernel Pipeline created
- [x] [Claude-Mgr] Arch → Programmer handoff created

### Product Manager
- [ ] [Prod-Mgr] Review and finalize REQ-0001: Core CUDA Python API Demonstrations
- [ ] [Prod-Mgr] Review and finalize REQ-0002: GPU-Accelerated ML Algorithm Demos
- [ ] [Prod-Mgr] Mark REQ-0001 and REQ-0002 as Active when approved

### Software Architect
- [x] [Arch] ARCH-001: Overall System Architecture (created in bootstrap)
- [x] [Arch] ARCH-002: CUDA Kernel Compilation and Launch Pipeline (created in bootstrap)
- [ ] [Arch] Review ARCH-001 and ARCH-002 — mark as Approved when satisfied
- [ ] [Arch] Issue Arch → Programmer handoff (update HANDOFF doc with approval)
- [ ] [Arch] Create pyproject.toml with ruff + pytest config

### Programmer
- [!] [Prog] Scaffold `src/utils/device.py` (blocked — await ARCH handoff approval)
- [!] [Prog] Scaffold `src/utils/memory.py` — DeviceBuffer context manager (blocked)
- [!] [Prog] Scaffold `src/utils/timing.py` — BenchmarkResult, BenchmarkRunner (blocked)
- [!] [Prog] Implement `src/kernels/compiler.py` — KernelCompiler via NVRTC (blocked)
- [!] [Prog] Implement `src/kernels/compiled_kernel.py` — CompiledKernel.launch() (blocked)
- [!] [Prog] Implement `demos/01_core_apis/` — device info + vector-add (blocked)
- [!] [Prog] Implement `demos/02_kmeans/` — GPU k-means with CPU baseline (blocked)
- [!] [Prog] Implement `benchmarks/run_all.py` — unified benchmark CLI (blocked)

### QA Agent
- [ ] [QA] Write test plan for REQ-0001: Core CUDA Python API Demonstrations (can start now)
- [ ] [QA] Write test plan for REQ-0002: GPU-Accelerated ML Algorithm Demos (can start now)
- [!] [QA] Implement `tests/conftest.py` — GPU pytest mark, device fixture (blocked — await src/ scaffold)
- [!] [QA] Implement `tests/test_device.py` (blocked)
- [!] [QA] Implement `tests/test_kernels.py` (blocked)
- [!] [QA] Implement `tests/test_kmeans.py` (blocked)

### Tech Lead
- [!] [Tech-Lead] Sprint 1 code review (blocked — triggers at sprint end, after Programmer delivers)

---

## Backlog (Future Sprints)

### Sprint 2 — Remaining Algorithm Demos
- [ ] [Prog] Implement `demos/03_pca/` — GPU PCA (REQ-0002)
- [ ] [Prog] Implement `demos/04_linear_model/` — GPU linear regression (REQ-0002)
- [ ] [Prog] Implement `demos/05_naive_bayes/` — GPU naive Bayes (REQ-0002)
- [ ] [Prog] Implement `demos/05_kernels/` — GEMM + activations (REQ-0003)
- [ ] [QA] Test plans for REQ-0003, REQ-0004

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

## Completed (Sprint 0 / Bootstrap)

- [x] [Claude-Mgr] PROJECT.md filled in and validated
- [x] [Claude-Mgr] Bootstrap initialization complete (2026-05-01)
