# agents/todo.md — CUDA Python ML Demos Task Tracker

## Sprint 3 — Interop, Memory, Notebooks (2026-05-01 → 2026-05-15)

### Claude Manager
- [x] [Claude-Mgr] Close Sprint 2, open Sprint 3

### Product Manager
- [x] [Prod-Mgr] Review and finalize REQ-0005: NumPy/CuPy/PyTorch Interop
- [x] [Prod-Mgr] Review and finalize REQ-0006: Memory Management Patterns

### Programmer
- [x] [Prog] Fix TL-S2-002: Add comment/note about wall-clock timing for k-means GPU in run_all.py
- [x] [Prog] Fix TL-S2-004: Document cache_key param in compiler.py compile() docstring
- [ ] [Prog] Implement `demos/05_naive_bayes/` — GPU Gaussian naive Bayes (P1 carry from Sprint 2) [DEFERRED]
- [x] [Prog] Implement `demos/06_interop/` — NumPy/CuPy/PyTorch interop (REQ-0005)
- [x] [Prog] Implement `demos/07_memory/` — Memory management patterns (REQ-0006)
- [ ] [Prog] Generate Jupyter notebook versions of demos 01 and 02

### QA Agent
- [x] [QA] Write test plans for REQ-0005, REQ-0006
- [x] [QA] Implement `tests/test_interop.py`

### Tech Lead
- [!] [Tech-Lead] Sprint 3 code review (triggers at sprint end)

---

## Backlog (Future Sprints)

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
- [x] [Prod-Mgr] REQ-0001, REQ-0002 marked Active
- [x] [Arch] ARCH-001, ARCH-002 approved; pyproject.toml created
- [x] [Prog] src/utils/ (device, memory, timing)
- [x] [Prog] src/kernels/ (compiler, compiled_kernel)
- [x] [Prog] demos/01_core_apis/ (device_info, vector_add, pinned_memory, main)
- [x] [Prog] demos/02_kmeans/ (cpu_kmeans, gpu_kmeans, main)
- [x] [Prog] benchmarks/run_all.py
- [x] [Prog] Fix TL-001 (pinned mem leak), TL-003 (RNG loop reset)
- [x] [QA] TEST-PLAN-REQ-0001.md, TEST-PLAN-REQ-0002.md
- [x] [QA] tests/conftest.py, test_device.py, test_kernels.py, test_kmeans.py
- [x] [Tech-Lead] Sprint 1 review — Conditional Approval (2026-05-01)

### Sprint 2 — Remaining Algorithm Demos (CLOSED 2026-05-01)
- [x] [Prod-Mgr] REQ-0003, REQ-0004 marked Active
- [x] [Prog] Fix TL-004 (test_memory.py — delivered by QA)
- [x] [Prog] Fix TL-005 (duplicate import in vector_add.py)
- [x] [Prog] Fix TL-006 (BenchmarkRunner in run_all.py)
- [x] [Prog] Fix TL-007 (compiler.py cache_key comment)
- [x] [Prog] demos/03_pca/ (cpu_pca, gpu_pca, main)
- [x] [Prog] demos/04_linear_model/ (cpu_linear, gpu_linear, main)
- [x] [Prog] demos/05_kernels/ (gemm, activations, main)
- [x] [Prog] Fix TL-S2-001 (ReLU timing fix), TL-S2-003 (benchmark runner Sprint 2 demos)
- [x] [QA] TEST-PLAN-REQ-0003.md, TEST-PLAN-REQ-0004.md
- [x] [QA] tests/test_memory.py, test_pca.py, test_linear.py
- [x] [Tech-Lead] Sprint 2 review — Conditional Approval (2026-05-01)
