# agents/todo.md — CUDA Python ML Demos Task Tracker

## Sprint 4 — Polish and Documentation (2026-05-01 → 2026-05-15)

### Claude Manager
- [x] [Claude-Mgr] Close Sprint 3, open Sprint 4

### Product Manager
- [x] [Prod-Mgr] Final review of all REQ documents against completed implementation

### Programmer
- [x] [Prog] Fix TL-S3-004 (Nit): simplify except clause in demos/07_memory/main.py demo_oom_recovery
- [x] [Prog] Fix TL-S3-005 (Nit): add clarifying comment to demo_basic_alloc
- [x] [Prog] Implement `demos/05_naive_bayes/` — GPU Gaussian naive Bayes (P1, deferred ×2)
- [ ] [Prog] Generate Jupyter notebook versions of demos 01 and 02

### Tech Lead
- [x] [Tech-Lead] Final architecture review (Sprint 4)

---

## Backlog

- [ ] README.md user-facing documentation

---

## Completed

### Sprint 0 / Bootstrap
- [x] [Claude-Mgr] PROJECT.md filled in and validated
- [x] [Claude-Mgr] Bootstrap initialization complete (2026-05-01)

### Sprint 1 — Core GPU Loop + K-Means Demo (CLOSED 2026-05-01)
- [x] [Claude-Mgr] CLAUDE.md, PROJECT_STATUS.md, REQ/ARCH documents created
- [x] [Prod-Mgr] REQ-0001, REQ-0002 marked Active
- [x] [Arch] ARCH-001, ARCH-002 approved; pyproject.toml created
- [x] [Prog] src/utils/ (device, memory, timing), src/kernels/ (compiler, compiled_kernel)
- [x] [Prog] demos/01_core_apis/, demos/02_kmeans/, benchmarks/run_all.py
- [x] [Prog] Fix TL-001 (pinned mem leak), TL-003 (RNG loop reset)
- [x] [QA] TEST-PLAN-REQ-0001/0002; tests/conftest.py, test_device, test_kernels, test_kmeans
- [x] [Tech-Lead] Sprint 1 review — Conditional Approval

### Sprint 2 — Remaining Algorithm Demos (CLOSED 2026-05-01)
- [x] [Prod-Mgr] REQ-0003, REQ-0004 marked Active
- [x] [Prog] demos/03_pca/, demos/04_linear_model/, demos/05_kernels/
- [x] [Prog] Fix TL-001–007 carry items; TL-S2-001, TL-S2-003 fixes
- [x] [QA] TEST-PLAN-REQ-0005/0006; test_memory.py, test_pca.py, test_linear.py
- [x] [Tech-Lead] Sprint 2 review — Conditional Approval

### Sprint 3 — Interop, Memory (CLOSED 2026-05-01)
- [x] [Prod-Mgr] REQ-0005, REQ-0006 marked Active
- [x] [Prog] Fix TL-S2-002, TL-S2-004 carry items
- [x] [Prog] demos/06_interop/ (cupy_interop, torch_interop, pipeline, main)
- [x] [Prog] demos/07_memory/main.py
- [x] [Prog] Fix TL-S3-001/002 (misleading comments), TL-S3-003 (NameError guard)
- [x] [QA] TEST-PLAN-REQ-0005/0006; tests/test_interop.py
- [x] [Tech-Lead] Sprint 3 review — Conditional Approval
