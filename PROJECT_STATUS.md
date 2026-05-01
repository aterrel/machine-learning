# PROJECT_STATUS.md — CUDA Python ML Demos

**Last Updated**: 2026-05-01
**Current Sprint**: Sprint 1
**Sprint Dates**: 2026-05-01 → 2026-05-15

---

## Project Health

| Area | Status | Notes |
|------|--------|-------|
| Requirements | 🟡 Yellow | REQ-0001–0006 drafted, not yet reviewed |
| Architecture | 🟡 Yellow | ARCH-001 created, ARCH-002 created; not yet reviewed |
| Implementation | 🔴 Red | Not started |
| Tests | 🔴 Red | Not started |
| Documentation | 🟡 Yellow | Project structure defined; inline docs pending |
| CI/Build | 🔴 Red | Not configured |

---

## Sprint 1 Goal

**Make the core GPU loop work end-to-end**: Stand up the repo structure, validate CUDA Python API access (`cuda.core` + `cuda.bindings`), and ship one fully working GPU-accelerated demo (k-means) with a CPU baseline comparison and correctness check.

---

## Sprint Backlog

### Pending

- [ ] [Prod-Mgr] Review and finalize REQ-0001 through REQ-0006
- [ ] [Arch] Review ARCH-001 (Overall Architecture) and ARCH-002 (CUDA Kernel Pipeline) and issue Approved status
- [ ] [Arch] Create ARCH-001 → Programmer handoff document
- [ ] [Prog] Scaffold `src/utils/` — device info, memory helpers, timing utilities
- [ ] [Prog] Implement `demos/01_core_apis/` — CUDA Python API tour (cuda.core + cuda.bindings)
- [ ] [Prog] Implement `demos/02_kmeans/` — GPU k-means with CPU baseline
- [ ] [Prog] Implement `benchmarks/benchmark_runner.py` — unified benchmark harness
- [ ] [QA] Write test plans for REQ-0001, REQ-0002
- [ ] [QA] Write GPU smoke tests (`tests/test_device.py`, `tests/test_kmeans.py`)
- [ ] [Tech Lead] Sprint 1 code review (triggers at sprint end)

### In Progress

_(none yet)_

### Completed (Sprint 0 / Bootstrap)

- [x] [Claude-Mgr] PROJECT.md filled in and validated
- [x] [Claude-Mgr] CLAUDE.md generated
- [x] [Claude-Mgr] PROJECT_STATUS.md initialized (this file)
- [x] [Claude-Mgr] agents/todo.md created
- [x] [Claude-Mgr] REQ-0001 through REQ-0006 created
- [x] [Claude-Mgr] ARCH-001: Overall System Architecture created
- [x] [Claude-Mgr] ARCH-002: CUDA Kernel Pipeline created
- [x] [Claude-Mgr] Arch → Programmer handoff created

---

## Agent Assignments

| Agent | Status | Current Task |
|-------|--------|-------------|
| Claude Manager | Active | Bootstrap complete; monitoring sprint |
| Product Manager | Pending | Review REQ-0001–0006 |
| Software Architect | Pending | Review and approve ARCH-001, ARCH-002 |
| Programmer | Blocked | Awaiting ARCH handoff |
| QA Agent | Available | Can begin test plans for REQ-0001, REQ-0002 |
| Tech Lead | Blocked | Awaiting sprint end |
| Project Manager | Available | Tracking sprint progress |

---

## Key File Locations

| Document | Path |
|----------|------|
| Project objectives | `PROJECT.md` |
| Project guidance for Claude | `CLAUDE.md` |
| Task tracker | `agents/todo.md` |
| Requirements | `agents/requirements/REQ-0001.md` – `REQ-0006.md` |
| Architecture (overall) | `agents/architecture/ARCH-001.md` |
| Architecture (kernels) | `agents/architecture/ARCH-002.md` |
| Arch → Prog handoff | `agents/handoffs/HANDOFF-arch-to-prog-all-reqs-2026-05-01.md` |
| Agent definitions | `agents/workflow/agents.md` |
| Session logs | `agents/session-logs/claude-manager/2026-05-01-001.md` |

---

## Open Blockers

- GPU availability not yet confirmed in CI — all GPU tests require a physical NVIDIA GPU
- `cuda-python` version compatibility with installed CUDA Toolkit needs validation at first run
