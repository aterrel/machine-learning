---
session: proj-mgr/sprint-1-retro-2026-05-01
agent: Project Manager
date: 2026-05-01
sprint: 1
---

# Sprint 1 Retrospective — CUDA Python ML Demos

## Sprint Goal (was)
Make the core GPU loop work end-to-end: stand up repo structure, validate CUDA Python API access, ship one fully working GPU-accelerated demo (k-means) with CPU baseline and correctness check.

## Outcome: COMPLETE (Conditional Approval)

All Sprint 1 P0 tasks delivered. Tech Lead issued Conditional Approval — no blockers, two major issues fixed within the sprint.

## Commits This Sprint

| Commit | Agent | Work |
|--------|-------|------|
| 526acbd | Prod-Mgr + Arch | REQ-0001/0002 Active, ARCH-001/002 Approved, pyproject.toml |
| 4ad22a9 | Programmer | src/utils/, src/kernels/, demos/01_core_apis/, demos/02_kmeans/, benchmarks/ |
| 0c1d6d7 | QA | tests/, test plans for REQ-0001/0002 |
| c027de8 | Tech Lead | Sprint 1 code review (Conditional Approval) |
| a286f62 | Programmer | Fixed TL-001 (pinned mem leak), TL-003 (RNG loop reset) |

## What Went Well
- All CPU baseline code correct and tested (11/11 tests pass)
- Memory safety discipline: DeviceBuffer context manager used throughout
- CUDA error checking on every cudaRT call
- Self-contained demo structure followed correctly
- QA test separation (CPU vs GPU) is clean and CI-friendly

## What Needs Improvement
- Pinned memory functions initially leaked GPU memory (TL-001) — caught by review
- Empty-cluster RNG was reset each iteration (TL-003) — subtle bug, caught by review
- test_memory.py was not implemented (TL-004) — carry into Sprint 2

## Remaining Minor Items (carry to Sprint 2)
- TL-004: Implement tests/test_memory.py (DeviceBuffer, alloc_pinned)
- TL-005: Remove duplicate import in vector_add.py
- TL-006: Use BenchmarkRunner in benchmarks/run_all.py instead of raw wall-clock
- TL-007: Document cache_key=None behavior in compiler.py

## Sprint 2 Goals
Implement remaining ML algorithm demos: GPU PCA (demos/03_pca/), GPU linear regression (demos/04_linear_model/), GPU naive Bayes (demos/05_naive_bayes/), plus GEMM kernels demo. Carry minor TL fixes from Sprint 1.
