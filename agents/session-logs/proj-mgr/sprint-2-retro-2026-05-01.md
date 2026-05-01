---
session: proj-mgr/sprint-2-retro-2026-05-01
agent: Project Manager
date: 2026-05-01
sprint: 2
---

# Sprint 2 Retrospective — CUDA Python ML Demos

## Sprint Goal (was)
Implement remaining ML algorithm demos: GPU PCA, GPU linear regression, GEMM/activation kernels. Fix Sprint 1 TL minor issues.

## Outcome: COMPLETE (Conditional Approval)

All Sprint 2 P0 tasks delivered. Tech Lead issued Conditional Approval — no blockers, two minor issues fixed within the sprint.

## Commits This Sprint

| Commit | Agent | Work |
|--------|-------|------|
| ef91e05 | Prod-Mgr | REQ-0003, REQ-0004 marked Active |
| b8cfa59 | Programmer | demos/03_pca/, demos/04_linear_model/, demos/05_kernels/; TL fixes |
| 1b79ad9 | QA | test_memory.py, test_pca.py, test_linear.py, test plans |
| 148651d | Tech Lead | Sprint 2 review (Conditional Approval) |
| c2c98f5 | Programmer | Fixed TL-S2-001 (ReLU timing), TL-S2-003 (benchmark runner) |

## Test Counts
- CPU-only tests: 22 passing
- GPU tests: 9 deselected (require hardware)
- Total test files: 6

## What Went Well
- Algorithm implementations (PCA, linear regression, GEMM, ReLU, softmax) all correct
- Memory safety: all DeviceBuffers in context managers across all new files
- Sign-flip PCA comparison handled correctly
- CPU baselines are clean numpy implementations

## What Needs Improvement
- ReLU timing initially included H2D transfer inside timed closure (caught by TL review)
- Benchmark runner not updated to include Sprint 2 demos (caught by TL review)

## Remaining Minor Items (carry to Sprint 3)
- TL-S2-002: k-means GPU timing uses time_cpu instead of time_gpu (wall-clock vs CUDA events)
- TL-S2-004: compiler.py compile() docstring missing cache_key parameter docs

## Sprint 3 Goals
Interop demos (NumPy/CuPy/PyTorch), memory management patterns demo, naive Bayes (P1 from Sprint 2), Jupyter notebook versions of all demos. Address TL-S2-002/004.
