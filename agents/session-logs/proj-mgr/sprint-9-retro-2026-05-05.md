# Sprint 9 Retrospective — CI/CD Pipeline + Jupyter Notebooks
Date: 2026-05-05
Sprint: 9
Verdict: Conditional Approval

## Delivered
- `.github/workflows/ci.yml` — lint (ruff) + CPU tests (96 passing) on every push/PR
- `.github/workflows/gpu-ci.yml` — manual-trigger GPU suite; self-hosted runner config documented
- `notebooks/01_core_apis.ipynb` — 13-cell interactive DeviceSpec/Occupancy/Roofline walkthrough; GPU-guarded; cleared output
- `notebooks/02_kmeans.ipynb` — 11-cell interactive k-means demo; CPU baseline always runs; GPU cell gated

## What Went Well
- CI/CD architecture exactly matched ARCH-006 spec
- Notebook GPU-guard pattern (GPU_AVAILABLE flag) is clean and reusable
- TL accepted the ARCH-007 import strategy deviation as pragmatic (ARCH-007 spec was aspirational)
- 0 critical, 0 major findings

## Minor Findings Tracked (non-blocking)
- m-1: CI badge missing from README (REQ-0012-F5) — Sprint 10 deliverable
- m-2: gpu-ci.yml missing cupy/torch install (interop test gap) — Sprint 10 fix
- m-3: 01_core_apis.ipynb GPU cell is CLI instructions only, not live code — tracked for post-Sprint 10

## Sprint 10 Setup
Sprint 10: README.md + Physical GPU Validation
