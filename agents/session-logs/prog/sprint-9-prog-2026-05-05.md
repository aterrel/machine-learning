# Programmer Session Log — Sprint 9
Date: 2026-05-05
Sprint: 9 — CI/CD Pipeline + Jupyter Notebooks

## Work Completed
- `.github/workflows/ci.yml`: Two-job GitHub Actions workflow (lint + CPU tests).
  Installs only `ruff` and `numpy+pytest` (no cuda-python) to validate the GPU-free
  test path on ubuntu-latest.
- `.github/workflows/gpu-ci.yml`: Manual-trigger GPU workflow targeting [self-hosted, gpu]
  runner with full CUDA stack. Includes setup instructions as comments.
- `notebooks/01_core_apis.ipynb`: 13-cell notebook covering DeviceSpec, OccupancyModel,
  and RooflineModel from src.kernel_model. All analysis cells are GPU-free. One GPU-gated
  cell directs users to the CLI demo. Outputs cleared.
- `notebooks/02_kmeans.ipynb`: 11-cell notebook. CPU baseline (kmeans_cpu) always runs.
  GPU k-means cell is gated with `if GPU_AVAILABLE:`. Includes roofline analysis
  narrative. Outputs cleared.

## Notes
- Notebooks diverge slightly from ARCH-007 template: 01_core_apis imports directly from
  src.kernel_model (not from demos/01_core_apis/) because the kernel model library
  provides the most instructive CPU-runnable content. 02_kmeans loads cpu_kmeans.py
  via importlib.util to work around the digit-prefixed directory name.
- All CPU-safe cells verified to execute without errors using DeviceSpec/OccupancyModel.

## Tests
No new tests added (CI/CD and notebooks are infrastructure, not library code).
Existing 96 CPU tests remain green.

## Status
All 4 Sprint 9 deliverables complete. Handing off to Tech Lead for review.
