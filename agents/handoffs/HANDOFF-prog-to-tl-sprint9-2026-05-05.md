# Handoff: Programmer → Tech Lead
Sprint: 9 — CI/CD Pipeline + Jupyter Notebooks
Date: 2026-05-05
From: Programmer
To: Tech Lead

## What Was Built
Four files added in this sprint:
1. `.github/workflows/ci.yml` — Lint (ruff) + CPU-safe test job on ubuntu-latest, no cuda-python
2. `.github/workflows/gpu-ci.yml` — GPU test job with manual trigger + self-hosted runner config
3. `notebooks/01_core_apis.ipynb` — 13-cell notebook: DeviceSpec/Occupancy/Roofline analysis (all CPU), GPU redirect cell
4. `notebooks/02_kmeans.ipynb` — 11-cell notebook: CPU k-means baseline always runs, GPU cell is gated

## Architecture Deviation to Flag
ARCH-007 specified that notebooks should call `run_demo()` / `run_cpu_baseline()` / `run_gpu_version()` from the demos/ modules. The actual implementation:
- `01_core_apis.ipynb` imports directly from `src.kernel_model` (DeviceSpec, OccupancyModel, RooflineModel) — provides richer CPU-only interactive analysis
- `02_kmeans.ipynb` loads `demos/02_kmeans/cpu_kmeans.py` directly via importlib.util (needed because "02_kmeans" starts with a digit)

Both approaches are functional and arguably better for learners. TL should evaluate whether this deviation from ARCH-007 is acceptable.

## REQ Compliance
- REQ-0012 (CI/CD): ci.yml triggers on push/PR; gpu-ci.yml is manual-trigger with self-hosted runner ✓
- REQ-0013 (Notebooks): Both notebooks committed with cleared output; GPU-guarded cells ✓

## Files to Review
- `.github/workflows/ci.yml`
- `.github/workflows/gpu-ci.yml`
- `notebooks/01_core_apis.ipynb`
- `notebooks/02_kmeans.ipynb`
