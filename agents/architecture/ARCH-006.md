# ARCH-006: CI/CD Pipeline (GitHub Actions)

**Status**: Draft
**Created**: 2026-05-05
**Author**: Software Architect
**REQ Reference**: REQ-0012
**Approved By**: —

---

## Overview

This document defines the CI/CD pipeline architecture for the CUDA Python ML Demos project using GitHub Actions. The primary challenge is that the project requires an NVIDIA GPU for its core tests, but GitHub-hosted runners do not provide GPUs. The architecture addresses this by splitting the pipeline into a GPU-free tier (runs on every push) and a GPU tier (runs on-demand or via self-hosted runner), ensuring basic code quality is always validated without requiring hardware.

---

## Context

The project currently has `pyproject.toml` in place but no CI pipeline (CI/Build status: Red). The project contains:
- 72 CPU-safe tests (`pytest tests/ -m "not gpu"`) — runnable anywhere
- 26 GPU tests (`pytest tests/ -m "gpu"`) — require NVIDIA GPU + CUDA 12.x
- `ruff` lint + format checks
- Jupyter notebooks (Sprint 9 deliverable)

GitHub-hosted runners (ubuntu-latest) provide no GPU. Options for GPU CI:
1. Self-hosted GitHub Actions runner on an NVIDIA GPU machine
2. NVIDIA LaunchPad / DGX Cloud CI integration
3. CPU-only CI with GPU tests skipped (most portable, zero infrastructure cost)

Option 3 is selected as the default; Option 1 is documented as the upgrade path.

---

## Design

### Pipeline Structure

Two workflows:

#### Workflow 1: `ci.yml` — Code Quality + CPU Tests (runs on every push and PR)

```yaml
name: CI
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff
      - run: ruff check src/ demos/ tests/
      - run: ruff format --check src/ demos/ tests/

  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install numpy pytest
      # Note: cuda-python is NOT installed — GPU-free path
      - run: pytest tests/ -m "not gpu" -v
```

**Rationale**: `cuda-python` is not installed in the CPU job. All `src/kernel_model/` modules (Sprint 7+8) import zero GPU libraries at module level, so they run cleanly. `DeviceSpec.from_device()` is not called (it would fail at call time, not import time).

#### Workflow 2: `gpu-ci.yml` — GPU Tests (manual trigger or self-hosted runner)

```yaml
name: GPU CI
on:
  workflow_dispatch:   # manual trigger
  # Uncomment below to add self-hosted GPU runner:
  # push:
  #   branches: [main]
jobs:
  test-gpu:
    runs-on: [self-hosted, gpu]   # requires self-hosted runner with NVIDIA GPU
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install cuda-python numpy cupy torch pytest ruff
      - run: pytest tests/ -v   # runs all tests including GPU tests
```

### File Layout

```
.github/
  workflows/
    ci.yml          # lint + CPU-safe tests (every push)
    gpu-ci.yml      # GPU tests (manual trigger / self-hosted)
pyproject.toml      # existing (already has ruff config)
```

---

## Key Design Decisions

### 1. Split CPU and GPU workflows

GPU CI cannot run on GitHub-hosted runners. Rather than fail loudly on every push, the CPU-only pipeline gives fast feedback on lint and logic correctness. The GPU pipeline is a separate concern — it can be run on-demand or wired to a self-hosted runner when hardware is available.

**Tradeoff**: GPU tests are not automatically validated on every push. This is acceptable given the project's demo/educational focus and the absence of a self-hosted GPU runner in the default configuration.

### 2. No `cuda-python` install in CPU CI

Installing `cuda-python` on a GPU-free runner would succeed at install time but fail at first GPU API call. To avoid confusing failures, `cuda-python` is explicitly excluded from the CPU CI job. Modules that defer GPU imports (`DeviceSpec.from_device()`) work correctly without it.

**Tradeoff**: If a developer accidentally adds a top-level `import cuda` somewhere, the CPU CI will fail at import time with a clear `ModuleNotFoundError` — which is the correct signal.

### 3. ruff for both lint and format

`ruff check` (lint) and `ruff format --check` (format diff check) are run as separate steps to give granular failure messages. Format check uses `--check` (no-modify) mode so CI does not commit changes.

---

## Module Interaction with Existing src/

No source code changes are required to implement the CI pipeline. The workflows consume the existing `pyproject.toml` ruff config and the existing `pytest` test suite. The only new files are `.github/workflows/ci.yml` and `.github/workflows/gpu-ci.yml`.

---

## Consequences

### Positive
- Every push gets lint + CPU test feedback within ~2 minutes
- GPU tests are documented and runnable on self-hosted runners
- No infrastructure cost for the default (CPU-only) path
- Clear upgrade path to full GPU CI

### Negative / Trade-offs
- GPU tests are not validated automatically on every push
- Self-hosted runner setup is out of scope for Sprint 9; must be documented as a manual step

### Risks
- If `pyproject.toml` ruff config evolves, CI must stay in sync
- Python version matrix (3.11 only for now) may need to expand if compatibility is required
- `cuda-python` version pinning: the GPU CI job should pin the `cuda-python` version to match what is tested locally

---

## Testing Strategy

The CI pipeline itself is tested by pushing a branch and observing the GitHub Actions run. No unit tests for the CI config itself are needed. A `README.md` badge linking to the CI status is a Sprint 9 deliverable.

---

## Implementation Notes for Programmer

1. Create `.github/workflows/` directory
2. Create `ci.yml` using the template above — no GPU dependencies
3. Create `gpu-ci.yml` using the template above — manual trigger only
4. Verify `pyproject.toml` has ruff config (it already does per Sprint 1)
5. Commit and push; check that GitHub Actions runs the `CI` workflow automatically
6. Confirm `pytest tests/ -m "not gpu"` passes in CI (72 CPU tests)

---

## ADRs

- ADR: CPU/GPU split CI avoids blocking every push on unavailable GPU hardware
