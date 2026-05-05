# ARCH-007: Jupyter Notebook Coverage for Demos 01 and 02

**Status**: Draft
**Created**: 2026-05-05
**Author**: Software Architect
**REQ Reference**: REQ-0013
**Approved By**: —

---

## Overview

This document defines the architecture for Jupyter notebook versions of demos 01 (`01_core_apis`) and 02 (`02_kmeans`). The notebooks are thin wrappers around the existing `demos/` Python modules — they import and call the existing `run_demo()` functions and add markdown narrative cells explaining the CUDA Python concepts. No new library code is introduced.

---

## Context

The project currently has Python scripts for all 8 demo directories (`demos/01_core_apis/` through `demos/08_comparison/`). Learners using Jupyter expect an interactive, cell-by-cell experience where they can modify parameters (e.g., `n_samples`, `block_size`) and re-run. The backlog item is scoped to demos 01 and 02 as the most foundational, with the remaining demos as a P2 follow-on.

The `notebooks/` directory exists in the project structure (per CLAUDE.md) but is currently empty.

---

## Design

### Module Structure

```
notebooks/
  01_core_apis.ipynb     # Notebook: CUDA Python core API walkthrough
  02_kmeans.ipynb        # Notebook: GPU k-means clustering demo
```

### Notebook Architecture

Each notebook follows a consistent structure:

```
Cell 1 (Markdown): Title + learning objectives
Cell 2 (Code):     pip install check + imports
Cell 3 (Markdown): Concept introduction (what is CUDA Python / what is k-means on GPU)
Cell 4 (Code):     Parameter setup (n_samples, seed, block_size — user-editable)
Cell 5 (Markdown): Step walkthrough narrative
Cell 6 (Code):     run_cpu_baseline() call + timing output
Cell 7 (Code):     run_gpu_version() call + timing output  [requires GPU; cell marked]
Cell 8 (Code):     BenchmarkResult comparison + speedup
Cell 9 (Markdown): Explanation of results + next steps
```

### GPU-Conditional Cells

Cells that require a GPU are marked with a comment `# REQUIRES GPU` at the top. The notebook includes a check cell:

```python
import subprocess
try:
    subprocess.run(["nvidia-smi"], check=True, capture_output=True)
    GPU_AVAILABLE = True
except (FileNotFoundError, subprocess.CalledProcessError):
    GPU_AVAILABLE = False
    print("No GPU detected — GPU cells will be skipped.")
```

GPU cells use `if GPU_AVAILABLE:` guards to avoid hard failures when run on CPU-only machines.

### Relation to Existing Code

Notebooks import directly from `demos/`:

```python
# In 01_core_apis.ipynb
import sys
sys.path.insert(0, "..")   # repo root
from demos.demo_01_core_apis import run_demo, run_cpu_baseline, run_gpu_version
```

This ensures the notebook stays in sync with the demo module automatically — no code duplication.

---

## Key Design Decisions

### 1. Notebooks call existing demo modules, not re-implement

Re-implementing logic in notebooks would create maintenance burden (two copies of the same code). The notebook is a presentation layer only.

**Tradeoff**: The user must have the full repo checked out for the `sys.path` import to work. Standalone notebook distribution is out of scope.

### 2. GPU-conditional cells, not GPU-required notebooks

A notebook that hard-errors on the first GPU cell discourages users without hardware. GPU cells are guarded so the narrative and CPU baseline always render correctly.

**Tradeoff**: The speedup comparison cell is only populated if GPU is available; otherwise it prints "N/A — no GPU detected."

### 3. Scoped to demos 01 and 02 only (Sprint 9)

Demos 03-08 are lower priority and structurally similar. Delivering 01 and 02 first establishes the pattern; the remaining notebooks are a P2 backlog item.

---

## Consequences

### Positive
- Interactive learner experience for the two most foundational demos
- No new library code; notebooks are thin wrappers
- Pattern established for future notebook coverage of demos 03-08

### Negative / Trade-offs
- Notebooks require `jupyter` to be installed (already in CLAUDE.md dev dependencies)
- `sys.path` hack is not elegant but is the standard approach for repo-relative notebook imports

### Risks
- Kernel restart resets all state — users may be confused if they run cells out of order; address with cell ordering warnings in markdown
- `.ipynb` files contain output JSON which can grow large if re-run repeatedly — document `jupyter nbconvert --clear-output` in README

---

## Testing Strategy

Notebooks are tested using `nbconvert --execute`:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_core_apis.ipynb --output /tmp/01_test.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_kmeans.ipynb --output /tmp/02_test.ipynb
```

This executes every cell and fails if any cell raises an exception. GPU cells are guarded with `if GPU_AVAILABLE:` so they do not fail on CPU-only CI.

A `pytest` integration test using `nbval` or `testbook` is a P2 enhancement.

---

## Implementation Notes for Programmer

1. Ensure `notebooks/` directory exists (already in project structure)
2. Create `notebooks/01_core_apis.ipynb` — follow cell structure above
3. Create `notebooks/02_kmeans.ipynb` — follow cell structure above
4. Test with `jupyter nbconvert --execute` on a CPU-only machine first
5. Clear output before committing: `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`
6. Verify notebooks render correctly on GitHub (GitHub renders `.ipynb` natively)

---

## ADRs

- ADR: Notebooks as presentation wrappers, not independent implementations, ensures single source of truth for demo logic
