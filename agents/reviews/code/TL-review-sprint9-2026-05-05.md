---
review: TL-review-sprint9-2026-05-05
agent: Tech Lead
date: 2026-05-05
sprint: 9
commit: acd39e1
---

# Tech Lead Code Review — Sprint 9
Date: 2026-05-05
Sprint: 9 — CI/CD Pipeline + Jupyter Notebooks
Reviewer: Tech Lead

## Verdict: Conditional Approval

## Summary

Sprint 9 delivers functional CI/CD workflows and two working Jupyter notebooks.
The CI/CD implementation matches ARCH-006 faithfully. The notebooks deviate from
ARCH-007's import strategy, but the deviation is accepted (see below) because the
`run_demo()` / `run_cpu_baseline()` / `run_gpu_version()` interface mandated by
REQ-0013-F5 and ARCH-007 was never implemented in the demos/ modules — the notebooks
could not have followed the spec. Two P1 items are not delivered: no CI badge in
README.md (REQ-0012-F5) and no notebooks/README.md (REQ-0013-F8). These are tracked
as minor findings and deferred to Sprint 10.

---

## CI/CD Workflows (REQ-0012, ARCH-006)

### ci.yml

**Positive:**
- Triggers on both `push` and `pull_request` — satisfies REQ-0012-F1 and AC-1.
- Lint and CPU test are separate jobs (`lint`, `test-cpu`) — satisfies REQ-0012-F7.
  A lint failure will not suppress test results.
- `cuda-python` is intentionally absent from `test-cpu` dependencies — satisfies
  REQ-0012-F2 and ARCH-006 design decision 2. The inline comment explaining the
  rationale is a good documentation practice.
- Python 3.11 used in both jobs — satisfies REQ-0012-F6.
- `ruff format --check` runs as a distinct step from `ruff check` — satisfies
  ARCH-006 design decision 3; granular failure messages.
- `PYTHONPATH=.` set on the pytest invocation — correct; required since tests
  import from `src/` using repo-relative paths.
- `ruff>=0.4` pinned — matches REQ-0012 notes on avoiding format drift.

**Findings:**
- m-1: REQ-0012-F5 (P1) — No CI status badge added to README.md. The existing
  README.md is the agentic starter template, not the project README, but the
  requirement is unambiguous. Deferred to Sprint 10.

### gpu-ci.yml

**Positive:**
- `workflow_dispatch` trigger only (no push trigger by default) — satisfies
  REQ-0012-F3 and ARCH-006 design.
- `runs-on: [self-hosted, gpu]` — satisfies REQ-0012-F4.
- Self-hosted runner setup requirements documented in inline comments (GPU,
  CUDA 12.x, runner registration, labels) — satisfies REQ-0012-NF3.
- `PYTHONPATH=. pytest tests/ -v` runs full suite including GPU tests — correct.
- Python 3.11 — satisfies REQ-0012-F6.

**Findings:**
- m-2: gpu-ci.yml installs `cuda-python>=12.3 numpy pytest ruff` but omits
  `cupy` and `torch` that ARCH-006 showed in the GPU CI install step. Tests
  in `test_interop.py` and `test_cupy_variants.py` may require those packages.
  Low risk since gpu-ci.yml is manual-trigger only and the runner admin can
  amend, but it should be addressed before the self-hosted runner is registered.

---

## Jupyter Notebooks (REQ-0013, ARCH-007)

### 01_core_apis.ipynb

**Positive:**
- 13 cells (exceeds the minimum 8 required by REQ-0013-AC3).
- Outputs are cleared — satisfies REQ-0013-F7 / AC-5.
- `sys.path.insert(0, "..")` present — notebooks run from `notebooks/` dir correctly.
- GPU detection cell uses `nvidia-smi` subprocess + `GPU_AVAILABLE` flag —
  satisfies REQ-0013-F4 and ARCH-007 design decision 2.
- GPU cell (cell 11) is guarded with `if GPU_AVAILABLE:` and marked
  `# REQUIRES GPU` — will not hard-error on CPU-only machines.
- All CPU cells (DeviceSpec, OccupancyModel, RooflineModel, GPU-generation
  comparison) run without any GPU or cuda-python — satisfies REQ-0013-F6.
- Rich analytical content: occupancy sweep, roofline analysis, multi-GPU
  comparison — high educational value.
- Cells are top-to-bottom executable with no hidden state — satisfies REQ-0013-NF3.

**Findings:**
- M-1 (Architecture deviation — see dedicated section below): Notebook imports
  from `src.kernel_model` rather than `demos.01_core_apis`. REQ-0013-F5 requires
  importing from demos/ modules. This is evaluated as acceptable; see below.
- m-3: The GPU cell (cell 11) does not execute any live GPU code — it prints CLI
  instructions (`python demos/01_core_apis/main.py`). This partly satisfies
  REQ-0013 cell structure (GPU version cell) but provides no interactive GPU
  timing output in Jupyter even when a GPU is present. Consider making this cell
  actually call `vector_add.run_vector_add()` from `demos/01_core_apis/vector_add.py`
  when GPU_AVAILABLE is True. Non-blocking for Sprint 9 given the root cause
  (no standardised run_gpu_version() function exists), but should be addressed
  in a future sprint.

### 02_kmeans.ipynb

**Positive:**
- 11 cells (exceeds minimum 8).
- Outputs are cleared — satisfies REQ-0013-F7 / AC-5.
- `sys.path.insert(0, "..")` present in parameter setup cell (cell 3).
- GPU detection cell uses `nvidia-smi` + `GPU_AVAILABLE` — satisfies REQ-0013-F4.
- CPU baseline (cell 5) always runs: calls `kmeans_cpu()` and times it —
  satisfies the "CPU baseline cell" requirement of REQ-0013-F3.
- GPU cell (cell 7) is guarded with `if GPU_AVAILABLE:` and calls `kmeans_gpu()`
  — provides actual GPU execution and timing when GPU is present.
- Comparison cell (cell 8) prints speedup when GPU ran, or a graceful skip
  message otherwise — satisfies the "results comparison cell" requirement.
- CUDA kernel design walkthrough in markdown (cell 6) is high-quality narrative.
- `importlib.util.spec_from_file_location` used correctly to load both
  `cpu_kmeans.py` and `gpu_kmeans.py` — avoids the digit-prefix import collision.
- Cells are top-to-bottom executable — satisfies REQ-0013-NF3.

**Findings:**
- M-2 (Architecture deviation — see dedicated section below): Uses `importlib.util`
  to load functions from file paths rather than calling a standardised
  `run_cpu_baseline()` / `run_gpu_version()` interface. Evaluated as acceptable.
- m-4: `BenchmarkResult` dataclass (defined in `src/utils/timing.py`) is not used
  in the comparison cell; the notebook rolls its own timing with `time.perf_counter`
  and prints fields manually. Not a correctness issue, but inconsistent with the
  project's established `BenchmarkResult` pattern. Minor style note.

---

## Architecture Deviation: ARCH-007 Import Strategy

### What ARCH-007 specified

ARCH-007 and REQ-0013-F5 required notebooks to import and call:
```python
from demos.demo_01_core_apis import run_demo, run_cpu_baseline, run_gpu_version
```

### What was implemented

- `01_core_apis.ipynb` imports from `src.kernel_model` (DeviceSpec, OccupancyModel,
  RooflineModel).
- `02_kmeans.ipynb` loads `demos/02_kmeans/cpu_kmeans.py` and `gpu_kmeans.py` via
  `importlib.util.spec_from_file_location`.

### Evaluation: Deviation is Acceptable

**Root cause**: The `run_demo()` / `run_cpu_baseline()` / `run_gpu_version()` interface
specified in ARCH-007 / REQ-0013 / CLAUDE.md API Structure section was never
implemented in the demos/ modules. A grep across all demos/ confirms zero definitions
of these functions. The demos expose module-level functions (`kmeans_cpu`, `kmeans_gpu`,
`vector_add.run_vector_add`, etc.) but not the standardised wrapper interface.
The Programmer could not follow the spec because the interface did not exist.

**For 02_kmeans.ipynb**: The `importlib.util` workaround for the digit-prefix directory
name is the correct Python approach. It directly loads the implementation functions
without code duplication, satisfying REQ-0013's "no code duplication" intent.

**For 01_core_apis.ipynb**: Importing from `src.kernel_model` rather than `demos/`
is a layer violation relative to ARCH-007, but it produces a better educational
notebook — the CPU-only analytical content (occupancy, roofline) is richer than
anything a `run_demo()` wrapper would provide for demo 01, which is primarily a
GPU-kernel demo with minimal CPU-only content.

**Action required (non-blocking)**: The CLAUDE.md API structure section and ARCH-007
should be updated to reflect the actual demo module interface (direct function imports
rather than `run_demo` wrappers), and the demo interface contract should either be
implemented or the spec retracted. Track as Sprint 10 follow-on (see below).

---

## Findings Summary

| ID | Severity | File | Finding |
|----|----------|------|---------|
| M-1 | Major | `notebooks/01_core_apis.ipynb` | Imports from `src.kernel_model` rather than `demos/` (ARCH-007 deviation). Accepted: root cause is missing `run_demo()` interface in demos/. |
| M-2 | Major | `notebooks/02_kmeans.ipynb` | Uses `importlib.util` file-path loading rather than package import (ARCH-007 deviation). Accepted: required workaround for digit-prefixed directory name; interface functions don't exist. |
| m-1 | Minor | `README.md` | CI status badge missing (REQ-0012-F5, P1). Deferred to Sprint 10. |
| m-2 | Minor | `.github/workflows/gpu-ci.yml` | `cupy` and `torch` not installed; GPU interop tests may fail on self-hosted runner. |
| m-3 | Minor | `notebooks/01_core_apis.ipynb` | GPU cell prints CLI instructions rather than executing live GPU code. Limited interactivity for GPU users. |
| m-4 | Minor | `notebooks/02_kmeans.ipynb` | Timing uses raw `time.perf_counter` instead of project's `BenchmarkResult` dataclass. |

**Totals: 0 critical, 2 major (both accepted deviations), 4 minor**

---

## Definition of Done Checklist

- [x] CI workflow triggers on push/PR
- [x] CPU tests run without cuda-python installed
- [x] gpu-ci.yml is manual-trigger with self-hosted runner config
- [x] Both notebooks have cleared outputs
- [x] Both notebooks have GPU-guarded cells (`if GPU_AVAILABLE:`)
- [x] sys.path set correctly so imports work from notebooks/ directory
- [ ] CI status badge in README.md (REQ-0012-F5 — deferred Sprint 10)
- [ ] notebooks/README.md or README section for notebook usage (REQ-0013-F8 — deferred Sprint 10)

---

## Follow-on Items (non-blocking, P2)

1. **Sprint 10 / README**: Add CI badge to project README.md; add notebooks/ usage
   section. Closes REQ-0012-F5 and REQ-0013-F8.
2. **Sprint 10 / gpu-ci.yml**: Add `cupy` and `torch` to GPU CI pip install to
   prevent interop test failures on self-hosted runner.
3. **Backlog / ARCH-007**: Retract or revise the `run_demo()` / `run_cpu_baseline()` /
   `run_gpu_version()` interface contract in ARCH-007 and CLAUDE.md API structure
   section to match the actual per-module function exports. Alternatively, implement
   the wrappers in demos/01 and demos/02 so the spec becomes real.
4. **Backlog / 01_core_apis.ipynb**: When the GPU interface is clarified, make the
   GPU cell execute live code (e.g. call `vector_add.run_vector_add()`) rather than
   just printing CLI instructions.
5. **Backlog / Notebooks 03-08**: Extend notebook coverage to remaining demos
   (REQ-0013 P2 backlog item already tracked in todo.md).
