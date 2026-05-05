# Tech Lead Code Review — Sprint 10
Date: 2026-05-05
Sprint: 10 — README + Physical GPU Validation
Reviewer: Tech Lead

## Verdict: Approved

## Summary

The README.md covers all REQ-0014 functional requirements and renders cleanly, but the "Run all demos" quickstart command (`python -m demos.run_all`) is broken — `demos/run_all.py` does not exist and `demos/` has no `__init__.py`, so the command fails with "No module named demos.run_all". This is a user-visible breakage on the very first GPU quickstart anyone would try. The gpu-ci.yml m-2 fix is correct.

---

## README.md (REQ-0014)

### Correctness

**C-1 (Critical) — Broken `python -m demos.run_all` command**

The "Full GPU Path" quickstart block contains:
```bash
# Run all demos
python -m demos.run_all
```
This command fails:
```
/usr/bin/python3: No module named demos.run_all
```
`demos/run_all.py` does not exist (confirmed: `ls demos/run_all.py` → missing). `demos/__init__.py` also does not exist. The correct script is `benchmarks/run_all.py`, which is the run-all entry point documented since Sprint 1 (`python benchmarks/run_all.py`).

**N-1 (Note) — `-m` form with digit-prefixed directories**

The instructions flagged a suspected issue with `python -m demos.09_kernel_model.main` due to digit-prefixed directory names not being valid Python identifiers. Testing confirms this is NOT broken: Python's `-m` flag uses `runpy`'s filesystem-based module lookup, which does not require valid identifier names. Commands verified to work:
- `python3 -m demos.09_kernel_model.main` — SUCCESS (CPU-only, no GPU needed)
- `python3 -m demos.10_ptx_tracer.main` — SUCCESS (CPU-only, no GPU needed)
- `python3 -m demos.01_core_apis.main` — Fails with `ModuleNotFoundError: No module named 'numpy'` (expected: numpy not installed in this environment, not a README defect)

The `-m` quickstart pattern is valid and correct for all digit-prefixed demos.

### Content Coverage (REQ-0014-F1 through F8)

| ID | Requirement | Status |
|----|-------------|--------|
| REQ-0014-F1 | README at root: title, description, prerequisites, install, quickstart, structure, demo list | PASS |
| REQ-0014-F2 | Prerequisites: Python 3.11+, NVIDIA GPU, CUDA 12.x, cuda-python, numpy | PASS |
| REQ-0014-F3 | CPU-only mode note (demos 09+10 run without GPU) | PASS — present in Prerequisites and Quickstart |
| REQ-0014-F4 | Demo Overview table: all 10 demos (01–10) with description and GPU requirement | PASS |
| REQ-0014-F5 | CI status badge at top | PASS — badge present on line 3 |
| REQ-0014-F6 | Link to `docs/slides/` | PASS — link present in Slides section; directory exists |
| REQ-0014-F7 | Running Tests section with `pytest tests/ -m "not gpu"` and GPU test commands | PASS |
| REQ-0014-F8 | Development section with lint/format commands | PASS |

REQ-0014-F9 (GPU SKU Reference) — P2, out of scope for this sprint.

Note on Demo Table: The table uses `05a` and `05b` rows for the two Sprint 2/4 demos (`05_kernels` and `05_naive_bayes`). This is acceptable and clearer than having two `05` rows.

### Non-Functional

**REQ-0014-NF1 (GitHub rendering)**: Markdown tables, code blocks, and badge all use standard syntax — renders correctly.

**REQ-0014-NF2 (Line limit)**: README is 156 lines. Passes the 300-line ceiling.

**REQ-0014-NF3 (Link validity)**:
- `notebooks/01_core_apis.ipynb` — EXISTS (confirmed)
- `notebooks/02_kmeans.ipynb` — EXISTS (confirmed)
- `docs/slides/` — EXISTS (confirmed: contains 10 subdirectories + README.md)
- External link `https://github.com/nvidia/cuda-python` — stable upstream URL, acceptable
- Badge URL `https://github.com/aterrel/agentic-project-starter/actions/workflows/ci.yml` — valid reference to the actual repo

AC-2 from REQ-0014: The quickstart example `python demos/09_kernel_model/main.py` appears in REQ-0014's interface design but the README uses the `-m` form instead. The `-m` form works correctly (verified above). Minor divergence from the acceptance criterion wording, but functionally equivalent and arguably cleaner for users who don't want to manage PYTHONPATH.

---

## gpu-ci.yml (Sprint 9 m-2 fix)

The m-2 finding from Sprint 9 TL review required adding `cupy` and `torch` to the `gpu-ci.yml` pip install step. Confirmed fix:

```yaml
run: pip install "cuda-python>=12.3" "numpy>=1.26" "pytest>=8.0" "ruff>=0.4" "cupy-cuda12x>=12.0" "torch>=2.0"
```

Both `cupy-cuda12x>=12.0` and `torch>=2.0` are now present. The m-2 finding is resolved.

---

## Findings Summary

| ID | Severity | File | Finding |
|----|----------|------|---------|
| C-1 | Critical | README.md | `python -m demos.run_all` fails — file does not exist; correct command is `python benchmarks/run_all.py` |
| N-1 | Note | README.md | `-m` form with digit-prefixed directories (e.g., `demos.09_kernel_model.main`) is confirmed working — not a defect |

---

## Required Changes

### Fix C-1: Replace broken `python -m demos.run_all` with correct command

In README.md, the "Full GPU Path" quickstart block (lines 48–55):

**Current (broken):**
```bash
# Run all demos
python -m demos.run_all
```

**Required (working):**
```bash
# Run all demos
python benchmarks/run_all.py
```

No other changes required. The single-demo commands (`python -m demos.01_core_apis.main`, etc.) are correct and functional.

---

## Definition of Done Checklist

- [x] README.md at project root, renders correctly on GitHub
- [x] Demo Overview table lists all 10 demos
- [x] All relative links resolve to existing files
- [x] Quickstart commands are executable (digit-prefixed `-m` form verified working)
- [x] `python benchmarks/run_all.py` — FIXED (C-1 resolved)
- [x] CI badge present
- [x] README under 300 lines (156 lines)
- [x] gpu-ci.yml has cupy and torch in pip install

---

## Re-Review: C-1 Fix (commit 24dbfd9)
Date: 2026-05-05

Fix verified: `python -m demos.run_all` replaced with `python benchmarks/run_all.py`.
`benchmarks/run_all.py` confirmed to exist. No new issues introduced.

**Final Verdict: Approved**
