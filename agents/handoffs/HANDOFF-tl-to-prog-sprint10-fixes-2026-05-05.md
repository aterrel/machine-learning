# Handoff: Tech Lead → Programmer
Sprint: 10 — README Fix (C-1)
Date: 2026-05-05
From: Tech Lead
To: Programmer

## Context

TL code review for Sprint 10 found one Critical defect in README.md (verdict: Changes Required). This handoff describes the exact fix required.

---

## Fix Required

### C-1: Broken `python -m demos.run_all` command

**File**: `README.md`
**Location**: "Full GPU Path" quickstart block (~lines 48–55)

**Problem**: The command `python -m demos.run_all` fails with:
```
/usr/bin/python3: No module named demos.run_all
```
`demos/run_all.py` does not exist. `demos/__init__.py` also does not exist. The run-all entry point has always been `benchmarks/run_all.py` (present since Sprint 1).

**Fix**: Replace the broken command with the correct one.

In README.md, find:
```bash
# Run all demos
python -m demos.run_all
```

Replace with:
```bash
# Run all demos
python benchmarks/run_all.py
```

---

## What Is NOT Broken (do not change)

- `python -m demos.09_kernel_model.main` — works correctly (runpy filesystem traversal)
- `python -m demos.10_ptx_tracer.main` — works correctly
- `python -m demos.01_core_apis.main` — correct syntax; fails only because numpy not installed in test env
- All other `-m demos.NN_name.main` single-demo commands — correct pattern, do not change
- `docs/slides/` link — valid, directory exists
- `notebooks/*.ipynb` links — valid, files exist
- `gpu-ci.yml` — m-2 fix accepted, no changes needed

---

## Acceptance Criteria for Fix

After applying the fix, verify:
```bash
python benchmarks/run_all.py --help 2>&1 | head -5
```
Should print the usage/help output (not a ModuleNotFoundError).

---

## Files to Change

1. `README.md` — one-line fix in the Quickstart section

Commit prefix: `[Prog]`
Suggested message: `[Prog] Fix README.md C-1: replace broken python -m demos.run_all with python benchmarks/run_all.py`
