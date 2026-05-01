---
reviewer: Tech Lead
sprint: 5
date: 2026-05-01
verdict: Conditional Approval
---

# Tech Lead Code Review — Sprint 5

## Commits Reviewed
- `8743952` — [Prog] Sprint 5: Numba+CuPy variants for all demos, comparison demo, benchmark --backend flag
- `b5a93bd` — [QA] Sprint 5: test_numba_variants, test_cupy_variants, test_comparison (33 CPU tests)

## Files Reviewed
- `demos/01_core_apis/numba_vector_add.py`, `cupy_vector_add.py`
- `demos/02_kmeans/numba_kmeans.py`, `cupy_kmeans.py`
- `demos/03_pca/numba_pca.py`, `cupy_pca.py`
- `demos/04_linear_model/numba_linear.py`, `cupy_linear.py`
- `demos/05_naive_bayes/numba_nb.py`, `cupy_nb.py`
- `demos/05_kernels/numba_kernels.py`, `cupy_kernels.py`
- `demos/08_comparison/main.py`, `backends/cuml_backends.py`
- `benchmarks/run_all.py`
- `tests/test_numba_variants.py`, `test_cupy_variants.py`, `test_comparison.py`

## Test Results
- 72 CPU tests pass (`pytest tests/ -m "not gpu"`) — 39 existing + 33 new
- 26 GPU tests deselected (require hardware)
- Zero regressions against Sprint 4 baseline

## Strengths
1. Consistent import guard pattern (`_NUMBA_AVAILABLE`, `_CUPY_AVAILABLE`) across all 12 variant files — any file is importable even without the backend
2. Numba kernels correctly use 1D flat indexing for 2D work (avoids confusion with 2D grid tuples)
3. `nb_cuda.atomic.add(arr, (row, col), val)` correct for 2D atomics in k-means accumulation
4. CuPy k-means uses the `||x||^2 - 2x·c^T + ||c||^2` identity to avoid the memory-intensive (n, k, d) intermediate
5. Comparison demo correctly catches both `ImportError` and `SystemExit` per backend — robust to missing backends including cuda-python itself
6. `_print_baseline_row` vs `_print_table_row` separation is clean
7. All new tests are CPU-safe and use `_AVAIL` flag patching rather than heavy mocking

## Issues

### TL-S5-001 (Minor — Nit) — `benchmarks/run_all.py` single-backend dispatch for numba/cupy is fragile
The `elif args.backend in ("numba", "cupy")` block builds module names with string concatenation (`f"{pkg}.{suffix}_{demo_key.split('_', 1)[-1]}"`) which is error-prone and has dead code (the intermediate `mod_name` assignment is unused). The `--backend all` comparison functions handle this cleanly. For the single-backend path, just call the same comparison helpers with a filtered backend list.

**Fix**: Remove the fragile string-building dispatch in `main()`. Instead, route `--backend numba` and `--backend cupy` through the comparison helper functions, passing only the relevant backend row.

### TL-S5-002 (Minor — Nit) — `cupy_kmeans.py` loop calls `cp.mean` inside Python `for j in range(k)` 
The centroid update loop iterates over clusters in Python, calling `cp.mean(X_gpu[mask], axis=0)` per cluster. For large `k` this creates many small GPU kernels and synchronization points. The cuda-python variant does the same (accumulate via atomicAdd then normalize on CPU), but the CuPy variant could do this more idiomatically with `cp.zeros` + scatter-add.

This is a P2 performance note only — the current implementation is correct and readable. No fix required for Conditional Approval.

### TL-S5-003 (Minor — Nit) — `test_numba_variants.py::test_numba_vector_add_raises_when_absent` uses a convoluted approach
The test creates a `type(mod)(mod.__name__)` copy and patches `_NUMBA_AVAILABLE = False` on it. This doesn't actually test the real guard because `_NUMBA_AVAILABLE` is checked inside the function in the *original* module, not the copy. The test passes only coincidentally. Fix by patching `mod._NUMBA_AVAILABLE` directly (same pattern as the other tests in the file).

## Required Fixes Before Close
- TL-S5-001: Fix the fragile `--backend numba/cupy` dispatch in `benchmarks/run_all.py`
- TL-S5-003: Fix the vector_add absent-test to patch `mod._NUMBA_AVAILABLE` directly

## Verdict: **Conditional Approval**
Two nit fixes required (TL-S5-001, TL-S5-003). TL-S5-002 is a P2 performance note only. Once fixes are committed, sprint may close.
