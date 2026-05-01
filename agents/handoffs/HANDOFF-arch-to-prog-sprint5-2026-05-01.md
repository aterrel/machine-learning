---
from: Software Architect
to: Programmer
date: 2026-05-01
sprint: 5
---

# Handoff: Sprint 5 Implementation

## What I Did
Created ARCH-003 defining the multi-backend design pattern for REQ-0007 and REQ-0008.

## What You Need to Do

Read ARCH-003 before starting. Key decisions:
- Side-by-side files: `numba_*.py` and `cupy_*.py` alongside existing `gpu_*.py`
- All variants: numpy-in / numpy-out, same return signatures as existing functions
- Optional import guards with `_NUMBA_AVAILABLE` / `_CUPY_AVAILABLE` sentinels
- Kernels defined inside the `try: import numba` block so file is importable when absent
- `BenchmarkRunner(warmup=1)` handles JIT warmup — no external warmup needed
- Comparison demo calls existing functions via importlib — NO re-implementation

## Implementation Order (P0 first)

### Phase 1 — Core algorithm variants (REQ-0007)
1. demos/01_core_apis/numba_vector_add.py  — `run_vector_add_numba(n)`
2. demos/01_core_apis/cupy_vector_add.py   — `run_vector_add_cupy(n)`
3. demos/02_kmeans/numba_kmeans.py         — `kmeans_numba(X, k, seed)` → (centroids, labels, inertia)
4. demos/02_kmeans/cupy_kmeans.py          — `kmeans_cupy(X, k, seed)` → same
5. demos/03_pca/numba_pca.py               — `pca_numba(X, n_components)` → (components, expl_var, X_transformed)
6. demos/03_pca/cupy_pca.py               — `pca_cupy(X, n_components)` → same
7. demos/04_linear_model/numba_linear.py  — `linear_regression_numba(X, y)` → (weights, r2, mse)
8. demos/04_linear_model/cupy_linear.py  — `linear_regression_cupy(X, y)` → same
9. demos/05_naive_bayes/numba_nb.py       — `gaussian_nb_numba(X_tr, y_tr, X_te, n_classes)` → (preds, means, vars, log_priors)
10. demos/05_naive_bayes/cupy_nb.py       — `gaussian_nb_cupy(...)` → same

### Phase 2 — Benchmark runner update (REQ-0007)
11. benchmarks/run_all.py — add --backend flag, comparison table for --backend all

### Phase 3 — Comparison demo (REQ-0008)
12. demos/08_comparison/__init__.py
13. demos/08_comparison/main.py
14. demos/08_comparison/backends/__init__.py
15. demos/08_comparison/backends/cuml_backends.py

### Phase 4 — P1 kernel variants (REQ-0007)
16. demos/05_kernels/numba_kernels.py
17. demos/05_kernels/cupy_kernels.py

## Return Signature Reference

Match existing signatures exactly:
- kmeans: `(centroids: np.ndarray, labels: np.ndarray, inertia: float)`
- pca: `(components: np.ndarray, explained_variance: np.ndarray, X_transformed: np.ndarray)`
- linear: `(weights: np.ndarray, r2_score: float, mse: float)`
- naive_bayes: `(predictions: np.ndarray, means: np.ndarray, vars_: np.ndarray, log_priors: np.ndarray)`

## Commit Prefix
Use `[Prog]` for all implementation commits.
