# ARCH-003: Multi-Backend Demo Pattern (Numba-CUDA + CuPy + cuML)

**Status**: Approved
**Created**: 2026-05-01
**Author**: Software Architect
**REQ Reference**: REQ-0007, REQ-0008
**Approved By**: Software Architect (2026-05-01)

---

## Overview

This document defines the design for adding Numba-CUDA and CuPy variants to the existing algorithm demos, and for the new `demos/08_comparison/` unified comparison demo with cuML support. The goal is a four-way comparison: NumPy CPU / cuda-python / Numba-CUDA / CuPy — plus cuML as an optional fifth column.

---

## Design Decisions

### 1. Side-by-Side Files (not subdirectories)

Each demo directory gets `numba_*.py` and `cupy_*.py` files alongside the existing `cpu_*.py` and `gpu_*.py`:

```
demos/02_kmeans/
  cpu_kmeans.py       # existing NumPy reference
  gpu_kmeans.py       # existing cuda-python
  numba_kmeans.py     # new: Numba-CUDA
  cupy_kmeans.py      # new: CuPy
```

**Rationale**: The side-by-side pattern is the most pedagogically clear — a reader browsing any demo directory immediately sees all backend options. Subdirectories would require navigating deeper and break the existing flat-import pattern.

**Tradeoff**: Slightly more files per directory, but each is self-contained and independently runnable.

### 2. Uniform Interface Contract

All variants — existing and new — follow the same signature contract:

```python
# Accepts numpy arrays; returns numpy arrays.
# GPU computation is an internal detail.
def algorithm_numba(X: np.ndarray, **kwargs) -> tuple[np.ndarray, ...]: ...
def algorithm_cupy(X: np.ndarray, **kwargs) -> tuple[np.ndarray, ...]: ...
```

Return types match the existing `cpu_*` and `gpu_*` functions exactly so `benchmarks/run_all.py` and `demos/08_comparison/` can call any backend interchangeably.

### 3. Optional Import Guards

Each variant file guards its backend import at module level:

```python
try:
    from numba import cuda as nb_cuda
    import math
    _NUMBA_AVAILABLE = True
    
    @nb_cuda.jit
    def _kernel(...): ...
    
except ImportError:
    _NUMBA_AVAILABLE = False

def algorithm_numba(X, **kwargs):
    if not _NUMBA_AVAILABLE:
        raise ImportError("numba required: pip install numba")
    ...
```

This means the file is always importable, even without the backend installed. Callers check `_NUMBA_AVAILABLE` or catch `ImportError`.

### 4. JIT Warmup via BenchmarkRunner

Numba's `@cuda.jit` compiles kernels on first call (JIT). `BenchmarkRunner(warmup=1)` performs one warmup call before starting timed runs — this warmup call compiles the kernel. No external warmup call is needed.

CuPy's `cp.RawKernel` similarly JIT-compiles on first call; the same `BenchmarkRunner(warmup=1)` pattern handles this.

### 5. Benchmark Runner Extension

`benchmarks/run_all.py` gains a `--backend` flag:

```
--backend cuda-python   # existing behavior (default)
--backend numba         # run only Numba variants
--backend cupy          # run only CuPy variants
--backend all           # run all backends, print comparison table
```

For `--backend all`, the output format is a per-demo comparison table:

```
[02_kmeans]
  Backend          CPU(s)    GPU(s)   Speedup  Correct
  -------------------------------------------------------
  cuda-python       2.341     0.187    12.5x    True
  numba             2.341     0.231    10.1x    True
  cupy              2.341     0.204    11.5x    True
```

The CPU time column is the shared NumPy baseline time. Each row's GPU time is the backend's GPU time.

### 6. Comparison Demo (demos/08_comparison/)

`demos/08_comparison/main.py` calls into existing demo module functions via `importlib` — it does NOT re-implement algorithms. This ensures the comparison reflects exactly the same code users can inspect in individual demo directories.

```
demos/08_comparison/
  __init__.py
  main.py                   # CLI: --algorithm, --n-samples; prints per-algorithm tables
  backends/
    __init__.py
    cuml_backends.py        # cuML wrappers (try/except guarded)
```

The cuML wrappers follow the same numpy-in / numpy-out contract. If RAPIDS is installed, cuML results appear in the table; otherwise that row shows `SKIPPED (not installed)`.

### 7. cuML: numpy-in / numpy-out Wrappers

cuML accepts both numpy arrays and cudf DataFrames. The wrappers always accept numpy and convert internally, hiding the cudf complexity from callers:

```python
def kmeans_cuml(X: np.ndarray, k: int, seed: int = 42):
    import cuml.cluster
    model = cuml.cluster.KMeans(n_clusters=k, random_state=seed, output_type="numpy")
    model.fit(X)
    return np.asarray(model.cluster_centers_), np.asarray(model.labels_), float(model.inertia_)
```

`output_type="numpy"` (available in recent cuML versions) avoids cudf Series outputs. Fall back to `np.asarray()` on older versions.

---

## Module Structure (Sprint 5 additions)

```
demos/
  01_core_apis/
    numba_vector_add.py     # @cuda.jit vector add + run_vector_add_numba()
    cupy_vector_add.py      # cp.RawKernel vector add + run_vector_add_cupy()
  02_kmeans/
    numba_kmeans.py         # @cuda.jit Lloyd's + kmeans_numba()
    cupy_kmeans.py          # cp broadcast distances + kmeans_cupy()
  03_pca/
    numba_pca.py            # @cuda.jit mean-center + cov + pca_numba()
    cupy_pca.py             # cp.linalg.eigh + pca_cupy()
  04_linear_model/
    numba_linear.py         # @cuda.jit XtX/Xty + linear_regression_numba()
    cupy_linear.py          # cp.linalg.solve + linear_regression_cupy()
  05_kernels/
    numba_kernels.py        # @cuda.jit GEMM + ReLU (P1)
    cupy_kernels.py         # cp.matmul + cp.RawKernel ReLU (P1)
  05_naive_bayes/
    numba_nb.py             # @cuda.jit log-likelihood + argmax + gaussian_nb_numba()
    cupy_nb.py              # cp broadcast log-likelihood + gaussian_nb_cupy()
  08_comparison/
    __init__.py
    main.py                 # unified comparison CLI
    backends/
      __init__.py
      cuml_backends.py      # cuML wrappers (all 4 algorithms)
benchmarks/
  run_all.py                # extended with --backend flag
tests/
  test_numba_variants.py    # CPU-safe: import check + graceful skip
  test_cupy_variants.py     # CPU-safe: import check + graceful skip
  test_comparison.py        # CPU-safe: SKIPPED rows when backends absent
```

---

## Numba Kernel Design

### Thread Indexing Convention
All Numba kernels use 1D grids with flat indexing for simplicity:

```python
@nb_cuda.jit
def _kernel(args...):
    i = nb_cuda.grid(1)          # flat 1D index
    total = some_dim
    if i >= total:
        return
    # work on element i
```

2D work (e.g., covariance matrix) is mapped to 1D: `i = tid // n_cols`, `j = tid % n_cols`.

### Atomic Operations
Use `nb_cuda.atomic.add(array, index, value)`:
- 1D array: `nb_cuda.atomic.add(arr, i, val)`
- 2D array: `nb_cuda.atomic.add(arr, (row, col), val)`

### Math Functions
Import `math` at module level; `math.log`, `math.sqrt`, `math.pi` are available inside `@cuda.jit` kernels.

---

## CuPy Design

### High-Level Ops (preferred)
Use cupy's NumPy-compatible API where possible:
- `cp.asarray(X)` — upload numpy array to GPU
- `cp.linalg.eigh`, `cp.linalg.solve` — LAPACK via cuSOLVER
- `cp.argmin`, `cp.argmax` — reduction ops via cuBLAS/custom
- `cp.asnumpy(arr)` — download to host

### RawKernel (for kernel demos)
For `demos/01_core_apis/` and `demos/05_kernels/`, use `cp.RawKernel` to show CUDA C kernel compilation via CuPy:

```python
_ADD_SRC = r"""
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""
kernel = cp.RawKernel(_ADD_SRC, "vector_add")
kernel((grid,), (block,), (d_a, d_b, d_c, n))
```

### Stream Sync
Before stopping wall-clock timers: `cp.cuda.Stream.null.synchronize()`.

---

## Testing Strategy

All new tests are CPU-safe:
- Check that variant files import without error even when backend is absent
- Verify correct `ImportError` raised when backend missing (mock `sys.modules`)
- Shape and dtype checks using small numpy arrays (mocked GPU path)
- `@pytest.mark.gpu` for tests that actually launch kernels

---

## Consequences

### Positive
- Users see the same algorithm at four abstraction levels in one place
- cuML column shows the "just use the library" option without any CUDA knowledge
- Benchmark runner comparison table is a single command

### Negative / Tradeoffs
- 12 new source files across demo directories
- RAPIDS installation for cuML is complex (conda/Docker recommended)
- Numba and CuPy are additional optional dependencies

### Risks
- Numba's CUDA backend version requirements must match installed CUDA Toolkit
- cuML `output_type="numpy"` may not exist in older RAPIDS versions; fallback with `np.asarray()` is required
