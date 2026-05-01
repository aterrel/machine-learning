# Handoff: Tech Lead → Programmer
**Date**: 2026-05-01
**From**: Tech Lead
**To**: Programmer
**Sprint**: 2 (post-review fixes)
**Review file**: `agents/reviews/code/TL-review-sprint2-2026-05-01.md`

---

## Context

Sprint 2 received **Conditional Approval**. All new algorithm implementations (PCA, linear
regression, GEMM, ReLU, softmax) are correct and memory-safe. Two items must be fixed before
benchmark work begins in Sprint 3, and two more are "should fix in Sprint 3" quality items.

---

## Must Fix (before Sprint 3 benchmark work)

### TL-S2-003 — benchmarks/run_all.py missing Sprint 2 demos

**File**: `benchmarks/run_all.py`

The benchmark runner was not updated to include the three Sprint 2 demos. Running
`python -m benchmarks.run_all` silently skips PCA, linear regression, and the custom kernels
demo.

**Required changes**:
1. Add `run_03_pca(n_samples)`, `run_04_linear_model(n_samples)`, and
   `run_05_kernels()` functions following the pattern of `run_01_core_apis()`.
2. Add `"03_pca"`, `"04_linear_model"`, `"05_kernels"` to the `--demo` argparse choices.
3. Call all three in `main()` when `args.demo in ("<demo>", "all")`.

For reference, `run_03_pca` should call `pca_cpu` and `pca_gpu` from their respective
modules, use `BenchmarkRunner.time_cpu` for both (since pca_gpu does its own stream
management), and compare eigenvectors using the sign-flip diagonal alignment check.

`run_05_kernels` can delegate to `run_gemm_demo()` and `run_activations_demo()` which already
return `BenchmarkResult`.

---

### TL-S2-001 — ReLU GPU timing includes H2D transfer

**File**: `demos/05_kernels/activations.py`, lines 119–123

Inside `relu_fn()` (the closure passed to `runner.time_gpu`), there is a `_h2d(...)` call
that uploads the input before the kernel launch:

```python
def relu_fn() -> None:
    kernel_x = x_cpu.copy()
    _h2d(d_x.handle, kernel_x, n_bytes)   # <-- this is timed
    relu_kernel.launch(...)                # <-- this is also timed
```

The `_h2d` call should be outside the timed closure. The intent of a ReLU kernel benchmark is
to measure kernel throughput, not transfer + kernel.

**Fix**: Restructure so `relu_fn` only contains the kernel launch. If correctness requires
fresh data each iteration (because ReLU is in-place and modifies the buffer), do the upload
as a setup step before the warmup, or re-upload in a pre-step hook. The simplest correct
approach:

```python
# Upload once before timing
_h2d(d_x.handle, x_cpu, n_bytes)

def relu_fn() -> None:
    # Re-upload original data so in-place ReLU gets fresh input each rep
    _h2d(d_x.handle, x_cpu, n_bytes)
    relu_kernel.launch(grid_relu, block_1d, [d_x.handle, np.int32(n)], stream=stream)
```

Wait — if you put `_h2d` inside `relu_fn` again, it's timed again. The cleanest approach for
an in-place kernel is to time only the kernel, not the setup:

```python
# Pre-upload once before warmup
_h2d(d_x.handle, x_cpu, n_bytes)

def relu_fn() -> None:
    relu_kernel.launch(grid_relu, block_1d, [d_x.handle, np.int32(n)], stream=stream)

gpu_time = runner.time_gpu(relu_fn, stream)
```

Accept that the buffer gets modified across timing reps (idempotent after first call since
all-negative values remain zero). Then do a clean re-upload and final launch afterward to
capture the result for correctness checking. This is the pattern used in `gemm.py` — follow
that model.

---

## Should Fix in Sprint 3

### TL-S2-002 — benchmarks/run_all.py uses time_cpu for GPU k-means

**File**: `benchmarks/run_all.py` line 47

```python
gpu_time = runner.time_cpu(gpu_mod.kmeans_gpu, X, k=8, seed=42)
```

Should use a `time_gpu` pattern with CUDA events for consistency with other demos. Since
k-means manages its own stream internally, this requires restructuring to pass a pre-allocated
stream. Alternatively, at minimum add a comment noting the timing is wall-clock, not
CUDA-event-measured.

### TL-S2-004 — compiler.py docstring missing cache_key documentation

**File**: `src/kernels/compiler.py`

Add a `Parameters` section to the `compile()` docstring documenting `cache_key`:

```
cache_key:
    If provided, the compiled kernel is cached under this key and returned on
    subsequent calls without recompilation. Pass ``None`` to skip caching (the
    kernel is compiled fresh on every call). Note: when ``None`` is passed, the
    auto-generated lookup key (based on kernel_name + source hash) is still checked
    against the cache before compiling.
```

### TL-S2-005 — agents/todo.md stale item

Mark `[ ] [Prog] Fix TL-004: Implement tests/test_memory.py` as done (`[x]`). The fix was
delivered by QA. The tracking state is inconsistent.

---

## Summary of issue IDs

| ID | Severity | When to fix | File |
|----|----------|-------------|------|
| TL-S2-001 | Minor | Before Sprint 3 benchmarks | `demos/05_kernels/activations.py` |
| TL-S2-002 | Minor | Sprint 3 | `benchmarks/run_all.py` |
| TL-S2-003 | Minor | Before Sprint 3 benchmarks | `benchmarks/run_all.py` |
| TL-S2-004 | Nit | Sprint 3 | `src/kernels/compiler.py` |
| TL-S2-005 | Nit | Immediately | `agents/todo.md` |

---

*Tech Lead sign-off: Sprint 2 code is correct and safe to proceed. Address TL-S2-001 and
TL-S2-003 before running benchmark comparisons or presenting benchmark output to stakeholders.*
