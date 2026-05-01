# Tech Lead Final Review — Sprint 4 + Project Close
**Date**: 2026-05-01
**Reviewer**: Tech Lead
**Sprint**: 4 (Final)
**Verdict**: Approved

---

## Sprint 4 Deliverables Review

Sprint 4 delivered two items: the `demos/05_naive_bayes/` Gaussian Naive Bayes demo (deferred
twice from prior sprints) and fixes for TL-S3-004 and TL-S3-005 from the Sprint 3 conditional
approval. Both items are confirmed delivered in commit `eda689c`.

### `demos/05_naive_bayes/cpu_nb.py`

The CPU Gaussian Naive Bayes baseline is mathematically correct and well-annotated.

- **Prior computation** (`log(count / n_train)`): correct.
- **Variance** (`X_c.var(axis=0) + 1e-9`): uses population variance — appropriate for a Bayes
  model; the `1e-9` epsilon for numerical stability is the standard convention.
- **Log-likelihood** (`-0.5 * log(2*pi*var) - 0.5 * (x - mean)^2 / var`): formula is correct
  Gaussian log-density. Both terms are summed over features before adding the log-prior.
- **Degenerate class handling**: falls back to uniform prior and zero-variance guard correctly.
- **Dtype discipline**: trains and accumulates statistics in float64, returns predictions as
  int32 — correct; avoids precision loss during accumulation.
- **Return signature**: `(predictions, class_means, class_vars, log_priors)` — intentionally
  broader than the minimal API contract so `main.py` can pass statistics to the GPU path. The
  return type annotation in the docstring says `tuple[np.ndarray, float]` but the function
  returns four values; this is a documentation nit, not a runtime issue.

### `demos/05_naive_bayes/gpu_nb.py`

**Kernel correctness — `compute_log_likelihood`**

```c
ll += -0.5f * logf(2.0f * 3.14159265f * var) - 0.5f * (x - mu) * (x - mu) / var;
```

Formula matches the Gaussian log-density exactly: `-0.5 * log(2*pi*var) - 0.5 * (x-mu)^2/var`.
The log-prior is initialized from `log_priors[cls]` before the feature loop and accumulated
correctly. The `M_PI` constant is hardcoded as `3.14159265f` — acceptable precision for float32.

**Grid/block mapping**: x-dimension covers test samples, y-dimension covers classes. Each thread
handles exactly one `(sample, class)` pair. The bounds check `if (sample >= n_test || cls >= n_classes)` is correct. Row-major indexing into `log_probs[sample * n_classes + cls]` matches
the argmax kernel's read pattern.

**Kernel correctness — `argmax_predictions`**

Linear scan with scalar best-value tracking. Correct. No race conditions (each thread owns one
output element). Integer index `best_cls` starts at 0 and updates correctly.

**Fit on CPU, predict on GPU**: the fit step (computing means, variances, priors) is done on
CPU as documented. Both mean and variance arrays are computed in float64 then cast to float32
for GPU transfer — good practice to minimize precision loss during statistics computation.

**Memory safety**: all six `DeviceBuffer` allocations are inside a single `with` block using
Python 3.10+ parenthesized context managers. All buffers are freed atomically on exit, even on
exception — correct and idiomatic.

**Stream lifecycle**: `stream.sync()` is called inside the `with` block before D2H copy, then
again after the `with` block closes (redundant but harmless), then `stream.close()` — correct.

**H2D / D2H copies**: `cudaMemcpy` with `cudaMemcpyHostToDevice` / `cudaMemcpyDeviceToHost`
uses `.ctypes.data` for the host pointer — correct. Error codes are checked and raised.

**Import guard**: `sys.exit(1)` on missing cuda-python or missing GPU — consistent with other
demos.

One minor note: `gaussian_nb_gpu` returns `(predictions_host, class_means, class_vars, log_priors)` where the last three are the CPU-computed statistics, not GPU outputs. This is not
wrong — `main.py` uses them for comparison printing — but the return signature could note this
more explicitly. Not a blocker.

### `demos/05_naive_bayes/main.py`

- Imports `gaussian_nb_cpu` and `gaussian_nb_gpu` as relative imports from within the package
  (`from .cpu_nb import ...`) — correct; no cross-demo imports.
- `BenchmarkResult` is instantiated with the correct field names (`demo_name`, `cpu_time_mean_s`,
  `gpu_time_mean_s`, `speedup`, `correct`, `max_abs_error`) matching `src/utils/timing.py`.
- Correctness check (`max_abs_error = float(np.max(np.abs(cpu_preds - gpu_preds)))`) is computed
  over integer prediction arrays; this is a valid match metric (0 = perfect agreement).
- Data generation in `_generate_data` uses Gaussian blobs with per-class cluster centers spaced
  3.0 standard deviations apart — well-separated classes that make the demo pedagogically clear
  and ensure high accuracy.
- Degenerate class handling in the data generator: uses `n_total - base * (n_classes - 1)` for
  the last class, ensuring sample counts sum to exactly `n_total` — correct.
- The `check_cuda_available()` early-exit guard is consistent with other GPU demos.

### TL-S3-004 fix — `demo_oom_recovery` except clause

Confirmed: `except (RuntimeError, MemoryError, Exception)` has been simplified to
`except Exception`. TL-S3-004 is closed.

### TL-S3-005 fix — `demo_basic_alloc` comment

Confirmed: the inline comment "Intentionally using explicit close() here (contrast with
demo_context_manager which shows the context manager pattern)" is now on line 43.
TL-S3-005 is closed.

---

## Holistic Project Assessment

### Architecture Adherence

The self-contained demo requirement is met across the codebase. Each demo module imports only
from `src/` (shared kernel/device/memory/timing utilities) and from within its own package.
The single tolerated exception is `demos/07_memory/main.py::demo_pinned_transfer`, which
imports `demos.01_core_apis.pinned_memory` via `importlib` to reuse the pinned transfer
benchmark rather than duplicating it. This is appropriately guarded with `try/except` and
documented with an explanatory comment. It is an intentional pedagogical cross-reference, not
an architectural violation. `demos/06_interop/main.py` imports from within its own package
(`demos.06_interop`), which is correct.

### Source Utilities (`src/`)

- `src/utils/device.py`: clean, correct, no issues.
- `src/utils/memory.py`: `DeviceBuffer` is correct. The `close()` method guards against double-
  free with `if self._buf is not None`. `alloc_pinned` / `free_pinned` pair is symmetric and
  correct. `query_device_memory` returns a dict with `free_gb` / `total_gb` — consistent with
  all call sites.
- `src/utils/timing.py`: `BenchmarkResult` fields match all demo usage sites. `BenchmarkRunner`
  uses CUDA events for GPU timing — correct and standard. CUDA event objects are created and
  destroyed in pairs within each timing iteration — no event handle leaks.
- `src/kernels/compiler.py`: kernel cache is keyed by `cache_key` when provided, correctly
  avoids re-compilation. The `cache_key=None` path compiles fresh and stores by
  `f"{kernel_name}:{hash(source)}"` — note this means a None-keyed kernel IS cached by this
  fallback key. The docstring says `cache_key=None` skips caching, but the implementation
  stores it. This is a latent documentation inconsistency: the condition `if cache_key is not
  None: self._cache[key] = compiled` correctly limits storage to explicit-key calls, and the
  comment on line 49 clarifies intent. The actual behavior matches the documented intent;
  the only confusion is the fallback key assignment `key = cache_key or f"..."` which runs
  even when `cache_key is None`, though storage is correctly gated. This is a nit.
- `src/kernels/compiled_kernel.py`: `CompiledKernel.launch` is thin and correct. `compute_grid_1d` is a useful static helper.

### `pyproject.toml`

Well-structured. Python 3.11+ correctly enforced. `cuda-python >= 12.3` pinned. Optional deps
(`cupy`, `torch`, `jupyter`) are in `[project.optional-dependencies].extras`. Ruff is
configured for line length 100 with E501 ignored (consistent with the demos' 100-char lines).
The `[tool.setuptools.packages.find]` only includes `src*`, leaving `demos/`, `tests/`, and
`benchmarks/` outside the installable package — appropriate for this project structure.

### Test Coverage

32 CPU tests passing, 14 GPU tests deferred. The test suite covers:
- `test_device.py` — device availability and property inspection
- `test_kernels.py` — KernelCompiler and CompiledKernel
- `test_kmeans.py` — CPU baseline and GPU k-means
- `test_memory.py` — DeviceBuffer, pinned allocation, memory queries
- `test_pca.py` — CPU PCA correctness and GPU PCA
- `test_linear.py` — linear regression CPU/GPU
- `test_interop.py` — `__cuda_array_interface__` structure, graceful skip without CuPy/PyTorch

**Gap**: `demos/05_naive_bayes/` has no test file. The CPU `gaussian_nb_cpu` function is
directly testable without a GPU (it is pure NumPy). A `test_naive_bayes.py` covering the CPU
baseline and verifying prediction shape/type, high accuracy on well-separated blobs, and
matching output format would be consistent with the coverage pattern established for every
other algorithm demo. This omission is the only notable gap in coverage across the project.

**Gap (pre-existing, not a Sprint 4 regression)**: `demos/07_memory/` has no dedicated unit
tests for `demo_basic_alloc`, `demo_context_manager`, or `demo_oom_recovery`. Memory patterns
are exercised indirectly via `test_memory.py` which tests `DeviceBuffer` and related utilities.
This was noted as acceptable in prior reviews; no change.

### Benchmark Runner (`benchmarks/run_all.py`)

Covers demos 01 through 05_kernels. Does not cover 05_naive_bayes (new this sprint). This is
consistent with the prior-sprint pattern where `demos/06_interop` and `demos/07_memory` are
also absent — pattern demos and non-benchmark demos are excluded by design. However,
`05_naive_bayes` is a performance demo (fit + inference with timing and correctness check) and
fits the benchmark runner's model. Adding it would complete coverage. This is a backlog item,
not a blocker for project close.

### Pedagogy and Code Clarity

The project succeeds at its primary mission. Each demo is runnable from the command line,
produces clearly labeled output, and follows the CPU-baseline-then-GPU-comparison pattern
consistently. Kernel source strings are embedded in the same file as their Python wrappers,
making the GPU code immediately visible without file-switching. Comments explain the CUDA
API choices, not just what the code does. Error messages are human-readable and guide the user
toward the fix. The `BenchmarkResult.summary_line()` output is uniform across all demos.

---

## Issues Found

| ID | Severity | File | Description |
|----|----------|------|-------------|
| TL-S4-001 | Nit | `demos/05_naive_bayes/cpu_nb.py` | Docstring return annotation says `tuple[np.ndarray, float]` but function returns four values `(predictions, class_means, class_vars, log_priors)`. No runtime impact. |
| TL-S4-002 | Nit | `demos/05_naive_bayes/gpu_nb.py` | Return docstring does not note that `class_means`, `class_vars`, `log_priors` are the CPU-computed statistics, not GPU outputs. Minor clarity gap for readers. |
| TL-S4-003 | Minor | `tests/` | No `test_naive_bayes.py`. CPU Naive Bayes baseline (`gaussian_nb_cpu`) is pure NumPy and testable without GPU. Missing coverage breaks the pattern established for every other algorithm demo. Backlog candidate. |
| TL-S4-004 | Nit | `src/kernels/compiler.py` | When `cache_key=None`, the function still builds a fallback key `f"{kernel_name}:{hash(source)}"` and assigns `key = fallback`, but storage is correctly gated by `if cache_key is not None`. The docstring says `cache_key=None` skips caching; the behavior matches the intent, but the `key` variable is computed unnecessarily. Minor code clarity issue only. |
| TL-S4-005 | Info | `benchmarks/run_all.py` | `demos/05_naive_bayes` is not included in the benchmark runner. Fits the runner's model. Backlog candidate. |

---

## Final Verdict and Project Status

**Approved. Project closed.**

All P0 requirements across REQ-0001 through REQ-0006 are delivered. The six sprint conditional
approvals from prior Tech Lead reviews have been cleared: every Major and Minor issue from
Sprints 1–3 was fixed before the next sprint opened, and the two Sprint 3 nit fixes
(TL-S3-004, TL-S3-005) were confirmed applied in this sprint.

The Sprint 4 naive Bayes deliverable is correct: the Gaussian log-likelihood formula is
mathematically exact in both CPU and GPU implementations, memory is safely managed with
context managers, and the demo follows the established API contract (CPU baseline +
GPU inference + `BenchmarkResult` summary). The two nit-level docstring issues (TL-S4-001,
TL-S4-002) do not affect correctness or pedagogy in any meaningful way.

The only gap of substance (TL-S4-003) — missing CPU-safe Naive Bayes tests — is a backlog
item, not a blocking issue for a project in its final sprint. For educational software, the
existing test coverage across all other algorithm demos demonstrates the pattern clearly.

The codebase is self-consistent, memory-safe, and pedagogically clear throughout. It achieves
its stated mission: hands-on demonstrations showing CUDA Python accelerating common ML
algorithms, bridging the gap between high-level frameworks and raw CUDA programming.

**Remaining backlog** (not blocking, for future maintenance reference):
- `test_naive_bayes.py` — CPU baseline unit tests
- Add `demos/05_naive_bayes` to `benchmarks/run_all.py`
- Jupyter notebook versions of demos 01 and 02 (deferred from Sprint 4 Programmer tasks)
- README.md user-facing documentation
