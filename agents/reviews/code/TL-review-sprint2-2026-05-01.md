# Tech Lead Code Review — Sprint 2
**Date**: 2026-05-01
**Reviewer**: Tech Lead
**Sprint**: 2
**Verdict**: Conditional Approval

---

## Summary

Sprint 2 delivered three new algorithm demos (GPU PCA, GPU linear regression, and a custom
kernel showcase covering naive GEMM, ReLU, and softmax), plus a new `tests/test_memory.py`
that closes the TL-004 gap from Sprint 1. All new code follows the established patterns
correctly: every `DeviceBuffer` is managed by a context manager, CUDA error codes are checked
after every API call, stream synchronisation ordering is correct, and each GPU implementation
is validated against its NumPy baseline. The PCA sign-flip problem is properly handled in
both the demo's correctness check and the GPU test.

Three Sprint 1 fixes (TL-005, TL-006, TL-007) are partially complete: TL-005 is fully fixed,
TL-007 has a minimal inline comment but the docstring is untouched, and TL-006 uses
`BenchmarkRunner` but calls `time_cpu` for the GPU path rather than `time_gpu`, so CUDA-event
timing is still not applied to k-means. More importantly, `benchmarks/run_all.py` was not
updated to include the three new demos (PCA, linear regression, kernels), leaving the
benchmark runner stale with only Sprint 1 coverage.

The ReLU benchmark in `activations.py` includes an H2D copy inside the timed function, which
means the reported GPU time measures data transfer plus kernel execution rather than kernel
throughput alone. For a demo whose stated goal is to show kernel performance, this is a
meaningful methodological error.

There are no blockers to GPU correctness — all kernel designs are sound, boundary checks are
present, and the mathematical algorithms are correctly implemented.

---

## Scope

Files reviewed:
- `demos/03_pca/cpu_pca.py`
- `demos/03_pca/gpu_pca.py`
- `demos/03_pca/main.py`
- `demos/04_linear_model/cpu_linear.py`
- `demos/04_linear_model/gpu_linear.py`
- `demos/04_linear_model/main.py`
- `demos/05_kernels/gemm.py`
- `demos/05_kernels/activations.py`
- `demos/05_kernels/main.py`
- `tests/test_memory.py`
- `tests/test_pca.py`
- `tests/test_linear.py`
- `benchmarks/run_all.py` (TL-006 fix)
- `demos/01_core_apis/vector_add.py` (TL-005 fix)
- `src/kernels/compiler.py` (TL-007 fix)

---

## TL Sprint-1 Fix Verification

### TL-005 — Duplicate BenchmarkResult import in vector_add.py
**Status: Fixed.** The second `from src.utils.timing import BenchmarkResult` that previously
appeared at line 112 is gone. Only one import exists at line 29 inside the try block.

### TL-006 — Use BenchmarkRunner in benchmarks/run_all.py
**Status: Partially fixed.** `BenchmarkRunner` is now instantiated and used in
`run_02_kmeans()`. However, the GPU path calls `runner.time_cpu(gpu_mod.kmeans_gpu, …)`
rather than `runner.time_gpu(gpu_fn, stream)`. This measures wall-clock time including Python
overhead, not CUDA-event-measured kernel time. The fix moved from raw single-shot timing to
`BenchmarkRunner` but did not use the CUDA-event path. See TL-S2-002.

Additionally, `run_all.py` has not been updated to include demos 03_pca, 04_linear_model, or
05_kernels. The `--demo` choices list still only offers `01_core_apis`, `02_kmeans`, and
`all` — running `--demo all` silently skips three Sprint 2 demos. See TL-S2-003.

### TL-007 — Document cache_key=None behaviour in compiler.py
**Status: Minimally fixed.** An inline comment (`# cache_key=None: kernel is compiled fresh
each call and not stored in cache`) was added at line 41. The public docstring for `compile()`
on line 22 was not updated, so the parameter's opt-out semantics remain invisible to callers
reading the API signature. The inline comment is better than nothing but does not satisfy the
original intent of documenting the API contract. See TL-S2-004.

There is also a minor logic inconsistency: when `cache_key=None`, the key is computed as
`f"{kernel_name}:{hash(source)}"` (line 23) and is used for the cache lookup on line 24.
This means a kernel compiled with `cache_key=None` could be found in the cache if a previous
call with `cache_key=f"{kernel_name}:{hash(source)}"` had stored it. The comment says
"compile fresh each call" but that is only guaranteed if no other caller used the same
auto-generated key. This is a nit-level edge case but the comment is misleading.

### TL-004 — tests/test_memory.py
**Status: Fixed by QA.** The file exists and covers: `alloc_pinned` and `free_pinned`
ImportError paths (CPU-safe), `DeviceBuffer` context manager (`buf._buf is None` after exit),
and `query_device_memory` key/value assertions. The Programmer task item remains unchecked in
`agents/todo.md` while the QA item is checked — minor tracking inconsistency, not a code
issue.

---

## File-by-File Notes

### demos/03_pca/cpu_pca.py

Correct NumPy PCA via `eigh` on the unbiased covariance matrix (divided by `n_samples - 1`).
`eigh` returns eigenvalues in ascending order; the code correctly reverses with `argsort`
descending. Components are extracted as column slices of the eigenvector matrix then
transposed to row form — matching the `(n_components, n_features)` output contract.
Projection uses `X_c @ eigenvectors[:, :n_components]`, which is correct. No issues.

### demos/03_pca/gpu_pca.py

**Kernel correctness (mean_center)**: 1D flat-index kernel with `f = i % n_features` to
select the mean to subtract. Boundary guard `if (i < n_samples * n_features)` is correct.
In-place modification of `d_X` is intentional — the centered matrix is then used directly by
the covariance kernel.

**Kernel correctness (compute_cov)**: 2D grid (one block per `(i, j)` pair), upper-triangle
only with `if (j < i) return`. Each thread loops over all `k` samples and accumulates the dot
product. Divides by `n_samples - 1` (unbiased). Writes both `C[i,j]` and `C[j,i]`
(symmetry fill). Correct.

**No race condition**: Because each `(i, j)` pair is handled by exactly one thread and the
symmetry fill `C[j,i] = s` writes to a cell owned by the `(j, i)` thread (which also
computes the same value), there is no write conflict — every thread writes different cells.
This is safe.

**Eigendecomposition upgrade**: The code correctly upcasts `C_host` to `float64` before
calling `np.linalg.eigh`, reducing floating-point error in the eigendecomposition. Results
are then cast back to `float32`. This is good practice.

**Memory safety**: All three `DeviceBuffer` allocations are within the `with (...)` block.
`stream.sync()` is called inside the block before the D2H copies and again after the block
exits. Stream is closed after the context manager. Clean.

**Sign-flip handling**: `main.py` computes `|diag(cpu_comps @ gpu_comps.T)|` and checks
`> 1.0 - 1e-3`. This is the correct approach for comparing eigenvectors that may differ by
sign. The GPU test uses the same pattern. Correct.

### demos/03_pca/main.py

Clean argument parsing. CUDA availability check before import. Uses `BenchmarkRunner` with
`time_cpu` for both CPU and GPU paths — consistent with the existing pattern. The correctness
check uses the diagonal-alignment approach, which is correct. No issues.

### demos/04_linear_model/cpu_linear.py

Uses `np.linalg.lstsq` with `rcond=None`, which handles rank-deficient cases and is the
correct CPU solver. Works in `float64` internally and returns weights as `float32` — sensible
for the demo. R² and MSE computation are correct. No issues.

### demos/04_linear_model/gpu_linear.py

**Kernel correctness (xtx_xty)**: 2D grid. Thread `(i, j)` computes `XtX[i, j] = sum_k
X[k,i] * X[k,j]`. Thread `(i, 0)` additionally computes `Xty[i] = sum_k X[k,i] * y[k]`.

One correctness concern: the design computes the full `XtX` matrix (all `n_features^2`
threads), not just the upper triangle. This means `XtX[i,j]` and `XtX[j,i]` are computed
redundantly by separate threads but both write the correct value — no race, no error, just
wasted work. For correctness this is fine.

The `Xty` computation happens only when `j == 0`. This is correct — one thread per row `i`
computes the full dot product `X[:,i] · y`. No atomics needed, no race.

**Normal equation solve**: `np.linalg.solve(XtX, Xty)` after upcasting to `float64` is
correct when XtX is full rank (which it is for well-conditioned regression problems). The demo
data generator in `main.py` uses `n_features=32, n_samples=10000`, so XtX will be full rank.
For degenerate inputs the solve could fail with `LinAlgError` — acceptable for a demo.

**Memory safety**: All four `DeviceBuffer` objects are within the `with (...)` block. Correct.

### demos/05_kernels/gemm.py

**Kernel correctness (naive_gemm)**: Standard naive GEMM. Thread `(row, col)` computes
`C[row, col] = sum_k A[row,k] * B[k,col]`. Row-major indexing: `A[row * K + k]`,
`B[k * N + col]`, `C[row * N + col]`. Boundary guards check `row >= M || col >= N`. Correct.

**Grid/block assignment**: `blockIdx.x` maps to `col`, `blockIdx.y` to `row`. `threadIdx.x`
maps to x-dimension (col), `threadIdx.y` to y-dimension (row). Grid is `(ceil(N/16),
ceil(M/16), 1)` and block is `(16, 16, 1)`. This correctly tiles the M×N output. Correct.

**Benchmark methodology**: Uses `runner.time_cpu(lambda: A @ B)` for CPU and
`runner.time_gpu(gpu_fn, stream)` for GPU — CUDA events are used for GPU timing. This is the
correct approach and a contrast to the activations demo. Good.

**Memory safety**: All three `DeviceBuffer`s are within the `with (...)` block. Correct.

### demos/05_kernels/activations.py

**ReLU kernel correctness**: `fmaxf(0.0f, x[i])` with standard 1D boundary check. Correct.

**Softmax kernel correctness**: Per-row softmax with numerically stable max subtraction.
Thread 0 does all work — non-parallel, intentionally simple for educational purposes. Correct:
finds max, shifts by max (numerical stability), computes exp and sum, normalises. The
correctness check (`row_sums ≈ 1.0`) is appropriate.

**ReLU GPU timing methodology (Minor issue)**: Inside `relu_fn()`, the function performs
`_h2d(d_x.handle, kernel_x, n_bytes)` before launching the kernel. Since `relu_fn` is what
`runner.time_gpu` times, the reported GPU time includes an H2D transfer of `n=1_000_000`
float32 elements (4 MB) plus the kernel launch. The benchmark title and result claim to
measure ReLU kernel throughput, but the number includes data transfer. This inflates measured
GPU time and reduces reported speedup, misrepresenting kernel performance. See TL-S2-001.

**Memory safety**: Both `DeviceBuffer`s (`d_x` and `d_sm`) use `with` blocks. Correct.

### tests/test_memory.py

Two CPU-safe tests correctly mock `sys.modules["cuda.bindings.cudart"] = None` and verify
`RuntimeError` is raised by `alloc_pinned` and `free_pinned`. The mock teardown correctly
restores the original value (or removes the key if it was absent). No issues.

Two GPU tests (marked `@pytest.mark.gpu`) test `DeviceBuffer` context manager exit behaviour
(`buf._buf is None`) and `query_device_memory` return structure. Both tests use the
`gpu_device` fixture. The `DeviceBuffer` post-exit assertion `buf._buf is None` correctly
tests the internal state without requiring another GPU call. Good.

### tests/test_pca.py

Five CPU tests: shape, orthonormality (`components @ components.T ≈ I`), positivity of
eigenvalues, descending order of eigenvalues, and determinism. All are meaningful. The
orthonormality check uses `atol=1e-6` — appropriate for float32 arithmetic.

One GPU test: sign-flip-tolerant alignment check `|diag(cpu @ gpu.T)| > 0.999`. Correct and
sufficient. Uses `importlib` correctly for numeric-prefix module path. No issues.

### tests/test_linear.py

Four CPU tests: shape, weight recovery (max deviation < 0.1 for low-noise data), R² > 0.99,
MSE > 0 for noisy data. All meaningful and well-parameterised.

One GPU test: weight agreement within 1e-3. Tolerance is appropriate for float32 normal
equation vs float64 lstsq. No issues.

---

## Issues Found

| ID | Severity | File | Description |
|----|----------|------|-------------|
| TL-S2-001 | Minor | `demos/05_kernels/activations.py` line 119–123 | `relu_fn` includes an H2D transfer (`_h2d`) inside the function timed by `runner.time_gpu`. Reported GPU time measures transfer + kernel, not kernel alone. The function should only launch the kernel; data should be pre-uploaded before the timing loop. |
| TL-S2-002 | Minor | `benchmarks/run_all.py` line 47 | TL-006 partial fix: `runner.time_cpu(gpu_mod.kmeans_gpu, …)` is used instead of `runner.time_gpu(gpu_fn, stream)`. Wall-clock timing for the GPU path means k-means benchmark timing is inconsistent with vector_add and gemm which use CUDA events. |
| TL-S2-003 | Minor | `benchmarks/run_all.py` lines 74–76 | `--demo choices` and `main()` dispatch do not include `03_pca`, `04_linear_model`, or `05_kernels`. Running `python -m benchmarks.run_all` skips all Sprint 2 demos silently. |
| TL-S2-004 | Nit | `src/kernels/compiler.py` line 22 | TL-007 partial fix: inline comment documents `cache_key=None` opt-out, but the `compile()` docstring does not describe the parameter's semantics. Callers reading only the signature remain uninformed. |
| TL-S2-005 | Nit | `agents/todo.md` line 13 | Programmer's `[ ] Fix TL-004` item is unchecked despite the fix being delivered (by QA). The tracking state is inconsistent. |

---

## Positive Observations

- **Algorithmic correctness is solid throughout**: Unbiased covariance (`n - 1`), numerically
  stable softmax (max subtraction), `float64` upcasting for eigendecomposition and normal
  equation solve, sign-flip-tolerant eigenvector comparison — every nontrivial algorithmic
  decision was made correctly.
- **Memory safety is maintained**: All `DeviceBuffer` allocations in every new file are
  within `with` blocks. The pattern established in Sprint 1 is being followed consistently.
- **CUDA error checking is thorough**: Every `cudaMemcpy`, `cudaMemset`, and D2H/H2D call
  checks error codes and raises `RuntimeError` with context. No call is silently ignored.
- **Sign-flip handling is implemented correctly**: Both `demos/03_pca/main.py` and
  `tests/test_pca.py` use `|diag(cpu_comps @ gpu_comps.T)| > threshold` — the correct test
  for eigenvector agreement under arbitrary sign convention.
- **Test quality is high**: The PCA orthonormality test (`components @ components.T ≈ I`),
  the linear regression weight-recovery test, and the memory post-exit assertion are
  non-trivial and would catch real implementation bugs.
- **The GEMM benchmark uses CUDA events correctly**: `runner.time_gpu(gpu_fn, stream)` is
  used for the GEMM kernel — a good contrast to activations. The GEMM demo is the cleaner
  model for future kernel benchmarks.
- **No cross-demo imports**: All new demos import only from `src/`. Self-containment
  requirement from ARCH-001 is honoured.

---

## Verdict and Rationale

**Conditional Approval**. The Sprint 2 algorithm implementations are correct and safe to
merge. No kernel has a correctness bug, no device memory leaks through unmanaged allocations,
and the new tests are meaningful. The GPU correctness path for PCA and linear regression is
sound.

Two items must be addressed before Sprint 3 closes (or earlier if any demo is run in
benchmark mode):

1. **TL-S2-003** (Minor, functional gap): `benchmarks/run_all.py` does not include Sprint 2
   demos. Anyone running the benchmark runner gets incomplete results without warning. Add
   `run_03_pca`, `run_04_linear_model`, and `run_05_kernels` helper functions and include
   them in the `--demo` choices and `main()` dispatch.

2. **TL-S2-001** (Minor, methodological): The ReLU GPU benchmark includes an H2D copy in the
   timed region. Move `_h2d(d_x.handle, x_cpu, n_bytes)` outside the `relu_fn` closure
   (into the pre-timing setup), so `runner.time_gpu` measures only the kernel launch.

Items TL-S2-002 and TL-S2-004 should be tracked and fixed in Sprint 3. TL-S2-005 is a
tracking housekeeping nit.

---

## Required Changes

### Must fix before Sprint 3 benchmark work begins

- [ ] **[Programmer] Fix TL-S2-003**: Add `run_03_pca()`, `run_04_linear_model()`, and
  `run_05_kernels()` to `benchmarks/run_all.py` and update `--demo choices` and `main()`
  dispatch to include `03_pca`, `04_linear_model`, `05_kernels`. Each function should follow
  the existing `run_01_core_apis` pattern.

- [ ] **[Programmer] Fix TL-S2-001**: In `demos/05_kernels/activations.py`, restructure the
  ReLU timing so the `relu_fn` closure passed to `runner.time_gpu` only contains the kernel
  launch (not the H2D copy). Pre-upload `x_cpu` before the timing block and re-upload between
  timing repetitions by hooking into a pre-step callback or restructuring to a manual loop.

### Should fix in Sprint 3

- [ ] **[Programmer] Fix TL-S2-002**: In `benchmarks/run_all.py::run_02_kmeans()`, replace
  `runner.time_cpu(gpu_mod.kmeans_gpu, …)` with a `runner.time_gpu(gpu_fn, stream)`
  pattern consistent with how GEMM and vector_add are timed.

- [ ] **[Programmer] Fix TL-S2-004**: Add parameter documentation for `cache_key` in the
  `compile()` docstring in `src/kernels/compiler.py` — specifically note that `None` opts out
  of caching and explain the auto-generated fallback key behaviour.

- [ ] **[Programmer] Fix TL-S2-005**: Mark the `[Prog] Fix TL-004` item in `agents/todo.md`
  as done (the fix was delivered by QA; the programmer item should be closed).

---

## Checklist

| Category | Status | Notes |
|----------|--------|-------|
| Lint/format compliance | Pass | Consistent style throughout; no obvious ruff violations |
| Test coverage — CPU | Pass | 22 tests: shapes, correctness, orthonormality, weight recovery, R², MSE, memory error paths |
| Test coverage — GPU | Pass (skipped, no GPU) | 4 GPU tests properly marked; structure is sound |
| Algorithm correctness — PCA | Pass | Unbiased covariance, correct eigh sort, sign-flip handled |
| Algorithm correctness — linear | Pass | Normal equation correct; float64 upcast for solve |
| Algorithm correctness — GEMM | Pass | Standard naive GEMM; correct grid/block assignment |
| Algorithm correctness — ReLU | Pass | fmaxf, correct boundary guard |
| Algorithm correctness — softmax | Pass | Numerically stable, thread-0 only by design |
| Memory safety (DeviceBuffer) | Pass | All allocations in context managers across all new files |
| CUDA error checking | Pass | Every memcpy, memset checked |
| Stream sync ordering | Pass | sync inside with-block before D2H; sync + close after |
| TL-005 fix (duplicate import) | Pass | Removed |
| TL-006 fix (BenchmarkRunner) | Partial | Runner used, but time_cpu not time_gpu for GPU path |
| TL-007 fix (docstring) | Partial | Inline comment added; docstring not updated |
| TL-004 fix (test_memory.py) | Pass | File exists with meaningful CPU + GPU tests |
| Benchmark runner covers Sprint 2 | Fail | TL-S2-003: new demos not in run_all.py |
| ReLU benchmark timing correct | Fail | TL-S2-001: H2D inside timed fn |
| Self-contained demos | Pass | No cross-demo imports |
| Interface matches CLAUDE.md contract | Pass | run_demo / cpu/gpu functions present; BenchmarkResult returned |

---

*Review by: Tech Lead*
*Handoff to: Programmer (fix TL-S2-001 and TL-S2-003 before benchmark work)*
