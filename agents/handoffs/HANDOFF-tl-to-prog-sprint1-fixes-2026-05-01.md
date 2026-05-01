# HANDOFF: Tech Lead → Programmer
# Topic: Sprint 1 Code Review Fixes (Conditional Approval)
# Date: 2026-05-01

---

## What Was Completed

- Tech Lead reviewed all Sprint 1 deliverables: src/utils/, src/kernels/, demos/01_core_apis/, demos/02_kmeans/, benchmarks/run_all.py, and all tests.
- Verdict: **Conditional Approval** — Sprint 1 passes with two required fixes and several minor items.
- Full review is at: `agents/reviews/code/TL-review-sprint1-2026-05-01.md`
- All 11 CPU-only tests pass. GPU execution path is architecturally correct.
- GPU timing, memory management (DeviceBuffer), kernel launch, and error handling are all solid.

---

## What You Need to Do

### Task 1 (Required — fix before Sprint 2 closes): TL-001 — Device memory leak in pinned_memory.py

**File**: `demos/01_core_apis/pinned_memory.py`

**Problem**: Both `load_from_disk_to_device()` and `load_from_url_to_device()` allocate device memory with `device.allocate(n_bytes, stream=stream)` and return only a raw `int` pointer. The caller has no way to free this GPU memory. This violates REQ-0001-NF2 ("All device memory must be freed before demo exits").

**Fix**: Change the return type of both functions to return a `DeviceBuffer` instead of a raw `int`. The caller is then responsible for closing it (e.g., with a context manager). Alternatively, if a raw pointer return is intentional (e.g., for interop), add a companion `release_device_buffer(ptr: int)` helper and document in the docstring that the caller must call it. The simplest fix is the `DeviceBuffer` return.

Example of the corrected pattern:
```python
# Instead of:
buf = device.allocate(n_bytes, stream=stream)
# ... memcpy ...
return int(buf.handle)

# Do:
from src.utils.memory import DeviceBuffer
with DeviceBuffer(n_bytes, stream=stream, device=device) as buf:
    # ... memcpy ...
    # but note: returning from inside a context manager closes it immediately
    # so you need a different approach — either return the DeviceBuffer before closing:

buf = DeviceBuffer(n_bytes, stream=stream, device=device)
# ... memcpy ...
return buf  # caller must call buf.close() or use as context manager
```

### Task 2 (Required — fix before Sprint 2 closes): TL-002 — AC-8 and AC-9 not exercised in main()

**File**: `demos/01_core_apis/main.py`

**Problem**: `load_from_disk_to_device()` and `load_from_url_to_device()` exist in `pinned_memory.py` but are never called from `main()`. REQ-0001 acceptance criteria AC-8 and AC-9 require the demo to visibly demonstrate these operations.

**Fix**: Add a short section to `main()` that:
1. Writes a small synthetic numpy array to a temp file using `tempfile.mkstemp(suffix='.npy')` + `np.save(path, arr)`.
2. Calls `load_from_disk_to_device(path)` and prints a confirmation line (e.g., "Loaded 1024 floats from disk to device pointer 0x...").
3. Calls `load_from_url_to_device(url, fallback_shape=(1024,))` with a plausible but likely-offline URL to trigger the fallback path and show the graceful warning.
4. Frees the returned device buffer (after fixing TL-001 to return a `DeviceBuffer`).

### Task 3 (Should fix in Sprint 2): TL-003 — GPU k-means empty-cluster RNG bug

**File**: `demos/02_kmeans/gpu_kmeans.py`, line 201

**Problem**: The line `rng_empty = np.random.default_rng(seed)` is inside the iteration loop, so it resets the RNG to the same seed on every iteration. When an empty cluster is encountered, the fallback always selects the same data point.

**Fix**: Hoist `rng_empty` out of the loop (or just reuse the existing `rng` variable defined at line 97). The simplest fix:

```python
# Before the loop, replace the inner rng_empty usage with the outer rng:
# Line 201: delete `rng_empty = np.random.default_rng(seed)` 
# Line 207: change `rng_empty.integers(n_samples)` to `rng.integers(n_samples)`
```

### Task 4 (Should fix in Sprint 2): TL-004 — Add tests/test_memory.py

**Problem**: ARCH-001 lists `tests/test_memory.py` as a planned test file. `DeviceBuffer`, `alloc_pinned`, and `free_pinned` have no unit tests.

**Fix**: Create `tests/test_memory.py` with at minimum:
- A CPU-safe test verifying `DeviceBuffer.__init__` raises `RuntimeError` when `cuda-python` is absent (mock the import).
- A CPU-safe test verifying `alloc_pinned` raises `RuntimeError` when `cuda-python` is absent.
- A GPU test (marked `@pytest.mark.gpu`) verifying `DeviceBuffer` allocates, provides a non-zero handle, and closes cleanly.

### Tasks 5–7 (Nits — address in Sprint 3)

- **TL-005**: Remove the duplicate `from src.utils.timing import BenchmarkResult` import on line 112 of `demos/01_core_apis/vector_add.py` (already imported on line 29).
- **TL-006**: Use `BenchmarkRunner.time_gpu()` (or consistent wall-clock with warmup) in `benchmarks/run_all.py::run_02_kmeans()` instead of a single raw `time.perf_counter()` measurement.
- **TL-007**: Document the `cache_key=None` behaviour in `src/kernels/compiler.py::KernelCompiler.compile()` docstring — callers passing `None` opt out of caching and will recompile every call.

---

## Relevant Files

| File | Status | Notes |
|------|--------|-------|
| `agents/reviews/code/TL-review-sprint1-2026-05-01.md` | Complete | Full review — read this first |
| `demos/01_core_apis/pinned_memory.py` | Needs fix | TL-001 (device memory leak), TL-002 (not called) |
| `demos/01_core_apis/main.py` | Needs fix | TL-002 (add calls to disk/URL loaders) |
| `demos/02_kmeans/gpu_kmeans.py` | Needs fix | TL-003 (RNG bug) |
| `tests/test_memory.py` | Missing | TL-004 (create this file) |
| `demos/01_core_apis/vector_add.py` | Needs minor fix | TL-005 (duplicate import) |
| `benchmarks/run_all.py` | Needs minor fix | TL-006 (noisy timing) |
| `src/kernels/compiler.py` | Needs nit | TL-007 (docstring) |
| `agents/requirements/REQ-0001.md` | Reference | AC-8, AC-9 to satisfy |
| `agents/architecture/ARCH-001.md` | Reference | Memory discipline rules |

---

## Open Questions / Decisions Needed

- [ ] For TL-001: should `load_from_disk_to_device` return a `DeviceBuffer` object, or is a raw int pointer intentional for interop? If it must return an int (e.g., for passing to downstream CUDA APIs directly), the function docstring must state the caller is responsible for freeing via `DeviceBuffer(... ).close()` or equivalent.

---

## Acceptance Criteria

The fixes are complete when:

- [ ] AC-TL-001: `load_from_disk_to_device` and `load_from_url_to_device` in `pinned_memory.py` have no unfreed device allocations (verified by code inspection).
- [ ] AC-TL-002: `demos/01_core_apis/main()` calls both disk-loading and URL-loading functions and prints confirmation output.
- [ ] AC-TL-003: In `gpu_kmeans.py`, the RNG used for empty-cluster fallback is not re-seeded inside the iteration loop.
- [ ] AC-TL-004: `tests/test_memory.py` exists with at least one CPU-safe test and one GPU-marked test.
- [ ] All 11 existing CPU tests still pass after changes (`pytest tests/ -m "not gpu"`).

---

## Context / Notes

- The review found GPU kernel logic, CUDA event timing, error checking, and stream synchronisation to all be correct. Do not change those patterns.
- The memory leak (TL-001) is the most important fix — unfreed GPU memory persists across Python object lifetimes and accumulates on repeated demo invocations.
- The RNG bug (TL-003) only manifests with empty clusters, which requires small datasets or very unlucky centroid initialisation. It will not show up in normal benchmark runs, but it is a correctness bug.
- Sprint 2 work (PCA, linear model, naive Bayes) should proceed in parallel — these fixes are small and should not block Sprint 2 demo development.

---

*From: Tech Lead*
*To: Programmer*
*Next step: Fix TL-001 and TL-002 first (they are the conditions for the Conditional Approval). Then address TL-003 and TL-004. Run `pytest tests/ -m "not gpu"` after each fix to confirm no regressions.*
