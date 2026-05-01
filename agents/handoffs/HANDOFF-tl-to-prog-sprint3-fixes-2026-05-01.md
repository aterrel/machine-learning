# Handoff: Tech Lead → Programmer — Sprint 3 Fixes
**Date**: 2026-05-01
**From**: Tech Lead
**To**: Programmer
**Related Review**: `agents/reviews/code/TL-review-sprint3-2026-05-01.md`
**Sprint**: 3 (post-review fix pass)

---

## Context

Sprint 3 code review completed. Verdict: Conditional Approval. All deliverables are
functionally correct and memory-safe. Two Major issues and two Nits must be resolved before
the sprint is fully closed.

---

## Required Fixes

### TL-S3-001 (Major) — `demos/06_interop/cupy_interop.py`: misleading `interface` dict comment

**File**: `demos/06_interop/cupy_interop.py`
**Function**: `demo_cuda_python_to_cupy`
**Lines**: 76–83

The `interface` dict is built and printed but is dead code — CuPy is actually given the
device pointer via `cp.cuda.UnownedMemory`, not via this dict. The comment
"This is the zero-copy handoff: CuPy reads the same device pointer" is attached to the dict
and is incorrect.

**Fix**: Update the comment block so it is clear the dict is illustrative/documentary only:

```python
# __cuda_array_interface__ dict — shown here for documentation only.
# This is the protocol CuPy/PyTorch use internally to discover the device pointer.
# The actual zero-copy handoff below uses cp.cuda.UnownedMemory, which reads
# the raw pointer (buf.handle) directly — no dict assignment is required on our side.
interface = {
    "shape": (n,),
    "typestr": "<f4",
    "data": (buf.handle, False),
    "version": 3,
}
```

Remove or correct the old comment "This is the zero-copy handoff: CuPy reads the same device pointer".

---

### TL-S3-002 (Major) — `demos/06_interop/torch_interop.py`: same misleading comment

**File**: `demos/06_interop/torch_interop.py`
**Function**: `demo_cuda_python_to_torch`
**Lines**: 68–74

Same issue as TL-S3-001. The `interface` dict is printed but is never passed to PyTorch.
The comment should clarify it is illustrative.

**Fix**: Same comment update as TL-S3-001.

---

### TL-S3-003 (Minor) — `demos/06_interop/pipeline.py`: `cp_arr` potentially undefined

**File**: `demos/06_interop/pipeline.py`
**Function**: `run_end_to_end_pipeline`
**Lines**: ~91–115

`cp_arr` is assigned inside a `try: import cupy` block (step 3). In step 4, it is referenced
via `torch.as_tensor(cp_arr, device="cuda")` inside `if results["cupy_available"]`. If step 3
raises a non-ImportError exception (e.g., a CuPy device error), `cp_arr` is undefined and
step 4 raises `NameError`.

**Fix**: Initialise `cp_arr = None` before step 3's try block:

```python
cp_arr = None
# Step 3: Optional CuPy ...
try:
    import cupy as cp
    results["cupy_available"] = True
    mem = cp.cuda.MemoryPointer(...)
    cp_arr = cp.ndarray(...)
    ...
except ImportError:
    ...

# Step 4: Optional PyTorch
try:
    import torch
    results["torch_available"] = True
    if results["cupy_available"] and cp_arr is not None:
        t = torch.as_tensor(cp_arr, device="cuda")
    else:
        # D2H fallback ...
```

---

### TL-S3-004 (Nit) — `demos/07_memory/main.py`: redundant exception types

**File**: `demos/07_memory/main.py`
**Function**: `demo_oom_recovery`
**Line**: 165

```python
# Before
except (RuntimeError, MemoryError, Exception) as exc:
# After
except Exception as exc:
```

`Exception` is a superclass of both `RuntimeError` and `MemoryError`; listing them explicitly
is redundant and suggests they have different handling (they don't).

---

### TL-S3-005 (Nit) — `demos/07_memory/main.py`: missing contrast comment in `demo_basic_alloc`

**File**: `demos/07_memory/main.py`
**Function**: `demo_basic_alloc`
**After line**: 43 (after `buf = DeviceBuffer(n_bytes, device=device)`)

Add a comment clarifying the intentional lack of a context manager:

```python
buf = DeviceBuffer(n_bytes, device=device)
# No 'with' block — this demo shows the explicit alloc/free pattern.
# See demo_context_manager() for the safer context-manager form.
```

---

## Non-Issues (Do Not Fix)

- Jupyter notebooks: deferred, not a Sprint 3 goal.
- Naive Bayes demo: explicitly deferred.
- `benchmarks/run_all.py` not including interop/memory demos: these demos do not produce
  `BenchmarkResult` objects by design; exclusion is correct.

---

## Testing After Fixes

```bash
# Verify all CPU tests still pass
pytest tests/ -v -m "not gpu"

# Lint
ruff check demos/06_interop/ demos/07_memory/
```

No new tests are required for these fixes (comments only, except TL-S3-003 which is a
defensive initialisation).

---

## Commit Convention

```
[Prog] Fix TL-S3-001 through TL-S3-005: interop comment accuracy and defensive init
```
