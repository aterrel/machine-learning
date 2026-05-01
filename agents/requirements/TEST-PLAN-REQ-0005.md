# Test Plan — REQ-0005: Interoperability with NumPy, PyTorch, and CuPy

**Author**: QA Agent
**Date**: 2026-05-01
**REQ Reference**: agents/requirements/REQ-0005.md
**Implementation**: Sprint 3 (main branch)

---

## Scope

This plan covers acceptance criteria AC-1 through AC-5 of REQ-0005, testing:
- `demo_cuda_python_to_cupy()` graceful skip when CuPy is not installed
- `demo_cupy_to_cuda_python()` graceful skip when CuPy is not installed
- `demo_cuda_python_to_torch()` graceful skip when PyTorch is not installed
- `demo_torch_to_cuda_python()` graceful skip when PyTorch is not installed
- `__cuda_array_interface__` dict structural correctness (shape, typestr, data, version)
- `run_end_to_end_pipeline()` return dict key presence
- Zero-copy pointer verification when GPU + CuPy are available

Tests are split into CPU-safe (no GPU required) and GPU-required categories.

---

## Test Environment

- Runtime: Python 3.11+, NumPy >= 1.26
- GPU: NVIDIA GPU with CUDA 12.x (required for GPU tests only)
- CuPy: optional (GPU tests that exercise CuPy skip if absent)
- PyTorch: optional (GPU tests that exercise PyTorch skip if absent)
- External services: None (offline safe)
- Test command:
  ```bash
  # All tests (GPU tests auto-skip if no GPU):
  pytest tests/test_interop.py -v

  # CPU-only (CI safe):
  pytest tests/test_interop.py -v -m "not gpu"

  # GPU tests only:
  pytest tests/test_interop.py -v -m gpu
  ```

---

## Test Cases

### Happy Path

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-001 | `__cuda_array_interface__` shape key present | Construct interface dict as in source | `"shape"` key exists, value is a tuple | No |
| TC-002 | `__cuda_array_interface__` typestr key present | Construct interface dict as in source | `"typestr"` key exists, value is a string | No |
| TC-003 | `__cuda_array_interface__` data key present | Construct interface dict as in source | `"data"` key exists, value is a 2-tuple | No |
| TC-004 | `__cuda_array_interface__` version key present | Construct interface dict as in source | `"version"` key exists, value equals 3 | No |
| TC-005 | typestr is little-endian float32 | Construct interface dict as in source | `typestr == "<f4"` | No |
| TC-006 | data tuple has exactly 2 elements | Construct interface dict as in source | `len(interface["data"]) == 2` | No |
| TC-007 | data[1] is False (not read-only) | Construct interface dict as in source | `interface["data"][1] is False` | No |
| TC-008 | `run_end_to_end_pipeline()` returns dict | Call function with GPU available | Returns a dict, not None | Yes |
| TC-009 | pipeline result has `n` key | Call `run_end_to_end_pipeline(n=1000)` | `result["n"] == 1000` | Yes |
| TC-010 | pipeline result has `cupy_available` key | Call `run_end_to_end_pipeline()` | `"cupy_available"` in result, value is bool | Yes |
| TC-011 | pipeline result has `torch_available` key | Call `run_end_to_end_pipeline()` | `"torch_available"` in result, value is bool | Yes |
| TC-012 | pipeline result has `relu_correct` key | Call `run_end_to_end_pipeline()` | `"relu_correct"` in result, value is True | Yes |
| TC-013 | pipeline result has `cupy_norm` key | Call `run_end_to_end_pipeline()` | `"cupy_norm"` in result | Yes |
| TC-014 | pipeline result has `torch_sum` key | Call `run_end_to_end_pipeline()` | `"torch_sum"` in result | Yes |
| TC-015 | Zero-copy: same pointer, no data copy | Call `demo_cuda_python_to_cupy()`; inspect CuPy mem ptr vs DeviceBuffer handle | `cp_arr.data.ptr == buf.handle` | Yes (GPU + CuPy) |

### Error / Skip Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-020 | `demo_cuda_python_to_cupy` skips when CuPy missing | Patch `sys.modules["cupy"] = None`, call function | Returns without raising any exception | No |
| TC-021 | `demo_cupy_to_cuda_python` skips when CuPy missing | Patch `sys.modules["cupy"] = None`, call function | Returns without raising any exception | No |
| TC-022 | `demo_cuda_python_to_torch` skips when torch missing | Patch `sys.modules["torch"] = None`, call function | Returns without raising any exception | No |
| TC-023 | `demo_torch_to_cuda_python` skips when torch missing | Patch `sys.modules["torch"] = None`, call function | Returns without raising any exception | No |
| TC-024 | Skip message printed for missing CuPy | Patch `sys.modules["cupy"] = None`, capture stdout | "CuPy not installed" appears in output | No |
| TC-025 | Skip message printed for missing PyTorch | Patch `sys.modules["torch"] = None`, capture stdout | "PyTorch not installed" appears in output | No |

### Edge Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-030 | Interface data[0] is an integer (pointer) | Construct interface dict | `isinstance(interface["data"][0], int)` | No |
| TC-031 | Interface shape is a tuple | Construct interface dict | `isinstance(interface["shape"], tuple)` | No |
| TC-032 | pipeline n=1 works without error | Call `run_end_to_end_pipeline(n=1)` | Returns dict, `relu_correct` is True | Yes |

---

## Acceptance Criteria Coverage

| AC from REQ-0005 | Test Case(s) | Status |
|------------------|--------------|--------|
| AC-1: CUDA Python buffer used in CuPy ops without H2D/D2H copy | TC-015 (zero-copy pointer match) | Pending |
| AC-2: CUDA Python buffer consumed as PyTorch CUDA tensor | TC-008 through TC-014 (pipeline), TC-011 | Pending |
| AC-3: End-to-end pipeline: custom kernel → CuPy norm → PyTorch sum | TC-008 through TC-014 | Pending |
| AC-4: Without PyTorch, prints message and exits cleanly | TC-022, TC-023, TC-025 | Pending |
| AC-5: Source comments identify zero-copy vs copy sites (code review) | Manual review | Deferred |

---

## Test Data Strategy

**Interface dict tests (TC-001 through TC-007, TC-030, TC-031):**
```python
interface = {
    "shape": (1_000_000,),
    "typestr": "<f4",
    "data": (12345678, False),  # fake ptr, read_only=False
    "version": 3,
}
```
Constructed in-process without touching GPU memory — safe on any machine.

**Skip tests (TC-020 through TC-025):**
```python
import sys
import unittest.mock
import importlib

with unittest.mock.patch.dict("sys.modules", {"cupy": None}):
    mod = importlib.import_module("demos.06_interop.cupy_interop")
    mod.demo_cuda_python_to_cupy()  # must not raise
```

**Pipeline tests (TC-008 through TC-015):**
```python
import importlib
pipeline = importlib.import_module("demos.06_interop.pipeline")
result = pipeline.run_end_to_end_pipeline(n=100_000)
assert isinstance(result, dict)
```

---

## Out of Scope

- JAX interop
- DLPack-primary path (not implemented in Sprint 3)
- Multi-stream interop across CuPy and CUDA Python
- Performance benchmarks for zero-copy vs copy paths

---

## Bugs Found

| Bug ID | Description | Severity | Status |
|--------|-------------|----------|--------|
| — | No bugs found during test plan authoring | — | — |
