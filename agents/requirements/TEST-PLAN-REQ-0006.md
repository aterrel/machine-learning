# Test Plan — REQ-0006: GPU Memory Management Patterns for ML Workloads

**Author**: QA Agent
**Date**: 2026-05-01
**REQ Reference**: agents/requirements/REQ-0006.md
**Implementation**: Sprint 3 (main branch)

---

## Scope

This plan covers acceptance criteria AC-1 through AC-6 of REQ-0006, testing:
- `DeviceBuffer` context manager frees device memory on `__exit__` even when an exception is raised
- `query_device_memory()` returns a dict with `free_gb` and `total_gb` keys, both > 0
- `alloc_pinned()` raises `RuntimeError` when `cuda.bindings` is not installed
- `demo_context_manager()` does not propagate an exception (catches it internally)
- `demo_oom_recovery()` continues after a CUDA OOM error
- `demo_basic_alloc()` prints memory stats in expected format

Tests are split into CPU-safe (no GPU required) and GPU-required categories.

---

## Test Environment

- Runtime: Python 3.11+, NumPy >= 1.26
- GPU: NVIDIA GPU with CUDA 12.x (required for GPU tests only)
- External services: None (offline safe)
- Test command:
  ```bash
  # All tests (GPU tests auto-skip if no GPU):
  pytest tests/test_memory.py -v

  # CPU-only (CI safe):
  pytest tests/test_memory.py -v -m "not gpu"

  # GPU tests only:
  pytest tests/test_memory.py -v -m gpu
  ```

Note: `tests/test_memory.py` already contains CPU-only and GPU tests from Sprint 1/2.
Sprint 3 adds additional GPU tests for context-manager-on-exception and memory queries.

---

## Test Cases

### Happy Path

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-001 | `query_device_memory` returns dict with `free_gb` | Call `query_device_memory(device)` | `"free_gb"` in result, value > 0 | Yes |
| TC-002 | `query_device_memory` returns dict with `total_gb` | Call `query_device_memory(device)` | `"total_gb"` in result, value > 0 | Yes |
| TC-003 | `free_gb <= total_gb` | Call `query_device_memory(device)` | `result["free_gb"] <= result["total_gb"]` | Yes |
| TC-004 | Context manager frees memory on clean exit | Use `with DeviceBuffer(...) as buf:` with no exception | After exit, `buf._buf is None` | Yes |
| TC-005 | Context manager frees memory on exception | Use `with DeviceBuffer(...) as buf:` then raise inside | After catching exception, `buf._buf is None` | Yes |
| TC-006 | `demo_context_manager()` does not raise | Call `demo_context_manager()` directly | Function returns normally (exception is caught internally) | Yes |
| TC-007 | `demo_basic_alloc()` runs without error | Call `demo_basic_alloc()` | No exception raised | Yes |
| TC-008 | `demo_oom_recovery()` runs without error | Call `demo_oom_recovery()` | No exception propagates to caller | Yes |
| TC-009 | Memory recovered after DeviceBuffer freed | Allocate 256 MB, free via `buf.close()`, query memory | `free_gb` after close > `free_gb` during allocation | Yes |

### Error / Import Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-020 | `alloc_pinned` raises `RuntimeError` when `cuda.bindings` missing | Patch `sys.modules["cuda.bindings.cudart"] = None`, call `alloc_pinned(1024)` | Raises `RuntimeError` | No |
| TC-021 | `free_pinned` raises `RuntimeError` when `cuda.bindings` missing | Patch `sys.modules["cuda.bindings.cudart"] = None`, call `free_pinned(0)` | Raises `RuntimeError` | No |
| TC-022 | `query_device_memory` raises `RuntimeError` when `cuda.bindings` missing | Patch `sys.modules["cuda.bindings.cudart"] = None`, call `query_device_memory(None)` | Raises `RuntimeError` | No |

### Edge Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-030 | `DeviceBuffer.close()` is idempotent | Call `buf.close()` twice | No error on second call; `buf._buf` remains `None` | Yes |
| TC-031 | Allocate minimum-size buffer (1 byte) | `DeviceBuffer(1, device=device)` inside context manager | No error; `buf.handle > 0` | Yes |
| TC-032 | OOM recovery leaves device usable | Call `demo_oom_recovery()`, then allocate a small buffer | Small allocation succeeds | Yes |

---

## Acceptance Criteria Coverage

| AC from REQ-0006 | Test Case(s) | Status |
|------------------|--------------|--------|
| AC-1: `python demos/07_memory/main.py` shows memory usage before/after each pattern | TC-007, TC-008 (smoke tests); manual smoke run | Pending |
| AC-2: Context manager raises exception inside block; memory freed after | TC-005, TC-006 | Pending |
| AC-3: Pinned memory benchmark shows ≥ 2x speedup for large transfers | Manual performance test (hardware-dependent) | Deferred |
| AC-4: Memory pool demo sub-allocates 10 chunks (not implemented in Sprint 3) | Deferred | Deferred |
| AC-5: OOM recovery catches error and prints remaining free memory | TC-008, TC-032 | Pending |
| AC-6: Every allocation has explicit free or context-manager guarantee | Code review (TC-004, TC-005 verify one pattern) | Pending |

---

## Test Data Strategy

**Import-error tests (TC-020, TC-021, TC-022) — CPU-only:**
```python
import sys
import pytest
from src.utils import memory as memory_mod

saved = sys.modules.get("cuda.bindings.cudart")
sys.modules["cuda.bindings.cudart"] = None
try:
    with pytest.raises(RuntimeError):
        memory_mod.alloc_pinned(1024)
finally:
    if saved is None:
        sys.modules.pop("cuda.bindings.cudart", None)
    else:
        sys.modules["cuda.bindings.cudart"] = saved
```

**Context-manager-on-exception test (TC-005):**
```python
from src.utils.memory import DeviceBuffer

exception_raised = False
try:
    with DeviceBuffer(1024, device=gpu_device) as buf:
        raise ValueError("simulated error")
except ValueError:
    exception_raised = True

assert exception_raised
assert buf._buf is None
```

**Memory recovery test (TC-009):**
```python
from src.utils.memory import DeviceBuffer, query_device_memory

n_bytes = 256 * 1024 * 1024
buf = DeviceBuffer(n_bytes, device=gpu_device)
mem_during = query_device_memory(gpu_device)
buf.close()
mem_after = query_device_memory(gpu_device)
assert mem_after["free_gb"] > mem_during["free_gb"]
```

---

## Out of Scope

- Unified memory (managed memory) — deferred per REQ-0006
- Multi-GPU peer-to-peer transfers
- Memory pool sub-allocation (not implemented in Sprint 3)
- Nsight profiler integration
- Stream-ordered allocation (cudaMallocAsync) — P1 carry, not in Sprint 3 scope

---

## Bugs Found

| Bug ID | Description | Severity | Status |
|--------|-------------|----------|--------|
| — | No bugs found during test plan authoring | — | — |
