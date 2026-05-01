# Test Plan — REQ-0001: Core CUDA Python API Demonstrations

**Author**: QA Agent
**Date**: 2026-05-01
**REQ Reference**: agents/requirements/REQ-0001.md
**Implementation**: Sprint 1 (commit on main branch)

---

## Scope

This plan covers acceptance criteria AC-1 through AC-9 of REQ-0001, testing:
- Device enumeration and property inspection (`device_info.py`)
- Vector-add kernel correctness against NumPy baseline (`vector_add.py`)
- Graceful error handling when no CUDA GPU is present
- Pinned vs pageable memory transfer benchmark (`pinned_memory.py`)
- Loading a `.npy` dataset from disk into pinned memory and transferring to device

Tests are split into CPU-safe (no GPU required) and GPU-required categories so that CI
can run the CPU-safe subset unconditionally.

---

## Test Environment

- Runtime: Python 3.11+
- GPU: NVIDIA GPU with CUDA 12.x (required for GPU tests only)
- External services: None (offline safe)
- Test command:
  ```bash
  # All tests (GPU tests auto-skip if no GPU):
  pytest tests/test_device.py tests/test_kernels.py -v

  # CPU-only (CI safe):
  pytest tests/test_device.py tests/test_kernels.py -v -m "not gpu"

  # GPU tests only:
  pytest tests/test_device.py tests/test_kernels.py -v -m gpu
  ```

---

## Test Cases

### Happy Path

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-001 | `check_cuda_available` returns bool | Import and call `check_cuda_available()` | Returns `True` or `False` (always a `bool`) | No |
| TC-002 | Device object obtained successfully | Call `get_device(0)` with GPU present | Returns non-None device object, no exception | Yes |
| TC-003 | Device props dict has required keys | Call `get_device_props(device)` | Dict contains keys: `name`, `compute_capability`, `total_memory_gb`, `arch` | Yes |
| TC-004 | Device prop values are sensible | Inspect returned props dict | `total_memory_gb > 0.5`, `compute_capability` is a 2-tuple of ints | Yes |
| TC-005 | Vector-add result is correct | Call `run_vector_add(n=10_000)` | `result.correct is True`, `result.max_abs_error < 1e-5` | Yes |
| TC-006 | `compute_grid_1d` covers all elements | Call `compute_grid_1d(n, block)` for various n/block combos | `grid[0] * block_size >= n` for all inputs | No |
| TC-007 | `BenchmarkResult.summary_line()` format | Construct a `BenchmarkResult` and call `summary_line()` | String contains `GPU:`, `CPU:`, `Speedup:`, and `Correct:` substrings | No |

### Error Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-010 | `get_device` raises when cuda-python missing | Patch `cuda.core.experimental` import to raise `ImportError` | `get_device()` raises `RuntimeError` with helpful message | No |
| TC-011 | `get_device` raises when no GPU | Call `get_device(0)` on machine without NVIDIA GPU | Raises `RuntimeError` (not an unhandled `ImportError`) | No |

### Edge Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-020 | `compute_grid_1d` exact multiple | Call `compute_grid_1d(256, 256)` | Grid is `(1, 1, 1)`, covers exactly n elements | No |
| TC-021 | `compute_grid_1d` non-multiple | Call `compute_grid_1d(257, 256)` | Grid is `(2, 1, 1)`, covers all 257 elements | No |
| TC-022 | `compute_grid_1d` large n | Call `compute_grid_1d(1_000_001, 256)` | `grid[0] * 256 >= 1_000_001` | No |

---

## Acceptance Criteria Coverage

| AC from REQ-0001 | Test Case(s) | Status |
|------------------|--------------|--------|
| AC-1: Device name, compute capability, and VRAM printed | TC-002, TC-003, TC-004 | Pending |
| AC-2: Source shows cuda.core equivalent for each bindings call | Manual code review | Out of scope for automated tests |
| AC-3: Vector-add kernel launches and produces correct output vs NumPy | TC-005 | Pending |
| AC-4: No GPU prints clear error and exits with code 1 | TC-010, TC-011 | Pending |
| AC-5: No CUDA memory or pinned host memory leaked after demo | Manual inspection (no auto-leak detector in scope) | Pending |
| AC-6: Demo completes in under 30 seconds | TC-005 (implicit via run_vector_add timing) | Pending |
| AC-7: Pinned vs pageable benchmark prints GB/s for both | Manual integration test (pinned_memory.py) | Out of scope for unit tests |
| AC-8: Loads .npy from disk into pinned memory, confirms integrity | Manual integration test | Out of scope for unit tests |
| AC-9: Attempts URL fetch, falls back gracefully if offline | Manual integration test | Out of scope for unit tests |

---

## Out of Scope

- Testing that pinned memory is faster than pageable (requires GPU measurement)
- Multi-GPU device enumeration (single GPU only per REQ-0001)
- Network URL fetch demo (`load_from_url_to_device`) — no network in test environment
- Memory leak detection (no automated CUDA memory profiler in scope)
- End-to-end `main.py` CLI invocation tests

---

## Bugs Found

| Bug ID | Description | Severity | Status |
|--------|-------------|----------|--------|
| — | No bugs found during test plan authoring | — | — |
