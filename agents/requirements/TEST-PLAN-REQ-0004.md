# Test Plan — REQ-0004: CPU vs GPU Benchmarking with Speedup Metrics

**Author**: QA Agent
**Date**: 2026-05-01
**REQ Reference**: agents/requirements/REQ-0004.md
**Implementation**: Sprint 2 (main branch)

---

## Scope

This plan covers acceptance criteria AC-1 through AC-6 of REQ-0004, testing:
- `BenchmarkResult` dataclass field presence and types
- `BenchmarkResult.summary_line()` output format
- `BenchmarkRunner.time_cpu()` timing accuracy for CPU functions
- `BenchmarkRunner.time_gpu()` timing using CUDA events
- Correctness flag (`correct: True`) when outputs match within tolerance
- Correctness flag (`correct: False`) when outputs diverge
- Graceful OOM handling (error row printed, benchmark continues)

Tests are split into CPU-safe (no GPU required) and GPU-required categories.

---

## Test Environment

- Runtime: Python 3.11+, NumPy >= 1.26
- GPU: NVIDIA GPU with CUDA 12.x (required for GPU tests only)
- External services: None (offline safe)
- Test command:
  ```bash
  # All tests (GPU tests auto-skip if no GPU):
  pytest tests/test_benchmarks.py -v

  # CPU-only (CI safe):
  pytest tests/test_benchmarks.py -v -m "not gpu"

  # GPU tests only:
  pytest tests/test_benchmarks.py -v -m gpu
  ```

---

## Test Cases

### Happy Path

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-001 | BenchmarkResult fields present | Construct `BenchmarkResult` with all required fields | All attributes accessible with correct types | No |
| TC-002 | summary_line() format | Construct `BenchmarkResult`, call `.summary_line()` | String contains `"GPU:"`, `"CPU:"`, `"Speedup:"`, `"Correct:"` | No |
| TC-003 | summary_line() speedup value | Construct with `cpu_time_mean_s=2.0, gpu_time_mean_s=0.5` | String contains `"4.0x"` speedup | No |
| TC-004 | summary_line() correct=True | Construct with `correct=True` | String contains `"Correct: True"` | No |
| TC-005 | summary_line() correct=False | Construct with `correct=False` | String contains `"Correct: False"` | No |
| TC-006 | BenchmarkRunner.time_cpu() returns float | Call `time_cpu(fn)` with a simple lambda | Returns `float`, value > 0 | No |
| TC-007 | BenchmarkRunner.time_cpu() measures non-trivially | Call `time_cpu` on a 10ms sleep function | Returned time >= 0.01 seconds | No |
| TC-008 | correct=True when outputs match | Build BenchmarkResult where `max_abs_error < tolerance` | `correct is True` | No |
| TC-009 | correct=False when outputs diverge | Build BenchmarkResult where `max_abs_error >= tolerance` | `correct is False` | No |
| TC-010 | BenchmarkRunner.time_gpu() uses CUDA events | Call `time_gpu(fn, stream)` on a GPU kernel function | Returns float > 0; no Python-only timing | Yes |

### Error Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-020 | time_gpu() raises when cuda-python missing | Patch `cuda.bindings` import to fail, call `time_gpu()` | Raises `RuntimeError` with clear message | No |
| TC-021 | OOM handled gracefully | Set artificially large n_samples to trigger CUDA OOM | Prints error row for failed demo, continues to next benchmark | Yes |

### Edge Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-030 | speedup field is ratio of cpu to gpu time | Construct `BenchmarkResult(cpu=1.0, gpu=0.25, speedup=4.0)` | `result.speedup == 4.0` | No |
| TC-031 | max_abs_error=0.0 → correct=True | Set `max_abs_error=0.0`, tolerance=1e-5 | `correct is True` | No |
| TC-032 | n_repeats=1 still returns valid time | Construct `BenchmarkRunner(n_repeats=1)`, call `time_cpu` | Returns valid float, no exception | No |
| TC-033 | warmup=0 still returns valid time | Construct `BenchmarkRunner(n_repeats=3, warmup=0)`, call `time_cpu` | Returns valid float, no exception | No |

---

## Acceptance Criteria Coverage

| AC from REQ-0004 | Test Case(s) | Status |
|------------------|--------------|--------|
| AC-1: Table with Algorithm, CPU time, GPU time, Speedup, Correct | TC-001, TC-002 (format), integration test of `benchmarks/run_all.py` | Pending |
| AC-2: GPU timing uses CUDA event records | TC-010 (verify source + runtime behavior) | Pending |
| AC-3: Same benchmark with same seed produces timing within 10% variance | Manual performance test (requires GPU, non-deterministic) | Deferred |
| AC-4: Incorrect GPU output reports `Correct: False` | TC-005, TC-009 | Pending |
| AC-5: JSON file written to benchmarks/results/ with required fields | Integration test of `benchmarks/run_all.py` | Deferred |
| AC-6: OOM prints error row and continues | TC-021 | Pending |

---

## Test Data Strategy

**BenchmarkResult construction (TC-001 through TC-009):**
```python
from src.utils.timing import BenchmarkResult

result = BenchmarkResult(
    demo_name="test_demo",
    cpu_time_mean_s=1.0,
    gpu_time_mean_s=0.25,
    speedup=4.0,
    correct=True,
    max_abs_error=0.0,
)
```
All CPU-only tests construct `BenchmarkResult` directly without touching GPU code.

**BenchmarkRunner CPU timing (TC-006, TC-007):**
```python
import time
from src.utils.timing import BenchmarkRunner

runner = BenchmarkRunner(n_repeats=3, warmup=1)
elapsed = runner.time_cpu(lambda: time.sleep(0.01))
assert elapsed >= 0.01
```

**Correctness flag construction (TC-008, TC-009):**
The `correct` field is set directly on `BenchmarkResult` at construction — not computed internally. Tests verify the field is respected when `summary_line()` formats output.

---

## Out of Scope

- Multi-GPU scaling benchmarks
- Profiler integration (Nsight, nvtx ranges)
- Automated regression detection across runs
- Memory bandwidth utilization reporting
- End-to-end `benchmarks/run_all.py` CLI invocation (integration test, not unit test)

---

## Bugs Found

| Bug ID | Description | Severity | Status |
|--------|-------------|----------|--------|
| — | No bugs found during test plan authoring | — | — |
