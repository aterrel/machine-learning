# Test Plan — REQ-0003: Custom CUDA Kernel Authoring for ML Operations

**Author**: QA Agent
**Date**: 2026-05-01
**REQ Reference**: agents/requirements/REQ-0003.md
**Implementation**: Sprint 2 (main branch)

---

## Scope

This plan covers acceptance criteria AC-1 through AC-6 of REQ-0003, testing:
- GEMM kernel correctness versus NumPy matrix multiply
- ReLU kernel correctness versus `np.maximum(0, x)`
- Softmax kernel output (each row sums to 1.0)
- Kernel reuse: compile once, launch twice with different data
- NVRTC compilation time constraint (< 5 seconds per kernel)
- `CompiledKernel.launch()` with both scalar and pointer arguments

Tests are split into CPU-safe (no GPU required) and GPU-required categories.

---

## Test Environment

- Runtime: Python 3.11+, NumPy >= 1.26
- GPU: NVIDIA GPU with CUDA 12.x (required for GPU tests only)
- External services: None (offline safe)
- Test command:
  ```bash
  # All tests (GPU tests auto-skip if no GPU):
  pytest tests/test_kernels.py -v

  # CPU-only (CI safe):
  pytest tests/test_kernels.py -v -m "not gpu"

  # GPU tests only:
  pytest tests/test_kernels.py -v -m gpu
  ```

---

## Test Cases

### Happy Path

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-001 | GEMM correctness vs NumPy | Compile GEMM kernel, allocate M×K and K×N matrices, launch, copy back | `max(|gpu_result - np_result|) < 1e-3` for float32 | Yes |
| TC-002 | ReLU correctness vs NumPy | Compile ReLU kernel, allocate random float array with negative values, launch in-place | Result matches `np.maximum(0, x)` within 1e-5 | Yes |
| TC-003 | Softmax row-sum equals 1.0 | Compile softmax kernel, allocate batch × features matrix, launch row-wise softmax | Each row sums to `1.0 ± 1e-5` | Yes |
| TC-004 | Kernel reuse: compile once, launch twice | Compile GEMM kernel once, launch with matrix A, then launch with matrix B | Both results match NumPy; no re-compilation between launches | Yes |
| TC-005 | CompiledKernel.launch() with scalar args | Launch a kernel passing `np.int32` scalar as arg | Kernel uses scalar correctly; no exception raised | Yes |
| TC-006 | CompiledKernel.launch() with pointer args | Launch a kernel passing device buffer handle (int) as arg | Kernel reads/writes through pointer correctly; no exception | Yes |

### Performance Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-010 | NVRTC compile time < 5 seconds | Time `KernelCompiler.compile()` for each of GEMM, ReLU, softmax kernels | Each compile completes within 5 seconds wall clock | Yes |

### Error Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-020 | Invalid CUDA C source raises error | Pass syntactically invalid CUDA C to `KernelCompiler.compile()` | Raises an exception (not a silent failure) | Yes |

### Edge Cases

| ID | Scenario | Steps | Expected Result | GPU? |
|----|----------|-------|-----------------|------|
| TC-030 | ReLU all-positive input unchanged | Launch ReLU on array of all positive values | Output equals input | Yes |
| TC-031 | ReLU all-negative input zeroed | Launch ReLU on array of all negative values | Output is all zeros | Yes |
| TC-032 | Softmax single-row input sums to 1 | Launch softmax on a single-row (1 × n) matrix | Row sums to `1.0 ± 1e-5` | Yes |
| TC-033 | GEMM with 1×1 matrices | GEMM on (1,1) @ (1,1) | Result matches scalar multiplication | Yes |

---

## Acceptance Criteria Coverage

| AC from REQ-0003 | Test Case(s) | Status |
|------------------|--------------|--------|
| AC-1: GEMM kernel launches with correct output vs NumPy | TC-001, TC-033 | Pending |
| AC-2: ReLU and softmax kernels validate against reference | TC-002, TC-003, TC-030, TC-031, TC-032 | Pending |
| AC-3: Kernel source in Python strings, no separate .cu files | Manual code review of `demos/05_kernels/` | Out of scope for automated tests |
| AC-4: Demo configures launch grid for arbitrary input sizes | TC-001 (various M/N/K), TC-002 (various n) | Pending |
| AC-5: Fused linear+ReLU measurable latency reduction | Performance benchmark (manual, requires GPU) | Deferred |
| AC-6: NVRTC compile step clearly separated and timed | TC-010 | Pending |

---

## Test Data Strategy

**GEMM (TC-001, TC-004):**
```python
rng = np.random.default_rng(42)
A = rng.standard_normal((128, 64)).astype(np.float32)
B = rng.standard_normal((64, 128)).astype(np.float32)
expected = A @ B  # reference
```
Tolerance of 1e-3 accommodates float32 accumulation rounding in CUDA.

**ReLU (TC-002):**
```python
rng = np.random.default_rng(0)
x = rng.standard_normal(1_000_000).astype(np.float32)  # ~50% negative values
expected = np.maximum(0, x)
```

**Softmax (TC-003):**
```python
rng = np.random.default_rng(7)
X = rng.standard_normal((256, 128)).astype(np.float32)  # 256 rows × 128 features
# Each row of output must sum to 1.0 ± 1e-5
```

**Kernel reuse (TC-004):**
Two independently-generated matrices are fed sequentially to the same compiled kernel object to confirm state is not shared between launches.

---

## Out of Scope

- Tensor core (FP16/BF16) intrinsics
- PTX assembly authoring
- CUDA graphs
- Multi-GPU kernel dispatch
- Profiler integration (Nsight, nvtx)
- End-to-end `demos/05_kernels/main.py` CLI invocation

---

## Bugs Found

| Bug ID | Description | Severity | Status |
|--------|-------------|----------|--------|
| — | No bugs found during test plan authoring | — | — |
