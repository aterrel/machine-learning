---
review: TL-review-sprint7-2026-05-05
agent: Tech Lead
date: 2026-05-05
sprint: 7
commit: post-7376db8
---

# TL Review — Sprint 7 — Kernel Performance Model
**Date**: 2026-05-05
**Verdict**: Conditional Approval
**Reviewer**: Tech Lead

---

## Scope

Sprint 7 deliverables: `src/kernel_model/` — pure-Python occupancy + roofline library; `demos/09_kernel_model/`; `tests/test_kernel_model.py`. REQ: REQ-0010. ARCH: ARCH-004 (Conditional Approval — `sm_version: str` required).

---

## Review Checklist

| Item | Result |
|------|--------|
| All P0 deliverables present | PASS — 7 files |
| ARCH-004 critical condition: `sm_version: str` in DeviceSpec | PASS |
| Zero GPU imports at module level | PASS |
| 11/11 CPU tests pass | PASS |
| Demo runs without GPU | PASS |
| Occupancy three-limiter algorithm correct | PASS |
| Roofline math correct | PASS |
| REQ-0010 API matches spec exactly | PASS |
| Ruff installed and clean | N/A — ruff not installed in venv; manual style check performed |

---

## Findings

### Critical (block sprint close if any)

None.

### Major (fix before close)

None.

### Minor (fix or note)

**M-1: Dead code in `demos/09_kernel_model/main.py` line 47**

`mem_bw_util` is computed but never used or printed:

```python
mem_bw_util = (roof.predicted_gflops / device.memory_bandwidth_gbs) * 1000 if roof.bound == "memory" else 0.0
```

The actual bandwidth display uses a separately-computed `effective_bw` on line 54. `mem_bw_util` should be removed. This would also be caught by ruff F841 (local variable assigned but never used) once ruff is installed.

**M-2: Misleading `Memory bandwidth use` for compute-bound kernels (demo line 57-58)**

In the `else` branch (compute-bound), the demo prints `device.memory_bandwidth_gbs` as "Memory bandwidth use" — implying 100% peak bandwidth utilization when the kernel is actually compute-bound. Either remove this line for the compute-bound case or label it differently (e.g., "Peak memory bandwidth").

**M-3: `sm_version` for RTX 5090 may be incorrect**

RTX 5090 uses the GB202 die (desktop Blackwell), which carries compute capability sm_120, not sm_100. sm_100 is correct for server Blackwell (GB100/B100). The current table assigns sm_100 to both B100 and RTX 5090. This should be verified against NVIDIA documentation and corrected to sm_120 for RTX 5090 if confirmed. ARCH-004 already flagged B100/RTX 5090 as needing datasheet verification.

**M-4: Block size validation uses SM-capacity threshold instead of CUDA block limit**

`occupancy.py` line 31: `max_block_threads = dev.max_warps_per_sm * 32`. On A100 this is 2048, but the CUDA hardware limit for block size is 1024 threads per block. The current check will not raise `ValueError` for block sizes 1025–2048 on A100, even though CUDA itself would reject them. Since `DeviceSpec` does not include a `max_threads_per_block` field (not in ARCH-004), this is a known model approximation. The error message correctly describes the limit as "max threads per SM" (not "per block"), so the messaging is accurate for what the model checks. Low impact for an analytical model, but worth noting for future DeviceSpec additions.

**M-5: `noqa: F401` comment on deferred `Device` import**

`device_spec.py` line 41: `from cuda.core.experimental import Device  # noqa: F401 — deferred import`. The `Device` class is imported but not used in the method body; the import serves only as an availability guard (fails fast if cuda.core not installed). The `noqa: F401` suppression is appropriate and necessary for ruff. The inline comment could be clearer: "guard import — verify cuda.core.experimental is available" rather than "deferred import" (the import IS deferred to the method, but the class itself is unused).

**M-6: `sweep_intensities` uses a stylistically unusual calling convention**

`roofline.py` line 40: `self.compute(flops=intensity * 1.0, bytes_accessed=1.0)`. Multiplying by `1.0` is unnecessary (intensity is already a float in a `list[float]`). The logic is correct (AI = intensity/1.0 = intensity) but the expression is mildly confusing. Equivalent and cleaner: `self.compute(flops=intensity, bytes_accessed=1.0)`.

**M-7: Ruff not installed in venv**

The handoff notes ruff was unavailable during development; a manual syntax check via `py_compile` was used instead. The venv should have ruff installed per `pyproject.toml` dev dependencies (`ruff>=0.4`). Lines exceeding 100 characters were found in device_spec.py (line 44, 101 chars), roofline.py (line 40, 105 chars), demos/09_kernel_model/main.py (line 47, 113 chars), and tests/test_kernel_model.py (line 34, 101 chars). Since `E501` is suppressed in pyproject.toml, these are not violations, but `ruff check` should be run before the next sprint to confirm no other issues.

---

## Acceptance Criteria Check

| AC | Description | Result |
|----|-------------|--------|
| AC-1 | `from src.kernel_model import OccupancyModel, RooflineModel, DeviceSpec` on no-GPU machine | PASS |
| AC-2 | `OccupancyModel(DeviceSpec.from_name("A100")).compute(256, 0, 32)` returns `OccupancyResult` with occupancy in [0,1]; verified 1.0 | PASS |
| AC-3 | `RooflineModel(DeviceSpec.from_name("A100")).compute(1e9, 1e8)` returns `RooflineResult` with valid `bound` and positive `predicted_gflops` | PASS |
| AC-4 | `DeviceSpec.from_name("RTX 9999")` raises `KeyError` with valid names in message | PASS |
| AC-5 | `python demos/09_kernel_model/main.py` prints occupancy table + roofline summary without GPU | PASS |
| AC-6 | `DeviceSpec.from_device(device)` populates all fields from live device | PASS (GPU test deselected in this env; logic verified by code inspection) |
| AC-7 | CPU tests cover occupancy edge cases: too-large block, zero shmem, register limiter, shmem limiter | PASS |
| AC-8 | CPU tests cover roofline edge cases: memory-bound, compute-bound, at ridge point | PASS |

---

## Math Verification

**Occupancy (A100, block_size=256, shmem=0, regs=32)**
- warps_per_block = ceil(256/32) = 8
- max_blocks_by_warps = floor(64/8) = 8
- max_blocks_by_shmem = inf (shmem=0)
- regs_per_warp_alloc = ceil(32*32/256)*256 = 1024
- regs_per_block = 1024 * 8 = 8192
- max_blocks_by_regs = floor(65536/8192) = 8
- active_blocks = min(8, inf, 8, 32) = 8
- active_warps = 8 * 8 = 64
- occupancy = 64/64 = 1.0 **CORRECT**

**Occupancy — shmem limiter (A100, block_size=256, shmem=40000)**
- max_blocks_by_shmem = floor(167936/40000) = 4; limits to 4 blocks **CORRECT**

**Occupancy — register limiter (A100, block_size=256, regs=255)**
- regs_per_warp_alloc = ceil(255*32/256)*256 = 8192
- regs_per_block = 8192 * 8 = 65536
- max_blocks_by_regs = floor(65536/65536) = 1 **CORRECT**

**Roofline (A100 80GB)**
- ridge_point = 19500.0 / 2039.0 = 9.5635 FLOP/byte
- vector_add: AI = 1M / (3 * 1M * 4) = 0.0833 FLOP/byte < ridge → memory-bound **CORRECT**
- predicted_gflops = 0.0833 * 2039 = 169.9 GFLOP/s **CORRECT**

**`inf`-for-inactive-limiter fix**: When shmem_per_block=0 or regs_per_block=0, inactive limiters correctly return `float("inf")` rather than `max_blocks_per_sm`. The `min()` then ignores them properly. **CORRECT — bug fix verified.**

---

## API Consistency

All public names, signatures, and dataclass fields match REQ-0010 API specification exactly:
- `OccupancyResult`: all 7 fields present with correct names and types
- `RooflineResult`: all 4 fields present with correct names and types
- `DeviceSpec`: all 8 REQ-0010 fields + `sm_version` (ARCH-004 requirement) present
- `OccupancyModel.compute()`, `sweep_block_sizes()`: signatures match spec
- `RooflineModel.compute()`, `sweep_intensities()`: signatures match spec
- `from_name()` raises `KeyError` with valid names; `from_device()` defers GPU import

`__init__.py` re-exports: `DeviceSpec`, `OccupancyModel`, `OccupancyResult`, `RooflineModel`, `RooflineResult` — all correct.

---

## Code Quality

- Stdlib-only at module level (`math`, `dataclasses`): confirmed
- Type annotations present throughout
- No dead imports at module level
- `from __future__ import annotations` used correctly for forward references
- `_register()` helper for SKU table is clean and avoids repetition
- `from_name()` normalization handles spaces, hyphens, and mixed case correctly (tested: "RTX 4090", "A100 80GB", "a100-40gb" all resolve correctly)
- Dead code: `mem_bw_util` in main.py (M-1 above)

---

## Summary

Sprint 7 delivers a well-structured, mathematically correct pure-Python occupancy and roofline library. All 11 CPU tests pass, the demo runs cleanly, the ARCH-004 critical condition (`sm_version: str`) is met, and zero GPU imports exist at module level. Seven minor issues were identified (dead code, one possibly incorrect sm_version for RTX 5090, minor demo output misleading for compute-bound case, unnecessary `* 1.0` in sweep_intensities, and ruff not yet installed in venv); none block sprint closure. Sprint 7 is conditionally approved — minor findings M-1 through M-7 should be addressed in a follow-on cleanup pass or early in Sprint 8.
