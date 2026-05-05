---
from: Programmer
to: Tech Lead
date: 2026-05-05
sprint: 7
---

# Handoff: Sprint 7 Implementation Complete — Kernel Performance Model

## What I Did

Implemented all Sprint 7 P0 deliverables for `src/kernel_model/` per REQ-0010 and ARCH-004 (Conditional Approval with `sm_version: str` condition met).

Commit: `[Prog] Sprint 7: implement src/kernel_model/ (occupancy + roofline library), demo 09, tests`

## Files Created

| File | Purpose |
|------|---------|
| `src/kernel_model/__init__.py` | Re-exports DeviceSpec, OccupancyModel/Result, RooflineModel/Result |
| `src/kernel_model/device_spec.py` | DeviceSpec dataclass + GPU SKU table + from_name() + from_device() |
| `src/kernel_model/occupancy.py` | OccupancyResult dataclass + OccupancyModel (compute + sweep_block_sizes) |
| `src/kernel_model/roofline.py` | RooflineResult dataclass + RooflineModel (compute + sweep_intensities) |
| `demos/09_kernel_model/__init__.py` | Empty package init |
| `demos/09_kernel_model/main.py` | CLI demo: occupancy table + roofline for vector-add kernel on A100 |
| `tests/test_kernel_model.py` | 11 CPU-safe unit tests + 1 @pytest.mark.gpu test |

## Test Results

```
pytest tests/test_kernel_model.py -v -m "not gpu"
11 passed, 1 deselected in 0.04s
```

## Acceptance Criteria Status

- AC-1: `from src.kernel_model import OccupancyModel, RooflineModel, DeviceSpec` — PASS (no GPU needed)
- AC-2: OccupancyModel compute on A100 returns valid OccupancyResult — PASS
- AC-3: RooflineModel compute on A100 returns valid RooflineResult — PASS
- AC-4: DeviceSpec.from_name("RTX 9999") raises KeyError with valid names list — PASS
- AC-5: `python demos/09_kernel_model/main.py` runs and prints table + summary without GPU — PASS
- AC-6: from_device() — requires GPU, covered by @pytest.mark.gpu test
- AC-7: CPU tests cover occupancy edge cases (too-large block, shmem limiter, register limiter) — PASS
- AC-8: CPU tests cover roofline edge cases (ridge point, memory-bound, compute-bound) — PASS

## Architecture Notes for TL Review

1. **ARCH-004 critical condition satisfied**: `DeviceSpec.sm_version: str` field is present in all SKU table entries and in `from_device()` (derived from `device.compute_capability`).

2. **Zero GPU imports at module level**: All three `src/kernel_model/` modules use only stdlib (`math`, `dataclasses`). The `cuda.core.experimental` import is deferred inside `DeviceSpec.from_device()`.

3. **Occupancy algorithm**: Implements exactly the three-limiter model specified in ARCH-004 — warps, shared memory, registers (with 256-register warp allocation granularity), plus max blocks/SM hardware limit.

4. **Register limiter**: `regs_per_warp_alloc = ceil(registers_per_thread * 32 / 256) * 256` — intentionally conservative per ARCH-004 design decision 2.

5. **from_name() normalization**: Strips whitespace, lowercases, removes spaces and hyphens before lookup. All aliases in `_GPU_TABLE` are already normalized. A secondary loop also normalizes the table alias keys for comparison.

6. **Tie-breaking in limiting_resource**: When two limiters are equal (e.g., block_size=32 with shmem=0 ties "shared_memory" and "blocks" at 32), the limiter returned is the first one in dict insertion order (warps → shared_memory → registers → blocks). The spec does not define tie-breaking behavior.

7. **Demo memory bandwidth line**: For memory-bound kernels, "Memory bandwidth use" is shown as `arithmetic_intensity * bandwidth_gbs` (effective bandwidth demand), not 100% of peak.

## Known Limitations

- `from_device()` for unknown GPU SKUs (not in table) raises `ValueError` with instructions to construct `DeviceSpec` manually — per ARCH-004 design decision 3.
- ruff was not available in the venv; all files were syntax-checked via `py_compile`. Code follows project style conventions (ruff line-length=100, no unused imports, type annotations).

## What TL Needs to Review

1. Occupancy math correctness — spot-check the limiter calculations against NVIDIA's occupancy calculator
2. GPU SKU table accuracy — verify register counts, SM counts, bandwidth figures against NVIDIA datasheets
3. Test coverage adequacy — 11 CPU tests + 1 GPU test
4. `from_name()` normalization logic — handles "A100 80GB", "a100-80gb", "a100" aliases
5. Demo output format matches ARCH-004 spec approximately (exact column widths may differ)
