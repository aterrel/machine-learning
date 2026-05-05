# Sprint 7 Retrospective — CUDA Python ML Demos

**Author**: Project Manager
**Date**: 2026-05-05
**Sprint**: 7
**Sprint Duration**: 2026-05-05 → 2026-05-05
**Tech Lead Verdict**: Conditional Approval

---

## Sprint Goal

Implement `src/kernel_model/` — a pure-Python library for GPU kernel occupancy and roofline analysis, plus `demos/09_kernel_model/` CLI demo and `tests/test_kernel_model.py`. Fully functional without a GPU. REQ-0010 / ARCH-004.

## Outcome

**Achieved**

All 7 planned deliverables delivered. 11 CPU tests pass, 1 GPU test marked appropriately. Demo runs without GPU. ARCH-004 critical condition (`sm_version: str` field in DeviceSpec) met. Tech Lead issued Conditional Approval — no critical or major findings, 7 minor findings tracked.

---

## Completed

| Task | Owner | Notes |
|------|-------|-------|
| `src/kernel_model/__init__.py` | Programmer | Re-exports: DeviceSpec, OccupancyModel, OccupancyResult, RooflineModel, RooflineResult |
| `src/kernel_model/device_spec.py` | Programmer | DeviceSpec + `sm_version` field + 8-SKU table + `from_name()` + `from_device()` |
| `src/kernel_model/occupancy.py` | Programmer | OccupancyResult + OccupancyModel with `compute()` + `sweep_block_sizes()` |
| `src/kernel_model/roofline.py` | Programmer | RooflineResult + RooflineModel with `compute()` + `sweep_intensities()` |
| `demos/09_kernel_model/__init__.py` | Programmer | Empty package init |
| `demos/09_kernel_model/main.py` | Programmer | CLI demo: occupancy table + roofline for vector-add on A100 |
| `tests/test_kernel_model.py` | Programmer | 11 CPU-safe tests + 1 `@pytest.mark.gpu` test |
| TL Sprint 7 code review | Tech Lead | Conditional Approval — 7 minor findings, M-1 fixed post-review |

---

## Not Completed (rolled to follow-on)

| Task | Owner | Reason | Priority |
|------|-------|--------|----------|
| Fix M-2: misleading "Memory bandwidth use" for compute-bound case | Programmer | Minor, non-blocking — follow-on cleanup | P2 |
| Fix M-3: verify RTX 5090 sm_version (sm_100 vs sm_120) | Programmer | Needs hardware datasheet verification | P2 |
| Fix M-4: block size validation uses SM capacity vs CUDA 1024 limit | Programmer | Known model approximation per ARCH-004 scope | P2 |
| Fix M-5: clarify `noqa: F401` comment on deferred Device import | Programmer | Cosmetic wording | P2 |
| Fix M-6: unnecessary `* 1.0` in `sweep_intensities` | Programmer | Cosmetic, correctness unaffected | P2 |
| Fix M-7: install ruff in venv; run `ruff check` before Sprint 8 | Programmer | ruff unavailable during Sprint 7 dev | P1 |

M-1 (dead `mem_bw_util` variable in main.py) was fixed in commit post-TL-review and is **closed**.

---

## Tech Lead Code Review Summary

**Verdict**: Conditional Approval — `TL-review-sprint7-2026-05-05.md`

Critical issues found: 0
Major issues found: 0
Minor issues found: 7

Issues resolved before close: M-1 (dead code `mem_bw_util`)
Issues tracked for Sprint 8 / follow-on: M-2, M-3, M-4, M-5, M-6, M-7

**Math verification (by TL)**:
- Occupancy three-limiter algorithm: PASS (A100 standard case, shmem limiter, register limiter — all verified by hand)
- `inf`-for-inactive-limiter fix: PASS
- Roofline ridge point: PASS (A100 80GB: 9.56 FLOP/byte)
- Roofline vector-add AI: PASS (memory-bound at 0.083 FLOP/byte)

**API consistency**: PASS — all public names, signatures, and dataclass fields match REQ-0010 exactly.

---

## Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| P0 deliverables | 7 | 7 |
| CPU tests written | 11 | 11 |
| GPU tests written | 1 | 1 |
| CPU test pass rate | 100% | 100% |
| Critical/Major TL findings | 0 | 0 |
| Minor TL findings | — | 7 |
| Ruff installed | Yes | No (M-7) |

---

## What Went Well

- All 7 deliverables completed in a single programmer pass with no re-work required.
- Three-limiter occupancy math was correct first time; TL math verification confirmed all edge cases.
- ARCH-004 critical condition (`sm_version: str`) was satisfied exactly — no clarification needed.
- GPU-free guarantee met throughout: zero GPU imports at module level, demo runs cleanly without hardware.
- `from_name()` normalization handles spaces, hyphens, and mixed case correctly (e.g., "a100-40gb", "RTX 4090").

---

## What Could Improve

- Ruff should be installed in the project venv before Sprint 8 begins (M-7). Line-length issues in 4 files were identified by TL manual review; `E501` is suppressed but other rules should be checked.
- M-3 (RTX 5090 sm_version) highlights that hardware datasheet verification should be a documented step in the ARCH template before any new SKU is added to the table.
- M-2 (compute-bound bandwidth label) is a demo output quality issue — the output is not wrong but is misleading. A pre-delivery self-review checklist for demo print statements would catch this class of issue.

---

## Sprint 8 Readiness Assessment

**Status**: READY TO OPEN

Prerequisites for Sprint 8 (REQ-0011 / ARCH-005):
- [x] `DeviceSpec` with `sm_version: str` field — delivered and verified (M-1 fixed)
- [x] `_ARCH_TABLE` keyed by `sm_version` can now integrate with `DeviceSpec.sm_version` directly
- [x] ARCH-005 Conditional Approval — `_MMA_LATENCY` values must be sourced from `ptx-tracer-research.md`; sm_86 ArchSpec placeholders must be filled
- [x] REQ-0011 authored with complete API spec, acceptance criteria, and 9 ordered deliverables
- [ ] M-7 (ruff install) should be resolved before or during Sprint 8

Sprint 8 goal: `src/kernel_model/ptx_tracer.py` — PTX instruction tracer covering Ampere, Ada, Hopper, and Blackwell. 9 deliverables, 13 CPU-safe tests.

---

## Sprint 8 Preview

**Goal**: PTX Kernel Execution Tracer — pure-Python scanner classifying PTX instruction mix by category with arch-specific annotations.

**P0 Tasks**:
- [ ] `src/kernel_model/_taxonomy.py` — `_INSTRUCTION_TAXONOMY` dict
- [ ] `src/kernel_model/_arch_table.py` — `ArchSpec` + `_ARCH_TABLE` for sm_80/86/89/90/100 + `_MMA_LATENCY`
- [ ] `src/kernel_model/ptx_tracer.py` — PTXTracer, TracerResult, InstructionRecord
- [ ] `src/kernel_model/__init__.py` — add PTXTracer, TracerResult to re-exports
- [ ] `demos/10_ptx_tracer/` — package init + PTX fixtures + CLI demo
- [ ] `tests/test_ptx_tracer.py` — 13 CPU-safe tests

**Agents assigned**:
- Programmer: all P0 deliverables (9 files)
- Tech Lead: Sprint 8 code review

*Approved by: Claude Manager*
*Sprint 8 kickoff: 2026-05-05*
