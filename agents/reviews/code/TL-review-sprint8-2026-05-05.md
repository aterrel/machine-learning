---
review: TL-review-sprint8-2026-05-05
agent: Tech Lead
date: 2026-05-05
sprint: 8
commit: a33dbd9
---

# TL Review — Sprint 8 — PTX Kernel Execution Tracer
**Date**: 2026-05-05
**Verdict**: Approved
**Reviewer**: Tech Lead

---

## Scope

Sprint 8 deliverables: `src/kernel_model/ptx_tracer.py`, `_taxonomy.py`, `_arch_table.py`;
`demos/10_ptx_tracer/`; `tests/test_ptx_tracer.py`. REQ: REQ-0011. ARCH: ARCH-005 (Draft).

---

## Review Checklist

| Item | Result |
|------|--------|
| All P0 deliverables present | PASS — 9 files |
| Zero GPU imports at module level (NF1) | PASS |
| 13/13 CPU tests pass | PASS |
| 11/11 Sprint 7 regression tests pass | PASS |
| Demo runs without GPU in under 5 s | PASS (~0.04 s) |
| Classification: mma.sync → mma_warp | PASS |
| Classification: wgmma.mma_async → mma_warpgroup | PASS |
| Classification: tcgen05.mma → mma_cta | PASS |
| Longest-prefix match correct for all critical mnemonics | PASS |
| Architecture mismatch detection (tcgen05 on A100) | PASS |
| smem parsing (.shared regex) | PASS |
| tmem parsing (tcgen05.alloc n_cols × 512) | PASS |
| Async copy group counting (no double-count) | PASS |
| Bottleneck cycle-budget math matches ARCH-005 | PASS |
| ArchSpec values match ARCH-005 for sm_80/86/89/90/100 | PASS |
| TracerResult fields match REQ-0011 interface design | PASS |
| AC-1 through AC-8 all met | PASS |

---

## Findings

### Critical (block sprint close if any)

None.

### Major (fix before close)

None.

### Minor (fix or note)

**M-1: Unused imports in `src/kernel_model/ptx_tracer.py`**

`ptx_tracer.py` imports `field` (from `dataclasses`), `InstructionRecord`, `_INSTRUCTION_TAXONOMY`,
and `ArchSpec` but does not directly reference them in the module body. They are transitively
used (the `classify()` function uses `_INSTRUCTION_TAXONOMY` internally; `InstructionRecord` and
`ArchSpec` are the return types of the functions that are called), but a strict linter (ruff F401)
would flag these as unused direct imports. Recommend either removing the re-exports (`field`,
`InstructionRecord`, `_INSTRUCTION_TAXONOMY`, `ArchSpec`) or documenting them as intentional
public re-exports in `__all__`. Low risk — no functional impact.

**M-2: "other" category at 73.7% in vector_add.ptx demo output**

The vector_add.ptx fixture has 19 classified instructions; 14 fall into "other"
(`ld.param`, `mov.u32`, `setp`, `cvt`, `shl`, `add`, `ret`). This is correct behavior — the
taxonomy intentionally covers only hardware-resource-relevant categories — but the high "other"
percentage in the demo output may confuse users unfamiliar with PTX. A short explanatory comment
in `_print_result()` or a note in the demo header would improve clarity. No functional defect.

**M-3: `arithmetic_intensity` uses fixed 256 FLOPs-per-MMA heuristic**

`ptx_tracer.py` line 120: `mma_total * 256` as a floor estimate for MMA FLOP count. This is
documented as a rough estimate in the code and in ARCH-005 ("estimated" label in output), but
256 FLOPs/instruction is conservative (an m16n8k16 mma.sync is 16×8×16×2 = 4096 FLOPs for the
full warp). The REQ-0011-F10 (P1) says "estimated FLOP/byte", so this is acceptable for the
current sprint, but the value should be flagged as architecture-and-shape-dependent in a
follow-up. Not blocking.

**M-4: sm_70 smem_latency_cycles value not sourced in ARCH-005**

`_ARCH_TABLE["sm_70"]` uses `smem_latency_cycles=28` and `global_l2_latency_cycles=193`.
ARCH-005 only specifies sm_80 through sm_100; the programmer notes these were sourced from
the Volta whitepaper. The values are plausible (Volta smem latency ~28 cycles, L2 ~193 cycles
per ptx-tracer-research.md), but they are not cross-referenced in the spec. Recommend adding
an inline comment with the source citation. Not blocking.

---

## Acceptance Criteria Check

| AC | Requirement | Status | Verified by |
|----|-------------|--------|-------------|
| AC-1 | `from src.kernel_model import PTXTracer` succeeds on no-GPU machine | PASS | import check + test run |
| AC-2 | mma.sync against A100 → mma_warp_count==1, mma_warpgroup_count==0 | PASS | `test_trace_mma_sync` + manual verify |
| AC-3 | wgmma.mma_async against H100 → mma_warpgroup_count==1 | PASS | `test_trace_wgmma` |
| AC-4 | tcgen05.mma against RTX 5090 → mma_cta_count==1 | PASS | `test_trace_tcgen05` + manual verify (sm_100) |
| AC-5 | tcgen05.mma against A100 → warning in TracerResult.notes | PASS | `test_arch_mismatch_warning` + manual verify |
| AC-6 | Demo prints vector_add + GEMM vs A100/H100, no GPU | PASS | demo run confirmed |
| AC-7 | 5,000-line PTX < 500 ms | PASS | `test_performance_10k_lines`: 10k lines in ~40 ms |
| AC-8 | CPU tests cover all categories, arch mismatch, smem/tmem parsing | PASS | 13 test functions examined |

All 8 acceptance criteria met.

---

## Classification Correctness Verification

Longest-prefix match tested for all critical mnemonics:

| Mnemonic (abbreviated) | Expected | Result |
|------------------------|----------|--------|
| `mma.sync.aligned...` | mma_warp | PASS |
| `mma.sp.sync.aligned...` | mma_warp | PASS |
| `wgmma.mma_async.sync...` | mma_warpgroup | PASS |
| `tcgen05.mma.cta_group::1...` | mma_cta | PASS |
| `tcgen05.mma.sp.sync.aligned` | mma_cta | PASS (matches mma.sp before mma) |
| `tcgen05.mma.ws.sync.aligned` | mma_cta | PASS |
| `cp.async.bulk.commit_group` | async_tma | PASS (bulk before non-bulk) |
| `cp.async.commit_group` | async_copy | PASS (no double-count confirmed) |
| `ld.global.f32` | global_mem | PASS |
| `ld.shared.f32` | smem | PASS |
| `tcgen05.alloc.cta_group::1...` | tmem | PASS |

Double-count analysis: `"cp.async.bulk.commit_group"` does NOT contain
`"cp.async.commit_group"` as a substring (the `.bulk.` segment breaks the match), so
`ptx_source.count()` for both strings is safe. Verified programmatically.

---

## Architecture Table Verification

All sm_80/86/89/90/100 ArchSpec values match the ARCH-005 specification exactly:
smem latency, global L2 latency, mma_class, mma_warp_scope, max_warps_per_sm all correct.
`_MMA_LATENCY` values (sm_80=17, sm_90=64, sm_100=11) match microbenchmark sources cited
in ptx-tracer-research.md.

Bottleneck cycle-budget formula verified: 50 mma.sync + 1 ld.global on sm_80 → ratio 0.819
(compute); 50 ld.global → ratio 1.0 (memory). Both exceed the 0.6 threshold correctly.

---

## Summary

Sprint 8 is a clean, complete implementation. All 9 deliverables are present, all 8 acceptance
criteria are met, and all 13 CPU tests (plus 11 Sprint 7 regressions) pass. The PTX tracer
correctly distinguishes the three MMA scopes (warp/warpgroup/CTA) across Ampere through
Blackwell, detects architecture mismatches with informative notes, parses smem and tmem
allocations, and runs well within the performance budget (10k lines in 40 ms vs the 1 s limit).

The four minor findings (M-1 through M-4) are cosmetic or documentation gaps; none affects
correctness or test results. No changes are required before sprint close.

**Verdict: Approved.**
