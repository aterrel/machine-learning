# Handoff: Programmer → Tech Lead — Sprint 8 Complete

**Date**: 2026-05-05
**From**: Programmer
**To**: Tech Lead
**Sprint**: Sprint 8 — PTX Kernel Execution Tracer (REQ-0011)
**Status**: Implementation Complete — Ready for Review

---

## Summary

All 9 Sprint 8 deliverables implemented and tested. The pure-Python PTX instruction tracer
(`PTXTracer`) is fully operational with zero GPU dependencies.

---

## Files Created / Modified

### New files
| File | Description |
|------|-------------|
| `src/kernel_model/_taxonomy.py` | `InstructionRecord` dataclass + `_INSTRUCTION_TAXONOMY` dict (35 entries across 10 categories) + `classify()` longest-prefix lookup |
| `src/kernel_model/_arch_table.py` | `ArchSpec` dataclass + `_ARCH_TABLE` (sm_70/80/86/89/90/100) + `_MMA_LATENCY` dict |
| `src/kernel_model/ptx_tracer.py` | `PTXTracer`, `TracerResult`; implements `trace()`, `trace_file()`, `bottleneck()` |
| `demos/10_ptx_tracer/__init__.py` | Empty package init |
| `demos/10_ptx_tracer/ptx_fixtures/vector_add.ptx` | Minimal handwritten PTX for vector-add (sm_80) |
| `demos/10_ptx_tracer/ptx_fixtures/gemm_mma.ptx` | Minimal handwritten PTX with mma.sync (sm_80) |
| `demos/10_ptx_tracer/main.py` | CLI demo: traces both fixtures vs A100 and H100 |
| `tests/test_ptx_tracer.py` | 13 CPU-safe unit tests |

### Modified files
| File | Change |
|------|--------|
| `src/kernel_model/__init__.py` | Added `PTXTracer`, `TracerResult` to re-exports and `__all__` |
| `agents/todo.md` | Marked all Sprint 8 Programmer tasks `[x]` |

---

## Test Results

```
tests/test_ptx_tracer.py  — 13/13 passed (not gpu)
tests/test_kernel_model.py — 11/11 passed (not gpu)   ← Sprint 7 regression: green
```

All 13 required tests pass, including:
- `test_trace_vector_add` — global_mem ≥ 2, fused_fp ≥ 1, all MMA counts = 0
- `test_trace_mma_sync` — mma_warp_count == 1 on sm_80
- `test_trace_wgmma` — mma_warpgroup_count == 1 on sm_90
- `test_trace_tcgen05` — mma_cta_count == 1 on sm_100
- `test_arch_mismatch_warning` — tcgen05 flagged as unsupported on sm_80
- `test_smem_declaration_parse` — 8192 bytes detected from `.shared` declaration
- `test_tmem_alloc_parse` — 32768 bytes (64 cols × 512) from tcgen05.alloc
- `test_async_copy_group_count` — 3 cp.async.commit_group counted correctly
- `test_unknown_instruction` — unknown mnemonic lands in "other", no exception
- `test_bottleneck_compute` — 50 mma.sync → "compute"
- `test_bottleneck_memory` — 50 ld.global → "memory"
- `test_performance_10k_lines` — 10k lines traced in < 1.0s (actual: ~0.04s)
- `test_arch_table_coverage` — all 8 SKUs (v100→rtx5090) have _ARCH_TABLE entries

---

## Demo Output

`python demos/10_ptx_tracer/main.py` produces a structured per-fixture, per-architecture
report including instruction mix, MMA scope breakdown, smem/tmem allocation, async copy groups,
estimated arithmetic intensity, and bottleneck classification. Runs in < 1 second, no GPU.

---

## Implementation Notes

1. **_taxonomy.py extended with sm_70**: `_ARCH_TABLE` includes sm_70 (V100) to satisfy
   `test_arch_table_coverage`. The ARCH-005 spec only listed sm_80+, but V100 is in the
   DeviceSpec SKU table and the test exercises all SKUs.

2. **Longest-prefix match correctness**: `tcgen05.mma.sp.sync.aligned` correctly matches
   `tcgen05.mma.sp` before `tcgen05.mma`. `cp.async.bulk.commit_group` matches
   `cp.async.bulk.commit_group` before `cp.async.bulk`.

3. **PTX directive skipping**: Lines without leading whitespace (labels, `.version`, `.target`,
   `.visible`, `.entry`, `.reg`, `.param`, `.shared`) are skipped via the
   `r'^\s+([\w.]+)'` regex — only indented instructions are classified.

4. **cp.async.commit_group double-count**: `cp.async.bulk.commit_group` does NOT contain
   `cp.async.commit_group` as a substring (the `.bulk.` is between them), so the two
   `ptx_source.count()` calls are safe with no double-counting.

5. **TracerResult.tmem_alloc_bytes**: Only non-None for sm_100 targets. The `tcgen05.alloc`
   regex captures the column argument; bytes = n_cols × 512 (128 lanes × 4 bytes).

---

## Acceptance Criteria Status

| AC | Status |
|----|--------|
| AC-1: `from src.kernel_model import PTXTracer` on no-GPU machine | PASS |
| AC-2: mma.sync → mma_warp_count == 1, mma_warpgroup_count == 0 | PASS |
| AC-3: wgmma.mma_async → mma_warpgroup_count == 1 | PASS |
| AC-4: tcgen05.mma on sm_100 → mma_cta_count == 1 | PASS |
| AC-5: tcgen05.mma on sm_80 → note in TracerResult.notes | PASS |
| AC-6: demo prints for vector-add and GEMM vs A100/H100, no GPU | PASS |
| AC-7: 5,000-line PTX < 500 ms (10k-line test: ~40 ms actual) | PASS |
| AC-8: CPU tests cover all categories, arch mismatch, smem/tmem parsing | PASS |

---

## Review Focus Areas for Tech Lead

1. **Bottleneck logic** — `bottleneck()` in `ptx_tracer.py` uses the cycle-budget model from
   ARCH-005. The `gemm_mma.ptx` fixture classifies as "memory" because the fixture has more
   global_mem ops than MMA ops (minimal PTX); a real GEMM would be compute-bound. This is
   expected behavior for the given fixture.

2. **sm_70 in _ARCH_TABLE** — Added to pass `test_arch_table_coverage`. Values sourced from
   Volta whitepaper (23-cycle smem, 193-cycle global L2, mma_sync class, 512 FP16 FMAs/SM/clk).
   TL should verify these are reasonable.

3. **async_copy_groups counting** — Uses simple `str.count()` on the full source. This is
   correct and fast; the alternative (counting classified instructions per line) would
   under-count because `cp.async.commit_group` appears as a single line without operands.
