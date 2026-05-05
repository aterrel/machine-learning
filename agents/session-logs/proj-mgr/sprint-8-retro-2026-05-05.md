---
session: sprint-8-retro
agent: Project Manager
date: 2026-05-05
sprint: 8
verdict: Approved
---

# Sprint 8 Retrospective — PTX Kernel Execution Tracer

**Date**: 2026-05-05
**Sprint**: 8
**TL Verdict**: Approved
**Commit**: a33dbd9 (TL review at 87b07d4)

---

## Goal

Implement `src/kernel_model/ptx_tracer.py` — a pure-Python PTX instruction scanner classifying instruction mix by category (smem, tmem, mma_warp, mma_warpgroup, mma_cta, async_copy, async_tma, fused_fp, global_mem) with architecture-specific annotations covering Ampere, Ada, Hopper, and Blackwell.

**REQ ref**: REQ-0011 | **ARCH ref**: ARCH-005 (Conditional Approval)

---

## Deliverables

9 files delivered (all P0):

| # | File | Status |
|---|------|--------|
| 1 | `src/kernel_model/_taxonomy.py` | Done |
| 2 | `src/kernel_model/_arch_table.py` | Done |
| 3 | `src/kernel_model/ptx_tracer.py` | Done |
| 4 | `src/kernel_model/__init__.py` (updated) | Done |
| 5 | `demos/10_ptx_tracer/__init__.py` | Done |
| 6 | `demos/10_ptx_tracer/ptx_fixtures/vector_add.ptx` | Done |
| 7 | `demos/10_ptx_tracer/ptx_fixtures/gemm_mma.ptx` | Done |
| 8 | `demos/10_ptx_tracer/main.py` | Done |
| 9 | `tests/test_ptx_tracer.py` | Done |

---

## Test Results

- **New CPU tests**: 13 (all pass)
- **Regression**: 11 Sprint 7 CPU tests — all pass (0 regressions)
- **Total CPU tests**: 96 (72 Sprint 1–5 baseline + 11 Sprint 7 + 13 Sprint 8)
- **GPU tests**: 26 (unchanged — no GPU available in CI)

---

## TL Findings Summary

**Verdict: Approved** — 0 critical, 0 major, 4 minor (none blocking)

| Finding | Severity | Status |
|---------|----------|--------|
| M-1: Unused imports in `ptx_tracer.py` (ruff F401 may flag) | Minor | Open (non-blocking) |
| M-2: 73.7% "other" category in vector_add demo output may confuse users | Minor | Open (non-blocking) |
| M-3: Fixed 256 FLOPs-per-MMA heuristic is conservative and shape-dependent | Minor | Open (non-blocking) |
| M-4: sm_70 smem_latency/L2 values not cross-referenced in ARCH-005 | Minor | Open (non-blocking) |

All 8 acceptance criteria met. Demo runs in ~0.04 s (budget: 5 s). 10k-line PTX parsed in ~40 ms (budget: 500 ms).

---

## Key Design Decisions

1. **Longest-prefix mnemonic matching**: PTXTracer scans each PTX line against taxonomy keys sorted by descending length, ensuring `tcgen05.mma.sp.sync` matches `tcgen05.mma` (mma_cta) before `mma` (mma_warp). This correctly distinguishes all three MMA scopes.

2. **Cycle-budget bottleneck**: `bottleneck()` computes compute vs memory cycle budgets using ArchSpec values (`_MMA_LATENCY`, `global_l2_latency_cycles`) sourced from ptx-tracer-research.md microbenchmarks. Ratio > 0.6 → compute-bound; ≤ 0.6 → memory-bound.

3. **sm_70 (V100) added to `_ARCH_TABLE`**: ARCH-005 spec covered sm_80–sm_100. Sprint 8 extended the table to include sm_70 for V100 coverage, using Volta whitepaper latency values. This aligns with the `sm_version` field added to `DeviceSpec` in Sprint 7.

4. **Architecture mismatch detection**: Tracing Blackwell-only instructions (tcgen05.mma) against older architectures (A100, sm_80) correctly populates `TracerResult.unsupported_instructions` and adds a human-readable note.

---

## What Went Well

- Clean implementation with zero GPU imports at module level — the GPU-guard discipline established in Sprint 7 carried forward.
- All acceptance criteria met on first delivery; no mandatory fixes before sprint close.
- Performance budget (500 ms for 5k-line PTX) exceeded: 10k lines processed in 40 ms.
- Demo output clearly contrasts vector-add (global_mem-bound) vs GEMM-with-MMA (compute-bound) on both A100 and H100.

## What Could Improve

- M-1 (linter): Clarifying whether re-exported symbols from taxonomy/arch_table modules are intentional public API would prevent future ruff confusion.
- M-4 (citation): Inline source citations for sm_70 latency values would prevent future reviewers from questioning the numbers.

---

## Next: Sprint 9

**Sprint 9 — OPEN: CI/CD Pipeline + Jupyter Notebooks**

- REQ-0012: `.github/workflows/ci.yml` + `.github/workflows/gpu-ci.yml`
- REQ-0013: `notebooks/01_core_apis.ipynb` + `notebooks/02_kmeans.ipynb`
- 4 Programmer tasks + 1 TL review task
- CI workflow files are valid YAML and can be pushed to GitHub once a remote is configured
