# ARCH-005: PTX Kernel Execution Tracer (src/kernel_model/ptx_tracer.py)

**Status**: Draft
**Created**: 2026-05-05
**Author**: Software Architect
**REQ Reference**: REQ-0011
**Depends On**: ARCH-004 (DeviceSpec, OccupancyModel, RooflineModel)
**Approved By**: —

---

## Overview

`PTXTracer` is a pure-Python analytical tool that scans a PTX assembly source string, classifies every instruction into a hardware-relevant category, annotates each category with architecture-specific throughput and latency from a static table, and returns a `TracerResult` describing the instruction mix, memory allocation footprint, and estimated bottleneck. It extends `src/kernel_model/` (ARCH-004) without modifying existing classes.

---

## Context

The roofline model (ARCH-004) answers whether a kernel is compute-bound or memory-bound given its arithmetic intensity. The PTX tracer answers a finer question: *which specific instruction classes are present, in what proportion, and how do those classes map to hardware resources on a given SM architecture?* This matters because modern GPU architectures have fundamentally different MMA scopes and accumulator locations:

| Generation | MMA instruction | Scope | Accumulator |
|---|---|---|---|
| Ampere / Ada | `mma.sync` | 1 warp (32 threads) | Register file |
| Hopper | `wgmma.mma_async` | 1 warpgroup (128 threads) | Register file |
| Blackwell | `tcgen05.mma` | 1 thread issues (CTA executes) | TMEM (256 KB/SM) |

A PTX file targeting Hopper that uses `wgmma` cannot be profiled by an Ampere occupancy model alone — it requires knowing that MMA is warpgroup-scoped and that the accumulator register pressure is distributed across 128 threads, not 32.

---

## Design

### Module Structure (additions to ARCH-004 layout)

```
src/
  kernel_model/
    __init__.py           # add PTXTracer, TracerResult to re-exports
    device_spec.py        # existing (ARCH-004)
    occupancy.py          # existing (ARCH-004)
    roofline.py           # existing (ARCH-004)
    ptx_tracer.py         # new: PTXTracer + TracerResult + InstructionRecord
    _taxonomy.py          # new: _INSTRUCTION_TAXONOMY prefix table
    _arch_table.py        # new: _ARCH_TABLE per-SM spec dict
demos/
  10_ptx_tracer/
    __init__.py
    main.py               # new: demo — trace vector-add + GEMM vs A100 and H100
    ptx_fixtures/
      vector_add.ptx      # minimal handwritten PTX
      gemm_mma.ptx        # minimal PTX showing mma.sync
tests/
  test_ptx_tracer.py      # new: CPU-safe unit tests
```

---

## Key Data Structures

### ArchSpec (in `_arch_table.py`)

```python
@dataclass
class ArchSpec:
    sm_version: str              # "sm_80" | "sm_86" | "sm_89" | "sm_90" | "sm_100"
    max_warps_per_sm: int        # 64 for sm_80/sm_90/sm_100; 48 for sm_86/sm_89
    warp_schedulers: int         # always 4
    fp32_fmas_per_sm_clock: int  # CUDA core FFMA throughput
    smem_latency_cycles: int     # approximate ld.shared latency
    global_l2_latency_cycles: int
    mma_class: str               # "mma_sync" | "wgmma" | "tcgen05"
    mma_warp_scope: int          # threads per MMA: 32 (mma.sync) | 128 (wgmma) | 1 issue (tcgen05)
    mma_fp16_throughput: int     # FP16 FMAs/SM/clock via Tensor Cores
    mma_fp8_throughput: int | None
    wgmma_supported: bool
    tmem_supported: bool
    tmem_kb_per_sm: int | None   # None unless tmem_supported
    fp8_mma_supported: bool
```

`_ARCH_TABLE`:

```python
_ARCH_TABLE = {
    "sm_80": ArchSpec(
        sm_version="sm_80", max_warps_per_sm=64, warp_schedulers=4,
        fp32_fmas_per_sm_clock=64, smem_latency_cycles=23, global_l2_latency_cycles=188,
        mma_class="mma_sync", mma_warp_scope=32, mma_fp16_throughput=1024,
        mma_fp8_throughput=None, wgmma_supported=False, tmem_supported=False,
        tmem_kb_per_sm=None, fp8_mma_supported=False,
    ),
    "sm_86": ArchSpec(..., max_warps_per_sm=48, fp32_fmas_per_sm_clock=128,
                      mma_fp16_throughput=1024, ...),
    "sm_89": ArchSpec(..., max_warps_per_sm=48, fp32_fmas_per_sm_clock=128,
                      mma_fp16_throughput=1024, mma_fp8_throughput=2048,
                      fp8_mma_supported=True, ...),
    "sm_90": ArchSpec(
        sm_version="sm_90", max_warps_per_sm=64, warp_schedulers=4,
        fp32_fmas_per_sm_clock=128, smem_latency_cycles=30, global_l2_latency_cycles=273,
        mma_class="wgmma", mma_warp_scope=128, mma_fp16_throughput=2048,
        mma_fp8_throughput=4096, wgmma_supported=True, tmem_supported=False,
        tmem_kb_per_sm=None, fp8_mma_supported=True,
    ),
    "sm_100": ArchSpec(
        sm_version="sm_100", max_warps_per_sm=64, warp_schedulers=4,
        fp32_fmas_per_sm_clock=128, smem_latency_cycles=35, global_l2_latency_cycles=240,
        mma_class="tcgen05", mma_warp_scope=1, mma_fp16_throughput=4096,
        mma_fp8_throughput=8192, wgmma_supported=False, tmem_supported=True,
        tmem_kb_per_sm=256, fp8_mma_supported=True,
    ),
}
```

`DeviceSpec` gains an `sm_version` field (populated by `from_name()` and `from_device()`) used as the key into `_ARCH_TABLE`.

---

### _INSTRUCTION_TAXONOMY (in `_taxonomy.py`)

A dict mapping PTX mnemonic prefixes (longest-match wins) to an `InstructionRecord`:

```python
_INSTRUCTION_TAXONOMY: dict[str, InstructionRecord] = {
    # smem
    "ld.shared":           InstructionRecord(category="smem", arch_introduced="sm_10", ...),
    "st.shared":           InstructionRecord(category="smem", arch_introduced="sm_10", ...),
    "ldmatrix.sync":       InstructionRecord(category="smem", arch_introduced="sm_75", ...),

    # global mem
    "ld.global":           InstructionRecord(category="global_mem", ...),
    "st.global":           InstructionRecord(category="global_mem", ...),

    # async copy
    "cp.async.ca":         InstructionRecord(category="async_copy", arch_introduced="sm_80", ...),
    "cp.async.cg":         InstructionRecord(category="async_copy", arch_introduced="sm_80", ...),
    "cp.async.commit_group": InstructionRecord(category="async_copy", ...),
    "cp.async.wait_group": InstructionRecord(category="async_copy", ...),

    # TMA
    "cp.async.bulk":       InstructionRecord(category="async_tma", arch_introduced="sm_90", ...),
    "cp.reduce.async.bulk": InstructionRecord(category="async_tma", ...),
    "multimem.cp.async":   InstructionRecord(category="async_tma", ...),

    # warp MMA
    "mma.sync":            InstructionRecord(category="mma_warp", arch_introduced="sm_70", ...),
    "mma.sp.sync":         InstructionRecord(category="mma_warp", arch_introduced="sm_80", ...),

    # warpgroup MMA
    "wgmma.mma_async":     InstructionRecord(category="mma_warpgroup", arch_introduced="sm_90", ...),
    "wgmma.fence":         InstructionRecord(category="mma_warpgroup", ...),
    "wgmma.commit_group":  InstructionRecord(category="mma_warpgroup", ...),
    "wgmma.wait_group":    InstructionRecord(category="mma_warpgroup", ...),

    # CTA MMA (Blackwell)
    "tcgen05.mma":         InstructionRecord(category="mma_cta", arch_introduced="sm_100", ...),
    "tcgen05.mma.sp":      InstructionRecord(category="mma_cta", arch_introduced="sm_100", ...),
    "tcgen05.mma.ws":      InstructionRecord(category="mma_cta", arch_introduced="sm_100", ...),

    # tmem (Blackwell)
    "tcgen05.alloc":       InstructionRecord(category="tmem", arch_introduced="sm_100", ...),
    "tcgen05.dealloc":     InstructionRecord(category="tmem", ...),
    "tcgen05.ld":          InstructionRecord(category="tmem", ...),
    "tcgen05.st":          InstructionRecord(category="tmem", ...),
    "tcgen05.cp":          InstructionRecord(category="tmem", ...),
    "tcgen05.shift":       InstructionRecord(category="tmem", ...),
    "tcgen05.fence":       InstructionRecord(category="tmem", ...),
    "tcgen05.commit":      InstructionRecord(category="tmem", ...),
    "tcgen05.wait":        InstructionRecord(category="tmem", ...),
    "tcgen05.relinquish_alloc_permit": InstructionRecord(category="tmem", ...),

    # fused FP (CUDA cores)
    "fma.rn":              InstructionRecord(category="fused_fp", ...),
    "fma.rm":              InstructionRecord(category="fused_fp", ...),
    "fma.rp":              InstructionRecord(category="fused_fp", ...),
    "fma.rz":              InstructionRecord(category="fused_fp", ...),
    "mad.lo":              InstructionRecord(category="fused_fp", ...),
    "mad.hi":              InstructionRecord(category="fused_fp", ...),
    "dp4a":                InstructionRecord(category="fused_fp", ...),
}
```

**Lookup algorithm** — longest prefix match:
```python
def _classify(mnemonic: str) -> InstructionRecord | None:
    for length in range(len(mnemonic), 0, -1):
        prefix = mnemonic[:length]
        if prefix in _INSTRUCTION_TAXONOMY:
            return _INSTRUCTION_TAXONOMY[prefix]
    return None  # → "other"
```

This handles e.g. `tcgen05.mma.sp.sync.aligned` matching `tcgen05.mma.sp` before `tcgen05.mma`.

---

### PTXTracer.trace() Algorithm

```
1. Split ptx_source on newlines
2. For each line:
   a. Strip whitespace, skip blank lines and comment lines (starting with //)
   b. Match r'^\s*([\w.]+)' to extract leading mnemonic token
   c. Look up mnemonic in _INSTRUCTION_TAXONOMY (longest prefix match)
   d. If found: increment by_category[category], increment mma_*_count if MMA category
   e. If not found: increment by_category["other"]
   f. If arch_introduced > target sm_version: add to unsupported_instructions
3. Scan for .shared directives: r'\.shared\s+\.align\s+\d+\s+\.\w+\s+\w+\[(\d+)\]'
   → sum bytes → smem_alloc_bytes
4. Scan for tcgen05.alloc: extract n_cols argument → tmem_alloc_bytes = n_cols × 512
5. Count cp.async.commit_group + cp.async.bulk.commit_group → async_copy_groups
6. Estimate arithmetic_intensity:
   flop_estimate = fused_fp_count × 2 + mma_warp_count × tile_flops
   byte_estimate = (global_ld_count + global_st_count) × 4  (assume 32-bit words as a floor)
   arithmetic_intensity = flop_estimate / byte_estimate if byte_estimate > 0 else None
7. Run bottleneck() → classify based on relative category counts and arch throughputs
8. Return TracerResult
```

---

### Bottleneck Classification

```python
def bottleneck(self, result: TracerResult) -> str:
    arch = _ARCH_TABLE[result.sm_version]

    # Estimated cycles per category, normalized per warp scheduler
    mma_total = result.mma_warp_count + result.mma_warpgroup_count + result.mma_cta_count
    smem_total = result.by_category.get("smem", 0)
    global_total = result.by_category.get("global_mem", 0)
    fp_total = result.by_category.get("fused_fp", 0)

    # Cycle estimates: instruction_count × latency / (schedulers × throughput scaling)
    mma_cycles    = mma_total * _MMA_LATENCY[result.sm_version]
    smem_cycles   = smem_total * arch.smem_latency_cycles
    global_cycles = global_total * arch.global_l2_latency_cycles
    fp_cycles     = fp_total * 4  # FP32 FFMA latency = 4 cycles on all arches

    dominant = max(mma_cycles, smem_cycles, global_cycles, fp_cycles)
    total = mma_cycles + smem_cycles + global_cycles + fp_cycles

    if total == 0:
        return "mixed"
    if dominant / total > 0.6:
        if dominant == mma_cycles:   return "compute"
        if dominant == global_cycles: return "memory"
        if dominant == smem_cycles:  return "latency"
    return "mixed"
```

---

## Integration with DeviceSpec

`DeviceSpec` needs one new field:

```python
@dataclass
class DeviceSpec:
    ...
    sm_version: str    # "sm_80" | "sm_86" | "sm_89" | "sm_90" | "sm_100"
```

`from_name()` populates `sm_version` from the GPU SKU table. `from_device()` reads it from `device.compute_capability` (major, minor) → `f"sm_{major}{minor}"`.

`PTXTracer.__init__` receives a `DeviceSpec` and calls `_ARCH_TABLE[device.sm_version]` to get the `ArchSpec`.

---

## Demo Output Format

```
=== PTX Kernel Execution Tracer ===

Kernel: vector_add.ptx
Target: A100 (sm_80)
─────────────────────────────────────────────────────
Total instructions : 42

Instruction mix:
  global_mem        : 18  (42.9%)   ← ld.global + st.global
  smem              :  0  ( 0.0%)
  fused_fp          : 16  (38.1%)   ← fma.rn.f32
  mma_warp          :  0  ( 0.0%)
  async_copy        :  6  (14.3%)   ← cp.async
  other             :  2  ( 4.8%)

MMA scope breakdown:
  mma.sync (warp, 32t) :  0
  wgmma    (wgrp, 128t):  0
  tcgen05  (CTA, tmem) :  0

Smem allocation   : 4,096 bytes  (from .shared declarations)
TMEM allocation   : n/a          (not sm_100)
Async copy groups : 2

Estimated arith. intensity: 0.089 FLOP/byte
Bottleneck        : memory       (global_mem dominates cycle budget)

─────────────────────────────────────────────────────

Kernel: gemm_mma.ptx
Target: A100 (sm_80)   │   Target: H100 (sm_90)
────────────────────── │ ──────────────────────
mma_warp      : 16     │ mma_warpgroup : 4
smem          : 48     │ smem          : 48
async_copy    : 12     │ async_tma     : 2
Bottleneck: compute    │ Bottleneck: compute
MMA throughput: 1,024 FMA/SM/clk  │  2,048 FMA/SM/clk
MMA scope: 1 warp (32t)           │  1 warpgroup (128t)
Accum in: RF (per warp)           │  RF (per warpgroup)
─────────────────────────────────────────────────────
```

---

## Design Decisions

### 1. Regex scanner, not full grammar parser

PTX is complex enough that a full grammar (Lark, pyparsing) would require significant maintenance as new instructions are added (PTX ISA 8.5 added tcgen05; 8.7 added MX FP4 variants). A regex mnemonic extractor + prefix-trie lookup correctly classifies >95% of instructions in real PTX files (validated by the pyptx project's round-trip tests on 218+ files), with unrecognized instructions falling into `other` without error. This approach is simpler to maintain and extend.

**Tradeoff**: The tracer sees PTX instruction types but not operand values. It cannot determine the MMA tile N dimension (which varies by the `wgmma` operand), only that a wgmma instruction is present.

### 2. Longest-prefix match for taxonomy lookup

PTX mnemonic modifiers are dot-delimited: `tcgen05.mma.sp.sync.aligned`. The taxonomy uses longest-prefix matching (try `tcgen05.mma.sp`, then `tcgen05.mma`, then `tcgen05`) so sparse variants are distinguished from dense variants. This is simpler than a trie and correct for the taxonomy depth required.

### 3. sm_version added to DeviceSpec (minimal ARCH-004 change)

Rather than creating a parallel lookup system, `sm_version: str` is added to `DeviceSpec` as a new field. Existing `from_name()` entries populate it from the GPU SKU table; `from_device()` derives it from `compute_capability`. This is the minimum change to ARCH-004.

### 4. Bottleneck as cycle-budget estimation, not cycle-accurate simulation

The bottleneck estimate compares `instruction_count × latency` per category — a simplified cycle-budget model. This is correct for identifying the dominant stressor but ignores latency hiding (which depends on occupancy) and pipeline overlap. The `OccupancyModel` result from ARCH-004 can be used alongside `TracerResult` to estimate latency-hiding effectiveness — that cross-model analysis is left to the user and documented in the demo.

### 5. PTX fixtures are handwritten, not cuobjdump-extracted

Handwritten PTX fixtures are reproducible, readable, and portable. Extracting PTX from compiled kernels via `cuobjdump` requires a GPU and an NVIDIA toolkit, breaking the "no GPU required" guarantee. Real kernel PTX can be fed to the tracer at runtime — the demo's handwritten fixtures serve as readable reference examples.

---

## Consequences

### Positive
- PTX tracer is GPU-free and fast (regex scan, no parsing)
- Surfaces the Ampere/Ada vs Hopper vs Blackwell MMA scope difference concretely
- Integrates cleanly into existing `src/kernel_model/` alongside `OccupancyModel` and `RooflineModel`
- `_ARCH_TABLE` and `_INSTRUCTION_TAXONOMY` are independently testable and extensible

### Negative / Tradeoffs
- Cannot determine MMA tile dimensions (N for wgmma, shape for tcgen05) without parsing operands
- Arithmetic intensity estimate uses instruction counts × assumed word size — less accurate than counting actual bytes; labelled as "estimated" in output
- PTX ≠ SASS: `ptxas` may reorder, fuse, or eliminate instructions; the tracer models PTX intent, not the final hardware schedule. This is a known, documented limitation.
- `_INSTRUCTION_TAXONOMY` must be updated when new PTX ISA versions introduce new instructions (e.g., FP4 MX variants in ISA 8.7)

### Risks
- Blackwell FP4/MX instruction variants (`tcgen05.mma.ws`, `kind::mxf4nvf4`) require ISA 8.7 coverage; the taxonomy table must include them
- pyptx (the most capable existing PTX parser) is not pip-installable — if regex proves insufficient, a vendored copy of pyptx's parser module may be needed
- Hopper wgmma requires 4 contiguous warps with specific warp-rank alignment; the tracer cannot verify alignment from PTX alone (this is a runtime property)

---

## Testing Strategy

All tests in `tests/test_ptx_tracer.py` are CPU-safe:

```
test_trace_vector_add        — inline PTX fixture; checks global_mem > 0, fused_fp > 0, mma_* == 0
test_trace_mma_sync          — fixture with mma.sync; checks mma_warp_count == 1, sm_80 target
test_trace_wgmma             — fixture with wgmma.mma_async; checks mma_warpgroup_count == 1
test_trace_tcgen05           — fixture with tcgen05.mma + tcgen05.alloc; checks mma_cta_count == 1
test_arch_mismatch_warning   — tcgen05 against sm_80 target; checks unsupported_instructions non-empty
test_smem_declaration_parse  — .shared .align 128 .b8 name[8192]; checks smem_alloc_bytes == 8192
test_tmem_alloc_parse        — tcgen05.alloc with n=64; checks tmem_alloc_bytes == 64 × 512 = 32768
test_async_copy_group_count  — fixture with 3 commit_group; checks async_copy_groups == 3
test_unknown_instruction     — nonsense mnemonic; checks it lands in "other" without exception
test_bottleneck_compute      — high mma_count, low global_count; checks bottleneck == "compute"
test_bottleneck_memory       — high global_count, low mma_count; checks bottleneck == "memory"
test_performance_10k_lines   — trace a 10,000-line PTX string; assert duration < 1 second
test_arch_table_coverage     — all DeviceSpec SKU names resolve to an _ARCH_TABLE entry
```

---

## References

- Research doc: `agents/architecture/ptx-tracer-research.md` (full instruction taxonomy, latency tables, existing tool survey)
- [PTX ISA 8.7 — tcgen05 instructions](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.7.pdf)
- [pyptx — Python PTX DSL with round-trip parser](https://github.com/patrick-toulme/pyptx)
- [arxiv 2512.02189 — Blackwell microbenchmarks (latency table source)](https://arxiv.org/html/2512.02189v1)
- [arxiv 2501.12084 — Hopper microbenchmarks (latency table source)](https://arxiv.org/html/2501.12084v1)
