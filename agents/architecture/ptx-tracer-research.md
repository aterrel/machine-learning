# Research: PTX Kernel Execution Tracer — Architecture and Feasibility

**Related to**: ARCH-004, REQ-0010
**Date**: 2026-05-05
**Purpose**: Feasibility and design research for a PTX-based instruction tracer that models how a CUDA warp scheduler allocates smem, tmem, fused ops, regular ops, and MMA operations across Ampere, Ada, Hopper, and Blackwell GPU architectures.

---

## Overview

A PTX tracer takes a PTX assembly file or string, classifies every instruction into a hardware-relevant category (smem, global mem, MMA, async copy, fused FP op, etc.), and produces a per-category breakdown annotated with architecture-specific throughput and latency. The result answers: *given this kernel on this GPU, what is the mix of hardware work, what is the expected bottleneck, and how does the scheduler interleave these instruction types?*

This is distinct from a cycle-accurate simulator. The goal is an analytical execution model — fast, offline, GPU-free — that gives useful qualitative and semi-quantitative insight.

---

## 1. PTX Instruction Taxonomy

### 1.1 Shared Memory Operations

All architectures (sm_80 through sm_100):

```
ld.shared.{b8,b16,b32,b64,b128}          scalar load from smem
st.shared.{b8,b16,b32,b64,b128}          scalar store to smem
ld.shared.v2.{b32,b64}, .v4.b32          vectorized smem loads
ldmatrix.sync.aligned.m8n8.{x1,x2,x4}   warp-collective matrix load from smem
  {.trans}.shared.b16                    → feeds directly into mma.sync fragment A/B
```

`ldmatrix` is the primary high-throughput path for loading MMA A/B operands. Introduced in sm_75 but used heavily on Ampere through Hopper before being superseded by TMA on Hopper.

### 1.2 Global Memory Operations

```
ld.global.{.ca,.cg,.cs,.cv}.{b32,b64}    load from global (various cache policies)
st.global.{.wb,.cg,.cs,.wt}.{b32,b64}    store to global
ld.global.v2.b32, .v4.b32                vectorized global loads
```

### 1.3 Async Copy — Ampere+ (cp.async)

```
cp.async.ca.shared.global [smem], [global], size    cache-all policy
cp.async.cg.shared.global [smem], [global], size    cache-global policy
cp.async.commit_group                               commit pending copies to a group
cp.async.wait_group N                               wait until ≤ N groups are in flight
cp.async.wait_all                                   wait for all pending copies
```

`size` must be a literal 4, 8, or 16. These instructions are the software pipelining primitive on Ampere/Ada before TMA existed.

### 1.4 Async Copy — Hopper+ (TMA / cp.async.bulk)

```
cp.async.bulk.tensor.{1..5}d              TMA load: global → smem
  .shared::cluster.global.tile
  .mbarrier::complete_tx::bytes
  [smem-addr], [gmem-desc], [coords], [mbar], size

cp.async.bulk.tensor.{...}.global         TMA store: smem → global
  .shared::cta.tile.bulk_group
  [gmem-desc], [coords], [smem-addr]

cp.async.bulk.prefetch.tensor.{...}       TMA prefetch into L2
cp.reduce.async.bulk.tensor.{...}         TMA with reduction (e.g., atomicAdd in global)
multimem.cp.async.bulk                    multicast TMA across cluster SMs

cp.async.bulk.commit_group                commit a TMA group
cp.async.bulk.wait_group N                wait for TMA completion
```

TMA is Hopper's primary data movement primitive for GEMM pipelines, replacing cp.async in high-performance kernels.

### 1.5 Warp-Level MMA — mma.sync (Ampere sm_80/86, Ada sm_89)

Mnemonic pattern:
```
mma.sync.aligned.mMnNkK.row.col.dtype.atype.btype.ctype {d}, {a}, {b}, {c}
```

| Type combo | Tile (M×N×K) | SM |
|---|---|---|
| f16/bf16 → f32 or f16 | 16×8×16 | sm_80+ |
| tf32 → f32 | 16×8×8 | sm_80+ |
| s8/u8 → s32 | 16×8×32 | sm_80+ |
| e4m3/e5m2 (FP8) → f32 | 16×8×32 | sm_89 (Ada) only |
| s4/u4 → s32 | 16×8×64 | sm_80+ |

Sparse variants: `mma.sp.sync.aligned` doubles the K dimension with 50% structured sparsity.  
Scope: **1 warp = 32 threads**. All 32 threads collectively hold the A, B, and C/D register fragments.

Accumulator register distribution (per thread, for a 16×8 tile):
- f16 → f32 accumulator: 4 × f32 regs per thread
- s8/u8 → s32: 4 × s32 regs per thread
- tf32 → f32: 4 × f32 regs per thread
- fp8 → f32 (Ada): 4 × f32 regs per thread

### 1.6 Warpgroup MMA — wgmma.mma_async (Hopper sm_90)

Mnemonic pattern:
```
wgmma.mma_async.sync.aligned.m64nNkK.acctype.atype.btype
  [desc-A-or-regs], [smem-desc-B], {acc-regs}, scale-A, scale-B
```

Scope: **1 warpgroup = 4 contiguous warps = 128 threads**. The warpgroup's first warp must have warp-rank divisible by 4.

| Type combo | M | N range | K | K (bytes) |
|---|---|---|---|---|
| f16/bf16 → f32 or f16 | 64 | 8–256 (step 8) | 16 | 32 |
| tf32 → f32 | 64 | 8–256 (step 8) | 8 | 32 |
| e4m3/e5m2 (FP8) → f32 | 64 | 8–256 (step 8) | 32 | 32 |
| i8 → i32 | 64 | 8–256 (step 8) | 32 | 32 |
| i4 → i32 | 64 | 8–256 (step 8) | 64 | 32 |

Note: M is fixed at 64 because the 128-thread warpgroup contributes 64 output rows (each thread covers half a row cooperatively).

Accumulator register distribution — for m64n64k16 f16→f32: 8 × f32 per thread (1024 total across 128 threads). For m64n256k16: 32 × f32 per thread.

Required bracketing instructions:
```
wgmma.fence.sync.aligned         separates wgmma groups
wgmma.commit_group.sync.aligned  commit an async group
wgmma.wait_group.sync.aligned N  wait until ≤ N groups pending
```

### 1.7 Blackwell MMA — tcgen05.mma (sm_100)

Mnemonic pattern:
```
tcgen05.mma.cta_group::1.kind::{tf32,f16,i8,f8f6f4,mxf8f6f4,...}
  [tmem-accum-addr], [smem-desc-A], [smem-desc-B], {scale-regs}, enable-input-d
```

Scope: **issued by a single thread** on behalf of the entire CTA. No warp or warpgroup synchronization at issue time — the hardware handles thread sequencing.

Accumulator location: **TMEM** (not RF). This is the fundamental departure from all previous generations.

1SM tile shapes: 64×64, 64×128, 64×192, 64×256, 128×64, 128×128, 128×192, 128×256  
2SM variant (`tcgen05.mma.2sm`): 128×64 through 256×256

Sparse variant: `tcgen05.mma.sp` (50% structured sparsity, 2× K)  
With-scale variant: `tcgen05.mma.ws` for block-scaled FP4/FP8 formats

Throughput relative to Hopper: ~2× for legacy types (f16/bf16/tf32/i8), ~4× for FP4/MX formats.

### 1.8 Tensor Memory Operations — tcgen05 (Blackwell sm_100 only)

**Hopper has no tmem.** Hopper wgmma accumulates in RF registers.

```
tcgen05.alloc.cta_group::1.sync.aligned b32-reg, n-cols
  # allocate n columns of TMEM (power-of-2, min 32); returns base address

tcgen05.dealloc.cta_group::1.sync.aligned b32-reg, n-cols
  # release allocation; must call before kernel exit

tcgen05.ld.sync.aligned.{shape}.b32 {regs}, [tmem-addr]
  # synchronous per-warp load from TMEM → registers (used in epilogue)

tcgen05.st.sync.aligned.{shape}.b32 [tmem-addr], {regs}
  # store from registers → TMEM

tcgen05.cp.sync.aligned.{...}
  # smem → TMEM copy (pre-loading B or bias into TMEM)

tcgen05.shift.sync.aligned
  # shift TMEM data (used for warp-rotation streaming patterns)

tcgen05.fence.before_thread_sync
tcgen05.fence.after_thread_sync
  # ordering fences between TMEM and thread synchronization

tcgen05.relinquish_alloc_permit
  # hand off allocation rights (used in dual-CTA patterns)
```

### 1.9 Fused Arithmetic

```
fma.rn.f32 d, a, b, c          FP32 FMA — the canonical FFMA (CUDA cores)
fma.rn.f16 d, a, b, c          FP16 FMA (CUDA cores, not Tensor Cores)
fma.rn.f16x2 d, a, b, c        FP16×2 packed FMA
fma.rn.bf16x2 d, a, b, c       BF16×2 FMA (Ampere+)
fma.rn.f64 d, a, b, c          FP64 FMA
fma.rn.f32x2 d, a, b, c        FP32 SIMD pair (Ada/Hopper)
mad.lo.s32, mad.hi.s32          integer multiply-add
dp4a.s32.s32, dp4a.u32.u32      dot product of 4 INT8 elements (CUDA core shortcut)
```

---

## 2. Architecture-Specific Throughputs

### 2.1 CUDA Core FP32 Throughput (FFMA ops/SM/clock)

| Architecture | SM | FP32 CUDA Cores/SM | FP32 FMAs/SM/clock |
|---|---|---|---|
| Ampere A100 | sm_80 | 64 | 64 |
| Ampere GA10x | sm_86 | 128 | 128 |
| Ada Lovelace | sm_89 | 128 | 128 |
| Hopper H100 | sm_90 | 128 | 128 |
| Blackwell B100/B200 | sm_100 | 128 | ~128 (estimated) |

### 2.2 Tensor Core Throughput (FMAs/SM/clock via MMA instructions)

These are the MMA instruction throughputs relevant to the tracer's bottleneck classification:

| Precision | Ampere sm_80 (A100) | Ada sm_89 | Hopper sm_90 (H100) | Blackwell sm_100 |
|---|---|---|---|---|
| FP16 | 1,024 | ~1,024 | 2,048 | ~4,096 |
| BF16 | 1,024 | ~1,024 | 2,048 | ~4,096 |
| TF32 | 512 | ~512 | 1,024 | ~2,048 |
| FP8 | not supported | ~2,048 | 4,096 | ~8,192 |
| FP4 / MX | not supported | not supported | not supported | ~16,384 |
| INT8 | 2,048 | ~2,048 | 4,096 | ~8,192 |

**Derivation**: Ampere A100 = 4 TCs/SM × 256 FP16 FMAs/TC/clock = 1,024. Hopper doubles per-SM TC throughput. Blackwell values estimated from CUTLASS-documented ~2× gain for legacy types, ~4× for narrow types. Official per-clock-per-SM table is not published for sm_100; these are architectural ratios.

### 2.3 Warp Scheduler Configuration

All four architectures share the same top-level scheduler structure:

| Architecture | SM | Warp Schedulers/SM | Max Resident Warps/SM |
|---|---|---|---|
| Ampere A100 | sm_80 | 4 | 64 |
| Ampere GA10x | sm_86 | 4 | 48 |
| Ada Lovelace | sm_89 | 4 | 48 |
| Hopper H100 | sm_90 | 4 | 64 |
| Blackwell B200 | sm_100 | 4 | 64 |

Each scheduler issues 1 instruction per cycle. 4 schedulers = up to 4 instructions issued across the SM per clock. Warp schedulers interleave warps to hide latency; the tracer's per-category latency table enables latency-hiding analysis.

---

## 3. Instruction Latencies by Architecture

Sources: microbenchmark papers (Hopper: arxiv 2501.12084; Blackwell: arxiv 2507.10789, 2512.02189).

| Operation | Ampere sm_80 | Ada sm_89 | Hopper sm_90 | Blackwell sm_100 |
|---|---|---|---|---|
| `ld.shared` | ~22–24 cycles | ~22 | ~29–31 | ~30–40 |
| `ld.global` (L1 hit) | ~33 | ~33 | ~33 | ~22–40 |
| `ld.global` (L2 hit) | ~188 | ~188 | ~273 | ~128–358 |
| `ld.global` (DRAM) | ~600+ | ~600+ | ~657 | ~877 (GB203) / ~4,200 (B200) |
| `fma.rn.f32` | 4 | 4 | 4 | 4 |
| `mma.sync` (f16, m16n8k16) | ~16–17 | ~16–17 | — | — |
| `wgmma.mma_async` (N=64) | — | — | ~64–128 | — |
| `tcgen05.mma` (m64n64) | — | — | — | ~11.0 |
| `tcgen05.mma` (m128n128) | — | — | — | ~11.3 |
| `tcgen05.ld` (TMEM load) | — | — | — | ~10–20 |

The DRAM latency gap for B200 (~4,200 cycles) reflects HBM3e on a large B200 SXM die with a more complex memory subsystem.

---

## 4. Warp/Warpgroup Scope Summary

| Instruction class | Architecture | Scope | Threads | Accumulator location |
|---|---|---|---|---|
| `mma.sync` | Ampere, Ada | 1 warp | 32 | RF (distributed across 32 threads) |
| `wgmma.mma_async` | Hopper | 1 warpgroup | 128 (4 warps) | RF (distributed across 128 threads) |
| `tcgen05.mma` | Blackwell | 1 thread (on behalf of CTA) | 1 (issue) / 128 (execution) | TMEM |
| `tcgen05.ld/st` | Blackwell | 1 warp | 32 | RF ← TMEM |

**Key implication for the tracer**: to model "how many mma instructions can be in flight simultaneously," you must know:
- Ampere/Ada: count warps resident on SM; mma.sync is 1 warp → N warps = N independent mma.sync pipelines
- Hopper: count warpgroups (groups of 4 warps); wgmma is 1 warpgroup → N/4 independent wgmma pipelines
- Blackwell: tcgen05.mma is issued per CTA; the TMEM accumulator address space determines concurrency

---

## 5. Tensor Memory (tmem) Architecture — Blackwell sm_100

### Physical Layout
- **Total capacity**: 256 KB per SM
- **Organization**: 512 columns × 128 rows (lanes) of 32-bit cells
- **Sub-partition mapping**: 4 sub-partitions, each 64 KB (32 lanes × 2 KB per lane)
- **Allocation unit**: power-of-2 columns, minimum 32 columns

### Address Format
```
bits [31:16] = lane ID (0–127)
bits [15:0]  = column index (0–511)
```

### Per-Warp Access Restriction
TMEM is per-SM but access is warp-partitioned:
- Warp 0 → lanes 0–31
- Warp 1 → lanes 32–63
- Warp 2 → lanes 64–95
- Warp 3 → lanes 96–127

An epilogue over a 128-row TMEM tile requires all 4 warps of the warpgroup to cooperate, each reading its 32-lane slice via `tcgen05.ld`.

### Hopper has no tmem
On Hopper, wgmma accumulators live in the RF of the 128-thread warpgroup. If a kernel needs to save/restore accumulators (e.g., during a smem double-buffer swap), it must manually spill to smem. Blackwell eliminates this by keeping accumulators in TMEM throughout the mainloop.

---

## 6. Existing PTX Parsing Tools

### pyptx (github.com/patrick-toulme/pyptx)
The most relevant tool. A Python DSL for writing PTX targeting Hopper and Blackwell that includes a real PTX parser and emitter achieving byte-identical round-trips on 218+ real PTX files from CUTLASS, Triton, ThunderKittens, DeepGEMM, and LLVM test suites. Full instruction coverage through tcgen05/TMEM (Blackwell). CLI: `python -m pyptx.codegen kernel.ptx --sugar --name my_kernel`.

**Recommendation**: use as a parsing dependency or reference implementation. It has already solved the hard part of PTX parsing for modern architectures.

### ptx-parser (PyPI: ptx-parser)
A Rust-backed Python extension for parsing PTX. Likely fast for bulk parsing. API not fully documented publicly.

### GPGPU-Sim
C++ PTX parser (`src/cuda-sim/ptx_parser.cc`) with a full warp scheduling simulation model. Too heavy as a dependency but a valuable reference for scheduler modeling logic.

### CuAssembler (github.com/cloudcores/CuAssembler)
Pure-Python SASS (not PTX) assembler/analyzer. Operates below PTX; not directly applicable, but shows patterns for instruction mnemonic parsing into modifier key-value pairs.

### Regex-based approach (pragmatic)
PTX instructions have a highly regular line structure:
```
opcode.modifier1.modifier2.type operands ;
```
A regex like `r'^\s+([\w.]+)\s' ` plus a prefix-trie or dict lookup against the taxonomy in section 1 correctly categorizes >95% of instructions without a full grammar. This is the recommended approach for a tracer that needs category counts, not a full parse tree.

### LLVM Python bindings
LLVM (llvmlite, MLIR Python) does not expose PTX-level analysis. LLVM emits PTX as a text string; PTX is not an LLVM IR. Analysis must happen on the PTX text directly.

---

## 7. Design Sketch for the Tracer

Given the research above, a PTX tracer in `src/kernel_model/` would have this structure:

```python
# src/kernel_model/ptx_tracer.py

@dataclass
class InstructionRecord:
    mnemonic: str              # e.g. "wgmma.mma_async.sync.aligned"
    category: str              # smem | tmem | mma_warp | mma_warpgroup | mma_cta |
                               # async_copy | async_tma | fused_fp | global_mem | other
    arch_introduced: str       # "sm_80" | "sm_89" | "sm_90" | "sm_100"
    latency_cycles: dict       # {"sm_80": 17, "sm_90": 64, ...}  — None if not applicable
    throughput_per_sm: dict    # {"sm_80": 1024, ...}  — FMAs/SM/clock for MMA; None for others

@dataclass
class TracerResult:
    arch: str                  # target architecture
    total_instructions: int
    by_category: dict[str, int]          # category → count
    mma_warp_count: int                  # mma.sync instructions (Ampere/Ada scope: 1 warp each)
    mma_warpgroup_count: int             # wgmma instructions (Hopper scope: 1 warpgroup each)
    mma_cta_count: int                   # tcgen05.mma instructions (Blackwell)
    smem_alloc_bytes: int | None         # if parseable from .shared .align directives
    tmem_alloc_cols: int | None          # from tcgen05.alloc; None if not Blackwell
    async_copy_groups: int               # number of cp.async.commit_group or bulk equivalents
    bottleneck: str                      # "compute" | "memory" | "latency" | "mixed"
    notes: list[str]                     # human-readable observations

class PTXTracer:
    def __init__(self, device: DeviceSpec): ...

    def trace(self, ptx_source: str) -> TracerResult:
        # 1. scan lines, extract instruction mnemonics (regex or pyptx)
        # 2. classify each into InstructionRecord category
        # 3. accumulate counts and annotate with arch-specific latency/throughput
        # 4. compute bottleneck estimate: compare mma throughput vs smem/global load throughput
        # 5. return TracerResult
        ...
```

### Bottleneck Classification Logic

For a given function body the tracer can estimate:
- **MMA bound**: `mma_count × tile_flops / mma_throughput_per_sm`
- **Smem bound**: `smem_load_count × latency_smem / warps_resident` (latency-hiding estimate)
- **Global memory bound**: correlate with roofline result from `RooflineModel`

The category with the highest estimated cycles is the bottleneck.

### Architecture Dispatch Table

Rather than `if arch == "sm_80"` chains, use a `_ARCH_TABLE` dict keyed by SM version:

```python
_ARCH_TABLE = {
    "sm_80": ArchSpec(max_warps=64, schedulers=4, smem_latency=23, fp32_fmas=64,
                      mma_class="mma_sync", wgmma_supported=False, tmem_supported=False, ...),
    "sm_86": ArchSpec(..., fp32_fmas=128, ...),
    "sm_89": ArchSpec(..., fp8_mma=True, ...),
    "sm_90": ArchSpec(..., mma_class="wgmma", wgmma_supported=True, tmem_supported=False, ...),
    "sm_100": ArchSpec(..., mma_class="tcgen05", tmem_supported=True, tmem_kb=256, ...),
}
```

This table is the core of the instruction dispatch and throughput annotation.

---

## 8. Complexity Assessment

| Component | Effort | Notes |
|---|---|---|
| PTX line scanner + regex classifier | Low | 1–2 days; regex on mnemonic prefix is sufficient |
| Instruction taxonomy dict (all categories, all arches) | Medium | 2–3 days; data from this doc |
| Architecture dispatch table (_ARCH_TABLE) | Low | 1 day; data from this doc |
| TracerResult + bottleneck estimator | Medium | 2–3 days; needs latency/throughput table |
| tmem column allocation parser | Medium | 2 days; parse tcgen05.alloc arguments |
| smem .shared declaration parser | Low | 1 day; parse `.shared .align N .bNN` directives |
| async copy group counting | Low | 1 day; count commit_group instructions |
| Integration with DeviceSpec + RooflineModel | Low | 1 day; feed TracerResult into RooflineModel |
| Tests (CPU-safe, no GPU) | Medium | 2 days; use PTX snippets as fixtures |
| **Total** | **~Medium-to-Complex** | **12–16 days; best as a separate sprint** |

---

## 9. Key Risks and Constraints

1. **PTX dialects evolve fast**: tcgen05 instructions were introduced with sm_100 in PTX ISA 8.5 (Nov 2024). FP4 / MXF4 variants in 8.7 (Feb 2025). The taxonomy table must be versioned.

2. **Accumulator register count is not statically determinable from PTX alone**: `wgmma` writes to `{acc}` register lists whose size depends on N. The tracer needs to match the N dimension of the wgmma instruction to compute register pressure.

3. **smem allocation may be dynamic**: `.shared .align` declarations give static smem size, but kernels using dynamic shared memory (`extern __shared__`) have runtime-determined sizes — the tracer would need that as an input parameter.

4. **Blackwell 2SM patterns**: `tcgen05.mma.2sm` requires NVLink cluster coordination. Modeling this correctly requires knowing the cluster shape — a tracer input parameter, not derivable from PTX alone.

5. **Instruction reordering by the assembler**: The PTX-to-SASS translation (done by `ptxas`) may reorder, fuse, or split instructions. The tracer models PTX-level intent, not final hardware scheduling. This is a known limitation to document clearly.

---

## 10. References

- [PTX ISA 9.2 Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [PTX ISA 8.7 PDF — Feb 2025](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.7.pdf)
- [PTX ISA 8.5 PDF — Nov 2024](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.5.pdf)
- [Colfax Research — CUTLASS Tutorial: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
- [Colfax Research — CUTLASS Tutorial: GEMM with Tensor Memory for Blackwell](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [NVIDIA CUTLASS Docs — Blackwell SM100 GEMM Functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [tcgen05 for Dummies — gau-nernst](https://gau-nernst.github.io/tcgen05/)
- [SemiAnalysis — Dissecting Nvidia Blackwell: Tensor Cores, PTX Instructions, SASS](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor)
- [SemiAnalysis — NVIDIA Tensor Core Evolution: Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [arxiv 2507.10789 — Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/html/2507.10789v2)
- [arxiv 2512.02189 — Microbenchmarking NVIDIA Blackwell Architecture](https://arxiv.org/html/2512.02189v1)
- [arxiv 2501.12084 — Dissecting the NVIDIA Hopper Architecture](https://arxiv.org/html/2501.12084v1)
- [arxiv 2402.13499 — Benchmarking and Dissecting the NVIDIA Hopper GPU Architecture](https://arxiv.org/pdf/2402.13499)
- [NVIDIA Developer Forums — SM100 TMEM Per-Warp Access Restriction](https://forums.developer.nvidia.com/t/sm100-tmem-rationale-for-per-warp-access-restriction-tcgen05-ld-st/361833)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [pyptx — Python PTX DSL for Hopper and Blackwell](https://github.com/patrick-toulme/pyptx)
- [ptx-parser on PyPI](https://pypi.org/project/ptx-parser/)
- [GPGPU-Sim Distribution](https://github.com/gpgpu-sim/gpgpu-sim_distribution)
- [ProjectPhysX/PTXprofiler](https://github.com/ProjectPhysX/PTXprofiler)
- [The Longest Nvidia PTX Instruction — Ash Vardanian](https://ashvardanian.com/posts/longest-ptx-instruction/)
