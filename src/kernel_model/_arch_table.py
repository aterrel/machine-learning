from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ArchSpec:
    sm_version: str
    max_warps_per_sm: int
    warp_schedulers: int
    fp32_fmas_per_sm_clock: int
    smem_latency_cycles: int
    global_l2_latency_cycles: int
    mma_class: str              # "mma_sync" | "wgmma" | "tcgen05"
    mma_warp_scope: int         # threads per MMA issue: 32 | 128 | 1
    mma_fp16_throughput: int    # FP16 FMAs/SM/clock via Tensor Cores
    mma_fp8_throughput: int | None
    wgmma_supported: bool
    tmem_supported: bool
    tmem_kb_per_sm: int | None
    fp8_mma_supported: bool


_ARCH_TABLE: dict[str, ArchSpec] = {
    "sm_70": ArchSpec("sm_70", max_warps_per_sm=64, warp_schedulers=4,
                      fp32_fmas_per_sm_clock=64, smem_latency_cycles=28,
                      global_l2_latency_cycles=193, mma_class="mma_sync",
                      mma_warp_scope=32, mma_fp16_throughput=512,
                      mma_fp8_throughput=None, wgmma_supported=False,
                      tmem_supported=False, tmem_kb_per_sm=None, fp8_mma_supported=False),
    "sm_80": ArchSpec("sm_80", max_warps_per_sm=64, warp_schedulers=4,
                      fp32_fmas_per_sm_clock=64, smem_latency_cycles=23,
                      global_l2_latency_cycles=188, mma_class="mma_sync",
                      mma_warp_scope=32, mma_fp16_throughput=1024,
                      mma_fp8_throughput=None, wgmma_supported=False,
                      tmem_supported=False, tmem_kb_per_sm=None, fp8_mma_supported=False),
    "sm_86": ArchSpec("sm_86", max_warps_per_sm=48, warp_schedulers=4,
                      fp32_fmas_per_sm_clock=128, smem_latency_cycles=22,
                      global_l2_latency_cycles=188, mma_class="mma_sync",
                      mma_warp_scope=32, mma_fp16_throughput=1024,
                      mma_fp8_throughput=None, wgmma_supported=False,
                      tmem_supported=False, tmem_kb_per_sm=None, fp8_mma_supported=False),
    "sm_89": ArchSpec("sm_89", max_warps_per_sm=48, warp_schedulers=4,
                      fp32_fmas_per_sm_clock=128, smem_latency_cycles=22,
                      global_l2_latency_cycles=188, mma_class="mma_sync",
                      mma_warp_scope=32, mma_fp16_throughput=1024,
                      mma_fp8_throughput=2048, wgmma_supported=False,
                      tmem_supported=False, tmem_kb_per_sm=None, fp8_mma_supported=True),
    "sm_90": ArchSpec("sm_90", max_warps_per_sm=64, warp_schedulers=4,
                      fp32_fmas_per_sm_clock=128, smem_latency_cycles=30,
                      global_l2_latency_cycles=273, mma_class="wgmma",
                      mma_warp_scope=128, mma_fp16_throughput=2048,
                      mma_fp8_throughput=4096, wgmma_supported=True,
                      tmem_supported=False, tmem_kb_per_sm=None, fp8_mma_supported=True),
    "sm_100": ArchSpec("sm_100", max_warps_per_sm=64, warp_schedulers=4,
                       fp32_fmas_per_sm_clock=128, smem_latency_cycles=35,
                       global_l2_latency_cycles=240, mma_class="tcgen05",
                       mma_warp_scope=1, mma_fp16_throughput=4096,
                       mma_fp8_throughput=8192, wgmma_supported=False,
                       tmem_supported=True, tmem_kb_per_sm=256, fp8_mma_supported=True),
}

_MMA_LATENCY: dict[str, int] = {
    "sm_70": 17,
    "sm_80": 17,
    "sm_86": 17,
    "sm_89": 17,
    "sm_90": 64,
    "sm_100": 11,
}
