from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class InstructionRecord:
    mnemonic: str
    category: str           # one of 10 categories
    arch_introduced: str    # "sm_10" | "sm_70" | "sm_75" | "sm_80" | "sm_89" | "sm_90" | "sm_100"
    latency_cycles: dict[str, int | None] = field(default_factory=dict)  # sm_version → cycles
    throughput_note: str = ""


_INSTRUCTION_TAXONOMY: dict[str, InstructionRecord] = {
    # smem
    "ld.shared":              InstructionRecord("ld.shared",    "smem",       "sm_10",  {"sm_80":23,"sm_86":22,"sm_89":22,"sm_90":30,"sm_100":35}),
    "st.shared":              InstructionRecord("st.shared",    "smem",       "sm_10",  {"sm_80":23,"sm_86":22,"sm_89":22,"sm_90":30,"sm_100":35}),
    "ldmatrix.sync":          InstructionRecord("ldmatrix.sync","smem",       "sm_75",  {"sm_80":23,"sm_86":22,"sm_89":22,"sm_90":30}),

    # global mem
    "ld.global":              InstructionRecord("ld.global",    "global_mem", "sm_10",  {"sm_80":188,"sm_86":188,"sm_89":188,"sm_90":273,"sm_100":240}),
    "st.global":              InstructionRecord("st.global",    "global_mem", "sm_10",  {"sm_80":188,"sm_86":188,"sm_89":188,"sm_90":273,"sm_100":240}),

    # async copy (Ampere+)
    "cp.async.ca":            InstructionRecord("cp.async.ca",  "async_copy", "sm_80"),
    "cp.async.cg":            InstructionRecord("cp.async.cg",  "async_copy", "sm_80"),
    "cp.async.commit_group":  InstructionRecord("cp.async.commit_group","async_copy","sm_80"),
    "cp.async.wait_group":    InstructionRecord("cp.async.wait_group",  "async_copy","sm_80"),
    "cp.async.wait_all":      InstructionRecord("cp.async.wait_all",    "async_copy","sm_80"),

    # async TMA (Hopper+)
    "cp.async.bulk.commit_group": InstructionRecord("cp.async.bulk.commit_group","async_tma","sm_90"),
    "cp.async.bulk.wait_group":   InstructionRecord("cp.async.bulk.wait_group",  "async_tma","sm_90"),
    "cp.async.bulk":          InstructionRecord("cp.async.bulk","async_tma","sm_90"),
    "cp.reduce.async.bulk":   InstructionRecord("cp.reduce.async.bulk","async_tma","sm_90"),
    "multimem.cp.async":      InstructionRecord("multimem.cp.async","async_tma","sm_90"),

    # warp MMA (Ampere/Ada)
    "mma.sp.sync":            InstructionRecord("mma.sp.sync",  "mma_warp",   "sm_80",  {"sm_80":17,"sm_86":17,"sm_89":17}),
    "mma.sync":               InstructionRecord("mma.sync",     "mma_warp",   "sm_70",  {"sm_80":17,"sm_86":17,"sm_89":17}),

    # warpgroup MMA (Hopper)
    "wgmma.mma_async":        InstructionRecord("wgmma.mma_async",    "mma_warpgroup","sm_90",{"sm_90":64}),
    "wgmma.fence":            InstructionRecord("wgmma.fence",         "mma_warpgroup","sm_90"),
    "wgmma.commit_group":     InstructionRecord("wgmma.commit_group",  "mma_warpgroup","sm_90"),
    "wgmma.wait_group":       InstructionRecord("wgmma.wait_group",    "mma_warpgroup","sm_90"),

    # CTA MMA (Blackwell)
    "tcgen05.mma.sp":         InstructionRecord("tcgen05.mma.sp", "mma_cta","sm_100",{"sm_100":11}),
    "tcgen05.mma.ws":         InstructionRecord("tcgen05.mma.ws", "mma_cta","sm_100",{"sm_100":11}),
    "tcgen05.mma":            InstructionRecord("tcgen05.mma",    "mma_cta","sm_100",{"sm_100":11}),

    # tmem (Blackwell)
    "tcgen05.alloc":                   InstructionRecord("tcgen05.alloc",                  "tmem","sm_100",{"sm_100":10}),
    "tcgen05.dealloc":                 InstructionRecord("tcgen05.dealloc",                "tmem","sm_100"),
    "tcgen05.relinquish_alloc_permit": InstructionRecord("tcgen05.relinquish_alloc_permit","tmem","sm_100"),
    "tcgen05.ld":                      InstructionRecord("tcgen05.ld",   "tmem","sm_100",{"sm_100":15}),
    "tcgen05.st":                      InstructionRecord("tcgen05.st",   "tmem","sm_100"),
    "tcgen05.cp":                      InstructionRecord("tcgen05.cp",   "tmem","sm_100"),
    "tcgen05.shift":                   InstructionRecord("tcgen05.shift","tmem","sm_100"),
    "tcgen05.fence":                   InstructionRecord("tcgen05.fence","tmem","sm_100"),
    "tcgen05.commit":                  InstructionRecord("tcgen05.commit","tmem","sm_100"),
    "tcgen05.wait":                    InstructionRecord("tcgen05.wait", "tmem","sm_100"),

    # fused FP (CUDA cores)
    "fma.rn":   InstructionRecord("fma.rn","fused_fp","sm_10",{"sm_80":4,"sm_86":4,"sm_89":4,"sm_90":4,"sm_100":4}),
    "fma.rm":   InstructionRecord("fma.rm","fused_fp","sm_10",{"sm_80":4,"sm_86":4,"sm_89":4,"sm_90":4,"sm_100":4}),
    "fma.rp":   InstructionRecord("fma.rp","fused_fp","sm_10",{"sm_80":4,"sm_86":4,"sm_89":4,"sm_90":4,"sm_100":4}),
    "fma.rz":   InstructionRecord("fma.rz","fused_fp","sm_10",{"sm_80":4,"sm_86":4,"sm_89":4,"sm_90":4,"sm_100":4}),
    "mad.lo":   InstructionRecord("mad.lo","fused_fp","sm_10"),
    "mad.hi":   InstructionRecord("mad.hi","fused_fp","sm_10"),
    "dp4a":     InstructionRecord("dp4a",  "fused_fp","sm_61"),
}


def classify(mnemonic: str) -> InstructionRecord | None:
    """Longest-prefix match lookup in _INSTRUCTION_TAXONOMY."""
    for length in range(len(mnemonic), 0, -1):
        prefix = mnemonic[:length]
        if prefix in _INSTRUCTION_TAXONOMY:
            return _INSTRUCTION_TAXONOMY[prefix]
    return None
