from __future__ import annotations
import re
from dataclasses import dataclass, field
from ._taxonomy import InstructionRecord, _INSTRUCTION_TAXONOMY, classify
from ._arch_table import ArchSpec, _ARCH_TABLE, _MMA_LATENCY
from .device_spec import DeviceSpec


@dataclass
class TracerResult:
    arch: str
    sm_version: str
    total_instructions: int
    by_category: dict[str, int]
    mma_warp_count: int
    mma_warpgroup_count: int
    mma_cta_count: int
    smem_alloc_bytes: int | None
    tmem_alloc_bytes: int | None
    async_copy_groups: int
    bottleneck: str
    arithmetic_intensity: float | None
    unsupported_instructions: list[str]
    notes: list[str]


def _sm_to_int(sm: str) -> int:
    """Convert "sm_80" → 80, "sm_100" → 100."""
    return int(sm[3:])


class PTXTracer:
    def __init__(self, device: DeviceSpec) -> None:
        self._device = device
        self._arch = _ARCH_TABLE[device.sm_version]

    def trace(self, ptx_source: str) -> TracerResult:
        by_category: dict[str, int] = {}
        mma_warp_count = 0
        mma_warpgroup_count = 0
        mma_cta_count = 0
        total_instructions = 0
        unsupported_instructions: list[str] = []
        notes: list[str] = []

        target_sm_int = _sm_to_int(self._device.sm_version)

        for line in ptx_source.splitlines():
            stripped = line.strip()

            # Skip blank lines and comment lines
            if not stripped or stripped.startswith("//") or stripped.startswith("#"):
                continue

            # PTX instructions are indented; labels/directives at column 0 start with
            # non-whitespace. Match only indented lines.
            m = re.match(r'^\s+([\w.]+)', line)
            if not m:
                continue

            mnemonic = m.group(1)

            # Skip lines where the leading token starts with . @ $ %
            # (these are directives, predicates, labels embedded in indented lines)
            if mnemonic.startswith(('.', '@', '$', '%')):
                continue

            record = classify(mnemonic)

            if record is not None:
                total_instructions += 1
                cat = record.category
                by_category[cat] = by_category.get(cat, 0) + 1

                if cat == "mma_warp":
                    mma_warp_count += 1
                elif cat == "mma_warpgroup":
                    mma_warpgroup_count += 1
                elif cat == "mma_cta":
                    mma_cta_count += 1

                # Architecture mismatch check
                intro_int = _sm_to_int(record.arch_introduced)
                if intro_int > target_sm_int:
                    unsupported_instructions.append(mnemonic)
                    notes.append(
                        f"Instruction '{mnemonic}' requires {record.arch_introduced} "
                        f"but target is {self._device.sm_version}."
                    )
            else:
                total_instructions += 1
                by_category["other"] = by_category.get("other", 0) + 1

        # Parse .shared declarations for smem_alloc_bytes
        shared_matches = re.findall(r'\.shared\s+[^[]+\[(\d+)\]', ptx_source)
        smem_alloc_bytes: int | None = sum(int(x) for x in shared_matches) if shared_matches else None

        # Parse tcgen05.alloc for tmem_alloc_bytes
        tmem_alloc_bytes: int | None = None
        if self._device.sm_version == "sm_100":
            tmem_matches = re.findall(r'tcgen05\.alloc[^,]+,\s*(\d+)', ptx_source)
            if tmem_matches:
                n_cols = int(tmem_matches[0])
                tmem_alloc_bytes = n_cols * 512

        # Count async copy groups
        async_copy_groups = (
            ptx_source.count("cp.async.commit_group")
            + ptx_source.count("cp.async.bulk.commit_group")
        )
        # cp.async.bulk.commit_group also contains "cp.async.commit_group" as a substring
        # — avoid double-counting by subtracting the bulk occurrences already counted above.
        # Actually "cp.async.bulk.commit_group" does NOT contain "cp.async.commit_group"
        # as a substring (it has ".bulk." in between), so no double-counting issue.

        # Arithmetic intensity estimate
        fused_fp = by_category.get("fused_fp", 0)
        global_mem = by_category.get("global_mem", 0)
        mma_total = mma_warp_count + mma_warpgroup_count + mma_cta_count
        flop_estimate = fused_fp * 2 + mma_total * 256  # rough: 256 flops per MMA instruction
        byte_estimate = global_mem * 4  # assume 32-bit word per global op
        arithmetic_intensity: float | None = (
            flop_estimate / byte_estimate if byte_estimate > 0 else None
        )

        # Build partial result to pass to bottleneck()
        result = TracerResult(
            arch=self._device.name,
            sm_version=self._device.sm_version,
            total_instructions=total_instructions,
            by_category=by_category,
            mma_warp_count=mma_warp_count,
            mma_warpgroup_count=mma_warpgroup_count,
            mma_cta_count=mma_cta_count,
            smem_alloc_bytes=smem_alloc_bytes,
            tmem_alloc_bytes=tmem_alloc_bytes,
            async_copy_groups=async_copy_groups,
            bottleneck="mixed",  # placeholder; set below
            arithmetic_intensity=arithmetic_intensity,
            unsupported_instructions=unsupported_instructions,
            notes=notes,
        )
        result.bottleneck = self.bottleneck(result)
        return result

    def trace_file(self, path: str) -> TracerResult:
        with open(path) as f:
            return self.trace(f.read())

    def bottleneck(self, result: TracerResult) -> str:
        arch = _ARCH_TABLE.get(result.sm_version)
        if arch is None:
            return "mixed"

        mma_total = result.mma_warp_count + result.mma_warpgroup_count + result.mma_cta_count
        smem_total = result.by_category.get("smem", 0)
        global_total = result.by_category.get("global_mem", 0)
        fp_total = result.by_category.get("fused_fp", 0)

        mma_cycles    = mma_total    * _MMA_LATENCY.get(result.sm_version, 17)
        smem_cycles   = smem_total   * arch.smem_latency_cycles
        global_cycles = global_total * arch.global_l2_latency_cycles
        fp_cycles     = fp_total     * 4

        total = mma_cycles + smem_cycles + global_cycles + fp_cycles
        if total == 0:
            return "mixed"
        dominant = max(mma_cycles, smem_cycles, global_cycles, fp_cycles)
        if dominant / total > 0.6:
            if dominant == mma_cycles:    return "compute"
            if dominant == global_cycles: return "memory"
            if dominant == smem_cycles:   return "latency"
        return "mixed"
