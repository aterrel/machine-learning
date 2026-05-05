from __future__ import annotations
import pathlib
from src.kernel_model import DeviceSpec, PTXTracer, TracerResult

FIXTURES = pathlib.Path(__file__).parent / "ptx_fixtures"


def _print_result(result: TracerResult, title: str) -> None:
    total = result.total_instructions
    print(f"\nKernel : {title}")
    print(f"Target : {result.arch} ({result.sm_version})")
    print("-" * 55)
    print(f"Total instructions : {total}")

    print("\nInstruction mix:")
    for cat, count in sorted(result.by_category.items()):
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"  {cat:<18}: {count:4d}  ({pct:5.1f}%)")

    print("\nMMA scope breakdown:")
    print(f"  mma.sync  (warp,  32t) : {result.mma_warp_count}")
    print(f"  wgmma     (wgrp, 128t) : {result.mma_warpgroup_count}")
    print(f"  tcgen05   (CTA,  tmem) : {result.mma_cta_count}")

    smem_str = f"{result.smem_alloc_bytes:,} bytes" if result.smem_alloc_bytes is not None else "n/a (no .shared declaration found)"
    tmem_str = f"{result.tmem_alloc_bytes:,} bytes" if result.tmem_alloc_bytes is not None else "n/a (not sm_100)"
    print(f"\nSmem allocation    : {smem_str}")
    print(f"TMEM allocation    : {tmem_str}")
    print(f"Async copy groups  : {result.async_copy_groups}")

    ai_str = f"{result.arithmetic_intensity:.3f} FLOP/byte" if result.arithmetic_intensity is not None else "n/a (no global mem ops)"
    print(f"\nEst. arith. intensity : {ai_str}")
    print(f"Bottleneck            : {result.bottleneck}")

    if result.unsupported_instructions:
        print(f"\n[!] Unsupported instructions for {result.sm_version}:")
        for instr in sorted(set(result.unsupported_instructions)):
            print(f"    {instr}")

    if result.notes:
        print(f"\nNotes ({len(result.notes)}):")
        seen: set[str] = set()
        for note in result.notes:
            if note not in seen:
                print(f"  - {note}")
                seen.add(note)

    print("-" * 55)


def main() -> None:
    a100 = DeviceSpec.from_name("a100")
    h100 = DeviceSpec.from_name("h100")

    print("=" * 55)
    print("    PTX Kernel Execution Tracer — Sprint 8 Demo")
    print("=" * 55)

    for fixture_name in ["vector_add.ptx", "gemm_mma.ptx"]:
        ptx = (FIXTURES / fixture_name).read_text()
        for device in [a100, h100]:
            tracer = PTXTracer(device)
            result = tracer.trace(ptx)
            _print_result(result, f"{fixture_name} -> {device.name}")


if __name__ == "__main__":
    main()
