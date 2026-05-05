"""Demo 09: Kernel Performance Model — occupancy + roofline for the vector-add kernel."""

from __future__ import annotations

from src.kernel_model import DeviceSpec, OccupancyModel, RooflineModel


def main() -> None:
    device = DeviceSpec.from_name("a100")

    n = 1_000_000
    shared_mem_per_block = 0
    registers_per_thread = 16
    flops = float(n)
    bytes_accessed = 3 * n * 4  # 2 reads + 1 write of float32

    print("=== Kernel Performance Model Demo ===")
    print(f"Kernel: vector_add  (n={n:,}, float32)")

    print(f"\n--- Occupancy Analysis ({device.name}) ---")
    model = OccupancyModel(device)
    results = model.sweep_block_sizes(
        shared_mem_per_block=shared_mem_per_block,
        registers_per_thread=registers_per_thread,
    )

    header = (
        f"{'Block Size':>10}  {'Warps/Block':>11}  {'Active Blocks/SM':>16}  "
        f"{'Active Warps/SM':>15}  {'Occupancy':>9}  {'Limiter'}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)
    for r in results:
        print(
            f"{r.block_size:>10}  {r.warps_per_block:>11}  {r.active_blocks_per_sm:>16}  "
            f"{r.active_warps_per_sm:>15}  {r.occupancy * 100:>8.1f}%  {r.limiting_resource}"
        )

    best = max(results, key=lambda r: r.occupancy)
    print(f"\nRecommended block size: {best.block_size} ({best.occupancy * 100:.1f}% occupancy)")

    print(f"\n--- Roofline Analysis ({device.name}) ---")
    roof_model = RooflineModel(device)
    roof = roof_model.compute(flops=flops, bytes_accessed=bytes_accessed)

    print(f"{'Arithmetic intensity':<20} : {roof.arithmetic_intensity:.3f} FLOP/byte")
    print(f"{'Ridge point':<20} : {roof.ridge_point:.2f} FLOP/byte")
    print(f"{'Bound':<20} : {roof.bound}")
    print(f"{'Predicted throughput':<20} : {roof.predicted_gflops:,.1f} GFLOP/s  "
          f"(of {device.peak_fp32_gflops:,.0f} GFLOP/s peak)")
    if roof.bound == "memory":
        effective_bw = roof.arithmetic_intensity * device.memory_bandwidth_gbs
        print(f"{'Memory bandwidth use':<20} : {effective_bw:,.0f} GB/s"
              f"     (of {device.memory_bandwidth_gbs:,.0f} GB/s peak)")
    else:
        print(f"{'Memory bandwidth use':<20} : {device.memory_bandwidth_gbs:,.0f} GB/s"
              f"     (of {device.memory_bandwidth_gbs:,.0f} GB/s peak)")


if __name__ == "__main__":
    main()
