from __future__ import annotations

import math
from dataclasses import dataclass

from .device_spec import DeviceSpec


@dataclass
class OccupancyResult:
    block_size: int
    warps_per_block: int
    active_blocks_per_sm: int
    active_warps_per_sm: int
    max_warps_per_sm: int
    occupancy: float
    limiting_resource: str


class OccupancyModel:
    def __init__(self, device: DeviceSpec) -> None:
        self._device = device

    def compute(
        self,
        block_size: int,
        shared_mem_per_block: int,
        registers_per_thread: int,
    ) -> OccupancyResult:
        dev = self._device
        max_block_threads = dev.max_warps_per_sm * 32
        if block_size > max_block_threads:
            raise ValueError(
                f"block_size={block_size} exceeds max threads per SM "
                f"({max_block_threads} = {dev.max_warps_per_sm} warps * 32 threads/warp)"
            )

        warps_per_block = math.ceil(block_size / 32)

        max_blocks_by_warps = math.floor(dev.max_warps_per_sm / warps_per_block)

        if shared_mem_per_block > 0:
            max_blocks_by_shmem = math.floor(dev.shared_mem_per_sm / shared_mem_per_block)
        else:
            max_blocks_by_shmem = dev.max_blocks_per_sm

        regs_per_warp_alloc = math.ceil(registers_per_thread * 32 / 256) * 256
        regs_per_block = regs_per_warp_alloc * warps_per_block
        if regs_per_block > 0:
            max_blocks_by_regs = math.floor(dev.registers_per_sm / regs_per_block)
        else:
            max_blocks_by_regs = dev.max_blocks_per_sm

        limits = {
            "warps": max_blocks_by_warps,
            "shared_memory": max_blocks_by_shmem,
            "registers": max_blocks_by_regs,
            "blocks": dev.max_blocks_per_sm,
        }

        active_blocks = min(limits.values())
        limiting_resource = min(limits, key=limits.__getitem__)

        active_warps = active_blocks * warps_per_block
        occupancy = active_warps / dev.max_warps_per_sm

        return OccupancyResult(
            block_size=block_size,
            warps_per_block=warps_per_block,
            active_blocks_per_sm=active_blocks,
            active_warps_per_sm=active_warps,
            max_warps_per_sm=dev.max_warps_per_sm,
            occupancy=occupancy,
            limiting_resource=limiting_resource,
        )

    def sweep_block_sizes(
        self,
        shared_mem_per_block: int,
        registers_per_thread: int,
        block_sizes: list[int] | None = None,
    ) -> list[OccupancyResult]:
        if block_sizes is None:
            block_sizes = [32, 64, 128, 256, 512, 1024]
        return [
            self.compute(bs, shared_mem_per_block, registers_per_thread)
            for bs in block_sizes
        ]
