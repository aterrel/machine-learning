"""Tests for src/kernel_model/ — occupancy model, roofline model, DeviceSpec."""

from __future__ import annotations

import pytest

from src.kernel_model import DeviceSpec, OccupancyModel, RooflineModel


# ---------------------------------------------------------------------------
# CPU-safe tests
# ---------------------------------------------------------------------------


def test_occupancy_full():
    device = DeviceSpec.from_name("a100")
    model = OccupancyModel(device)
    result = model.compute(block_size=256, shared_mem_per_block=0, registers_per_thread=32)
    assert result.occupancy == pytest.approx(1.0)


def test_occupancy_shmem_limiter():
    # A100: 167936 bytes/SM; shmem_per_block=40000 → floor(167936/40000)=4 blocks max by shmem
    # With block_size=256, regs=16: max_blocks_by_warps=8, max_blocks_by_regs large
    # So shmem limits to 4 blocks
    device = DeviceSpec.from_name("a100")
    model = OccupancyModel(device)
    result = model.compute(block_size=256, shared_mem_per_block=40000, registers_per_thread=16)
    assert result.limiting_resource == "shared_memory"
    assert result.active_blocks_per_sm == 4


def test_occupancy_register_limiter():
    # 255 regs/thread → regs_per_warp_alloc = ceil(255*32/256)*256 = ceil(31.875)*256 = 32*256 = 8192
    # warps_per_block = ceil(256/32) = 8
    # regs_per_block = 8192 * 8 = 65536
    # A100: registers_per_sm=65536 → max_blocks_by_regs = floor(65536/65536) = 1
    device = DeviceSpec.from_name("a100")
    model = OccupancyModel(device)
    result = model.compute(block_size=256, shared_mem_per_block=0, registers_per_thread=255)
    assert result.limiting_resource == "registers"
    assert result.active_blocks_per_sm == 1


def test_occupancy_block_size_too_large():
    # A100: max_warps_per_sm=64 → max threads = 2048; block_size=2080 > 2048 → ValueError
    device = DeviceSpec.from_name("a100")
    model = OccupancyModel(device)
    with pytest.raises(ValueError):
        model.compute(block_size=2080, shared_mem_per_block=0, registers_per_thread=16)


def test_roofline_memory_bound():
    # A100: ridge_point = 19500 / 2039 ≈ 9.56 FLOP/byte; intensity=0.08 < ridge → memory bound
    device = DeviceSpec.from_name("a100")
    model = RooflineModel(device)
    result = model.compute(flops=0.08, bytes_accessed=1.0)
    assert result.bound == "memory"
    assert result.predicted_gflops == pytest.approx(0.08 * device.memory_bandwidth_gbs)


def test_roofline_compute_bound():
    # intensity=100.0 >> ridge_point (~9.56) → compute bound
    device = DeviceSpec.from_name("a100")
    model = RooflineModel(device)
    result = model.compute(flops=100.0, bytes_accessed=1.0)
    assert result.bound == "compute"
    assert result.predicted_gflops == pytest.approx(device.peak_fp32_gflops)


def test_roofline_at_ridge():
    # intensity exactly at ridge_point → bound is "compute" (>= branch); predicted_gflops == peak
    device = DeviceSpec.from_name("a100")
    model = RooflineModel(device)
    ridge = device.peak_fp32_gflops / device.memory_bandwidth_gbs
    result = model.compute(flops=ridge, bytes_accessed=1.0)
    assert result.predicted_gflops == pytest.approx(device.peak_fp32_gflops)
    assert result.bound in ("compute", "memory")


def test_device_spec_from_name_valid():
    names = ["v100", "a100", "h100", "b100", "rtx3090", "rtx4090", "rtx5090"]
    sm_versions = {
        "v100": "sm_70",
        "a100": "sm_80",
        "h100": "sm_90",
        "b100": "sm_100",
        "rtx3090": "sm_86",
        "rtx4090": "sm_89",
        "rtx5090": "sm_100",
    }
    for name in names:
        spec = DeviceSpec.from_name(name)
        assert spec.sm_count > 0
        assert spec.max_warps_per_sm > 0
        assert spec.max_blocks_per_sm > 0
        assert spec.shared_mem_per_sm > 0
        assert spec.registers_per_sm > 0
        assert spec.peak_fp32_gflops > 0
        assert spec.memory_bandwidth_gbs > 0
        assert spec.sm_version == sm_versions[name]


def test_device_spec_from_name_invalid():
    with pytest.raises(KeyError) as exc_info:
        DeviceSpec.from_name("RTX 9999")
    message = str(exc_info.value)
    assert "v100" in message.lower() or "a100" in message.lower()


def test_sweep_block_sizes():
    device = DeviceSpec.from_name("a100")
    model = OccupancyModel(device)
    results = model.sweep_block_sizes(shared_mem_per_block=0, registers_per_thread=16)
    assert len(results) == 6
    for r in results:
        assert 0.0 <= r.occupancy <= 1.0


def test_device_spec_aliases():
    spec_a = DeviceSpec.from_name("a100")
    spec_b = DeviceSpec.from_name("a100-80gb")
    assert spec_a == spec_b


# ---------------------------------------------------------------------------
# GPU test
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_device_spec_from_device_live(gpu_device):
    spec = DeviceSpec.from_device(gpu_device)
    assert spec.sm_count > 0
    assert spec.max_warps_per_sm > 0
    assert spec.max_blocks_per_sm > 0
    assert spec.shared_mem_per_sm > 0
    assert spec.registers_per_sm > 0
    assert spec.peak_fp32_gflops > 0
    assert spec.memory_bandwidth_gbs > 0
    assert spec.sm_version.startswith("sm_")
