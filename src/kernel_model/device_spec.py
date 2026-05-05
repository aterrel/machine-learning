from __future__ import annotations

from dataclasses import dataclass

_GPU_TABLE: dict[str, "DeviceSpec"] = {}


def _register(*aliases: str, **kwargs):
    spec = DeviceSpec(**kwargs)
    for alias in aliases:
        _GPU_TABLE[alias] = spec


@dataclass
class DeviceSpec:
    name: str
    sm_count: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    shared_mem_per_sm: int
    registers_per_sm: int
    peak_fp32_gflops: float
    memory_bandwidth_gbs: float
    sm_version: str

    @classmethod
    def from_name(cls, gpu_name: str) -> "DeviceSpec":
        key = gpu_name.lower().strip().replace(" ", "").replace("-", "")
        # Also try with hyphens removed from table keys
        for alias, spec in _GPU_TABLE.items():
            normalized_alias = alias.replace("-", "")
            if key == normalized_alias:
                return spec
        valid = sorted(set(_GPU_TABLE.keys()))
        raise KeyError(
            f"Unknown GPU name {gpu_name!r}. Valid names: {valid}"
        )

    @classmethod
    def from_device(cls, device) -> "DeviceSpec":
        from cuda.core.experimental import Device  # noqa: F401 — deferred import

        props = device.properties
        name = props.name.decode() if isinstance(props.name, (bytes, bytearray)) else str(props.name)
        cc = device.compute_capability
        major, minor = cc.major, cc.minor
        sm_version = f"sm_{major}{minor}"

        sm_count = props.multiProcessorCount
        max_warps_per_sm = props.maxThreadsPerMultiProcessor // props.warpSize
        max_blocks_per_sm = props.maxBlocksPerMultiprocessor
        shared_mem_per_sm = props.sharedMemPerMultiprocessor
        registers_per_sm = props.regsPerMultiprocessor

        # Peak FP32 and bandwidth are not in device properties — look up SKU table
        name_key = name.lower().strip().replace(" ", "").replace("-", "")
        matched_spec = None
        for alias, spec in _GPU_TABLE.items():
            if alias.replace("-", "") in name_key or name_key in alias.replace("-", ""):
                matched_spec = spec
                break

        if matched_spec is None:
            raise ValueError(
                f"GPU {name!r} not found in the SKU table. "
                f"peak_fp32_gflops and memory_bandwidth_gbs cannot be determined "
                f"automatically from live device properties. "
                f"Please construct DeviceSpec manually: "
                f"DeviceSpec(name={name!r}, sm_count={sm_count}, "
                f"max_warps_per_sm={max_warps_per_sm}, max_blocks_per_sm={max_blocks_per_sm}, "
                f"shared_mem_per_sm={shared_mem_per_sm}, registers_per_sm={registers_per_sm}, "
                f"peak_fp32_gflops=<value>, memory_bandwidth_gbs=<value>, "
                f"sm_version={sm_version!r})"
            )

        return cls(
            name=name,
            sm_count=sm_count,
            max_warps_per_sm=max_warps_per_sm,
            max_blocks_per_sm=max_blocks_per_sm,
            shared_mem_per_sm=shared_mem_per_sm,
            registers_per_sm=registers_per_sm,
            peak_fp32_gflops=matched_spec.peak_fp32_gflops,
            memory_bandwidth_gbs=matched_spec.memory_bandwidth_gbs,
            sm_version=sm_version,
        )


# V100 16GB
_register(
    "v100",
    name="V100 16GB",
    sm_count=80,
    max_warps_per_sm=64,
    max_blocks_per_sm=32,
    shared_mem_per_sm=98304,
    registers_per_sm=65536,
    peak_fp32_gflops=15700.0,
    memory_bandwidth_gbs=900.0,
    sm_version="sm_70",
)

# A100 40GB
_register(
    "a100-40gb",
    name="A100 40GB",
    sm_count=108,
    max_warps_per_sm=64,
    max_blocks_per_sm=32,
    shared_mem_per_sm=167936,
    registers_per_sm=65536,
    peak_fp32_gflops=19500.0,
    memory_bandwidth_gbs=1555.0,
    sm_version="sm_80",
)

# A100 80GB
_register(
    "a100",
    "a100-80gb",
    name="A100 80GB",
    sm_count=108,
    max_warps_per_sm=64,
    max_blocks_per_sm=32,
    shared_mem_per_sm=167936,
    registers_per_sm=65536,
    peak_fp32_gflops=19500.0,
    memory_bandwidth_gbs=2039.0,
    sm_version="sm_80",
)

# H100 80GB SXM
_register(
    "h100",
    "h100-sxm",
    name="H100 80GB SXM",
    sm_count=132,
    max_warps_per_sm=64,
    max_blocks_per_sm=32,
    shared_mem_per_sm=233472,
    registers_per_sm=65536,
    peak_fp32_gflops=51200.0,
    memory_bandwidth_gbs=3350.0,
    sm_version="sm_90",
)

# B100 80GB SXM
_register(
    "b100",
    "b100-sxm",
    name="B100 80GB SXM",
    sm_count=148,
    max_warps_per_sm=64,
    max_blocks_per_sm=32,
    shared_mem_per_sm=233472,
    registers_per_sm=65536,
    peak_fp32_gflops=67000.0,
    memory_bandwidth_gbs=8000.0,
    sm_version="sm_100",
)

# RTX 3090
_register(
    "rtx3090",
    name="RTX 3090",
    sm_count=82,
    max_warps_per_sm=48,
    max_blocks_per_sm=16,
    shared_mem_per_sm=102400,
    registers_per_sm=65536,
    peak_fp32_gflops=35600.0,
    memory_bandwidth_gbs=936.0,
    sm_version="sm_86",
)

# RTX 4090
_register(
    "rtx4090",
    name="RTX 4090",
    sm_count=128,
    max_warps_per_sm=48,
    max_blocks_per_sm=24,
    shared_mem_per_sm=102400,
    registers_per_sm=65536,
    peak_fp32_gflops=82600.0,
    memory_bandwidth_gbs=1008.0,
    sm_version="sm_89",
)

# RTX 5090
_register(
    "rtx5090",
    name="RTX 5090",
    sm_count=170,
    max_warps_per_sm=48,
    max_blocks_per_sm=24,
    shared_mem_per_sm=102400,
    registers_per_sm=65536,
    peak_fp32_gflops=104800.0,
    memory_bandwidth_gbs=1792.0,
    sm_version="sm_100",
)
