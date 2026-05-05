# ARCH-004: Kernel Performance Model Library (src/kernel_model/)

**Status**: Draft
**Created**: 2026-05-05
**Author**: Software Architect
**REQ Reference**: REQ-0010
**Approved By**: —

---

## Overview

`src/kernel_model/` is a pure-Python analytical library that models GPU kernel performance without executing any GPU code. It implements two classical GPU performance models — occupancy and roofline — parameterized by a `DeviceSpec` dataclass that can be populated from a hardcoded GPU SKU table or from a live `cuda.core.Device` at runtime.

---

## Context

The project already delivers working CUDA demos but lacks a conceptual framework for reasoning about *why* a kernel runs at a given speed. This library fills that gap. Because both models are pure arithmetic, the library has zero GPU dependencies at module level: it runs on any Python 3.11 machine and is suitable for teaching, pre-implementation analysis, and offline exploration. GPU dependency is optional, deferred to `DeviceSpec.from_device()`.

---

## Design

### Module Structure

```
src/
  kernel_model/
    __init__.py           # re-exports: DeviceSpec, OccupancyModel, OccupancyResult,
    │                     #             RooflineModel, RooflineResult
    device_spec.py        # DeviceSpec dataclass + GPU SKU table + from_device()
    occupancy.py          # OccupancyModel + OccupancyResult
    roofline.py           # RooflineModel + RooflineResult
demos/
  09_kernel_model/
    __init__.py
    main.py               # CLI demo: occupancy table + roofline summary
tests/
  test_kernel_model.py    # CPU-safe unit tests; all edge cases
```

---

## Key Classes and Data Flow

### DeviceSpec

```python
@dataclass
class DeviceSpec:
    name: str
    sm_count: int
    max_warps_per_sm: int
    max_blocks_per_sm: int
    shared_mem_per_sm: int       # bytes
    registers_per_sm: int
    peak_fp32_gflops: float
    memory_bandwidth_gbs: float

    @classmethod
    def from_name(cls, gpu_name: str) -> "DeviceSpec":
        # Case-insensitive lookup in _GPU_TABLE dict
        # Raises KeyError with list of valid names if not found

    @classmethod
    def from_device(cls, device) -> "DeviceSpec":
        # Deferred import of cuda.core — only called when GPU is present
        # Reads device.properties: max_threads_per_multiprocessor,
        # max_shared_memory_per_multiprocessor, regs_per_multiprocessor,
        # warp_size, multi_processor_count
        # peak_fp32_gflops and memory_bandwidth_gbs require cuBLAS/bandwidth
        # probe or must be set from the SKU table as a fallback
```

**GPU SKU table** (`_GPU_TABLE` dict, keyed by canonical lowercase name):

| Key | GPU |
|-----|-----|
| `"v100"` | Tesla V100 16GB |
| `"a100-40gb"` | A100 40GB SXM |
| `"a100"`, `"a100-80gb"` | A100 80GB SXM |
| `"h100"`, `"h100-sxm"` | H100 80GB SXM |
| `"b100"`, `"b100-sxm"` | B100 80GB SXM |
| `"rtx3090"` | RTX 3090 |
| `"rtx4090"` | RTX 4090 |
| `"rtx5090"` | RTX 5090 |

`from_name()` normalizes the input (lowercase, strip whitespace, remove spaces) before lookup. Multiple aliases map to the same `DeviceSpec`.

---

### OccupancyModel

```python
class OccupancyModel:
    def __init__(self, device: DeviceSpec): ...

    def compute(self,
                block_size: int,
                shared_mem_per_block: int,
                registers_per_thread: int) -> OccupancyResult: ...

    def sweep_block_sizes(self,
                          shared_mem_per_block: int,
                          registers_per_thread: int,
                          block_sizes: list[int] | None = None) -> list[OccupancyResult]:
        # Default block_sizes: [32, 64, 128, 256, 512, 1024]
        ...
```

**Occupancy calculation** (follows NVIDIA's occupancy calculator methodology):

```
warps_per_block = ceil(block_size / 32)

# Limiter 1 — warp count
max_blocks_by_warps = floor(max_warps_per_sm / warps_per_block)

# Limiter 2 — shared memory (skip if shared_mem_per_block == 0)
max_blocks_by_shmem = (
    floor(shared_mem_per_sm / shared_mem_per_block)
    if shared_mem_per_block > 0
    else max_blocks_per_sm
)

# Limiter 3 — registers
# Registers are allocated per warp in multiples of 256 (register allocation granularity)
regs_per_warp_alloc = ceil(registers_per_thread * 32 / 256) * 256
regs_per_block = regs_per_warp_alloc * warps_per_block
max_blocks_by_regs = (
    floor(registers_per_sm / regs_per_block)
    if regs_per_block > 0
    else max_blocks_per_sm
)

# Binding limiter
active_blocks = min(
    max_blocks_by_warps,
    max_blocks_by_shmem,
    max_blocks_by_regs,
    max_blocks_per_sm,
)
active_warps = active_blocks * warps_per_block
occupancy = active_warps / max_warps_per_sm
limiting_resource = argmin(max_blocks_by_warps, max_blocks_by_shmem,
                           max_blocks_by_regs, max_blocks_per_sm)
```

`OccupancyResult`:

```python
@dataclass
class OccupancyResult:
    block_size: int
    warps_per_block: int
    active_blocks_per_sm: int
    active_warps_per_sm: int
    max_warps_per_sm: int
    occupancy: float                 # 0.0–1.0
    limiting_resource: str           # "warps" | "shared_memory" | "registers" | "blocks"
```

---

### RooflineModel

```python
class RooflineModel:
    def __init__(self, device: DeviceSpec): ...

    def compute(self, flops: float, bytes_accessed: float) -> RooflineResult: ...

    def sweep_intensities(self,
                          intensity_range: list[float]) -> list[RooflineResult]: ...
```

**Roofline calculation**:

```
arithmetic_intensity = flops / bytes_accessed          # FLOP/byte
ridge_point = peak_fp32_gflops / memory_bandwidth_gbs  # FLOP/byte

if arithmetic_intensity < ridge_point:
    bound = "memory"
    predicted_gflops = arithmetic_intensity * memory_bandwidth_gbs
else:
    bound = "compute"
    predicted_gflops = peak_fp32_gflops
```

`RooflineResult`:

```python
@dataclass
class RooflineResult:
    arithmetic_intensity: float   # FLOP/byte
    ridge_point: float            # FLOP/byte
    bound: str                    # "compute" | "memory"
    predicted_gflops: float
```

---

## Demo: demos/09_kernel_model/main.py

The demo uses the vector-add kernel from REQ-0001 as a worked example. Its parameters are known:
- Block size: 256 threads
- Shared memory per block: 0 bytes (no shared mem used)
- Registers per thread: ~16 (typical for a simple elementwise kernel)
- FLOPs: `n` additions
- Bytes accessed: `3 * n * 4` bytes (two reads + one write of float32)

The demo prints:

```
=== Kernel Performance Model Demo ===
Kernel: vector_add  (n=1,000,000, float32)

--- Occupancy Analysis (A100 80GB) ---
Block Size  Warps/Block  Active Blocks/SM  Active Warps/SM  Occupancy  Limiter
----------  -----------  ----------------  ---------------  ---------  -------
        32            1                32               32     50.0%   blocks
        64            2                32               64    100.0%   blocks
       128            4                32              128     100.0%   blocks (limit: 32 blocks/SM)
       256            8                16              128    100.0%   warps (tie: all full)
...

Recommended block size: 256 (100.0% occupancy)

--- Roofline Analysis (A100 80GB) ---
Arithmetic intensity : 0.083 FLOP/byte
Ridge point          : 12.55 FLOP/byte
Bound                : memory
Predicted throughput : 162.1 GFLOP/s  (of 19,500 GFLOP/s peak)
Memory bandwidth use : 1,955 GB/s     (of 2,039 GB/s peak)
```

---

## Design Decisions

### 1. Zero GPU imports at module level

`src/kernel_model/` never imports `cuda`, `cupy`, or any GPU library at module level. `DeviceSpec.from_device()` performs a deferred `import cuda.core` inside the method body. This guarantees the library is fully usable without a GPU installed, matching REQ-0010-NF1/NF2.

**Tradeoff**: `from_device()` will fail at call time if `cuda-python` is not installed, not at import time. This is the correct behavior for an optional GPU path.

### 2. Registers per thread rounded to warp allocation granularity

CUDA allocates registers per warp in multiples of 256 (on Volta+). Ignoring this granularity produces optimistic occupancy estimates. The occupancy model always rounds up before computing register pressure.

**Tradeoff**: For small register counts (e.g., 16 regs/thread × 32 threads = 512 registers/warp — already a multiple of 256), the rounding has no effect. For 1–7 regs/thread, the rounding doubles apparent register usage — intentionally conservative.

### 3. DeviceSpec.from_device() does not probe peak FP32 or memory bandwidth

Live device properties (`cuda.core.Device.properties`) expose SM count, warp size, shared mem, and register counts, but **not** peak FP32 GFLOP/s or memory bandwidth — those are product-level specs, not hardware register values. `from_device()` reads what it can from `Device.properties`, then attempts a name match against the GPU name string (e.g., `"NVIDIA A100-SXM4-80GB"`) to fill in `peak_fp32_gflops` and `memory_bandwidth_gbs`. If the name does not match any table entry, it raises a `ValueError` with instructions to construct a `DeviceSpec` manually.

**Tradeoff**: `from_device()` is not fully automatic for unknown SKUs. This is the honest trade-off: peak performance specs are not available programmatically without running a benchmark or reading a datasheet.

### 4. sweep_block_sizes defaults to [32, 64, 128, 256, 512, 1024]

These are the common power-of-two block sizes used in practice. The caller can pass any list; the default covers the useful range without producing a wall of output.

---

## Module Interaction with Existing src/

`src/kernel_model/` is a standalone sub-package — it does not import from `src/kernels/` or `src/utils/`. The only coupling is via `DeviceSpec.from_device()` accepting a `cuda.core.Device` object (which the caller obtains from `src/utils/device.py`). This keeps the dependency direction clean: `kernel_model` is a pure analysis layer that happens to accept a CUDA device as optional input.

```
src/utils/device.py  →  (caller passes Device to)  →  DeviceSpec.from_device()
src/kernel_model/    →  (no imports from src/utils or src/kernels)
```

---

## Testing Strategy

All tests in `tests/test_kernel_model.py` are CPU-safe (no `@pytest.mark.gpu` needed for the model itself):

- `test_occupancy_full` — block_size=256, shmem=0, regs=32 on A100: expect ~100% occupancy
- `test_occupancy_shmem_limiter` — large shmem_per_block forces shmem to be the limiter
- `test_occupancy_register_limiter` — 255 regs/thread forces register limiter
- `test_occupancy_block_size_too_large` — block_size=2048: expect `ValueError` (exceeds max threads per block)
- `test_roofline_memory_bound` — low arithmetic intensity (e.g., 0.08 FLOP/byte on A100): `bound == "memory"`
- `test_roofline_compute_bound` — high arithmetic intensity (e.g., 100 FLOP/byte): `bound == "compute"`
- `test_roofline_at_ridge` — intensity exactly at ridge point: either classification is acceptable; `predicted_gflops == peak_fp32_gflops`
- `test_device_spec_from_name_valid` — all table entries round-trip correctly
- `test_device_spec_from_name_invalid` — unknown name raises `KeyError` with list of valid options
- `test_sweep_block_sizes` — returns one result per block size; occupancy is non-decreasing for simple kernels with no shmem/register pressure
- `test_device_spec_aliases` — `"a100"` and `"a100-80gb"` return identical specs

One `@pytest.mark.gpu` test: `test_device_spec_from_device_live` — calls `from_device()` with a real device and checks all fields are positive.

---

## Consequences

### Positive
- Works without a GPU — broadens audience to learners who don't yet have CUDA access
- Analytical models are deterministic and fast (sub-millisecond per call)
- Pedagogically valuable: makes visible the hidden trade-offs in block size and register usage

### Negative / Tradeoffs
- Models are approximate — they assume ideal conditions (no bank conflicts, perfect memory coalescing, no instruction throughput bottlenecks)
- `peak_fp32_gflops` and `memory_bandwidth_gbs` are not readable from live device properties; from_device() must do GPU name matching
- Register count per thread must be obtained from `nvcc --ptxas-info` or profiler — it is not a value a developer can compute analytically without compiling the kernel

### Risks
- GPU specs for B100 and RTX 5090 should be verified against official NVIDIA datasheets before the table is considered authoritative
- `cuda.core.Device.properties` field names may differ across cuda-python versions; from_device() should be tested against the installed version
