# ARCH-001: Overall System Architecture — CUDA Python ML Demos

**Status**: Draft
**Created**: 2026-05-01
**Author**: Software Architect
**REQ Reference**: REQ-0001, REQ-0002, REQ-0003, REQ-0004, REQ-0005, REQ-0006
**Approved By**: Software Architect (2026-05-01)

---

## Overview

CUDA Python ML Demos is a structured collection of self-contained Python demos and a shared utility library, organized so each demo is independently runnable while all sharing common abstractions for device management, memory, kernel compilation, and benchmarking. There is no server, database, or web layer — the system is pure Python running locally on a CUDA-capable machine.

---

## Context

The project targets ML practitioners who know Python and NumPy but are new to direct GPU programming. The architecture must be learnable (each demo self-contained, minimal dependencies), accurate (correct CUDA usage patterns), and benchmarkable (every demo participates in a unified timing framework). The constraint that all demos use `cuda-python` directly (not CuPy/PyTorch as the primary path) shapes the entire design.

---

## Design

### Module Structure

```
cuda-python-ml-demos/
├── src/
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── compiler.py        # KernelCompiler: NVRTC wrapper around cuda.core.Program
│   │   └── compiled_kernel.py # CompiledKernel: typed launch helper
│   └── utils/
│       ├── __init__.py
│       ├── device.py          # Device init, property queries, CUDA availability check
│       ├── memory.py          # DeviceBuffer context manager, pinned alloc, OOM helpers
│       └── timing.py          # CUDA event-based timer, BenchmarkRunner, BenchmarkResult
├── demos/
│   ├── 01_core_apis/
│   │   ├── main.py            # Entry point: device info + vector-add demo
│   │   ├── device_info.py     # print_device_info()
│   │   └── vector_add.py      # Vector-add kernel: compile → launch → verify
│   ├── 02_kmeans/
│   │   ├── main.py
│   │   ├── gpu_kmeans.py      # GPU k-means (Lloyd's algorithm)
│   │   └── cpu_kmeans.py      # NumPy reference
│   ├── 03_pca/
│   │   ├── main.py
│   │   ├── gpu_pca.py         # GPU PCA via covariance matrix
│   │   └── cpu_pca.py
│   ├── 04_linear_model/
│   │   ├── main.py
│   │   ├── gpu_linear.py      # GPU linear regression (normal equation)
│   │   └── cpu_linear.py
│   ├── 05_naive_bayes/
│   │   ├── main.py
│   │   ├── gpu_nb.py
│   │   └── cpu_nb.py
│   └── 06_interop/
│       ├── cupy_interop.py
│       ├── torch_interop.py
│       └── pipeline.py
├── benchmarks/
│   ├── run_all.py             # CLI: runs all demos, prints summary table
│   └── results/               # JSON output (gitignored)
├── notebooks/                 # Jupyter versions (one per demo)
├── tests/
│   ├── conftest.py            # pytest GPU mark, device fixture
│   ├── test_device.py
│   ├── test_kernels.py
│   ├── test_kmeans.py
│   ├── test_pca.py
│   ├── test_linear.py
│   └── test_memory.py
└── pyproject.toml             # ruff config, pytest config, dependencies
```

### Key Classes / Interfaces

```python
# src/utils/device.py
def check_cuda_available() -> bool: ...
def get_device(index: int = 0) -> cuda.core.Device: ...
def get_device_props(device: cuda.core.Device) -> dict: ...

# src/utils/memory.py
class DeviceBuffer:
    def __init__(self, n_bytes: int): ...
    def __enter__(self) -> "DeviceBuffer": ...
    def __exit__(self, *args) -> None: ...
    @property
    def ptr(self) -> int: ...
    @property
    def __cuda_array_interface__(self) -> dict: ...

# src/utils/timing.py
@dataclass
class BenchmarkResult:
    demo_name: str
    cpu_time_mean_s: float
    gpu_time_mean_s: float
    speedup: float
    correct: bool
    max_abs_error: float

class BenchmarkRunner:
    def run(self, cpu_fn, gpu_fn, n_repeats: int = 5, **kwargs) -> BenchmarkResult: ...

# src/kernels/compiler.py
class KernelCompiler:
    def compile(self, source: str, kernel_name: str) -> "CompiledKernel": ...

# src/kernels/compiled_kernel.py
class CompiledKernel:
    def launch(self, grid: tuple[int,int,int], block: tuple[int,int,int],
               args: list, stream=None) -> None: ...
```

### Data Flow

```
User invokes demo
    │
    ▼
demos/XX_name/main.py
    │
    ├── generate synthetic data (numpy, seeded)
    │
    ├── CPU baseline
    │   └── cpu_impl.py → result_cpu, time_cpu
    │
    └── GPU path
        ├── src/utils/device.py    → Device, check availability
        ├── src/utils/memory.py    → DeviceBuffer (alloc H2D)
        ├── src/kernels/compiler.py → CompiledKernel (NVRTC)
        ├── CompiledKernel.launch() → kernel executes on GPU
        ├── DeviceBuffer → copy D2H → result_gpu
        └── src/utils/timing.py    → BenchmarkResult
            │
            └── print: GPU Xs | CPU Xs | Speedup Xx | Correct: True/False
```

---

## Key Design Decisions

1. **Each demo is self-contained**: Demos import only from `src/` — never from each other. This ensures any demo can be read and run in isolation without understanding the full repo. Tradeoff: some boilerplate repetition in main.py files, but this is intentional for pedagogical clarity.

2. **`cuda.core` as the primary API, `cuda.bindings` for contrast only**: `cuda.core` is the stable, idiomatic surface. We drop to `cuda.bindings` only in REQ-0001 to show the raw layer for educational contrast. Using raw bindings everywhere would make demos unnecessarily verbose and harder to read.

3. **No framework in `src/kernels/`**: The kernel compiler and launcher use only `cuda-python` — no CuPy, no PyTorch. This forces the demos to demonstrate actual CUDA Python usage, not just wrap a higher-level library. Interop is isolated to `demos/06_interop/`.

4. **BenchmarkResult is a dataclass, not a class hierarchy**: Simplest structure that works. All demos return the same shape so `benchmarks/run_all.py` can tabulate them uniformly without polymorphism.

5. **pytest markers for GPU tests**: `@pytest.mark.gpu` lets CI skip GPU tests on machines without a GPU. CPU-only tests (data generation, result parsing) always run.

---

## Consequences

### Positive
- Any demo can be opened and understood without reading the rest of the repo
- BenchmarkRunner makes adding a new demo trivial (implement cpu_fn + gpu_fn, call runner)
- GPU/CPU correctness check is always present — no demo can "pass" without verifying its output

### Negative / Trade-offs
- Some boilerplate repetition across demo main.py files
- No shared state between demos means each re-initializes the CUDA device — small overhead
- Without a proper memory pool in src/, demos that allocate repeatedly may be slower than necessary for benchmarking

### Risks
- `cuda-python` API surface is still evolving — `cuda.core` was introduced in 12.3; demos may need version guards
- NVRTC compilation at import time could slow first-run experience; consider lazy compilation

---

## Testing Strategy

- `tests/conftest.py` defines a `cuda_device` fixture that skips if no GPU is present
- CPU baseline tests run without GPU (fast, always run in CI)
- GPU tests tagged `@pytest.mark.gpu` run only in GPU CI or locally
- Each demo has a corresponding test that runs the demo function and checks `BenchmarkResult.correct == True`

---

## Implementation Notes for Programmer

1. Start with `src/utils/device.py` and `src/utils/memory.py` — everything else depends on these
2. Implement `demos/01_core_apis/` next — it's the integration smoke test for the entire src/ layer
3. Then implement `src/kernels/compiler.py` (KernelCompiler + CompiledKernel) before moving to demos 02-05
4. `benchmarks/run_all.py` should be implemented after at least 2 demos are working, so it has something to benchmark
5. Notebooks are generated last — copy demo code into notebook cells

**Key gotchas**:
- `cuda.core.Device` must be activated before any memory allocation on it: `device.set_current()`
- Kernel arguments must be passed as a list of ctypes pointers, not Python objects directly
- Always synchronize the stream before reading device memory back to host
- `cuda.core.Program` takes the kernel source as a string and the kernel function name separately

---

## ADRs

- ADR-001: Use cuda.core as primary API (vs raw cuda.bindings) — see ARCH-001 decision #2
- ADR-002: Self-contained demos (vs shared demo base class) — see ARCH-001 decision #1
