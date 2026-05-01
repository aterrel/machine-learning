# ARCH-002: CUDA Kernel Compilation and Launch Pipeline

**Status**: Draft
**Created**: 2026-05-01
**Author**: Software Architect
**REQ Reference**: REQ-0003, REQ-0001, REQ-0002
**Approved By**: Software Architect (2026-05-01)

---

## Overview

This document specifies the design of the kernel compilation and launch pipeline — the subsystem that takes CUDA C kernel source strings, compiles them to PTX via NVRTC (using `cuda.core.Program`), and launches them on the device. This is the most technically complex part of the project and is used by all ML algorithm demos.

---

## Context

CUDA Python's `cuda.core.Program` wraps NVRTC (NVIDIA Runtime Compilation), enabling JIT compilation of CUDA C source strings to PTX at Python runtime. This avoids needing `nvcc` separately and enables kernel source to live inline in Python files — a critical property for keeping demos self-contained and readable. The pipeline must be usable with minimal boilerplate so demo code focuses on the ML algorithm, not CUDA infrastructure.

---

## Design

### Module Structure

```
src/kernels/
├── __init__.py
├── compiler.py        # KernelCompiler — wraps cuda.core.Program
└── compiled_kernel.py # CompiledKernel — wraps launched kernel with typed arg handling
```

### Key Classes / Interfaces

```python
# src/kernels/compiler.py
import cuda.core.experimental as cudax
from cuda.core.experimental import Program, ProgramOptions

class KernelCompiler:
    """Compiles CUDA C source strings to launchable kernels via NVRTC."""

    def __init__(self, device: cudax.Device):
        self._device = device
        self._cache: dict[str, "CompiledKernel"] = {}

    def compile(
        self,
        source: str,
        kernel_name: str,
        options: list[str] | None = None,
        cache_key: str | None = None,
    ) -> "CompiledKernel":
        """Compile CUDA C source and return a launchable kernel.

        Args:
            source: CUDA C kernel source as a Python string
            kernel_name: Name of the __global__ function to extract
            options: Extra NVRTC compile flags (e.g., ["-arch=sm_86"])
            cache_key: If set, caches compiled kernel by this key
        Returns:
            CompiledKernel ready to launch
        """
        ...

# src/kernels/compiled_kernel.py
from cuda.core.experimental import LaunchConfig, launch

class CompiledKernel:
    """Wraps a compiled CUDA kernel with a typed launch interface."""

    def __init__(self, kernel, device: cudax.Device):
        self._kernel = kernel
        self._device = device

    def launch(
        self,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list,
        stream=None,
    ) -> None:
        """Launch the kernel on the given grid/block configuration.

        Args:
            grid: (grid_x, grid_y, grid_z) number of blocks
            block: (block_x, block_y, block_z) threads per block
            args: list of kernel arguments (ctypes values or device pointers)
            stream: cuda.core Stream to launch on (uses default if None)
        """
        ...

    @staticmethod
    def compute_grid_1d(n: int, block_size: int) -> tuple[int, int, int]:
        """Compute 1D grid dimensions for n elements."""
        return ((n + block_size - 1) // block_size, 1, 1)
```

### Data Flow

```
CUDA C source string (in demo Python file)
    │
    ▼
KernelCompiler.compile(source, kernel_name)
    │
    ├── cuda.core.experimental.Program(source, code_type="c++")
    ├── program.compile(options=["-arch=sm_XX"])   # NVRTC → PTX
    ├── module = program.get_kernel(kernel_name)   # extract kernel object
    └── CompiledKernel(module, device)
            │
            ▼
    CompiledKernel.launch(grid, block, args, stream)
        │
        ├── LaunchConfig(grid=grid, block=block)
        ├── launch(stream, config, kernel, *args)
        └── stream.sync()   # caller's responsibility
```

### Typical Demo Usage Pattern

```python
# Inside a demo file:
KERNEL_SOURCE = r"""
extern "C" __global__
void relu_inplace(float* x, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(0.0f, x[i]);
}
"""

compiler = KernelCompiler(device)
relu_kernel = compiler.compile(KERNEL_SOURCE, "relu_inplace")

with DeviceBuffer(n * 4) as buf:
    # copy data to buf.ptr ...
    grid = CompiledKernel.compute_grid_1d(n, block_size=256)
    relu_kernel.launch(grid, (256, 1, 1), args=[buf.ptr, n], stream=stream)
    stream.sync()
    # copy result back ...
```

---

## Key Design Decisions

1. **Lazy NVRTC compilation (compile on first call, cache)**: Kernel compilation takes 100ms–2s. Caching by `cache_key` means repeated demo runs don't re-pay the compile cost. Tradeoff: slightly more complex KernelCompiler state; acceptable given the performance impact.

2. **`extern "C"` required on all kernels**: NVRTC C++ mode mangles function names. All kernel functions must use `extern "C"` to get a predictable symbol name that `get_kernel()` can look up. This is a non-obvious gotcha for users — document it in every example.

3. **Caller is responsible for stream sync**: `CompiledKernel.launch()` does NOT synchronize the stream. This is intentional — it allows callers to pipeline multiple kernel launches before synchronizing, which is the correct GPU performance pattern. Demos that need results immediately call `stream.sync()` after launch.

4. **Architecture flag derived from device at runtime**: The `-arch=sm_XX` flag is computed from `device.compute_capability` at compile time, not hardcoded. This makes demos portable across GPU generations without code changes.

5. **No shared kernel cache across demos**: Each demo instantiates its own `KernelCompiler`. A global cache would save compile time across demos in `run_all.py`, but would couple demos together. For an educational project, demo isolation wins.

---

## Consequences

### Positive
- Kernel source is readable Python strings — no separate `.cu` files, no Makefile
- NVRTC compile errors surface as Python exceptions with helpful CUDA error messages
- `compute_grid_1d` eliminates boilerplate for the most common 1D parallel pattern
- Caching means users can iterate on demo code without paying compile cost each run

### Negative / Trade-offs
- First-run compile latency (100ms–2s per kernel) is noticeable — users must be informed this is expected
- `args` must be ctypes-compatible; Python ints and floats need explicit casting (e.g., `ctypes.c_int(n)`)
- No support for dynamic parallelism or cooperative groups in this design

### Risks
- `cuda.core.experimental` module name suggests API instability — monitor cuda-python releases
- NVRTC does not support all CUDA C++ features (templates require specific flags) — kernel authors must be aware

---

## Testing Strategy

- `tests/test_kernels.py` — compile and launch each standard kernel (vector add, ReLU, softmax) and verify output against NumPy
- Test compile error handling: pass invalid CUDA C, confirm Python exception is raised with useful message
- Test grid computation: `compute_grid_1d` with various n and block_size values (unit test, no GPU needed)
- GPU tests tagged `@pytest.mark.gpu`

---

## Implementation Notes for Programmer

1. Implement `KernelCompiler` before any demo that needs a custom kernel
2. Use `cuda.core.experimental` (not deprecated `cuda.core`) — check installed cuda-python version first
3. The NVRTC option for SM architecture: use `f"-arch=sm_{major}{minor}"` from `device.compute_capability`
4. For argument passing: integer scalars need `ctypes.c_int()`, float scalars need `ctypes.c_float()`, device pointers pass as raw int (the `.ptr` attribute of DeviceBuffer)
5. NVRTC compilation error messages come from `CUDAError` exception — print the full error, it's human-readable

**Reference kernel pattern** (vector add):
```cuda
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

---

## ADRs

- ADR-003: Lazy kernel compilation with per-compiler cache (vs eager compile at module import)
- ADR-004: Caller-managed stream sync (vs auto-sync after each launch)
