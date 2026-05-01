# Kernel Compilation with NVRTC

CUDA kernels are written as CUDA C strings and compiled at runtime by NVRTC (NVIDIA Runtime Compilation). `Program(source, code_type="c++", options=opts)` feeds the source to NVRTC, and `prog.compile("cubin")` produces a device binary. `mod.get_kernel(name)` extracts the `__global__` function by name. The `KernelCompiler` wrapper adds a cache so the same kernel string is only compiled once per process.

The `arch` option in `ProgramOptions` must match the device's compute capability. The compiler queries the device's `compute_capability` attribute and assembles `sm_{major}{minor}` automatically, so the same Python code works on any supported GPU.

```python
from cuda.core.experimental import Program, ProgramOptions

cc   = device.compute_capability
arch = f"sm_{cc.major}{cc.minor}"
opts = ProgramOptions(std="c++17", arch=arch)

prog   = Program(source, code_type="c++", options=opts)
mod    = prog.compile("cubin")
kernel = mod.get_kernel(kernel_name)
```

**Source:** `src/kernels/compiler.py:40–45`
