# What Is CUDA Python?

CUDA Python is NVIDIA's official Python package that exposes the full CUDA runtime and driver API from pure Python. It ships two layers: `cuda.bindings` (thin wrappers around the C API) and `cuda.core.experimental` (a higher-level, Pythonic interface). Together they let you allocate GPU memory, compile CUDA C kernels at runtime via NVRTC, and launch those kernels — without writing a single line of C or setting up a C++ build system.

Unlike CuPy or Numba, which hide the kernel programming model behind array operators or decorators, CUDA Python exposes the raw programming model directly. You write the CUDA C kernel as a Python string, compile it, and control every aspect of the launch — grid dimensions, block size, stream assignment. This makes it the right tool when you need exact control: custom memory layouts, novel algorithms, or GPU programming education.

```python
from cuda.core.experimental import Device, Program, ProgramOptions

device = Device(0)
device.set_current()

src = r"""
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"""
opts = ProgramOptions(std="c++17", arch="sm_89")
prog = Program(src, code_type="c++", options=opts)
mod  = prog.compile("cubin")
kernel = mod.get_kernel("vector_add")
```

**Source:** `src/kernels/compiler.py:36–45`
