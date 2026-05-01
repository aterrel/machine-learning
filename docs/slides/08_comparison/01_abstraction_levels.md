# Abstraction Levels in GPU Python

The five backends in `demos/08_comparison/` represent a ladder of abstraction. At the bottom, cuda-python writes CUDA C kernels and manages memory manually — maximum control, maximum code. Numba-CUDA writes kernels in Python syntax with `@cuda.jit` — same control, less friction. CuPy provides array operators that dispatch to cuBLAS/cuDNN — no kernels, but limited to existing operators. cuML mirrors the scikit-learn API — one line of code, no GPU knowledge required.

Choosing the right level is a tradeoff between control and convenience. Novel algorithms or custom memory layouts require cuda-python or Numba. Standard ML algorithms on regular arrays belong in CuPy or cuML. The comparison demo makes this concrete by running all four on the same problem and printing timing side-by-side.

```
Abstraction ladder (low → high):

cuda-python    Custom CUDA C kernels + explicit memory management
               ↑ most control, most code

numba-cuda     Python-syntax @cuda.jit kernels, to_device/copy_to_host
               ↑ CUDA semantics, Python syntax

cupy           Array operators dispatched to cuBLAS/cuDNN
               ↑ no kernel code; limited to built-in operations

cuml           scikit-learn-compatible API (fit/predict/transform)
               ↑ least code; fixed algorithms, no customization
```

**Source:** `demos/08_comparison/main.py`, `agents/architecture/ARCH-003.md`
