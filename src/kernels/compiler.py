"""NVRTC kernel compiler wrapping cuda.core.experimental.Program."""

from __future__ import annotations

from .compiled_kernel import CompiledKernel


class KernelCompiler:
    """Compiles CUDA C source strings to launchable kernels via NVRTC."""

    def __init__(self, device):
        self._device = device
        self._cache: dict[str, CompiledKernel] = {}

    def compile(
        self,
        source: str,
        kernel_name: str,
        cache_key: str | None = None,
        stream=None,
    ) -> CompiledKernel:
        """Compile CUDA C source and return a launchable CompiledKernel."""
        key = cache_key or f"{kernel_name}:{hash(source)}"
        if key in self._cache:
            return self._cache[key]

        try:
            from cuda.core.experimental import Program, ProgramOptions
        except ImportError as exc:
            raise RuntimeError("cuda-python >= 12.3 is required.") from exc

        cc = self._device.compute_capability
        arch = f"sm_{cc.major}{cc.minor}"
        opts = ProgramOptions(std="c++17", arch=arch)
        prog = Program(source, code_type="c++", options=opts)
        mod = prog.compile("cubin")
        kernel = mod.get_kernel(kernel_name)

        compiled = CompiledKernel(kernel, stream)
        if cache_key is not None:
            # cache_key=None: kernel is compiled fresh each call and not stored in cache
            self._cache[key] = compiled
        return compiled
