"""CompiledKernel: typed launch helper wrapping a cuda.core kernel object."""

from __future__ import annotations


class CompiledKernel:
    """Wraps a compiled CUDA kernel with a typed launch interface."""

    def __init__(self, kernel, stream=None):
        self._kernel = kernel
        self._stream = stream

    def launch(
        self,
        grid: tuple,
        block: tuple,
        args: list,
        stream=None,
    ) -> None:
        """Launch the kernel on the given grid/block configuration.

        Caller is responsible for calling stream.sync() before reading results.
        """
        try:
            from cuda.core.experimental import LaunchConfig, launch
        except ImportError as exc:
            raise RuntimeError("cuda-python >= 12.3 is required.") from exc

        s = stream if stream is not None else self._stream
        cfg = LaunchConfig(grid=grid, block=block)
        launch(s, cfg, self._kernel, *args)

    @staticmethod
    def compute_grid_1d(n: int, block_size: int = 256) -> tuple:
        """Compute 1D grid dimensions covering n elements."""
        return ((n + block_size - 1) // block_size, 1, 1)
