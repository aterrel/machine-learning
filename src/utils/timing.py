"""Benchmarking utilities: BenchmarkResult dataclass and BenchmarkRunner."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    demo_name: str
    cpu_time_mean_s: float
    gpu_time_mean_s: float
    speedup: float
    correct: bool
    max_abs_error: float

    def summary_line(self) -> str:
        return (
            f"GPU: {self.gpu_time_mean_s:.3f}s | CPU: {self.cpu_time_mean_s:.3f}s"
            f" | Speedup: {self.speedup:.1f}x | Correct: {self.correct}"
        )


class BenchmarkRunner:
    def __init__(self, n_repeats: int = 5, warmup: int = 1):
        self._n_repeats = n_repeats
        self._warmup = warmup

    def time_cpu(self, fn, *args, **kwargs) -> float:
        """Run fn(*args, **kwargs) n_repeats times and return mean wall-clock seconds."""
        for _ in range(self._warmup):
            fn(*args, **kwargs)
        times = []
        for _ in range(self._n_repeats):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)
        return sum(times) / len(times)

    def time_gpu(self, fn, stream, *args, **kwargs) -> float:
        """Run fn(*args, **kwargs) n_repeats times and return mean time in seconds.

        Uses CUDA events for accurate GPU timing.
        """
        try:
            from cuda.bindings import cudart
        except ImportError as exc:
            raise RuntimeError("cuda-python is not installed.") from exc

        for _ in range(self._warmup):
            fn(*args, **kwargs)
            stream.sync()

        elapsed_ms_total = 0.0
        for _ in range(self._n_repeats):
            err, start = cudart.cudaEventCreate()
            if err.value != 0:
                raise RuntimeError(f"cudaEventCreate failed: {err.value}")
            err, stop = cudart.cudaEventCreate()
            if err.value != 0:
                raise RuntimeError(f"cudaEventCreate failed: {err.value}")

            cudart.cudaEventRecord(start, stream.handle)
            fn(*args, **kwargs)
            cudart.cudaEventRecord(stop, stream.handle)
            cudart.cudaEventSynchronize(stop)

            err, elapsed_ms = cudart.cudaEventElapsedTime(start, stop)
            if err.value != 0:
                raise RuntimeError(f"cudaEventElapsedTime failed: {err.value}")
            elapsed_ms_total += elapsed_ms

            cudart.cudaEventDestroy(start)
            cudart.cudaEventDestroy(stop)

        return (elapsed_ms_total / self._n_repeats) / 1000.0
