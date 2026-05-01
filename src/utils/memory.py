"""Device memory management utilities for CUDA Python demos."""

from __future__ import annotations

import ctypes

import numpy as np


class DeviceBuffer:
    """Context manager for device memory allocation and deallocation."""

    def __init__(self, n_bytes: int, stream=None, device=None):
        self._n_bytes = n_bytes
        self._stream = stream
        self._device = device
        self._buf = None

        if device is not None:
            device.set_current()
        try:
            from cuda.core.experimental import Device

            if device is None:
                dev = Device(0)
                dev.set_current()
                self._device = dev
            self._buf = self._device.allocate(n_bytes, stream=stream)
        except ImportError as exc:
            raise RuntimeError("cuda-python is not installed.") from exc

    def __enter__(self) -> "DeviceBuffer":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if self._buf is not None:
            self._buf.close(self._stream)
            self._buf = None

    @property
    def handle(self) -> int:
        """Raw device pointer as integer."""
        return int(self._buf.handle)

    @property
    def n_bytes(self) -> int:
        return self._n_bytes


def alloc_pinned(n_bytes: int) -> tuple[int, np.ndarray]:
    """Allocate n_bytes of pinned (page-locked) host memory for float32 data.

    Returns (ptr_int, np_array_view).
    """
    try:
        from cuda.bindings import cudart
    except ImportError as exc:
        raise RuntimeError("cuda-python is not installed.") from exc

    err, ptr = cudart.cudaHostAlloc(n_bytes, cudart.cudaHostAllocDefault)
    if err.value != 0:
        raise RuntimeError(f"cudaHostAlloc failed with error code {err.value}")
    arr = np.frombuffer(ctypes.string_at(ptr, n_bytes), dtype=np.float32)
    return int(ptr), arr


def free_pinned(ptr: int) -> None:
    """Free pinned host memory previously allocated with alloc_pinned."""
    try:
        from cuda.bindings import cudart
    except ImportError as exc:
        raise RuntimeError("cuda-python is not installed.") from exc

    (err,) = cudart.cudaFreeHost(ptr)
    if err.value != 0:
        raise RuntimeError(f"cudaFreeHost failed with error code {err.value}")


def query_device_memory(device) -> dict:
    """Return dict with free_gb and total_gb for the given device."""
    try:
        from cuda.bindings import cudart
    except ImportError as exc:
        raise RuntimeError("cuda-python is not installed.") from exc

    err, free_bytes, total_bytes = cudart.cudaMemGetInfo()
    if err.value != 0:
        raise RuntimeError(f"cudaMemGetInfo failed with error code {err.value}")
    return {
        "free_gb": free_bytes / (1024**3),
        "total_gb": total_bytes / (1024**3),
    }
