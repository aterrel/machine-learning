"""Pinned vs pageable memory transfer benchmark and data loading demos."""

from __future__ import annotations

import sys
import time
import urllib.request

import numpy as np


def demo_pinned_vs_pageable(n_mb: int = 128) -> dict:
    """Compare H2D transfer speed: pinned vs pageable memory.

    Returns {"pinned_gb_s": float, "pageable_gb_s": float, "speedup": float}.
    """
    try:
        from cuda.bindings import cudart

        from src.utils.device import get_device
        from src.utils.memory import DeviceBuffer, alloc_pinned, free_pinned
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        sys.exit(1)

    try:
        device = get_device(0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    n_bytes = n_mb * 1024 * 1024
    n_floats = n_bytes // 4
    stream = device.create_stream()

    # --- Pinned memory benchmark ---
    ptr, pinned_arr = alloc_pinned(n_bytes)
    rng = np.random.default_rng(0)
    pinned_arr[:] = rng.random(n_floats, dtype=np.float32)

    N_REPS = 5
    pinned_times = []
    with DeviceBuffer(n_bytes, stream=stream, device=device) as d_buf:
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            (err,) = cudart.cudaMemcpy(
                d_buf.handle,
                ptr,
                n_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            stream.sync()
            pinned_times.append(time.perf_counter() - t0)
            if err.value != 0:
                raise RuntimeError(f"Pinned H2D copy failed: {err.value}")

    free_pinned(ptr)
    pinned_mean_s = sum(pinned_times) / len(pinned_times)
    pinned_gb_s = (n_bytes / (1024**3)) / pinned_mean_s

    # --- Pageable memory benchmark ---
    pageable_arr = rng.random(n_floats, dtype=np.float32)
    pageable_times = []
    with DeviceBuffer(n_bytes, stream=stream, device=device) as d_buf:
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            (err,) = cudart.cudaMemcpy(
                d_buf.handle,
                pageable_arr.ctypes.data,
                n_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            stream.sync()
            pageable_times.append(time.perf_counter() - t0)
            if err.value != 0:
                raise RuntimeError(f"Pageable H2D copy failed: {err.value}")

    stream.sync()
    stream.close()

    pageable_mean_s = sum(pageable_times) / len(pageable_times)
    pageable_gb_s = (n_bytes / (1024**3)) / pageable_mean_s
    speedup = pinned_gb_s / pageable_gb_s if pageable_gb_s > 0 else float("inf")

    return {"pinned_gb_s": pinned_gb_s, "pageable_gb_s": pageable_gb_s, "speedup": speedup}


def load_from_disk_to_device(path: str) -> int:
    """Load .npy file into pinned buffer, transfer to device; return device pointer."""
    try:
        from cuda.bindings import cudart

        from src.utils.device import get_device
        from src.utils.memory import DeviceBuffer, alloc_pinned, free_pinned
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency: {exc}") from exc

    device = get_device(0)
    stream = device.create_stream()

    arr = np.load(path).astype(np.float32).ravel()
    n_bytes = arr.nbytes

    ptr, pinned_arr = alloc_pinned(n_bytes)
    pinned_arr[:] = arr

    buf = device.allocate(n_bytes, stream=stream)
    (err,) = cudart.cudaMemcpy(
        int(buf.handle),
        ptr,
        n_bytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    stream.sync()
    stream.close()
    free_pinned(ptr)

    if err.value != 0:
        raise RuntimeError(f"H2D copy failed: {err.value}")

    return int(buf.handle)


def load_from_url_to_device(url: str, fallback_shape: tuple) -> int:
    """Fetch .npy data from URL or fall back to random array; return device pointer."""
    try:
        from cuda.bindings import cudart

        from src.utils.device import get_device
        from src.utils.memory import alloc_pinned, free_pinned
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency: {exc}") from exc

    arr: np.ndarray | None = None
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = resp.read()
        import io

        arr = np.load(io.BytesIO(data)).astype(np.float32).ravel()
        print(f"Loaded {arr.nbytes / 1024:.1f} KB from {url}")
    except Exception as exc:
        print(f"WARNING: Could not fetch from URL ({exc}). Falling back to random array.")
        rng = np.random.default_rng(0)
        arr = rng.random(fallback_shape, dtype=np.float32).ravel()

    device = get_device(0)
    stream = device.create_stream()

    n_bytes = arr.nbytes
    ptr, pinned_arr = alloc_pinned(n_bytes)
    pinned_arr[:] = arr

    buf = device.allocate(n_bytes, stream=stream)
    (err,) = cudart.cudaMemcpy(
        int(buf.handle),
        ptr,
        n_bytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    stream.sync()
    stream.close()
    free_pinned(ptr)

    if err.value != 0:
        raise RuntimeError(f"H2D copy failed: {err.value}")

    return int(buf.handle)
