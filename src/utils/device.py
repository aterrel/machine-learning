"""Device management utilities for CUDA Python demos."""

from __future__ import annotations


def check_cuda_available() -> bool:
    """Return True if cuda.core is importable and at least one device exists."""
    try:
        from cuda.core.experimental import Device

        dev = Device(0)
        dev.set_current()
        return True
    except Exception:
        return False


def get_device(index: int = 0):
    """Return a cuda.core Device, set as current."""
    try:
        from cuda.core.experimental import Device
    except ImportError as exc:
        raise RuntimeError(
            "cuda-python is not installed. Install with: pip install cuda-python"
        ) from exc

    try:
        dev = Device(index)
        dev.set_current()
        return dev
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize CUDA device {index}. "
            "Ensure an NVIDIA GPU is present and CUDA 12.x is installed."
        ) from exc


def get_device_props(device) -> dict:
    """Return dict with name, compute_capability, total_memory_gb, arch."""
    try:
        cc = device.compute_capability
        major, minor = cc.major, cc.minor
        arch = f"{major}{minor}"
        props = device.properties
        name = props.name.decode() if isinstance(props.name, (bytes, bytearray)) else str(props.name)
        total_mem_bytes = props.totalGlobalMem
        total_mem_gb = total_mem_bytes / (1024**3)
        return {
            "name": name,
            "compute_capability": (major, minor),
            "total_memory_gb": total_mem_gb,
            "arch": arch,
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to query device properties: {exc}") from exc
