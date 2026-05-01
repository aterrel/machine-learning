"""Print CUDA device properties for device 0."""

from __future__ import annotations


def print_device_info() -> dict:
    """Get device 0 properties, print them, and return as dict."""
    try:
        from src.utils.device import get_device, get_device_props
    except ImportError:
        print("ERROR: src/utils not found. Run from project root.")
        return {}

    try:
        device = get_device(0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return {}

    try:
        props = get_device_props(device)
    except RuntimeError as exc:
        print(f"ERROR: Could not query device properties: {exc}")
        return {}

    major, minor = props["compute_capability"]
    print(f"Device name          : {props['name']}")
    print(f"Compute capability   : {major}.{minor}")
    print(f"Total VRAM           : {props['total_memory_gb']:.2f} GB")
    print(f"SM architecture      : sm_{props['arch']}")

    return props
