"""Memory management pattern demos for CUDA Python.

Five self-contained functions demonstrate:
  1. Basic device alloc / free and free-memory queries.
  2. Context manager guaranteeing cleanup even on exception.
  3. Pinned vs pageable H2D transfer speed (cross-reference demos/01_core_apis).
  4. OOM error catch and recovery.

Run with: python -m demos.07_memory.main
"""

from __future__ import annotations

import sys

import numpy as np


# ---------------------------------------------------------------------------
# Demo 1: Basic alloc / free + memory queries
# ---------------------------------------------------------------------------

def demo_basic_alloc() -> None:
    """Show basic device alloc/free and query free memory before/after."""
    try:
        from src.utils.device import get_device
        from src.utils.memory import DeviceBuffer, query_device_memory
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        sys.exit(1)

    try:
        device = get_device(0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    n_bytes = 256 * 1024 * 1024  # 256 MB

    mem_before = query_device_memory(device)
    print(f"  Before alloc : free={mem_before['free_gb']:.3f} GB / total={mem_before['total_gb']:.3f} GB")

    # Intentionally using explicit close() here (contrast with demo_context_manager which shows the context manager pattern)
    buf = DeviceBuffer(n_bytes, device=device)
    mem_during = query_device_memory(device)
    alloc_gb = n_bytes / (1024 ** 3)
    print(f"  After alloc  : free={mem_during['free_gb']:.3f} GB  (allocated {alloc_gb:.3f} GB)")

    buf.close()
    mem_after = query_device_memory(device)
    print(f"  After free   : free={mem_after['free_gb']:.3f} GB  (restored)")

    recovered = mem_after["free_gb"] - mem_during["free_gb"]
    print(f"  Memory recovered: {recovered:.3f} GB  (expected ~{alloc_gb:.3f} GB)")


# ---------------------------------------------------------------------------
# Demo 2: Context manager guarantees cleanup on exception
# ---------------------------------------------------------------------------

def demo_context_manager() -> None:
    """Show DeviceBuffer context manager guarantees cleanup even on exception."""
    try:
        from src.utils.device import get_device
        from src.utils.memory import DeviceBuffer, query_device_memory
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        sys.exit(1)

    try:
        device = get_device(0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    n_bytes = 128 * 1024 * 1024  # 128 MB

    mem_before = query_device_memory(device)
    print(f"  Before alloc : free={mem_before['free_gb']:.3f} GB")

    exception_caught = False
    try:
        with DeviceBuffer(n_bytes, device=device) as buf:
            mem_during = query_device_memory(device)
            print(f"  Inside with  : free={mem_during['free_gb']:.3f} GB  (buffer handle={buf.handle:#x})")
            # Simulate an error mid-flight — __exit__ still fires
            raise ValueError("Simulated error inside with-block")
    except ValueError as exc:
        exception_caught = True
        print(f"  Exception caught: {exc}")

    # Memory should be freed despite the exception
    mem_after = query_device_memory(device)
    print(f"  After with   : free={mem_after['free_gb']:.3f} GB  (buffer freed by __exit__)")
    print(f"  Exception caught: {exception_caught}  |  Memory restored: {mem_after['free_gb'] > mem_during['free_gb']}")


# ---------------------------------------------------------------------------
# Demo 3: Pinned vs pageable H2D transfer speed
# ---------------------------------------------------------------------------

def demo_pinned_transfer(n_mb: int = 64) -> None:
    """Benchmark pinned vs pageable H2D transfer speed.

    This reuses the same logic demonstrated in demos/01_core_apis/pinned_memory.py
    (see demo_pinned_vs_pageable).  Results are printed in GB/s for both modes.
    """
    # Cross-reference: full implementation lives in demos/01_core_apis/pinned_memory.py
    # Use importlib because the directory name starts with a digit
    import importlib

    try:
        pm = importlib.import_module("demos.01_core_apis.pinned_memory")
        demo_pinned_vs_pageable = pm.demo_pinned_vs_pageable
    except Exception as exc:
        print(f"  Could not import pinned_memory demo: {exc}")
        return

    print(f"  Benchmarking {n_mb} MB H2D transfer (pinned vs pageable)...")
    result = demo_pinned_vs_pageable(n_mb=n_mb)
    print(f"  Pinned   : {result['pinned_gb_s']:.2f} GB/s")
    print(f"  Pageable : {result['pageable_gb_s']:.2f} GB/s")
    print(f"  Speedup  : {result['speedup']:.2f}x")
    print("  (Full demo: python -m demos.01_core_apis.main)")


# ---------------------------------------------------------------------------
# Demo 4: OOM recovery
# ---------------------------------------------------------------------------

def demo_oom_recovery() -> None:
    """Show how to catch and recover from a CUDA OOM error.

    Attempts to allocate more memory than the device has, catches the error,
    and demonstrates that execution continues normally.
    """
    try:
        from src.utils.device import get_device
        from src.utils.memory import DeviceBuffer, query_device_memory
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        sys.exit(1)

    try:
        device = get_device(0)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    mem = query_device_memory(device)
    free_gb = mem["free_gb"]
    total_gb = mem["total_gb"]
    print(f"  Device memory : free={free_gb:.3f} GB / total={total_gb:.3f} GB")

    # Request more than the total device memory (guaranteed OOM)
    oom_bytes = int((total_gb + 1.0) * (1024 ** 3))
    oom_gb = oom_bytes / (1024 ** 3)
    print(f"  Requesting {oom_gb:.1f} GB (OOM expected) ...")

    recovered = False
    try:
        buf = DeviceBuffer(oom_bytes, device=device)
        # Should not reach here
        buf.close()
        print("  ERROR: OOM was NOT raised — unexpected success")
    except Exception as exc:
        recovered = True
        print(f"  OOM caught: {type(exc).__name__}: {exc}")

    # Verify execution continues and device memory is unchanged
    mem_after = query_device_memory(device)
    print(f"  After OOM    : free={mem_after['free_gb']:.3f} GB  (unchanged — no allocation occurred)")
    print(f"  Recovered cleanly: {recovered}")

    # Confirm we can still allocate small buffers after the OOM
    small_bytes = 1024 * 1024  # 1 MB
    try:
        with DeviceBuffer(small_bytes, device=device) as small_buf:
            print(f"  Small alloc after OOM: OK (handle={small_buf.handle:#x})")
    except Exception as exc:
        print(f"  Small alloc after OOM FAILED: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all memory management demos in sequence."""
    try:
        from src.utils.device import check_cuda_available
    except ImportError as exc:
        print(f"ERROR: src/utils not available: {exc}")
        sys.exit(1)

    if not check_cuda_available():
        print(
            "ERROR: No CUDA GPU detected or cuda-python not installed.\n"
            "Install CUDA Toolkit 12.x and run: pip install cuda-python"
        )
        sys.exit(1)

    print("=" * 60)
    print("Demo 07: Memory Management Patterns")
    print("=" * 60)

    print("\n[1] Basic alloc / free + memory queries")
    demo_basic_alloc()

    print("\n[2] Context manager cleanup on exception")
    demo_context_manager()

    print("\n[3] Pinned vs pageable transfer speed")
    demo_pinned_transfer(n_mb=64)

    print("\n[4] OOM error catch and recovery")
    demo_oom_recovery()

    print("\n" + "=" * 60)
    print("All memory demos complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
