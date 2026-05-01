"""Entry point for demo 01: Core CUDA Python APIs."""

from __future__ import annotations

import sys


def main() -> None:
    from src.utils.device import check_cuda_available

    if not check_cuda_available():
        print(
            "ERROR: No CUDA GPU detected or cuda-python not installed.\n"
            "Install CUDA Toolkit 12.x and run: pip install cuda-python"
        )
        sys.exit(1)

    print("=" * 60)
    print("Demo 01: Core CUDA Python APIs")
    print("=" * 60)

    print("\n--- Device Info ---")
    from .device_info import print_device_info

    print_device_info()

    print("\n--- Vector Add Kernel ---")
    from .vector_add import run_vector_add

    result = run_vector_add(n=1_000_000)
    print(result.summary_line())
    print(f"Max absolute error   : {result.max_abs_error:.2e}")

    print("\n--- Pinned vs Pageable Memory Transfer ---")
    from .pinned_memory import demo_pinned_vs_pageable

    bw = demo_pinned_vs_pageable(n_mb=128)
    print(f"Pinned   : {bw['pinned_gb_s']:.2f} GB/s")
    print(f"Pageable : {bw['pageable_gb_s']:.2f} GB/s")
    print(f"Speedup  : {bw['speedup']:.2f}x")

    print("\nDemo 01 complete.")


if __name__ == "__main__":
    main()
