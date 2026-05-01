"""Entry point for demo 01: Core CUDA Python APIs."""

from __future__ import annotations

import sys
import tempfile

import numpy as np


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

    print("\n--- Load from Disk to Device (AC-8) ---")
    from .pinned_memory import load_from_disk_to_device

    rng = np.random.default_rng(42)
    synthetic = rng.random(1024, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name
    np.save(tmp_path, synthetic)
    dev_ptr, dev_buf = load_from_disk_to_device(tmp_path)
    print(f"Loaded {synthetic.nbytes} bytes from disk to device pointer 0x{dev_ptr:x}")
    dev_buf.close()

    print("\n--- Load from URL to Device (AC-9) ---")
    from .pinned_memory import load_from_url_to_device

    dev_ptr2, dev_buf2 = load_from_url_to_device(
        "https://example.invalid/data.npy", fallback_shape=(1024,)
    )
    print(f"Data on device at pointer 0x{dev_ptr2:x} ({1024 * 4} bytes)")
    dev_buf2.close()

    print("\nDemo 01 complete.")


if __name__ == "__main__":
    main()
