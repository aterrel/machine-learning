"""Entry point for demos/06_interop — runs all interop sub-demos."""

from __future__ import annotations

import sys


def main() -> None:
    """Run all interop demos and report which libraries were available."""
    from src.utils.device import check_cuda_available

    if not check_cuda_available():
        print(
            "ERROR: No CUDA GPU detected or cuda-python not installed.\n"
            "Install CUDA Toolkit 12.x and run: pip install cuda-python"
        )
        sys.exit(1)

    # Detect optional library availability once at the top level for reporting
    cupy_ok = False
    torch_ok = False
    try:
        import cupy  # noqa: F401
        cupy_ok = True
    except ImportError:
        pass
    try:
        import torch  # noqa: F401
        torch_ok = True
    except ImportError:
        pass

    print("=" * 60)
    print("Demo 06: NumPy / CuPy / PyTorch Interop")
    print("=" * 60)
    print(f"  CuPy available  : {cupy_ok}")
    print(f"  PyTorch available: {torch_ok}")
    print()

    from demos.06_interop import cupy_interop, pipeline, torch_interop

    print("[CuPy Interop]")
    cupy_interop.demo_cuda_python_to_cupy()
    print()
    cupy_interop.demo_cupy_to_cuda_python()
    print()

    print("[PyTorch Interop]")
    torch_interop.demo_cuda_python_to_torch()
    print()
    torch_interop.demo_torch_to_cuda_python()
    print()

    print("[End-to-End Pipeline]")
    results = pipeline.run_end_to_end_pipeline()
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  CuPy available  : {results['cupy_available']}")
    print(f"  PyTorch available: {results['torch_available']}")
    print(f"  ReLU correct    : {results['relu_correct']}")
    if results["cupy_norm"] is not None:
        print(f"  CuPy L2 norm    : {results['cupy_norm']:.4f}")
    if results["torch_sum"] is not None:
        print(f"  PyTorch sum     : {results['torch_sum']:.4f}")


if __name__ == "__main__":
    main()
