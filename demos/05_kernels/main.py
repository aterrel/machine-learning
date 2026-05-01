"""Custom CUDA kernel demos: naive GEMM, ReLU, and softmax."""

from __future__ import annotations

import importlib
import sys


def main() -> None:
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

    kernels_mod = importlib.import_module("demos.05_kernels.activations")
    gemm_mod = importlib.import_module("demos.05_kernels.gemm")
    run_activations_demo = kernels_mod.run_activations_demo
    run_gemm_demo = gemm_mod.run_gemm_demo

    print("=" * 60)
    print("Custom CUDA Kernel Demos")
    print("=" * 60)

    print("\n[Naive GEMM — 512x512x512]")
    gemm_result = run_gemm_demo(M=512, N=512, K=512)
    print(f"  Max abs error : {gemm_result.max_abs_error:.2e}")
    print(f"  Correct       : {gemm_result.correct}")
    print(f"  {gemm_result.summary_line()}")

    print("\n[ReLU + Softmax — n=1_000_000]")
    relu_result = run_activations_demo(n=1_000_000)
    print(f"  ReLU max error: {relu_result.max_abs_error:.2e}")
    print(f"  ReLU correct  : {relu_result.correct}")
    print(f"  {relu_result.summary_line()}")

    print("\n" + "=" * 60)
    print("All kernel demos complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
