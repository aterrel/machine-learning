"""Tests for src/kernels/ and the vector-add demo.

CPU-safe tests cover CompiledKernel.compute_grid_1d() and BenchmarkResult.summary_line().
GPU tests cover end-to-end vector-add kernel correctness.
"""

from __future__ import annotations

import pytest

from src.kernels.compiled_kernel import CompiledKernel
from src.utils.timing import BenchmarkResult


# ---------------------------------------------------------------------------
# CPU-only tests — no GPU required
# ---------------------------------------------------------------------------


def test_compute_grid_1d_exact_multiple():
    """Grid covers exactly n elements when n is an exact multiple of block_size."""
    n, block_size = 256, 256
    grid = CompiledKernel.compute_grid_1d(n, block_size)
    assert grid == (1, 1, 1)
    assert grid[0] * block_size >= n


def test_compute_grid_1d_non_multiple():
    """Grid must cover all n elements even when n is not a multiple of block_size."""
    n, block_size = 257, 256
    grid = CompiledKernel.compute_grid_1d(n, block_size)
    assert grid == (2, 1, 1)
    assert grid[0] * block_size >= n


def test_compute_grid_1d_large_n():
    """Grid covers all elements for a large n."""
    n, block_size = 1_000_001, 256
    grid = CompiledKernel.compute_grid_1d(n, block_size)
    assert grid[0] * block_size >= n
    # Verify it's the tightest possible grid (no unnecessary extra blocks)
    assert (grid[0] - 1) * block_size < n


def test_benchmark_result_summary_line():
    """BenchmarkResult.summary_line() must contain the four expected substrings."""
    result = BenchmarkResult(
        demo_name="test_op",
        cpu_time_mean_s=1.0,
        gpu_time_mean_s=0.25,
        speedup=4.0,
        correct=True,
        max_abs_error=0.0,
    )
    line = result.summary_line()
    assert "GPU:" in line, f"summary_line() missing 'GPU:': {line!r}"
    assert "CPU:" in line, f"summary_line() missing 'CPU:': {line!r}"
    assert "Speedup:" in line, f"summary_line() missing 'Speedup:': {line!r}"
    assert "Correct:" in line, f"summary_line() missing 'Correct:': {line!r}"


# ---------------------------------------------------------------------------
# GPU tests — require NVIDIA GPU (auto-skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_vector_add_correctness(gpu_device):  # noqa: ARG001  (fixture ensures GPU is ready)
    """run_vector_add produces a correct result with max_abs_error < 1e-5."""
    import importlib

    vector_add_mod = importlib.import_module("demos.01_core_apis.vector_add")
    run_vector_add = vector_add_mod.run_vector_add

    result = run_vector_add(n=10_000)
    assert result.correct is True, (
        f"vector_add reported incorrect result; max_abs_error={result.max_abs_error}"
    )
    assert result.max_abs_error < 1e-5, (
        f"max_abs_error={result.max_abs_error} exceeds threshold 1e-5"
    )
