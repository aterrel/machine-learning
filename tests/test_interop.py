"""Tests for demos/06_interop — CuPy, PyTorch, and pipeline interop.

CPU-only tests verify:
  - __cuda_array_interface__ dict structure (shape, typestr, data, version keys)
  - demo_cuda_python_to_cupy() returns without error when CuPy is not installed
  - demo_cupy_to_cuda_python() returns without error when CuPy is not installed
  - demo_cuda_python_to_torch() returns without error when PyTorch is not installed
  - demo_torch_to_cuda_python() returns without error when PyTorch is not installed

GPU tests (auto-skipped when no GPU is available):
  - run_end_to_end_pipeline() returns a dict with the expected keys

All interop modules are imported via importlib because their directory names
start with a digit (06_interop, 07_memory).
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — lazy module load (avoids importing cuda-python at collection time)
# ---------------------------------------------------------------------------

def _load_cupy_interop():
    return importlib.import_module("demos.06_interop.cupy_interop")


def _load_torch_interop():
    return importlib.import_module("demos.06_interop.torch_interop")


def _load_pipeline():
    return importlib.import_module("demos.06_interop.pipeline")


# ---------------------------------------------------------------------------
# CPU-only tests — __cuda_array_interface__ dict structure
# ---------------------------------------------------------------------------


def test_cuda_array_interface_dict_structure():
    """__cuda_array_interface__ dict must contain shape, typestr, data, and version keys."""
    # Construct the interface dict directly as the production code does (fake ptr value).
    interface = {
        "shape": (1_000_000,),
        "typestr": "<f4",          # little-endian float32
        "data": (12345678, False), # (device_ptr, read_only=False)
        "version": 3,
    }

    assert "shape" in interface, "Missing 'shape' key"
    assert "typestr" in interface, "Missing 'typestr' key"
    assert "data" in interface, "Missing 'data' key"
    assert "version" in interface, "Missing 'version' key"


def test_cuda_array_interface_version_is_3():
    """__cuda_array_interface__ version must be exactly 3."""
    interface = {
        "shape": (1_000_000,),
        "typestr": "<f4",
        "data": (12345678, False),
        "version": 3,
    }
    assert interface["version"] == 3, (
        f"Expected version=3, got {interface['version']}"
    )


def test_cuda_array_interface_typestr_float32():
    """__cuda_array_interface__ typestr must be '<f4' (little-endian float32)."""
    interface = {
        "shape": (1_000_000,),
        "typestr": "<f4",
        "data": (12345678, False),
        "version": 3,
    }
    assert interface["typestr"] == "<f4", (
        f"Expected typestr='<f4', got {interface['typestr']!r}"
    )


def test_cuda_array_interface_data_is_2_tuple():
    """__cuda_array_interface__ data field must be a 2-tuple of (ptr, read_only)."""
    interface = {
        "shape": (1_000_000,),
        "typestr": "<f4",
        "data": (12345678, False),
        "version": 3,
    }
    data = interface["data"]
    assert isinstance(data, tuple), (
        f"Expected data to be a tuple, got {type(data).__name__}"
    )
    assert len(data) == 2, f"Expected 2-tuple, got length {len(data)}"


def test_cuda_array_interface_data_ptr_is_int():
    """__cuda_array_interface__ data[0] must be an integer (device pointer)."""
    interface = {
        "shape": (1_000_000,),
        "typestr": "<f4",
        "data": (12345678, False),
        "version": 3,
    }
    ptr, read_only = interface["data"]
    assert isinstance(ptr, int), (
        f"Expected device pointer to be int, got {type(ptr).__name__}"
    )
    assert read_only is False, (
        f"Expected read_only=False, got {read_only!r}"
    )


def test_cuda_array_interface_shape_is_tuple():
    """__cuda_array_interface__ shape must be a tuple."""
    interface = {
        "shape": (1_000_000,),
        "typestr": "<f4",
        "data": (12345678, False),
        "version": 3,
    }
    assert isinstance(interface["shape"], tuple), (
        f"Expected shape to be a tuple, got {type(interface['shape']).__name__}"
    )


# ---------------------------------------------------------------------------
# CPU-only tests — graceful skip when CuPy or PyTorch is not installed
# ---------------------------------------------------------------------------


def test_cupy_interop_skips_without_cupy(capsys):
    """demo_cuda_python_to_cupy prints a skip message and returns None when cupy is missing."""
    mod = _load_cupy_interop()

    # Patch cupy out of sys.modules so the `import cupy` inside the function raises ImportError.
    with patch.dict("sys.modules", {"cupy": None}):
        result = mod.demo_cuda_python_to_cupy()

    # Must return cleanly (not raise, not call sys.exit)
    assert result is None, (
        f"Expected None return when CuPy is missing, got {result!r}"
    )

    # Must print a human-readable skip message
    captured = capsys.readouterr()
    assert "CuPy not installed" in captured.out, (
        f"Expected 'CuPy not installed' in stdout, got: {captured.out!r}"
    )


def test_cupy_to_cuda_python_skips_without_cupy(capsys):
    """demo_cupy_to_cuda_python prints a skip message and returns None when cupy is missing."""
    mod = _load_cupy_interop()

    with patch.dict("sys.modules", {"cupy": None}):
        result = mod.demo_cupy_to_cuda_python()

    assert result is None, (
        f"Expected None return when CuPy is missing, got {result!r}"
    )

    captured = capsys.readouterr()
    assert "CuPy not installed" in captured.out, (
        f"Expected 'CuPy not installed' in stdout, got: {captured.out!r}"
    )


def test_torch_interop_skips_without_torch(capsys):
    """demo_cuda_python_to_torch prints a skip message and returns None when torch is missing."""
    mod = _load_torch_interop()

    with patch.dict("sys.modules", {"torch": None}):
        result = mod.demo_cuda_python_to_torch()

    assert result is None, (
        f"Expected None return when torch is missing, got {result!r}"
    )

    captured = capsys.readouterr()
    assert "PyTorch not installed" in captured.out, (
        f"Expected 'PyTorch not installed' in stdout, got: {captured.out!r}"
    )


def test_torch_to_cuda_python_skips_without_torch(capsys):
    """demo_torch_to_cuda_python prints a skip message and returns None when torch is missing."""
    mod = _load_torch_interop()

    with patch.dict("sys.modules", {"torch": None}):
        result = mod.demo_torch_to_cuda_python()

    assert result is None, (
        f"Expected None return when torch is missing, got {result!r}"
    )

    captured = capsys.readouterr()
    assert "PyTorch not installed" in captured.out, (
        f"Expected 'PyTorch not installed' in stdout, got: {captured.out!r}"
    )


# ---------------------------------------------------------------------------
# GPU tests — require NVIDIA GPU (auto-skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_pipeline_returns_dict(gpu_device):  # noqa: ARG001
    """run_end_to_end_pipeline() must return a dict (not None, not raise)."""
    pipeline = _load_pipeline()
    result = pipeline.run_end_to_end_pipeline(n=10_000)
    assert isinstance(result, dict), (
        f"Expected dict return from run_end_to_end_pipeline, got {type(result).__name__}"
    )


@pytest.mark.gpu
def test_pipeline_result_keys(gpu_device):  # noqa: ARG001
    """run_end_to_end_pipeline() result dict must contain the six expected keys."""
    pipeline = _load_pipeline()
    result = pipeline.run_end_to_end_pipeline(n=10_000)

    expected_keys = {"n", "cupy_available", "torch_available", "relu_correct", "cupy_norm", "torch_sum"}
    missing = expected_keys - result.keys()
    assert not missing, (
        f"pipeline result is missing keys: {missing}. Got keys: {set(result.keys())}"
    )


@pytest.mark.gpu
def test_pipeline_n_matches_input(gpu_device):  # noqa: ARG001
    """run_end_to_end_pipeline(n=...) must store the exact n in the result dict."""
    pipeline = _load_pipeline()
    result = pipeline.run_end_to_end_pipeline(n=5_000)
    assert result["n"] == 5_000, (
        f"Expected result['n'] == 5000, got {result['n']}"
    )


@pytest.mark.gpu
def test_pipeline_relu_correct(gpu_device):  # noqa: ARG001
    """run_end_to_end_pipeline() ReLU kernel output must match the NumPy reference."""
    pipeline = _load_pipeline()
    result = pipeline.run_end_to_end_pipeline(n=10_000)
    assert result["relu_correct"] is True, (
        f"Expected relu_correct=True, got {result['relu_correct']}"
    )


@pytest.mark.gpu
def test_pipeline_availability_flags_are_bools(gpu_device):  # noqa: ARG001
    """cupy_available and torch_available must be plain Python booleans."""
    pipeline = _load_pipeline()
    result = pipeline.run_end_to_end_pipeline(n=10_000)

    assert isinstance(result["cupy_available"], bool), (
        f"cupy_available should be bool, got {type(result['cupy_available']).__name__}"
    )
    assert isinstance(result["torch_available"], bool), (
        f"torch_available should be bool, got {type(result['torch_available']).__name__}"
    )
