"""Tests for src/utils/device.py.

CPU-safe tests run without any GPU.
GPU tests are marked @pytest.mark.gpu and use the gpu_device fixture,
which auto-skips when no CUDA device is present.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.utils.device import check_cuda_available


# ---------------------------------------------------------------------------
# CPU-only tests — no GPU required
# ---------------------------------------------------------------------------


def test_check_cuda_available_returns_bool():
    """check_cuda_available() must always return a plain bool, never raise."""
    result = check_cuda_available()
    assert isinstance(result, bool), (
        f"check_cuda_available() returned {type(result).__name__}, expected bool"
    )


def test_get_device_raises_without_cuda():
    """get_device() must raise RuntimeError (not ImportError) when cuda-python is absent."""
    # Simulate cuda-python not being installed by making the import fail.
    with patch.dict("sys.modules", {"cuda": None, "cuda.core": None, "cuda.core.experimental": None}):
        from src.utils import device as device_mod

        with patch.object(device_mod, "get_device") as mock_get:
            mock_get.side_effect = RuntimeError(
                "cuda-python is not installed. Install with: pip install cuda-python"
            )
            with pytest.raises(RuntimeError, match="cuda-python"):
                device_mod.get_device(0)


# ---------------------------------------------------------------------------
# GPU tests — require NVIDIA GPU (auto-skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_get_device_success(gpu_device):
    """get_device() returns a non-None device object when a GPU is present."""
    assert gpu_device is not None


@pytest.mark.gpu
def test_get_device_props_shape(gpu_device):
    """get_device_props() returns a dict with the four required keys."""
    from src.utils.device import get_device_props

    props = get_device_props(gpu_device)
    required_keys = {"name", "compute_capability", "total_memory_gb", "arch"}
    missing = required_keys - props.keys()
    assert not missing, f"get_device_props() is missing keys: {missing}"


@pytest.mark.gpu
def test_device_props_values_sensible(gpu_device):
    """Device props must have sensible values: memory > 0.5 GB, compute_capability is 2-tuple."""
    from src.utils.device import get_device_props

    props = get_device_props(gpu_device)

    assert props["total_memory_gb"] > 0.5, (
        f"total_memory_gb={props['total_memory_gb']:.3f} is unexpectedly small"
    )

    cc = props["compute_capability"]
    assert isinstance(cc, tuple) and len(cc) == 2, (
        f"compute_capability should be a 2-tuple, got {cc!r}"
    )
    major, minor = cc
    assert isinstance(major, int) and isinstance(minor, int), (
        f"compute_capability elements should be ints, got ({type(major)}, {type(minor)})"
    )
    assert major >= 3, f"Unexpectedly old compute capability: {major}.{minor}"
