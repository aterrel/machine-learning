"""Tests for src/utils/memory.py.

CPU-only tests cover ImportError handling for alloc_pinned and free_pinned.
GPU tests cover DeviceBuffer context manager and query_device_memory.
All GPU tests use the gpu_device fixture and are marked @pytest.mark.gpu.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from src.utils import memory as memory_mod


# ---------------------------------------------------------------------------
# CPU-only tests — no GPU required
# ---------------------------------------------------------------------------


def test_alloc_pinned_import_error():
    """alloc_pinned raises RuntimeError when cuda.bindings is not installed."""
    # Patch cuda.bindings in sys.modules to simulate it being absent,
    # then call the function directly (it does a local import inside the body).
    saved = sys.modules.get("cuda.bindings.cudart")
    sys.modules["cuda.bindings.cudart"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError):
            memory_mod.alloc_pinned(1024)
    finally:
        if saved is None:
            sys.modules.pop("cuda.bindings.cudart", None)
        else:
            sys.modules["cuda.bindings.cudart"] = saved


def test_free_pinned_import_error():
    """free_pinned raises RuntimeError when cuda.bindings is not installed."""
    saved = sys.modules.get("cuda.bindings.cudart")
    sys.modules["cuda.bindings.cudart"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError):
            memory_mod.free_pinned(0)
    finally:
        if saved is None:
            sys.modules.pop("cuda.bindings.cudart", None)
        else:
            sys.modules["cuda.bindings.cudart"] = saved


# ---------------------------------------------------------------------------
# GPU tests — require NVIDIA GPU (auto-skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_device_buffer_context_manager(gpu_device):  # noqa: ARG001
    """DeviceBuffer context manager allocates memory and frees it on exit."""
    from src.utils.memory import DeviceBuffer

    with DeviceBuffer(1024, device=gpu_device) as buf:
        assert buf.handle > 0, f"Expected device pointer > 0, got {buf.handle}"
        assert buf.n_bytes == 1024

    # After __exit__ the internal buffer should be freed (buf._buf is None)
    assert buf._buf is None, "Expected buffer to be freed after context manager exit"


@pytest.mark.gpu
def test_query_device_memory_keys(gpu_device):
    """query_device_memory returns dict with 'free_gb' and 'total_gb', both > 0."""
    from src.utils.memory import query_device_memory

    result = query_device_memory(gpu_device)

    assert "free_gb" in result, f"Expected 'free_gb' key, got keys: {list(result.keys())}"
    assert "total_gb" in result, f"Expected 'total_gb' key, got keys: {list(result.keys())}"
    assert result["free_gb"] > 0, f"Expected free_gb > 0, got {result['free_gb']}"
    assert result["total_gb"] > 0, f"Expected total_gb > 0, got {result['total_gb']}"
