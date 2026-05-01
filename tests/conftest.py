"""pytest configuration: GPU marker and device fixture."""

from __future__ import annotations

import pytest

from src.utils.device import check_cuda_available


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires NVIDIA GPU")


@pytest.fixture(scope="session")
def gpu_device():
    """Return CUDA device 0, or skip the test if no GPU is available."""
    if not check_cuda_available():
        pytest.skip("No CUDA GPU available")
    from src.utils.device import get_device

    return get_device(0)
