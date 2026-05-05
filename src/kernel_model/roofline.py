from __future__ import annotations

from dataclasses import dataclass

from .device_spec import DeviceSpec


@dataclass
class RooflineResult:
    arithmetic_intensity: float
    ridge_point: float
    bound: str
    predicted_gflops: float


class RooflineModel:
    def __init__(self, device: DeviceSpec) -> None:
        self._device = device

    def compute(self, flops: float, bytes_accessed: float) -> RooflineResult:
        dev = self._device
        arithmetic_intensity = flops / bytes_accessed
        ridge_point = dev.peak_fp32_gflops / dev.memory_bandwidth_gbs

        if arithmetic_intensity < ridge_point:
            bound = "memory"
            predicted_gflops = arithmetic_intensity * dev.memory_bandwidth_gbs
        else:
            bound = "compute"
            predicted_gflops = dev.peak_fp32_gflops

        return RooflineResult(
            arithmetic_intensity=arithmetic_intensity,
            ridge_point=ridge_point,
            bound=bound,
            predicted_gflops=predicted_gflops,
        )

    def sweep_intensities(self, intensity_range: list[float]) -> list[RooflineResult]:
        return [self.compute(flops=intensity * 1.0, bytes_accessed=1.0) for intensity in intensity_range]
