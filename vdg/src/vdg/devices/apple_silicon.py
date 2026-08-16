"""Apple Silicon device profiles.

Grounded in the edge-deployment report (edge-video-dit-deployment-2026.md,
section A, lines 12, 42-45):

  * M4 Max  : 64GB unified, 546 GB/s, ~480W system. BF16 + FP8 via MLX; NO FP4
              hardware (Apple GPUs lack Blackwell-style FP4 tensor cores).
  * M3 Ultra: 512GB unified, 819 GB/s, ~480W system. Largest unified pool.
  * M2 Ultra: 192GB unified, 800 GB/s, ~370W system.

The report explicitly states Apple does NOT publish a TFLOPS figure
('M3 Ultra 819 GB/s (bandwidth-dominated, no public TFLOPS scale)'), and
bandwidth is the real bottleneck for attention (memory-bound). The compute
numbers below are therefore conservative estimates (FP8 via MLX software/Metal,
not a hardware tensor peak) and are marked as such; they follow the same
2x-per-precision-step convention used by the foundation's M4 Max fixture
(fp32 -> bf16/fp16 -> fp8 each 2x). Apple Silicon ships NO FP4 path -- fp4 is
deliberately absent from compute_tflops and supported_precisions.

All profiles are unified-memory devices. 'is_available' checks both MPS
availability and a chip-name match (so an M4 Max profile only reports available
on an actual M4 Max host, not on any Mac). 'measure_power' returns None: the
report gives only system-level wattage and live per-GPU power is not readable
without root 'powermetrics' polling, which we do not attempt here.
"""
from __future__ import annotations

from ..core.contracts import DeviceCategory, DeviceProfile, DeviceSpec
from ..core.registry import register_device
from .detector import apple_chip_name, try_mps


def _apple_available(chip_token: str) -> bool:
    """True if MPS is available AND the host chip name matches 'chip_token'."""
    if not try_mps():
        return False
    chip = apple_chip_name()
    if not chip:
        return False
    return chip_token.lower() in chip.lower()


@register_device
class M4_Max(DeviceProfile):
    """Apple M4 Max. 64GB unified; MLX runs Wan2.1 1.3B ~90 s/it, 14B ~230 s/it."""

    _CHIP_TOKEN = "m4 max"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="M4 Max",
            category=DeviceCategory.APPLE_SILICON,
            memory_gb=64.0,
            memory_bandwidth_gbps=546.0,
            # Estimates (Apple publishes no TFLOPS); FP8 via MLX software/Metal.
            compute_tflops={
                "fp32": 27.0,
                "bf16": 54.0,
                "fp16": 54.0,
                "fp8": 108.0,
            },
            tdp_w=480.0,
            idle_power_w=10.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8"],
            attention_backends=["mlx_sdpa", "math"],
            unified_memory=True,
        )

    def is_available(self) -> bool:
        return _apple_available(self._CHIP_TOKEN)

    def measure_power(self) -> float | None:
        return None


@register_device
class M3_Ultra(DeviceProfile):
    """Apple M3 Ultra. 512GB unified, 819 GB/s -- largest unified-memory pool.

    Unified memory lets a 14B model reside entirely on-GPU (~36GB), but 819 GB/s
    is far below a discrete GPU, so attention stays memory-bound.
    """

    _CHIP_TOKEN = "m3 ultra"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="M3 Ultra",
            category=DeviceCategory.APPLE_SILICON,
            memory_gb=512.0,
            memory_bandwidth_gbps=819.0,
            # Estimates (Apple publishes no TFLOPS); ~2x M4 Max GPU-core count.
            compute_tflops={
                "fp32": 54.0,
                "bf16": 108.0,
                "fp16": 108.0,
                "fp8": 216.0,
            },
            tdp_w=480.0,
            idle_power_w=15.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8"],
            attention_backends=["mlx_sdpa", "math"],
            unified_memory=True,
        )

    def is_available(self) -> bool:
        return _apple_available(self._CHIP_TOKEN)

    def measure_power(self) -> float | None:
        return None


@register_device
class M2_Ultra(DeviceProfile):
    """Apple M2 Ultra. 192GB unified, 800 GB/s, ~370W system."""

    _CHIP_TOKEN = "m2 ultra"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="M2 Ultra",
            category=DeviceCategory.APPLE_SILICON,
            memory_gb=192.0,
            memory_bandwidth_gbps=800.0,
            # Estimates (Apple publishes no TFLOPS).
            compute_tflops={
                "fp32": 27.0,
                "bf16": 54.0,
                "fp16": 54.0,
                "fp8": 108.0,
            },
            tdp_w=370.0,
            idle_power_w=15.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8"],
            attention_backends=["mlx_sdpa", "math"],
            unified_memory=True,
        )

    def is_available(self) -> bool:
        return _apple_available(self._CHIP_TOKEN)

    def measure_power(self) -> float | None:
        return None
