"""NVIDIA Jetson (Tegra SoC) edge device profiles.

Grounded in the edge-deployment report (edge-video-dit-deployment-2026.md,
section C, hardware table lines 169-171):

  * Jetson AGX Thor / T5000: 2070 TFLOPS FP4-sparse, 128GB LPDDR5X unified,
                             273 GB/s, 40-130W, FP4/FP8/INT8 (Blackwell 5th-gen TC).
  * Jetson Thor T4000:       1200 TFLOPS FP4-sparse, 64GB LPDDR5X, 273 GB/s,
                             40-70W, FP4/FP8/INT8.
  * Jetson AGX Orin 64GB:    275 TOPS INT8-sparse, 64GB LPDDR5, 204.8 GB/s,
                             15-60W, INT8/FP16 (Ampere).

These are unified-memory edge modules. The report flags that Jetson Thor video-
DiT latency is NOT publicly benchmarked; expectation is multi-minute, bandwidth-
limited 480p short clips (273 GB/s is ~1/4 of a 4090). All three are categorized
EDGE_NPU per the VDG taxonomy (edge accelerators).

Dense TFLOPS derivation (peak, tensor cores; sparse AI-TOPS halved to a dense-
equivalent since DiT matmuls are dense):

  Thor T5000  FP4-sparse 2070 -> dense 1035; Blackwell 2x-step:
              FP8 517.5 | BF16/FP16 258.75 | INT8 517.5.
              FP32 (CUDA) = 2560 cores x 1.57 GHz x 2 / 1e12 = 8.0.
  Thor T4000  FP4-sparse 1200 -> dense 600; FP8 300 | BF16/FP16 150 | INT8 300.
              FP32 (CUDA) = 1536 cores x 1.53 GHz x 2 / 1e12 = 4.7.
  Orin 64     INT8-sparse 275 -> dense 137.5; Ampere: FP16/BF16 = INT8/2 = 68.75.
              FP32 (CUDA) = 2048 cores x 1.3 GHz x 2 / 1e12 = 5.3.

TDP is stored as the module's max sustained power (Thor 130W, T4000 70W, Orin
60W) so worst-case energy planning stays conservative; idle is ~0.1x.

Jetson modules are Tegra SoCs and are usually NOT visible to desktop NVML, so
'is_available' uses detector.try_jetson (checks /proc/device-tree/model and
nv_tegra_release) rather than NVML. 'measure_power' attempts NVML first (some
Jetson setups expose it) and falls back to None.
"""
from __future__ import annotations

from ..core.contracts import DeviceCategory, DeviceProfile, DeviceSpec
from ..core.registry import register_device
from .detector import nvml_power_for, try_jetson, try_nvml


@register_device
class Jetson_Thor_T5000(DeviceProfile):
    """NVIDIA Jetson AGX Thor / T5000. Blackwell edge; native FP4/INT8 via TensorRT."""

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="Jetson AGX Thor T5000",
            category=DeviceCategory.EDGE_NPU,
            memory_gb=128.0,
            memory_bandwidth_gbps=273.0,
            compute_tflops={
                "fp32": 8.0,
                "bf16": 258.75,
                "fp16": 258.75,
                "fp8": 517.5,
                "nvfp4": 1035.0,
                "fp4": 1035.0,
                "int8": 517.5,
            },
            tdp_w=130.0,
            idle_power_w=10.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8", "nvfp4", "fp4", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "sage3", "triton", "math"],
            unified_memory=True,
        )

    def is_available(self) -> bool:
        return try_jetson()

    def measure_power(self) -> float | None:
        # Some Jetson setups expose power via NVML; otherwise unreadable here.
        if try_nvml():
            return nvml_power_for("")
        return None


@register_device
class Jetson_Thor_T4000(DeviceProfile):
    """NVIDIA Jetson Thor T4000. Lower-tier Blackwell edge (1536 cores)."""

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="Jetson Thor T4000",
            category=DeviceCategory.EDGE_NPU,
            memory_gb=64.0,
            memory_bandwidth_gbps=273.0,
            compute_tflops={
                "fp32": 4.7,
                "bf16": 150.0,
                "fp16": 150.0,
                "fp8": 300.0,
                "nvfp4": 600.0,
                "fp4": 600.0,
                "int8": 300.0,
            },
            tdp_w=70.0,
            idle_power_w=8.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8", "nvfp4", "fp4", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "sage3", "triton", "math"],
            unified_memory=True,
        )

    def is_available(self) -> bool:
        return try_jetson()

    def measure_power(self) -> float | None:
        if try_nvml():
            return nvml_power_for("")
        return None


@register_device
class Jetson_Orin_64(DeviceProfile):
    """NVIDIA Jetson AGX Orin 64GB. Ampere edge; INT8-primary (no FP4)."""

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="Jetson AGX Orin 64GB",
            category=DeviceCategory.EDGE_NPU,
            memory_gb=64.0,
            memory_bandwidth_gbps=204.8,
            compute_tflops={
                "fp32": 5.3,
                "bf16": 68.75,
                "fp16": 68.75,
                "int8": 137.5,
            },
            tdp_w=60.0,
            idle_power_w=6.0,
            supported_precisions=["fp32", "bf16", "fp16", "int8"],
            attention_backends=["flash", "sdpa", "sage1", "math"],
            unified_memory=True,
        )

    def is_available(self) -> bool:
        return try_jetson()

    def measure_power(self) -> float | None:
        if try_nvml():
            return nvml_power_for("")
        return None
