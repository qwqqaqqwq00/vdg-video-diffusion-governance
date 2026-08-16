"""Consumer / workstation NVIDIA GPU device profiles.

Grounded in the edge-deployment report (edge-video-dit-deployment-2026.md,
section B, hardware table lines 108-114):

  * RTX 4090    (Ada Lovelace): 24GB GDDR6X, 1008 GB/s, 450W, FP8 e4m3, no FP4.
  * RTX 5090    (Blackwell):    32GB GDDR7, ~1.79 TB/s (512-bit / 28 Gbps), 575W,
                                3352 AI TOPS FP4-sparse, native NVFP4.
  * RTX 6000 Ada(Ada Lovelace): 48GB GDDR6, 960 GB/s, 300W, FP8 e4m3, no FP4.

Dense TFLOPS derivation (peak, tensor cores):

  RTX 4090   FP32 = 82.6 (16384 CUDA x 2.52 GHz x 2 / 1e12, the report's anchor).
            Tensor (4th-gen, Ada 2x-per-precision-step from FP16):
            BF16/FP16 330 | FP8 660 | INT8 660.
  RTX 6000 Ada FP32 = 87.5 (18176 CUDA x 2.405 GHz x 2 / 1e12, same method).
            Scaled by the Ada tensor ratio: BF16/FP16 350 | FP8 700 | INT8 700.
  RTX 5090  FP32 = 104.8 (21760 CUDA x 2.41 GHz x 2 / 1e12, per the report).
            FP4-sparse 3352 -> dense-equivalent 1676 (matmuls are dense, so the
            2:4 sparse rating is halved; documented per the contracts guidance).
            Blackwell 2x-per-step: FP8 838 | BF16/FP16 419 | INT8 838.

Each profile probes the host via NVML and reads live power via detector. Both
degrade to False/None on a host without NVIDIA hardware -- never raise.
"""
from __future__ import annotations

from ..core.contracts import DeviceCategory, DeviceProfile, DeviceSpec
from ..core.registry import register_device
from .detector import nvml_has_name, nvml_power_for


@register_device
class RTX4090(DeviceProfile):
    """NVIDIA RTX 4090 (Ada Lovelace). 24GB; the report's 480p baseline GPU.

    Wan2.1-1.3B ~4min/RTX4090; LightX2V HunyuanVideo-1.5 fits 24GB.
    """

    _NVML_KEY = "4090"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="RTX 4090",
            category=DeviceCategory.CONSUMER_NV,
            memory_gb=24.0,
            memory_bandwidth_gbps=1008.0,
            compute_tflops={
                "fp32": 82.6,
                "bf16": 330.0,
                "fp16": 330.0,
                "fp8": 660.0,
                "int8": 660.0,
            },
            tdp_w=450.0,
            idle_power_w=45.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "math"],
            unified_memory=False,
            cost_per_hour_usd=0.5,
        )

    def is_available(self) -> bool:
        return nvml_has_name(self._NVML_KEY)

    def measure_power(self) -> float | None:
        return nvml_power_for(self._NVML_KEY)


@register_device
class RTX5090(DeviceProfile):
    """NVIDIA RTX 5090 (Blackwell consumer). 32GB; native NVFP4 tensor cores.

    Differentiator vs Ada: FP4/NVFP4. LightX2V Wan2.2-NVFP4-Sparse >50x speedup
    on a single 5090; SageAttention3 (microscaling FP4) reaches ~560T here.
    """

    _NVML_KEY = "5090"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="RTX 5090",
            category=DeviceCategory.CONSUMER_NV,
            memory_gb=32.0,
            memory_bandwidth_gbps=1790.0,
            compute_tflops={
                "fp32": 104.8,
                "bf16": 419.0,
                "fp16": 419.0,
                "fp8": 838.0,
                "nvfp4": 1676.0,
                "fp4": 1676.0,
                "int8": 838.0,
            },
            tdp_w=575.0,
            idle_power_w=55.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8", "nvfp4", "fp4", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "sage3", "triton", "math"],
            unified_memory=False,
            cost_per_hour_usd=0.7,
        )

    def is_available(self) -> bool:
        return nvml_has_name(self._NVML_KEY)

    def measure_power(self) -> float | None:
        return nvml_power_for(self._NVML_KEY)


@register_device
class RTX6000_Ada(DeviceProfile):
    """NVIDIA RTX 6000 Ada (Ada Lovelace workstation). 48GB; larger batch / 1080p."""

    _NVML_KEY = "6000"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="RTX 6000 Ada",
            category=DeviceCategory.CONSUMER_NV,
            memory_gb=48.0,
            memory_bandwidth_gbps=960.0,
            compute_tflops={
                "fp32": 87.5,
                "bf16": 350.0,
                "fp16": 350.0,
                "fp8": 700.0,
                "int8": 700.0,
            },
            tdp_w=300.0,
            idle_power_w=30.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "math"],
            unified_memory=False,
            cost_per_hour_usd=1.0,
        )

    def is_available(self) -> bool:
        return nvml_has_name(self._NVML_KEY)

    def measure_power(self) -> float | None:
        return nvml_power_for(self._NVML_KEY)
