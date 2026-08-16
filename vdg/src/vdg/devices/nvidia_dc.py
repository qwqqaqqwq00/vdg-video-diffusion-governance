"""Datacenter NVIDIA GPU device profiles.

Grounded in the training-side report (video-gen-training-nvidia-2025-2026.md,
hardware table lines 53-57) and canonical public NVIDIA dense-tensor specs:

  * H100 SXM (Hopper):   80GB HBM3, 3.35 TB/s, 700W, 4th-gen TC, FP8.
  * H200 SXM (Hopper):  141GB HBM3e, 4.8 TB/s, 700W, 4th-gen TC, FP8.
                         (Same Hopper compute die as H100; larger/faster HBM3e.)
  * B200 SXM (Blackwell):192GB HBM3e, 8 TB/s, ~1000W, 5th-gen TC, MXFP8/MXFP6/NVFP4.
  * GB300 NVL72 (Blackwell Ultra): per-GPU 192GB, ~1200W, NVFP4; official
    '1.5x dense FP4 FLOPS' vs Blackwell.

Dense TFLOPS (peak, tensor cores -- the report references these as the public
spec convention; sparse AI-TOPS are halved to a dense-equivalent where used):

  H100/H200  FP32 67 | TF32/BF16/FP16 989 | FP8 1979 | INT8 3958   (no FP4)
  B200       FP32 80 | TF32 1125 | BF16/FP16 2250 | FP8 4500 | FP4 9000 | INT8 4500
  GB300      = B200, with FP4/NVFP4 = 9000 * 1.5 = 13500 (confirmed Ultra uplift);
              FP8/BF16 anchored to B200 dense values (no confirmed Ultra number).

All four profiles probe the host via NVML (detector.try_nvml / nvml_has_name)
and read live power via detector.nvml_power_for. Both degrade to False/None on
any host without NVIDIA hardware -- they never raise.
"""
from __future__ import annotations

from ..core.contracts import DeviceCategory, DeviceProfile, DeviceSpec
from ..core.registry import register_device
from .detector import nvml_has_name, nvml_power_for


@register_device
class H100(DeviceProfile):
    """NVIDIA H100 SXM (Hopper). Primary training + high-end inference GPU."""

    # NVML markets this as e.g. 'NVIDIA H100 80GB HBM3'.
    _NVML_KEY = "h100"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="H100 SXM",
            category=DeviceCategory.DATACENTER,
            memory_gb=80.0,
            memory_bandwidth_gbps=3350.0,
            compute_tflops={
                "fp32": 67.0,
                "tf32": 989.0,
                "bf16": 989.0,
                "fp16": 989.0,
                "fp8": 1979.0,
                "int8": 3958.0,
            },
            tdp_w=700.0,
            idle_power_w=70.0,
            supported_precisions=["fp32", "tf32", "bf16", "fp16", "fp8", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "triton", "math"],
            unified_memory=False,
            cost_per_hour_usd=2.5,
        )

    def is_available(self) -> bool:
        return nvml_has_name(self._NVML_KEY)

    def measure_power(self) -> float | None:
        return nvml_power_for(self._NVML_KEY)


@register_device
class H200(DeviceProfile):
    """NVIDIA H200 SXM (Hopper, 141GB HBM3e). Same compute as H100, more memory.

    Open-Sora 2.0 trained on 192-224x H200 141GB (arXiv 2503.09642).
    """

    _NVML_KEY = "h200"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="H200 SXM",
            category=DeviceCategory.DATACENTER,
            memory_gb=141.0,
            memory_bandwidth_gbps=4800.0,
            # Hopper compute die -- identical dense tensor throughput to H100.
            compute_tflops={
                "fp32": 67.0,
                "tf32": 989.0,
                "bf16": 989.0,
                "fp16": 989.0,
                "fp8": 1979.0,
                "int8": 3958.0,
            },
            tdp_w=700.0,
            idle_power_w=70.0,
            supported_precisions=["fp32", "tf32", "bf16", "fp16", "fp8", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "triton", "math"],
            unified_memory=False,
            cost_per_hour_usd=3.0,
        )

    def is_available(self) -> bool:
        return nvml_has_name(self._NVML_KEY)

    def measure_power(self) -> float | None:
        return nvml_power_for(self._NVML_KEY)


@register_device
class B200(DeviceProfile):
    """NVIDIA B200 SXM (Blackwell). 192GB HBM3e, native NVFP4/MXFP8 tensor cores."""

    _NVML_KEY = "b200"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="B200 SXM",
            category=DeviceCategory.DATACENTER,
            memory_gb=192.0,
            memory_bandwidth_gbps=8000.0,
            compute_tflops={
                "fp32": 80.0,
                "tf32": 1125.0,
                "bf16": 2250.0,
                "fp16": 2250.0,
                "fp8": 4500.0,
                "nvfp4": 9000.0,
                "fp4": 9000.0,
                "int8": 4500.0,
            },
            tdp_w=1000.0,
            idle_power_w=100.0,
            supported_precisions=["fp32", "tf32", "bf16", "fp16", "fp8", "nvfp4", "fp4", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "sage3", "triton", "math"],
            unified_memory=False,
            cost_per_hour_usd=4.0,
        )

    def is_available(self) -> bool:
        return nvml_has_name(self._NVML_KEY)

    def measure_power(self) -> float | None:
        return nvml_power_for(self._NVML_KEY)


@register_device
class GB300_NVL72(DeviceProfile):
    """NVIDIA GB300 NVL72 (Blackwell Ultra), per-GPU profile.

    Per-GPU: 192GB HBM3e, ~1200W. Officially '1.5x dense FP4 FLOPS' vs
    Blackwell, so FP4/NVFP4 dense = 9000 * 1.5 = 13500 TFLOPS. FP8/BF16 are
    anchored to the B200 dense values (no confirmed Ultra multiplier published);
    the report notes GB300 also delivers '2x attention performance', which the
    FP4 uplift partially captures. Bandwidth is the documented 8 TB/s GB200
    floor (GB300 is higher but unconfirmed in public specs).
    """

    # NVML may report 'NVIDIA GB300' or 'NVIDIA B300'; 'b300' matches both.
    _NVML_KEY = "b300"

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="GB300 NVL72",
            category=DeviceCategory.DATACENTER,
            memory_gb=192.0,
            memory_bandwidth_gbps=8000.0,
            compute_tflops={
                "fp32": 80.0,
                "tf32": 1125.0,
                "bf16": 2250.0,
                "fp16": 2250.0,
                "fp8": 4500.0,
                "nvfp4": 13500.0,
                "fp4": 13500.0,
                "int8": 4500.0,
            },
            tdp_w=1200.0,
            idle_power_w=120.0,
            supported_precisions=["fp32", "tf32", "bf16", "fp16", "fp8", "nvfp4", "fp4", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "sage3", "triton", "math"],
            unified_memory=False,
            cost_per_hour_usd=6.0,
        )

    def is_available(self) -> bool:
        return nvml_has_name(self._NVML_KEY)

    def measure_power(self) -> float | None:
        return nvml_power_for(self._NVML_KEY)
