"""Non-NVIDIA NPU / accelerator device profiles.

Grounded in the edge-deployment report (edge-video-dit-deployment-2026.md,
section C, lines 14, 162-176, 195). These are the "industrial-edge / NPU" class
the report groups together (server NPUs like Ascend/MLU alongside edge NPUs like
RK3588/Hexagon). All are categorized EDGE_NPU per the VDG taxonomy (non-GPU
accelerators), matching the report's own 'C class NPU' grouping.

  * Ascend 910B    (Huawei, server): ~310W, INT8/FP16. Report: '数百 TFLOPS FP16'
                    (hundreds). LightX2V-adapted for Wan/HunyuanVideo/LTX.
  * Cambricon MLU590 (server):       vendor-public approximate specs; LightX2V
                    + ComfyUI adapted.
  * RK3588 (Rockchip, edge):         <=32GB LPDDR4x, ~50 GB/s, 5-15W, INT8/INT4,
                    6 TOPS INT8 NPU (3 cores). Report: low-end, video DiT NOT
                    feasible without extreme distillation+quantization.
  * Qualcomm Hexagon (mobile):       ~45 TOPS INT8, INT8/INT4. Snapdragon class.

Dense TFLOPS notes:
  * Ascend 910B: FP16 anchored at 320 (representative of '数百/hundreds'); INT8
    = 2x FP16 = 640; BF16 ~= FP16; FP32 is a conservative vector-unit estimate.
    All marked approximate (vendor does not publish a clean dense-tensor sheet).
  * MLU590: vendor-public approximate (INT8 ~512 TOPS, FP16 ~128 TFLOPS).
  * RK3588: the 6 TOPS INT8 rating is the Rockchip sparse figure -> dense 3.0
    TFLOPS; INT4 = 2x INT8 = 6.0.
  * Hexagon: ~45 TOPS INT8 taken at face value as the rated mobile peak; INT4 90.

'is_available': Ascend probes via detector.try_ascend (torch_npu or driver
paths). MLU590 / RK3588 / Hexagon have no reliable cross-host detector, so they
return False gracefully (they are target deployment devices, not dev-host
hardware). 'measure_power' returns None for all NPU profiles (no portable
power read path without vendor SDKs).
"""
from __future__ import annotations

from ..core.contracts import DeviceCategory, DeviceProfile, DeviceSpec
from ..core.registry import register_device
from .detector import try_ascend


@register_device
class Ascend_910B(DeviceProfile):
    """Huawei Ascend 910B (server NPU). LightX2V-adapted for video DiT."""

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="Ascend 910B",
            category=DeviceCategory.EDGE_NPU,
            memory_gb=64.0,
            memory_bandwidth_gbps=1200.0,
            # Estimates: FP16 anchored to the report's '数百 TFLOPS' (320 used);
            # INT8 = 2x FP16. Vendor does not publish a clean dense sheet.
            compute_tflops={
                "fp32": 40.0,
                "bf16": 320.0,
                "fp16": 320.0,
                "int8": 640.0,
            },
            tdp_w=310.0,
            idle_power_w=30.0,
            supported_precisions=["fp32", "bf16", "fp16", "int8"],
            attention_backends=["vendor_attn", "math"],
            unified_memory=False,
        )

    def is_available(self) -> bool:
        return try_ascend()

    def measure_power(self) -> float | None:
        return None


@register_device
class Cambricon_MLU590(DeviceProfile):
    """Cambricon MLU590 (server NPU). LightX2V + ComfyUI adapted for video DiT."""

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="Cambricon MLU590",
            category=DeviceCategory.EDGE_NPU,
            memory_gb=48.0,
            memory_bandwidth_gbps=1024.0,
            # Vendor-public approximate specs (INT8 ~512 TOPS, FP16 ~128 TFLOPS).
            compute_tflops={
                "fp32": 32.0,
                "bf16": 128.0,
                "fp16": 128.0,
                "int8": 512.0,
            },
            tdp_w=300.0,
            idle_power_w=30.0,
            supported_precisions=["fp32", "bf16", "fp16", "int8"],
            attention_backends=["vendor_attn", "math"],
            unified_memory=False,
        )

    def is_available(self) -> bool:
        # No portable detector for Cambricon on a generic host.
        return False

    def measure_power(self) -> float | None:
        return None


@register_device
class RK3588(DeviceProfile):
    """Rockchip RK3588 (low-end edge NPU). Video DiT infeasible per the report.

    6 TOPS INT8 NPU (3 cores); the report states low-end NPUs like this require
    extreme 4-step distillation + INT8/INT4 + low resolution to run at all.
    """

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="Rockchip RK3588",
            category=DeviceCategory.EDGE_NPU,
            memory_gb=32.0,
            memory_bandwidth_gbps=50.0,
            # 6 TOPS INT8 is Rockchip's sparse rating -> dense 3.0; INT4 = 6.0.
            compute_tflops={
                "int8": 3.0,
                "int4": 6.0,
            },
            tdp_w=15.0,
            idle_power_w=1.0,
            supported_precisions=["int8", "int4"],
            attention_backends=["math"],
            unified_memory=True,
        )

    def is_available(self) -> bool:
        return False

    def measure_power(self) -> float | None:
        return None


@register_device
class Qualcomm_Hexagon(DeviceProfile):
    """Qualcomm Hexagon NPU (Snapdragon mobile class). ~45 TOPS INT8.

    Mobile NPU; video DiT would need extreme distillation + INT8/INT4 + low res.
    """

    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="Qualcomm Hexagon",
            category=DeviceCategory.EDGE_NPU,
            memory_gb=16.0,
            memory_bandwidth_gbps=51.2,
            # ~45 TOPS INT8 (rated mobile peak); INT4 = 2x INT8.
            compute_tflops={
                "int8": 45.0,
                "int4": 90.0,
            },
            tdp_w=8.0,
            idle_power_w=0.5,
            supported_precisions=["int8", "int4"],
            attention_backends=["math"],
            unified_memory=True,
        )

    def is_available(self) -> bool:
        return False

    def measure_power(self) -> float | None:
        return None
