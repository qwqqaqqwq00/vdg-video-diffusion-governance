"""Quantization acceleration skill (post-training weight / activation quant).

Quantization serves two benefits: fitting a large DiT into smaller memory
(fit) and accelerating via low-precision tensor-core matmuls. Three methods
are modeled, each targeting a different device family.

Grounding (video-dit-inference-acceleration-report.md, Section 5 / 8.6):
  * GGUF Q4_K_M (ComfyUI-GGUF, city96): HunyuanVideo-I2V 25.6 GB -> 7.88 GB;
    Wan2.1-T2V-14B 29.1 GB -> 10.1 GB. Primary benefit is fit, not speed
    (README reports no speedup). Works on Apple Silicon (needs torch 2.4.1 on
    macOS Sequoia) and consumer NVIDIA; not reported on edge NPU.
  * NVFP4 (SVDQuant / 6Bit-Diffusion, Blackwell): SVDQuant 3.1x on RTX 5090;
    6Bit-Diffusion NVFP4/INT8 dynamic 1.92x/3.32x. Needs Blackwell FP4 tensor
    cores. 4-bit residual preserves quality.
  * INT8 (edge NPU): HiF8 on Ascend (Wan2.1) keeps all 5 VBench dims >= BF16;
    TensorRT INT8 on Jetson gives 9x latency but rejects transformer layers
    (only 0.9% volume gain) -> modest effective speedup.

Applicability design: the base applicable(device, load) signature carries no
config, so it cannot see a method override. The three methods target different
device families (gguf_q4: apple+nv; nvfp4: nv+fp4; int8: edge_npu), so
applicable checks the union (is there ANY method that applies to this
device?). predict then guards the specific configured method: if it cannot
run on the device, predict returns a neutral no-op impact (speedup 1.0) with a
note, so an impossible gain is never credited.

VDG model (config key 'method' in {gguf_q4, nvfp4, int8}):
  * gguf_q4:  speedup 1.1, memory_ratio 0.35, quality_delta -1.0,
              energy_ratio 1.0, applies_to [apple_silicon, consumer_nv].
              (memory_ratio 0.35 grounded in Wan14B 10.1/29.1 = 0.347.)
  * nvfp4:    speedup 3.0, memory_ratio 0.5,  quality_delta -2.0,
              energy_ratio 0.7, applies_to [consumer_nv] (requires device fp4).
  * int8:     speedup 2.0, memory_ratio 0.5,  quality_delta -0.5,
              energy_ratio 0.85, applies_to [edge_npu].
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceCategory, DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["Quantization"]

_FP4 = "fp4"

# Per-method impact and applicability.
_METHODS: dict[str, dict[str, Any]] = {
    "gguf_q4": {
        "speedup": 1.1,
        "memory_ratio": 0.35,
        "quality_delta": -1.0,
        "energy_ratio": 1.0,
        "applies_to": [DeviceCategory.APPLE_SILICON, DeviceCategory.CONSUMER_NV],
        "runtime": "comfyui",
        "needs": "Apple Silicon or consumer NVIDIA",
        "notes": "GGUF Q4_K_M (ComfyUI-GGUF city96); Wan14B 29.1->10.1 GB "
                 "(0.347), HunyuanI2V 25.6->7.88 GB; fit not speed; MPS needs "
                 "torch 2.4.1.",
    },
    "nvfp4": {
        "speedup": 3.0,
        "memory_ratio": 0.5,
        "quality_delta": -2.0,
        "energy_ratio": 0.7,
        "applies_to": [DeviceCategory.CONSUMER_NV],
        "runtime": "tensorrt",
        "needs": "consumer NVIDIA with Blackwell FP4 tensor cores",
        "notes": "NVFP4 (SVDQuant 3.1x on 5090; 6Bit-Diffusion). Blackwell FP4 "
                 "tensor cores required; 4-bit residual preserves quality.",
    },
    "int8": {
        "speedup": 2.0,
        "memory_ratio": 0.5,
        "quality_delta": -0.5,
        "energy_ratio": 0.85,
        "applies_to": [DeviceCategory.EDGE_NPU],
        "runtime": "tensorrt",
        "needs": "edge NPU (Ascend HiF8 / Jetson TensorRT INT8)",
        "notes": "INT8 on edge NPU (HiF8 Ascend/Wan2.1: 5 VBench dims >= BF16; "
                 "Jetson TRT INT8 9x but rejects transformer layers -> modest "
                 "effective gain).",
    },
}


def _method_applicable(device: DeviceProfile, method: str) -> bool:
    """Whether a specific quantization method can run on the device."""
    spec = device.spec()
    m = _METHODS[method]
    if spec.category not in m["applies_to"]:
        return False
    if method == "nvfp4" and not spec.supports(_FP4):
        return False  # Blackwell FP4 tensor cores required.
    return True


@register_skill("quantization")
class Quantization(Skill):
    """Weight / activation quantization (GGUF Q4 / NVFP4 / INT8)."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Union gate: is there ANY method that applies to this device? The
        # method-specific guard is enforced in predict() (see docstring).
        return any(_method_applicable(device, m) for m in _METHODS)

    def default_config(self) -> dict[str, Any]:
        return {"method": "gguf_q4"}

    def _method(self, config: dict[str, Any]) -> str:
        method = str(config.get("method", "gguf_q4")).lower()
        return method if method in _METHODS else "gguf_q4"

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        method = self._method(cfg)
        # Guard the configured method against the device. If it cannot run,
        # return a neutral no-op so an impossible gain is never credited.
        if not _method_applicable(device, method):
            m = _METHODS[method]
            alt = "gguf_q4" if method == "nvfp4" else (
                "nvfp4 on Blackwell" if method == "gguf_q4" else "gguf_q4"
            )
            return SkillImpact(
                speedup=1.0,
                memory_ratio=1.0,
                quality_delta=0.0,
                energy_ratio=1.0,
                applies_to=[],
                notes="Quantization " + method + " needs " + str(m["needs"])
                      + "; " + device.spec().name + " lacks it -> no effect. "
                      + "Consider " + alt + ".",
            )
        m = _METHODS[method]
        return SkillImpact(
            speedup=float(m["speedup"]),
            memory_ratio=float(m["memory_ratio"]),
            quality_delta=float(m["quality_delta"]),
            energy_ratio=float(m["energy_ratio"]),
            applies_to=list(m["applies_to"]),
            notes=str(m["notes"]),
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        method = self._method(cfg)
        m = _METHODS[method]
        runtime_cfg: dict[str, Any] = {"method": method, "enable": True}
        if method == "gguf_q4":
            runtime_cfg["file_type"] = "Q4_K_M"
            runtime_cfg["loader"] = "ComfyUI-GGUF Unet Loader (GGUF)"
        elif method == "nvfp4":
            runtime_cfg["format"] = "nvfp4"
            runtime_cfg["residual_bits"] = 4
        elif method == "int8":
            runtime_cfg["format"] = "int8"
            runtime_cfg["calibrate"] = True
        applied = False
        hook = getattr(model_or_pipeline, "enable_quantization", None)
        if callable(hook):
            try:
                hook(method=method)
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="quantization",
            runtime=str(m["runtime"]),
            config=runtime_cfg,
            applied=applied,
            notes=m["notes"] + " Stub: no kernel patched; runtime applies config.",
        )
