"""FlashAttention acceleration skill (FA-2 / FA-3 / FA-4).

IO-aware exact attention: fuses QK^T -> softmax -> V into a single tiled kernel
resident in SRAM, never materializing the N x N matrix. This is the universal
CUDA baseline that every other attention skill is measured against.

Grounding (video-dit-inference-acceleration-report.md, Section 3A):
  * FA-2: ~10x memory saving at seq 2K, ~20x at seq 4K (A100) vs PyTorch
    eager; v2.0 ~2x over FA-1. Exact attention, no quality loss (max numeric
    error <= 2x PyTorch baseline).
  * FA-3: Hopper-only (warp specialization overlapping TensorCore with TMA
    loads, interleaved matmul/softmax, FP8 block quantization): 1.5-2.0x over
    FA-2 on H100; FP16 740 TFLOPs/s (75% MFU); FP8 ~1.2 PFLOPs/s; FP8 numeric
    error 2.6x lower than baseline FP8.
  * FA-4 (CuTeDSL rewrite): targets Hopper AND Blackwell (H100, B200, RTX
    5090); 'pip install flash-attn-4'; no measured numbers in the report.
  * Devices: MPS no / consumer-NV yes (3090/4090/5090, Jetson Orin-Ampere) /
    NPU no (CUDA/ROCm only). FA-3 additionally requires Hopper datacenter.

VDG model (config key 'version'): 'fa2' (default, speedup 1.5) or 'fa3'
(Hopper-only, speedup 1.8). predict guards fa3: on a non-Hopper device it
returns a neutral no-op so an impossible 1.8x is never credited. FA-4 is
Blackwell (5090) but unreported -- notes only.
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceCategory, DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["FlashAttention"]

# Per-version impact (see module docstring).
_VERSIONS: dict[str, dict[str, float]] = {
    "fa2": {"speedup": 1.5},
    "fa3": {"speedup": 1.8},
}


def _version_applicable(device: DeviceProfile, version: str) -> bool:
    """Whether a specific FlashAttention version can run on the device."""
    spec = device.spec()
    if spec.category not in (DeviceCategory.CONSUMER_NV, DeviceCategory.DATACENTER):
        return False  # CUDA/ROCm only; no Metal / NPU backend.
    if version == "fa3" and spec.category != DeviceCategory.DATACENTER:
        return False  # FA-3 is Hopper datacenter-only (H100/H800/H20).
    return True


@register_skill("flash_attention")
class FlashAttention(Skill):
    """IO-aware fused exact attention (FA-2 / FA-3). CUDA only."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Broad family gate: any FA version runs -> consumer_nv / datacenter.
        # The per-version (fa3-Hopper) guard is enforced in predict().
        return any(_version_applicable(device, v) for v in _VERSIONS)

    def default_config(self) -> dict[str, Any]:
        return {"version": "fa2"}

    def _version(self, config: dict[str, Any]) -> str:
        version = str(config.get("version", "fa2")).lower().replace("-", "")
        short = version.lstrip("flash_attn").lstrip("v")
        if short in ("2", "3", "4"):
            version = "fa" + short
        return version if version in _VERSIONS else "fa2"

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        version = self._version(cfg)
        if not _version_applicable(device, version):
            need = "Hopper datacenter (H100/H800/H20)" if version == "fa3" \
                else "consumer-NV or datacenter CUDA"
            return SkillImpact(
                speedup=1.0,
                memory_ratio=1.0,
                quality_delta=0.0,
                energy_ratio=1.0,
                applies_to=[],
                notes="FlashAttention " + version.upper() + " needs " + need
                      + "; " + device.spec().name + " lacks it -> no effect. "
                      + ("Use fa2 instead." if version == "fa3" else ""),
            )
        return SkillImpact(
            speedup=_VERSIONS[version]["speedup"],
            memory_ratio=1.0,
            quality_delta=0.0,
            energy_ratio=1.0,
            applies_to=[DeviceCategory.CONSUMER_NV, DeviceCategory.DATACENTER],
            notes="FlashAttention "
                  + ("2" if version == "fa2" else "3")
                  + ": exact fused attention, no quality loss. "
                  + ("FA-2: ~10x memory saving at seq 2K vs eager." if version == "fa2"
                     else "FA-3 Hopper-only: 1.5-2.0x over FA-2 on H100; FA-4 "
                         "(CuTeDSL) targets Hopper+Blackwell, unreported."),
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        version = self._version(cfg)
        runtime_cfg = {
            "enable": True,
            "version": version,
            "backend": "flash_attn",
            "window_size": None,  # FA-2 v2.3+ optional sliding-window.
        }
        applied = False
        hook = getattr(model_or_pipeline, "enable_flash_attention", None)
        if callable(hook):
            try:
                hook(version=version)
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="flash_attention",
            runtime="comfyui",
            config=runtime_cfg,
            applied=applied,
            notes="ComfyUI: torch SDPA already uses the FA kernel on NVIDIA "
                  "when available ('--use-sage-attention' overrides it); "
                  "explicit FA: F.scaled_dot_product_attention with "
                  "'enable_flash_sdp' or 'pip install flash-attn'. FA-3 needs "
                  "Hopper (flash_attn_3); FA-4 (flash-attn-4, CuTeDSL) runs "
                  "on Hopper + Blackwell 5090. Stub: runtime applies config.",
        )
