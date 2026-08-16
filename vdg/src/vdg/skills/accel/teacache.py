"""TeaCache acceleration skill (CVPR 2025 Highlight, training-free).

TeaCache caches and reuses transformer outputs across denoising steps when the
"timestep-embedding-modulated input residual" is below a threshold, skipping
the full DiT forward. It is device-agnostic (pure output caching): Apple
Silicon, consumer NVIDIA and edge NPU all support it.

Grounding (video-dit-inference-acceleration-report.md, Section 2 / 8.4):
  * Open-Sora-Plan 4.41x-4.91x (99.65s -> 22.62s); -0.07% VBench at 4.41x.
  * Per-model measured: Wan2.1 1.4-2.9x, HunyuanVideo 1.6x/2.1x,
    CogVideoX1.5 1.3-2.1x, LTX-Video 1.6x/2.1x, Mochi 1.5x/2.1x,
    Cosmos 1.4x/2.0x.
  * Kijai ComfyUI tuning: with coefficients the 0.25-0.30 range works well;
    start_step can be 0 once an aggressive threshold is set, to avoid
    early-motion corruption.

VDG model: a user threshold in [0.05, 0.25] maps linearly to speedup
[1.6, 4.0] and quality_delta [-0.07, -1.0]. A low threshold is conservative
(mostly recompute -> high quality, small speedup); a high threshold is
aggressive (more caching -> lower quality, larger speedup). Values outside the
range are clamped to the documented endpoints.
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import clamp, runtime_envelope

__all__ = ["TeaCache"]

# Threshold -> speedup / quality mapping (see module docstring).
_THR_MIN = 0.05
_THR_MAX = 0.25
_SPEEDUP_MIN = 1.6    # threshold 0.05 -> 1.6x
_SPEEDUP_MAX = 4.0    # threshold 0.25 -> 4.0x
_QUALITY_MIN = -0.07  # threshold 0.05 -> -0.07 VBench (negligible)
_QUALITY_MAX = -1.0   # threshold 0.25 -> -1.0 VBench


def _impact(threshold: float) -> tuple[float, float]:
    """Return (speedup, quality_delta) for a TeaCache threshold."""
    t = clamp(float(threshold), _THR_MIN, _THR_MAX)
    frac = (t - _THR_MIN) / (_THR_MAX - _THR_MIN)
    speedup = _SPEEDUP_MIN + frac * (_SPEEDUP_MAX - _SPEEDUP_MIN)
    quality = _QUALITY_MIN + frac * (_QUALITY_MAX - _QUALITY_MIN)
    return speedup, quality


@register_skill("teacache")
class TeaCache(Skill):
    """Cross-step caching (TeaCache). Device-agnostic, training-free."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Pure output caching -> applies to every device category.
        return True

    def default_config(self) -> dict[str, Any]:
        return {"threshold": 0.1}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        threshold = float(cfg.get("threshold", 0.1))
        speedup, quality = _impact(threshold)
        return SkillImpact(
            speedup=speedup,
            memory_ratio=1.0,
            quality_delta=quality,
            energy_ratio=1.0,
            applies_to=[],
            notes="TeaCache 1.6-4.0x (threshold 0.05->1.6x, 0.25->4.0x); "
                  "-0.07% VBench at 4.41x (CVPR 2025); device-agnostic caching.",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        threshold = float(cfg.get("threshold", 0.1))
        runtime_cfg = {
            "enable": True,
            # ComfyUI TeaCache node exposes this as 'rel_l1_thresh'.
            "rel_l1_thresh": threshold,
            "start_step": 0,
            "end_step": None,
        }
        applied = False
        # Best-effort in-process patch for diffusers-style pipelines exposing a
        # TeaCache hook; otherwise the runtime consumes runtime_cfg.
        hook = getattr(model_or_pipeline, "enable_teacache", None)
        if callable(hook):
            try:
                hook(threshold=threshold)
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="teacache",
            runtime="comfyui",
            config=runtime_cfg,
            applied=applied,
            notes="ComfyUI: TeaCache4Wan2.1 / TeaCacheHunyuanVideo node arg "
                  "'rel_l1_thresh' (Kijai 0.25-0.30 with coefficients). "
                  "diffusers: pipe.enable_teacache(threshold=...). "
                  "Stub: no kernel patched; runtime applies config.",
        )
