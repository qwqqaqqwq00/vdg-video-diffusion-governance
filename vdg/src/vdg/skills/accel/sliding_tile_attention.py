"""Sliding Tile Attention (STA) acceleration skill (ICML 2025, training-free).

Replaces full 3D self-attention with a hardware-friendly per-tile sliding
window: each query attends only within a local 3D spatio-temporal tile, which
matches the observation that video-DiT attention concentrates on local windows.
FA-2 supports per-token sliding windows (``window_size``) since v2.3. STA is
pure-inference plug-in (CUDA only) and is the mechanism behind HunyuanVideo
1.5's native selective + sliding tile attention (SSTA).

Grounding (video-dit-inference-acceleration-report.md, Section 3C / 8.4):
  * Attention kernel 2.8-17x over FA-2, 1.6-10x over FA-3 (58.79% MFU).
  * End-to-end HunyuanVideo: 945 s (FA3) -> 685 s (training-free, no quality
    loss) -> 268 s (finetuned, only 0.09% VBench drop).
  * Training-free: 945/685 ~= 1.38x; the VDG model uses the conservative 1.4x.
  * Finetuned (FastVideo): the report's 945/268 ~= 3.5x is end-to-end on a
    finetuned model; VDG models the finetuned path conservatively at 2.5x
    (requires retraining, unlike the plug-in training-free path).
  * STA and SageAttention3 both rewrite attention (partial overlap) -- a
    planner should not stack both multiplicatively.

VDG model (config key 'mode'): 'training_free' (default) -> speedup 1.4,
quality_delta 0.0; 'finetuned' -> speedup 2.5, quality_delta -0.09.
CUDA + custom kernel only: applies to consumer_nv (Ampere/Ada/Hopper/Blackwell).
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceCategory, DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["SlidingTileAttention"]

# Per-mode impact (see module docstring). Quality_delta in VBench-proxy points.
_MODES: dict[str, dict[str, float]] = {
    "training_free": {"speedup": 1.4, "quality_delta": 0.0},
    "finetuned": {"speedup": 2.5, "quality_delta": -0.09},
}

_MODE_NOTES = {
    "training_free": (
        "STA training-free: HunyuanVideo 945 s -> 685 s (no quality loss); "
        "attention kernel 2.8-17x over FA-2 (ICML 2025, FastVideo)."
    ),
    "finetuned": (
        "STA finetuned: HunyuanVideo 945 s -> 268 s, -0.09 VBench (FastVideo); "
        "requires retraining, not a plug-in. VDG uses conservative 2.5x."
    ),
}


@register_skill("sliding_tile_attention")
class SlidingTileAttention(Skill):
    """Sliding tile attention (3D local window). Consumer NVIDIA only."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # STA is a CUDA custom kernel (FastVideo / HunyuanVideo SSTA); no
        # Metal or NPU backend exists.
        return device.spec().category == DeviceCategory.CONSUMER_NV

    def default_config(self) -> dict[str, Any]:
        return {"mode": "training_free"}

    def _mode(self, config: dict[str, Any]) -> str:
        mode = str(config.get("mode", "training_free")).lower()
        if mode not in _MODES:
            # Tolerate tf / ft / fine_tuned style inputs.
            if mode in ("tf", "trainingfree"):
                mode = "training_free"
            elif mode in ("ft", "finetune", "fine_tuned"):
                mode = "finetuned"
        return mode if mode in _MODES else "training_free"

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        mode = self._mode(cfg)
        m = _MODES[mode]
        return SkillImpact(
            speedup=m["speedup"],
            memory_ratio=1.0,
            quality_delta=m["quality_delta"],
            energy_ratio=1.0,
            applies_to=[DeviceCategory.CONSUMER_NV],
            notes=_MODE_NOTES[mode] + " CUDA-only kernel; FA-2 v2.3+ supports "
                  "per-token window_size. HunyuanVideo 1.5 uses native SSTA.",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        mode = self._mode(cfg)
        runtime_cfg = {
            "enable": True,
            "mode": mode,
            # FastVideo / ComfyUI-KJNodes STA node parameter (query local window).
            "window_size": 16,
        }
        applied = False
        hook = getattr(model_or_pipeline, "enable_sliding_tile_attention", None)
        if callable(hook):
            try:
                hook(window_size=runtime_cfg["window_size"])
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="sliding_tile_attention",
            runtime="comfyui",
            config=runtime_cfg,
            applied=applied,
            notes="ComfyUI: FastVideo / KJNodes STA node (window_size); "
                  "FA-2 'window_size' arg for per-token sliding windows. "
                  "Finetuned mode needs FastVideo retraining (HunyuanVideo "
                  "945->268 s, -0.09 VBench). Stub: runtime applies config.",
        )
