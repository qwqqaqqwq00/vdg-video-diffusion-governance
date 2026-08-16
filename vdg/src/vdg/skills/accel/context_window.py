"""Kijai context-window chunking skill (long-video VRAM control).

Processes a long video by splitting the latent/frame stream into overlapping
context windows (chunked denoising) so peak activation memory stays bounded
regardless of total frame count. The windows are re-fused with overlap blending
at the end. This is the Kijai WanVideoWrapper context-window node -- the
standard ComfyUI mechanism for 1000+ frame generations on consumer cards.

Grounding (video-dit-inference-acceleration-report.md, Section 4.5 / 8.1):
  * Kijai WanVideoWrapper, consumer NVIDIA: 1.3B Wan T2V, 1025 frames via
    context window (window 81 frames, overlap 16) -> <5 GB VRAM on RTX 5090,
    ~10 min.
  * Context-window node chunking (cli_args.py / context_windows/): window +
    overlap control peak activation memory per chunk instead of per full
    sequence.

VDG model (config keys 'window'/'overlap'): memory_ratio 0.35 for long-video
scenarios (1025 frames -> 81-frame windows is roughly an 8-13x activation cut;
0.35 is a conservative end-to-end figure accounting for weights + overlap
recompute), speedup 0.9 (overlap recompute + re-fusion cost). applies_to all
categories: chunking is runtime-level and device-agnostic (ComfyUI runs on
NVIDIA, Apple Silicon MPS, and NPU frontends).
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["ContextWindow"]


@register_skill("context_window")
class ContextWindow(Skill):
    """Overlapping context-window chunking for long video. All devices."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Chunked denoising is runtime-level and device-agnostic.
        return True

    def default_config(self) -> dict[str, Any]:
        # Kijai WanVideoWrapper long-video setting (1025 frames -> window 81,
        # overlap 16, <5 GB VRAM on RTX 5090).
        return {"window": 81, "overlap": 16}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        window = int(cfg.get("window", 81))
        overlap = int(cfg.get("overlap", 16))
        return SkillImpact(
            speedup=0.9,
            memory_ratio=0.35,
            quality_delta=0.0,
            energy_ratio=1.0,
            applies_to=[],
            notes="Kijai context window: 1025 frames -> window " + str(window)
                  + " + overlap " + str(overlap)
                  + " -> <5 GB VRAM on RTX 5090 (~10 min, 1.3B Wan T2V). "
                  + "memory_ratio 0.35 for long-video scenarios; speedup 0.9 "
                  + "from overlap recompute + re-fusion.",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        window = int(cfg.get("window", 81))
        overlap = int(cfg.get("overlap", 16))
        runtime_cfg = {
            "enable": True,
            "chunking": "context_window",
            "window": window,
            "overlap": overlap,
        }
        applied = False
        hook = getattr(model_or_pipeline, "enable_context_window", None)
        if callable(hook):
            try:
                hook(window=window, overlap=overlap)
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="context_window",
            runtime="comfyui",
            config=runtime_cfg,
            applied=applied,
            notes="ComfyUI: Kijai WanVideoWrapper context-window node "
                  "(context_windows/; e.g. 1025 frames -> 81-frame window with "
                  "16-frame overlap). Chunked denoising bounds activation "
                  "memory per chunk; windows re-fused with overlap blending. "
                  "Stub: runtime applies config.",
        )
