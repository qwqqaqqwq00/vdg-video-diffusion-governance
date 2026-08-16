"""Diffusion-forcing / frame-packing skill (CogVideoX-style long video).

CogVideoX 1.5 trains with chunked denoising in which a long video is
decomposed into frames that are denoised in parallel with shared latent
context, passing KV state across chunks (diffusion forcing lineage: arXiv
2407.01392 / 2408.06072). This trades memory for speed on long sequences and
is a TRAINING-SIDE architecture change -- an existing model must be retrained /
finetuned with the frame-packing scheme; it is not a plug-in inference patch.

Grounding (video-dit-inference-acceleration-report.md, Section 6.1):
  * CogVideoX1.5 model card: '3-4x speed / 3x memory' tradeoff for long
    videos.
  * Diffusion Forcing (arXiv 2407.01392): chunked denoising with KV
    cross-chunk propagation; per-variant A100/H100 seconds not transcribed
    (HTML column misalignment) -- only the unambiguous memory figure and the
    overall tradeoff are cited.
  * Long-video memory grows linear/quadratic with frames; mitigation families:
    (1) diffusion forcing lineage (chunked denoising + cross-chunk KV),
    (2) autoregressive latent-chunk generation (rolling KV),
    (3) frame-batch / chunked denoising (Flex-Forcing, LongLive chunked VAE,
    MiniWorld async pipeline),
    (4) cross-chunk latent/feature caching.

VDG model: speedup 2.0 (conservative, below the report's 3-4x), memory_ratio
0.6, quality_delta -0.5 (frame-packing can perturb temporal consistency).
applies_to empty (architecture change -> device-agnostic once trained).
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["DiffusionForcing"]


@register_skill("diffusion_forcing")
class DiffusionForcing(Skill):
    """Chunked frame-packing denoising (CogVideoX1.5). Training-side change."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Training-side architecture: the resulting model runs on any device.
        return True

    def default_config(self) -> dict[str, Any]:
        # CogVideoX1.5 frame-packing: 16-frame chunks with shared context.
        return {"chunk_size": 16, "kv_cross_chunk": True}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        return SkillImpact(
            speedup=2.0,
            memory_ratio=0.6,
            quality_delta=-0.5,
            energy_ratio=1.0,
            applies_to=[],
            notes="Diffusion forcing / frame-packing (CogVideoX1.5): 3-4x "
                  "speed with ~3x memory tradeoff on long video (model card); "
                  "VDG uses conservative speedup 2.0, memory_ratio 0.6, "
                  "quality_delta -0.5. Training-side architecture -- needs "
                  "retraining/finetune, not a plug-in.",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        runtime_cfg = {
            "architecture": "frame_packing",
            "chunk_size": int(cfg.get("chunk_size", 16)),
            "kv_cross_chunk": bool(cfg.get("kv_cross_chunk", True)),
            "requires_retraining": True,
        }
        return runtime_envelope(
            skill="diffusion_forcing",
            runtime="comfyui",
            config=runtime_cfg,
            applied=False,
            notes="Training-side (CogVideoX1.5 / diffusion-forcing lineage, "
                  "arXiv 2407.01392 / 2408.06072): chunked denoising + "
                  "cross-chunk KV. Load a frame-packed checkpoint instead of "
                  "patching. No plug-in patch exists. Stub: runtime applies "
                  "config.",
        )
