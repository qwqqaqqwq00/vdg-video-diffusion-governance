"""Linear attention acceleration skill (SANA-Video 2.0, training-side).

Replaces the softmax O(N^2) attention with kernelized O(N) linear attention
(recurrent state), optionally keeping a few softmax "anchor" tokens to restore
full-rank expressivity (mixed linear-softmax). This is an ARCHITECTURE-LEVEL
change: the acceleration happens at training time and the resulting model is
fast at inference, but existing models cannot get it as a plug-in patch -- they
must be retrained / finetuned.

Grounding (video-dit-inference-acceleration-report.md, Section 3D / 8.5):
  * SANA-Video (arXiv 2509.24695): linear DiT + resident block linear KV cache;
    16x faster than Wan 2.1-1.3B; NVFP4 -> 2.4x more (71 s -> 29 s, 5s 720p).
  * SANA-Video 2.0 (arXiv 2607.21553): mixed Linear-Softmax (3:1) + Block
    Attention Residuals; VBench 84.30; compiled DiT fwd 3.2x at 720p/60s;
    single H100 120x faster than Wan 2.2-A14B; +Sol-Engine 3.58x.
  * Quality: SANA-Video VBench comparable to Wan 2.1-1.3B / SkyReel-V2-1.3B;
    SANA-Video 2.0 VBench 84.30; ARL2 / SALAD report on-par quality.
  * Devices: MPS partial (portable principle, no fused impl); consumer-NV and
    datacenter (H100/5090) full; NPU partial (most NPU-friendly attention
    form in principle, no production impl yet).

VDG model: for an EXISTING load the skill is not a plug-in, so predict uses the
conservative speedup 3.0 with a training-side caveat (retraining cost, VBench
quality delta -1.0 to reflect that the ported model is not the tuned
SANA-Video 2.0). applies_to is empty (architecture change -> any device family
could host the resulting model).
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["LinearAttention"]


@register_skill("linear_attention")
class LinearAttention(Skill):
    """O(N) linear attention (SANA-Video 2.0 style). Training-side change."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Architecture-level change: the trained model runs on any device
        # family (consumer-NV, datacenter, MPS in principle, NPU in principle).
        return True

    def default_config(self) -> dict[str, Any]:
        # 3:1 linear:softmax anchors (SANA-Video 2.0) with block attention
        # residuals; resident linear KV cache.
        return {"linear_softmax_ratio": 3, "kv_cache": "block_resident"}

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
            speedup=3.0,
            memory_ratio=1.0,
            quality_delta=-1.0,
            energy_ratio=1.0,
            applies_to=[],
            notes="Linear attention (SANA-Video 2.0: mixed Linear-Softmax 3:1 "
                  "+ Block Attention Residuals, VBench 84.30). Architecture "
                  "16x vs Wan1.3B is training-side -- existing models need "
                  "retraining/finetune, so VDG credits a conservative 3.0x "
                  "with quality_delta -1.0. Not a plug-in patch.",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        # No runtime can retrofit linear attention onto a trained softmax
        # model -- this is a retraining / model-selection decision. The
        # envelope documents the target architecture for the training run.
        runtime_cfg = {
            "architecture": "linear_attention",
            "linear_softmax_ratio": cfg.get("linear_softmax_ratio", 3),
            "kv_cache": cfg.get("kv_cache", "block_resident"),
            "requires_retraining": True,
        }
        return runtime_envelope(
            skill="linear_attention",
            runtime="comfyui",
            config=runtime_cfg,
            applied=False,
            notes="Training-side: SANA-Video / SANA-Video 2.0 (arXiv "
                  "2509.24695 / 2607.21553). Load the pretrained linear-DiT "
                  "checkpoint instead of patching. No plug-in patch exists. "
                  "Stub: runtime applies config.",
        )
