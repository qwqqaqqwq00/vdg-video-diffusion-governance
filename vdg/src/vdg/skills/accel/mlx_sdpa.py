"""MLX fused scaled-dot-product attention skill (Apple Silicon).

Uses Apple MLX's hand-written fused attention primitive
``mx.fast.scaled_dot_product_attention`` (Metal native, Flash-style kernel
using tiling + shared memory) instead of PyTorch MPS eager SDPA. This is the
Apple-silicon counterpart to FlashAttention on CUDA: MLX's fusion keeps the
attention matrix in SRAM/registers, so the T x T matrix is never materialized.

Grounding (video-dit-inference-acceleration-report.md, Section 7.5 / 8.3):
  * mx.fast.scaled_dot_product_attention ~1.4-1.5x faster than eager, AND
    avoids materializing the 4.3 GB T x T attention matrix at T=8192 (the
    memory win is the primary benefit).
  * MLX fusion primitives (mx.fast.*): elementwise memory-bound up to 17x
    (GELU 4k x 4k: 12 ms -> 0.7 ms), typical ~4x; mx.fast.layer_norm 2-7x vs
    eager; whole training step ~1.2x.
  * Inference: mx.compile alone is only 1.04-1.10x/token -- the real inference
    win on Apple silicon is mx.fast SDPA, not compile.
  * Cold start is low (lazy/dynamic graph; changing parameter shapes does not
    trigger slow recompilation).

VDG model: speedup 1.4 (attention kernel), memory_ratio 0.85 (avoids the 4.3 GB
T x T materialization at T=8192). Device-gated to apple_silicon; MLX also has a
Linux CUDA backend but the skill targets the Apple unified-memory path.
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceCategory, DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["MlxSdpa"]


@register_skill("mlx_sdpa")
class MlxSdpa(Skill):
    """MLX mx.fast.scaled_dot_product_attention. Apple Silicon only."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # MLX Metal path is Apple-silicon only (MPS is the PyTorch fallback;
        # MLX itself has a Linux CUDA backend but this skill is the Metal path).
        return device.spec().category == DeviceCategory.APPLE_SILICON

    def default_config(self) -> dict[str, Any]:
        # Report figure: 4.3 GB T x T matrix at T=8192 with 16 heads in fp32
        # (16 * 8192^2 * 4 bytes = 4.29 GB).
        return {"seq_len": 8192, "heads": 16}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        seq_len = int(cfg.get("seq_len", 8192))
        heads = int(cfg.get("heads", 16))
        # Report figure: the materialized H x T x T fp32 attention matrix is
        # 16 * 8192^2 * 4 B = 4.29 GB at T=8192 (MX SDPA keeps it in registers).
        materialized_gb = seq_len * seq_len * heads * 4 / 1e9
        return SkillImpact(
            speedup=1.4,
            memory_ratio=0.85,
            quality_delta=0.0,
            energy_ratio=1.0,
            applies_to=[DeviceCategory.APPLE_SILICON],
            notes="mx.fast.scaled_dot_product_attention (MLX kernel fusion, "
                  "Flash-style tiled Metal kernel): 1.4-1.5x on attention; "
                  "avoids materializing the "
                  + format(materialized_gb, ".2f")
                  + " GB attention matrix (" + str(heads) + " x " + str(seq_len)
                  + "^2 x fp32) at T=" + str(seq_len)
                  + " -- the report's 4.3 GB at T=8192 is the primary win. "
                  + "Exact attention, no quality loss.",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        runtime_cfg = {
            "enable": True,
            "backend": "mlx_mx_fast_sdpa",
            "seq_len": int(cfg.get("seq_len", 8192)),
            "heads": int(cfg.get("heads", 16)),
        }
        applied = False
        hook = getattr(model_or_pipeline, "enable_mlx_sdpa", None)
        if callable(hook):
            try:
                hook()
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="mlx_sdpa",
            runtime="mlx",
            config=runtime_cfg,
            applied=applied,
            notes="MLX: import mlx.core as mx; mx.fast.scaled_dot_product_attention "
                  "(requires 'pip install mlx'; Metal native, unified memory). "
                  "Replaces PyTorch MPS eager SDPA in the attention module. "
                  "Compile alone is only 1.04-1.10x/token on inference -- use "
                  "mx.fast SDPA. Stub: runtime applies config.",
        )
