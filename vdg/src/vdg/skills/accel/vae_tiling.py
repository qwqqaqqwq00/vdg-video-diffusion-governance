"""VAE tiling acceleration skill (spatial + temporal tiling of the 3D VAE).

The video 3D VAE expands the latent along spatial and temporal axes; its
*activation* memory (not weight memory) dominates and can OOM a consumer / edge
device before the transformer even runs. Tiling splits the latent into
overlapping spatial and temporal tiles, decodes each, and blends the overlap
bands, bounding peak activation memory by tile size rather than full resolution.

Grounding (video-dit-inference-acceleration-report.md, Section 4 / 8.7):
  * Temporal tiling: ComfyUI v0.3.10 HunyuanVideo VAE 32 GB -> 8 GB (4x cut)
    with tile_size=128, overlap=32, temporal_size=32, temporal_overlap=4.
  * Spatial tiling: SD v1.5 1024^2 RTX 3070 VAE 56.3% -> 16.0% (~3.5x cut),
    latency +77%/+63%; on RTX 4090 +103% time with no memory benefit.
  * Streaming / causal VAE (CogVideoX): 49f@720x480 19 GB -> 11 GB;
    sequential offload -> <4 GB (slow).
  * ComfyUI VAEDecodeTiled defaults: tile_size 512 (64-4096, step 32),
    overlap 64, temporal_size 64 (8-4096, step 4), temporal_overlap 8.

VDG model:
  * memory_ratio 0.25 (the VAE-segment peak cut, grounded in HunyuanVideo
    32 GB -> 8 GB). NOTE: the simulator applies memory_ratio to total
    footprint; this ratio models the VAE segment, so it is most accurate when
    VAE activations dominate (e.g. a HunyuanVideo VAE OOM). A governance agent
    should compose it accordingly.
  * speedup 0.9 (tiling is memory-fit, not speed; tiled decode is slightly
    slower than monolithic due to overlap recompute).
  * quality_delta -0.1 (tile-seam artifacts, mitigated by overlap >= 64;
    report: FID/FVD not reported, seams nearly invisible at overlap >= 64).
  * applies_to all categories (MPS / consumer NV / edge NPU all support tiling).
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["VAETiling"]


@register_skill("vae_tiling")
class VAETiling(Skill):
    """Spatial + temporal VAE tiling (peak activation memory cut)."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Tiling is a pure-PyTorch memory technique; all categories support it.
        return True

    def default_config(self) -> dict[str, Any]:
        # ComfyUI VAEDecodeTiled defaults (tile_size, overlap, temporal,
        # temporal_overlap). 'temporal' is the temporal tile size in frames.
        return {
            "tile_size": 512,
            "overlap": 64,
            "temporal": 64,
            "temporal_overlap": 8,
        }

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        return SkillImpact(
            speedup=0.9,
            memory_ratio=0.25,
            quality_delta=-0.1,
            energy_ratio=1.0,
            applies_to=[],
            notes="VAE tiling: HunyuanVideo 32 GB -> 8 GB (0.25 VAE-segment "
                  "ratio); tiled decode slightly slower (0.9x); seam artifacts "
                  "mitigated by overlap >= 64. All devices.",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        tile_size = int(cfg.get("tile_size", 512))
        overlap = int(cfg.get("overlap", 64))
        temporal = int(cfg.get("temporal", 64))
        temporal_overlap = int(cfg.get("temporal_overlap", 8))
        runtime_cfg = {
            "enable_tiling": True,
            "tile_size": tile_size,
            "overlap": overlap,
            "temporal_size": temporal,
            "temporal_overlap": temporal_overlap,
        }
        applied = False
        # diffusers AutoencoderKL* exposes enable_tiling(); ComfyUI uses the
        # VAEDecode(Tiled) node instead of a model method.
        hook = getattr(model_or_pipeline, "enable_tiling", None)
        if callable(hook):
            try:
                hook(
                    tile_size=tile_size,
                    overlap=overlap,
                    temporal_size=temporal,
                    temporal_overlap=temporal_overlap,
                )
                applied = True
            except TypeError:
                # Some signatures take no kwargs; fall back to bare enable.
                try:
                    hook()
                    applied = True
                except Exception:
                    applied = False
            except Exception:
                applied = False
        return runtime_envelope(
            skill="vae_tiling",
            runtime="comfyui",
            config=runtime_cfg,
            applied=applied,
            notes="ComfyUI: VAEDecode(Tiled) node args tile_size/overlap/"
                  "temporal_size/temporal_overlap (defaults 512/64/64/8). "
                  "diffusers: vae.enable_tiling(tile_size=, overlap=, "
                  "temporal_size=, temporal_overlap=). Stub: runtime applies "
                  "config.",
        )
