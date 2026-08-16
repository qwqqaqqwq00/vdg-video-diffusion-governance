"""Model offload acceleration skill (CPU block-swap / sequential offload).

Moves transformer blocks (or whole sub-modules) between CPU RAM and device
memory so only the active blocks are resident, trading latency for peak memory.
Async prefetch streams overlap weight transfer with compute. On Apple Silicon
unified memory this is low-value (no discrete VRAM to page to), but it still
runs.

Grounding (video-dit-inference-acceleration-report.md, Section 4.5 / 8.2):
  * Kijai WanVideoWrapper block-swap (consumer NVIDIA): 14B Wan T2V
    @512x512x81 ~16 GB VRAM (20/40 blocks swapped); 1.3B Wan T2V, 1025 frames
    via context window (81 frames, 16 overlap) <5 GB VRAM, ~10 min on RTX 5090.
    Flags: '--async-offload <NUM_STREAMS>' (default 2, NVIDIA default on),
    '--cpu-vae', '--cache-ram'. LoRA weights now follow block swap.
  * diffusers: enable_model_cpu_offload() 33 GB -> 19 GB;
    enable_sequential_cpu_offload() -> <4 GB (slow).
  * ComfyUI-TiledVaeLite GTX 970: VAE on cuda:0, partial-load ~1184 MB resident
    / ~1504 MB offloaded.
  * No quality loss (same compute, different device); pays latency not quality.

VDG model (config key 'block_swap_ratio' in [0.0, 1.0]):
  * memory_ratio in [0.3, 0.5]: 0.0 swap -> 0.5 (full resident baseline),
    1.0 swap -> 0.3 (max paging). memory_ratio = 0.5 - 0.2 * block_swap_ratio.
  * speedup 0.6 (offload is slower than resident; async mitigates but stays
    below 1.0).
  * quality_delta 0.0 (no quality loss).
  * applies_to all categories.
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import clamp, runtime_envelope

__all__ = ["Offload"]

# memory_ratio = _RATIO_MAX - _RATIO_SPAN * block_swap_ratio, clamped to the
# documented [0.3, 0.5] range.
_RATIO_MAX = 0.5   # no swap -> 0.5 of footprint resident-relevant
_RATIO_MIN = 0.3   # full swap -> 0.3
_RATIO_SPAN = _RATIO_MAX - _RATIO_MIN  # 0.2


@register_skill("offload")
class Offload(Skill):
    """CPU block-swap / sequential offload (memory for latency)."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Offload runs on any device with host RAM; low-value on unified memory
        # but still applicable.
        return True

    def default_config(self) -> dict[str, Any]:
        # Kijai: 20/40 blocks for 14B Wan -> 0.5 swap ratio.
        return {"block_swap_ratio": 0.5}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        ratio = clamp(float(cfg.get("block_swap_ratio", 0.5)), 0.0, 1.0)
        memory_ratio = _RATIO_MAX - _RATIO_SPAN * ratio
        return SkillImpact(
            speedup=0.6,
            memory_ratio=memory_ratio,
            quality_delta=0.0,
            energy_ratio=1.0,
            applies_to=[],
            notes="Block-swap offload: memory_ratio "
                  + format(memory_ratio, ".2f")
                  + " (block_swap_ratio=" + format(ratio, ".2f")
                  + "); Kijai 14B Wan 20/40 blocks ~16 GB; diffusers "
                  "sequential offload <4 GB. No quality loss; pays latency.",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        ratio = clamp(float(cfg.get("block_swap_ratio", 0.5)), 0.0, 1.0)
        runtime_cfg = {
            "enable_offload": True,
            "block_swap_ratio": ratio,
            "async_offload_streams": 2,
            "cpu_vae": False,
        }
        applied = False
        # diffusers exposes enable_model_cpu_offload / sequential variant.
        hook = getattr(model_or_pipeline, "enable_model_cpu_offload", None)
        if callable(hook):
            try:
                hook()
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="offload",
            runtime="diffusers",
            config=runtime_cfg,
            applied=applied,
            notes="ComfyUI: '--async-offload 2' (NVIDIA default on) + Kijai "
                  "block-swap (e.g. 20/40 blocks for 14B Wan). diffusers: "
                  "pipe.enable_model_cpu_offload() or "
                  "enable_sequential_cpu_offload() (<4 GB, slower). Apple "
                  "unified memory: low-value (no discrete VRAM). Stub: runtime "
                  "applies config.",
        )
