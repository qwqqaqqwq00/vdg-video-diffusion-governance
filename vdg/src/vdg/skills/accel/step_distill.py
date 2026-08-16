"""Step-distillation acceleration skill (few-step sampling).

Distillation is the "train on a datacenter GPU, profit at inference" paradigm:
a many-step teacher (commonly 50 steps) is distilled into a 1-8 step student, so
inference samples with far fewer denoise steps. The distillation itself is
training-side; the speedup is realized purely at inference.

Grounding (video-dit-inference-acceleration-report.md, Section 1.2):
  * CoDMD (Consistency Trajectory / DMD for video): Wan 50 -> 4 steps, ~25x
    speedup; VBench 84.5-84.9 (vs DMD 83.4 / rCM 82.8).
  * Dynamic-in-Few-Step: Wan-14B ~30x over 50 steps.
  * WanToFight: 30 FPS on RTX 5090.
  * LTX-Video realtime target: flow-matching default ~30-step, distilled to
    ~4-8 steps (LightX2V 4-step).
  * Causal Forcing++: frame-level 2-step; +0.335 VisionReward.

VDG model (config key 'steps' in {4, 8}):
  * speedup is MARGINAL: min(baseline_steps / distilled_steps, 10.0), where
    baseline_steps is the step count the workload would actually sample
    WITHOUT distillation (injected by the selector / simulator agent / CLI as
    the simulated step count, defaulting to load.default_steps). This prevents
    two failure modes: (a) a spurious multiplier on an already-distilled
    scenario (e.g. edge_npu 4-step: baseline==distilled -> 1.0x), and (b) a
    double-count when a caller also lowers config['steps'] (the speedup would
    then re-divide an already-reduced denoise time). When baseline_steps <=
    distilled steps the speedup is 1.0 (no further reduction possible).
  * quality_delta in [-3.0, -1.0]: 4 steps -> -3.0 (most quality loss),
    8 steps -> -1.0 (least). Linearly interpolated and clamped.
  * applies_to all categories (distillation is model-side, device-agnostic).

The cap models the report's observation that below ~4 steps the quality cliff
and non-attention overhead (VAE, TE) dominate, so raw step-ratio speedups do not
fully translate end-to-end.
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import clamp, runtime_envelope

__all__ = ["StepDistill"]

_SPEEDUP_CAP = 10.0
# Quality mapping: 4 steps -> -3.0, 8 steps -> -1.0 (linear, clamped).
_STEPS_LO = 4
_STEPS_HI = 8
_QUALITY_LO = -3.0
_QUALITY_HI = -1.0


def _quality_delta(steps: int) -> float:
    s = clamp(float(steps), float(_STEPS_LO), float(_STEPS_HI))
    frac = (s - _STEPS_LO) / (_STEPS_HI - _STEPS_LO)
    return _QUALITY_LO + frac * (_QUALITY_HI - _QUALITY_LO)


@register_skill("step_distill")
class StepDistill(Skill):
    """Few-step distillation (CoDMD / Dynamic-in-Few-Step style)."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Distillation is model-side; applies to every device category.
        return True

    def default_config(self) -> dict[str, Any]:
        return {"steps": 4}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        steps = max(1, int(cfg.get("steps", 4)))
        default_steps = max(1, int(load.characteristics().default_steps))
        # The speedup is MARGINAL: relative to the step count the workload would
        # actually sample WITHOUT distillation (baseline_steps), not the model
        # default. Callers (accel selector, simulator agent, CLI) inject
        # baseline_steps = the simulated step count so that:
        #   * an already-distilled scenario (e.g. edge_npu 4-step) gets no
        #     spurious multiplier (baseline == distilled -> 1.0x), and
        #   * lowering config['steps'] AND applying this speedup cannot
        #     double-count the step reduction.
        baseline_steps = int(cfg.get("baseline_steps", default_steps))
        if baseline_steps > steps:
            speedup = min(baseline_steps / steps, _SPEEDUP_CAP)
        else:
            speedup = 1.0
        quality = _quality_delta(steps)
        return SkillImpact(
            speedup=speedup,
            memory_ratio=1.0,
            quality_delta=quality,
            energy_ratio=1.0,
            applies_to=[],
            notes="Step distillation: marginal speedup = min(baseline_steps/"
                  "distilled_steps, " + str(_SPEEDUP_CAP) + ") = "
                  + format(speedup, ".2f") + "x (baseline_steps="
                  + str(baseline_steps) + ", distilled=" + str(steps)
                  + "). CoDMD 50->4 ~25x (VBench 84.5-84.9). Baselines already "
                  "at/below the distilled count get 1.0x (no double-count).",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        steps = max(1, int(cfg.get("steps", 4)))
        runtime_cfg = {
            "steps": steps,
            "scheduler": "flow_matching_distilled",
            "guidance_scale": 1.0,
            "use_distilled_weights": True,
        }
        applied = False
        hook = getattr(model_or_pipeline, "set_num_inference_steps", None)
        if callable(hook):
            try:
                hook(steps)
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="step_distill",
            runtime="diffusers",
            config=runtime_cfg,
            applied=applied,
            notes="diffusers: pipe.set_num_inference_steps(steps) with a "
                  "distilled checkpoint (CoDMD/LightX2V 4-step, SDXL-Lightning "
                  "8-step). Requires a distilled student model; training-side "
                  "cost (CoDMD ~A100-class) is not modeled here. Stub: runtime "
                  "applies the step count config.",
        )
