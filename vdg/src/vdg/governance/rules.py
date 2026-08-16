"""Governance rule engine: device/quality/energy guard rules.

The rule engine sits between diagnosis and accel-selection. It inspects the
device, load, scenario, policy and the *unskilled baseline* simulation result,
and emits a RuleOutcome of:

  * decisions           -- GovernanceDecision recommendations to apply
                                (e.g. add boundary-block bf16 protection).
  * disabled_skills     -- skill names the accel selector must NOT include
                                (e.g. SageAttention on Apple Silicon).
  * preferred_skills    -- skill names to boost in Pareto ranking
                                (e.g. step distillation when energy is blown).
  * config_overrides    -- config keys to force on every candidate
                                (e.g. allowed quantization methods, VAE
                                precision, boundary-block count).
  * notes               -- human-readable rationale lines for the report.

Four canonical rules (grounded in the synthesis report section 4.2 and the
cross-device robustness report section 7):

  R1  device == apple_silicon  -> disable CUDA-only attention skills
      (FlashAttention/SageAttention are CUDA+Triton only; MPS SDPA falls back
      to a non-fused math path. Apple Silicon attention acceleration is limited
      to MLX mx.fast SDPA + TeaCache -- both device-agnostic.)

  R2  baseline energy > energy_budget -> prefer step_distill
      (Step distillation 50->4 is the single largest energy lever: ~12x energy
      cut, produced once on the training side.)

  R3  quality_target > 85 -> limit quantization
      (Restrict the quantization skill to the mild gguf_q4 method; disallow
      nvfp4/int8 whose quality deltas are too large to clear a >85 floor.)

  R4  fp8 weights on an int8-only device -> add boundary-block bf16 protection
      (When a model trained with fp8 weights is deployed on a device that has
      int8 tensor cores but NO fp8 hardware -- consumer Ampere/Jetson -- the
      format mismatch risks divergence on the first/last transformer blocks.
      Wan2.1-T2V-14B HiFloat8 keeps the first 2 + last 3 blocks in bf16; this
      rule recommends the same boundary-block guard.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.contracts import (
    DeviceCategory,
    DeviceProfile,
    GovernanceDecision,
    LoadModel,
    SkillImpact,
)
from ..core.simulator import SimulationResult
from ..core.scenario import Scenario
from .policy import Policy

__all__ = [
    "RuleOutcome",
    "RuleEngine",
    "SKILL_TEACACHE",
    "SKILL_SAGE",
    "SKILL_DISTILL",
    "SKILL_QUANT",
    "SKILL_VAE_TILING",
    "SKILL_COMPILE",
    "SKILL_OFFLOAD",
    "SKILL_STA",
    "SKILL_LINEAR_ATTN",
    "SKILL_ADALN_FP32",
    "SKILL_GELU_FP32",
    "SKILL_RMSNORM_FP32",
    "SKILL_SOFTMAX_FP32",
    "SKILL_VAE_FP32",
    "SKILL_BOUNDARY_BF16",
]

# Canonical skill names -- aligned to the phase-2 skill plugins registered via
# @register_skill in vdg/skills/. Configurable skills carry their variant in a
# config key (sage_attention: version; quantization: method; step_distill:
# steps; compile_graph: backend; teacache: threshold; offload: block_swap_ratio).
SKILL_TEACACHE = "teacache"
SKILL_SAGE = "sage_attention"          # version v1/v2/v3
SKILL_DISTILL = "step_distill"         # steps 4/8
SKILL_QUANT = "quantization"           # method gguf_q4/nvfp4/int8
SKILL_VAE_TILING = "vae_tiling"
SKILL_COMPILE = "compile_graph"        # backend torch_compile/trt
SKILL_OFFLOAD = "offload"              # block_swap_ratio

# Conceptual skills referenced by recipe presets but not yet registered as
# plugins; the accel selector degrades to a documented estimate for these.
SKILL_STA = "sliding_tile_attention"
SKILL_LINEAR_ATTN = "linear_attention"

# Registered granular repair skills (kind="repair").
SKILL_ADALN_FP32 = "adaln_fp32"
SKILL_GELU_FP32 = "gelu_fp32"
SKILL_RMSNORM_FP32 = "rmsnorm_fp32"
SKILL_SOFTMAX_FP32 = "softmax_fp32"
SKILL_VAE_FP32 = "vae_fp32"

# Block-level repair (boundary bf16 guard); registered as the
# "boundary_block_bf16" skill plugin (vdg/skills/repair/boundary_bf16.py).
SKILL_BOUNDARY_BF16 = "boundary_block_bf16"

# CUDA+Triton-only attention skills that cannot run on Apple Silicon / non-NV
# NPUs (synthesis report 4.2 constraint 1).
CUDA_ONLY_ATTENTION_SKILLS = (
    SKILL_SAGE,
    SKILL_STA,
)


@dataclass
class RuleOutcome:
    """Aggregate output of the rule engine for one (device, load, scenario)."""

    decisions: list[GovernanceDecision] = field(default_factory=list)
    disabled_skills: set[str] = field(default_factory=set)
    preferred_skills: set[str] = field(default_factory=set)
    config_overrides: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def merge(self, other: "RuleOutcome") -> "RuleOutcome":
        self.decisions.extend(other.decisions)
        self.disabled_skills |= other.disabled_skills
        self.preferred_skills |= other.preferred_skills
        for k, v in other.config_overrides.items():
            self.config_overrides[k] = v
        self.notes.extend(other.notes)
        return self


class RuleEngine:
    """Applies the canonical governance rules to a deployment context.

    The engine is stateless: apply inspects its arguments and returns a
    fresh RuleOutcome. baseline_result may be None when no baseline
    has been simulated yet (R2 then treats the budget as tight conservatively).
    """

    def apply(
        self,
        device: DeviceProfile,
        load: LoadModel,
        scenario: Scenario,
        policy: Policy,
        baseline_result: SimulationResult | None = None,
    ) -> RuleOutcome:
        outcome = RuleOutcome()
        spec = device.spec()
        outcome.merge(self._r1_apple_silicon(spec))
        outcome.merge(self._r2_energy_budget(spec, scenario, policy, baseline_result))
        outcome.merge(self._r3_quality_floor(scenario, policy))
        outcome.merge(self._r4_fp8_on_int8(spec, load))
        return outcome

    # -- R1 ----------------------------------------------------------------
    def _r1_apple_silicon(self, spec) -> RuleOutcome:
        if spec.category != DeviceCategory.APPLE_SILICON:
            return RuleOutcome()
        outcome = RuleOutcome()
        outcome.disabled_skills.update(CUDA_ONLY_ATTENTION_SKILLS)
        # TeaCache is the only device-agnostic attention-side acceleration that
        # runs on MPS (pure output caching; no CUDA/Triton/Metal-fused kernel).
        # With FlashAttention/SageAttention/compile_graph all CUDA-only and thus
        # disabled, prefer TeaCache so the Pareto selector boosts it over
        # step-distillation (which is model-side, not attention-side) when both
        # are policy-feasible -- matching the report's Apple-Silicon guidance.
        outcome.preferred_skills.add(SKILL_TEACACHE)
        outcome.notes.append(
            "R1: Apple Silicon has no fused attention kernels "
            "(FlashAttention/SageAttention are CUDA+Triton only); disabled "
            + ", ".join(CUDA_ONLY_ATTENTION_SKILLS) + ". Prefer " + SKILL_TEACACHE
            + " (device-agnostic output caching) for attention-side gains; "
            "use MLX mx.fast SDPA for the fused path."
        )
        return outcome

    # -- R2 ----------------------------------------------------------------
    def _r2_energy_budget(
        self, spec, scenario: Scenario, policy: Policy,
        baseline: SimulationResult | None,
    ) -> RuleOutcome:
        if policy.energy_budget_j == float("inf"):
            return RuleOutcome()
        if baseline is not None:
            exceeded = baseline.energy_j > policy.energy_budget_j
            headroom = baseline.energy_j / policy.energy_budget_j if policy.energy_budget_j > 0 else 1.0
        else:
            exceeded = True
            headroom = 1.0
        if not exceeded:
            return RuleOutcome()
        outcome = RuleOutcome()
        outcome.preferred_skills.add(SKILL_DISTILL)
        outcome.notes.append(
            "R2: energy budget exceeded; prefer " + SKILL_DISTILL
            + " (ratio " + format(headroom, ".2f") + "). Step distillation is the "
            "largest energy lever (50->4 steps ~ 12x energy cut); the accel "
            "selector boosts distill combos in Pareto ranking."
        )
        return outcome

    # -- R3 ----------------------------------------------------------------
    def _r3_quality_floor(self, scenario: Scenario, policy: Policy) -> RuleOutcome:
        # Fires when the effective quality target (scenario target OR the CLI
        # --quality-floor policy) exceeds 85: limit quantization to the mild
        # gguf_q4 method, disallowing nvfp4/int8 whose quality deltas are too
        # large to clear a strict floor.
        effective = max(scenario.quality_target, policy.quality_floor)
        if effective <= 85.0:
            return RuleOutcome()
        outcome = RuleOutcome()
        # Restrict quantization to the mild gguf_q4 method; nvfp4/int8 carry
        # quality deltas too large for a >85 floor.
        outcome.config_overrides["quant_methods_allowed"] = ["gguf_q4"]
        outcome.notes.append(
            "R3: effective quality target " + format(effective, ".1f")
            + " > 85; restricting quantization to gguf_q4 (disallow nvfp4/int8) "
            "to protect the quality floor."
        )
        return outcome

    # -- R4 ----------------------------------------------------------------
    def _r4_fp8_on_int8(self, spec, load: LoadModel) -> RuleOutcome:
        has_int8 = spec.supports("int8")
        has_fp8 = spec.supports("fp8")
        # int8-only device (Ampere consumer/Jetson): int8 yes, fp8 no.
        if not (has_int8 and not has_fp8):
            return RuleOutcome()
        outcome = RuleOutcome()
        first_blocks = 2
        last_blocks = 3
        outcome.config_overrides["boundary_first_blocks"] = first_blocks
        outcome.config_overrides["boundary_last_blocks"] = last_blocks
        outcome.decisions.append(GovernanceDecision(
            skill_name=SKILL_BOUNDARY_BF16,
            config={
                "first_blocks": first_blocks,
                "last_blocks": last_blocks,
                "precision": "bf16",
            },
            predicted_impact=SkillImpact(
                speedup=1.0,
                memory_ratio=1.0,
                quality_delta=0.0,
                energy_ratio=1.0,
                applies_to=[DeviceCategory.CONSUMER_NV, DeviceCategory.EDGE_NPU],
                notes="Keep first 2 + last 3 transformer blocks in bf16 (Wan2.1-T2V-14B "
                      "HiFloat8 boundary protection, arXiv:2606.00957).",
            ),
            rationale=(
                "R4: int8-only device (" + spec.name + ") deploying fp8-trained "
                "weights (" + load.characteristics().model_name + ") risks divergence "
                "on the boundary blocks from the fp8->int8 format mismatch. Add "
                "boundary-block bf16 protection (first " + str(first_blocks) + " + last "
                + str(last_blocks) + " blocks) per the HiFloat8 recipe."
            ),
        ))
        outcome.notes.append(
            "R4: int8-only device with fp8/int8 quant path; recommend "
            + SKILL_BOUNDARY_BF16 + " (first " + str(first_blocks) + " + last "
            + str(last_blocks) + " blocks)."
        )
        return outcome
