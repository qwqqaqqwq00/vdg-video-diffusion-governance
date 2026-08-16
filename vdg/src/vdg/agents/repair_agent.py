"""RepairAgent: applies recommended repair skills -> patched config + instructions.

The repair agent consumes the repair GovernanceDecision recommendations
produced by the diagnostic agent (the granular adaln_fp32 / gelu_fp32 /
rmsnorm_fp32 / softmax_fp32 / vae_fp32 skills from the probe's
repair_skills_suggested) and the rule engine (the boundary_block_bf16
block-level guard from R4). It resolves each repair skill from the
registry, applies it to a real pipeline when one is supplied, and emits:

  * a **patched config** -- the deployment config with repair settings merged
    in (per-op fp32 guard flags, VAE precision, boundary-block counts) so the
    downstream SimulatorAgent simulates the *repaired* operating point.
  * **patch instructions** -- grounded, actionable patch text (the LTX-2.3
    three-cast fp32 template from MPS_BLACK_VIDEO_FIX.md, the boundary-block
    bf16 recipe, the PREC_GUARD_OPS op-level template).
  * **repair_skills** -- the resolved repair Skill instances, so the
    pipeline can include them in the final simulation (their predict() impacts
    model the repair's latency/quality cost).

If a recommended repair skill is not registered, the agent emits
instructions only.
"""
from __future__ import annotations

from typing import Any

from ..core.contracts import GovernanceDecision, Skill
from ..core.registry import REGISTRY
from ..core.simulator import AgentContext
from .base import GovernanceAgent

__all__ = ["RepairAgent", "PREC_GUARD_OPS"]

# The device-agnostic op-level precision-guard template: on low-precision
# backends, force fp32 intermediate computation for these sensitive ops.
# Grounded in the cross-device robustness report section 7 (PREC_GUARD_OPS).
PREC_GUARD_OPS = (
    "gelu",
    "gelu_tanh",
    "silu",
    "adln_modulate",
    "rmsnorm",
    "layernorm",
    "groupnorm",
    "softmax",
    "timestep_embed_mlp",
    "vae_decode",
)

# Granular repair skill -> (guard flag key, op name) for patched_config mapping.
_REPAIR_SKILL_MAP = {
    "adaln_fp32": ("guard_adaln", "adln_modulate"),
    "gelu_fp32": ("guard_gelu", "gelu_tanh"),
    "rmsnorm_fp32": ("guard_rmsnorm", "rmsnorm"),
    "softmax_fp32": ("guard_softmax", "softmax"),
    "vae_fp32": ("guard_vae", "vae_decode"),
}

_BOUNDARY_BF16 = "boundary_block_bf16"


class RepairAgent(GovernanceAgent):
    """Applies repair-skill decisions and produces patched config + instructions."""

    name = "repair"
    role = "repair"

    def __init__(self, name: str | None = None, role: str | None = None) -> None:
        super().__init__(name, role)
        self.last_patched_config: dict[str, Any] = {}
        self.last_instructions: list[str] = []

    def run(self, context: AgentContext) -> dict[str, Any]:
        cfg = dict(context.config or {})
        repair_decisions: list[GovernanceDecision] = list(cfg.get("repair_decisions", []) or [])
        overrides = dict(cfg.get("config_overrides", {}))

        patched_config: dict[str, Any] = {}
        patched_config.update(overrides)
        patched_config.setdefault("precision_guard_ops", list(PREC_GUARD_OPS))
        instructions: list[str] = []
        applied: list[str] = []
        repair_skills: list[Skill] = []
        pipeline = cfg.get("pipeline")

        for decision in repair_decisions:
            skill_name = decision.skill_name
            skill = self._resolve_skill(skill_name)
            if skill is not None:
                repair_skills.append(skill)
                if pipeline is not None:
                    try:
                        skill.apply(pipeline, decision.config)
                        applied.append(skill_name + " (applied to pipeline)")
                    except Exception as exc:
                        applied.append(skill_name + " (apply failed: " + repr(exc) + ")")
                else:
                    applied.append(skill_name + " (resolved; no pipeline to apply)")
            else:
                applied.append(skill_name + " (skill not registered; instructions only)")

            self._merge_repair(patched_config, instructions, decision, skill_name)

        # Boundary overrides present without an explicit decision still get
        # documented (rule R4 may set config_overrides directly).
        if "boundary_first_blocks" in overrides and not any(
            d.skill_name == _BOUNDARY_BF16 for d in repair_decisions
        ):
            instructions.append(self._boundary_instruction(overrides))

        notes = self._summarize(repair_decisions, applied)
        self.last_patched_config = patched_config
        self.last_instructions = instructions

        return {
            "agent": self.name,
            "role": self.role,
            "decisions": repair_decisions,
            "notes": notes,
            "extra": {
                "patched_config": patched_config,
                "patch_instructions": instructions,
                "applied": applied,
                "repair_skills": repair_skills,
            },
        }

    # -- helpers -----------------------------------------------------------
    def _resolve_skill(self, skill_name: str) -> Skill | None:
        cls = REGISTRY.get("skill", skill_name)
        if cls is None:
            return None
        try:
            return cls()
        except Exception:
            return None

    def _merge_repair(
        self, patched_config: dict[str, Any], instructions: list[str],
        decision: GovernanceDecision, skill_name: str,
    ) -> None:
        if skill_name in _REPAIR_SKILL_MAP:
            flag_key, op = _REPAIR_SKILL_MAP[skill_name]
            patched_config[flag_key] = True
            guard_ops = list(patched_config.get("precision_guard_ops", PREC_GUARD_OPS))
            if op not in guard_ops:
                guard_ops.append(op)
            patched_config["precision_guard_ops"] = guard_ops
            if skill_name == "vae_fp32":
                # Forcing VAE decode to fp32 is the one config the simulator
                # reads directly (vae_precision drives the VAE roofline).
                patched_config["vae_precision"] = "fp32"
            instructions.append(self._op_fp32_instruction(skill_name, op, decision))
        elif skill_name == _BOUNDARY_BF16:
            first = int(decision.config.get("first_blocks", 2))
            last = int(decision.config.get("last_blocks", 3))
            patched_config["boundary_first_blocks"] = first
            patched_config["boundary_last_blocks"] = last
            patched_config["boundary_precision"] = decision.config.get("precision", "bf16")
            instructions.append(self._boundary_instruction({
                "boundary_first_blocks": first,
                "boundary_last_blocks": last,
                "boundary_precision": decision.config.get("precision", "bf16"),
            }))
        else:
            for k, v in decision.config.items():
                patched_config["repair_" + k] = v
            instructions.append(
                "Apply repair skill '" + skill_name + "' with config "
                + repr(decision.config) + "."
            )

    # -- grounded instruction text ----------------------------------------
    def _op_fp32_instruction(self, skill_name: str, op: str, decision: GovernanceDecision) -> str:
        cast_lines = {
            "adaln_fp32": (
                "  AdaLN modulation (the critical black-frame fix): cast the six\n"
                "    modulation tensors scale_msa/shift_msa/gate_msa and\n"
                "    scale_mlp/shift_mlp/gate_mlp to fp32, cast x to fp32, and compute\n"
                "    rms_norm(x)*(1.0+scale)+shift in fp32. The (1+scale) step MUST be\n"
                "    fp32 to avoid catastrophic cancellation when scale ~= -1 (bf16\n"
                "    7-bit mantissa). Self-attn runs fp32; MLP modulated input casts\n"
                "    back to x.dtype before ff (faithful to LTX model.py)."
            ),
            "gelu_fp32": (
                "  GELU: F.gelu(x.float(), approximate='tanh').to(dtype=x.dtype)\n"
                "    when x.device.type == 'mps' (bf16 fused Metal kernel NaNs at\n"
                "    |x|>=15; fp16 x^3 overflows 65504 at |x|>40)."
            ),
            "rmsnorm_fp32": (
                "  RMSNorm: compute mean(x^2) and the rsqrt division in fp32 (fp16\n"
                "    square-sum overflows 65504 for |x|>256); cast input/output only."
            ),
            "softmax_fp32": (
                "  Softmax / attention scores: run the (x - max) reduction and exp\n"
                "    in fp32 (MPS large-tensor softmax NaN, PyTorch #96602; fp16 exp\n"
                "    underflow zeros rows when the score gap > ~11)."
            ),
            "vae_fp32": (
                "  VAE decode: keep the full VAE decode in fp16/fp32 (do NOT co-quantize\n"
                "    the VAE with the DiT); GroupNorm/upsample key steps in fp32. The\n"
                "    VAE is the quality hard floor (fp8 VAE shows visible artifacts)."
            ),
        }
        body = cast_lines.get(skill_name, "  Force fp32 intermediate computation for " + op + ".")
        return (
            "REPAIR: " + skill_name + " -- op-level fp32 guard for " + op + "\n"
            "  Principle (robustness report section 7): on a low-precision backend,\n"
            "  force fp32 intermediate computation for sensitive ops, then cast the\n"
            "  boundary tensor back to the backend precision.\n"
            + body + "\n"
            "  Locate the site: grep -n '" + op + "\\|scale_msa\\|gate_msa\\|gelu' "
            "comfy/ldm/<model>/model.py\n"
            "  Verify: --force-fp32 renders correctly; --fp16-unet black before patch,\n"
            "  correct after; no 'invalid value encountered in cast' log."
        )

    def _boundary_instruction(self, cfg: dict[str, Any]) -> str:
        first = int(cfg.get("boundary_first_blocks", cfg.get("first_blocks", 2)))
        last = int(cfg.get("boundary_last_blocks", cfg.get("last_blocks", 3)))
        prec = cfg.get("boundary_precision", cfg.get("precision", "bf16"))
        return (
            "REPAIR: boundary_block_bf16 -- keep boundary transformer blocks high-precision\n"
            "  Keep the first " + str(first) + " + last " + str(last) + " transformer blocks"
            " in " + prec + " while the interior runs int8/HiF8.\n"
            "  Grounding: Wan2.1-T2V-14B HiFloat8 keeps the first 2 + last 3 blocks in bf16\n"
            "  (arXiv:2606.00957); Ideogram 4.0 INT8 protects a 'high-vulnerability layer\n"
            "  set' in bf16 (arXiv:2606.12280). This is the block-level granularity of the\n"
            "  same keep-high-precision principle as the op-level fp32 guard.\n"
            "  Config: boundary_first_blocks=" + str(first)
            + ", boundary_last_blocks=" + str(last)
            + ", boundary_precision=" + str(prec) + "."
        )

    # -- summary -----------------------------------------------------------
    def _summarize(
        self, decisions: list[GovernanceDecision], applied: list[str],
    ) -> str:
        if not decisions:
            return "No repair decisions to apply; patched config carries only overrides."
        lines = ["Applied " + str(len(decisions)) + " repair decision(s):"]
        lines.extend("  - " + a for a in applied)
        return "\n".join(lines)
