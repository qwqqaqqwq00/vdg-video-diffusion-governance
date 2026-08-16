"""SimulatorAgent: runs the final simulation with the selected skills.

The simulator agent is the last step of the governance pipeline. It takes the
skill combination selected by the accel selector and the patched config
produced by the repair agent (merged with any rule-engine overrides) and runs
one authoritative PerformanceEnergySimulator.simulate call, returning the
final SimulationResult plus the policy violations for the report.

It is intentionally thin: the heavy lifting (combo enumeration, repair patching)
happens upstream. Its job is to produce the single, authoritative final result
the GovernanceReport carries.
"""
from __future__ import annotations

from typing import Any

from ..core.contracts import GovernanceDecision, Skill
from ..core.registry import REGISTRY
from ..core.simulator import (
    AgentContext,
    PerformanceEnergySimulator,
    SimulationResult,
)
from .base import GovernanceAgent
from ..governance.policy import Policy

__all__ = ["SimulatorAgent"]


class SimulatorAgent(GovernanceAgent):
    """Runs the final simulation with selected skills + patched config."""

    name = "simulator"
    role = "simulate"

    def __init__(
        self,
        name: str | None = None,
        role: str | None = None,
        simulator: PerformanceEnergySimulator | None = None,
    ) -> None:
        super().__init__(name, role)
        self.simulator = simulator or PerformanceEnergySimulator()
        self.last_result: SimulationResult | None = None

    def run(self, context: AgentContext) -> dict[str, Any]:
        device = context.device
        load = context.load
        scenario = context.scenario
        cfg = dict(context.config or {})

        # Selected skills: either explicit Skill instances or names resolved
        # from the registry (by the accel selector's decisions).
        skills: list[Skill] = list(cfg.get("selected_skills", []) or [])
        selected_names: list[str] = list(cfg.get("selected_skill_names", []) or [])
        if not skills and selected_names:
            for nm in selected_names:
                cls = REGISTRY.get("skill", nm)
                if cls is not None:
                    try:
                        inst = cls()
                        if inst.applicable(device, load):
                            skills.append(inst)
                    except Exception:
                        continue

        # Build the final config: start from the scenario base, then layer the
        # SELECTED COMBO's operating point (attention_backend, steps, precision,
        # skill_configs) so the final run matches the accel selector's pick, then
        # the repair patched_config (vae_precision, guard flags) on top.
        config: dict[str, Any] = {
            "precision": "bf16",
            "attention_backend": self._default_backend(device),
            "steps": scenario.steps,
            "utilization": 0.75,
        }
        combo_config = dict(cfg.get("combo_config", {}) or {})
        if combo_config:
            # Apply the selected combo's operating point. Steps are deliberately
            # NOT taken from the combo: the accel selector simulates every
            # candidate at scenario.steps and models distillation's step
            # reduction via the skill's marginal speedup, so the final run must
            # also use scenario.steps to stay consistent with the ranked
            # alternatives (taking combo steps here would re-introduce a
            # double-count for distill combos).
            sc = combo_config.get("skill_configs")
            if sc:
                config["skill_configs"] = dict(sc)
            ab = combo_config.get("attention_backend")
            if ab:
                config["attention_backend"] = ab
            pc = combo_config.get("precision")
            if pc and pc != "bf16":
                config["precision"] = pc
        patched = dict(cfg.get("patched_config", {}) or {})
        # The repair patch carries guard flags + vae_precision; it must NOT
        # clobber the combo's operating point (backend/steps/precision/skill_configs).
        _op_keys = ("skill_configs", "attention_backend", "steps", "precision")
        for k, v in patched.items():
            if k in _op_keys:
                continue
            config[k] = v
        for k, v in dict(cfg.get("config_overrides", {}) or {}).items():
            if k in ("quant_methods_allowed", "boundary_first_blocks", "boundary_last_blocks"):
                continue
            config.setdefault(k, v)
        # Distillation: its step reduction is encoded by the skill's marginal
        # speedup (baseline_steps in skill_configs), NOT by lowering config
        # steps here -- that would double-count the reduction (denoise reduced
        # by fewer steps AND latency /speedup). Mark the run distilled so the
        # simulator skips the naive-few-step quality penalty, and ensure
        # baseline_steps is present so the skill's predict() computes the
        # correct marginal speedup relative to the actual step count.
        has_distill = (
            any(s.registry_name() == "step_distill" for s in skills)
            or "step_distill" in selected_names
        )
        if has_distill:
            config["distilled"] = True
            sc = config.get("skill_configs")
            if not isinstance(sc, dict):
                sc = {}
            dc = dict(sc.get("step_distill", {}))
            dc.setdefault("baseline_steps", scenario.steps)
            sc["step_distill"] = dc
            config["skill_configs"] = sc

        result = self.simulator.simulate(
            device, load, skills_applied=skills, config=config, scenario=scenario,
        )
        self.last_result = result

        policy: Policy = cfg.get("policy") or Policy.from_scenario(scenario, device)
        violations = policy.enforce(result)

        decisions: list[GovernanceDecision] = []
        for s in skills:
            sc = s.default_config()
            impact = s.predict(device, load, sc)
            decisions.append(GovernanceDecision(
                skill_name=s.registry_name(),
                config=sc,
                predicted_impact=impact,
                rationale="Applied in final simulation.",
            ))

        notes = self._summarize(result, skills, violations, policy)
        return {
            "agent": self.name,
            "role": self.role,
            "decisions": decisions,
            "notes": notes,
            "extra": {
                "result": result,
                "violations": violations,
                "policy": policy,
                "skills": [s.registry_name() for s in skills],
                "config": config,
            },
        }

    # -- helpers -----------------------------------------------------------
    def _default_backend(self, device) -> str:
        backends = device.spec().attention_backends
        if backends:
            for preferred in ("flash", "sdpa", "mlx_sdpa", "sage2", "triton"):
                if preferred in backends:
                    return preferred
            return backends[0]
        return "math"

    def _summarize(
        self, result: SimulationResult, skills: list[Skill],
        violations, policy: Policy,
    ) -> str:
        lines = [
            "Final simulation with " + str(len(skills)) + " skill(s): "
            + (", ".join(s.registry_name() for s in skills) if skills else "none"),
            "  latency=" + format(result.latency_s, ".2f") + "s"
            + " energy=" + format(result.energy_j, ".0f") + "J"
            + " quality=" + format(result.quality_score, ".2f")
            + " peak_mem=" + format(result.peak_memory_gb, ".2f") + "GB"
            + " tag=" + result.pareto_tag,
        ]
        if violations:
            lines.append("  Policy violations (" + str(len(violations)) + "):")
            for v in violations:
                lines.append("    - " + str(v))
        else:
            lines.append("  Policy: all constraints satisfied.")
        if result.warnings:
            lines.append("  Warnings: " + "; ".join(result.warnings))
        return "\n".join(lines)
