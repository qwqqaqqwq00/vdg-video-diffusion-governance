"""GovernancePipeline: orchestrates the multi-agent governance loop.

GovernancePipeline.run(device_name, load_name, scenario) resolves the
device and load from the registry, then runs:

  resolve -> diagnose -> rules -> select accel -> repair -> simulate -> report

producing a GovernanceReport. run_with accepts already-resolved device
and load instances (useful for tests and direct programmatic use).

The pipeline is the single entry point the CLI vdg govern subcommand calls.
Each stage's structured output feeds the next via a shared config dict; the
final report carries the decisions, the final + baseline simulation results,
the ranked alternatives, the repair recommendations and patch instructions,
and the policy violations.

Simulator override: the constructor accepts any PerformanceEnergySimulator
(including ``core.calibration.CalibratedSimulator``) and threads it through
the accel-selector and simulator agents, so a governance run can use
anchor-calibrated predictions end to end::

    from vdg.core.calibration import CalibratedSimulator
    report = GovernancePipeline(simulator=CalibratedSimulator()).run(...)

Default (``simulator=None``) builds the base roofline simulator -- the
historic behavior, unchanged.

Import note: the agent classes are imported lazily inside GovernancePipeline.
__init__ (and only named under TYPE_CHECKING for annotations) to break
the agents<->governance import cycle (agents.diagnostic imports
governance.rules for the canonical skill-name constants; this module imports
the agents). Both packages still fully expose all classes via their
__init__ files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.contracts import GovernanceDecision, DeviceProfile, LoadModel
from ..core.registry import REGISTRY
from ..core.scenario import Scenario
from ..core.simulator import (
    AgentContext,
    PerformanceEnergySimulator,
    SimulationResult,
)
from .policy import Policy, Violation
from .rules import RuleEngine, RuleOutcome

if TYPE_CHECKING:  # annotations only -- never imported at runtime (cycle-safe)
    from ..agents.diagnostic import DiagnosticAgent, DiagnosticReport
    from ..agents.accel_selector import AccelSelectorAgent
    from ..agents.repair_agent import RepairAgent
    from ..agents.simulator_agent import SimulatorAgent

__all__ = ["GovernanceReport", "GovernancePipeline"]


@dataclass
class GovernanceReport:
    """Final output of a governance run."""

    device_name: str
    load_name: str
    scenario_name: str
    decisions: list[GovernanceDecision] = field(default_factory=list)
    final_result: SimulationResult | None = None
    baseline_result: SimulationResult | None = None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    repair_recommendations: list[GovernanceDecision] = field(default_factory=list)
    probe_summary: str = ""
    probe_report: "DiagnosticReport | None" = None
    probe_mode: str = ""  # "simulated" | "measured" -- structured audit field
    rule_notes: list[str] = field(default_factory=list)
    patched_config: dict[str, Any] = field(default_factory=dict)
    patch_instructions: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    top_combo: str | None = None
    policy: Policy | None = None
    feasible: bool = False

    def summary(self) -> str:
        """One-paragraph human-readable summary for CLI/report output."""
        lines = [
            "VDG Governance Report",
            "  device: " + self.device_name,
            "  load: " + self.load_name,
            "  scenario: " + self.scenario_name,
        ]
        if self.baseline_result is not None:
            b = self.baseline_result
            lines.append(
                "  baseline: latency " + format(b.latency_s, ".2f") + "s, energy "
                + format(b.energy_j, ".0f") + "J, quality " + format(b.quality_score, ".2f")
            )
        if self.final_result is not None:
            f = self.final_result
            speedup = (
                self.baseline_result.latency_s / f.latency_s
                if self.baseline_result and f.latency_s > 0 else 0.0
            )
            lines.append(
                "  final:    latency " + format(f.latency_s, ".2f") + "s ("
                + format(speedup, ".1f") + "x), energy " + format(f.energy_j, ".0f")
                + "J, quality " + format(f.quality_score, ".2f") + " [" + f.pareto_tag + "]"
            )
        lines.append("  top combo: " + str(self.top_combo))
        lines.append("  feasible: " + str(self.feasible))
        if self.violations:
            lines.append("  violations: " + str(len(self.violations)))
            for v in self.violations:
                lines.append("    - " + str(v))
        if self.repair_recommendations:
            names = ", ".join(d.skill_name for d in self.repair_recommendations)
            lines.append("  repair: " + names)
        if self.alternatives:
            lines.append("  alternatives: " + str(len(self.alternatives)) + " combos evaluated")
        if self.patch_instructions:
            lines.append("  patch instructions: " + str(len(self.patch_instructions)) + " block(s)")
        return "\n".join(lines)


class GovernancePipeline:
    """Orchestrates diagnose -> rules -> select -> repair -> simulate."""

    def __init__(
        self,
        simulator: PerformanceEnergySimulator | None = None,
        diagnostic: "DiagnosticAgent | None" = None,
        accel_selector: "AccelSelectorAgent | None" = None,
        repair: "RepairAgent | None" = None,
        simulator_agent: "SimulatorAgent | None" = None,
        rule_engine: RuleEngine | None = None,
    ) -> None:
        # Lazy import breaks the agents<->governance import cycle.
        from ..agents.diagnostic import DiagnosticAgent
        from ..agents.accel_selector import AccelSelectorAgent
        from ..agents.repair_agent import RepairAgent
        from ..agents.simulator_agent import SimulatorAgent

        self.simulator = simulator or PerformanceEnergySimulator()
        self.diagnostic_agent = diagnostic or DiagnosticAgent()
        self.accel_agent = accel_selector or AccelSelectorAgent(simulator=self.simulator)
        self.repair_agent = repair or RepairAgent()
        self.simulator_agent = simulator_agent or SimulatorAgent(simulator=self.simulator)
        self.rule_engine = rule_engine or RuleEngine()

    # -- public API --------------------------------------------------------
    def run(
        self, device_name: str, load_name: str, scenario: Scenario,
        energy_budget_j: float | None = None,
        latency_slo_s: float | None = None,
        quality_floor: float | None = None,
        max_memory_gb: float | None = None,
        sim_probe: bool = True,
    ) -> GovernanceReport:
        """Resolve device/load by name, then run the full governance loop.

        ``sim_probe`` defaults to True so governance planning uses the
        NumericalProbe SIMULATED path (reproducible documented divergence
        thresholds); pass False (CLI ``--real-probe``) to measure the live host.
        """
        device_cls = REGISTRY.get("device", device_name)
        load_cls = REGISTRY.get("load", load_name)
        if device_cls is None:
            raise KeyError(
                "Unknown device: " + repr(device_name)
                + ". Registered: " + ", ".join(sorted(REGISTRY.names("device")))
                + ". (Install device plugins in vdg/devices.)"
            )
        if load_cls is None:
            raise KeyError(
                "Unknown load: " + repr(load_name)
                + ". Registered: " + ", ".join(sorted(REGISTRY.names("load")))
                + ". (Install load plugins in vdg/loads.)"
            )
        device = device_cls()
        load = load_cls()
        return self.run_with(
            device, load, scenario,
            device_name=device_name, load_name=load_name,
            energy_budget_j=energy_budget_j, latency_slo_s=latency_slo_s,
            quality_floor=quality_floor, max_memory_gb=max_memory_gb,
            sim_probe=sim_probe,
        )

    def run_with(
        self, device: DeviceProfile, load: LoadModel, scenario: Scenario,
        device_name: str | None = None, load_name: str | None = None,
        energy_budget_j: float | None = None, latency_slo_s: float | None = None,
        quality_floor: float | None = None, max_memory_gb: float | None = None,
        policy: Policy | None = None, sim_probe: bool = True,
    ) -> GovernanceReport:
        """Run the governance loop on already-resolved device/load instances."""
        device_name = device_name or device.spec().name
        load_name = load_name or load.characteristics().model_name
        if policy is None:
            policy = Policy.from_scenario(
                scenario, device,
                energy_budget_j=energy_budget_j, latency_slo_s=latency_slo_s,
                quality_floor=quality_floor, max_memory_gb=max_memory_gb,
            )

        # 1. Baseline (unskilled) simulation -- feeds rule R2 (energy budget).
        base_config = self._base_config(device, scenario)
        baseline = self.simulator.simulate(
            device, load, skills_applied=[], config=base_config, scenario=scenario,
        )

        # 2. Diagnose -- numerical probe + repair recommendations.
        diag_ctx = AgentContext(device, load, scenario, config=dict(base_config))
        diag_ctx.config["sim_probe"] = sim_probe
        diag_out = self.diagnostic_agent.run(diag_ctx)
        repair_from_diag: list[GovernanceDecision] = list(diag_out.get("decisions", []))
        probe_report = diag_out.get("extra", {}).get("probe_report")

        # 3. Rules -- disable/prefer skills, config overrides, repair decisions.
        outcome: RuleOutcome = self.rule_engine.apply(
            device, load, scenario, policy, baseline,
        )
        repair_from_rules = list(outcome.decisions)

        # 4. Select accel -- enumerate combos, simulate, Pareto-rank.
        accel_ctx = AgentContext(device, load, scenario, config={
            "disabled_skills": set(outcome.disabled_skills),
            "preferred_skills": set(outcome.preferred_skills),
            "config_overrides": dict(outcome.config_overrides),
            "policy": policy,
        })
        accel_out = self.accel_agent.run(accel_ctx)
        accel_decisions: list[GovernanceDecision] = list(accel_out.get("decisions", []))
        top_combo = accel_out.get("extra", {}).get("top_combo")
        top_skills = list(accel_out.get("extra", {}).get("top_skills", []))
        top_config = dict(accel_out.get("extra", {}).get("top_config", {}) or {})
        alternatives = list(accel_out.get("extra", {}).get("alternatives", []))

        # 5. Repair -- apply repair decisions, produce patched config + instructions.
        repair_decisions = repair_from_diag + repair_from_rules
        repair_ctx = AgentContext(device, load, scenario, config={
            "repair_decisions": repair_decisions,
            "config_overrides": dict(outcome.config_overrides),
        })
        repair_out = self.repair_agent.run(repair_ctx)
        patched_config = dict(repair_out.get("extra", {}).get("patched_config", {}))
        patch_instructions = list(repair_out.get("extra", {}).get("patch_instructions", []))
        repair_skills = list(repair_out.get("extra", {}).get("repair_skills", []))

        # 6. Simulate -- final authoritative run with selected accel skills +
        #    repair skills (their predict() impacts model the repair cost) and
        #    the patched config (e.g. vae_precision=fp32 from vae_fp32).
        sim_ctx = AgentContext(device, load, scenario, config={
            "selected_skills": list(top_skills) + list(repair_skills),
            "selected_skill_names": [d.skill_name for d in accel_decisions]
                                      + [d.skill_name for d in repair_decisions],
            "combo_config": top_config,
            "patched_config": patched_config,
            "config_overrides": dict(outcome.config_overrides),
            "policy": policy,
        })
        sim_out = self.simulator_agent.run(sim_ctx)
        final_result: SimulationResult = sim_out["extra"]["result"]
        violations = list(sim_out["extra"].get("violations", []))

        # 7. Report.
        all_decisions = list(accel_decisions) + list(repair_decisions)
        feasible = len(violations) == 0
        rationale = self._rationale(
            device, load, scenario, policy, baseline, final_result, outcome,
            repair_decisions, top_combo, probe_report,
        )
        return GovernanceReport(
            device_name=device_name,
            load_name=load_name,
            scenario_name=scenario.name,
            decisions=all_decisions,
            final_result=final_result,
            baseline_result=baseline,
            alternatives=alternatives,
            rationale=rationale,
            repair_recommendations=repair_decisions,
            probe_summary=(probe_report.summary if probe_report is not None else ""),
            probe_report=probe_report,
            probe_mode=("simulated" if (probe_report is not None and probe_report.simulated)
                        else ("measured" if probe_report is not None else "")),
            rule_notes=list(outcome.notes),
            patched_config=patched_config,
            patch_instructions=patch_instructions,
            violations=violations,
            top_combo=top_combo,
            policy=policy,
            feasible=feasible,
        )

    # -- helpers -----------------------------------------------------------
    def _base_config(self, device: DeviceProfile, scenario: Scenario) -> dict[str, Any]:
        backends = device.spec().attention_backends
        backend = "math"
        if backends:
            for preferred in ("flash", "sdpa", "mlx_sdpa", "sage2", "triton"):
                if preferred in backends:
                    backend = preferred
                    break
            else:
                backend = backends[0]
        return {
            "precision": "bf16",
            "attention_backend": backend,
            "steps": scenario.steps,
            "utilization": 0.75,
        }

    def _rationale(
        self, device, load, scenario, policy, baseline, final, outcome,
        repair_decisions, top_combo, probe_report,
    ) -> str:
        speedup = baseline.latency_s / final.latency_s if final.latency_s > 0 else 0.0
        parts = [
            "Governance run for " + device.spec().name + " / " + load.characteristics().model_name
            + " / " + scenario.name + ".",
            "Baseline latency " + format(baseline.latency_s, ".2f") + "s, energy "
            + format(baseline.energy_j, ".0f") + "J.",
        ]
        if probe_report is not None:
            parts.append(
                "Probe: " + ("divergence detected" if probe_report.has_failure
                             else "no divergence") + " ("
                + str(sum(1 for r in probe_report.results if r.status in ("divergence", "nan")))
                + "/" + str(len(probe_report.results)) + " ops divergent)."
            )
        if outcome.notes:
            parts.append("Rules: " + " | ".join(outcome.notes))
        if top_combo:
            parts.append(
                "Selected '" + str(top_combo) + "': latency " + format(final.latency_s, ".2f")
                + "s (" + format(speedup, ".1f") + "x), quality "
                + format(final.quality_score, ".2f") + "."
            )
        if repair_decisions:
            parts.append(
                "Repair: " + ", ".join(d.skill_name for d in repair_decisions) + "."
            )
        if final.warnings:
            parts.append("Warnings: " + "; ".join(final.warnings))
        return " ".join(parts)
