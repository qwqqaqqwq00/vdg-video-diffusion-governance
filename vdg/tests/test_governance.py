"""Tests for the governance layer (pipeline, rules, policy, agents).

Verifies:
  * GovernancePipeline.run returns a populated GovernanceReport (baseline +
    final result, alternatives, rationale, feasible flag).
  * The accel selector respects the energy budget: a tight budget makes the
    rule engine prefer step distillation (the largest energy lever) and the
    selector's top combo reduces energy below the baseline.
  * The rule engine disables SageAttention on Apple Silicon (CUDA+Triton only).
  * M4 Max + LTX-2.3 produces repair recommendations (the probe flags the
    AdaLN bf16 catastrophic cancellation, which is host-independent).

Host-independent (pure simulation; the AdaLN bf16 divergence is a precision
property, not a host property, so it fires on any machine).
"""
from __future__ import annotations

import pytest

from vdg import BUILTIN_SCENARIOS, DeviceCategory
from vdg.core.simulator import AgentContext, PerformanceEnergySimulator
from vdg.devices import get_device
from vdg.governance import GovernancePipeline, GovernanceReport, Policy, RuleEngine
from vdg.governance.rules import SKILL_DISTILL, SKILL_SAGE
from vdg.loads import LTX_2_3
from vdg.agents.accel_selector import AccelSelectorAgent
from vdg.agents.diagnostic import DiagnosticAgent

SCENARIO = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")


@pytest.fixture(autouse=True)
def _force_simulated_probe(monkeypatch):
    """Force the NumericalProbe SIMULATED path so governance tests are
    host-independent (no dependence on torch / MPS / CUDA availability). The
    simulated path encodes the documented divergence thresholds; the real torch
    path is a host optimization and would make these tests non-deterministic."""
    from vdg.skills.repair.numerical_probe import NumericalProbe
    monkeypatch.setattr(
        NumericalProbe, "probe_ops",
        lambda self, device_name="mps", precision="bf16":
            self._probe_simulated(device_name, precision),
    )


# ---------------------------------------------------------------------------
# GovernancePipeline.run returns a report
# ---------------------------------------------------------------------------
def test_pipeline_run_returns_report():
    pipe = GovernancePipeline()
    report = pipe.run("RTX4090", "LTX_2_3", SCENARIO)
    assert isinstance(report, GovernanceReport)
    assert report.device_name == "RTX4090"
    assert report.load_name == "LTX_2_3"
    assert report.scenario_name == SCENARIO.name
    # Baseline + final simulation results are populated.
    assert report.baseline_result is not None
    assert report.final_result is not None
    assert report.baseline_result.latency_s > 0
    assert report.final_result.latency_s > 0
    # Alternatives enumerated and ranked.
    assert isinstance(report.alternatives, list) and len(report.alternatives) > 0
    assert report.top_combo is not None
    assert isinstance(report.rationale, str) and report.rationale
    assert isinstance(report.feasible, bool)
    assert isinstance(report.summary(), str) and "Governance" in report.summary()


def test_pipeline_run_with_resolved_instances():
    pipe = GovernancePipeline()
    report = pipe.run_with(get_device("RTX5090"), LTX_2_3(), SCENARIO)
    assert isinstance(report, GovernanceReport)
    assert report.final_result is not None
    # RTX 5090 is Blackwell -> quantization (incl. the NVFP4-capable skill) is
    # applicable and appears among the evaluated combos. The selector enumerates
    # each config variant as a distinct candidate (quantization:gguf_q4,
    # quantization:nvfp4, quantization:int8), so NVFP4 is a real Pareto point,
    # not merely modeled via predict.
    quant_combos = [
        a for a in report.alternatives if "quantization" in a["skills"]
    ]
    assert len(quant_combos) > 0, "no quantization combo enumerated on RTX 5090"


def test_pipeline_unknown_device_raises():
    pipe = GovernancePipeline()
    with pytest.raises(KeyError):
        pipe.run("nonexistent_device", "LTX_2_3", SCENARIO)


def test_pipeline_unknown_load_raises():
    pipe = GovernancePipeline()
    with pytest.raises(KeyError):
        pipe.run("RTX4090", "nonexistent_load", SCENARIO)


def test_pipeline_final_improves_over_baseline():
    """The selected combo should not be worse than the baseline on latency."""
    pipe = GovernancePipeline()
    report = pipe.run("RTX4090", "LTX_2_3", SCENARIO)
    # The final run applies the top combo -> latency <= baseline (speedup >= 1).
    assert report.final_result.latency_s <= report.baseline_result.latency_s + 1e-9


# ---------------------------------------------------------------------------
# Rule engine: disable SageAttention on Apple Silicon (R1)
# ---------------------------------------------------------------------------
def test_rules_disable_sage_on_mps():
    engine = RuleEngine()
    dev = get_device("M4_Max")
    outcome = engine.apply(dev, LTX_2_3(), SCENARIO, Policy.from_scenario(SCENARIO, dev))
    assert SKILL_SAGE in outcome.disabled_skills
    assert "sliding_tile_attention" in outcome.disabled_skills
    # The rule emits a human-readable note.
    assert any("Apple Silicon" in n or "R1" in n for n in outcome.notes)


def test_rules_do_not_disable_sage_on_consumer_nv():
    engine = RuleEngine()
    dev = get_device("RTX4090")
    outcome = engine.apply(dev, LTX_2_3(), SCENARIO, Policy.from_scenario(SCENARIO, dev))
    assert SKILL_SAGE not in outcome.disabled_skills


# ---------------------------------------------------------------------------
# Rule engine: energy budget exceeded -> prefer step distillation (R2)
# ---------------------------------------------------------------------------
def test_rules_energy_budget_prefers_distill():
    """When the baseline energy exceeds the budget, R2 prefers step_distill."""
    engine = RuleEngine()
    dev = get_device("RTX4090")
    sim = PerformanceEnergySimulator()
    baseline = sim.simulate(dev, LTX_2_3(), skills_applied=[], config={
        "precision": "bf16", "attention_backend": "flash",
        "steps": SCENARIO.steps, "utilization": 0.75,
    }, scenario=SCENARIO)
    # Tight budget: a fraction of the baseline energy.
    tight = Policy.from_scenario(SCENARIO, dev, energy_budget_j=baseline.energy_j / 10.0)
    outcome = engine.apply(dev, LTX_2_3(), SCENARIO, tight, baseline)
    assert SKILL_DISTILL in outcome.preferred_skills
    assert any("R2" in n or "energy" in n.lower() for n in outcome.notes)


def test_rules_no_distill_when_budget_met():
    engine = RuleEngine()
    dev = get_device("RTX4090")
    sim = PerformanceEnergySimulator()
    baseline = sim.simulate(dev, LTX_2_3(), skills_applied=[], config={
        "precision": "bf16", "attention_backend": "flash",
        "steps": SCENARIO.steps, "utilization": 0.75,
    }, scenario=SCENARIO)
    # Generous budget: baseline is well within -> R2 does not fire.
    generous = Policy.from_scenario(SCENARIO, dev, energy_budget_j=baseline.energy_j * 100.0)
    outcome = engine.apply(dev, LTX_2_3(), SCENARIO, generous, baseline)
    assert SKILL_DISTILL not in outcome.preferred_skills


# ---------------------------------------------------------------------------
# Accel selector respects the energy budget
# ---------------------------------------------------------------------------
def test_accel_selector_reduces_energy_under_tight_budget():
    """Under a tight energy budget the selector's top combo reduces energy below
    the baseline (it picks a lower-energy combo, biased toward distillation)."""
    dev = get_device("RTX4090")
    load = LTX_2_3()
    sim = PerformanceEnergySimulator()
    baseline = sim.simulate(dev, load, skills_applied=[], config={
        "precision": "bf16", "attention_backend": "flash",
        "steps": SCENARIO.steps, "utilization": 0.75,
    }, scenario=SCENARIO)
    tight = Policy.from_scenario(SCENARIO, dev, energy_budget_j=baseline.energy_j / 10.0)
    selector = AccelSelectorAgent(simulator=sim)
    ctx = AgentContext(
        device=dev, load=load, scenario=SCENARIO,
        config={
            "preferred_skills": {SKILL_DISTILL},  # R2 would set this
            "policy": tight,
        },
    )
    out = selector.run(ctx)
    top = out["extra"]
    assert top["top_combo"] is not None
    top_energy = top["alternatives"][0]["energy_j"]
    assert top_energy < baseline.energy_j, (
        "top combo energy " + str(top_energy) + " >= baseline " + str(baseline.energy_j)
    )


def test_accel_selector_distill_in_alternatives_under_budget():
    """A distill combo is among the evaluated alternatives (energy lever)."""
    dev = get_device("RTX4090")
    load = LTX_2_3()
    sim = PerformanceEnergySimulator()
    baseline = sim.simulate(dev, load, skills_applied=[], config={
        "precision": "bf16", "attention_backend": "flash",
        "steps": SCENARIO.steps, "utilization": 0.75,
    }, scenario=SCENARIO)
    tight = Policy.from_scenario(SCENARIO, dev, energy_budget_j=baseline.energy_j / 10.0)
    selector = AccelSelectorAgent(simulator=sim)
    ctx = AgentContext(
        device=dev, load=load, scenario=SCENARIO,
        config={"preferred_skills": {SKILL_DISTILL}, "policy": tight},
    )
    out = selector.run(ctx)
    alts = out["extra"]["alternatives"]
    has_distill = any(
        "step_distill" in a["skills"] for a in alts
    )
    assert has_distill, "no step_distill combo in alternatives"


# ---------------------------------------------------------------------------
# M4 Max + LTX-2.3 produces repair recommendations
# ---------------------------------------------------------------------------
def test_m4max_ltx_produces_repair_recommendations():
    """The probe flags the AdaLN bf16 catastrophic cancellation (host-independent:
    -0.999 rounds to -1.0 in bf16 on any backend) -> repair recommended."""
    pipe = GovernancePipeline()
    report = pipe.run_with(get_device("M4_Max"), LTX_2_3(), SCENARIO)
    assert isinstance(report, GovernanceReport)
    assert len(report.repair_recommendations) > 0, "no repair recommendations on M4 Max"
    repair_names = {d.skill_name for d in report.repair_recommendations}
    # AdaLN divergence is a bf16 precision property -> always flagged.
    assert "adaln_fp32" in repair_names, "adaln_fp32 not recommended: " + str(repair_names)
    # The probe report is attached and reports a failure.
    assert report.probe_report is not None
    assert report.probe_report.has_failure is True
    # Patch instructions are emitted for the recommended repairs.
    assert len(report.patch_instructions) > 0


def test_m4max_ltx_disables_sage_in_pipeline():
    """The pipeline honors R1: SageAttention is disabled on Apple Silicon, so it
    never appears in the final selected skills."""
    pipe = GovernancePipeline()
    report = pipe.run_with(get_device("M4_Max"), LTX_2_3(), SCENARIO)
    selected = {d.skill_name for d in report.decisions}
    assert "sage_attention" not in selected


def test_datacenter_ltx_no_repair_recommendations():
    """A datacenter card (H100) reports no applicable repair skills (R1/low-precision
    guard does not fire on a datacenter category)."""
    pipe = GovernancePipeline()
    report = pipe.run_with(get_device("H100"), LTX_2_3(), SCENARIO)
    assert report.repair_recommendations == []


# ---------------------------------------------------------------------------
# Policy enforcement
# ---------------------------------------------------------------------------
def test_policy_enforce_feasible():
    dev = get_device("RTX4090")
    policy = Policy.from_scenario(SCENARIO, dev)
    sim = PerformanceEnergySimulator()
    res = sim.simulate(dev, LTX_2_3(), skills_applied=[], config={
        "precision": "bf16", "attention_backend": "flash",
        "steps": 4, "utilization": 0.75, "distilled": True,
    }, scenario=SCENARIO)
    # A 4-step distilled run should clear the latency SLO (120s) and budget.
    violations = policy.enforce(res)
    latency_violation = [v for v in violations if v.constraint == "latency_slo_s"]
    # Distilled 4-step is fast; latency should be within SLO.
    assert latency_violation == []


def test_policy_enforce_latency_violation():
    dev = get_device("RTX4090")
    policy = Policy(latency_slo_s=0.001, energy_budget_j=float("inf"),
                    quality_floor=0.0, max_memory_gb=float("inf"))
    sim = PerformanceEnergySimulator()
    res = sim.simulate(dev, LTX_2_3(), skills_applied=[], config={
        "precision": "bf16", "attention_backend": "flash",
        "steps": SCENARIO.steps, "utilization": 0.75,
    }, scenario=SCENARIO)
    violations = policy.enforce(res)
    assert any(v.constraint == "latency_slo_s" for v in violations)
    assert policy.is_feasible(res) is False


def test_policy_violation_message():
    """A breached SLO produces a human-readable Violation message."""
    dev = get_device("RTX4090")
    sim = PerformanceEnergySimulator()
    res = sim.simulate(dev, LTX_2_3(), skills_applied=[], config={
        "precision": "bf16", "attention_backend": "flash",
        "steps": SCENARIO.steps, "utilization": 0.75,
    }, scenario=SCENARIO)
    violations = Policy(latency_slo_s=0.001).enforce(res)
    assert violations
    msg = str(violations[0])
    assert "Latency" in msg or "exceeds" in msg


# ---------------------------------------------------------------------------
# Diagnostic agent
# ---------------------------------------------------------------------------
def test_diagnostic_agent_runs_probe_and_reports():
    agent = DiagnosticAgent()
    ctx = AgentContext(device=get_device("M4_Max"), load=LTX_2_3(), scenario=SCENARIO,
                       config={"precision": "bf16"})
    out = agent.run(ctx)
    assert out["agent"] == "diagnostic"
    report = out["extra"]["probe_report"]
    assert report is not None
    # bf16 always flags the AdaLN cancellation -> at least one repair recommended.
    assert report.has_failure is True
    decisions = out["decisions"]
    assert any(d.skill_name == "adaln_fp32" for d in decisions)
