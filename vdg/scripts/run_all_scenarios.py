#!/usr/bin/env python3
"""Run the full VDG governance pipeline across the heterogeneous scenario matrix.

Iterates >=6 deployment scenarios spanning every device category and the
headline acceleration strategies (NVFP4 on Blackwell, step distillation, INT8 on
edge NPU, Apple Silicon repair), runs the GovernancePipeline for each, and
writes a machine-readable summary to test_results/results.json.

This is PURE SIMULATION: it never touches a GPU, network, or torch runtime. The
governance pipeline + PerformanceEnergySimulator + NumericalProbe (simulated
fallback) are all numpy/stdlib, so the script runs offline on any machine.

Usage:
    python scripts/run_all_scenarios.py [--out test_results/results.json]

The results feed scripts/generate_report.py which renders docs/TEST_REPORT.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# --- path bootstrap: make 'import vdg' work even without an editable install --
_THIS = Path(__file__).resolve()
_SRC = _THIS.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vdg import BUILTIN_SCENARIOS, REGISTRY  # noqa: E402
from vdg.core.simulator import PerformanceEnergySimulator, SimulationResult  # noqa: E402
from vdg.governance import GovernancePipeline, GovernanceReport  # noqa: E402
from vdg.core.contracts import GovernanceDecision  # noqa: E402
from vdg.skills.repair.numerical_probe import NumericalProbe  # noqa: E402

# Force the NumericalProbe SIMULATED path so the scenario results are fully
# deterministic and host-independent (the "pure simulation" contract),
# regardless of whether torch + a real MPS/CUDA device are available on this
# host. The simulated path encodes the documented divergence thresholds from
# the cross-device robustness report; the real torch path is a host-specific
# optimization and would make results.json non-reproducible across machines.
NumericalProbe.probe_ops = (  # type: ignore[assignment]
    lambda self, device_name="mps", precision="bf16":
        self._probe_simulated(device_name, precision)
)


# ---------------------------------------------------------------------------
# Scenario matrix (device x model x workload x focus strategy)
# ---------------------------------------------------------------------------
# Each entry: (label, device_name, model_name, scenario_name, focus,
#              optional (skill_name, skill_config, precision) for a direct
#              focus simulation that captures the headline strategy's numbers).
SCENARIO_MATRIX: list[tuple[str, str, str, str, str, tuple | None]] = [
    (
        "H100 + LTX-2.3 + 720p",
        "H100", "LTX_2_3", "ltx_t2v_720p_129f", "720p high-res datacenter",
        None,
    ),
    (
        "RTX 5090 + LTX-2.3 + NVFP4",
        "RTX5090", "LTX_2_3", "ltx_t2v_480p_81f", "NVFP4 (Blackwell FP4)",
        ("quantization", {"method": "nvfp4"}, "nvfp4"),
    ),
    (
        "RTX 4090 + distill + LTX-2.3 + 480p",
        "RTX4090", "LTX_2_3", "ltx_t2v_480p_81f", "step distillation (4-step)",
        ("step_distill", {"steps": 4}, None),
    ),
    (
        "M4 Max + LTX-2.3 + 480p",
        "M4_Max", "LTX_2_3", "ltx_t2v_480p_81f", "Apple Silicon (repair)",
        None,
    ),
    (
        "Jetson Thor + LTX-2.3 distill",
        "Jetson_Thor_T5000", "LTX_2_3", "ltx_t2v_480p_81f", "edge distillation",
        ("step_distill", {"steps": 4}, None),
    ),
    (
        "Ascend 910B + LTX-2.3 + INT8",
        "Ascend_910B", "LTX_2_3", "ltx_t2v_480p_81f", "INT8 (edge NPU)",
        ("quantization", {"method": "int8"}, "int8"),
    ),
]


# ---------------------------------------------------------------------------
# Serialization helpers (dataclasses -> JSON-safe dicts)
# ---------------------------------------------------------------------------
def _result_dict(res: SimulationResult) -> dict[str, Any]:
    return {
        "latency_s": round(res.latency_s, 4),
        "energy_j": round(res.energy_j, 2),
        "peak_memory_gb": round(res.peak_memory_gb, 4),
        "quality": round(res.quality_score, 4),
        "throughput_tokens_s": round(res.throughput_tokens_s, 2),
        "pareto": res.pareto_tag,
        "tokens": res.tokens,
        "steps": res.steps,
        "precision": res.precision,
        "attention_backend": res.attention_backend,
        "warnings": list(res.warnings),
    }


def _decision_dict(d: GovernanceDecision) -> dict[str, Any]:
    imp = d.predicted_impact
    return {
        "skill": d.skill_name,
        "config": dict(d.config),
        "speedup": round(imp.speedup, 4),
        "memory_ratio": round(imp.memory_ratio, 4),
        "quality_delta": round(imp.quality_delta, 4),
        "energy_ratio": round(imp.energy_ratio, 4),
        "rationale": d.rationale,
    }


def _default_backend(device) -> str:
    backends = device.spec().attention_backends
    if backends:
        for preferred in ("flash", "sdpa", "mlx_sdpa", "sage2", "triton"):
            if preferred in backends:
                return preferred
        return backends[0]
    return "math"


# ---------------------------------------------------------------------------
# Focus simulation: a direct sim with the headline strategy's skill
# ---------------------------------------------------------------------------
def _focus_sim(device, load, scenario, skill_name, skill_config, precision):
    """Run a direct simulation with the focus skill applied (captures the
    headline strategy's numbers even when the selector's dedup hides the
    config variant from the ranked alternatives)."""
    cls = REGISTRY.get("skill", skill_name)
    if cls is None:
        return None
    skill = cls()
    if not skill.applicable(device, load):
        return {"applicable": False, "note": skill_name + " not applicable on " + device.spec().name}
    cfg: dict[str, Any] = {
        "precision": precision or "bf16",
        "attention_backend": _default_backend(device),
        "steps": scenario.steps,
        "utilization": 0.75,
        "skill_configs": {skill_name: dict(skill_config)},
    }
    if skill_name == "step_distill":
        sc = dict(cfg["skill_configs"]["step_distill"])
        sc.setdefault("baseline_steps", scenario.steps)
        cfg["skill_configs"]["step_distill"] = sc
        cfg["distilled"] = True
    sim = PerformanceEnergySimulator()
    res = sim.simulate(device, load, skills_applied=[skill], config=cfg, scenario=scenario)
    out = _result_dict(res)
    out["skill"] = skill_name
    out["applicable"] = True
    return out


# ---------------------------------------------------------------------------
# Run one scenario through the governance pipeline
# ---------------------------------------------------------------------------
def run_scenario(entry: tuple) -> dict[str, Any]:
    label, device_name, model_name, scenario_name, focus, focus_spec = entry
    scenario = BUILTIN_SCENARIOS.get(scenario_name)
    pipe = GovernancePipeline()
    report: GovernanceReport = pipe.run(device_name, model_name, scenario)

    baseline = _result_dict(report.baseline_result) if report.baseline_result else {}
    final = _result_dict(report.final_result) if report.final_result else {}
    speedup = (
        round(report.baseline_result.latency_s / report.final_result.latency_s, 3)
        if report.baseline_result and report.final_result and report.final_result.latency_s > 0
        else 0.0
    )

    record: dict[str, Any] = {
        "label": label,
        "device": device_name,
        "device_name": report.device_name,
        "model": model_name,
        "scenario": scenario_name,
        "resolution": list(scenario.resolution),
        "frames": scenario.frames,
        "focus": focus,
        "baseline": baseline,
        "final": final,
        "top_combo": report.top_combo,
        "speedup": speedup,
        "feasible": report.feasible,
        "violations": [
            {"constraint": v.constraint, "actual": round(v.actual, 4),
             "limit": round(v.limit, 4), "message": str(v)}
            for v in report.violations
        ],
        "repair_recommendations": [_decision_dict(d) for d in report.repair_recommendations],
        "decisions": [_decision_dict(d) for d in report.decisions],
        "alternatives_count": len(report.alternatives),
        "alternatives_top5": [
            {
                "combo": a["combo"],
                "latency_s": round(a["latency_s"], 4),
                "energy_j": round(a["energy_j"], 2),
                "quality": round(a["quality"], 4),
                "peak_memory_gb": round(a["peak_memory_gb"], 4),
                "pareto": a["pareto_tag"],
                "feasible": a["feasible"],
                "estimate_only": a["estimate_only"],
            }
            for a in report.alternatives[:5]
        ],
        "rule_notes": list(report.rule_notes),
        "patch_instructions_count": len(report.patch_instructions),
        "probe_summary": report.probe_summary,
    }

    # Optional direct focus simulation.
    if focus_spec is not None:
        sname, sconfig, sprec = focus_spec
        dev_cls = REGISTRY.get("device", device_name)
        load_cls = REGISTRY.get("load", model_name)
        record["focus_result"] = _focus_sim(dev_cls(), load_cls(), scenario, sname, sconfig, sprec)
    else:
        record["focus_result"] = None

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the VDG scenario matrix.")
    parser.add_argument("--out", default="test_results/results.json",
                        help="output JSON path (default: test_results/results.json)")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("VDG scenario matrix: running " + str(len(SCENARIO_MATRIX)) + " scenarios...")
    records = []
    for entry in SCENARIO_MATRIX:
        label = entry[0]
        print("  - " + label + " ...", flush=True)
        rec = run_scenario(entry)
        records.append(rec)
        fin = rec["final"]
        base = rec["baseline"]
        print(
            "      baseline " + format(base.get("latency_s", 0), ".2f") + "s / "
            + format(base.get("energy_j", 0), ".0f") + "J  ->  final "
            + format(fin.get("latency_s", 0), ".2f") + "s / "
            + format(fin.get("energy_j", 0), ".0f") + "J  (" + format(rec["speedup"], ".1f")
            + "x, " + ("feasible" if rec["feasible"] else "infeasible") + ")"
        )

    payload = {
        "platform": "VDG (Video Diffusion Governance)",
        "primary_model": "LTX-2.3",
        "scenario_count": len(records),
        "pure_simulation": True,
        "scenarios": records,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("")
    print("Wrote " + str(len(records)) + " scenario results to " + str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
