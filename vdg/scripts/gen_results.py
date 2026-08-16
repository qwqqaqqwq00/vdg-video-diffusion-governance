"""Generate results.json: grounded simulation data for docs/TEST_REPORT.md.

Runs the REAL VDG simulator + governance pipeline over the primary load
(LTX-2.3) and the built-in scenarios across the four headline device classes
(RTX 4090, RTX 5090, M4 Max, Jetson Thor T5000), plus an energy comparison
(joules per 480p/81f clip) and a skill-impact sweep. Numbers are computed by
the shipped code, not hand-written. Re-run anytime:

    python scripts/gen_results.py
"""
from __future__ import annotations

import json
import os
import sys

# Ensure the package is importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import vdg  # noqa: F401  (triggers auto-registration)
from vdg.core.registry import REGISTRY
from vdg.core.scenario import BUILTIN_SCENARIOS
from vdg.core.simulator import PerformanceEnergySimulator
from vdg.governance.pipeline import GovernancePipeline

REGISTRY.discover()

SIM = PerformanceEnergySimulator()


def _load(name):
    return REGISTRY.get("load", name)()


def _device(name):
    return REGISTRY.get("device", name)()


def _round(obj, nd=3):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, nd) for v in obj]
    return obj


def energy_comparison():
    """Joules per 480p/81f LTX-2.3 clip across the four headline devices."""
    scen = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    rows = []
    for dev_name in ("RTX4090", "RTX5090", "M4_Max", "Jetson_Thor_T5000"):
        dev = _device(dev_name)
        for steps, label in ((30, "baseline_30step"), (4, "distilled_4step")):
            cfg = {
                "precision": "bf16",
                "steps": steps,
                "utilization": 0.75,
                "distilled": steps <= 8,
            }
            # Pick a backend the device actually lists.
            backends = dev.spec().attention_backends
            cfg["attention_backend"] = next(
                (b for b in ("flash", "mlx_sdpa") if b in backends), backends[0] if backends else "math"
            )
            res = SIM.simulate(dev, _load("LTX_2_3"), skills_applied=[], config=cfg, scenario=scen)
            rows.append({
                "device": dev.spec().name,
                "device_class": dev.spec().category,
                "steps": steps,
                "label": label,
                "latency_s": res.latency_s,
                "energy_j": res.energy_j,
                "peak_memory_gb": res.peak_memory_gb,
                "quality": res.quality_score,
                "pareto_tag": res.pareto_tag,
                "tdp_w": dev.spec().tdp_w,
                "mem_bw_gbps": dev.spec().memory_bandwidth_gbps,
            })
    return {"scenario": scen.name, "model": "LTX-2.3", "rows": rows}


def scenario_matrix():
    """Baseline simulation per (device, scenario) for the headline devices."""
    out = []
    for dev_name in ("RTX4090", "RTX5090", "M4_Max", "Jetson_Thor_T5000"):
        dev = _device(dev_name)
        for scen in BUILTIN_SCENARIOS.all():
            cfg = {
                "precision": "bf16",
                "steps": scen.steps,
                "utilization": 0.75,
                "distilled": scen.steps <= 8,
            }
            backends = dev.spec().attention_backends
            cfg["attention_backend"] = next(
                (b for b in ("flash", "mlx_sdpa") if b in backends), backends[0] if backends else "math"
            )
            res = SIM.simulate(dev, _load("LTX_2_3"), skills_applied=[], config=cfg, scenario=scen)
            out.append({
                "device": dev.spec().name,
                "scenario": scen.name,
                "latency_s": res.latency_s,
                "energy_j": res.energy_j,
                "peak_memory_gb": res.peak_memory_gb,
                "quality": res.quality_score,
                "feasible": res.is_feasible(scen),
                "pareto_tag": res.pareto_tag,
            })
    return out


def skill_sweep():
    """Single-skill impact on RTX 5090 / LTX-2.3 / 480p-81f (baseline 30-step)."""
    scen = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    dev = _device("RTX5090")
    load = _load("LTX_2_3")
    base_cfg = {"precision": "bf16", "steps": 30, "utilization": 0.75, "attention_backend": "flash"}
    base = SIM.simulate(dev, load, skills_applied=[], config=base_cfg, scenario=scen)
    rows = [{"combo": "baseline", "latency_s": base.latency_s, "energy_j": base.energy_j,
             "quality": base.quality_score, "speedup": 1.0}]
    # Each entry: (combo_label, skill_name, skill_configs). The variant is
    # injected via skill_configs so predict() sees the real override (otherwise
    # the skill's default_config is used and e.g. sage v2/v3 collapse to v2).
    singles = [
        ("step_distill:4step", "step_distill", {"step_distill": {"steps": 4, "baseline_steps": 30}}),
        ("teacache:thr0.1", "teacache", {"teacache": {"threshold": 0.1}}),
        ("sage_attention:v2", "sage_attention", {"sage_attention": {"version": "v2"}}),
        ("sage_attention:v3", "sage_attention", {"sage_attention": {"version": "v3"}}),
        ("quantization:nvfp4", "quantization", {"quantization": {"method": "nvfp4"}}),
        ("quantization:gguf_q4", "quantization", {"quantization": {"method": "gguf_q4"}}),
        ("compile_graph:torch_compile", "compile_graph", {"compile_graph": {"backend": "torch_compile"}}),
        ("compile_graph:trt", "compile_graph", {"compile_graph": {"backend": "trt"}}),
        ("vae_tiling", "vae_tiling", {}),
        ("offload:0.5", "offload", {"offload": {"block_swap_ratio": 0.5}}),
    ]
    for label, name, sc in singles:
        cls = REGISTRY.get("skill", name)
        if cls is None:
            continue
        skill = cls()
        if not skill.applicable(dev, load):
            rows.append({"combo": label, "skipped": "not_applicable"})
            continue
        cfg = dict(base_cfg)
        cfg["distilled"] = name == "step_distill"
        if sc:
            cfg["skill_configs"] = sc
        res = SIM.simulate(dev, load, skills_applied=[skill], config=cfg, scenario=scen)
        rows.append({
            "combo": label,
            "skill": name,
            "latency_s": res.latency_s,
            "energy_j": res.energy_j,
            "quality": res.quality_score,
            "peak_memory_gb": res.peak_memory_gb,
            "speedup": round(base.latency_s / res.latency_s, 3) if res.latency_s > 0 else 0,
        })
    return {"device": "RTX 5090", "model": "LTX-2.3", "scenario": scen.name, "rows": rows}


def governance_runs():
    """Full governance pipeline runs for the canonical use-case devices."""
    out = []
    pipe = GovernancePipeline()
    cases = [
        ("RTX5090", "LTX_2_3", "ltx_t2v_480p_81f"),
        ("RTX4090", "LTX_2_3", "ltx_t2v_480p_81f"),
        ("M4_Max", "LTX_2_3", "ltx_t2v_480p_81f"),
        ("Jetson_Thor_T5000", "LTX_2_3", "edge_npu_shortclip"),
        ("RTX5090", "LTX_2_3", "long_video_1025f"),
    ]
    for dev_name, load_name, scen_name in cases:
        scen = BUILTIN_SCENARIOS.get(scen_name)
        report = pipe.run(dev_name, load_name, scen)
        out.append({
            "device": dev_name,
            "load": load_name,
            "scenario": scen_name,
            "baseline_latency_s": report.baseline_result.latency_s if report.baseline_result else None,
            "baseline_energy_j": report.baseline_result.energy_j if report.baseline_result else None,
            "final_latency_s": report.final_result.latency_s if report.final_result else None,
            "final_energy_j": report.final_result.energy_j if report.final_result else None,
            "final_quality": report.final_result.quality_score if report.final_result else None,
            "top_combo": report.top_combo,
            "feasible": report.feasible,
            "violations": [str(v) for v in report.violations],
            "repair": [d.skill_name for d in report.repair_recommendations],
            "alternatives": len(report.alternatives),
        })
    return out


def probe_table():
    """NumericalProbe divergence status across device/precision pairs."""
    from vdg.skills.repair.numerical_probe import NumericalProbe
    out = []
    for dev_name, probe_name in (("M4_Max", "mps"), ("RTX5090", "cuda"),
                                  ("RTX4090", "cuda"), ("Jetson_Thor_T5000", "Jetson AGX Thor T5000")):
        for prec in ("bf16", "fp16"):
            rep = NumericalProbe().probe_ops(probe_name, prec)
            out.append({
                "device": dev_name,
                "probe_target": probe_name,
                "precision": prec,
                "simulated": rep.simulated,
                "has_failure": rep.has_failure,
                "failures": [{"op": r.op, "status": r.status, "max_diff": r.max_diff}
                             for r in rep.results if r.status in ("divergence", "nan")],
                "suggested": rep.repair_skills_suggested,
            })
    return out


def main():
    data = {
        "platform": "VDG (Video Diffusion Governance)",
        "version": vdg.__version__,
        "primary_model": "LTX-2.3",
        "energy_comparison_480p_81f": _round(energy_comparison(), 3),
        "scenario_matrix": _round(scenario_matrix(), 3),
        "skill_sweep_5090": _round(skill_sweep(), 3),
        "governance_runs": _round(governance_runs(), 3),
        "probe_table": _round(probe_table(), 4),
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "results.json")
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print("wrote " + os.path.abspath(out_path))
    # Echo a compact summary so the doc author has the numbers inline.
    print("\n-- energy comparison (480p/81f LTX-2.3) --")
    for r in data["energy_comparison_480p_81f"]["rows"]:
        print("  %-22s %-18s latency=%7.2fs energy=%8.0fJ mem=%5.2fGB q=%.1f"
              % (r["device"], r["label"], r["latency_s"], r["energy_j"], r["peak_memory_gb"], r["quality"]))
    print("\n-- skill sweep (RTX5090 / LTX-2.3 / 480p-81f) --")
    for r in data["skill_sweep_5090"]["rows"]:
        if "skipped" in r:
            print("  %-18s skipped" % r["combo"])
        else:
            print("  %-18s latency=%7.2fs energy=%8.0fJ q=%.2f speedup=%.2fx"
                  % (r["combo"], r["latency_s"], r["energy_j"], r["quality"], r["speedup"]))


if __name__ == "__main__":
    main()
