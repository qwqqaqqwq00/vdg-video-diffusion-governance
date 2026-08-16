#!/usr/bin/env python3
"""Render docs/TEST_REPORT.md from test_results/results.json.

Reads the scenario-matrix results produced by scripts/run_all_scenarios.py and
emits a Markdown report with:

  * a main results table (scenario x latency x energy x memory x quality x
    skills x pareto),
  * an energy comparison section (baseline vs final, % saved),
  * a focus-strategy table (the headline acceleration strategy per scenario),
  * a repair-recommendations summary,
  * per-scenario top-5 ranked alternatives.

Usage:
    python scripts/generate_report.py [--in test_results/results.json] [--out docs/TEST_REPORT.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _f(x: Any, digits: int = 2) -> str:
    """Format a number for a Markdown table cell (handles None / non-numeric)."""
    if x is None:
        return "-"
    try:
        return format(float(x), "." + str(digits) + "f")
    except (TypeError, ValueError):
        return str(x)


def _pct_savings(baseline: float, final: float) -> str:
    if not baseline or baseline <= 0:
        return "-"
    saving = (baseline - final) / baseline * 100.0
    if saving >= 0:
        return format(saving, ".1f") + "%"
    # Negative saving: the final run used MORE energy (e.g. repair cost).
    return "+" + format(-saving, ".1f") + "% (more)"


def _skills_short(combo: str | None, decisions: list[dict[str, Any]]) -> str:
    """Compact skills cell: top combo + repair skills applied."""
    parts = []
    if combo:
        parts.append(combo)
    repair = [d["skill"] for d in decisions if d["skill"] in (
        "gelu_fp32", "adaln_fp32", "rmsnorm_fp32", "softmax_fp32", "vae_fp32",
        "boundary_block_bf16",
    )]
    if repair:
        parts.append("repair:" + "+".join(repair))
    return " | ".join(parts) if parts else "baseline"


def build_report(payload: dict[str, Any]) -> str:
    scenarios: list[dict[str, Any]] = payload.get("scenarios", [])
    lines: list[str] = []

    lines.append("# VDG Test Report -- Scenario Matrix")
    lines.append("")
    lines.append("**Platform:** " + payload.get("platform", "VDG"))
    lines.append("**Primary model:** " + payload.get("primary_model", "LTX-2.3"))
    lines.append("**Scenarios:** " + str(payload.get("scenario_count", len(scenarios))))
    lines.append("**Mode:** " + ("pure simulation (offline)" if payload.get("pure_simulation") else "measured"))
    lines.append("")
    lines.append("Every scenario runs the full GovernancePipeline "
                 "(diagnose -> rules -> select accel -> repair -> simulate -> report). "
                 "The governance layer auto-selects the best *policy-feasible* skill combo "
                 "under each scenario's latency SLO, energy budget, and quality floor.")
    lines.append("")

    # -------------------------------------------------------------------
    # Main results table
    # -------------------------------------------------------------------
    lines.append("## Results")
    lines.append("")
    lines.append("| # | Scenario | Device | Workload | Baseline lat (s) | Final lat (s) | Speedup | Energy (J) | Memory (GB) | Quality | Skills | Pareto | Feasible |")
    lines.append("|--:|----------|--------|----------|-----------------:|--------------:|--------:|-----------:|------------:|--------:|--------|--------|:--------:|")
    for i, s in enumerate(scenarios, 1):
        base = s.get("baseline", {}) or {}
        fin = s.get("final", {}) or {}
        workload = str(s.get("resolution")) + " " + str(s.get("frames")) + "f"
        lines.append(
            "| " + str(i) + " | " + s.get("label", "") + " | " + s.get("device", "")
            + " | " + workload
            + " | " + _f(base.get("latency_s"))
            + " | " + _f(fin.get("latency_s"))
            + " | " + _f(s.get("speedup"), 3) + "x"
            + " | " + _f(fin.get("energy_j"), 0)
            + " | " + _f(fin.get("peak_memory_gb"))
            + " | " + _f(fin.get("quality"))
            + " | " + _skills_short(s.get("top_combo"), s.get("decisions", []))
            + " | " + str(fin.get("pareto", "-"))
            + " | " + ("yes" if s.get("feasible") else "no") + " |"
        )
    lines.append("")

    # -------------------------------------------------------------------
    # Energy comparison
    # -------------------------------------------------------------------
    lines.append("## Energy Comparison")
    lines.append("")
    lines.append("Baseline (unskilled) vs governance-final energy per scenario, "
                 "with the percent energy saved. Negative values mean the final "
                 "run used *more* energy (e.g. Apple Silicon repair skills trade "
                 "latency/energy for numerical correctness).")
    lines.append("")
    lines.append("| Scenario | Device | Baseline energy (J) | Final energy (J) | Energy saved | Baseline lat (s) | Final lat (s) |")
    lines.append("|----------|--------|--------------------:|-----------------:|-------------:|-----------------:|--------------:|")
    for s in scenarios:
        base = s.get("baseline", {}) or {}
        fin = s.get("final", {}) or {}
        lines.append(
            "| " + s.get("label", "") + " | " + s.get("device", "")
            + " | " + _f(base.get("energy_j"), 0)
            + " | " + _f(fin.get("energy_j"), 0)
            + " | " + _pct_savings(base.get("energy_j"), fin.get("energy_j"))
            + " | " + _f(base.get("latency_s"))
            + " | " + _f(fin.get("latency_s")) + " |"
        )
    lines.append("")

    # -------------------------------------------------------------------
    # Focus strategy table
    # -------------------------------------------------------------------
    lines.append("## Focus Strategy (Direct Simulation)")
    lines.append("")
    lines.append("Each scenario's headline acceleration strategy, simulated "
                 "directly (independent of the governance auto-selection). This "
                 "shows the strategy's raw numbers -- governance may reject it "
                 "when it violates the quality floor or SLO.")
    lines.append("")
    lines.append("| Scenario | Focus strategy | Applicable | Lat (s) | Energy (J) | Memory (GB) | Quality | vs baseline lat |")
    lines.append("|----------|----------------|:----------:|--------:|-----------:|------------:|--------:|----------------:|")
    for s in scenarios:
        fr = s.get("focus_result")
        base = s.get("baseline", {}) or {}
        if not fr:
            lines.append(
                "| " + s.get("label", "") + " | " + s.get("focus", "")
                + " | - | - | - | - | - | - |"
            )
            continue
        applicable = fr.get("applicable", True)
        if not applicable:
            lines.append(
                "| " + s.get("label", "") + " | " + s.get("focus", "")
                + " | no | - | - | - | - | - |"
            )
            continue
        vs_base = _pct_savings(base.get("latency_s"), fr.get("latency_s"))
        lines.append(
            "| " + s.get("label", "") + " | " + s.get("focus", "")
            + " | yes | " + _f(fr.get("latency_s"))
            + " | " + _f(fr.get("energy_j"), 0)
            + " | " + _f(fr.get("peak_memory_gb"))
            + " | " + _f(fr.get("quality"))
            + " | " + vs_base + " |"
        )
    lines.append("")

    # -------------------------------------------------------------------
    # Repair recommendations
    # -------------------------------------------------------------------
    lines.append("## Repair Recommendations")
    lines.append("")
    lines.append("NumericalProbe-driven repair skills recommended per scenario "
                 "(AdaLN/GELU fp32 guards on low-precision backends; boundary-block "
                 "bf16 on int8-only devices).")
    lines.append("")
    any_repair = False
    for s in scenarios:
        repairs = s.get("repair_recommendations", [])
        if not repairs:
            continue
        any_repair = True
        names = ", ".join(r["skill"] for r in repairs)
        lines.append("- **" + s.get("label", "") + "** (" + s.get("device", "") + "): " + names)
        for r in repairs:
            lines.append("    - " + r["skill"] + ": speedup " + _f(r.get("speedup"), 3)
                         + ", quality_delta " + _f(r.get("quality_delta"))
                         + " -- " + r.get("rationale", "").split(".")[0] + ".")
    if not any_repair:
        lines.append("_No repair recommendations across the scenario matrix._")
    lines.append("")

    # -------------------------------------------------------------------
    # Top-5 alternatives per scenario
    # -------------------------------------------------------------------
    lines.append("## Top-5 Ranked Alternatives")
    lines.append("")
    for s in scenarios:
        lines.append("### " + s.get("label", ""))
        lines.append("")
        alts = s.get("alternatives_top5", [])
        if not alts:
            lines.append("_No alternatives._")
            lines.append("")
            continue
        lines.append("| Combo | Lat (s) | Energy (J) | Memory (GB) | Quality | Pareto | Feasible |")
        lines.append("|-------|--------:|-----------:|------------:|--------:|--------|:--------:|")
        for a in alts:
            lines.append(
                "| " + a.get("combo", "") + " | " + _f(a.get("latency_s"))
                + " | " + _f(a.get("energy_j"), 0)
                + " | " + _f(a.get("peak_memory_gb"))
                + " | " + _f(a.get("quality"))
                + " | " + str(a.get("pareto", "-"))
                + " | " + ("yes" if a.get("feasible") else "no") + " |"
            )
        lines.append("")

    # -------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------
    lines.append("---")
    lines.append("Generated by scripts/generate_report.py from "
                 "test_results/results.json (produced by "
                 "scripts/run_all_scenarios.py). All numbers are roofline + "
                 "energy-model simulations; no GPU or network was used.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate docs/TEST_REPORT.md.")
    parser.add_argument("--in", dest="inp", default="test_results/results.json",
                        help="input results JSON (default: test_results/results.json)")
    parser.add_argument("--out", default="docs/TEST_REPORT.md",
                        help="output markdown path (default: docs/TEST_REPORT.md)")
    args = parser.parse_args(argv)

    in_path = Path(args.inp)
    if not in_path.exists():
        print("error: input not found: " + str(in_path), file=sys.stderr)
        print("Run scripts/run_all_scenarios.py first.", file=sys.stderr)
        return 1
    payload = json.loads(in_path.read_text())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(payload)
    out_path.write_text(report)

    print("Wrote report to " + str(out_path)
          + " (" + str(len(payload.get("scenarios", []))) + " scenarios, "
          + str(out_path.stat().st_size) + " bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
