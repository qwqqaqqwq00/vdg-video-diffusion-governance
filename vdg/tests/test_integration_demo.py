"""Integration regression test: locks the killer governance demo contract.

Verifies the end-to-end governance highlight documented in README
"M4 Max + LTX-2.3 end-to-end governance" and INTEGRATION_REPORT.md section 6.
The GovernancePipeline defaults to the SIMULATED numerical probe (reproducible,
host-independent documented thresholds), so this test is deterministic on any
machine regardless of torch / MPS / CUDA availability.

Contract under test (vdg govern --device M4_Max --model LTX_2_3
--scenario ltx_t2v_480p):
  (a) detect MPS low-precision        -> probe reports a failure
  (b) recommend gelu/adaln fp32       -> both in repair_recommendations
  (c) select teacache, no sage        -> top_combo startswith "teacache";
                                         sage_attention absent from alternatives
      distill in alternatives         -> step_distill present somewhere
  (d) output a SimulationResult       -> final_result populated, latency < baseline
"""
from __future__ import annotations

import pytest

from vdg import BUILTIN_SCENARIOS
from vdg.core.simulator import SimulationResult
from vdg.devices import get_device
from vdg.governance import GovernancePipeline, GovernanceReport
from vdg.loads import LTX_2_3


def test_scenario_alias_ltx_t2v_480p_resolves():
    """The CLI/documentation shorthand resolves to the canonical scenario."""
    s = BUILTIN_SCENARIOS.get("ltx_t2v_480p")
    assert s.name == "ltx_t2v_480p_81f"
    # The alias must NOT inflate the documented scenario set.
    assert "ltx_t2v_480p" not in BUILTIN_SCENARIOS.names()


def test_govern_m4_ltx_demo_contract():
    """The killer demo: M4 Max + LTX-2.3 + ltx_t2v_480p governance run."""
    pipe = GovernancePipeline()
    report = pipe.run("M4_Max", "LTX_2_3", BUILTIN_SCENARIOS.get("ltx_t2v_480p"))
    assert isinstance(report, GovernanceReport)

    # (a) MPS low-precision detected (simulated probe -> documented thresholds).
    assert report.probe_report is not None
    assert report.probe_report.has_failure is True
    assert report.probe_mode == "simulated"

    # (b) Recommend gelu_fp32 AND adaln_fp32 repair skills.
    repair_names = {d.skill_name for d in report.repair_recommendations}
    assert "adaln_fp32" in repair_names, repair_names
    assert "gelu_fp32" in repair_names, repair_names
    assert len(report.patch_instructions) >= 2  # one block per repair skill

    # (c) Select teacache; sage disabled (absent everywhere); distill evaluated.
    assert report.top_combo is not None
    assert report.top_combo.startswith("teacache"), report.top_combo
    all_skills = set()
    for alt in report.alternatives:
        all_skills.update(alt.get("skills", []))
    assert "sage_attention" not in all_skills, "sage must be disabled on Apple Silicon"
    assert "step_distill" in all_skills, "distill must be among evaluated alternatives"

    # R1 note records the sage disable + teacache preference.
    assert any("Apple Silicon" in n or "R1" in n for n in report.rule_notes)

    # (d) Final SimulationResult populated and faster than baseline.
    assert isinstance(report.final_result, SimulationResult)
    assert report.baseline_result is not None
    assert report.final_result.latency_s < report.baseline_result.latency_s
    assert report.feasible is True
    assert report.final_result.latency_s > 0


def test_govern_m4_ltx_real_probe_still_selects_teacache():
    """The --real-probe path (measured) still selects teacache and is feasible.

    On a host without MPS the measured probe falls back to simulated; on this
    M4 Max (torch 2.12, GELU kernel fixed) it measures adaln only. Either way
    the governance selection (teacache) and feasibility must hold -- the probe
    mode only changes which repair skills are recommended, not the accel pick.
    """
    pipe = GovernancePipeline()
    report = pipe.run("M4_Max", "LTX_2_3",
                      BUILTIN_SCENARIOS.get("ltx_t2v_480p"), sim_probe=False)
    assert report.top_combo is not None
    assert report.top_combo.startswith("teacache")
    assert report.feasible is True
    # AdaLN bf16 cancellation is a precision property -> always flagged.
    assert "adaln_fp32" in {d.skill_name for d in report.repair_recommendations}
