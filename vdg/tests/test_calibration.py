"""Anchor-based simulation calibration tests (core/calibration.py).

Uses the real registered plugins (devices/loads) so anchors match by their
canonical registry names. The key invariant: simulating AT an anchor's exact
operating point must reproduce the measured latency (calibrated = predicted x
scale = measured), and runs away from any anchor must stay engineering
estimates (scale 1.0 + warning).
"""
from __future__ import annotations

import pytest

from vdg.core.calibration import (
    ANCHORS,
    CalibrationReport,
    CalibratedSimulator,
    find_anchor,
)
from vdg.core.registry import REGISTRY
from vdg.core.simulator import PerformanceEnergySimulator

REGISTRY.discover()


def _device(name: str):
    cls = REGISTRY.get("device", name)
    assert cls is not None, "device " + name + " must be registered"
    return cls()


def _load(name: str):
    cls = REGISTRY.get("load", name)
    assert cls is not None, "load " + name + " must be registered"
    return cls()


def _anchor(device: str, load: str):
    hits = [
        a for a in ANCHORS
        if a.device_name == device and a.load_name == load and a.kind == "latency"
    ]
    assert hits, "expected a latency anchor for " + device + "/" + load
    return hits[0]


# --------------------------------------------------------------------------
# ANCHORS data integrity
# --------------------------------------------------------------------------
def test_anchors_cover_the_five_grounded_data_points():
    pairs = {
        (a.device_name, a.load_name) for a in ANCHORS if a.kind == "latency"
    }
    # Single-device latency anchors only (multi-GPU ones are report-only).
    assert ("M4_Max", "Wan21_T2V_1_3B") in pairs
    assert ("RTX4090", "Wan21_I2V_14B") in pairs
    assert ("H100", "Wan21_I2V_14B") in pairs
    assert ("H100", "HunyuanVideo_13B") in pairs
    # Skill/memory anchors present with honest kinds.
    kinds = {a.kind for a in ANCHORS}
    assert "speedup" in kinds and "memory" in kinds
    speedup = next(a for a in ANCHORS if a.kind == "speedup")
    assert speedup.measured_latency_s == pytest.approx(22.62)
    memory = next(a for a in ANCHORS if a.kind == "memory")
    assert memory.measured_latency_s is None


def test_every_latency_anchor_has_a_resolution_and_measured_latency():
    for a in ANCHORS:
        if a.kind != "latency":
            continue
        assert a.resolution is not None
        assert a.frames is not None and a.frames > 0
        assert a.steps is not None and a.steps > 0
        assert a.measured_latency_s is not None and a.measured_latency_s > 0
        assert a.source, "every anchor needs a citation"


def test_multi_gpu_anchors_never_match_single_devices():
    # H100_x8 anchors must never be the anchor returned for a single-device H100
    # simulation -- the single-GPU anchor is the one that may match.
    for a in ANCHORS:
        if a.device_name != "H100_x8":
            continue
        got = find_anchor("H100", a.load_name, a.resolution or (1280, 720))
        assert got is not a
        assert got is None or got.device_name == "H100"


# --------------------------------------------------------------------------
# find_anchor matching
# --------------------------------------------------------------------------
def test_find_anchor_matches_registry_and_display_names():
    a1 = find_anchor("M4_Max", "Wan21_T2V_1_3B", (854, 480))
    a2 = find_anchor("M4 Max", "Wan2.1-T2V-1.3B", (832, 480))
    assert a1 is not None and a2 is not None
    assert a1 is a2  # same anchor under both spellings


def test_find_anchor_resolution_tolerance():
    anchor = _anchor("H100", "HunyuanVideo_13B")
    # 1280x720 anchor: within 20% pixel count.
    assert find_anchor("H100", "HunyuanVideo_13B", (1280, 720)) is anchor
    assert find_anchor("H100", "HunyuanVideo_13B", (1104, 832)) is anchor  # ~0.3% off
    # 960x540 is ~44% fewer pixels -> out of tolerance.
    assert find_anchor("H100", "HunyuanVideo_13B", (960, 540)) is None


def test_find_anchor_returns_none_for_unregistered_pairs():
    assert find_anchor("RTX5090", "LTX_2_3", (854, 480)) is None
    # M4_Max/LTX_2_3 now has a real measured anchor (ComfyUI benchmark 2026-08-17)
    assert find_anchor("M4_Max", "LTX_2_3", (768, 512)) is not None


# --------------------------------------------------------------------------
# CalibratedSimulator
# --------------------------------------------------------------------------
def test_calibrated_sim_reproduces_measured_at_anchor_operating_point():
    """At the anchor's exact workload the calibrated latency == measured."""
    dev, ld = _device("RTX4090"), _load("Wan21_I2V_14B")
    anchor = _anchor("RTX4090", "Wan21_I2V_14B")
    config = {
        "resolution": tuple(anchor.resolution),
        "frames": int(anchor.frames),
        "steps": int(anchor.steps),
        "task": "i2v",
        "precision": "bf16",
        "attention_backend": "flash",
    }
    result = CalibratedSimulator().simulate(dev, ld, [], config, None)
    assert result.latency_s == pytest.approx(anchor.measured_latency_s, rel=1e-6)
    assert any("calibration: anchor" in w for w in result.warnings)
    assert result.steps == anchor.steps
    assert result.tokens > 0
    # Throughput consistent with the scaled latency.
    assert result.throughput_tokens_s == pytest.approx(result.tokens / result.latency_s)


def test_calibration_scales_energy_and_breakdown():
    dev, ld = _device("H100"), _load("HunyuanVideo_13B")
    anchor = _anchor("H100", "HunyuanVideo_13B")
    config = {
        "resolution": tuple(anchor.resolution),
        "frames": int(anchor.frames),
        "steps": int(anchor.steps),
        "precision": "bf16",
    }
    sim = CalibratedSimulator()
    result = sim.simulate(dev, ld, [], config, None)
    base = sim.last_base_result
    scale = result.latency_s / base.latency_s
    assert scale != pytest.approx(1.0)
    assert result.energy_j == pytest.approx(base.energy_j * scale)
    assert result.breakdown["denoise"] == pytest.approx(base.breakdown["denoise"] * scale)
    # Unscaled fields survive calibration.
    assert result.peak_memory_gb == base.peak_memory_gb
    assert result.quality_score == base.quality_score
    assert result.pareto_tag == base.pareto_tag


def test_no_anchor_is_engineering_estimate():
    dev, ld = _device("RTX5090"), _load("LTX_2_3")
    config = {"resolution": (854, 480), "frames": 81, "steps": 30, "precision": "bf16"}
    sim = CalibratedSimulator()
    result = sim.simulate(dev, ld, [], config, None)
    assert result.latency_s == pytest.approx(sim.last_base_result.latency_s)
    assert any("no anchor, engineering estimate" in w for w in result.warnings)


def test_manual_calibration_scale_knob():
    dev, ld = _device("RTX5090"), _load("LTX_2_3")
    config = {"resolution": (854, 480), "frames": 81, "steps": 30, "precision": "bf16"}
    sim = CalibratedSimulator(calibration_scale=2.0)
    result = sim.simulate(dev, ld, [], config, None)
    assert result.latency_s == pytest.approx(sim.last_base_result.latency_s * 2.0)
    assert result.energy_j == pytest.approx(sim.last_base_result.energy_j * 2.0)


def test_calibration_composes_with_skills():
    """Skill composition stays multiplicative on top of the anchor scale."""
    from vdg.core.contracts import Skill, SkillImpact
    from vdg.core.simulator import COMBINATION_EXPONENT

    class FakeSpeedup(Skill):
        def predict(self, device, load, config=None):
            return SkillImpact(speedup=2.0)

    dev, ld = _device("M4_Max"), _load("Wan21_T2V_1_3B")
    anchor = _anchor("M4_Max", "Wan21_T2V_1_3B")
    config = {
        "resolution": tuple(anchor.resolution),
        "frames": int(anchor.frames),
        "steps": int(anchor.steps),
        "precision": "bf16",
    }
    cal_plain = CalibratedSimulator().simulate(dev, ld, [], config, None)
    cal_skilled = CalibratedSimulator().simulate(dev, ld, [FakeSpeedup()], config, None)
    # Single-skill speedups still compose sub-multiplicatively (2.0 ** 0.85).
    assert cal_skilled.latency_s == pytest.approx(
        cal_plain.latency_s / (2.0 ** COMBINATION_EXPONENT)
    )


# --------------------------------------------------------------------------
# CalibrationReport
# --------------------------------------------------------------------------
def test_report_rows_cover_all_anchor_kinds():
    dev, ld = _device("H100"), _load("HunyuanVideo_13B")
    report = CalibrationReport.compare(dev, ld, None)
    kinds = {r.anchor.kind for r in report.rows}
    assert kinds == {"latency", "memory"}
    latency_row = next(r for r in report.rows if r.anchor.kind == "latency")
    assert latency_row.predicted_latency_s is not None
    assert latency_row.relative_error_pct is not None
    assert latency_row.relative_error_pct < 0  # roofline under-predicts H100
    memory_row = next(r for r in report.rows if r.anchor.kind == "memory")
    assert memory_row.predicted_latency_s is None
    assert memory_row.relative_error_pct is None


def test_report_render_is_a_text_table():
    dev, ld = _device("H100"), _load("HunyuanVideo_13B")
    text = CalibrationReport.compare(dev, ld, None).render()
    assert "Calibration report" in text
    assert "measured(s)" in text
    assert "rel.err" in text


def test_report_empty_for_pairs_without_anchors():
    dev, ld = _device("RTX5090"), _load("LTX_2_3")
    report = CalibrationReport.compare(dev, ld, None)
    assert report.rows == []
    assert "no anchors" in report.render()


# --------------------------------------------------------------------------
# GovernancePipeline integration (simulator override)
# --------------------------------------------------------------------------
def test_governance_pipeline_accepts_calibrated_simulator():
    from vdg.core.scenario import BUILTIN_SCENARIOS
    from vdg.governance.pipeline import GovernancePipeline

    scenario = BUILTIN_SCENARIOS.get("ltx_t2v_720p_129f")
    cal_pipe = GovernancePipeline(simulator=CalibratedSimulator())
    cal_report = cal_pipe.run("H100", "HunyuanVideo_13B", scenario)
    base_pipe = GovernancePipeline()
    base_report = base_pipe.run("H100", "HunyuanVideo_13B", scenario)

    # Calibrated baseline must be strictly slower (roofline under-predicts).
    assert cal_report.baseline_result.latency_s > base_report.baseline_result.latency_s
    assert any(
        "calibration" in w for w in cal_report.baseline_result.warnings
    )
    # Both pipelines still produce a full, feasible-shaped report.
    assert cal_report.final_result is not None
    assert cal_report.top_combo is not None
    assert cal_report.decisions


def test_calibrated_simulator_is_a_drop_in_simulator_type():
    assert isinstance(CalibratedSimulator(), PerformanceEnergySimulator)
