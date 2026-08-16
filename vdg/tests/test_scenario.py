"""Scenario library tests."""
from __future__ import annotations

import pytest

from vdg import BUILTIN_SCENARIOS, Scenario, ScenarioLibrary


EXPECTED = {
    "ltx_t2v_480p_81f",
    "ltx_t2v_720p_129f",
    "ltx_i2v_480p",
    "long_video_1025f",
    "edge_npu_shortclip",
}


def test_builtin_scenario_names():
    assert set(BUILTIN_SCENARIOS.names()) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_scenario_has_required_fields(name):
    s = BUILTIN_SCENARIOS.get(name)
    assert isinstance(s, Scenario)
    assert s.task in ("t2v", "i2v")
    assert s.frames > 0 and s.fps > 0 and s.steps > 0
    assert s.width > 0 and s.height > 0
    assert s.quality_target > 0
    assert s.energy_budget_j > 0
    assert s.latency_slo_s > 0
    assert s.duration_s == s.frames / s.fps


def test_ltx_480p_81f_grounding():
    s = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    assert s.resolution == (854, 480)
    assert s.frames == 81
    assert s.fps == 16
    assert s.steps == 30


def test_ltx_720p_129f_grounding():
    s = BUILTIN_SCENARIOS.get("ltx_t2v_720p_129f")
    assert s.resolution == (1280, 720)
    assert s.frames == 129
    assert s.fps == 24


def test_long_video_1025f_grounding():
    s = BUILTIN_SCENARIOS.get("long_video_1025f")
    assert s.frames == 1025


def test_edge_npu_shortclip_4_step_distilled():
    s = BUILTIN_SCENARIOS.get("edge_npu_shortclip")
    assert s.steps == 4  # distilled
    assert s.quality_target < 84.0  # lower target for edge


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        BUILTIN_SCENARIOS.get("does_not_exist")


def test_library_add_and_all():
    lib = ScenarioLibrary()
    lib.add(Scenario(
        name="custom", task="t2v", resolution=(640, 360), frames=49, fps=16,
        steps=10, quality_target=80.0, energy_budget_j=1000.0, latency_slo_s=60.0,
    ))
    assert lib.get("custom").frames == 49
    assert len(lib.all()) == 1
