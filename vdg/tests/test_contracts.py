"""Contracts base-class tests (exact signatures + behavior)."""
from __future__ import annotations

import inspect

import pytest

from vdg import (
    DeviceCategory,
    DeviceProfile,
    DeviceSpec,
    GovernanceDecision,
    LoadModel,
    Skill,
    SkillImpact,
    VideoDiTLoad,
)
from vdg.core.roofline import GB


# --- DeviceSpec -----------------------------------------------------------
def test_devicespec_frozen():
    spec = DeviceSpec(
        name="x", category="c", memory_gb=1.0, memory_bandwidth_gbps=10.0,
        compute_tflops={"bf16": 1.0}, tdp_w=100.0, idle_power_w=10.0,
        supported_precisions=["bf16"], attention_backends=["math"],
    )
    with pytest.raises(Exception):
        spec.name = "y"  # frozen dataclass


def test_devicespec_peak_flops_and_bw():
    spec = DeviceSpec(
        name="x", category="c", memory_gb=1.0, memory_bandwidth_gbps=1000.0,
        compute_tflops={"bf16": 165.0}, tdp_w=100.0, idle_power_w=10.0,
        supported_precisions=["bf16"], attention_backends=["math"],
    )
    assert spec.peak_flops("bf16") == 165.0 * 1e12
    assert spec.mem_bw_bytes() == 1000.0 * 1e9
    assert spec.supports("bf16") is True
    assert spec.supports("fp8") is False


def test_devicespec_peak_flops_unknown_precision():
    spec = DeviceSpec(
        name="x", category="c", memory_gb=1.0, memory_bandwidth_gbps=10.0,
        compute_tflops={"bf16": 1.0}, tdp_w=100.0, idle_power_w=10.0,
        supported_precisions=["bf16"], attention_backends=["math"],
    )
    with pytest.raises(ValueError):
        spec.peak_flops("fp4")


def test_devicespec_optional_cost():
    spec = DeviceSpec(
        name="x", category="c", memory_gb=1.0, memory_bandwidth_gbps=10.0,
        compute_tflops={"bf16": 1.0}, tdp_w=100.0, idle_power_w=10.0,
        supported_precisions=["bf16"], attention_backends=["math"],
        cost_per_hour_usd=0.5,
    )
    assert spec.cost_per_hour_usd == 0.5


# --- DeviceProfile base ---------------------------------------------------
def test_deviceprofile_base_spec_raises():
    with pytest.raises(NotImplementedError):
        DeviceProfile().spec()


def test_deviceprofile_base_availability_defaults():
    p = DeviceProfile()
    assert p.is_available() is False
    assert p.measure_power() is None


# --- VideoDiTLoad ---------------------------------------------------------
def test_videoditload_d_ff():
    load = VideoDiTLoad(
        model_name="m", params_b=2.0, vae_compress=(8, 32, 32), patch_size=1,
        te_params_b=5.0, layers=48, hidden_dim=1536, heads=24, default_steps=30,
        supported_tasks=["t2v"], vae_params_m=175.0, ffn_expansion=4.0,
    )
    assert load.d_ff == 6144


def test_videoditload_default_ffn_expansion():
    load = VideoDiTLoad(
        model_name="m", params_b=2.0, vae_compress=(8, 32, 32), patch_size=1,
        te_params_b=5.0, layers=48, hidden_dim=1536, heads=24, default_steps=30,
        supported_tasks=["t2v"], vae_params_m=175.0,
    )
    assert load.ffn_expansion == 4.0
    assert load.d_ff == 6144


# --- LoadModel base (using LTX fixture via conftest) ----------------------
def test_loadmodel_tokens_for(ltx23):
    assert ltx23.tokens_for((854, 480), 81) == 4053


def test_loadmodel_per_step_flops(ltx23):
    out = ltx23.per_step_flops(4053, text_tokens=256)
    assert set(out) == {"attention", "ffn", "total"}
    assert out["total"] == out["attention"] + out["ffn"]


def test_loadmodel_memory_footprint(ltx23):
    mem = ltx23.memory_footprint("bf16", 4053)
    assert set(mem) == {"weights", "kv", "activations", "total_gb"}
    # weights = params_b * 1e9 * 2 / GB
    assert abs(mem["weights"] - (2.0 * 1e9 * 2 / GB)) < 1e-9
    # total is the sum
    assert abs(mem["total_gb"] - (mem["weights"] + mem["kv"] + mem["activations"])) < 1e-9


def test_loadmodel_memory_fp8_half_of_bf16(ltx23):
    bf16 = ltx23.memory_footprint("bf16", 4053)["total_gb"]
    fp8 = ltx23.memory_footprint("fp8", 4053)["total_gb"]
    assert abs(fp8 - bf16 / 2.0) < 1e-9


# --- SkillImpact ----------------------------------------------------------
def test_skillimpact_defaults():
    imp = SkillImpact()
    assert imp.speedup == 1.0
    assert imp.memory_ratio == 1.0
    assert imp.quality_delta == 0.0
    assert imp.energy_ratio == 1.0
    assert imp.applies_to == []
    assert imp.notes == ""


# --- Skill base -----------------------------------------------------------
def test_skill_base_kind_default():
    assert Skill.kind == "accel"


def test_skill_base_applicable_default_true():
    assert Skill().applicable(object(), object()) is True


def test_skill_base_default_config_empty():
    assert Skill().default_config() == {}


def test_skill_base_predict_raises():
    with pytest.raises(NotImplementedError):
        Skill().predict(None, None)


def test_skill_base_apply_returns_input():
    obj = object()
    assert Skill().apply(obj) is obj


# --- GovernanceDecision ---------------------------------------------------
def test_governance_decision_fields():
    imp = SkillImpact(speedup=2.0)
    d = GovernanceDecision(skill_name="teacache", config={"threshold": 0.1},
                           predicted_impact=imp, rationale="cuts latency")
    assert d.skill_name == "teacache"
    assert d.config == {"threshold": 0.1}
    assert d.predicted_impact is imp
    assert d.rationale == "cuts latency"


# --- Exact signature guard (phase-2 contract stability) ------------------
def test_loadmodel_method_signatures():
    sig = inspect.signature(LoadModel.tokens_for)
    assert list(sig.parameters) == ["self", "resolution", "frames"]
    sig = inspect.signature(LoadModel.per_step_flops)
    assert list(sig.parameters) == ["self", "tokens", "text_tokens"]
    sig = inspect.signature(LoadModel.memory_footprint)
    assert list(sig.parameters) == ["self", "precision", "tokens"]


def test_skill_method_signatures():
    sig = inspect.signature(Skill.predict)
    assert list(sig.parameters) == ["self", "device", "load", "config"]
    sig = inspect.signature(Skill.apply)
    assert list(sig.parameters) == ["self", "model_or_pipeline", "config"]


def test_deviceprofile_method_signatures():
    assert list(inspect.signature(DeviceProfile.spec).parameters) == ["self"]
    assert list(inspect.signature(DeviceProfile.is_available).parameters) == ["self"]
    assert list(inspect.signature(DeviceProfile.measure_power).parameters) == ["self"]
