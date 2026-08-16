"""Tests for the acceleration skills library (vdg.skills.accel).

Verifies the composable inference-acceleration stack:
  * SageAttention is NOT applicable to Apple Silicon (CUDA+Triton only).
  * TeaCache is applicable to every device category (device-agnostic caching).
  * NVFP4 quantization only credits a speedup on Blackwell FP4 hardware
    (RTX 5090); on non-FP4 devices predict returns a neutral no-op.
  * predict() impact numbers are in the documented grounded ranges.

Host-independent (pure-sim; no torch/GPU required).
"""
from __future__ import annotations

import pytest

from vdg import DeviceCategory, Skill, SkillImpact
from vdg.core.registry import REGISTRY
from vdg.devices import get_device
from vdg.loads import LTX_2_3
from vdg.skills.accel import (
    compile_graph,
    offload,
    quantization,
    sage_attention,
    step_distill,
    teacache,
    vae_tiling,
)
from vdg.skills.accel.sage_attention import SageAttention
from vdg.skills.accel.teacache import TeaCache
from vdg.skills.accel.quantization import Quantization
from vdg.skills.accel.step_distill import StepDistill

ACCEL_SKILLS = [
    "teacache", "sage_attention", "quantization", "step_distill",
    "vae_tiling", "compile_graph", "offload",
]


def _skill(name):
    cls = REGISTRY.get("skill", name)
    assert cls is not None, name + " not registered"
    return cls()


# ---------------------------------------------------------------------------
# Registration + kind
# ---------------------------------------------------------------------------
def test_accel_skills_registered():
    names = set(REGISTRY.names("skill"))
    for n in ACCEL_SKILLS:
        assert n in names, "accel skill " + n + " not registered"


def test_accel_skills_kind():
    for n in ACCEL_SKILLS:
        assert _skill(n).kind == "accel"


# ---------------------------------------------------------------------------
# SageAttention: CUDA + Triton only -> NOT applicable to Apple Silicon / NPU
# ---------------------------------------------------------------------------
def test_sage_not_applicable_to_apple_silicon():
    sk = SageAttention()
    assert sk.applicable(get_device("M4_Max"), LTX_2_3()) is False


def test_sage_not_applicable_to_edge_npu():
    sk = SageAttention()
    assert sk.applicable(get_device("Jetson_Thor_T5000"), LTX_2_3()) is False
    assert sk.applicable(get_device("Ascend_910B"), LTX_2_3()) is False


def test_sage_not_applicable_to_datacenter():
    """SageAttention targets consumer NVIDIA only (not datacenter)."""
    sk = SageAttention()
    assert sk.applicable(get_device("H100"), LTX_2_3()) is False


def test_sage_applicable_to_consumer_nv():
    sk = SageAttention()
    assert sk.applicable(get_device("RTX4090"), LTX_2_3()) is True
    assert sk.applicable(get_device("RTX5090"), LTX_2_3()) is True


def test_sage_v3_needs_blackwell_fp4():
    """v3 (microscaling FP4) only credits a speedup on Blackwell FP4 hardware."""
    sk = SageAttention()
    # RTX 5090 (Blackwell) -> real 5x.
    imp = sk.predict(get_device("RTX5090"), LTX_2_3(), {"version": "v3"})
    assert imp.speedup == pytest.approx(5.0)
    assert imp.energy_ratio == pytest.approx(0.7)
    # RTX 4090 (Ada, no FP4) -> neutral no-op, never credits an impossible 5x.
    imp_ada = sk.predict(get_device("RTX4090"), LTX_2_3(), {"version": "v3"})
    assert imp_ada.speedup == pytest.approx(1.0)
    assert imp_ada.quality_delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TeaCache: device-agnostic -> applicable to ALL categories
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dev_name", [
    "RTX4090", "RTX5090", "M4_Max", "M3_Ultra", "H100", "B200",
    "Jetson_Thor_T5000", "Ascend_910B", "RK3588",
])
def test_teacache_applicable_all(dev_name):
    sk = TeaCache()
    assert sk.applicable(get_device(dev_name), LTX_2_3()) is True


def test_teacache_predict_in_range():
    sk = TeaCache()
    dev = get_device("RTX4090")
    load = LTX_2_3()
    for thr in (0.05, 0.1, 0.15, 0.2, 0.25):
        imp = sk.predict(dev, load, {"threshold": thr})
        assert 1.6 <= imp.speedup <= 4.0, "thr=" + str(thr) + " speedup=" + str(imp.speedup)
        # Higher threshold -> larger speedup (monotonic mapping).
        assert imp.quality_delta <= 0.0


def test_teacache_threshold_endpoints():
    sk = TeaCache()
    dev = get_device("RTX4090")
    load = LTX_2_3()
    assert sk.predict(dev, load, {"threshold": 0.05}).speedup == pytest.approx(1.6)
    assert sk.predict(dev, load, {"threshold": 0.25}).speedup == pytest.approx(4.0)


def test_teacache_clamps_out_of_range_threshold():
    sk = TeaCache()
    dev = get_device("RTX4090")
    load = LTX_2_3()
    # Below the min -> clamped to 1.6x.
    assert sk.predict(dev, load, {"threshold": 0.0}).speedup == pytest.approx(1.6)
    # Above the max -> clamped to 4.0x.
    assert sk.predict(dev, load, {"threshold": 0.99}).speedup == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# NVFP4 quantization: only Blackwell FP4 credits a speedup
# ---------------------------------------------------------------------------
def test_nvfp4_credits_speedup_only_on_blackwell():
    """NVFP4 needs Blackwell FP4 tensor cores -> only RTX 5090 gets the 3x."""
    sk = Quantization()
    load = LTX_2_3()
    # RTX 5090 (Blackwell, fp4) -> real 3x.
    imp = sk.predict(get_device("RTX5090"), load, {"method": "nvfp4"})
    assert imp.speedup == pytest.approx(3.0)
    assert imp.memory_ratio == pytest.approx(0.5)


@pytest.mark.parametrize("dev_name", [
    "RTX4090", "RTX6000_Ada", "M4_Max", "H100", "B200", "GB300_NVL72",
    "Jetson_Thor_T5000", "Ascend_910B",
])
def test_nvfp4_noop_on_non_blackwell_consumer(dev_name):
    """On every non-RTX-5090 device, NVFP4 predict returns a neutral no-op
    (speedup 1.0) so an impossible gain is never credited."""
    sk = Quantization()
    imp = sk.predict(get_device(dev_name), LTX_2_3(), {"method": "nvfp4"})
    assert imp.speedup == pytest.approx(1.0), dev_name + " nvfp4 speedup=" + str(imp.speedup)
    assert imp.quality_delta == pytest.approx(0.0)


def test_gguf_q4_applicable_apple_and_nv():
    sk = Quantization()
    load = LTX_2_3()
    for dev_name in ("M4_Max", "RTX4090", "RTX5090"):
        imp = sk.predict(get_device(dev_name), load, {"method": "gguf_q4"})
        assert imp.speedup == pytest.approx(1.1)
        assert imp.memory_ratio == pytest.approx(0.35)


def test_int8_applicable_edge_npu():
    sk = Quantization()
    imp = sk.predict(get_device("Ascend_910B"), LTX_2_3(), {"method": "int8"})
    assert imp.speedup == pytest.approx(2.0)
    assert imp.memory_ratio == pytest.approx(0.5)


def test_quantization_not_applicable_to_datacenter():
    """No quantization method targets datacenter (gguf=nv+apple, nvfp4=nv, int8=edge)."""
    sk = Quantization()
    assert sk.applicable(get_device("H100"), LTX_2_3()) is False


# ---------------------------------------------------------------------------
# StepDistill: device-agnostic (model-side)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dev_name", [
    "RTX4090", "M4_Max", "H100", "Jetson_Thor_T5000", "Ascend_910B",
])
def test_step_distill_applicable_all(dev_name):
    assert StepDistill().applicable(get_device(dev_name), LTX_2_3()) is True


def test_step_distill_marginal_speedup():
    """Speedup is marginal: min(baseline_steps/distilled, 10), not double-counted."""
    sk = StepDistill()
    dev = get_device("RTX4090")
    load = LTX_2_3()
    # 30-step baseline -> 4-step distilled = 7.5x (capped at 10).
    imp = sk.predict(dev, load, {"steps": 4, "baseline_steps": 30})
    assert imp.speedup == pytest.approx(7.5)
    # Already-distilled baseline (4 -> 4) gets 1.0x (no spurious multiplier).
    imp_eq = sk.predict(dev, load, {"steps": 4, "baseline_steps": 4})
    assert imp_eq.speedup == pytest.approx(1.0)


def test_step_distill_quality_in_range():
    sk = StepDistill()
    dev = get_device("RTX4090")
    load = LTX_2_3()
    q4 = sk.predict(dev, load, {"steps": 4}).quality_delta
    q8 = sk.predict(dev, load, {"steps": 8}).quality_delta
    assert -3.0 <= q4 <= -1.0
    assert -3.0 <= q8 <= -1.0
    # Fewer steps -> more quality loss.
    assert q4 <= q8


# ---------------------------------------------------------------------------
# predict() numbers in documented ranges (all accel skills)
# ---------------------------------------------------------------------------
def test_all_accel_predict_returns_skillimpact():
    dev = get_device("RTX4090")
    load = LTX_2_3()
    for n in ACCEL_SKILLS:
        sk = _skill(n)
        if not sk.applicable(dev, load):
            continue
        imp = sk.predict(dev, load)
        assert isinstance(imp, SkillImpact)
        assert imp.speedup > 0
        assert 0.0 < imp.memory_ratio <= 1.0


def test_vae_tiling_memory_cut():
    sk = _skill("vae_tiling")
    imp = sk.predict(get_device("RTX4090"), LTX_2_3())
    assert imp.speedup == pytest.approx(0.9)
    assert imp.memory_ratio == pytest.approx(0.25)


def test_offload_memory_ratio_in_range():
    sk = _skill("offload")
    imp = sk.predict(get_device("RTX4090"), LTX_2_3())
    assert imp.speedup == pytest.approx(0.6)
    # memory_ratio in [0.3, 0.5] for block_swap_ratio in [0, 1].
    assert 0.3 <= imp.memory_ratio <= 0.5


# ---------------------------------------------------------------------------
# apply() returns a runtime envelope (stub, no kernel patched)
# ---------------------------------------------------------------------------
def test_apply_returns_runtime_envelope():
    sk = TeaCache()
    env = sk.apply(object(), {"threshold": 0.1})
    assert isinstance(env, dict)
    assert env["skill"] == "teacache"
    assert env["runtime"] == "comfyui"
    assert "config" in env
    assert env["applied"] is False  # stub path: no kernel patched
