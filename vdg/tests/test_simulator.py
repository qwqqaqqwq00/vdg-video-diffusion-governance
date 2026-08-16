"""Performance-energy simulator tests."""
from __future__ import annotations

import pytest

from vdg import (
    BUILTIN_SCENARIOS,
    PerformanceEnergySimulator,
    SimulationResult,
    Skill,
    SkillImpact,
)
from vdg.core.simulator import (
    ATTENTION_BACKEND_PRECISION,
    COMBINATION_EXPONENT,
    PRECISION_QUALITY_DELTA,
)


def _sim(device, load, skills=None, config=None, scenario=None):
    return PerformanceEnergySimulator().simulate(device, load, skills, config, scenario)


def test_simulate_returns_result_with_all_fields(rtx4090, ltx23):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "attention_backend": "math"}, scn)
    assert isinstance(r, SimulationResult)
    assert r.latency_s > 0
    assert r.energy_j > 0
    assert r.peak_memory_gb > 0
    assert 0.0 <= r.quality_score <= 100.0
    assert r.throughput_tokens_s > 0
    assert r.tokens == 4053
    assert r.steps == 30
    assert r.precision == "bf16"
    assert r.attention_backend == "math"
    assert isinstance(r.warnings, list)
    assert isinstance(r.pareto_tag, str)


def test_breakdown_has_all_phases(rtx4090, ltx23):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    r = _sim(rtx4090, ltx23, [], {"precision": "bf16"}, scn)
    assert set(r.breakdown) == {"denoise", "attention", "ffn", "vae_decode", "te_encode"}
    # attention + ffn should sum to denoise.
    assert abs(r.breakdown["attention"] + r.breakdown["ffn"] - r.breakdown["denoise"]) < 1e-6
    # VAE must NOT dominate absurdly (the geometric-mean fix).
    assert r.breakdown["vae_decode"] < r.latency_s


def test_oom_warning_for_oversized_load(rtx4090, wan14b):
    # Wan-14B in bf16 cannot fit in 24GB.
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    r = _sim(rtx4090, wan14b, [], {"precision": "bf16", "attention_backend": "flash"}, scn)
    assert r.peak_memory_gb > rtx4090.spec().memory_gb
    assert any("OOM" in w for w in r.warnings)


def test_fp8_reduces_memory_vs_bf16(rtx4090, wan14b):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    bf16 = _sim(rtx4090, wan14b, [], {"precision": "bf16"}, scn).peak_memory_gb
    fp8 = _sim(rtx4090, wan14b, [], {"precision": "fp8"}, scn).peak_memory_gb
    assert fp8 < bf16
    assert abs(fp8 - bf16 / 2.0) < 1e-6


def test_skill_speedup_reduces_latency(rtx4090, ltx23, teacache):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    cfg = {"precision": "bf16", "attention_backend": "flash"}
    base = _sim(rtx4090, ltx23, [], cfg, scn).latency_s
    skilled = _sim(rtx4090, ltx23, [teacache], cfg, scn).latency_s
    assert skilled < base


def test_skill_quality_delta_applied(rtx4090, ltx23, teacache):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    cfg = {"precision": "bf16", "attention_backend": "flash"}
    base_q = _sim(rtx4090, ltx23, [], cfg, scn).quality_score
    skilled_q = _sim(rtx4090, ltx23, [teacache], cfg, scn).quality_score
    # TeaCache quality_delta is -0.07.
    assert abs((base_q - skilled_q) - 0.07) < 1e-6


def test_skill_memory_ratio_applied(rtx4090, wan14b):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    cfg = {"precision": "bf16", "attention_backend": "flash"}

    class HalfMem(Skill):
        kind = "accel"

        def applicable(self, device, load):
            return True

        def default_config(self):
            return {}

        def predict(self, device, load, config=None):
            return SkillImpact(speedup=1.0, memory_ratio=0.5)

    base = _sim(rtx4090, wan14b, [], cfg, scn).peak_memory_gb
    cut = _sim(rtx4090, wan14b, [HalfMem()], cfg, scn).peak_memory_gb
    assert abs(cut - base * 0.5) < 1e-6


def test_skill_not_applicable_is_skipped_with_warning(rtx4090, ltx23, sage2):
    # SageAttention2 only applies to consumer_nv; it IS applicable here, so test
    # the skip path with a deliberately-restricted skill instead.
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    cfg = {"precision": "bf16"}

    class NVOnly(Skill):
        kind = "accel"

        def applicable(self, device, load):
            return device.spec().category == "apple_silicon"

        def default_config(self):
            return {}

        def predict(self, device, load, config=None):
            return SkillImpact(speedup=100.0)

    r = _sim(rtx4090, ltx23, [NVOnly()], cfg, scn)
    assert any("not applicable" in w for w in r.warnings)
    # The 100x speedup must NOT have been applied.
    base = _sim(rtx4090, ltx23, [], cfg, scn).latency_s
    assert abs(r.latency_s - base) < 1e-6


def test_attention_backend_sage2_uses_fp8_peak(rtx4090, ltx23):
    # sage2 maps to fp8 precision, which on a 4090 has higher TFLOPS than bf16,
    # so attention should be faster than the bf16 math path.
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    math_r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "attention_backend": "math"}, scn)
    sage_r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "attention_backend": "sage2"}, scn)
    assert sage_r.breakdown["attention"] < math_r.breakdown["attention"]


def test_math_backend_penalized(rtx4090, ltx23):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    math_r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "attention_backend": "math"}, scn)
    flash_r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "attention_backend": "flash"}, scn)
    # math path is 0.5x effective peak -> slower attention than flash.
    assert math_r.breakdown["attention"] > flash_r.breakdown["attention"]


def test_precision_quality_delta(rtx4090, ltx23):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    bf16 = _sim(rtx4090, ltx23, [], {"precision": "bf16", "attention_backend": "flash"}, scn).quality_score
    fp8 = _sim(rtx4090, ltx23, [], {"precision": "fp8", "attention_backend": "flash"}, scn).quality_score
    assert abs((bf16 - fp8) - (PRECISION_QUALITY_DELTA["bf16"] - PRECISION_QUALITY_DELTA["fp8"])) < 1e-6
    assert fp8 < bf16


def test_te_offload_adds_warning(rtx4090, ltx23):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "te_offload": True}, scn)
    assert any("offloaded" in w for w in r.warnings)


def test_pareto_tags(rtx4090, ltx23):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    # LTX-2.3 480p on 4090 is fast & efficient -> feasible/efficient.
    r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "attention_backend": "flash"}, scn)
    assert r.pareto_tag in ("efficient", "feasible_lowq", "feasible")
    assert r.is_feasible(scn) is True


def test_pareto_infeasible_for_oversized(rtx4090, wan14b):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    r = _sim(rtx4090, wan14b, [], {"precision": "bf16", "attention_backend": "flash"}, scn)
    # 14B bf16 on 4090 is slow -> likely misses the 120s SLO.
    assert r.pareto_tag in ("infeasible", "slow_efficient", "fast_energy_heavy")


def test_meets_slo_and_budget(rtx4090, ltx23):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "attention_backend": "flash"}, scn)
    assert r.meets_slo(scn) is True
    assert r.meets_budget(scn) is True


def test_combination_exponent_submultiplicative(rtx4090, ltx23, teacache, sage2):
    # Two skills with speedup 2.0 and 1.8: combined = (2*1.8)**0.85, not 2*1.8.
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    cfg = {"precision": "bf16", "attention_backend": "flash"}
    base = _sim(rtx4090, ltx23, [], cfg, scn).latency_s
    skilled = _sim(rtx4090, ltx23, [teacache, sage2], cfg, scn).latency_s
    combined = (2.0 * 1.8) ** COMBINATION_EXPONENT
    assert abs(skilled - base / combined) < 1e-3


def test_scenario_overrides_steps(rtx4090, ltx23):
    scn = BUILTIN_SCENARIOS.get("edge_npu_shortclip")
    r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "attention_backend": "flash"}, scn)
    assert r.steps == scn.steps  # 4 distilled


def test_config_overrides_scenario_steps(rtx4090, ltx23):
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")
    r = _sim(rtx4090, ltx23, [], {"precision": "bf16", "steps": 8}, scn)
    assert r.steps == 8


def test_attention_backend_precision_map_complete():
    for backend in ("sage3", "sage2", "sage1", "flash", "sdpa", "triton", "mlx_sdpa", "math"):
        assert backend in ATTENTION_BACKEND_PRECISION


def test_unsupported_precision_warns(rtx4090, ltx23):
    # M4 Max has no fp4; request fp4 -> warning but no crash.
    scn = BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f")

    class FakeM4(rtx4090.__class__):
        def spec(self):
            from vdg import DeviceSpec, DeviceCategory
            return DeviceSpec(
                name="M4 Max", category=DeviceCategory.APPLE_SILICON,
                memory_gb=64.0, memory_bandwidth_gbps=546.0,
                compute_tflops={"fp32": 27.0, "bf16": 54.0, "fp16": 54.0, "fp8": 108.0},
                tdp_w=480.0, idle_power_w=10.0,
                supported_precisions=["fp32", "bf16", "fp16", "fp8"],
                attention_backends=["mlx_sdpa", "math"], unified_memory=True,
            )

    r = _sim(FakeM4(), ltx23, [], {"precision": "fp4", "attention_backend": "math"}, scn)
    assert any("fp4" in w for w in r.warnings)
