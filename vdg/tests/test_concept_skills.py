"""Concept-skill tests: the new accel/repair skills register and gate correctly.

Covers the skills added in the runtime phase -- techniques grounded in the
research reports that are either device-gated (no kernel on other families),
training-side (not a plug-in patch), or block-level numerical guards:

* sliding_tile_attention  (STA, ICML 2025)      -- CUDA kernel, consumer_nv only
* mlx_sdpa                (MLX fused SDPA)      -- Apple Silicon only
* linear_attention        (SANA-Video 2.0)      -- training-side, device-agnostic
* flash_attention         (FA-2 / FA-3)         -- consumer_nv + datacenter
* context_window          (Kijai chunking)      -- device-agnostic
* diffusion_forcing       (CogVideoX frame-packing) -- training-side
* boundary_block_bf16     (block-level bf16 guard)  -- low-precision backends

Each test uses the real registered plugins (registry names as canonical
identifiers), so a renamed or unregistered skill fails the registration tests
rather than silently passing.
"""
from __future__ import annotations

import pytest

from vdg import REGISTRY, DeviceCategory, Skill
from vdg.core.contracts import SkillImpact


def _device(name: str):
    cls = REGISTRY.get("device", name)
    assert cls is not None, "device " + name + " must be registered"
    return cls()


def _load(name: str):
    cls = REGISTRY.get("load", name)
    assert cls is not None, "load " + name + " must be registered"
    return cls()


def _skill(name: str):
    cls = REGISTRY.get("skill", name)
    assert cls is not None, "skill " + name + " must be registered"
    return cls


# Devices used across the applicability matrix.
M4 = _device("M4_Max")              # apple_silicon
RTX = _device("RTX4090")            # consumer_nv (int8-capable)
ASCEND = _device("Ascend_910B")     # edge_npu (server NPU)
RK = _device("RK3588")              # edge_npu (low-end, int8/int4 only)
H100 = _device("H100")              # datacenter
LTX = _load("LTX_2_3")


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,kind", [
    ("sliding_tile_attention", "accel"),
    ("mlx_sdpa", "accel"),
    ("linear_attention", "accel"),
    ("flash_attention", "accel"),
    ("context_window", "accel"),
    ("diffusion_forcing", "accel"),
    ("boundary_block_bf16", "repair"),
])
def test_new_skills_register(name, kind):
    cls = _skill(name)
    assert issubclass(cls, Skill)
    assert cls().kind == kind


def test_all_new_skills_are_runtime_concept_skills():
    """Every new skill is an envelope-emitting Skill (predict + apply)."""
    for name in (
        "sliding_tile_attention", "mlx_sdpa", "linear_attention",
        "flash_attention", "context_window", "diffusion_forcing",
        "boundary_block_bf16",
    ):
        inst = _skill(name)()
        impact = inst.predict(M4, LTX) if inst.applicable(M4, LTX) \
            else inst.predict(RTX, LTX)
        assert isinstance(impact, SkillImpact)
        assert inst.apply(object()) is not None


# --------------------------------------------------------------------------
# Device gating
# --------------------------------------------------------------------------
def test_sliding_tile_attention_not_apple_silicon():
    """STA is a CUDA custom kernel: no Metal / NPU backend exists."""
    sta = _skill("sliding_tile_attention")()
    assert not sta.applicable(M4, LTX)          # apple_silicon -> no
    assert not sta.applicable(ASCEND, LTX)      # edge_npu -> no
    assert sta.applicable(RTX, LTX)             # consumer_nv -> yes
    # predict on a non-applicable device still returns a well-formed impact
    # (governance simulators call predict only after applicable()).
    imp = sta.predict(M4, LTX)
    assert isinstance(imp, SkillImpact)


def test_mlx_sdpa_only_apple_silicon():
    """mx.fast.scaled_dot_product_attention is the Apple Metal path."""
    sdpa = _skill("mlx_sdpa")()
    assert sdpa.applicable(M4, LTX)
    assert not sdpa.applicable(RTX, LTX)
    assert not sdpa.applicable(ASCEND, LTX)
    imp = sdpa.predict(M4, LTX)
    assert imp.applies_to == [DeviceCategory.APPLE_SILICON]


def test_boundary_bf16_applicable_to_edge_npu():
    """Block-level bf16 guard targets low-precision backends (NPU first)."""
    bb = _skill("boundary_block_bf16")()
    assert bb.applicable(ASCEND, LTX)   # edge_npu -> yes (report: Ascend 910B)
    assert bb.applicable(RK, LTX)       # int8/int4-only edge NPU -> yes
    assert bb.applicable(M4, LTX)       # apple_silicon low-precision -> yes
    # int8-capable consumer NV is a low-precision backend -> yes;
    # datacenter fp8/Blackwell is allowed to relax the guard -> no.
    assert bb.applicable(RTX, LTX)
    assert not bb.applicable(H100, LTX)


def test_flash_attention_cuda_families_only():
    fa = _skill("flash_attention")()
    assert fa.applicable(RTX, LTX)
    assert fa.applicable(H100, LTX)
    assert not fa.applicable(M4, LTX)       # no Metal backend
    assert not fa.applicable(ASCEND, LTX)   # CUDA/ROCm only


def test_device_agnostic_concept_skills():
    """Training-side / runtime-level skills apply everywhere they host."""
    for name in ("linear_attention", "context_window", "diffusion_forcing"):
        inst = _skill(name)()
        for device in (M4, RTX, ASCEND, RK):
            assert inst.applicable(device, LTX), name + " on " + device.spec().category


# --------------------------------------------------------------------------
# predict() value ranges (grounded in the research reports)
# --------------------------------------------------------------------------
def test_sliding_tile_attention_predict_ranges():
    sta = _skill("sliding_tile_attention")()
    tf = sta.predict(RTX, LTX)
    assert tf.speedup == pytest.approx(1.4)     # HunyuanVideo 945->685s
    assert tf.quality_delta == pytest.approx(0.0)  # training-free, no loss
    ft = sta.predict(RTX, LTX, {"mode": "finetuned"})
    assert ft.speedup == pytest.approx(2.5)
    assert ft.quality_delta == pytest.approx(-0.09)  # FastVideo -0.09 VBench
    # Mode aliases tolerated, unknown modes fall back to training_free.
    assert sta.predict(RTX, LTX, {"mode": "tf"}).speedup == pytest.approx(1.4)
    assert sta.predict(RTX, LTX, {"mode": "bogus"}).speedup == pytest.approx(1.4)


def test_mlx_sdpa_predict_ranges():
    sdpa = _skill("mlx_sdpa")()
    imp = sdpa.predict(M4, LTX)
    assert imp.speedup == pytest.approx(1.4)      # mx.fast SDPA ~1.4-1.5x
    assert imp.memory_ratio == pytest.approx(0.85)  # avoids T x T materialization
    assert imp.quality_delta == pytest.approx(0.0)  # exact attention
    # The materialized-matrix note reflects the default T=8192/16 heads.
    assert "4.29 GB" in imp.notes


def test_boundary_bf16_predict_ranges():
    bb = _skill("boundary_block_bf16")()
    imp = bb.predict(ASCEND, LTX)
    assert imp.speedup == pytest.approx(0.95)     # bf16 vs int8 tensor cores
    assert imp.quality_delta == pytest.approx(1.5)  # HiF8 keeps 5/5 VBench dims
    assert imp.memory_ratio == pytest.approx(1.0)
    assert imp.energy_ratio == pytest.approx(1.05)
    imp2 = bb.predict(ASCEND, LTX, {"n_first": 2, "n_last": 3, "dtype": "bfloat16"})
    assert imp2.quality_delta == pytest.approx(1.5)


def test_linear_attention_predict_ranges():
    la = _skill("linear_attention")()
    imp = la.predict(RTX, LTX)
    assert imp.speedup == pytest.approx(3.0)      # conservative vs SANA 16x
    assert imp.quality_delta == pytest.approx(-1.0)  # untuned port caveat
    assert imp.applies_to == []                    # architecture change


def test_flash_attention_predict_ranges():
    fa = _skill("flash_attention")()
    imp = fa.predict(RTX, LTX)
    assert imp.speedup == pytest.approx(1.5)       # FA-2 default
    assert imp.quality_delta == pytest.approx(0.0)  # exact attention
    fa3 = fa.predict(RTX, LTX, {"version": "fa3"})
    assert fa3.speedup == pytest.approx(1.0)       # Hopper-only -> neutral no-op
    fa3h = fa.predict(H100, LTX, {"version": "fa3"})
    assert fa3h.speedup == pytest.approx(1.8)      # Hopper datacenter -> FA-3
    assert fa3h.applies_to == [DeviceCategory.CONSUMER_NV, DeviceCategory.DATACENTER]


def test_context_window_predict_ranges():
    cw = _skill("context_window")()
    imp = cw.predict(RTX, LTX)
    assert imp.speedup == pytest.approx(0.9)       # overlap recompute cost
    assert imp.memory_ratio == pytest.approx(0.35)  # 1025f -> 81f windows
    assert imp.quality_delta == pytest.approx(0.0)


def test_diffusion_forcing_predict_ranges():
    df = _skill("diffusion_forcing")()
    imp = df.predict(RTX, LTX)
    assert imp.speedup == pytest.approx(2.0)       # conservative vs 3-4x report
    assert imp.memory_ratio == pytest.approx(0.6)  # 3x memory reduction inverted
    assert imp.quality_delta == pytest.approx(-0.5)  # temporal consistency caveat


def test_predict_returns_frozen_impact_shape():
    """Every new skill's predict returns a complete SkillImpact."""
    for name in (
        "sliding_tile_attention", "mlx_sdpa", "linear_attention",
        "flash_attention", "context_window", "diffusion_forcing",
        "boundary_block_bf16",
    ):
        inst = _skill(name)()
        device = M4 if inst.applicable(M4, LTX) else RTX
        imp = inst.predict(device, LTX)
        assert imp.speedup >= 0.0
        assert 0.0 < imp.memory_ratio <= 1.0
        assert -100.0 <= imp.quality_delta <= 100.0
        assert imp.energy_ratio > 0.0
        assert imp.notes  # grounded provenance present
