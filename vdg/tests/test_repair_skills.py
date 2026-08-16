"""Tests for the repair skills library (numerical-robustness fp32 guards).

These tests are HOST-INDEPENDENT: they exercise registration, applicability,
impact prediction, and the NumericalProbe SIMULATED path (which encodes the
known divergence thresholds from the robustness report section 8) so they pass
on any machine regardless of whether torch / a GPU is installed. The real-torch
probe path and the patch functions are guarded behind a torch availability skip.
"""
from __future__ import annotations

import pytest

import vdg
from vdg import REGISTRY
from vdg.core.contracts import (
    DeviceCategory,
    DeviceProfile,
    DeviceSpec,
    LoadModel,
    Skill,
    SkillImpact,
    VideoDiTLoad,
)
from vdg.skills.repair import NumericalProbe
from vdg.skills.repair._common import low_precision_backend

REPAIR_SKILLS = ["gelu_fp32", "adaln_fp32", "rmsnorm_fp32", "softmax_fp32", "vae_fp32"]


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Inline fixtures (loads/ is empty in the foundation; build a minimal LTX load).
# --------------------------------------------------------------------------
class _LTX23(LoadModel):
    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="LTX-2.3", params_b=2.0, vae_compress=(8, 32, 32),
            patch_size=1, te_params_b=0.4, layers=28, hidden_dim=2048, heads=16,
            default_steps=30, supported_tasks=["t2v", "i2v"], vae_params_m=175.0,
            ffn_expansion=4.0,
        )


def _apple_silicon_device() -> DeviceProfile:
    for _name, cls in REGISTRY.all("device").items():
        dev = cls()
        if dev.spec().category == DeviceCategory.APPLE_SILICON:
            return dev
    pytest.skip("no Apple Silicon device plugin registered")


def _spec(category: str, backends: list[str], precisions: list[str]) -> DeviceSpec:
    return DeviceSpec(
        name="probe-spec", category=category, memory_gb=16.0,
        memory_bandwidth_gbps=200.0, compute_tflops={"fp32": 10.0, "bf16": 20.0},
        tdp_w=100.0, idle_power_w=5.0, supported_precisions=precisions,
        attention_backends=backends,
    )


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def test_repair_skills_registered():
    names = REGISTRY.names("skill")
    for n in REPAIR_SKILLS:
        assert n in names, "repair skill " + n + " not registered"


def test_repair_skills_kind_and_base():
    for n in REPAIR_SKILLS:
        cls = REGISTRY.get("skill", n)
        assert cls is not None
        inst = cls()
        assert inst.kind == "repair"
        assert isinstance(inst, Skill)


# --------------------------------------------------------------------------
# Applicability + prediction
# --------------------------------------------------------------------------
def test_applicable_on_apple_silicon():
    dev = _apple_silicon_device()
    load = _LTX23()
    for n in REPAIR_SKILLS:
        sk = REGISTRY.get("skill", n)()
        assert sk.applicable(dev, load) is True


def test_low_precision_backend_predicate():
    assert low_precision_backend(_spec(DeviceCategory.APPLE_SILICON, ["mlx_sdpa"], ["bf16"]))
    assert low_precision_backend(_spec(DeviceCategory.EDGE_NPU, ["int8"], ["int8", "bf16"]))
    assert low_precision_backend(_spec(DeviceCategory.CONSUMER_NV, ["int8"], ["int8", "fp16"]))
    assert not low_precision_backend(_spec(DeviceCategory.DATACENTER, ["flash"], ["bf16", "fp32"]))
    # Blackwell-style fp8/fp4 on consumer NV RELAXES the guard (report S7).
    assert not low_precision_backend(_spec(DeviceCategory.CONSUMER_NV, ["flash"], ["fp8", "nvfp4", "bf16", "fp32"]))


def test_predict_impact_well_signed():
    dev = _apple_silicon_device()
    load = _LTX23()
    for n in REPAIR_SKILLS:
        sk = REGISTRY.get("skill", n)()
        imp = sk.predict(dev, load)
        assert isinstance(imp, SkillImpact)
        # fp32 cast costs perf -> speedup <= 1.0; fixes black frames -> quality up.
        assert imp.speedup <= 1.0
        assert imp.quality_delta > 0.0
        assert imp.energy_ratio >= 1.0
        assert DeviceCategory.APPLE_SILICON in imp.applies_to
        assert DeviceCategory.EDGE_NPU in imp.applies_to


def test_default_config_returns_dict():
    for n in REPAIR_SKILLS:
        sk = REGISTRY.get("skill", n)()
        assert isinstance(sk.default_config(), dict)


# --------------------------------------------------------------------------
# NumericalProbe -- SIMULATED path (host-independent, threshold-encoded)
# --------------------------------------------------------------------------
def test_probe_simulated_mps_bf16_gelu_nan_and_adaln_divergence():
    r = NumericalProbe()._probe_simulated("mps", "bf16")
    assert r.simulated is True
    by_op = {x.op: x for x in r.results}
    # MPS bf16 fused GELU kernel bug -> NaN at |x|>=15.
    assert by_op["gelu_tanh"].status == "nan"
    assert by_op["gelu_tanh"].nan_count == 1
    # AdaLN (1+scale) catastrophic cancellation: -0.999 -> -1.0 in bf16.
    assert by_op["adln_modulate"].status == "divergence"
    # bf16 does not overflow at |x|=300 (large exponent range).
    assert by_op["rmsnorm"].status == "ok"


def test_probe_simulated_mps_fp16_rmsnorm_overflow():
    r = NumericalProbe()._probe_simulated("mps", "fp16")
    by_op = {x.op: x for x in r.results}
    # fp16 |x|=300 -> x^2=90000 > 65504 overflow -> NaN.
    assert by_op["rmsnorm"].status == "nan"
    # fp16 has 10 mantissa bits so -0.999 does NOT round to -1.0 -> no cancellation.
    assert by_op["adln_modulate"].status == "ok"
    # fp16 gelu at x=15 is fine (15^3=3375 < 65504, no overflow).
    assert by_op["gelu_tanh"].status == "ok"


def test_probe_simulated_cpu_fp32_all_ok():
    r = NumericalProbe()._probe_simulated("cpu", "fp32")
    for x in r.results:
        assert x.status == "ok", x.op + " should be ok in fp32"


def test_probe_fallback_for_unknown_device_is_simulated():
    # "coreml" is not a torch device -> probe must fall back to simulated.
    r = NumericalProbe().probe_ops("coreml", "bf16")
    assert r.simulated is True
    assert r.has_failure is True  # adaln cancellation still flagged


def test_probe_repair_skills_suggested_mapping():
    r = NumericalProbe()._probe_simulated("mps", "bf16")
    suggested = r.repair_skills_suggested
    assert "gelu_fp32" in suggested
    assert "adaln_fp32" in suggested


def test_probe_report_summary_nonempty():
    r = NumericalProbe().probe_ops("mps", "bf16")
    assert isinstance(r.summary, str) and r.summary


# --------------------------------------------------------------------------
# Patch functions (torch-dependent; skipped without torch).
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_patch_gelu_fixes_mps_bf16_nan_or_preserves_dtype():
    import torch
    import torch.nn as nn
    from vdg.skills.repair import patch_gelu

    g = nn.GELU(approximate="tanh")
    patch_gelu(g)
    assert getattr(g, "_vdg_patched", None) == "gelu_fp32"
    # Run on CPU (always available) to verify the non-mps path works and the
    # module stays callable; dtype preserved on output.
    x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    y = g(x)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_patch_adaln_guard_on_non_adaln_module():
    import torch
    import torch.nn as nn
    from vdg.skills.repair import patch_adaln

    # A plain Linear has no scale_shift_table -> returned unchanged, no patch.
    lin = nn.Linear(8, 8)
    out = patch_adaln(lin)
    assert out is lin
    assert getattr(lin, "_vdg_patched", None) is None


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_patch_vae_wraps_decode_and_forward():
    import torch
    import torch.nn as nn
    from vdg.skills.repair import patch_vae

    class _VAE(nn.Module):
        def forward(self, x):
            return x * 2.0

        def decode(self, x):
            return x * 2.0

    vae = _VAE()
    patch_vae(vae)
    assert getattr(vae, "_vdg_patched", None) == "vae_fp32"
    # Non-mps path delegates to the original decode.
    x = torch.tensor([1.0, 2.0])
    assert torch.equal(vae.decode(x), x * 2.0)
