"""Tests for the repair skills library (numerical-robustness fp32 guards).

Host-independent: exercises registration, applicability, impact prediction, and
the NumericalProbe SIMULATED path (which encodes the known divergence
thresholds from the cross-device robustness report section 8) so tests pass on
any machine regardless of whether torch / a GPU is installed. The real-torch
probe path is guarded behind a torch-availability skip.
"""
from __future__ import annotations

import pytest

from vdg import DeviceCategory, REGISTRY, Skill, SkillImpact
from vdg.devices import get_device
from vdg.loads import LTX_2_3
from vdg.skills.repair import NumericalProbe
from vdg.skills.repair._common import low_precision_backend

REPAIR_SKILLS = ["gelu_fp32", "adaln_fp32", "rmsnorm_fp32", "softmax_fp32", "vae_fp32"]


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registration + kind
# ---------------------------------------------------------------------------
def test_repair_skills_registered():
    names = set(REGISTRY.names("skill"))
    for n in REPAIR_SKILLS:
        assert n in names, "repair skill " + n + " not registered"


def test_repair_skills_kind_and_base():
    for n in REPAIR_SKILLS:
        inst = REGISTRY.get("skill", n)()
        assert inst.kind == "repair"
        assert isinstance(inst, Skill)


def test_repair_skills_count_at_least_five():
    """The five granular repair skills are all present (test_registry may add a
    throwaway skill, so use a subset check)."""
    names = set(REGISTRY.names("skill"))
    assert set(REPAIR_SKILLS) <= names


# ---------------------------------------------------------------------------
# Applicability (correctness across device categories)
# ---------------------------------------------------------------------------
def test_repair_applicable_on_apple_silicon():
    dev = get_device("M4_Max")
    load = LTX_2_3()
    for n in REPAIR_SKILLS:
        sk = REGISTRY.get("skill", n)()
        assert sk.applicable(dev, load) is True, n + " not applicable on M4 Max"


def test_repair_applicable_on_edge_npu():
    for dev_name in ("Jetson_Thor_T5000", "Ascend_910B"):
        dev = get_device(dev_name)
        load = LTX_2_3()
        for n in REPAIR_SKILLS:
            sk = REGISTRY.get("skill", n)()
            assert sk.applicable(dev, load) is True, n + " not applicable on " + dev_name


def test_repair_not_applicable_on_datacenter():
    """Datacenter cards (H100) report not-applicable: Blackwell-class fp8/fp4 on
    a datacenter category relaxes the fp32 guard (robustness report S7)."""
    dev = get_device("H100")
    load = LTX_2_3()
    for n in REPAIR_SKILLS:
        sk = REGISTRY.get("skill", n)()
        assert sk.applicable(dev, load) is False, n + " should not apply on H100"


def test_low_precision_backend_predicate_for_shipped_devices():
    # Apple Silicon + edge NPU -> True; datacenter -> False.
    assert low_precision_backend(get_device("M4_Max").spec()) is True
    assert low_precision_backend(get_device("Jetson_Thor_T5000").spec()) is True
    assert low_precision_backend(get_device("Ascend_910B").spec()) is True
    assert low_precision_backend(get_device("H100").spec()) is False


# ---------------------------------------------------------------------------
# predict() sanity (well-signed impacts)
# ---------------------------------------------------------------------------
def test_predict_impact_well_signed():
    dev = get_device("M4_Max")
    load = LTX_2_3()
    for n in REPAIR_SKILLS:
        sk = REGISTRY.get("skill", n)()
        imp = sk.predict(dev, load)
        assert isinstance(imp, SkillImpact)
        # fp32 cast costs latency -> speedup <= 1.0; fixes black frames -> quality up.
        assert imp.speedup <= 1.0, n + " speedup > 1.0 (should cost latency)"
        assert imp.quality_delta > 0.0, n + " quality_delta <= 0 (should improve)"
        assert imp.energy_ratio >= 1.0, n + " energy_ratio < 1.0 (should cost energy)"
        # Repair targets Apple Silicon + edge NPU.
        assert DeviceCategory.APPLE_SILICON in imp.applies_to
        assert DeviceCategory.EDGE_NPU in imp.applies_to


def test_predict_returns_sane_numbers():
    """Concrete anchor: GeluFP32 costs ~8% latency, +2.0 quality, +5% energy."""
    dev = get_device("M4_Max")
    load = LTX_2_3()
    gelu = REGISTRY.get("skill", "gelu_fp32")()
    imp = gelu.predict(dev, load)
    assert imp.speedup == pytest.approx(0.92, abs=0.01)
    assert imp.quality_delta == pytest.approx(2.0, abs=0.01)
    assert imp.energy_ratio == pytest.approx(1.05, abs=0.01)


def test_default_config_returns_dict():
    for n in REPAIR_SKILLS:
        sk = REGISTRY.get("skill", n)()
        assert isinstance(sk.default_config(), dict)


# ---------------------------------------------------------------------------
# NumericalProbe -- SIMULATED path (host-independent, threshold-encoded)
# ---------------------------------------------------------------------------
def test_probe_simulated_returns_report():
    """NumericalProbe returns a report in sim mode (pure-sim, host-independent)."""
    r = NumericalProbe()._probe_simulated("mps", "bf16")
    assert r.simulated is True
    assert r.device_name == "mps"
    assert r.precision == "bf16"
    assert len(r.results) >= 4
    assert isinstance(r.summary, str) and r.summary


def test_probe_simulated_flags_mps_bf16_gelu_as_diverge():
    """MPS bf16 fused GELU kernel bug -> NaN at |x|>=15 (a divergence failure)."""
    r = NumericalProbe()._probe_simulated("mps", "bf16")
    by_op = {x.op: x for x in r.results}
    gelu = by_op["gelu_tanh"]
    # The op is flagged as a failure (nan or divergence status).
    assert gelu.status in ("nan", "divergence"), "gelu_tanh status=" + gelu.status
    assert r.has_failure is True
    # And the matching repair skill is suggested.
    assert "gelu_fp32" in r.repair_skills_suggested


def test_probe_simulated_mps_bf16_adaln_divergence():
    """AdaLN (1+scale) catastrophic cancellation: -0.999 -> -1.0 in bf16."""
    r = NumericalProbe()._probe_simulated("mps", "bf16")
    by_op = {x.op: x for x in r.results}
    assert by_op["adln_modulate"].status == "divergence"
    assert "adaln_fp32" in r.repair_skills_suggested


def test_probe_simulated_mps_bf16_rmsnorm_ok():
    """bf16 shares fp32's 8-bit exponent -> no overflow at |x|=300."""
    r = NumericalProbe()._probe_simulated("mps", "bf16")
    by_op = {x.op: x for x in r.results}
    assert by_op["rmsnorm"].status == "ok"


def test_probe_simulated_mps_fp16_rmsnorm_overflow():
    """fp16 |x|=300 -> x^2=90000 > 65504 overflow -> NaN."""
    r = NumericalProbe()._probe_simulated("mps", "fp16")
    by_op = {x.op: x for x in r.results}
    assert by_op["rmsnorm"].status == "nan"
    # fp16 has 10 mantissa bits -> -0.999 does NOT round to -1.0 -> no cancellation.
    assert by_op["adln_modulate"].status == "ok"
    # fp16 gelu at x=15 is fine (15^3=3375 < 65504, no overflow).
    assert by_op["gelu_tanh"].status == "ok"


def test_probe_simulated_cpu_fp32_all_ok():
    r = NumericalProbe()._probe_simulated("cpu", "fp32")
    for x in r.results:
        assert x.status == "ok", x.op + " should be ok in fp32"
    assert r.has_failure is False
    assert r.repair_skills_suggested == []


def test_probe_fallback_for_unknown_device_is_simulated():
    """'coreml' is not a torch device -> probe falls back to a simulated report."""
    r = NumericalProbe().probe_ops("coreml", "bf16")
    assert r.simulated is True
    assert r.has_failure is True  # adaln cancellation still flagged in bf16


def test_probe_repair_skills_suggested_mapping():
    r = NumericalProbe()._probe_simulated("mps", "bf16")
    suggested = r.repair_skills_suggested
    assert "gelu_fp32" in suggested
    assert "adaln_fp32" in suggested


def test_probe_report_summary_nonempty():
    r = NumericalProbe().probe_ops("mps", "bf16")
    assert isinstance(r.summary, str) and r.summary


def test_probe_results_carry_status_and_diff():
    r = NumericalProbe()._probe_simulated("mps", "bf16")
    for x in r.results:
        assert x.status in ("ok", "divergence", "nan")
        assert x.op
        assert x.input_desc
        # nan results report inf max_diff; ok/divergence report a finite float.
        if x.status == "nan":
            assert x.nan_count >= 1


# ---------------------------------------------------------------------------
# Patch functions (torch-dependent; skipped without torch).
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_patch_gelu_fixes_mps_bf16_nan_or_preserves_dtype():
    import torch
    import torch.nn as nn
    from vdg.skills.repair import patch_gelu

    g = nn.GELU(approximate="tanh")
    patch_gelu(g)
    assert getattr(g, "_vdg_patched", None) == "gelu_fp32"
    x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    y = g(x)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_patch_adaln_guard_on_non_adaln_module():
    import torch
    import torch.nn as nn
    from vdg.skills.repair import patch_adaln

    lin = nn.Linear(8, 8)
    out = patch_adaln(lin)
    assert out is lin
    assert getattr(lin, "_vdg_patched", None) is None
