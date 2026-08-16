"""Tests for the shipped device plugins (vdg.devices).

Every registered DeviceSpec is validated end-to-end: spec fields are sane,
peak_flops matches compute_tflops, unsupported precisions carry zero TFLOPS
(and raise on peak_flops), and the FP4 hardware differentiation (RTX 5090
Blackwell vs Apple M4 Max) is enforced. Counts use subset/registry-name checks
because test_registry.py registers throwaway devices into the global REGISTRY
at collection time (same defensive pattern as test_video_dit_loads.py).
"""
from __future__ import annotations

import pytest

from vdg import DeviceCategory
from vdg.core.contracts import DeviceSpec
from vdg.core.registry import REGISTRY
from vdg.devices import filter_by_category, get_device, list_all_devices

# The 17 shipped device plugins, keyed by their registry name (== class name).
EXPECTED_DEVICES: set[str] = {
    # consumer_nv
    "RTX4090", "RTX5090", "RTX6000_Ada",
    # apple_silicon
    "M4_Max", "M3_Ultra", "M2_Ultra",
    # jetson (edge NPU)
    "Jetson_Thor_T5000", "Jetson_Thor_T4000", "Jetson_Orin_64",
    # npu (edge NPU)
    "Ascend_910B", "Cambricon_MLU590", "RK3588", "Qualcomm_Hexagon",
    # nvidia_dc (datacenter)
    "H100", "H200", "B200", "GB300_NVL72",
}

VALID_CATEGORIES: set[str] = {
    DeviceCategory.CONSUMER_NV,
    DeviceCategory.APPLE_SILICON,
    DeviceCategory.EDGE_NPU,
    DeviceCategory.DATACENTER,
}

# Devices that ship native FP4 tensor cores (Blackwell-class).
FP4_DEVICES: set[str] = {"RTX5090", "B200", "GB300_NVL72", "Jetson_Thor_T5000", "Jetson_Thor_T4000"}


# ---------------------------------------------------------------------------
# Registration + count
# ---------------------------------------------------------------------------
def test_all_expected_devices_registered():
    names = set(REGISTRY.names("device"))
    missing = EXPECTED_DEVICES - names
    assert not missing, "Missing device plugins: " + ", ".join(sorted(missing))


def test_list_all_devices_returns_expected_count():
    """list_all_devices returns exactly the 17 shipped DeviceProfile instances.

    Other tests (test_registry.py) register throwaway non-DeviceProfile classes
    into the global REGISTRY, so we filter to the shipped set by registry name.
    """
    shipped = [d for d in list_all_devices() if d.registry_name() in EXPECTED_DEVICES]
    assert len(shipped) == len(EXPECTED_DEVICES) == 17


def test_list_all_devices_returns_deviceprofile_instances():
    for d in list_all_devices():
        if d.registry_name() not in EXPECTED_DEVICES:
            continue  # skip throwaway Registrable doubles from test_registry
        spec = d.spec()
        assert isinstance(spec, DeviceSpec)


# ---------------------------------------------------------------------------
# Spec validity (every shipped device)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(EXPECTED_DEVICES))
def test_every_device_spec_valid(name):
    dev = get_device(name)
    assert dev is not None, name + " not registered"
    spec = dev.spec()
    # Identity / category.
    assert spec.name, name + " has empty spec.name"
    assert spec.category in VALID_CATEGORIES, name + " bad category " + repr(spec.category)
    # Memory + bandwidth.
    assert spec.memory_gb > 0, name + " memory_gb <= 0"
    assert spec.memory_bandwidth_gbps > 0, name + " bandwidth <= 0"
    # Compute + power.
    assert spec.compute_tflops, name + " has no compute_tflops entries"
    for prec, tf in spec.compute_tflops.items():
        assert tf > 0, name + " compute_tflops[" + prec + "] <= 0"
    assert spec.tdp_w > 0, name + " tdp_w <= 0"
    assert spec.idle_power_w >= 0, name + " idle_power_w < 0"
    # Precision / backend lists non-empty.
    assert spec.supported_precisions, name + " no supported_precisions"
    assert spec.attention_backends, name + " no attention_backends"


@pytest.mark.parametrize("name", sorted(EXPECTED_DEVICES))
def test_supported_precisions_subset_of_known(name):
    known = {"fp32", "tf32", "bf16", "fp16", "fp8", "fp4", "nvfp4", "int8", "int4"}
    spec = get_device(name).spec()
    for p in spec.supported_precisions:
        assert p.lower() in known, name + " unknown precision " + repr(p)


# ---------------------------------------------------------------------------
# peak_flops / supports semantics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(EXPECTED_DEVICES))
def test_peak_flops_matches_compute_tflops(name):
    spec = get_device(name).spec()
    for prec, tf in spec.compute_tflops.items():
        assert spec.peak_flops(prec) == pytest.approx(tf * 1e12), (
            name + " peak_flops(" + prec + ") mismatch"
        )


@pytest.mark.parametrize("name", sorted(EXPECTED_DEVICES))
def test_supports_matches_supported_precisions(name):
    spec = get_device(name).spec()
    for p in spec.supported_precisions:
        assert spec.supports(p) is True, name + " should support " + p
    # A precision not in the list must report unsupported.
    assert spec.supports("fp4_bogus") is False


def test_unsupported_precision_has_zero_tflops():
    """An unsupported precision is absent from compute_tflops -> 0 TFLOPS."""
    spec = get_device("M4_Max").spec()
    # M4 Max ships no FP4 path at all.
    assert "fp4" not in spec.compute_tflops
    assert "nvfp4" not in spec.compute_tflops
    assert spec.compute_tflops.get("fp4", 0) == 0
    assert spec.compute_tflops.get("nvfp4", 0) == 0
    # H100 has no FP4 either.
    h100 = get_device("H100").spec()
    assert h100.compute_tflops.get("fp4", 0) == 0
    assert h100.compute_tflops.get("nvfp4", 0) == 0


def test_peak_flops_raises_for_unsupported_precision():
    spec = get_device("M4_Max").spec()
    with pytest.raises(ValueError):
        spec.peak_flops("fp4")
    with pytest.raises(ValueError):
        spec.peak_flops("nvfp4")


# ---------------------------------------------------------------------------
# FP4 hardware differentiation (RTX 5090 Blackwell vs Apple M4 Max)
# ---------------------------------------------------------------------------
def test_rtx5090_supports_fp4():
    spec = get_device("RTX5090").spec()
    assert spec.supports("fp4") is True
    assert spec.supports("nvfp4") is True
    assert "fp4" in spec.compute_tflops
    assert "nvfp4" in spec.compute_tflops
    assert spec.compute_tflops["fp4"] > 0
    assert spec.compute_tflops["nvfp4"] > 0
    # RTX 5090 is a Blackwell consumer card (sage3 backend for FP4 attention).
    assert "sage3" in spec.attention_backends


def test_m4_max_does_not_support_fp4():
    spec = get_device("M4_Max").spec()
    assert spec.supports("fp4") is False
    assert spec.supports("nvfp4") is False
    assert "fp4" not in spec.compute_tflops
    assert "nvfp4" not in spec.compute_tflops
    # Apple Silicon ships no FP4 hardware (report: no Blackwell-style FP4 TC).
    assert spec.category == DeviceCategory.APPLE_SILICON
    assert spec.unified_memory is True


def test_fp4_devices_and_non_fp4_devices_partition():
    for name in EXPECTED_DEVICES:
        spec = get_device(name).spec()
        has_fp4 = spec.supports("fp4")
        if name in FP4_DEVICES:
            assert has_fp4 is True, name + " should support fp4"
        # Apple Silicon never has FP4.
        if spec.category == DeviceCategory.APPLE_SILICON:
            assert has_fp4 is False, name + " (Apple Silicon) must not support fp4"


# ---------------------------------------------------------------------------
# Helpers: get_device / filter_by_category
# ---------------------------------------------------------------------------
def test_get_device_returns_instance():
    dev = get_device("RTX5090")
    assert dev is not None
    assert dev.spec().name == "RTX 5090"


def test_get_device_unknown_returns_none():
    assert get_device("nonexistent_device_xyz") is None


def test_filter_by_category_consumer_nv():
    devs = filter_by_category(DeviceCategory.CONSUMER_NV)
    names = {d.registry_name() for d in devs}
    assert {"RTX4090", "RTX5090", "RTX6000_Ada"} <= names
    # All returned devices are actually consumer NV.
    for d in devs:
        if d.registry_name() in EXPECTED_DEVICES:
            assert d.spec().category == DeviceCategory.CONSUMER_NV


def test_filter_by_category_apple_silicon():
    devs = filter_by_category(DeviceCategory.APPLE_SILICON)
    names = {d.registry_name() for d in devs}
    assert {"M4_Max", "M3_Ultra", "M2_Ultra"} <= names


def test_filter_by_category_datacenter():
    devs = filter_by_category(DeviceCategory.DATACENTER)
    names = {d.registry_name() for d in devs}
    assert {"H100", "H200", "B200", "GB300_NVL72"} <= names


def test_filter_by_category_edge_npu():
    devs = filter_by_category(DeviceCategory.EDGE_NPU)
    names = {d.registry_name() for d in devs}
    assert {
        "Jetson_Thor_T5000", "Jetson_Thor_T4000", "Jetson_Orin_64",
        "Ascend_910B", "Cambricon_MLU590", "RK3588", "Qualcomm_Hexagon",
    } <= names


# ---------------------------------------------------------------------------
# Grounded spec anchors (a few headline numbers from the reports)
# ---------------------------------------------------------------------------
def test_rtx5090_blackwell_anchors():
    spec = get_device("RTX5090").spec()
    assert spec.memory_gb == pytest.approx(32.0)
    assert spec.memory_bandwidth_gbps == pytest.approx(1790.0)
    assert spec.tdp_w == pytest.approx(575.0)
    # FP4 dense-equivalent (3352 AI TOPS FP4-sparse halved to dense).
    assert spec.compute_tflops["fp4"] == pytest.approx(1676.0)


def test_h100_hopper_anchors():
    spec = get_device("H100").spec()
    assert spec.memory_gb == pytest.approx(80.0)
    assert spec.memory_bandwidth_gbps == pytest.approx(3350.0)
    assert spec.tdp_w == pytest.approx(700.0)
    assert spec.compute_tflops["fp8"] == pytest.approx(1979.0)
    # Hopper has no FP4 tensor cores.
    assert not spec.supports("fp4")


def test_jetson_thor_is_edge_npu_with_fp4():
    spec = get_device("Jetson_Thor_T5000").spec()
    assert spec.category == DeviceCategory.EDGE_NPU
    assert spec.unified_memory is True
    assert spec.supports("fp4") is True
    assert spec.memory_bandwidth_gbps == pytest.approx(273.0)


def test_ascend_910b_anchors():
    spec = get_device("Ascend_910B").spec()
    assert spec.category == DeviceCategory.EDGE_NPU
    assert spec.supports("int8") is True
    assert not spec.supports("fp4")
    assert spec.compute_tflops["int8"] == pytest.approx(640.0)


def test_is_available_and_measure_power_never_raise():
    """is_available / measure_power must degrade gracefully on any host."""
    for name in EXPECTED_DEVICES:
        dev = get_device(name)
        # Must return a bool / float-or-None, never raise.
        avail = dev.is_available()
        assert isinstance(avail, bool)
        power = dev.measure_power()
        assert power is None or (isinstance(power, (int, float)) and power >= 0)
