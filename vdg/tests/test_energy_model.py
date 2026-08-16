"""Energy model tests."""
from __future__ import annotations

from vdg import DeviceSpec, MeasuredEnergyModel, TDPEnergyModel


def _spec(tdp=450.0, idle=45.0) -> DeviceSpec:
    return DeviceSpec(
        name="RTX 4090", category="consumer_nv", memory_gb=24.0,
        memory_bandwidth_gbps=1008.0, compute_tflops={"bf16": 165.0},
        tdp_w=tdp, idle_power_w=idle, supported_precisions=["bf16"],
        attention_backends=["math"],
    )


def test_tdp_idle_power():
    # utilization 0 -> idle power.
    spec = _spec()
    m = TDPEnergyModel()
    e = m.energy(spec, compute_time_s=10.0, utilization=0.0)
    assert abs(e - 45.0 * 10.0) < 1e-6


def test_tdp_peak_power():
    # utilization 1 -> tdp.
    spec = _spec()
    m = TDPEnergyModel()
    e = m.energy(spec, compute_time_s=10.0, utilization=1.0)
    assert abs(e - 450.0 * 10.0) < 1e-6


def test_tdp_linear_interpolation():
    spec = _spec(tdp=450.0, idle=45.0)
    m = TDPEnergyModel()
    e = m.energy(spec, compute_time_s=10.0, utilization=0.5)
    # power = 45 + (450-45)*0.5 = 247.5
    assert abs(e - 247.5 * 10.0) < 1e-6


def test_tdp_idle_defaults_to_tenth_of_tdp():
    # When idle_power_w == 0, idle defaults to 0.1 * tdp.
    spec = _spec(tdp=450.0, idle=0.0)
    m = TDPEnergyModel()
    e = m.energy(spec, compute_time_s=10.0, utilization=0.0)
    assert abs(e - 45.0 * 10.0) < 1e-6


def test_tdp_utilization_clamped():
    spec = _spec()
    m = TDPEnergyModel()
    below = m.energy(spec, 10.0, utilization=-1.0)
    above = m.energy(spec, 10.0, utilization=2.0)
    idle = m.energy(spec, 10.0, utilization=0.0)
    peak = m.energy(spec, 10.0, utilization=1.0)
    assert below == idle
    assert above == peak


def test_tdp_negative_time_raises():
    with __import__("pytest").raises(ValueError):
        TDPEnergyModel().energy(_spec(), -1.0)


def test_measured_falls_back_to_tdp_on_non_nvidia_host():
    # On a Mac/CI host without pynvml or NVIDIA driver, MeasuredEnergyModel
    # must NOT crash and must equal the TDP model result.
    spec = _spec()
    measured = MeasuredEnergyModel()
    tdp = TDPEnergyModel()
    e_m = measured.energy(spec, compute_time_s=10.0, utilization=0.75)
    e_t = tdp.energy(spec, compute_time_s=10.0, utilization=0.75)
    assert e_m == e_t


def test_measured_init_is_idempotent():
    m = MeasuredEnergyModel()
    first = m._try_init_pynvml()
    second = m._try_init_pynvml()
    assert first == second
