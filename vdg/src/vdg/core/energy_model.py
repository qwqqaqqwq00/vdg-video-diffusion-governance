"""Energy models for VDG.

Energy models turn a compute time and utilization into joules. They are
themselves pluggable (``EnergyModel`` subclasses register via
``@register_energy_model``), so a deployment can swap a TDP model for a measured
one without touching the simulator.

Grounded wattage anchors (from the edge-deployment / training reports):
  RTX 5090 575W, RTX 4090 450W, RTX 6000 Ada 300W, Mac Studio <=480W system,
  Jetson Thor 40-130W, Jetson Orin 15-60W, RK3588 5-15W, H100 700W, H200 700W,
  B200 ~1000W, Ascend 910B ~310W.
"""
from __future__ import annotations

from typing import Any

from .registry import Registrable, register_energy_model
from .contracts import DeviceSpec

__all__ = ["EnergyModel", "TDPEnergyModel", "MeasuredEnergyModel"]


class EnergyModel(Registrable):
    """Base class for energy models. Subclasses implement ``energy``."""

    def energy(
        self,
        device_spec: DeviceSpec,
        compute_time_s: float,
        utilization: float = 0.75,
    ) -> float:
        """Return energy in joules for ``compute_time_s`` at ``utilization``."""
        raise NotImplementedError("EnergyModel.energy must be implemented")


@register_energy_model("tdp")
class TDPEnergyModel(EnergyModel):
    """TDP-based energy model with idle-to-peak linear scaling.

    power = idle_power_w + (tdp_w - idle_power_w) * clamp(utilization, 0, 1)
    energy = power * compute_time_s

    idle defaults to 0.1 * tdp (matches the spec: idle=0.1*tdp, peak=tdp). If
    the device spec carries an explicit ``idle_power_w`` it is used directly.
    """

    def energy(
        self,
        device_spec: DeviceSpec,
        compute_time_s: float,
        utilization: float = 0.75,
    ) -> float:
        if compute_time_s < 0:
            raise ValueError("compute_time_s must be >= 0")
        u = max(0.0, min(1.0, float(utilization)))
        tdp = device_spec.tdp_w
        idle = device_spec.idle_power_w if device_spec.idle_power_w > 0 else 0.1 * tdp
        power = idle + (tdp - idle) * u
        return power * compute_time_s


@register_energy_model("measured")
class MeasuredEnergyModel(EnergyModel):
    """Energy model that reads live GPU power when possible, else falls back.

    On an NVIDIA GPU with ``pynvml`` installed, it samples instantaneous power
    during the interval and integrates. If pynvml is missing, no NVIDIA device is
    present, or any error occurs (Mac/NPU/CI), it transparently falls back to
    ``TDPEnergyModel``. This MUST NOT crash on hosts without NVIDIA hardware --
    that is the whole point of the graceful fallback.
    """

    def __init__(self) -> None:
        self._fallback = TDPEnergyModel()
        self._pynvml: Any = None
        self._pynvml_ok: bool | None = None

    def _try_init_pynvml(self) -> bool:
        if self._pynvml_ok is not None:
            return self._pynvml_ok
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            # Confirm at least one device is reachable.
            pynvml.nvmlDeviceGetCount()
            self._pynvml = pynvml
            self._pynvml_ok = True
        except Exception:
            # Any failure (no pynvml, no NVIDIA driver, permission error, NPU
            # host, Mac) -> fall back. Never raise.
            self._pynvml = None
            self._pynvml_ok = False
        return self._pynvml_ok

    def _instantaneous_power_w(self, device_spec: DeviceSpec) -> float | None:
        """Read instantaneous power for the matching NVIDIA device, else None."""
        if not self._try_init_pynvml() or self._pynvml is None:
            return None
        pynvml = self._pynvml
        try:
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                # Match by name substring (case-insensitive) against the spec.
                if device_spec.name.lower() in str(name).lower():
                    return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception:
            return None
        return None

    def energy(
        self,
        device_spec: DeviceSpec,
        compute_time_s: float,
        utilization: float = 0.75,
    ) -> float:
        if compute_time_s < 0:
            raise ValueError("compute_time_s must be >= 0")
        power = self._instantaneous_power_w(device_spec)
        if power is None:
            return self._fallback.energy(device_spec, compute_time_s, utilization)
        return power * compute_time_s
