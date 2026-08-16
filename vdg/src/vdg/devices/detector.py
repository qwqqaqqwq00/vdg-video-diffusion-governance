"""Host hardware detection for VDG device profiles.

Every public function here degrades gracefully: on any failure (missing
optional library such as pynvml/torch, no matching device, wrong OS, permission
error, subprocess timeout) it returns False / None / [] and NEVER raises.
Device profiles call these from 'is_available' and 'measure_power' so that
importing, listing, or simulating with VDG on ANY host (a Mac, an NPU box, a CI
runner, or a GPU server) cannot crash.

The four functions named in the device-profile task spec are:
  * 'try_nvml'               -- is any NVIDIA GPU reachable via NVML?
  * 'try_mps'                -- is Apple's Metal/MPS backend available?
  * 'try_apple_powermetrics' -- is the macOS 'powermetrics' tool present?
  * 'try_ascend'             -- is a Huawei Ascend NPU reachable?

A few extra helpers ('nvml_device_names', 'nvml_has_name',
'nvml_power_for', 'apple_chip_name', 'try_jetson') support the per-device
'is_available' / 'measure_power' logic. They are equally graceful.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess

__all__ = [
    "try_nvml",
    "nvml_device_names",
    "nvml_has_name",
    "nvml_power_for",
    "try_mps",
    "apple_chip_name",
    "try_apple_powermetrics",
    "try_ascend",
    "try_jetson",
]


# --------------------------------------------------------------------------
# Internal subprocess helper
# --------------------------------------------------------------------------
def _run(args: list[str], timeout: float = 2.0) -> str:
    """Run a command and return combined stdout+stderr, or "" on any failure."""
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return ""


# --------------------------------------------------------------------------
# NVIDIA NVML (datacenter, consumer, and some Jetson discrete GPUs)
# --------------------------------------------------------------------------
# Module-level cache so repeated is_available()/measure_power() calls across
# many device profiles do not re-init NVML. 'ok' is None = unprobed.
_NVML_STATE: dict[str, object] = {"pynvml": None, "ok": None, "names": []}


def _nvml_init() -> bool:
    """Lazily initialize NVML and enumerate device names. Idempotent."""
    if _NVML_STATE["ok"] is not None:
        return bool(_NVML_STATE["ok"])
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        names: list[str] = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            raw = pynvml.nvmlDeviceGetName(handle)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            names.append(str(raw))
        _NVML_STATE["pynvml"] = pynvml
        _NVML_STATE["names"] = names
        _NVML_STATE["ok"] = True
        return True
    except Exception:
        # pynvml not installed, no NVIDIA driver, permission denied, etc.
        _NVML_STATE["pynvml"] = None
        _NVML_STATE["names"] = []
        _NVML_STATE["ok"] = False
        return False


def try_nvml() -> bool:
    """True if any NVIDIA GPU is reachable via NVML (pynvml installed + driver)."""
    return _nvml_init()


def nvml_device_names() -> list[str]:
    """Marketing names of all NVML-visible GPUs (e.g. 'NVIDIA H100 80GB HBM3')."""
    if not _nvml_init():
        return []
    names = _NVML_STATE.get("names")
    if not isinstance(names, list):
        return []
    return list(names)


def nvml_has_name(name_substring: str) -> bool:
    """True if an NVML-visible GPU name contains 'name_substring' (case-insensitive).

    An empty substring means 'any NVIDIA GPU present'.
    """
    needle = (name_substring or "").lower()
    names = nvml_device_names()
    if not needle:
        return bool(names)
    return any(needle in nm.lower() for nm in names)


def nvml_power_for(name_substring: str) -> float | None:
    """Instantaneous power draw in watts for the matching NVIDIA GPU, else None.

    'name_substring' is matched case-insensitively against NVML device names
    (e.g. '5090' matches 'NVIDIA GeForce RTX 5090'). An empty substring returns
    the power of the first visible GPU. Returns None when NVML is unavailable,
    no device matches, or power read is unsupported -- never raises.
    """
    if not _nvml_init():
        return None
    pynvml = _NVML_STATE.get("pynvml")
    if pynvml is None:
        return None
    needle = (name_substring or "").lower()
    try:
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            raw = pynvml.nvmlDeviceGetName(handle)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            name = str(raw).lower()
            if needle in name:
                # nvmlDeviceGetPowerUsage returns milliwatts.
                return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------
# Apple Silicon (Metal Performance Shaders / MLX)
# --------------------------------------------------------------------------
def try_mps() -> bool:
    """True if Apple's MPS/Metal GPU backend is available on this host."""
    # Prefer torch's MPS backend when torch is installed (most reliable signal).
    try:
        import torch  # type: ignore[import-not-found]

        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and callable(getattr(mps, "is_available", None)):
            if mps.is_available():
                return True
    except Exception:
        pass
    # Fallback: Apple Silicon Mac with the Metal framework present.
    if platform.system() != "Darwin":
        return False
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return False
    return os.path.exists("/System/Library/Frameworks/Metal.framework")


def apple_chip_name() -> str | None:
    """The host Apple chip marketing name (e.g. 'Apple M4 Max'), or None."""
    if platform.system() != "Darwin":
        return None
    # sysctl brand string is typically 'Apple M4 Max' on Apple Silicon.
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    if brand and brand.lower().startswith("apple"):
        return brand
    # Fallback: parse system_profiler output ('Chip: Apple M3 Ultra').
    out = _run(["system_profiler", "SPHardwareDataType"], timeout=4.0)
    for line in out.splitlines():
        line = line.strip()
        if line.lower().startswith("chip:"):
            chip = line.split(":", 1)[1].strip()
            if chip:
                return chip
    return None


def try_apple_powermetrics() -> bool:
    """True if the macOS 'powermetrics' utility is on PATH (needs root to run)."""
    if platform.system() != "Darwin":
        return False
    return shutil.which("powermetrics") is not None


# --------------------------------------------------------------------------
# Huawei Ascend NPU
# --------------------------------------------------------------------------
def try_ascend() -> bool:
    """True if a Huawei Ascend NPU is reachable (torch_npu or driver present)."""
    try:
        import torch_npu  # type: ignore[import-not-found]

        npu = getattr(torch_npu, "npu", None)
        if npu is not None and callable(getattr(npu, "is_available", None)):
            if npu.is_available():
                return True
    except Exception:
        pass
    # Fallback: look for the Ascend driver/software install paths.
    for path in ("/usr/local/Ascend", "/etc/ascend_install.info"):
        if os.path.exists(path):
            return True
    return False


# --------------------------------------------------------------------------
# NVIDIA Jetson (Tegra SoCs -- often NOT visible to desktop NVML)
# --------------------------------------------------------------------------
def try_jetson() -> bool:
    """True if the host is an NVIDIA Jetson Tegra module."""
    if platform.system() != "Linux":
        return False
    model = _run(["cat", "/proc/device-tree/model"]).strip()
    if model and "jetson" in model.lower():
        return True
    if os.path.exists("/etc/nv_tegra_release"):
        return True
    return False
