"""Device plugins (phase 2).

Importing this package imports every device module so that classes decorated with
'@register_device' self-register into 'REGISTRY' on 'import vdg' (or
'REGISTRY.discover()'). Each module is grounded in the edge-deployment and
training research reports; see the per-module docstrings for spec provenance.

Modules:
  * detector      -- graceful host probing (NVML, MPS, powermetrics, Ascend, Jetson).
  * nvidia_dc     -- H100, H200, B200, GB300_NVL72 (datacenter).
  * consumer_nv   -- RTX 4090, RTX 5090, RTX 6000 Ada.
  * apple_silicon -- M4 Max, M3 Ultra, M2 Ultra (unified memory, no FP4).
  * jetson        -- Jetson Thor T5000/T4000, Jetson Orin 64 (edge).
  * npu           -- Ascend 910B, Cambricon MLU590, RK3588, Qualcomm Hexagon.

Helpers:
  * list_all_devices()      -- instantiate every registered DeviceProfile.
  * filter_by_category(cat) -- instantiate every registered profile in a category.
  * get_device(name)        -- instantiate a registered profile by registry name.
"""
from __future__ import annotations

from . import detector  # noqa: F401  (probes; import side effects are safe)
from . import nvidia_dc  # noqa: F401  (registers H100/H200/B200/GB300_NVL72)
from . import consumer_nv  # noqa: F401  (registers RTX 4090/5090/6000 Ada)
from . import apple_silicon  # noqa: F401  (registers M4 Max/M3 Ultra/M2 Ultra)
from . import jetson  # noqa: F401  (registers Thor T5000/T4000, Orin 64)
from . import npu  # noqa: F401  (registers Ascend/MLU590/RK3588/Hexagon)

from ..core.contracts import DeviceCategory, DeviceProfile
from ..core.registry import REGISTRY

__all__ = [
    "detector",
    "nvidia_dc",
    "consumer_nv",
    "apple_silicon",
    "jetson",
    "npu",
    "list_all_devices",
    "filter_by_category",
    "get_device",
]


def _instantiate(cls: type) -> DeviceProfile:
    """Instantiate a registered DeviceProfile class (zero-arg constructor)."""
    return cls()  # type: ignore[call-arg]


def list_all_devices() -> list[DeviceProfile]:
    """Return a fresh instance of every registered DeviceProfile, sorted by name."""
    classes = REGISTRY.all("device")
    pairs = sorted(classes.items(), key=lambda kv: kv[0])
    return [_instantiate(cls) for _, cls in pairs]


def filter_by_category(category: str) -> list[DeviceProfile]:
    """Return instances of every registered profile whose spec category matches.

    'category' is compared case-insensitively against 'DeviceSpec.category'.
    """
    needle = (category or "").lower()
    out: list[DeviceProfile] = []
    for profile in list_all_devices():
        try:
            if profile.spec().category.lower() == needle:
                out.append(profile)
        except Exception:
            # A broken profile must never break listing.
            continue
    return out


def get_device(name: str) -> DeviceProfile | None:
    """Instantiate a registered DeviceProfile by its registry name, or None."""
    cls = REGISTRY.get("device", name)
    if cls is None:
        return None
    return _instantiate(cls)
