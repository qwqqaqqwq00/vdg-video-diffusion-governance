"""Pluggable plugin registry for VDG.

This is the ComfyUI/pytest-style decorator registry. Concrete devices, loads,
skills and energy models register themselves at import time via the
``@register_*`` decorators. A user who drops a new file decorated with
``@register_device`` into ``vdg/devices/`` and imports the package (or calls
``REGISTRY.discover()``) automatically makes it available to the simulator and
governance agents -- no central edit required.

Design notes
------------
* ``Registry`` stores classes keyed by ``(kind, name)``. It is intentionally
  generic (it never imports the contract base classes) so there is no circular
  import between ``registry`` and ``contracts``.
* ``Registrable`` is a lightweight mixin marking a class as registry-managed and
  carrying the ``__registry_kind__`` / ``__registry_name__`` metadata that the
  decorators stamp on it.
* The decorators support BOTH usage styles::

      @register_device                    # bare: name derived from class __name__
      class RTX5090(DeviceProfile): ...

      @register_device("rtx_5090")        # named: explicit registry name
      class RTX5090(DeviceProfile): ...
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Iterator

__all__ = [
    "Registrable",
    "Registry",
    "REGISTRY",
    "register_device",
    "register_load",
    "register_skill",
    "register_energy_model",
]

# Canonical registry kinds. Keeping them as constants makes typos surface early
# and lets ``Registry.all`` / ``Registry.get`` validate the kind argument.
KIND_DEVICE = "device"
KIND_LOAD = "load"
KIND_SKILL = "skill"
KIND_ENERGY_MODEL = "energy_model"
_ALL_KINDS = (KIND_DEVICE, KIND_LOAD, KIND_SKILL, KIND_ENERGY_MODEL)

# Subpackages that ``discover()`` imports so their decorated modules execute
# and self-register. Energy models are NOT here: they live under
# ``vdg.core.energy_model`` and register when the core package is imported.
# Order is load -> skill -> device so that, once concrete plugins exist, a
# device's ``is_available``/``measure_power`` can reference an energy model.
_DISCOVER_SUBPACKAGES = ("loads", "skills", "devices")


class Registrable:
    """Mixin marking a class as registry-managed.

    Subclasses (``DeviceProfile``, ``LoadModel``, ``Skill``, ``EnergyModel``)
    inherit this so the decorators can stamp registration metadata on them
    without each subclass repeating boilerplate.
    """

    __registry_kind__: str = ""
    __registry_name__: str = ""

    @classmethod
    def registry_name(cls) -> str:
        return cls.__registry_name__ or cls.__name__

    @classmethod
    def registry_kind(cls) -> str:
        return cls.__registry_kind__


class Registry:
    """A simple ``(kind, name) -> class`` store with discovery support."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, type]] = {}

    # -- core operations -------------------------------------------------
    def register(self, kind: str, name: str, cls: type) -> None:
        if kind not in _ALL_KINDS:
            raise ValueError(
                "Unknown registry kind: " + repr(kind)
                + ". Expected one of " + repr(_ALL_KINDS)
            )
        if not name:
            raise ValueError("Cannot register under an empty name for kind " + repr(kind))
        bucket = self._items.setdefault(kind, {})
        bucket[name] = cls

    def get(self, kind: str, name: str) -> type | None:
        return self._items.get(kind, {}).get(name)

    def all(self, kind: str) -> dict[str, type]:
        """Return a shallow copy of all registered classes for ``kind``."""
        if kind not in _ALL_KINDS:
            raise ValueError("Unknown registry kind: " + repr(kind))
        return dict(self._items.get(kind, {}))

    def kinds(self) -> list[str]:
        return list(self._items.keys())

    def names(self, kind: str) -> list[str]:
        return list(self._items.get(kind, {}).keys())

    def items(self) -> Iterator[tuple[str, str, type]]:
        for kind, bucket in self._items.items():
            for name, cls in bucket.items():
                yield kind, name, cls

    def __contains__(self, kind: str) -> bool:
        return kind in self._items and bool(self._items[kind])

    def __len__(self) -> int:
        return sum(len(b) for b in self._items.values())

    def clear(self) -> None:
        self._items.clear()

    # -- discovery -------------------------------------------------------
    def discover(self) -> int:
        """Auto-import subpackages so decorated plugins self-register.

        Returns the number of classes registered after discovery. Importing the
        top-level ``vdg`` package already imports these subpackages, so this is
        mainly useful for late-loaded plugin modules or explicit refresh from
        the CLI. Failures (missing optional plugin modules) are swallowed so a
        half-populated tree never crashes a simulation.
        """
        before = len(self)
        for sub in _DISCOVER_SUBPACKAGES:
            module_name = "vdg." + sub
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                # Subpackage has no plugins yet (foundation ships empty
                # devices/loads/skills) -- fine, skip silently.
                continue
            except Exception as exc:  # pragma: no cover - defensive
                # A broken plugin module must not poison the whole registry.
                import sys
                print("vdg: discovery skipped " + module_name + ": " + repr(exc), file=sys.stderr)
        return len(self) - before


# Global singleton. All decorators and ``discover`` operate on this instance.
REGISTRY = Registry()


def _make_decorator(kind: str) -> Callable[..., Any]:
    """Build a decorator that supports both bare and named usage."""

    def decorator(name: Any = None) -> Any:
        # Bare usage: ``@register_device`` -> Python calls decorator(cls).
        if isinstance(name, type):
            cls = name
            REGISTRY.register(kind, cls.__name__, cls)
            cls.__registry_kind__ = kind
            cls.__registry_name__ = cls.__name__
            return cls

        # Named usage: ``@register_device("rtx_5090")`` -> returns a real deco.
        explicit_name = name

        def deco(cls: type) -> type:
            reg_name = explicit_name or cls.__name__
            REGISTRY.register(kind, reg_name, cls)
            cls.__registry_kind__ = kind
            cls.__registry_name__ = reg_name
            return cls

        return deco

    decorator.__name__ = "register_" + kind
    decorator.__doc__ = "Register a class as a VDG " + kind + " plugin."
    return decorator


register_device = _make_decorator(KIND_DEVICE)
register_load = _make_decorator(KIND_LOAD)
register_skill = _make_decorator(KIND_SKILL)
register_energy_model = _make_decorator(KIND_ENERGY_MODEL)
