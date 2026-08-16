"""VDG (Video Diffusion Governance).

Importing this package sets up the pluggable registry and core contracts, and
imports the subpackages so that phase-2 plugins (devices, loads, skills)
self-register on import. The primary modeled load is LTX-2.3 (Lightricks video
DiT); Wan/Hunyuan/CogVideoX are secondary reference loads.
"""
from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

# Core machinery + contracts (importing core also registers the built-in energy
# models via the @register_energy_model decorators in core.energy_model).
from .core.registry import (
    REGISTRY,
    Registry,
    Registrable,
    register_device,
    register_load,
    register_skill,
    register_energy_model,
)
from .core.contracts import (
    DeviceSpec,
    DeviceProfile,
    VideoDiTLoad,
    LoadModel,
    SkillImpact,
    Skill,
    GovernanceDecision,
    DeviceCategory,
)
from .core.roofline import (
    roofline,
    token_count,
    per_step_flops,
    predict_step_time,
    operational_intensity,
)
from .core.energy_model import EnergyModel, TDPEnergyModel, MeasuredEnergyModel
from .core.scenario import Scenario, ScenarioLibrary, BUILTIN_SCENARIOS
from .core.simulator import (
    PerformanceEnergySimulator,
    SimulationResult,
    AgentContext,
)
from .core.calibration import (
    CalibrationAnchor,
    CalibrationRow,
    CalibrationReport,
    CalibratedSimulator,
    ANCHORS,
    find_anchor,
)

__all__ = [
    "__version__",
    "REGISTRY",
    "Registry",
    "Registrable",
    "register_device",
    "register_load",
    "register_skill",
    "register_energy_model",
    "DeviceSpec",
    "DeviceProfile",
    "VideoDiTLoad",
    "LoadModel",
    "SkillImpact",
    "Skill",
    "GovernanceDecision",
    "DeviceCategory",
    "roofline",
    "token_count",
    "per_step_flops",
    "predict_step_time",
    "operational_intensity",
    "EnergyModel",
    "TDPEnergyModel",
    "MeasuredEnergyModel",
    "Scenario",
    "ScenarioLibrary",
    "BUILTIN_SCENARIOS",
    "PerformanceEnergySimulator",
    "SimulationResult",
    "AgentContext",
    "CalibrationAnchor",
    "CalibrationRow",
    "CalibrationReport",
    "CalibratedSimulator",
    "ANCHORS",
    "find_anchor",
]


def _import_subpackages() -> None:
    """Import plugin subpackages so decorated plugins auto-register.

    Failures are swallowed so a partially-installed tree never breaks
    ``import vdg``. Phase-2 plugins placed in these packages register on import.
    """
    for _sub in ("devices", "loads", "skills", "agents", "governance", "runtime"):
        try:
            import_module("vdg." + _sub)
        except Exception:
            pass


_import_subpackages()
