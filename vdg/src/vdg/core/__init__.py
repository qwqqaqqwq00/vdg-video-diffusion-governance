"""Core: registry, contracts, roofline, energy, scenario, simulator, calibration."""
from __future__ import annotations

from .registry import (
    REGISTRY,
    Registry,
    Registrable,
    register_device,
    register_load,
    register_skill,
    register_energy_model,
)
from .contracts import (
    DeviceSpec,
    DeviceProfile,
    VideoDiTLoad,
    LoadModel,
    SkillImpact,
    Skill,
    GovernanceDecision,
    DeviceCategory,
)
from .roofline import (
    roofline,
    token_count,
    per_step_flops,
    predict_step_time,
    operational_intensity,
)
from .energy_model import EnergyModel, TDPEnergyModel, MeasuredEnergyModel
from .scenario import Scenario, ScenarioLibrary, BUILTIN_SCENARIOS
from .simulator import PerformanceEnergySimulator, SimulationResult, AgentContext
from .calibration import (
    CalibrationAnchor,
    CalibrationRow,
    CalibrationReport,
    CalibratedSimulator,
    ANCHORS,
    find_anchor,
)

__all__ = [
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
