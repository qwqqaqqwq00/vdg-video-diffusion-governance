"""Governance agents.

Subclasses of GovernanceAgent that reason over an AgentContext and emit
GovernanceDecision recommendations:

  * DiagnosticAgent  -- runs NumericalProbe and recommends repair skills.
  * AccelSelectorAgent -- enumerates skill combos + 5 recipe presets,
    simulates each, ranks by Pareto under the governance policy.
  * RepairAgent      -- applies repair decisions -> patched config +
    grounded patch instructions.
  * SimulatorAgent   -- runs the final authoritative simulation.
"""
from __future__ import annotations

from .base import GovernanceAgent, AgentResult
from .diagnostic import (
    DiagnosticAgent,
    NumericalProbe,
    DiagnosticReport,
    OpResult,
)
from .accel_selector import (
    AccelSelectorAgent,
    RecipePreset,
    RECIPES,
)
from .repair_agent import RepairAgent, PREC_GUARD_OPS
from .simulator_agent import SimulatorAgent

__all__ = [
    "GovernanceAgent",
    "AgentResult",
    "DiagnosticAgent",
    "NumericalProbe",
    "DiagnosticReport",
    "OpResult",
    "AccelSelectorAgent",
    "RecipePreset",
    "RECIPES",
    "RepairAgent",
    "PREC_GUARD_OPS",
    "SimulatorAgent",
]
