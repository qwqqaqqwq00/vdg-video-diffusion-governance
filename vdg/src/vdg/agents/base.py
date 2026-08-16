"""Governance agent base classes.

A governance agent reasons over an ``AgentContext`` (device, load, scenario,
accumulated simulation results) and returns a structured decision dict --
typically a list of ``GovernanceDecision`` recommendations (which skills to
apply, with what config, and why) that a planner then simulates to build a
Pareto frontier.

Agents are pluggable: subclass ``GovernanceAgent``, decorate with
``@register_skill``-style registration in phase 2 (a dedicated
``@register_agent`` kind can be added when needed), and implement ``run``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.contracts import GovernanceDecision
from ..core.simulator import AgentContext

__all__ = ["GovernanceAgent", "AgentResult"]


@dataclass
class AgentResult:
    """Structured return value of ``GovernanceAgent.run``."""

    agent: str
    role: str
    decisions: list[GovernanceDecision] = field(default_factory=list)
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class GovernanceAgent:
    """Base class for all governance agents.

    Subclasses set ``name``/``role`` and implement ``run(context)``. The context
    carries the device, load, scenario, config, the skills already applied, and
    the simulation results gathered so far, so an agent can plan incrementally.
    """

    name: str = "base"
    role: str = "governance"

    def __init__(self, name: str | None = None, role: str | None = None) -> None:
        if name is not None:
            self.name = name
        if role is not None:
            self.role = role

    def run(self, context: AgentContext) -> dict[str, Any]:
        """Reason over the context and return a structured decision dict.

        The default implementation returns an empty ``AgentResult``-shaped dict.
        Concrete agents (repair-governor, accel-governor, pareto-planner) return
        ``{"agent", "role", "decisions", "notes"}`` where ``decisions`` is a list
        of ``GovernanceDecision``.
        """
        return {
            "agent": self.name,
            "role": self.role,
            "decisions": [],
            "notes": "Base agent does nothing; subclass to implement governance.",
        }
