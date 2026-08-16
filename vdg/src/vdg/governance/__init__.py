"""Governance layer: policy, rule engine, and the multi-agent pipeline.

  * Policy / Violation -- hard constraints (energy, latency, quality
    floor, memory) enforced on simulation results.
  * RuleEngine / RuleOutcome -- canonical guard rules (Apple Silicon
    disables SageAttention; energy overrun prefers distillation; strict quality
    floor limits quantization; fp8-on-int8 adds boundary-block bf16).
  * GovernancePipeline / GovernanceReport -- orchestrates
    resolve -> diagnose -> rules -> select accel -> repair -> simulate -> report.
"""
from __future__ import annotations

from .policy import Policy, Violation
from .rules import RuleEngine, RuleOutcome
from .pipeline import GovernancePipeline, GovernanceReport

__all__ = [
    "Policy",
    "Violation",
    "RuleEngine",
    "RuleOutcome",
    "GovernancePipeline",
    "GovernanceReport",
]
