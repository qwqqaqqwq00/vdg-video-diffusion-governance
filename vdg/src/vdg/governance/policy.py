"""Governance policy: hard constraints enforced on simulation results.

A Policy is the deployment-level envelope of acceptable operating points:
an energy budget (joules), a latency SLO (seconds), a quality floor (VBench-
proxy points) and a peak-memory ceiling (GB). It is distinct from a
Scenario (which fixes the *workload*) -- a policy fixes the *acceptance
bar* that any workload+skill combination must clear on a given device.

The pipeline builds a Policy from the scenario SLOs plus device memory and
any CLI overrides (--energy-budget), then calls enforce on every
candidate simulation result to filter out infeasible operating points and to
emit structured violations in the final report.

Grounding: the energy/latency/quality axes come directly from the synthesis
report's performance-energy trade-off section (step distillation is the
energy-first lever; VAE is the quality hard floor; 24/32GB consumer cards set
the memory ceiling). See CONTRACTS.md and the scenario library.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.simulator import SimulationResult

__all__ = ["Violation", "Policy"]


@dataclass(frozen=True)
class Violation:
    """A single policy constraint breach.

    constraint is the symbolic name of the limit that was breached,
    actual / limit carry the offending numbers, and message is a
    human-readable line suitable for CLI/report output.
    """

    constraint: str
    actual: float
    limit: float
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class Policy:
    """Acceptance envelope for a governance run.

    All fields default to "no limit" sentinels so a partially-specified policy
    (e.g. only an energy budget) still enforces cleanly. The pipeline fills
    sensible defaults from the scenario and device when constructing a policy.
    """

    energy_budget_j: float = float("inf")
    latency_slo_s: float = float("inf")
    quality_floor: float = 0.0
    max_memory_gb: float = float("inf")
    notes: str = ""

    @classmethod
    def from_scenario(
        cls,
        scenario,
        device=None,
        energy_budget_j: float | None = None,
        latency_slo_s: float | None = None,
        quality_floor: float | None = None,
        max_memory_gb: float | None = None,
    ) -> "Policy":
        """Build a policy from a scenario, optionally overridden by CLI args.

        Defaults: energy/latency come from the scenario SLOs; quality_floor
        defaults to the scenario quality target (a result at or above target is
        acceptable); max_memory defaults to the device memory if a device is
        given, else unbounded.
        """
        energy = scenario.energy_budget_j if energy_budget_j is None else float(energy_budget_j)
        slo = scenario.latency_slo_s if latency_slo_s is None else float(latency_slo_s)
        floor = scenario.quality_target if quality_floor is None else float(quality_floor)
        if max_memory_gb is None:
            max_memory_gb = device.spec().memory_gb if device is not None else float("inf")
        else:
            max_memory_gb = float(max_memory_gb)
        return cls(
            energy_budget_j=energy,
            latency_slo_s=slo,
            quality_floor=floor,
            max_memory_gb=max_memory_gb,
            notes="Built from scenario " + repr(scenario.name) + ".",
        )

    def enforce(self, result: SimulationResult) -> list[Violation]:
        """Return the list of policy constraints this result breaches.

        An empty list means the result is policy-feasible. Each violation is
        a structured Violation with a ready-to-print message.
        """
        violations: list[Violation] = []
        if result.latency_s > self.latency_slo_s:
            violations.append(Violation(
                constraint="latency_slo_s",
                actual=result.latency_s,
                limit=self.latency_slo_s,
                message=(
                    "Latency " + format(result.latency_s, ".2f") + "s exceeds SLO "
                    + format(self.latency_slo_s, ".2f") + "s."
                ),
            ))
        if result.energy_j > self.energy_budget_j:
            violations.append(Violation(
                constraint="energy_budget_j",
                actual=result.energy_j,
                limit=self.energy_budget_j,
                message=(
                    "Energy " + format(result.energy_j, ".0f") + "J exceeds budget "
                    + format(self.energy_budget_j, ".0f") + "J."
                ),
            ))
        if result.quality_score < self.quality_floor:
            violations.append(Violation(
                constraint="quality_floor",
                actual=result.quality_score,
                limit=self.quality_floor,
                message=(
                    "Quality " + format(result.quality_score, ".2f") + " below floor "
                    + format(self.quality_floor, ".2f") + "."
                ),
            ))
        if result.peak_memory_gb > self.max_memory_gb:
            violations.append(Violation(
                constraint="max_memory_gb",
                actual=result.peak_memory_gb,
                limit=self.max_memory_gb,
                message=(
                    "Peak memory " + format(result.peak_memory_gb, ".2f")
                    + "GB exceeds ceiling " + format(self.max_memory_gb, ".2f") + "GB."
                ),
            ))
        return violations

    def is_feasible(self, result: SimulationResult) -> bool:
        """True iff the result clears every policy constraint."""
        return not self.enforce(result)
