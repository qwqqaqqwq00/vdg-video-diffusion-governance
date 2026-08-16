"""DiagnosticAgent: runs NumericalProbe and recommends repair skills.

The diagnostic agent probes a (device, precision) deployment point for
numerical divergence using the canonical NumericalProbe from
vdg.skills.repair.numerical_probe (which has a real torch path when the
target device is available, and a deterministic numpy-simulated fallback
otherwise, so it runs on any machine). If the probe reports divergence, the
agent recommends the matching granular repair skills (adaln_fp32,
gelu_fp32, rmsnorm_fp32, softmax_fp32, vae_fp32) -- resolved
from the registry and gated by each skill's own applicable predicate -- and
returns them as a list of GovernanceDecision.

Grounding: the probe's boundary inputs and thresholds come from the cross-device
robustness report section 8 (GELU |x|>=15 MPS kernel bug; AdaLN scale=-0.999
catastrophic cancellation; RMSNorm |x|>256 fp16 overflow; softmax large-seq
underflow). The repair-skill mapping is the probe's
DiagnosticReport.repair_skills_suggested. The LTX-2.3 three-cast fp32 fix
that the repair skills encode is grounded in MPS_BLACK_VIDEO_FIX.md.
"""
from __future__ import annotations

from typing import Any

from ..core.contracts import (
    DeviceCategory,
    DeviceProfile,
    GovernanceDecision,
    LoadModel,
)
from ..core.registry import REGISTRY
from ..core.simulator import AgentContext
from .base import GovernanceAgent
from ..skills.repair.numerical_probe import (
    NumericalProbe,
    DiagnosticReport,
    OpResult,
)

__all__ = ["DiagnosticAgent", "NumericalProbe", "DiagnosticReport", "OpResult"]


class DiagnosticAgent(GovernanceAgent):
    """Runs NumericalProbe and recommends repair skills on divergence.

    The agent maps the governance DeviceProfile to the probe's device-name
    string (apple_silicon -> "mps"; consumer_nv/datacenter -> "cuda"; edge NPU
    -> the spec name, which yields a simulated report), runs the probe, and
    turns each suggested repair skill into a GovernanceDecision -- but only
    for skills that are registered AND applicable to the device (so a
    datacenter card, whose repair skills report not-applicable, gets no repair
    recommendation even if the raw bf16 probe flags a theoretical hazard).
    """

    name = "diagnostic"
    role = "diagnose"

    def __init__(self, name: str | None = None, role: str | None = None) -> None:
        super().__init__(name, role)
        self.probe: NumericalProbe | None = None
        self.last_report: DiagnosticReport | None = None

    def run(self, context: AgentContext) -> dict[str, Any]:
        device = context.device
        load = context.load
        config = dict(context.config or {})
        precision = str(config.get("precision", "bf16"))
        probe_name = self._probe_device_name(device)

        self.probe = NumericalProbe()
        # Governance planning defaults to the SIMULATED probe (reproducible,
        # documented divergence thresholds from the robustness report) so a
        # governance run is deterministic across hosts and PyTorch versions.
        # The real torch path is a host-specific measurement used by the
        # standalone `vdg probe` diagnostic and opt-in via --real-probe. We call
        # _probe_simulated directly (rather than a probe_ops flag) so existing
        # probe_ops monkeypatches in tests/scripts stay byte-compatible.
        if bool(config.get("sim_probe", False)):
            report = self.probe._probe_simulated(probe_name, precision)
        else:
            report = self.probe.probe_ops(probe_name, precision)
        self.last_report = report

        decisions = self._recommend_repairs(device, load, report)
        notes_lines = [report.summary]
        if decisions:
            names = ", ".join(d.skill_name for d in decisions)
            notes_lines.append(
                "Repair skills recommended: " + names + "."
            )
        else:
            notes_lines.append(
                "No applicable repair skills required for this device."
            )

        return {
            "agent": self.name,
            "role": self.role,
            "decisions": decisions,
            "notes": "\n".join(notes_lines),
            "extra": {"probe_report": report},
        }

    # -- helpers -----------------------------------------------------------
    def _probe_device_name(self, device: DeviceProfile) -> str:
        """Map a DeviceProfile to the NumericalProbe device-name string."""
        cat = device.spec().category
        if cat == DeviceCategory.APPLE_SILICON:
            return "mps"
        if cat in (DeviceCategory.CONSUMER_NV, DeviceCategory.DATACENTER):
            return "cuda"
        # edge_npu: not mps/cuda -> probe returns a simulated report.
        return device.spec().name

    def _recommend_repairs(
        self, device: DeviceProfile, load: LoadModel, report: DiagnosticReport,
    ) -> list[GovernanceDecision]:
        if not report.has_failure:
            return []
        decisions: list[GovernanceDecision] = []
        suggested = list(report.repair_skills_suggested)
        failing = {r.op for r in report.results if r.status in ("divergence", "nan")}
        for skill_name in suggested:
            cls = REGISTRY.get("skill", skill_name)
            if cls is None:
                continue
            try:
                inst = cls()
            except Exception:
                continue
            if not inst.applicable(device, load):
                continue
            cfg = inst.default_config()
            impact = inst.predict(device, load, cfg)
            decisions.append(GovernanceDecision(
                skill_name=skill_name,
                config=cfg,
                predicted_impact=impact,
                rationale=(
                    "NumericalProbe reported divergence on " + ", ".join(sorted(failing))
                    + " for " + device.spec().name + " @ " + report.precision
                    + (" (simulated)" if report.simulated else " (measured)")
                    + ". Apply " + skill_name + " to force fp32 intermediate "
                    "computation for the sensitive op (PREC_GUARD_OPS template, "
                    "robustness report section 7)."
                ),
            ))
        return decisions
