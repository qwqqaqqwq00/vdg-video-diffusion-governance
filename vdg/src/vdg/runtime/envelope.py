"""RuntimeEnvelope -- the structured handoff between governance and runtimes.

The accel/repair skills already emit a lightweight envelope dict via
vdg.skills.accel._common.runtime_envelope (keys: skill / runtime / config /
applied / notes). This module upgrades that concept into a first-class
dataclass that a real runtime can validate and act on:

* kind distinguishes whether the envelope is a pure config payload for the
  runtime to apply ("config"), an already-applied in-process patch ("patch"),
  or a generated workflow artifact ("workflow").
* target_runtime names the runtime that consumes the config -- one of
  comfyui / diffusers / lightx2v / mlx / torch / tensorrt.
* validate() checks that the required config keys exist for the
  (target_runtime, skill) pair, so a runtime can fail fast with a readable
  message instead of silently applying a broken config.
* from_dict() parses the existing envelope dicts the skills emit, so all
  current Skill.apply() outputs upgrade to RuntimeEnvelope cleanly.

The envelope is pure Python (no torch/diffusers) -- it must stay importable in
pure-sim environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "VALID_RUNTIMES",
    "VALID_KINDS",
    "KNOWN_SKILLS",
    "RuntimeEnvelope",
]

# Canonical runtime names a config can target.
VALID_RUNTIMES: frozenset[str] = frozenset(
    {"comfyui", "diffusers", "lightx2v", "mlx", "torch", "tensorrt"}
)

# Envelope kinds.
VALID_KINDS: frozenset[str] = frozenset({"config", "patch", "workflow"})

# Skills the runtime layer knows how to translate to a concrete runtime
# action. Concept skills (sliding_tile_attention / linear_attention) are not
# listed: they have no runtime binding and validate() flags them as advisory.
KNOWN_SKILLS: frozenset[str] = frozenset({
    # repair (torch patchable)
    "gelu_fp32", "adaln_fp32", "rmsnorm_fp32", "softmax_fp32", "vae_fp32",
    # accel
    "teacache", "vae_tiling", "quantization", "sage_attention",
    "step_distill", "offload", "compile_graph",
})

# Required config keys per (runtime, skill). A skill with no entry for a given
# runtime has no mandatory keys (validate passes with a note). These mirror the
# exact config keys the accel skills emit in their runtime_envelope config
# dicts (see vdg/skills/accel/*.py apply()).
_REQUIRED_CONFIG_KEYS: dict[tuple[str, str], tuple[str, ...]] = {
    ("comfyui", "teacache"): ("rel_l1_thresh",),
    ("comfyui", "vae_tiling"): ("tile_size", "overlap", "temporal_size", "temporal_overlap"),
    ("comfyui", "quantization"): ("method",),
    ("comfyui", "sage_attention"): ("version",),
    ("diffusers", "teacache"): ("rel_l1_thresh",),
    ("diffusers", "vae_tiling"): ("tile_size", "overlap", "temporal_size", "temporal_overlap"),
    ("diffusers", "step_distill"): ("steps",),
    ("diffusers", "offload"): ("enable_offload",),
    ("diffusers", "compile_graph"): ("backend",),
    ("lightx2v", "step_distill"): ("steps",),
    ("lightx2v", "quantization"): ("method",),
    ("mlx", "step_distill"): ("steps",),
    ("mlx", "quantization"): ("method",),
    ("torch", "gelu_fp32"): (),
    ("torch", "adaln_fp32"): (),
    ("torch", "rmsnorm_fp32"): (),
    ("torch", "softmax_fp32"): (),
    ("torch", "vae_fp32"): (),
}


@dataclass
class RuntimeEnvelope:
    """Validated handoff describing how a runtime should apply one skill.

    runtime preserves the legacy skill-emitted runtime string (e.g.
    "comfyui"); target_runtime is the normalized field the runtime layer
    actually switches on. kind records whether this is a config payload, an
    already-applied in-process patch, or a generated workflow artifact.
    """

    skill: str
    runtime: str
    config: dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    notes: str = ""
    kind: str = "config"
    target_runtime: str = "comfyui"

    # -- validation -------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of problems, empty when the envelope is well-formed.

        Checks, in order:

        * kind is one of VALID_KINDS,
        * target_runtime is one of VALID_RUNTIMES,
        * config is a dict,
        * the required config keys for (target_runtime, skill) are present,
        * skill is a known runtime-bound skill (unknown skills are
          reported as a note-level problem so concept skills surface clearly).
        """
        problems: list[str] = []
        if self.kind not in VALID_KINDS:
            problems.append(
                "kind " + repr(self.kind) + " not in " + sorted_str(VALID_KINDS)
            )
        if self.target_runtime not in VALID_RUNTIMES:
            problems.append(
                "target_runtime " + repr(self.target_runtime)
                + " not in " + sorted_str(VALID_RUNTIMES)
            )
        if not isinstance(self.config, dict):
            problems.append("config must be a dict, got " + type(self.config).__name__)
            return problems
        if self.skill not in KNOWN_SKILLS:
            problems.append(
                "skill " + repr(self.skill)
                + " has no runtime binding (concept skill or unknown); "
                + "envelope is advisory only"
            )
        required = _REQUIRED_CONFIG_KEYS.get((self.target_runtime, self.skill), ())
        for key in required:
            if key not in self.config:
                problems.append(
                    "missing required config key " + repr(key)
                    + " for (" + self.target_runtime + ", " + self.skill + ")"
                )
        return problems

    def validate_or_raise(self) -> "RuntimeEnvelope":
        """Like validate() but raises ValueError on the first problem."""
        problems = self.validate()
        if problems:
            raise ValueError(
                "invalid RuntimeEnvelope for skill " + repr(self.skill)
                + ": " + "; ".join(problems)
            )
        return self

    # -- parsing ----------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeEnvelope":
        """Build a RuntimeEnvelope from a skill-emitted envelope dict.

        Accepts both the legacy stub shape (skill / runtime / config / applied /
        notes -- what Skill.apply() returns today) and the extended shape
        (plus kind / target_runtime). Missing kind is derived from applied
        (True -> "patch", False -> "config"); missing target_runtime falls back
        to the legacy runtime string.
        """
        skill = str(data.get("skill", ""))
        runtime = str(data.get("runtime", "") or "comfyui")
        config = data.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        applied = bool(data.get("applied", False))
        notes = str(data.get("notes", "") or "")
        kind = str(data.get("kind", "") or ("patch" if applied else "config"))
        target_runtime = str(data.get("target_runtime", "") or runtime)
        return cls(
            skill=skill,
            runtime=runtime,
            config=dict(config),
            applied=applied,
            notes=notes,
            kind=kind,
            target_runtime=target_runtime,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to a plain dict (superset of the skill-emitted shape)."""
        return {
            "skill": self.skill,
            "runtime": self.runtime,
            "config": dict(self.config),
            "applied": self.applied,
            "notes": self.notes,
            "kind": self.kind,
            "target_runtime": self.target_runtime,
        }


def sorted_str(values: frozenset[str]) -> str:
    return "{" + ", ".join(sorted(values)) + "}"
