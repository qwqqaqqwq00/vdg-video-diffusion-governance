"""Shared helpers for acceleration skills.

These keep each skill file focused on its impact model and runtime config by
factoring out two reusable pieces:

* clamp -- bound a float to an interval (used by threshold/ratio mappings).
* runtime_envelope -- the structured dict that Skill.apply returns so a
  real runtime (ComfyUI / diffusers / LightX2V / MLX) can consume the exact
  config. The envelope always carries the skill name, target runtime, the
  resolved config kwargs, whether an in-process patch was applied, and a
  provenance note. Skills stub the actual kernel (the VDG foundation ships no
  runtime), so applied is normally False and the runtime applies the config.
"""
from __future__ import annotations

from typing import Any

__all__ = ["clamp", "runtime_envelope"]


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to the closed interval [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def runtime_envelope(
    skill: str,
    runtime: str,
    config: dict[str, Any],
    applied: bool,
    notes: str,
) -> dict[str, Any]:
    """Build the runtime config dict consumed by a video-DiT runtime.

    Parameters
    ----------
    skill:
        Registry name of the skill (e.g. "teacache").
    runtime:
        Target runtime that consumes the config: one of "comfyui",
        "diffusers", "lightx2v", "mlx", "tensorrt".
    config:
        Resolved runtime kwargs (already merged with defaults + overrides).
    applied:
        Whether an in-process patch was actually applied to the model/pipeline.
        False for the stub path (no kernel patched); the runtime applies config.
    notes:
        Provenance / caveats documenting how the runtime consumes the config.
    """
    return {
        "skill": skill,
        "runtime": runtime,
        "config": dict(config),
        "applied": bool(applied),
        "notes": notes,
    }
