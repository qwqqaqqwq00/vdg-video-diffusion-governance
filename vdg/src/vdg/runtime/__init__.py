"""VDG runtime bindings -- turn governance decisions into executable actions.

This package resolves known limitation #2 of the foundation ("Skill.apply()
was a stub envelope"): it provides REAL runtime bindings that consume the
governance decision list and apply it to actual runtimes:

* envelope          -- RuntimeEnvelope: validated handoff (kind,
                          target_runtime, per-runtime required-config checks).
* torch_runtime     -- real in-process repair patches on torch nn.Module
                          (locates sensitive modules in any DiT, applies the
                          vdg.skills.repair patch functions, unpatch support).
* diffusers_runtime -- LTX-Video diffusers pipeline binding (build_pipeline
                          is None-safe, apply_repairs via TorchRuntime,
                          apply_accel maps each accel skill to diffusers APIs).
* comfyui_emitter   -- renders a valid ComfyUI /prompt API workflow JSON
                          from the decisions + markdown instructions + an
                          executable torch patch script.
* lightx2v_emitter  -- renders a LightX2V launch command.
* mlx_emitter       -- renders an Apple Silicon MLX generation command.

All torch/diffusers imports are lazy: importing this package never requires
either, so pure-sim environments stay importable (the emitters and envelope
are pure Python; the torch/diffusers runtimes only import their backend at
call time).
"""
from __future__ import annotations

from .envelope import (
    KNOWN_SKILLS,
    VALID_KINDS,
    VALID_RUNTIMES,
    RuntimeEnvelope,
)
from .torch_runtime import REPAIR_PATCH_SITES, TorchRuntime
from .diffusers_runtime import DiffusersRuntime
from . import comfyui_emitter, lightx2v_emitter, mlx_emitter
from .comfyui_emitter import build_workflow, render_markdown, render_patch_script
from .lightx2v_emitter import render_command as render_lightx2v_command
from .mlx_emitter import render_command as render_mlx_command

__all__ = [
    # envelope
    "RuntimeEnvelope",
    "VALID_RUNTIMES",
    "VALID_KINDS",
    "KNOWN_SKILLS",
    # torch runtime
    "TorchRuntime",
    "REPAIR_PATCH_SITES",
    # diffusers runtime
    "DiffusersRuntime",
    # comfyui emitter
    "comfyui_emitter",
    "build_workflow",
    "render_markdown",
    "render_patch_script",
    # lightx2v / mlx emitters
    "lightx2v_emitter",
    "render_lightx2v_command",
    "mlx_emitter",
    "render_mlx_command",
]
