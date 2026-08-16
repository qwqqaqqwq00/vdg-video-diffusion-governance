"""Skill plugins: repair (numerical robustness) and accel (inference speedup).

Importing this package imports both subpackages so their decorated skills
self-register. Repair skills encode the battle-tested MPS three-cast fp32 fix
from ``MPS_BLACK_VIDEO_FIX.md`` (GELU / AdaLN-attn / AdaLN-MLP) plus the
op/block/layer precision-guard template from the cross-device robustness report.
Accel skills encode TeaCache, SageAttention 1/2/3, Sliding Tile Attention,
step distillation, GGUF/NVFP4 quantization, VAE tiling, torch.compile, MLX SDPA.
"""
from __future__ import annotations

from importlib import import_module


def _safe_import(name: str) -> None:
    try:
        import_module(name)
    except Exception:
        pass


_safe_import("vdg.skills.repair")
_safe_import("vdg.skills.accel")
