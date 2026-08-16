"""MLX emitter -- map governance decisions to an Apple Silicon MLX command.

MLX video generation (ml-explore/mlx-video, MLX-Community) runs LTX / Wan /
Hunyuan on Apple Silicon unified memory. This emitter renders a ready-to-run
command from a VDG decision list::

    mlx_video_generate --model <model> --width 854 --height 480 \
        --frames 81 --quantize 4 --steps 4 --sampler euler \
        --guidance 1.0 --use_teacache

Skill -> flag mapping:

* quantization    -> --quantize <bits> (gguf_q4/nvfp4 -> 4, int8 -> 8),
* step_distill    -> --steps <config.steps> + --sampler euler,
* teacache        -> --use_teacache,
* vae_tiling      -> --tiling,
* compile_graph   -> --compile,
* repair skills   -> no CLI flag (the MPS fp32 guards are applied in-process by
                     vdg.runtime.torch_runtime on the loaded model; noted here),
* sage_attention  -> no MLX flag (MLX uses its own fused attention kernels).
"""
from __future__ import annotations

import shlex
from typing import Any

__all__ = ["render_command"]

# quantization method -> --quantize bits on MLX.
_QUANT_BITS: dict[str, int] = {
    "gguf_q4": 4,
    "nvfp4": 4,
    "int8": 8,
}


def _as_pairs(decisions: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    pairs: list[tuple[str, dict[str, Any]]] = []
    for d in decisions:
        if isinstance(d, (tuple, list)) and len(d) >= 2:
            pairs.append((str(d[0]), dict(d[1] or {})))
        elif isinstance(d, dict):
            pairs.append((str(d.get("skill", d.get("skill_name", ""))),
                          dict(d.get("config") or {})))
        else:
            pairs.append((str(getattr(d, "skill_name", "")),
                          dict(getattr(d, "config", None) or {})))
    return pairs


def render_command(
    decisions: list[Any],
    model: str,
    resolution: tuple[int, int],
    frames: int,
) -> str:
    """Render an MLX video generation command from governance decisions.

    model is the MLX-converted model id/path (e.g.
    "mlx-community/LTX-Video-0.9.7-4bit"); resolution is (width, height).
    Returns a shell command string (paths shell-quoted).
    """
    pairs = _as_pairs(decisions)
    configs: dict[str, dict[str, Any]] = {}
    for skill, cfg in pairs:
        configs[skill] = dict(cfg)

    width, height = resolution
    tokens: list[str] = [
        "mlx_video_generate",
        "--model", model,
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frames),
        "--sampler", "euler",
    ]

    quant = configs.get("quantization", {})
    method = quant.get("method")
    if method in _QUANT_BITS:
        tokens += ["--quantize", str(_QUANT_BITS[method])]

    distill = configs.get("step_distill")
    if "step_distill" in configs and distill:
        tokens += ["--steps", str(int(distill.get("steps", 4)))]
        guidance = distill.get("guidance_scale")
        tokens += ["--guidance", str(float(guidance) if guidance is not None else 1.0)]
    else:
        tokens += ["--guidance", "4.0"]

    if "teacache" in configs:
        tokens += ["--use_teacache"]

    if "vae_tiling" in configs:
        tokens += ["--tiling"]

    if "compile_graph" in configs:
        tokens += ["--compile"]

    return " ".join(shlex.quote(t) for t in tokens)
