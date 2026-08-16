"""LightX2V emitter -- map governance decisions to a LightX2V launch command.

LightX2V is the distilled LTX-Video inference stack (4-step flow matching,
NVFP4 quantized, SageAttention, TeaCache -- the "1 video in 1 second on
consumer GPUs" pipeline). This emitter renders a ready-to-run shell command
from a VDG decision list::

    python -m lightx2v.infer --model_path <model> --width 854 --height 480 \
        --frames 81 --steps 4 --quant nvfp4 --attn sageattn --use_teacache

Skill -> flag mapping (grounded in the LightX2V CLI / the acceleration report):

* step_distill  -> --steps <config.steps> (4-step student weights),
* quantization  -> --quant <method> (nvfp4 / int8 / gguf_q4),
* sage_attention-> --attn sageattn,
* teacache      -> --use_teacache [--teacache_threshold <t>],
* vae_tiling    -> --vae_tiling (tiled decode),
* offload       -> --offload,
* compile_graph -> --compile (torch.compile of the DiT),
* repair skills -> --fp32_ops <comma list> (the ops the fp32 guards protect).
"""
from __future__ import annotations

import shlex
from typing import Any

__all__ = ["render_command"]

_QUANT_FLAG: dict[str, str] = {
    "gguf_q4": "gguf_q4",
    "nvfp4": "nvfp4",
    "int8": "int8",
}

# Repair skill -> guarded op token LightX2V understands for its fp32-ops flag.
_REPAIR_OP_TOKENS: dict[str, str] = {
    "gelu_fp32": "gelu",
    "adaln_fp32": "adaln",
    "rmsnorm_fp32": "rmsnorm",
    "softmax_fp32": "softmax",
    "vae_fp32": "vae",
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
    """Render a LightX2V launch command from governance decisions.

    model is the model path/id (e.g. "Lightricks/LTX-Video" or a local
    distilled checkpoint dir); resolution is (width, height). Returns a
    shell command string (paths shell-quoted). Every decision that maps to a
    flag is appended; unknown skills are skipped.
    """
    pairs = _as_pairs(decisions)
    configs: dict[str, dict[str, Any]] = {}
    for skill, cfg in pairs:
        configs[skill] = dict(cfg)

    width, height = resolution
    tokens: list[str] = [
        "python", "-m", "lightx2v.infer",
        "--model_path", model,
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frames),
    ]

    distill = configs.get("step_distill")
    if "step_distill" in configs and distill:
        tokens += ["--steps", str(int(distill.get("steps", 4)))]

    quant = configs.get("quantization", {})
    method = quant.get("method")
    if method in _QUANT_FLAG:
        tokens += ["--quant", _QUANT_FLAG[method]]

    if "sage_attention" in configs:
        tokens += ["--attn", "sageattn"]

    tea = configs.get("teacache", {})
    if "teacache" in configs:
        tokens += ["--use_teacache"]
        thr = tea.get("rel_l1_thresh", tea.get("threshold"))
        if thr is not None:
            tokens += ["--teacache_threshold", str(float(thr))]

    if "vae_tiling" in configs:
        tokens += ["--vae_tiling"]

    if "offload" in configs:
        tokens += ["--offload"]

    if "compile_graph" in configs:
        tokens += ["--compile"]

    repair_ops = [
        _REPAIR_OP_TOKENS[s] for s in sorted(configs)
        if s in _REPAIR_OP_TOKENS
    ]
    if repair_ops:
        tokens += ["--fp32_ops", ",".join(repair_ops)]

    return " ".join(shlex.quote(t) for t in tokens)
