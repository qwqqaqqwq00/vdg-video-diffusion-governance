"""ComfyUI emitter -- turn governance decisions into an executable ComfyUI workflow.

This is the bridge that makes a VDG governance report DIRECTLY executable in
ComfyUI: build_workflow renders the decisions into a valid ComfyUI
/prompt API-format workflow (nodes keyed "1", "2", ... with class_type +
inputs), render_markdown produces a human-readable instruction block, and
render_patch_script emits a ready-to-run python snippet that applies the
repair decisions in-process via vdg.runtime.torch_runtime (so the repair is
executable, not just described).

Skill -> node mapping:

* teacache        -> class_type "TeaCache" with rel_l1_thresh / start_step /
                     end_step (Kijai ComfyUI-TeaCache node),
* vae_tiling      -> VAEDecodeTiled with tile_size / overlap / temporal params,
* quantization    -> "UnetLoaderGGUF" when method is gguf_q4 (else note),
* step_distill    -> KSampler steps = config.steps, cfg 1.0 (distilled),
* sage_attention  -> documented as a launch flag (--use-sage-attention), not a
                     node; surfaced in the returned notes / markdown.

The returned workflow is a dict {"nodes": <api payload>, "meta": {...},
"notes": [...]} so metadata travels with the payload without polluting the
API-format nodes dict itself.
"""
from __future__ import annotations

import json
from typing import Any

__all__ = [
    "build_workflow",
    "render_markdown",
    "render_patch_script",
]

# Default sampler / scheduler names for video DiTs in ComfyUI.
_SAMPLER = "euler"
_SCHEDULER = "simple"
_DISTILLED_CFG = 1.0
_STANDARD_CFG = 4.0

# Repair skills are applied as an in-process torch patch, not a ComfyUI node.
_REPAIR_SKILLS = frozenset({
    "gelu_fp32", "adaln_fp32", "rmsnorm_fp32", "softmax_fp32", "vae_fp32",
})


# --------------------------------------------------------------------------
# Decision normalization
# --------------------------------------------------------------------------
def _as_pairs(decisions: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    """Normalize GovernanceDecision / dict / tuple items to (skill, config)."""
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


def _config_map(pairs: list[tuple[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Collapse pairs to {skill: config}, last config wins."""
    merged: dict[str, dict[str, Any]] = {}
    for skill, cfg in pairs:
        merged[skill] = dict(cfg)
    return merged


# --------------------------------------------------------------------------
# Workflow construction
# --------------------------------------------------------------------------
def build_workflow(
    decisions: list[Any],
    load_name: str,
    scenario: Any,
    comfyui_model_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render governance decisions into a ComfyUI /prompt API-format workflow.

    Parameters
    ----------
    decisions:
        Governance decisions (GovernanceDecision objects, (skill, config)
        tuples, or {"skill", "config"} dicts).
    load_name:
        Model name, used to derive default model file names when
        comfyui_model_paths does not provide them.
    scenario:
        A vdg.core.scenario.Scenario (resolution / frames / fps / steps).
    comfyui_model_paths:
        Optional mapping of ComfyUI model files, e.g.
        {"checkpoint": "models/checkpoints/ltx.safetensors",
        "unet": "models/diffusion_models/ltx.safetensors",
        "vae": "models/vae/ltx_vae.safetensors",
        "clip": "models/clip/ltx_clip.safetensors"}. When "unet" is present
        the model is loaded with UNETLoader (plus CLIPLoader + VAELoader);
        when a gguf_q4 quantization decision exists it is loaded with
        UnetLoaderGGUF; otherwise CheckpointLoaderSimple is used.

    Returns {"nodes": ..., "meta": ..., "notes": [...]}. nodes is the
    valid /prompt payload (keys "1".."N", each with class_type + inputs).
    """
    pairs = _as_pairs(decisions)
    configs = _config_map(pairs)
    paths = dict(comfyui_model_paths or {})
    width, height = scenario.resolution
    frames = int(scenario.frames)
    fps = int(scenario.fps)

    nodes: dict[str, dict[str, Any]] = {}
    notes: list[str] = []

    # --- model loader ----------------------------------------------------
    quant_method = configs.get("quantization", {}).get("method")
    use_gguf = quant_method == "gguf_q4"
    use_unet = bool(paths.get("unet"))
    if use_gguf:
        nodes["1"] = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": paths.get("unet", "models/diffusion_models/" + load_name + ".gguf"),
                "file_type": configs.get("quantization", {}).get("file_type", "Q4_K_M"),
            },
        }
        notes.append("GGUF quantization: model loaded via UnetLoaderGGUF "
                     "(ComfyUI-GGUF pack required).")
        model_node = "1"
        clip_node: str | None = None
        vae_node: str | None = None
    elif use_unet:
        nodes["1"] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": paths["unet"],
                "weight_dtype": "default",
            },
        }
        notes.append("Model loaded via UNETLoader (diffusion model only).")
        model_node = "1"
        clip_node = None
        vae_node = None
    else:
        nodes["1"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": paths.get(
                    "checkpoint", "models/checkpoints/" + load_name + ".safetensors"
                ),
            },
        }
        model_node = "1"
        clip_node = "1"      # output 1 (CLIP)
        vae_node = "1"       # output 2 (VAE)

    next_id = 2

    # --- CLIP / VAE loaders for UNET paths -------------------------------
    if clip_node is None:
        clip_type = configs.get("clip_type", "ltxv")
        nodes[str(next_id)] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": paths.get("clip", "models/clip/clip_l.safetensors"),
                "type": clip_type,
            },
        }
        clip_node = str(next_id)
        next_id += 1
    if vae_node is None:
        nodes[str(next_id)] = {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": paths.get("vae", "models/vae/" + load_name + "_vae.safetensors"),
            },
        }
        vae_node = str(next_id)
        next_id += 1

    # --- conditioning ----------------------------------------------------
    pos_text = configs.get("positive_prompt", "cinematic high-quality video")
    neg_text = configs.get("negative_prompt", "low quality, blurry, distorted")
    nodes[str(next_id)] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": [clip_node, 0], "text": pos_text},
    }
    pos_node = str(next_id)
    next_id += 1
    nodes[str(next_id)] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": [clip_node, 0], "text": neg_text},
    }
    neg_node = str(next_id)
    next_id += 1

    # --- latent ----------------------------------------------------------
    nodes[str(next_id)] = {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": width,
            "height": height,
            "length": frames,
            "batch_size": 1,
        },
    }
    notes.append("EmptyLTXVLatentVideo is a core ComfyUI node (LTX video "
                 "latent); on ComfyUI versions without it, replace with "
                 "EmptyLatentImage (width/height) + batch_size " + str(frames) + ".")
    latent_node = str(next_id)
    next_id += 1

    # --- TeaCache --------------------------------------------------------
    feed_model = model_node
    if "teacache" in configs:
        tc = configs["teacache"]
        nodes[str(next_id)] = {
            "class_type": "TeaCache",
            "inputs": {
                "model": [model_node, 0],
                "rel_l1_thresh": float(tc.get("rel_l1_thresh", tc.get("threshold", 0.1))),
                "start_step": int(tc.get("start_step", 0)),
                "end_step": int(tc.get("end_step", 0) or 0),
            },
        }
        notes.append("TeaCache node: class_type 'TeaCache' (Kijai ComfyUI-TeaCache "
                     "pack), rel_l1_thresh " + str(float(tc.get("rel_l1_thresh", tc.get("threshold", 0.1))))
                     + "; rel_l1_thresh 0.25-0.30 recommended with coefficients.")
        feed_model = str(next_id)
        next_id += 1

    # --- KSampler --------------------------------------------------------
    distill_cfg = configs.get("step_distill", {})
    steps = int(distill_cfg.get("steps", scenario.steps))
    distilled = bool(distill_cfg)
    cfg = float(distill_cfg.get("guidance_scale", _DISTILLED_CFG if distilled else _STANDARD_CFG))
    if distill_cfg:
        notes.append("Step distillation: KSampler steps=" + str(steps)
                     + ", cfg=1.0 (distilled checkpoint, e.g. LightX2V 4-step).")
    nodes[str(next_id)] = {
        "class_type": "KSampler",
        "inputs": {
            "model": [feed_model, 0],
            "positive": [pos_node, 0],
            "negative": [neg_node, 0],
            "latent_image": [latent_node, 0],
            "seed": 42,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": _SAMPLER,
            "scheduler": _SCHEDULER,
            "denoise": 1.0,
        },
    }
    sampler_node = str(next_id)
    next_id += 1

    # --- VAE decode ------------------------------------------------------
    tiling = configs.get("vae_tiling", {})
    if tiling:
        inputs: dict[str, Any] = {
            "samples": [sampler_node, 0],
            "vae": [vae_node, 0],
            "tile_size": int(tiling.get("tile_size", 512)),
            "overlap": int(tiling.get("overlap", 64)),
        }
        if tiling.get("temporal_size") is not None:
            inputs["temporal_size"] = int(tiling["temporal_size"])
        if tiling.get("temporal_overlap") is not None:
            inputs["temporal_overlap"] = int(tiling["temporal_overlap"])
        nodes[str(next_id)] = {"class_type": "VAEDecodeTiled", "inputs": inputs}
        notes.append("VAE tiling: VAEDecodeTiled with tile_size/overlap"
                     + (" plus temporal_size/temporal_overlap (ComfyUI-VideoHelperSuite)"
                        if "temporal_size" in inputs else "")
                     + "; 32GB -> ~8GB peak on long clips.")
    else:
        nodes[str(next_id)] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": [sampler_node, 0], "vae": [vae_node, 0]},
        }
    decode_node = str(next_id)
    next_id += 1

    # --- video output ----------------------------------------------------
    nodes[str(next_id)] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "video": [decode_node, 0],
            "frame_rate": fps,
            "loop_count": 0,
            "format": "video/h264-mp4",
            "filename_prefix": "vdg_" + load_name.replace("/", "_"),
        },
    }
    notes.append("VideoOutput: VHS_VideoCombine (ComfyUI-VideoHelperSuite) writes "
                 + str(fps) + " fps h264; pause_after_gen off.")

    # --- non-node skills surfaced as notes -------------------------------
    sage = configs.get("sage_attention", {})
    if sage:
        notes.append("SageAttention is a launch flag, not a node: start ComfyUI with "
                     "--use-sage-attention (version " + str(sage.get("version", "v2"))
                     + "); requires sageattention + Triton + CUDA.")
    offload = configs.get("offload", {})
    if offload:
        notes.append("Offload is a launch flag, not a node: start ComfyUI with "
                     "--async-offload " + str(offload.get("async_offload_streams", 2))
                     + " (block-swap via Kijai packs for >VRAM models).")
    compile_cfg = configs.get("compile_graph", {})
    if compile_cfg:
        notes.append("torch.compile: enable via ComfyUI Settings -> 'Torch compile' "
                     "mode (backend " + str(compile_cfg.get("backend", "torch_compile")) + ").")

    repair_names = sorted(s for s in configs if s in _REPAIR_SKILLS)
    if repair_names:
        notes.append("Repair skills (" + ", ".join(repair_names)
                     + ") are in-process torch patches, not nodes -- run "
                     "render_patch_script() output against the loaded model.")

    return {
        "nodes": nodes,
        "meta": {
            "title": "VDG governance workflow: " + load_name + " / " + scenario.name,
            "load": load_name,
            "scenario": scenario.name,
            "resolution": list(scenario.resolution),
            "frames": frames,
            "fps": fps,
        },
        "notes": notes,
    }


# --------------------------------------------------------------------------
# Human-readable rendering
# --------------------------------------------------------------------------
def render_markdown(workflow_json: dict[str, Any]) -> str:
    """Render a build_workflow() dict as a human-readable instruction block.

    Lists every node (class_type, key inputs, and what consumes its outputs),
    the launch flags the workflow depends on, and where to paste the payload
    (ComfyUI /prompt API). Also prints the raw JSON for direct paste.
    """
    nodes = workflow_json.get("nodes", {})
    meta = workflow_json.get("meta", {})
    notes = workflow_json.get("notes", [])
    lines: list[str] = []
    lines.append("# VDG ComfyUI workflow: " + str(meta.get("title", "")))
    lines.append("")
    lines.append("This workflow renders the VDG governance decisions for load "
                 + str(meta.get("load", "?")) + " / scenario "
                 + str(meta.get("scenario", "?")) + " ("
                 + "x".join(str(v) for v in meta.get("resolution", []))
                 + " @ " + str(meta.get("frames", "?")) + "f/"
                 + str(meta.get("fps", "?")) + "fps).")
    lines.append("")
    lines.append("## Where to run it")
    lines.append("1. Start ComfyUI (custom nodes: ComfyUI-VideoHelperSuite, "
                 "ComfyUI-TeaCache, and ComfyUI-GGUF if used).")
    lines.append("2. Dev Mode (Settings -> Enable Dev Mode Options).")
    lines.append("3. Either POST the JSON below to http://127.0.0.1:8188/prompt "
                 "with body {\"prompt\": <json>}, or paste it via 'Load (API "
                 "Format)' in the workflow editor.")
    lines.append("")
    lines.append("## Nodes")
    for nid in sorted(nodes, key=int):
        node = nodes[nid]
        ct = node.get("class_type", "?")
        inputs = node.get("inputs", {})
        consumed_by = [
            other for other in sorted(nodes, key=int)
            if other != nid and _references(nodes[other], nid)
        ]
        summary = "  [" + nid + "] " + ct
        lines.append(summary)
        for key, val in inputs.items():
            if isinstance(val, list):
                lines.append("      " + key + " -> node " + str(val[0]) + " output " + str(val[1]))
            else:
                lines.append("      " + key + " = " + str(val))
        if consumed_by:
            lines.append("      consumed by: " + ", ".join("[" + c + "]" for c in consumed_by))
    lines.append("")
    if notes:
        lines.append("## Notes")
        for note in notes:
            lines.append("- " + note)
        lines.append("")
    lines.append("## JSON payload (paste into /prompt)")
    lines.append(json.dumps(nodes, indent=2))
    return "\n".join(lines)


def _references(node: dict[str, Any], node_id: str) -> bool:
    for val in node.get("inputs", {}).values():
        if isinstance(val, list) and val and str(val[0]) == node_id:
            return True
    return False


# --------------------------------------------------------------------------
# Executable repair script
# --------------------------------------------------------------------------
def render_patch_script(decisions: list[Any]) -> str:
    """Emit a ready-to-run python snippet that patches a loaded model.

    The snippet imports vdg.runtime.torch_runtime.TorchRuntime, accepts a
    user-loaded model (model = ... placeholder), and applies every repair
    decision with apply_all. The repair is therefore actually executable
    in-process, not merely described. Non-repair decisions are listed as
    comments (they map to ComfyUI nodes / launch flags instead).
    """
    pairs = _as_pairs(decisions)
    repairs = [(s, c) for s, c in pairs if s in _REPAIR_SKILLS]
    others = [s for s, _c in pairs if s not in _REPAIR_SKILLS]

    lines: list[str] = []
    lines.append('"""VDG-generated repair patch script.')
    lines.append("")
    lines.append("Applies the governance repair decisions to a loaded torch model")
    lines.append("in-process (no diffusers required -- any nn.Module works).")
    lines.append('"""')
    lines.append("")
    lines.append("from vdg.runtime.torch_runtime import TorchRuntime")
    lines.append("")
    lines.append("# Load your model here (any torch nn.Module):")
    lines.append("#   from diffusers import LTXPipeline")
    lines.append("#   pipe = LTXPipeline.from_pretrained('Lightricks/LTX-Video')")
    lines.append("#   model = pipe.transformer        # DiT to repair")
    lines.append("#   vae = pipe.vae                  # VAE to repair")
    lines.append("model = ...  # <-- your model (nn.Module)")
    lines.append("")
    lines.append("rt = TorchRuntime()")
    lines.append("decisions = [")
    for skill, cfg in repairs:
        lines.append("    (" + repr(skill) + ", " + repr(cfg) + "),")
    lines.append("]")
    lines.append("results = rt.apply_all(model, decisions)")
    lines.append("for r in results:")
    lines.append("    print(r['skill'], 'applied=', r['applied'], 'targets=', r['targets'])")
    lines.append("print('patched sites:', sum(r['applied'] for r in results))")
    lines.append("")
    lines.append("# Undo all patches (restores _vdg_original_forward):")
    lines.append("# n = rt.unpatch(model)")
    if others:
        lines.append("")
        lines.append("# Non-repair decisions map to the runtime, not torch patches:")
        for skill in others:
            lines.append("#   " + skill + "  -> ComfyUI node / launch flag (see workflow JSON)")
    return "\n".join(lines)
