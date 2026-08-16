"""DiffusersRuntime -- LTX-Video diffusers pipeline binding.

Binds governance decisions to the HuggingFace diffusers LTX-Video pipeline:

* build_pipeline(model_id) loads an LTXPipeline (falling back to a
  generic DiffusionPipeline.from_pretrained) and returns None gracefully
  when diffusers is not installed (self.last_error records why).
* apply_repairs(pipe, decisions) delegates the repair skills to
  TorchRuntime applied on pipe.transformer and pipe.vae -- the same
  in-process fp32 guards the torch runtime performs, no diffusers-specific
  machinery needed.
* apply_accel(pipe, skill_name, config) maps each accel skill to the
  diffusers API surface (enable_teacache / vae tiling / model CPU offload /
  torch.compile / set_num_inference_steps), best-effort with try/except, and
  reports applied:bool + notes per skill.

All torch/diffusers imports are lazy -- importing this module never requires
either, so pure-sim environments stay importable.
"""
from __future__ import annotations

from typing import Any

from .torch_runtime import TorchRuntime

__all__ = ["DiffusersRuntime"]

# Repair skills handled by TorchRuntime on pipe.transformer / pipe.vae.
_REPAIR_SKILLS: frozenset[str] = frozenset({
    "gelu_fp32", "adaln_fp32", "rmsnorm_fp32", "softmax_fp32", "vae_fp32",
})


class DiffusersRuntime:
    """Binds governance decisions to a diffusers LTX-Video pipeline."""

    def __init__(self) -> None:
        self.last_error: str | None = None
        self._torch_runtime: TorchRuntime | None = None

    # -- pipeline construction -------------------------------------------
    def build_pipeline(self, model_id: str) -> Any | None:
        """Load an LTX-Video pipeline by HuggingFace id, or None on failure.

        Tries diffusers.LTXPipeline.from_pretrained first (the exact
        LTX-Video binding), then falls back to a generic
        DiffusionPipeline.from_pretrained. Returns None and records
        self.last_error if diffusers is missing or the load fails, so
        callers can degrade gracefully.
        """
        try:
            from diffusers import LTXPipeline  # type: ignore
        except ImportError:
            self.last_error = (
                "diffusers not installed; run 'pip install diffusers'. "
                "VDG simulation and envelope generation work without it."
            )
            return None
        try:
            pipe = LTXPipeline.from_pretrained(model_id)
            return pipe
        except Exception as ltx_exc:  # noqa: BLE001 - fall through to generic
            try:
                from diffusers import DiffusionPipeline  # type: ignore
                pipe = DiffusionPipeline.from_pretrained(model_id)
                return pipe
            except Exception as gen_exc:  # noqa: BLE001
                self.last_error = (
                    "LTXPipeline.from_pretrained failed (" + repr(ltx_exc)
                    + "); DiffusionPipeline.from_pretrained failed ("
                    + repr(gen_exc) + ")"
                )
                return None

    # -- repairs ----------------------------------------------------------
    def apply_repairs(
        self, pipe: Any, decisions: list[Any],
    ) -> dict[str, Any]:
        """Apply repair decisions to pipe.transformer and pipe.vae in-process.

        decisions items may be (skill_name, config) tuples, dicts with
        skill/config keys, or GovernanceDecision objects. Only repair
        skills are applied here (accel skills are handled by apply_accel).
        Returns a summary dict::

            {"skills": [...], "applied": total, "transformer": {...},
             "vae": {...}, "skipped": [...], "notes": str}
        """
        pairs = [(self._skill_of(d), self._config_of(d)) for d in decisions]
        repair_pairs = [(s, c) for s, c in pairs if s in _REPAIR_SKILLS]
        skipped = [s for s, _c in pairs if s not in _REPAIR_SKILLS]

        transformer_results: list[dict[str, Any]] = []
        vae_results: list[dict[str, Any]] = []
        dropped: list[str] = []
        applied_total = 0
        transformer = getattr(pipe, "transformer", None)
        vae = getattr(pipe, "vae", None)

        # vae_fp32 targets the VAE; the other repairs target the transformer
        # (or whatever DiT module the pipe exposes).
        for skill, cfg in repair_pairs:
            if skill == "vae_fp32":
                if vae is not None:
                    result = self._apply_one(vae, skill, cfg)
                    vae_results.append(result)
                    applied_total += int(result.get("applied", 0))
                else:
                    dropped.append(skill)
            else:
                if transformer is not None:
                    result = self._apply_one(transformer, skill, cfg)
                    transformer_results.append(result)
                    applied_total += int(result.get("applied", 0))
                elif vae is not None and skill == "gelu_fp32":
                    # Some pipelines only carry a VAE with GELU sites.
                    result = self._apply_one(vae, skill, cfg)
                    vae_results.append(result)
                    applied_total += int(result.get("applied", 0))
                else:
                    dropped.append(skill)

        trans_applied = sum(int(r.get("applied", 0)) for r in transformer_results)
        vae_applied = sum(int(r.get("applied", 0)) for r in vae_results)
        notes = "repaired " + str(applied_total) + " module(s) in-process "
        notes += "(transformer: " + str(trans_applied)
        notes += ", vae: " + str(vae_applied) + ")"
        if dropped:
            notes += "; no target module, skipped: " + ", ".join(dropped)
        return {
            "skills": [s for s, _c in repair_pairs],
            "applied": applied_total,
            "transformer": {"applied": trans_applied, "results": transformer_results},
            "vae": {"applied": vae_applied, "results": vae_results},
            "skipped": skipped + dropped,
            "notes": notes,
        }

    # -- accel ------------------------------------------------------------
    def apply_accel(
        self, pipe: Any, skill_name: str, config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Map one accel skill to the diffusers API surface, best-effort.

        Returns {"skill", "applied": bool, "notes": str}. Every mapping is
        wrapped in try/except: an absent API (e.g. a pipeline without
        enable_teacache) degrades to applied=False with a readable note
        rather than raising.
        """
        cfg = dict(config or {})
        if skill_name == "teacache":
            return self._accel_call(
                skill_name, pipe,
                lambda p: p.enable_teacache(
                    threshold=float(cfg.get("rel_l1_thresh", cfg.get("threshold", 0.1)))
                ),
                "pipe.enable_teacache(threshold=rel_l1_thresh)",
                fallback_module="transformer",
            )
        if skill_name == "vae_tiling":
            return self._accel_call(
                skill_name, pipe.vae if getattr(pipe, "vae", None) is not None else pipe,
                lambda v: v.enable_tiling(
                    tile_size=int(cfg.get("tile_size", 512)),
                    overlap=int(cfg.get("overlap", 64)),
                    temporal_size=int(cfg.get("temporal_size", 64)),
                    temporal_overlap=int(cfg.get("temporal_overlap", 8)),
                ),
                "vae.enable_tiling(tile_size/overlap/temporal_size/temporal_overlap)",
                also=self._enable_temporal_tiling,
            )
        if skill_name == "offload":
            return self._accel_call(
                skill_name, pipe,
                lambda p: p.enable_model_cpu_offload(),
                "pipe.enable_model_cpu_offload()",
            )
        if skill_name == "compile_graph":
            try:
                import torch  # lazy -- torch required for compile
            except ImportError:
                return {
                    "skill": skill_name, "applied": False,
                    "notes": "torch required for torch.compile; not installed",
                }
            transformer = getattr(pipe, "transformer", None)
            if transformer is None:
                return {
                    "skill": skill_name, "applied": False,
                    "notes": "pipe.transformer missing; cannot torch.compile",
                }
            try:
                pipe.transformer = torch.compile(transformer)  # type: ignore
                return {
                    "skill": skill_name, "applied": True,
                    "notes": "pipe.transformer = torch.compile(pipe.transformer)",
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "skill": skill_name, "applied": False,
                    "notes": "torch.compile failed: " + repr(exc),
                }
        if skill_name == "step_distill":
            return self._accel_call(
                skill_name, pipe,
                lambda p: p.set_num_inference_steps(int(cfg.get("steps", 4))),
                "pipe.set_num_inference_steps(steps)",
            )
        if skill_name == "sage_attention":
            return self._accel_call(
                skill_name, pipe,
                lambda p: p.enable_sage_attention(version=cfg.get("version", "v2")),
                "pipe.enable_sage_attention(version) (requires pip install sageattention)",
            )
        if skill_name == "quantization":
            return {
                "skill": skill_name, "applied": False,
                "notes": "diffusers has no in-process quantization API; use the "
                         "ComfyUI-GGUF loader or a TensorRT engine instead",
            }
        return {
            "skill": skill_name, "applied": False,
            "notes": "no diffusers binding for skill " + repr(skill_name),
        }

    # -- helpers ----------------------------------------------------------
    def _apply_one(self, module: Any, skill: str, cfg: dict[str, Any]) -> dict[str, Any]:
        if self._torch_runtime is None:
            self._torch_runtime = TorchRuntime()
        return self._torch_runtime.apply(module, skill, cfg)

    @staticmethod
    def _skill_of(decision: Any) -> str:
        if isinstance(decision, (tuple, list)) and len(decision) >= 1:
            return str(decision[0])
        if isinstance(decision, dict):
            return str(decision.get("skill", ""))
        return str(getattr(decision, "skill_name", ""))

    @staticmethod
    def _config_of(decision: Any) -> dict[str, Any]:
        if isinstance(decision, (tuple, list)) and len(decision) >= 2:
            return dict(decision[1] or {})
        if isinstance(decision, dict):
            return dict(decision.get("config") or {})
        return dict(getattr(decision, "config", None) or {})

    @staticmethod
    def _accel_call(
        skill_name: str,
        target: Any,
        call: Any,
        description: str,
        fallback_module: str | None = None,
        also: Any | None = None,
    ) -> dict[str, Any]:
        notes: list[str] = []
        if target is None:
            return {
                "skill": skill_name, "applied": False,
                "notes": "target object missing; cannot " + description,
            }
        applied = False
        try:
            call(target)
            applied = True
            notes.append(description + " ok")
        except AttributeError:
            notes.append(description + " not available on this pipeline")
        except TypeError:
            notes.append(description + " signature mismatch (kwargs unsupported)")
        except Exception as exc:  # noqa: BLE001
            notes.append(description + " failed: " + repr(exc))
        if not applied and fallback_module is not None:
            sub = getattr(target, fallback_module, None)
            if sub is not None:
                try:
                    call(sub)
                    applied = True
                    notes.append(description + " ok on " + fallback_module)
                except Exception as exc:  # noqa: BLE001
                    notes.append(fallback_module + " fallback failed: " + repr(exc))
        if also is not None and target is not None:
            try:
                also(target)
                applied = True
                notes.append("temporal tiling enabled")
            except Exception:
                pass
        return {"skill": skill_name, "applied": applied, "notes": "; ".join(notes)}

    @staticmethod
    def _enable_temporal_tiling(vae: Any) -> None:
        hook = getattr(vae, "enable_temporal_tiling", None)
        if callable(hook):
            hook()
