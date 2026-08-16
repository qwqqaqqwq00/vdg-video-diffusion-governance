"""TorchRuntime -- real in-process repair patch application on nn.Module.

This is the concrete runtime behind the previously-stub Skill.apply()
envelope: given an actual torch.nn.Module (any video DiT transformer / VAE
that torch can walk), it LOCATES the sensitive submodules and applies the real
repair patch functions from vdg.skills.repair (the battle-tested fp32
guards encoded from MPS_BLACK_VIDEO_FIX.md / the cross-device robustness
report section 7).

Supported skills (repair set):

    gelu_fp32      -> vdg.skills.repair.gelu_fp32.patch_gelu
    adaln_fp32     -> vdg.skills.repair.adaln_fp32.patch_adaln
    rmsnorm_fp32   -> vdg.skills.repair.rmsnorm_fp32.patch_rmsnorm
    softmax_fp32   -> vdg.skills.repair.softmax_fp32.patch_softmax
    vae_fp32       -> vdg.skills.repair.vae_fp32.patch_vae

find_sensitive_modules walks the module tree by class-name / attribute-name
heuristics (names containing gelu / adaln / rms / layer_norm / softmax / norm,
plus the LTX scale_shift_table marker for AdaLN blocks), so the runtime can
find patch sites in ANY DiT without knowing the exact architecture.

All torch imports are lazy: importing this module never requires torch; only
calling the apply/patch methods does. If torch is missing, those methods raise
RuntimeError with "torch required for runtime application".
"""
from __future__ import annotations

import importlib
from typing import Any, Iterable

__all__ = ["TorchRuntime", "REPAIR_PATCH_SITES"]

# skill_name -> (module path, patch function name) of the REAL patch backend.
REPAIR_PATCH_SITES: dict[str, tuple[str, str]] = {
    "gelu_fp32": ("vdg.skills.repair.gelu_fp32", "patch_gelu"),
    "adaln_fp32": ("vdg.skills.repair.adaln_fp32", "patch_adaln"),
    "rmsnorm_fp32": ("vdg.skills.repair.rmsnorm_fp32", "patch_rmsnorm"),
    "softmax_fp32": ("vdg.skills.repair.softmax_fp32", "patch_softmax"),
    "vae_fp32": ("vdg.skills.repair.vae_fp32", "patch_vae"),
}

# skill_name -> find_sensitive_modules bucket used to locate patch sites.
_SKILL_BUCKET: dict[str, str] = {
    "gelu_fp32": "gelu",
    "adaln_fp32": "adaln",
    "rmsnorm_fp32": "rmsnorm",
    "softmax_fp32": "softmax",
}

_TORCH_MISSING_MSG = (
    "torch required for runtime application: 'import torch' failed. "
    "Install torch (e.g. 'pip install torch') or run in a torch environment."
)


def _require_torch() -> Any:
    """Import torch or raise RuntimeError with the canonical missing message."""
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover - depends on host
        raise RuntimeError(_TORCH_MISSING_MSG) from exc


class TorchRuntime:
    """Locates and applies VDG repair patches on real torch modules."""

    # -- public API -------------------------------------------------------
    def apply(
        self,
        module: Any,
        skill_name: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply one repair skill to module (or its sensitive children).

        * vae_fp32 patches the module itself (the whole VAE).
        * the other repair skills patch the module itself when it matches the
          skill's sensitive-module heuristics, otherwise every matching child
          located by find_sensitive_modules.

        Returns a result dict::

            {"skill", "config", "targets": [names], "applied": n,
             "failed": [names], "notes": str}

        applied counts how many submodules were actually patched. Unknown
        skills / failed patches are reported in the result, never raised --
        so apply_all can keep going over a full decision list.
        """
        self._require_torch()
        cfg = dict(config or {})
        if skill_name not in REPAIR_PATCH_SITES:
            return {
                "skill": skill_name,
                "config": cfg,
                "targets": [],
                "applied": 0,
                "failed": [],
                "notes": "no torch patch site for skill "
                         + repr(skill_name) + "; not a repair skill",
            }
        patch_fn = self._patch_function(skill_name)
        if skill_name == "vae_fp32":
            # The whole module is the VAE -- patch its decode/forward entry.
            if getattr(module, "_vdg_patched", None):
                return {
                    "skill": skill_name,
                    "config": cfg,
                    "targets": ["self"],
                    "applied": 0,
                    "failed": [],
                    "notes": "vae_fp32: module already patched; skipped",
                }
            try:
                patch_fn(module, cfg)
                marked = getattr(module, "_vdg_patched", None)
                if marked == "vae_fp32":
                    return {
                        "skill": skill_name,
                        "config": cfg,
                        "targets": ["self"],
                        "applied": 1,
                        "failed": [],
                        "notes": "vae_fp32: whole-module decode/forward wrapped in fp32",
                    }
                # patch_vae does not mark a module with no callable decode/
                # forward -- treat as a no-op, not a successful patch.
                return {
                    "skill": skill_name,
                    "config": cfg,
                    "targets": ["self"],
                    "applied": 0,
                    "failed": ["self"],
                    "notes": "vae_fp32: module exposes no callable decode/forward; "
                             "nothing patched",
                }
            except Exception as exc:
                return {
                    "skill": skill_name,
                    "config": cfg,
                    "targets": ["self"],
                    "applied": 0,
                    "failed": ["self"],
                    "notes": "vae_fp32 patch failed: " + repr(exc),
                }

        bucket = _SKILL_BUCKET[skill_name]
        walk_items = list(self._walk(module))
        if self._module_matches(module, bucket):
            # Caller passed a leaf module (a single GELU / RMSNorm / ...).
            names = ["self"]
        else:
            names = [
                name for name, child in walk_items
                if name and self._module_matches(child, bucket)
            ]
        by_name = dict(walk_items)
        applied: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        for name in names:
            target = module if name == "self" else by_name[name]
            if getattr(target, "_vdg_patched", None):
                # Idempotency guard: the repair patch fns overwrite
                # _vdg_original_forward unconditionally, so re-applying to an
                # already-marked module would break unpatch().
                skipped.append(name)
                continue
            try:
                patch_fn(target, cfg)
                if getattr(target, "_vdg_patched", None) == skill_name:
                    applied.append(name)
                else:
                    failed.append(name + " (patch fn did not mark module)")
            except Exception as exc:
                failed.append(name + " (" + repr(exc) + ")")
        return {
            "skill": skill_name,
            "config": cfg,
            "targets": names,
            "applied": len(applied),
            "failed": failed,
            "notes": self._notes_for(skill_name, names, applied, skipped, failed),
        }

    def apply_all(
        self,
        module: Any,
        decisions: Iterable[Any],
    ) -> list[dict[str, Any]]:
        """Apply a list of (skill_name, config) pairs (or decision dicts).

        Each item may be a (skill_name, config) tuple, a dict with
        skill / config keys, or a GovernanceDecision (it exposes
        skill_name / config attributes). Returns one result dict per
        decision, in input order.
        """
        results: list[dict[str, Any]] = []
        for decision in decisions:
            skill_name, config = self._unpack_decision(decision)
            results.append(self.apply(module, skill_name, config))
        return results

    def unpatch(self, module: Any) -> int:
        """Remove every VDG patch in the module tree and restore originals.

        Walks the module recursively; for each module carrying _vdg_patched
        it restores forward / decode from the _vdg_original_*
        attributes the repair patches saved, then clears the marks. Returns the
        number of modules unpatched.
        """
        self._require_torch()
        count = 0
        for _name, child in self._walk(module):
            if not getattr(child, "_vdg_patched", None):
                continue
            for attr in ("forward", "decode"):
                original = getattr(child, "_vdg_original_" + attr, None)
                if original is not None:
                    try:
                        setattr(child, attr, original)
                    except Exception:
                        pass
                if hasattr(child, "_vdg_original_" + attr):
                    try:
                        delattr(child, "_vdg_original_" + attr)
                    except Exception:
                        pass
            if hasattr(child, "_vdg_patched"):
                try:
                    delattr(child, "_vdg_patched")
                except Exception:
                    pass
            count += 1
        return count

    def find_sensitive_modules(self, module: Any) -> dict[str, list[Any]]:
        """Locate sensitive submodules by class-name / attribute-name heuristics.

        Returns {"gelu": [...], "adaln": [...], "rmsnorm": [...],
        "softmax": [...]}. Classification precedence:

        * "gelu"     -- name or class name contains "gelu",
        * "adaln"    -- name/class contains "adaln" OR the module carries the
          LTX scale_shift_table attribute (AdaLN block marker),
        * "softmax"  -- name or class name contains "softmax",
        * "rmsnorm"  -- name contains "rms" / "layer_norm" / "norm", or the
          class name contains "rmsnorm" / "layernorm" / "rms".

        The root module itself (walk name "") is skipped; children are returned
        as (name, module) pairs where the name is the dotted path from the
        root (e.g. transformer_blocks.3.attn1.to_out.0). Classification
        matches on the LEAF path segment (attn1, norm1) and the class
        name, so a parent named e.g. adaln_block does not taint its
        descendants.
        """
        self._require_torch()
        found: dict[str, list[Any]] = {"gelu": [], "adaln": [], "rmsnorm": [], "softmax": []}
        for name, child in self._walk(module):
            if not name:
                continue  # the root itself
            bucket = self._classify(child, name)
            if bucket is not None:
                found[bucket].append((name, child))
        return found

    # -- internals --------------------------------------------------------
    def _require_torch(self) -> Any:
        return _require_torch()

    @staticmethod
    def _patch_function(skill_name: str) -> Any:
        module_path, func_name = REPAIR_PATCH_SITES[skill_name]
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name)

    @staticmethod
    def _walk(module: Any) -> Iterable[tuple[str, Any]]:
        try:
            return module.named_modules()
        except AttributeError:  # not a torch module
            return iter([("", module)])

    @staticmethod
    def _unpack_decision(decision: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(decision, (tuple, list)) and len(decision) >= 2:
            return str(decision[0]), dict(decision[1] or {})
        if isinstance(decision, dict):
            return str(decision.get("skill", "")), dict(decision.get("config") or {})
        # GovernanceDecision-style object (skill_name / config attributes).
        return str(getattr(decision, "skill_name", "")), dict(
            getattr(decision, "config", None) or {}
        )

    @staticmethod
    def _classify(module: Any, name: str) -> str | None:
        """Return the sensitive bucket a module belongs to, or None."""
        lowered = name.rsplit(".", 1)[-1].lower()
        cls_name = type(module).__name__.lower()
        if "gelu" in lowered or "gelu" in cls_name:
            return "gelu"
        if (
            "adaln" in lowered
            or "adaln" in cls_name
            or hasattr(module, "scale_shift_table")
        ):
            return "adaln"
        if "softmax" in lowered or "softmax" in cls_name:
            return "softmax"
        if (
            "rms" in lowered
            or "layer_norm" in lowered
            or "norm" in lowered
            or "rmsnorm" in cls_name
            or "layernorm" in cls_name
        ):
            return "rmsnorm"
        return None

    @staticmethod
    def _module_matches(module: Any, bucket: str) -> bool:
        try:
            return TorchRuntime._classify(module, "") == bucket
        except Exception:
            return False

    @staticmethod
    def _notes_for(
        skill_name: str, names: list[str], applied: list[str],
        skipped: list[str], failed: list[str],
    ) -> str:
        parts = [
            skill_name + ": patched " + str(len(applied)) + "/" + str(len(names))
            + " site(s) in-process via vdg.skills.repair"
        ]
        if applied:
            parts.append("sites: " + ", ".join(applied))
        if skipped:
            parts.append("already patched, skipped: " + ", ".join(skipped))
        if failed:
            parts.append("failed: " + "; ".join(failed))
        return "; ".join(parts)
