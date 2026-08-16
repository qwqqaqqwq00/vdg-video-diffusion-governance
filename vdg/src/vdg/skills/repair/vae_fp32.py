"""VAE fp32-cast repair skill.

Encodes the VAE-decode branch of the cross-device robustness report section 7
PREC_GUARD_OPS template (vae_decode). The video VAE operates over a large value
range (intermediate conv features are O(100), report section 8) and relies on
GroupNorm, which is numerically unstable in low precision; an int8 / fp16 VAE
decode produces black frames or banding even when the DiT latent is correct. The
fix runs the whole VAE decoder in fp32 (cast the latent input to fp32 so the
dtype propagates through every conv / GroupNorm / SiLU, then cast the pixel
output back). The report section 8 is explicit: VAE decode must stay fp16/fp32,
never int8.

This wraps the VAE decode entry point (the decode method when present, else
forward). The same pattern is used for the encoder when encoding a conditioning
image.
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import REPAIR_APPLIES_TO, low_precision_backend

__all__ = ["VAEFP32", "patch_vae"]


def _wrap_callable(obj: Any, attr: str) -> tuple[bool, Any]:
    """Return (had_attr, original) for a callable attribute, else (False, None)."""
    if hasattr(obj, attr) and callable(getattr(obj, attr)):
        return True, getattr(obj, attr)
    return False, None


def patch_vae(module: Any, config: dict[str, Any] | None = None) -> Any:
    """Wrap a VAE so decode runs end-to-end in fp32 on MPS.

    Faithful to the report section 7 guarded_op template applied to the VAE
    decode segment: on MPS, cast the latent input to fp32 (the dtype then
    propagates through the whole decoder -- convs, GroupNorm, SiLU all run in
    fp32), call the original decode, and cast the pixel output back to the
    input dtype. Non-MPS devices call the original unchanged.

    Both the decode method (primary VAE entry) and forward are wrapped when
    present. Imports torch lazily.
    """
    import torch  # noqa: F401  (runtime patch backend, used for isinstance check)

    cfg = config or {}
    # Optionally force a specific output dtype instead of mirroring the input.
    force_dtype = cfg.get("dtype", None)

    def _make_wrapper(original: Any) -> Any:
        def wrapped(x: Any, *args: Any, **kwargs: Any) -> Any:
            if hasattr(x, "device") and getattr(x.device, "type", "") == "mps":
                orig_dtype = x.dtype
                out = original(x.float(), *args, **kwargs)
                target = force_dtype if force_dtype is not None else orig_dtype
                if isinstance(out, torch.Tensor):
                    return out.to(dtype=target)
                # Some VAEs return (pixels, extra); cast the leading tensor.
                if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
                    out = list(out)
                    out[0] = out[0].to(dtype=target)
                    return tuple(out) if isinstance(out, tuple) else out
                return out
            return original(x, *args, **kwargs)

        return wrapped

    patched_any = False
    for attr in ("decode", "forward"):
        had, original = _wrap_callable(module, attr)
        if had:
            setattr(module, "_vdg_original_" + attr, original)
            setattr(module, attr, _make_wrapper(original))
            patched_any = True

    if patched_any:
        module._vdg_patched = "vae_fp32"
    return module


@register_skill("vae_fp32")
class VAEFP32(Skill):
    """Force fp32 VAE decode on low-precision backends (black-frame / banding fix)."""

    kind = "repair"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        return low_precision_backend(device.spec())

    def default_config(self) -> dict[str, Any]:
        return {"dtype": None}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        return SkillImpact(
            speedup=0.80,
            memory_ratio=1.0,
            quality_delta=2.0,
            energy_ratio=1.10,
            applies_to=list(REPAIR_APPLIES_TO),
            notes=(
                "VAE decode has large value range (features O(100)) and unstable "
                "GroupNorm; int8/fp16 VAE -> black frames / banding even with a "
                "correct latent. Whole-decoder fp32 fixes it. VAE is a big conv "
                "segment so fp32 costs ~20% of the VAE time. Report S7/S8."
            ),
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        return patch_vae(model_or_pipeline, config)
