"""RMSNorm fp32-cast repair skill.

Encodes the RMSNorm branch of the cross-device robustness report section 7
PREC_GUARD_OPS template (rmsnorm / layernorm / groupnorm). On low-precision
backends the normalization statistics (mean of squares, reciprocal square root)
must be computed in fp32: fp16 squares overflow 65504 for |x| > 256 (report
section 8, PyTorch MPS issue #96113), and the division loses precision. The fix
casts the input to fp32, computes the RMS statistic and the division in fp32,
then casts back to the original dtype. The affine weight is applied in the
original dtype to avoid storing a fp32 copy (the cancellation risk is in the
statistic, not the scale).

The LTX-2.3 DiT block normalizes via comfy.ldm.common_dit.rms_norm (a function,
not a module); for an nn.RMSNorm-style module this skill wraps forward directly.
For the inline functional call in an LTX block, the adaln_fp32 skill already
casts x to fp32 around rms_norm (section 5.2 (2)(3)).
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import REPAIR_APPLIES_TO, low_precision_backend

__all__ = ["RMSNormFP32", "patch_rmsnorm"]


def patch_rmsnorm(module: Any, config: dict[str, Any] | None = None) -> Any:
    """Wrap an RMSNorm module so the statistic + division run in fp32 on MPS.

    Faithful to the report section 7 guarded_op template:
        if op in PREC_GUARD_OPS and KEEP_FP32:
            orig_dtype = x.dtype
            x32 = x.float()
            y32 = op(x32, ...)
            return y32.to(orig_dtype)

    The module's eps and optional weight are honored. The RMS is computed as
    x * rsqrt(mean(x^2) + eps) over the last dimension. Imports torch lazily.
    """
    import torch  # noqa: F401  (runtime patch backend)

    cfg = config or {}
    eps = float(cfg.get("eps", getattr(module, "eps", 1e-6)))
    weight = getattr(module, "weight", None)
    dim = int(cfg.get("dim", getattr(module, "dim", -1)))
    original_forward = getattr(module, "forward", None)

    def forward(x: Any, *args: Any, **kwargs: Any) -> Any:
        if hasattr(x, "device") and getattr(x.device, "type", "") == "mps":
            orig_dtype = x.dtype
            x32 = x.float()
            # mean of squares + reciprocal sqrt in fp32 (avoids fp16 overflow).
            ms = x32.pow(2).mean(dim=dim, keepdim=True)
            y32 = x32 * torch.rsqrt(ms + eps)
            y = y32.to(dtype=orig_dtype)
            if weight is not None:
                y = y * weight
            return y
        return original_forward(x, *args, **kwargs)

    module._vdg_original_forward = original_forward
    module.forward = forward
    module._vdg_patched = "rmsnorm_fp32"
    return module


@register_skill("rmsnorm_fp32")
class RMSNormFP32(Skill):
    """Force fp32 RMSNorm statistics on low-precision backends (fp16 overflow fix)."""

    kind = "repair"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        return low_precision_backend(device.spec())

    def default_config(self) -> dict[str, Any]:
        return {"eps": 1e-6, "dim": -1}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        return SkillImpact(
            speedup=0.95,
            memory_ratio=1.0,
            quality_delta=1.0,
            energy_ratio=1.03,
            applies_to=list(REPAIR_APPLIES_TO),
            notes=(
                "fp16 RMSNorm square-sum overflows 65504 for |x|>256 (MPS #96113); "
                "bf16 loses precision in the variance. fp32 statistic+division fixes "
                "NaN/zero norms. RMSNorm is cheap so cast cost ~5%. Report S7/S8."
            ),
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        return patch_rmsnorm(model_or_pipeline, config)
