"""Softmax fp32-cast repair skill.

Encodes the softmax / attention-scores branch of the cross-device robustness
report section 7 PREC_GUARD_OPS template. On low-precision backends the softmax
reduction (x - max, exp, sum, divide) must run in fp32: for very large sequences
the exp underflows (fp16 exp underflows for a score difference > 11, report
section 8) and the sum loses precision, producing NaN / silently-zero
probabilities (PyTorch MPS issue #96602). The fix casts the logits to fp32,
computes softmax, and casts back.

This wraps an nn.Softmax module; for inline F.softmax inside a custom attention
block the same pattern is applied by wrapping that block's forward.
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import REPAIR_APPLIES_TO, low_precision_backend

__all__ = ["SoftmaxFP32", "patch_softmax"]


def patch_softmax(module: Any, config: dict[str, Any] | None = None) -> Any:
    """Wrap an nn.Softmax module so softmax runs in fp32 on MPS.

    Faithful to the report section 7 guarded_op template: cast input to fp32,
    compute, cast back. The module's dim is honored (config override wins).
    Imports torch lazily.
    """
    import torch.nn.functional as F  # noqa: F401  (runtime patch backend)

    cfg = config or {}
    default_dim = cfg.get("dim", getattr(module, "dim", -1))
    original_forward = getattr(module, "forward", None)

    def forward(x: Any, *args: Any, **kwargs: Any) -> Any:
        dim = kwargs.get("dim", None)
        if dim is None and args:
            # nn.Softmax.forward(x, dim) sometimes passes dim positionally.
            dim = args[0] if not isinstance(args[0], (int,)) else default_dim
        if dim is None:
            dim = default_dim
        if hasattr(x, "device") and getattr(x.device, "type", "") == "mps":
            orig_dtype = x.dtype
            return F.softmax(x.float(), dim=dim).to(dtype=orig_dtype)
        return F.softmax(x, dim=dim)

    module._vdg_original_forward = original_forward
    module.forward = forward
    module._vdg_patched = "softmax_fp32"
    return module


@register_skill("softmax_fp32")
class SoftmaxFP32(Skill):
    """Force fp32 softmax on low-precision backends (large-seq exp underflow fix)."""

    kind = "repair"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        return low_precision_backend(device.spec())

    def default_config(self) -> dict[str, Any]:
        return {"dim": -1}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        return SkillImpact(
            speedup=0.93,
            memory_ratio=1.0,
            quality_delta=1.5,
            energy_ratio=1.04,
            applies_to=list(REPAIR_APPLIES_TO),
            notes=(
                "Large-sequence softmax: fp16 exp underflows for score diff > 11 "
                "(MPS #96602); bf16 sum loses precision. fp32 reduction fixes NaN/"
                "zero-prob attention. Cast cost ~7%. Report S7/S8."
            ),
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        return patch_softmax(model_or_pipeline, config)
