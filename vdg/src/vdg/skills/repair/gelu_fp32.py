"""GELU fp32-cast repair skill.

Encodes the first of the three battle-tested LTX-2.3 MPS fixes
(MPS_BLACK_VIDEO_FIX.md section 5.2 (1)): on Apple Silicon MPS the fused bf16
GELU-tanh Metal kernel diverges to NaN for |x| >= 15. This is a kernel defect,
not an overflow -- bf16 shares fp32's 8-bit exponent so x^3 does not overflow;
only the MPS fused kernel is buggy (CPU bf16 GELU is correct). The fix casts the
input to fp32, computes GELU-tanh, and casts back to the original dtype. This is
the same workaround Hunyuan3D / HiDream ship (comfy/ldm/hunyuan3dv2_1).

Robustness report section 8 records the threshold: bf16 |x| >= 15 -> NaN on MPS;
fp16 |x| > 40 overflows 65504. Section 7 lists gelu / gelu_tanh / silu in
PREC_GUARD_OPS.
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import REPAIR_APPLIES_TO, low_precision_backend

__all__ = ["GeluFP32", "patch_gelu"]


def patch_gelu(module: Any, config: dict[str, Any] | None = None) -> Any:
    """Wrap an nn.GELU module so GELU-tanh runs in fp32 on MPS.

    Faithful to MPS_BLACK_VIDEO_FIX.md section 5.2 (1):

        if x.device.type == "mps":
            x = F.gelu(x.float(), approximate="tanh").to(dtype=x.dtype)
        else:
            x = F.gelu(x, approximate="tanh")

    The module's own approximate attribute is honored when present (LTX uses the
    tanh approximation); an explicit config override wins. Imports torch lazily
    so this module is importable in pure-sim environments without a torch
    runtime -- the patch only needs torch at apply time.
    """
    import torch.nn.functional as F  # noqa: F401  (runtime patch backend)

    cfg = config or {}
    approximate = cfg.get("approximate", None)
    if approximate is None:
        approximate = getattr(module, "approximate", "tanh")

    def forward(x: Any, *args: Any, **kwargs: Any) -> Any:
        # MPS fused bf16 GELU-tanh kernel -> NaN for |x|>=15; cast to fp32.
        if hasattr(x, "device") and getattr(x.device, "type", "") == "mps":
            out = F.gelu(x.float(), approximate=approximate)
            return out.to(dtype=x.dtype)
        return F.gelu(x, approximate=approximate)

    # Preserve the original for inspection / unpatching; the new forward fully
    # replaces a GELU activation module's behavior (its forward is just F.gelu).
    module._vdg_original_forward = getattr(module, "forward", None)
    module.forward = forward
    module._vdg_patched = "gelu_fp32"
    return module


@register_skill("gelu_fp32")
class GeluFP32(Skill):
    """Force fp32 GELU-tanh on low-precision backends (MPS bf16 NaN fix)."""

    kind = "repair"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        return low_precision_backend(device.spec())

    def default_config(self) -> dict[str, Any]:
        return {"approximate": "tanh"}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        return SkillImpact(
            speedup=0.92,
            memory_ratio=1.0,
            quality_delta=2.0,
            energy_ratio=1.05,
            applies_to=list(REPAIR_APPLIES_TO),
            notes=(
                "MPS bf16 fused GELU-tanh kernel -> NaN at |x|>=15 (kernel defect, "
                "not overflow); fp16 |x|>40 overflows 65504. fp32 cast fixes black "
                "frames. Cast costs ~8% latency on the GELU op. Robustness report S8."
            ),
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        return patch_gelu(model_or_pipeline, config)
