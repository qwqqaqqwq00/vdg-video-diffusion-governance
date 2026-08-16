"""AdaLN fp32-cast repair skill (the critical black-frame fix).

Encodes the second and third of the three battle-tested LTX-2.3 MPS fixes
(MPS_BLACK_VIDEO_FIX.md section 5.2 (2) and (3)), transcribed verbatim from the
user's patched Lightricks block (comfy/ldm/lightricks/model.py,
BasicTransformerBlock.forward). The cross-device robustness report section 1.3
marks AdaLN as the single most critical divergence point (starred): the
(1 + scale) term suffers catastrophic cancellation on bf16 because bf16's 7-bit
mantissa cannot represent a result near zero -- when |1 + scale| < 2^-7 ~= 0.0078
the whole scale branch is silently zeroed (report section 8).

The fix, on Apple Silicon MPS, casts the six modulation tensors
(scale_msa/shift_msa/gate_msa and scale_mlp/shift_mlp/gate_mlp, derived from
self.scale_shift_table) to fp32, casts x to fp32, and runs the modulation in
fp32 before handing the (cast-back) input to attn1 / ff. The self-attention path
runs attn1 itself in fp32 (its modulated input is fp32); the MLP path casts the
modulated input back to x.dtype before ff so ff stays in the original dtype,
matching the real LTX code exactly.

Robustness report section 7 generalizes this as the adln_modulate guarded_op:
    def adln_modulate(x, scale, shift, gate=None):
        if KEEP_FP32:
            x32, s32, sh32 = x.float(), scale.float(), shift.float()
            y = rms_norm_fp32(x32) * (1.0 + s32) + sh32
            if gate is not None:
                y = y * gate.float()
            return y.to(x.dtype)
        return rms_norm(x) * (1 + scale) + shift
"""
from __future__ import annotations

import types
from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import REPAIR_APPLIES_TO, low_precision_backend

__all__ = ["AdaLNFP32", "patch_adaln"]


def _get_rms_norm() -> Any:
    """Resolve the LTX rms_norm function, falling back to a pure implementation."""
    try:
        from comfy.ldm.common_dit import rms_norm as _fn  # type: ignore
        return _fn
    except Exception:
        return _rms_norm_fallback


def _rms_norm_fallback(x: Any, eps: float = 1e-6) -> Any:
    """No-affine RMSNorm: x * rsqrt(mean(x^2) + eps) over the last dimension."""
    import torch  # noqa: F401
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps)


def _get_apply_cross_attention_adaln() -> Any:
    """Resolve LTX apply_cross_attention_adaln, or None when unavailable."""
    try:
        from comfy.ldm.lightricks.model import (  # type: ignore
            apply_cross_attention_adaln as _fn,
        )
        return _fn
    except Exception:
        return None


def _is_mps(x: Any) -> bool:
    return hasattr(x, "device") and getattr(x.device, "type", "") == "mps"


def patch_adaln(block: Any, config: dict[str, Any] | None = None) -> Any:
    """Wrap an LTX/Lightricks AdaLN DiT block so modulation runs in fp32 on MPS.

    Faithful to comfy/ldm/lightricks/model.py BasicTransformerBlock.forward. On
    MPS the six modulation tensors (scale_msa/shift_msa/gate_msa and
    scale_mlp/shift_mlp/gate_mlp) and x are cast to fp32 and the modulation is
    computed in fp32; the self-attention input stays fp32 (attn1 runs in fp32),
    while the MLP modulated input is cast back to x.dtype before ff. On non-MPS
    devices the original forward is called unchanged, so the patch is a strict
    superset of the unpatched behavior outside MPS.

    Imports torch / comfy lazily so the module is importable in pure-sim
    environments. Only blocks exposing the LTX scale_shift_table + attn1 + ff
    interface are patched; others are returned unchanged.
    """
    import torch  # noqa: F401  (runtime patch backend)

    if getattr(block, "_vdg_patched", None) == "adaln_fp32":
        return block
    if not (
        hasattr(block, "scale_shift_table")
        and hasattr(block, "attn1")
        and hasattr(block, "ff")
    ):
        # Not an LTX-style AdaLN block we can patch in place.
        return block

    rms_norm = _get_rms_norm()
    apply_cross_adaln = _get_apply_cross_attention_adaln()
    original_forward = block.forward

    def forward(
        self: Any,
        x: Any,
        context: Any = None,
        attention_mask: Any = None,
        timestep: Any = None,
        pe: Any = None,
        transformer_options: Any = None,
        self_attention_mask: Any = None,
        prompt_timestep: Any = None,
        **kwargs: Any,
    ) -> Any:
        # Non-MPS: defer to the original forward byte-for-byte.
        if not _is_mps(x) or timestep is None:
            return original_forward(
                x,
                context=context,
                attention_mask=attention_mask,
                timestep=timestep,
                pe=pe,
                transformer_options=transformer_options,
                self_attention_mask=self_attention_mask,
                prompt_timestep=prompt_timestep,
                **kwargs,
            )

        transformer_options = transformer_options if transformer_options is not None else {}
        sst = self.scale_shift_table

        # Modulation vectors (faithful copy of LTX line 534).
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            sst[None, None, :6].to(device=x.device, dtype=x.dtype)
            + timestep.reshape(x.shape[0], timestep.shape[1], sst.shape[0], -1)[:, :, :6, :]
        ).unbind(dim=2)

        # --- (2) self-attention modulation, fp32 on MPS (LTX lines 537-540) ---
        scale_msa, shift_msa, gate_msa = (t.float() for t in (scale_msa, shift_msa, gate_msa))
        x_fp32 = x.float()
        attn_out = self.attn1(
            rms_norm(x_fp32) * (1.0 + scale_msa) + shift_msa,
            pe=pe,
            mask=self_attention_mask,
            transformer_options=transformer_options,
        ) * gate_msa
        x = (x_fp32 + attn_out).to(dtype=x.dtype)

        # --- cross-attention (unchanged by the patch; LTX lines 544-551) ---
        if getattr(self, "cross_attention_adaln", False) and apply_cross_adaln is not None:
            shift_q_mca, scale_q_mca, gate_q_mca = (
                sst[None, None, 6:9].to(device=x.device, dtype=x.dtype)
                + timestep.reshape(x.shape[0], timestep.shape[1], sst.shape[0], -1)[:, :, 6:9, :]
            ).unbind(dim=2)
            x = x + apply_cross_adaln(
                x, context, self.attn2, shift_q_mca, scale_q_mca, gate_q_mca,
                self.prompt_scale_shift_table, prompt_timestep,
                attention_mask, transformer_options,
            )
        else:
            x = x + self.attn2(
                x, context=context, mask=attention_mask, transformer_options=transformer_options,
            )

        # --- (3) MLP modulation, fp32 on MPS (LTX lines 555-559) ---
        # addcmul(y, y, scale) == y + y*scale == y*(1+scale), computed in fp32 to
        # avoid the (1+scale) cancellation; the modulated input is cast back to
        # x.dtype before ff so ff runs in the original dtype.
        scale_mlp_f, shift_mlp_f, gate_mlp_f = (t.float() for t in (scale_mlp, shift_mlp, gate_mlp))
        y_f32 = rms_norm(x.float())
        y_f32 = torch.addcmul(y_f32, y_f32, scale_mlp_f).add_(shift_mlp_f)
        y = self.ff(y_f32.to(dtype=x.dtype))
        x.addcmul_(y, gate_mlp_f.to(dtype=x.dtype))

        return x

    block._vdg_original_forward = original_forward
    block.forward = types.MethodType(forward, block)
    block._vdg_patched = "adaln_fp32"
    return block


@register_skill("adaln_fp32")
class AdaLNFP32(Skill):
    """Force fp32 AdaLN modulation on low-precision backends (the critical fix)."""

    kind = "repair"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        return low_precision_backend(device.spec())

    def default_config(self) -> dict[str, Any]:
        return {}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        return SkillImpact(
            speedup=0.90,
            memory_ratio=1.0,
            quality_delta=2.5,
            energy_ratio=1.06,
            applies_to=list(REPAIR_APPLIES_TO),
            notes=(
                "AdaLN (1+scale) catastrophic cancellation on bf16: |1+scale|<2^-7 "
                "(~0.0078) zeroes the whole scale branch (report S1.3, starred most "
                "critical). fp32 modulation fixes black frames / silent modulation "
                "loss. attn1 runs in fp32 on the patched path so cost ~10%. S7/S8."
            ),
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        return patch_adaln(model_or_pipeline, config)
