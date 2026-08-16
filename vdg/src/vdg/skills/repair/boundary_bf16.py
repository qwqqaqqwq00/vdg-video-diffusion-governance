"""Boundary-block bf16 protection repair skill (block-level precision guard).

Keeps the first ``n_first`` and last ``n_last`` transformer blocks in bf16
while the middle blocks run quantized (int8 / HiF8 / fp8). Boundary blocks are
the most fragile: they see the rawest inputs and the final accumulation, so
quantization error there propagates furthest. This is the block-level instance
of the same principle the op-level fp32 guards encode -- the robustness report
calls out the three-granularity stack: op-level fp32 (GELU/SiLU, AdaLN,
RMSNorm, softmax, VAE), block-level bf16 boundary protection, and layer-set
bf16 protection.

Grounding (DiT_跨设备数值鲁棒性研究报告.md, sections 2 / 7 / 10):
  * Wan2.1-T2V-14B HiFloat8 quantization (arXiv 2606.00957, ICME 2026) keeps
    the first 2 + last 3 blocks as BF16 -- demonstrated on Ascend 910B NPU.
  * Ideogram 4.0 INT8 quantization (arXiv 2606.12280) uses bf16 to protect a
    small set of high-vulnerability layers.
  * Report section 7 template: on low-precision backends, sensitive ops keep
    fp32 intermediate computation; boundary-block bf16 is the complementary
    block-level granularity and can be stacked with the op-level guards.
  * Consumer-NV int8: same template, with SmoothQuant + per-channel/per-token
    + boundary-block bf16. NPU (Ascend/Hexagon): HiFloat8/int8 + first/last
    block bf16.

VDG model: quality_delta +1.5 (recovers VBench dimensions that quantizing
boundary blocks would lose -- HiF8 on Ascend keeps all 5 VBench dims >= BF16),
speedup 0.95 (tiny cost: boundary blocks stay at bf16 tensor-core speed instead
of int8). applicable: edge_npu + apple_silicon + int8/int4 quantized backends
(low_precision_backend predicate -- same gate as the op-level repair skills).
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import REPAIR_APPLIES_TO, low_precision_backend

__all__ = ["BoundaryBF16", "patch_boundary_blocks"]

# Well-known transformer-block container attribute names, in preference order.
_BLOCK_ATTRS = (
    "blocks",
    "layers",
    "transformer_blocks",
    "diT_blocks",
    "dit_blocks",
    "model",
)


def _find_block_container(module: Any) -> tuple[Any, str | None]:
    """Locate the ordered container of transformer blocks.

    Returns ``(container, attr_name)`` where ``container`` is indexable and
    sized (nn.ModuleList / nn.Sequential / plain list) and attr_name is the
    attribute under which it hangs, or ``(None, None)`` when no known container
    attribute is present. A config-level ``block_attr`` override wins.
    """
    cfg_attr = getattr(module, "_vdg_block_attr", None)
    candidates = (_BLOCK_ATTRS if not cfg_attr else (cfg_attr,))
    for attr in candidates:
        container = getattr(module, attr, None)
        if container is not None and hasattr(container, "__len__"):
            return container, attr
    return None, None


def patch_boundary_blocks(
    module: Any,
    n_first: int = 2,
    n_last: int = 3,
    dtype: str = "bfloat16",
) -> Any:
    """Register hooks keeping boundary blocks in bf16 on a quantized model.

    Selects the first ``n_first`` and last ``n_last`` blocks of the model's
    transformer-block container (found via the known container attributes) and
    registers forward hooks that cast each boundary block's input and output to
    ``dtype`` (default bf16), so the quantized (int8/HiF8) path never sees the
    most fragile layers. Faithful to the Wan2.1-T2V-14B HiF8 scheme (first 2 +
    last 3 blocks bf16 on Ascend 910B, arXiv 2606.00957).

    Pure-sim safe: torch is imported lazily; a module without a recognized
    block container is returned unchanged (with ``_vdg_boundary_applied``
    recorded as False).
    """
    container, attr = _find_block_container(module)
    if container is None:
        module._vdg_boundary_applied = False
        return module

    import torch  # noqa: F401  (runtime patch backend)

    n = len(container)
    first = min(max(int(n_first), 0), n)
    last = min(max(int(n_last), 0), n - first)
    target_dtype = getattr(torch, dtype, torch.bfloat16)

    def _keep_bf16(input_tensor: Any) -> Any:
        """Cast a tensor (or tensor tuple/list) to the protected dtype."""
        if isinstance(input_tensor, tuple):
            return tuple(_keep_bf16(t) for t in input_tensor)
        if isinstance(input_tensor, list):
            return [_keep_bf16(t) for t in input_tensor]
        if hasattr(input_tensor, "dtype") and input_tensor.is_floating_point():
            return input_tensor.to(dtype=target_dtype)
        return input_tensor

    for idx in list(range(first)) + list(range(n - last, n)):
        block = container[idx]
        # Pre-hook: cast the incoming hidden state to bf16 so the block never
        # sees quantized activations; post-hook: keep the output bf16 for the
        # next (possibly quantized) block boundary.
        block.register_forward_pre_hook(lambda _m, args: _keep_bf16(args))
        block.register_forward_hook(lambda _m, _inp, out: _keep_bf16(out))

    module._vdg_boundary_applied = True
    module._vdg_boundary_config = {
        "n_first": first,
        "n_last": last,
        "dtype": dtype,
        "block_attr": attr,
    }
    return module


@register_skill("boundary_block_bf16")
class BoundaryBF16(Skill):
    """Block-level bf16 guard for quantized backends (boundary blocks)."""

    kind = "repair"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # edge_npu + apple_silicon + int8/int4 quantized backends -- the same
        # low-precision gate as the op-level fp32 repair skills.
        return low_precision_backend(device.spec())

    def default_config(self) -> dict[str, Any]:
        # Wan2.1-T2V-14B HiF8 scheme: first 2 + last 3 blocks bf16.
        return {"n_first": 2, "n_last": 3, "dtype": "bfloat16"}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        n_first = int(cfg.get("n_first", 2))
        n_last = int(cfg.get("n_last", 3))
        return SkillImpact(
            speedup=0.95,
            memory_ratio=1.0,
            quality_delta=1.5,
            energy_ratio=1.05,
            applies_to=list(REPAIR_APPLIES_TO),
            notes="Boundary-block bf16 protection: keeps first "
                  + str(n_first) + " + last " + str(n_last)
                  + " blocks bf16 under quantized inference (Wan2.1-T2V-14B "
                  "HiF8 on Ascend 910B keeps all 5 VBench dims >= BF16; "
                  "arXiv 2606.00957). Block-level complement to op-level fp32 "
                  "guards -- three-granularity stack (op / block / layer-set). "
                  "Costs ~5% on boundary blocks (bf16 vs int8).",
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        n_first = int(cfg.get("n_first", 2))
        n_last = int(cfg.get("n_last", 3))
        dtype = str(cfg.get("dtype", "bfloat16"))
        applied = False
        try:
            patched = patch_boundary_blocks(
                model_or_pipeline,
                n_first=n_first,
                n_last=n_last,
                dtype=dtype,
            )
            applied = bool(getattr(patched, "_vdg_boundary_applied", False))
        except Exception:
            applied = False
        return {
            "skill": "boundary_block_bf16",
            "runtime": "diffusers",
            "config": {
                "n_first": n_first,
                "n_last": n_last,
                "dtype": dtype,
                "enabled": True,
            },
            "applied": applied,
            "notes": "patch_boundary_blocks(module, n_first=" + str(n_first)
                     + ", n_last=" + str(n_last)
                     + ") registers forward hooks keeping boundary blocks in "
                     + dtype + " (Wan2.1-T2V-14B HiF8 / Ascend 910B scheme). "
                     + "Needs a recognized transformer-block container "
                     + "(blocks/layers/transformer_blocks). Complement to "
                     + "op-level fp32 repair; stack together for quantized "
                     + "backends.",
        }
