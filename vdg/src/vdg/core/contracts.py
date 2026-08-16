"""Base-class contracts for VDG.

Every concrete device, load, skill and governance agent in later phases
subclasses the classes defined here. The public method signatures in this file
are FROZEN -- do not rename parameters or change return shapes, because phase-2
plugins and agents build against them. See ``CONTRACTS.md`` for the canonical
reference.

All numeric modeling constants used by the default ``LoadModel`` implementations
live here (and in ``roofline.py``) so they are visible at the contract layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .registry import Registrable
from .roofline import (
    GB,
    bytes_per_element,
    ffn_flops,
    attention_flops,
    per_step_flops,
    token_count,
)

__all__ = [
    "DeviceSpec",
    "DeviceProfile",
    "VideoDiTLoad",
    "LoadModel",
    "SkillImpact",
    "Skill",
    "GovernanceDecision",
    "DeviceCategory",
]


# Canonical device categories. Used by ``SkillImpact.applies_to`` so a skill can
# declare which device families it targets (e.g. a SageAttention skill applies
# only to consumer_nv, not apple_silicon/edge_npu).
class DeviceCategory:
    CONSUMER_NV = "consumer_nv"
    APPLE_SILICON = "apple_silicon"
    EDGE_NPU = "edge_npu"
    DATACENTER = "datacenter"


# --------------------------------------------------------------------------
# Device contracts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceSpec:
    """Immutable hardware specification of a compute device.

    ``compute_tflops`` maps precision name -> peak dense TFLOPs (1 TFLOPS =
    1e12 FLOPs/s). Sparse ratings (e.g. RTX 5090 "3352 AI TOPS FP4-sparse")
    should be stored as their dense-equivalent or documented in the subclass;
    the simulator multiplies by 1e12 to get FLOPs/s.

    Grounded example values (from the edge-deployment report) live in the
    phase-2 device plugins; this dataclass only defines the schema.
    """

    name: str
    category: str
    memory_gb: float
    memory_bandwidth_gbps: float
    compute_tflops: dict[str, float]
    tdp_w: float
    idle_power_w: float
    supported_precisions: list[str]
    attention_backends: list[str]
    unified_memory: bool = False
    cost_per_hour_usd: float | None = None

    def peak_flops(self, precision: str) -> float:
        """Peak FLOPs/s for a precision (1e12 * compute_tflops[precision])."""
        key = precision.lower()
        if key not in self.compute_tflops:
            raise ValueError(
                "Precision " + repr(precision) + " not in compute_tflops for "
                + self.name + ". Available: " + ", ".join(sorted(self.compute_tflops))
            )
        return self.compute_tflops[key] * 1e12

    def mem_bw_bytes(self) -> float:
        """Memory bandwidth in bytes/s (1e9 * memory_bandwidth_gbps)."""
        return self.memory_bandwidth_gbps * 1e9

    def supports(self, precision: str) -> bool:
        return precision.lower() in {p.lower() for p in self.supported_precisions}


class DeviceProfile(Registrable):
    """Base class for all device plugins.

    Subclasses are decorated with ``@register_device`` and implement ``spec``.
    ``is_available`` and ``measure_power`` probe the real host gracefully and
    MUST never raise -- they return ``False``/``None`` when the device is absent
    or the probing library is missing (e.g. pynvml on a Mac/NPU).
    """

    def spec(self) -> DeviceSpec:
        raise NotImplementedError("DeviceProfile.spec must be implemented by subclasses")

    def is_available(self) -> bool:
        """Return True if this device is present on the current host.

        Default implementation returns False. Subclasses should probe via
        ``torch.cuda.is_available()``, nvidia-ml, ``powermetrics`` (Apple) or
        the relevant NPU SDK, wrapped so any failure degrades to False.
        """
        return False

    def measure_power(self) -> float | None:
        """Instantaneous power draw in watts, or None if unreadable.

        Default returns None. NVIDIA subclasses may read via pynvml; Apple
        Silicon via powermetrics; NPUs via vendor SDK. Must not raise.
        """
        return None


# --------------------------------------------------------------------------
# Load (video DiT) contracts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VideoDiTLoad:
    """Characteristics of a video diffusion transformer load.

    ``vae_compress`` is ``(temporal, height, width)`` spatial-temporal
    compression of the 3D VAE. ``patch_size`` is the DiT patchify factor.
    ``vae_params_m`` is the VAE parameter count in millions.
    """

    model_name: str
    params_b: float
    vae_compress: tuple[int, int, int]
    patch_size: int
    te_params_b: float
    layers: int
    hidden_dim: int
    heads: int
    default_steps: int
    supported_tasks: list[str]
    vae_params_m: float
    # FFN expansion ratio relative to hidden_dim. Wan/LTX/Hunyuan commonly use
    # ~4x (SwiGLU/GELU); stored explicitly so per_step_flops is grounded.
    ffn_expansion: float = 4.0

    @property
    def d_ff(self) -> int:
        return int(self.hidden_dim * self.ffn_expansion)


class LoadModel(Registrable):
    """Base class for all video-DiT load plugins (LTX-2.3 is primary)."""

    def characteristics(self) -> VideoDiTLoad:
        raise NotImplementedError("LoadModel.characteristics must be implemented")

    def tokens_for(self, resolution: tuple[int, int], frames: int) -> int:
        """DiT token count for a (width, height) resolution and frame count."""
        c = self.characteristics()
        width, height = resolution
        return token_count(frames, height, width, c.vae_compress, c.patch_size)

    def per_step_flops(self, tokens: int, text_tokens: int = 0) -> dict[str, int]:
        """Per-denoise-step FLOP breakdown: {attention, ffn, total}."""
        c = self.characteristics()
        return per_step_flops(
            tokens,
            hidden_dim=c.hidden_dim,
            layers=c.layers,
            d_ff=c.d_ff,
            heads=c.heads,
            text_tokens=text_tokens,
        )

    def memory_footprint(self, precision: str, tokens: int) -> dict[str, float]:
        """Memory footprint in GB for weights + KV + activations.

        weights    = params_b * 1e9 * bytes_per_elem / GB
        kv         = tokens * d * L * 2(K&V) * bytes_per_elem / GB
        activations= tokens * d * L * bytes_per_elem / GB
        total_gb   = weights + kv + activations
        """
        c = self.characteristics()
        bpe = bytes_per_element(precision)
        weights = c.params_b * 1e9 * bpe / GB
        kv = tokens * c.hidden_dim * c.layers * 2 * bpe / GB
        activations = tokens * c.hidden_dim * c.layers * bpe / GB
        total = weights + kv + activations
        return {
            "weights": weights,
            "kv": kv,
            "activations": activations,
            "total_gb": total,
        }


# --------------------------------------------------------------------------
# Skill contracts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SkillImpact:
    """Predicted effect of applying a skill, relative to the unskilled baseline.

    * ``speedup``: end-to-end latency multiplier (1.0 = no effect, 2.0 = 2x
      faster). The simulator composes multiple skills sub-multiplicatively to
      model bottleneck transfer / scope overlap (see simulator docs).
    * ``memory_ratio``: peak-memory multiplier (1.0 = no effect, 0.3 = 70%
      memory cut, e.g. VAE temporal tiling 32GB->8GB on the VAE segment).
    * ``quality_delta``: VBench-proxy delta in points (0.0 = no effect, negative
      = worse). Grounded where possible (TeaCache -0.07, STA finetune -0.09).
    * ``energy_ratio``: energy multiplier BEYOND the latency speedup (1.0 = no
      extra effect; the simulator already divides energy by the combined
      speedup). Use <1.0 for power-efficiency gains (e.g. FP4 tensor cores).
    * ``applies_to``: list of ``DeviceCategory`` strings the skill targets; an
      empty list means "all categories".
    * ``notes``: free-form provenance / caveats.
    """

    speedup: float = 1.0
    memory_ratio: float = 1.0
    quality_delta: float = 0.0
    energy_ratio: float = 1.0
    applies_to: list[str] = field(default_factory=list)
    notes: str = ""


class Skill(Registrable):
    """Base class for all repair and acceleration skills.

    A skill is pluggable: it declares whether it applies to a (device, load)
    pair, provides a default config, predicts its impact for governance
    planning, and (optionally) applies itself to a real model/pipeline.

    ``kind`` is either "repair" (numerical-robustness patches, e.g. the MPS
    three-cast fp32 fix encoded from MPS_BLACK_VIDEO_FIX.md) or "accel"
    (inference acceleration, e.g. TeaCache / SageAttention / distillation).
    """

    kind: str = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        """Whether this skill can run on the given (device, load)."""
        return True

    def default_config(self) -> dict[str, Any]:
        """Sensible default configuration for ``predict``/``apply``."""
        return {}

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        """Predict the impact of applying this skill (for governance planning)."""
        raise NotImplementedError("Skill.predict must be implemented")

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        """Apply this skill to a real model or pipeline.

        The foundation ships without a runtime, so the default is a documented
        no-op stub returning the object unchanged. Concrete skills with a real
        backend (ComfyUI/LightX2V/MLX) override this to perform the actual
        patch (e.g. the MPS fp32 casts, TeaCache hooks, SageAttention swap).
        """
        return model_or_pipeline


@dataclass(frozen=True)
class GovernanceDecision:
    """A governance agent's decision to apply a skill with a rationale."""

    skill_name: str
    config: dict[str, Any]
    predicted_impact: SkillImpact
    rationale: str
