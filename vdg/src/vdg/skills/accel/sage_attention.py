"""SageAttention acceleration skill (quantized attention, post-training).

Replaces scaled_dot_product_attention with a quantized QK^T (INT8 or INT4) plus
FP8/FP16 PV matmul. It is pure-inference but CUDA + Triton only, so it targets
consumer NVIDIA (Ampere / Ada / Hopper / Blackwell). It is NOT applicable to
Apple Silicon (no Metal backend) or to a non-NVIDIA edge NPU.

Grounding (video-dit-inference-acceleration-report.md, Section 3B / 8.5):
  * v1 (INT8 QK + FP8/FP16 PV): 2-5x over FlashAttention; CogVideoX1.5-5B on
    H20 25'34" -> 12'07" (~2.1x); RTX 5090 560 TOPS, 2.7x over FA2. Negligible
    end-to-end loss.
  * v2 (INT4 QK + FP8 PV, two-level accumulate): ~3x FA2 / ~4.5x xformers on
    RTX 4090; on Hopper ~= FA3-FP8 speed but higher precision. Negligible loss;
    authors recommend v2 for precision-sensitive scenes.
  * v3 (microscaling FP4, Blackwell): RTX 5090 1038 TOPS = 5x over the fastest
    FlashAttention on 5090. FP4 precision is lower than INT4/INT8; authors
    recommend v2 for precision-sensitive workloads.

Applicability design: the base applicable(device, load) signature carries no
config, so it cannot see a version override. Therefore applicable checks the
broad family gate (any version applies -> consumer NVIDIA), and predict
guards the specific configured version: if v3 is requested on a device lacking
FP4, predict returns a neutral no-op impact (speedup 1.0) with a note, so an
impossible 5x is never credited and the planner is told to use v2 instead.

VDG model:
  * v1: speedup 2.5, quality_delta -0.2, energy_ratio 0.9.
  * v2: speedup 3.0 (RTX 4090), quality_delta -0.1 (~0, negligible), energy 0.9.
  * v3: speedup 5.0 (RTX 5090 FP4), quality_delta -0.5 (FP4 lower precision),
    energy 0.7 (FP4 tensor cores). Requires device fp4 support (Blackwell).

Note: the simulator also models SageAttention via the attention_backend
config ("sage1"/"sage2"/"sage3"), which maps to a roofline precision peak and an
ATTENTION_BACKEND_QUALITY_DELTA. This Skill is the composable governance
representation; a planner should use one representation, not both at once.
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceCategory, DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["SageAttention"]

# Per-version impact. Speedups are end-to-end attention-kernel multipliers
# grounded in the report; quality_delta is in VBench-proxy points; energy_ratio
# is the BEYOND-speedup power multiplier (quantized matmuls run on lower-power
# tensor cores, so < 1.0).
_VERSIONS: dict[str, dict[str, float]] = {
    "v1": {"speedup": 2.5, "quality_delta": -0.2, "energy_ratio": 0.9},
    "v2": {"speedup": 3.0, "quality_delta": -0.1, "energy_ratio": 0.9},
    "v3": {"speedup": 5.0, "quality_delta": -0.5, "energy_ratio": 0.7},
}

# Runtime install / pip hints per version (provenance for the apply envelope).
_VERSION_NOTES = {
    "v1": "SageAttention v1 (INT8 QK, ICLR 2025); 2-5x over FA; negligible loss.",
    "v2": "SageAttention2 (INT4 QK + FP8 PV, ICML 2025); ~3x FA2 on 4090; "
          "negligible loss; recommended for precision-sensitive scenes.",
    "v3": "SageAttention3 (microscaling FP4, Blackwell, NeurIPS 2025); 5x over "
          "FA on 5090; FP4 precision lower than INT4/INT8.",
}

# FP4 hardware requirement (Blackwell-only).
_FP4 = "fp4"


def _variant_applicable(device: DeviceProfile, version: str) -> bool:
    """Whether a specific SageAttention version can run on the device."""
    spec = device.spec()
    if spec.category != DeviceCategory.CONSUMER_NV:
        return False  # CUDA + Triton only; no Metal/NPU backend.
    if version == "v3" and not spec.supports(_FP4):
        return False  # v3 needs Blackwell FP4 tensor cores.
    return True


@register_skill("sage_attention")
class SageAttention(Skill):
    """Quantized attention (SageAttention v1/v2/v3). Consumer NVIDIA only."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Broad family gate: any version applies -> consumer NVIDIA. The
        # version-specific FP4 guard is enforced in predict() (see docstring).
        return any(_variant_applicable(device, v) for v in _VERSIONS)

    def default_config(self) -> dict[str, Any]:
        return {"version": "v2"}

    def _version(self, config: dict[str, Any]) -> str:
        version = str(config.get("version", "v2")).lower()
        if version not in _VERSIONS:
            # Tolerate "sage2"/"sage3"/"2"/"3" style inputs.
            short = version.lstrip("sage_")
            if short in ("1", "2", "3"):
                version = "v" + short
        return version if version in _VERSIONS else "v2"

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        version = self._version(cfg)
        # Guard the configured version against the device. If it cannot run,
        # return a neutral no-op so an impossible speedup is never credited.
        if not _variant_applicable(device, version):
            need = "Blackwell FP4 tensor cores" if version == "v3" else "consumer NVIDIA"
            return SkillImpact(
                speedup=1.0,
                memory_ratio=1.0,
                quality_delta=0.0,
                energy_ratio=1.0,
                applies_to=[],
                notes="SageAttention " + version + " needs " + need + "; "
                      + device.spec().name + " lacks it -> no effect. "
                      + ("Use v2 instead." if version == "v3" else ""),
            )
        v = _VERSIONS[version]
        return SkillImpact(
            speedup=v["speedup"],
            memory_ratio=1.0,
            quality_delta=v["quality_delta"],
            energy_ratio=v["energy_ratio"],
            applies_to=[DeviceCategory.CONSUMER_NV],
            notes=_VERSION_NOTES[version],
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        version = self._version(cfg)
        runtime_cfg = {
            "enable": True,
            "version": version,
            "backend": "sage" + version.lstrip("v"),
            "pv_accum_dtype": "fp16" if version == "v2" else "fp32+fp16",
        }
        applied = False
        hook = getattr(model_or_pipeline, "enable_sage_attention", None)
        if callable(hook):
            try:
                hook(version=version)
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="sage_attention",
            runtime="comfyui",
            config=runtime_cfg,
            applied=applied,
            notes="ComfyUI: '--use-sage-attention' CLI flag or Kijai "
                  "WanVideoWrapper sageattn node. diffusers: replace "
                  "F.scaled_dot_product_attention with sageattn. Requires "
                  "'pip install sageattention' + Triton + CUDA (Ada fp8 >=12.4, "
                  "Blackwell/SA2++ >=12.8). v3 needs Blackwell FP4.",
        )
