"""Shared helpers for repair skills (numerical-robustness fp32 guards).

The repair skills all encode the same device-agnostic principle from the
cross-device robustness report section 7: on a low-precision backend, force fp32
intermediate computation for ops that carry additive-cancellation risk (AdaLN
1+scale), polynomial nonlinearity (GELU/SiLU), normalization statistics
(RMSNorm/LayerNorm/GroupNorm), large-sequence reduction (softmax), or large
value range (VAE decode). This module centralizes the applicability predicate
and the shared applies_to device list so each skill stays DRY.
"""
from __future__ import annotations

from ...core.contracts import DeviceCategory
from ...core.contracts import DeviceSpec

__all__ = [
    "REPAIR_APPLIES_TO",
    "LOW_PRECISION_BACKENDS",
    "LOW_PRECISION_PRECISIONS",
    "low_precision_backend",
]

# Repair skills target Apple Silicon (MPS) and edge-NPU (int8/int4/CoreML/ANE)
# backends. Consumer-NV fp8/Blackwell is handled by accel skills instead: the
# robustness report section 7 notes Blackwell fp8 can relax the fp32 guard.
REPAIR_APPLIES_TO: list[str] = [DeviceCategory.APPLE_SILICON, DeviceCategory.EDGE_NPU]

# Attention backends that signal a low-precision execution path needing fp32
# guards (report section 7 KEEP_FP32 rule: device == mps OR backend in
# {int8, int4, coreml_ane}).
LOW_PRECISION_BACKENDS: set[str] = {"coreml", "coreml_ane", "mps", "ane", "int8", "int4"}

# Precisions that indicate the device runs quantized inference where
# sensitive ops must be kept in fp32. Only int8/int4 trigger the repair guard:
# the robustness report section 7 notes Blackwell fp8 / nvfp4 / fp4 can RELAX
# the fp32 guard (handled by accel/quantization skills instead), while
# consumer-NV int8 and NPU int8/int4 still need it (report S7/S8).
LOW_PRECISION_PRECISIONS: set[str] = {"int8", "int4"}


def low_precision_backend(spec: DeviceSpec) -> bool:
    """Return True when a device runs a low-precision backend needing fp32 guards.

    Mirrors the report section 7 KEEP_FP32 condition:
        KEEP_FP32 = (device == "mps") or (backend in {"int8", "int4", "coreml_ane"})

    Concretely true when any of:
      * the device category is Apple Silicon or edge NPU,
      * an attention backend name is mps / coreml / ane / int8 / int4,
      * the device supports int8 / int4 / fp8 / nvfp4 / fp4 AND is not a
        datacenter card (datacenter fp8/Blackwell is allowed to relax).
    """
    if spec.category in (DeviceCategory.APPLE_SILICON, DeviceCategory.EDGE_NPU):
        return True
    backends = {b.lower() for b in spec.attention_backends}
    if backends & LOW_PRECISION_BACKENDS:
        return True
    # Precision-in-precision trigger: only int8/int4 (Blackwell fp8/fp4 relaxes).
    precisions = {p.lower() for p in spec.supported_precisions}
    if precisions & LOW_PRECISION_PRECISIONS and spec.category != DeviceCategory.DATACENTER:
        return True
    return False
