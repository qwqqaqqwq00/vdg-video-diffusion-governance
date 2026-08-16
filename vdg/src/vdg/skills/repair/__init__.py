"""Repair skills (numerical robustness) -- phase 2.

Encodes the device-agnostic precision-guard template: on low-precision backends,
force fp32 intermediate computation for sensitive ops (GELU/SiLU, AdaLN (1+scale)
modulation, RMSNorm/LayerNorm/GroupNorm, softmax, timestep-embed MLP, VAE
decode), at op / boundary-block / high-vulnerability-layer-set granularity.

Grounded in 'MPS_BLACK_VIDEO_FIX.md' (LTX-2.3 three-cast fix: GELU,
scale_msa/shift_msa/gate_msa, scale_mlp/shift_mlp/gate_mlp) and the cross-device
robustness report (six crash points, PREC_GUARD_OPS). Each repair Skill is a
subclass of vdg.core.contracts.Skill with kind="repair", registered via
@register_skill; importing this package makes them available to the
simulator and governance agents.

The numerical probe (NumericalProbe) is the diagnostic that justifies
applying these skills: it tests each sensitive op on boundary inputs comparing
cpu-fp32 vs the target device/precision, and falls back to a pure-sim (numpy)
report when torch or the device is unavailable, so the platform runs anywhere.
"""
from __future__ import annotations

# Importing the modules executes the @register_skill decorators, registering
# each repair skill in the global REGISTRY. The probe module is imported last
# so callers can also reach it as vdg.skills.repair.NumericalProbe.
from .gelu_fp32 import GeluFP32, patch_gelu
from .adaln_fp32 import AdaLNFP32, patch_adaln
from .rmsnorm_fp32 import RMSNormFP32, patch_rmsnorm
from .softmax_fp32 import SoftmaxFP32, patch_softmax
from .vae_fp32 import VAEFP32, patch_vae
from .boundary_bf16 import BoundaryBF16, patch_boundary_blocks
from .numerical_probe import (
    NumericalProbe,
    DiagnosticReport,
    OpResult,
    probe,
)

__all__ = [
    # Repair skills (registered via @register_skill).
    "GeluFP32",
    "AdaLNFP32",
    "RMSNormFP32",
    "SoftmaxFP32",
    "VAEFP32",
    "BoundaryBF16",
    # Patch functions (the real backend code).
    "patch_gelu",
    "patch_adaln",
    "patch_rmsnorm",
    "patch_softmax",
    "patch_vae",
    "patch_boundary_blocks",
    # Diagnostic.
    "NumericalProbe",
    "DiagnosticReport",
    "OpResult",
    "probe",
]
