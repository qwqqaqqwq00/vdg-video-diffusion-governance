"""Video DiT load plugins (phase 2).

The primary load is LTX-2.3 (Lightricks). Secondary reference loads (Wan 2.1/2.2,
HunyuanVideo, CogVideoX, Open-Sora 2.0) provide comparability. Each module
defines a 'LoadModel' subclass decorated with '@register_load' so it
self-registers when this package is imported.

Importing this package (which 'vdg.__init__' does via '_import_subpackages')
causes every decorated load class to register in the global 'REGISTRY' under
the 'load' kind, making them available to the simulator and governance agents.
"""
from __future__ import annotations

# Importing video_dit triggers @register_load on all 9 load classes, adding
# them to the global REGISTRY. The import itself is the registration side-effect.
from . import video_dit  # noqa: F401

# Re-export the load classes and helpers for convenient top-level access.
from .video_dit import (
    LTX_2_3,
    Wan21_T2V_1_3B,
    Wan21_T2V_14B,
    Wan21_I2V_14B,
    Wan22_A14B_MoE,
    Wan22_TI2V_5B_Dense,
    HunyuanVideo_13B,
    CogVideoX_5B,
    OpenSora2_11B,
    list_all_loads,
    recommended_model_for,
)

__all__ = [
    "LTX_2_3",
    "Wan21_T2V_1_3B",
    "Wan21_T2V_14B",
    "Wan21_I2V_14B",
    "Wan22_A14B_MoE",
    "Wan22_TI2V_5B_Dense",
    "HunyuanVideo_13B",
    "CogVideoX_5B",
    "OpenSora2_11B",
    "list_all_loads",
    "recommended_model_for",
]
