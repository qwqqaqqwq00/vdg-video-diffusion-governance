"""Acceleration skills -- phase 2.

Encodes the composable inference-acceleration stack with grounded impact
numbers from the acceleration report (video-dit-inference-acceleration-report.md):
  * step distillation (CoDMD 50->4 ~25x; Dynamic-in-Few-Step 30x; LTX-2.3
    default ~30-step -> 4-8 step LightX2V)
  * cross-step caching (TeaCache 1.6-4.0x via threshold 0.05->0.25, -0.07%
    VBench at 4.41x; device-agnostic)
  * quantized attention (SageAttention v1 2.5x / v2 3x FA2 on 4090 / v3 5x on
    5090 FP4; consumer NVIDIA only)
  * fused exact attention (FlashAttention-2 1.5x / FA-3 1.8x Hopper; consumer
    NVIDIA + datacenter; FA-4 Blackwell)
  * sliding tile attention (STA: HunyuanVideo 945s->685s training-free,
    1.4x, no quality loss; 2.5x finetuned, -0.09 VBench; consumer NVIDIA)
  * linear attention (SANA-Video 2.0 architecture, 16x vs Wan1.3B
    training-side; conservative 3.0x for existing models, needs finetune)
  * MLX fused SDPA (mx.fast.scaled_dot_product_attention 1.4x + avoids 4.3GB
    T x T matrix; Apple Silicon only)
  * quantization (GGUF Q4_K_M Wan14B 29->10 GB; NVFP4 3x on 5090; INT8 edge NPU)
  * VAE tiling (HunyuanVideo 32 GB -> 8 GB; speedup 0.9)
  * offload / block-swap (Kijai 14B Wan 20/40 blocks ~16 GB; <5 GB with context
    window; speedup 0.6)
  * context window (Kijai long-video chunking: 1025 frames -> window 81 +
    overlap 16 -> <5 GB VRAM on 5090; memory_ratio 0.35, speedup 0.9)
  * diffusion forcing (CogVideoX1.5 frame-packing 3-4x speed / 3x memory;
    conservative 2.0x; training-side architecture)
  * compile / graph (torch.compile 1.5x FLUX / 20-40% video; TensorRT 2.5x
    backbone Firefly)

Each skill subclasses vdg.core.contracts.Skill (kind="accel") and registers
via @register_skill. Importing this package imports every submodule so the
decorated classes self-register in REGISTRY.
"""
from __future__ import annotations

from . import (
    compile_graph,
    context_window,
    diffusion_forcing,
    flash_attention,
    linear_attention,
    mlx_sdpa,
    offload,
    quantization,
    sage_attention,
    sliding_tile_attention,
    step_distill,
    teacache,
    vae_tiling,
)

__all__ = [
    "teacache",
    "sage_attention",
    "quantization",
    "step_distill",
    "vae_tiling",
    "offload",
    "compile_graph",
    "sliding_tile_attention",
    "linear_attention",
    "mlx_sdpa",
    "flash_attention",
    "context_window",
    "diffusion_forcing",
]
