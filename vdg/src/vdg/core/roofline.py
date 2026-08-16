"""Roofline performance model for video diffusion transformers.

All formulas here are the HPC roofline model applied to a video DiT denoising
step. The constants are derived from standard transformer FLOP accounting using
the 2*M*N*K matmul convention (one multiply + one add counts as 2 FLOPs):

* Self-attention, per layer:
    - QK^T:   (N, d) @ (d, N)  -> 2 * N^2 * d
    - attn@V: (N, N) @ (N, d)  -> 2 * N^2 * d
    - QKV proj: 2 * N * d * 3d = 6 * N * d^2
    - out proj: 2 * N * d * d  = 2 * N * d^2
  => attention ~ (4 * N^2 * d + 8 * N * d^2) * L      [task spec: O(N^2 * d * L)]
* Feed-forward, per layer (expansion d_ff):
    - up:   (N, d) @ (d, d_ff)   -> 2 * N * d * d_ff
    - down: (N, d_ff) @ (d_ff, d)-> 2 * N * d * d_ff
  => ffn ~ 4 * N * d * d_ff * L                         [task spec: O(N * d * d_ff * L)]

Token count for a 3D spatiotemporal VAE:
    tokens = frames * H * W / (vae_t * vae_h * vae_w * patch^2)

Units convention
----------------
* ``peak_flops`` and ``mem_bw`` are in FLOPs/s and bytes/s (NOT TFLOPS/GB/s).
  Callers convert from device specs (``compute_tflops * 1e12`` and
  ``memory_bandwidth_gbps * 1e9``).
* ``GB`` means 1e9 bytes (decimal), matching how ``memory_gb`` is expressed in
  ``DeviceSpec``.
"""
from __future__ import annotations

__all__ = [
    "PRECISION_BYTES",
    "bytes_per_element",
    "roofline",
    "token_count",
    "attention_flops",
    "ffn_flops",
    "per_step_flops",
    "operational_intensity",
    "predict_step_time",
    "vae_decode_flops",
    "text_encoder_flops",
    "GB",
]

# 1 GB in bytes (decimal). Used everywhere memory is converted to GB.
GB: float = 1e9

# Precision -> bytes per element. fp4/int4 use 0.5 bytes (packed); NVFP4/MXFP4
# likewise count as 0.5 bytes per element for memory modeling.
PRECISION_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "tf32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "fp4": 0.5,
    "nvfp4": 0.5,
    "int8": 1.0,
    "int4": 0.5,
}


def bytes_per_element(precision: str) -> float:
    """Bytes per element for a precision name (case-insensitive)."""
    key = precision.lower()
    if key not in PRECISION_BYTES:
        raise ValueError(
            "Unknown precision: " + repr(precision)
            + ". Known: " + ", ".join(sorted(PRECISION_BYTES))
        )
    return PRECISION_BYTES[key]


def roofline(
    arithmetic_intensity: float,
    peak_flops: float,
    mem_bw: float,
) -> float:
    """Achievable FLOPs/s under the roofline model.

    achievable = min(peak_flops, arithmetic_intensity * mem_bw)

    A workload is *compute-bound* when the result equals ``peak_flops`` and
    *memory-bound* when it is limited by ``ai * mem_bw``.
    """
    if arithmetic_intensity < 0:
        raise ValueError("arithmetic_intensity must be >= 0")
    if peak_flops <= 0:
        raise ValueError("peak_flops must be > 0")
    if mem_bw <= 0:
        raise ValueError("mem_bw must be > 0")
    mem_bound = arithmetic_intensity * mem_bw
    return min(peak_flops, mem_bound)


def token_count(
    frames: int,
    height: int,
    width: int,
    vae_compress: tuple[int, int, int],
    patch_size: int,
) -> int:
    """Number of DiT tokens for a video clip under a 3D VAE + patchify.

    tokens = frames * H * W / (vae_t * vae_h * vae_w * patch^2)

    ``vae_compress`` is ``(temporal, height, width)`` compression. Always >= 1
    so degenerate inputs do not produce zero tokens.
    """
    if frames <= 0 or height <= 0 or width <= 0:
        raise ValueError("frames/height/width must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    vae_t, vae_h, vae_w = vae_compress
    denom = vae_t * vae_h * vae_w * (patch_size * patch_size)
    if denom <= 0:
        raise ValueError("vae_compress components and patch_size must be positive")
    tokens = (frames * height * width) // denom
    return max(1, int(tokens))


def attention_flops(
    tokens: int,
    hidden_dim: int,
    layers: int,
    text_tokens: int = 0,
) -> int:
    """Self-attention (plus optional cross-attention) FLOPs for one denoise step.

    Self-attention: (4 * N^2 * d + 8 * N * d^2) * L
    Cross-attention (text): 4 * N * M_t * d * L   (QK^T + attn@V; projections
    are small relative to the self-attention terms and folded above).
    """
    if tokens <= 0 or hidden_dim <= 0 or layers <= 0:
        raise ValueError("tokens, hidden_dim, layers must be positive")
    n, d, l = tokens, hidden_dim, layers
    self_attn = (4 * n * n * d + 8 * n * d * d) * l
    cross_attn = (4 * n * text_tokens * d) * l if text_tokens > 0 else 0
    return int(self_attn + cross_attn)


def ffn_flops(tokens: int, hidden_dim: int, d_ff: int, layers: int) -> int:
    """Feed-forward FLOPs for one denoise step: 4 * N * d * d_ff * L."""
    if tokens <= 0 or hidden_dim <= 0 or d_ff <= 0 or layers <= 0:
        raise ValueError("tokens, hidden_dim, d_ff, layers must be positive")
    return int(4 * tokens * hidden_dim * d_ff * layers)


def per_step_flops(
    tokens: int,
    hidden_dim: int,
    layers: int,
    d_ff: int,
    heads: int = 0,
    text_tokens: int = 0,
) -> dict[str, int]:
    """Per-step FLOP breakdown for a video DiT.

    Returns ``{"attention": .., "ffn": .., "total": ..}`` where total is the
    sum. ``heads`` is accepted for API symmetry (it does not change the FLOP
    count under standard MHA) so callers can pass full load characteristics.
    """
    attn = attention_flops(tokens, hidden_dim, layers, text_tokens)
    ffn = ffn_flops(tokens, hidden_dim, d_ff, layers)
    return {"attention": attn, "ffn": ffn, "total": attn + ffn}


def operational_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte moved (arithmetic intensity). +inf if no bytes moved."""
    if bytes_moved <= 0:
        return float("inf")
    return flops / bytes_moved


def predict_step_time(
    flops: float,
    peak_flops: float,
    mem_bw: float,
    bytes_moved: float,
) -> float:
    """Seconds for one step via the roofline: flops / achievable_flops."""
    ai = operational_intensity(flops, bytes_moved)
    achievable = roofline(ai, peak_flops, mem_bw)
    return flops / achievable


# --------------------------------------------------------------------------
# Auxiliary phase FLOPs (VAE decode, text encoder) -- separate rooflines.
# These are modeling estimates; the acceleration report flags precise per-video
# VAE FLOPs as a data gap, so constants here are documented and calibratable.
# --------------------------------------------------------------------------

# Conv-decoder pyramid factor: sum of relative spatial positions across an
# upsampling pyramid (1 + 1/r + 1/r^2 + ...). For 2x spatial stages r=4 this is
# 4/3 ~ 1.33; we use 1.5 as a conservative middle-ground for mixed-ratio VAEs.
_VAE_PYRAMID_FACTOR: float = 1.5

# Conv VAEs achieve far below tensor-core peak (low occupancy, elementwise
# GroupNorm/SiLU, no large GEMM). Effective fraction of peak FLOPs.
VAE_EFFICIENCY: float = 0.15


def vae_decode_flops(
    frames: int,
    height: int,
    width: int,
    vae_params_m: float,
    vae_compress: tuple[int, int, int] | None = None,
) -> int:
    """Estimated FLOPs to decode a latent video to pixels.

    A conv VAE does NOT apply all its parameters to every output pixel: the
    high-parameter layers operate near the (compressed) latent resolution while
    only shallow final layers see the full output resolution. We therefore use
    the *geometric mean* of latent and output spatiotemporal positions as the
    effective positions the parameters are applied to -- a standard aggregate
    for a decoder pyramid where param-count and resolution trade off:

        R            = vae_t * vae_h * vae_w      (total compression)
        eff_positions = frames * H * W / sqrt(R)   (geometric mean)
        flops        = 2 * vae_params * eff_positions * PYRAMID_FACTOR

    Validation anchors: for SD's image VAE (50M params, 8x spatial, 512x512) this
    yields ~0.1-0.2s at a 165 TFLOPS device * VAE_EFFICIENCY, matching observed
    SD VAE decode times; for a 175M video VAE at 480p/81f it yields ~5-10s,
    consistent with ComfyUI tiled-VAE observations and a fraction of a 14B
    DiT's denoise time (which dominates as expected for large models).
    """
    if frames <= 0 or height <= 0 or width <= 0 or vae_params_m <= 0:
        raise ValueError("frames/height/width/vae_params_m must be positive")
    output_positions = frames * height * width
    if vae_compress is not None:
        vae_t, vae_h, vae_w = vae_compress
        r = vae_t * vae_h * vae_w
        if r > 0:
            eff_positions = output_positions / (r ** 0.5)
        else:
            eff_positions = output_positions
    else:
        eff_positions = output_positions
    return int(2.0 * vae_params_m * 1e6 * eff_positions * _VAE_PYRAMID_FACTOR)


def text_encoder_flops(te_params_b: float, text_tokens: int) -> int:
    """Estimated FLOPs for one text-encoder forward pass.

    Model: each parameter is applied roughly once per text token (standard
    transformer-encoder accounting): ``2 * te_params * text_tokens``.
    """
    if te_params_b < 0 or text_tokens < 0:
        raise ValueError("te_params_b and text_tokens must be >= 0")
    if te_params_b == 0 or text_tokens == 0:
        return 0
    return int(2.0 * te_params_b * 1e9 * text_tokens)
