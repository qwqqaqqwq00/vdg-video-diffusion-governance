"""Video diffusion transformer (video DiT) load plugins for VDG.

This module defines the load library -- concrete 'LoadModel' subclasses that
describe the characteristics of real open-source video generation models. Each
class is decorated with '@register_load' so it self-registers on import and
becomes available to the simulator and governance agents.

PRIMARY MODEL: LTX-2.3 (Lightricks) -- the hero load. It is the model the user
has prepared and battle-tested on Apple Silicon (MPS), documented in
MPS_BLACK_VIDEO_FIX.md. Its high-compression 3D VAE (8x32x32 = 8192x total)
yields very low token counts, enabling realtime-capable generation at 1216x704.

SECONDARY REFERENCE LOADS (for cross-model comparability):
  - Wan 2.1 T2V 1.3B / 14B, I2V 14B (Wan-VAE 3D causal, UMT5-XXL text encoder)
  - Wan 2.2 A14B MoE (27B total / 14B active) and TI2V-5B dense (Wan2.2-VAE)
  - HunyuanVideo 13B (Causal 3D VAE, MLLM text encoder)
  - CogVideoX 5B (3D Causal VAE, T5-XXL, 226-token text cap)
  - Open-Sora 2.0 11B (Video DC-AE high-compression, FLUX-initialized MMDiT)

GROUNDING PROVENANCE
--------------------
All architecture numbers (layers, hidden_dim, heads, ffn_dim, patch_size) are
fetched directly from the HuggingFace model config.json / transformer source
code of each model. Parameter counts are grounded from safetensors metadata or
checkpoint file sizes. VAE compression ratios are grounded from model configs
and official model cards / papers. Where a value could not be directly verified
(notably some VAE parameter counts and the Open-Sora 2.0 layer count), it is
explicitly marked as an estimate in the docstring with its rationale.

Key sources:
  - LTX-Video: huggingface.co/Lightricks/LTX-Video (transformer/config.json,
    vae/config.json, diffusers transformer_ltx.py source, safetensors metadata)
  - Wan 2.1: huggingface.co/Wan-AI/Wan2.1-{T2V-1.3B,T2V-14B,I2V-14B-480P}
    (config.json, safetensors metadata, VAE file size, UMT5-XXL file size)
  - Wan 2.2: huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers + Wan2.2-TI2V-5B
    (transformer/config.json), github.com/Wan-Video/Wan2.2 README (MoE + VAE)
  - HunyuanVideo: huggingface.co/tencent/HunyuanVideo (vae/config.json, README,
    fp8 checkpoint size), diffusers transformer_hunyuan_video.py source
  - CogVideoX: huggingface.co/THUDM/CogVideoX-5b (transformer/config.json)
  - Open-Sora 2.0: arXiv 2503.09642 (FLUX-init, DC-AE 4x32x32, 11B), report
    cross-references; layer count estimated from the FLUX base architecture.

The 'LoadModel' base class (in 'vdg.core.contracts') already implements
'tokens_for', 'per_step_flops' and 'memory_footprint' from the roofline
model. Each subclass here ONLY implements 'characteristics()' returning a
'VideoDiTLoad' dataclass -- it does not re-implement those methods.
"""
from __future__ import annotations

from ..core.contracts import DeviceCategory, DeviceSpec, LoadModel, VideoDiTLoad
from ..core.registry import REGISTRY, register_load
from ..core.roofline import GB, bytes_per_element

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


# --------------------------------------------------------------------------
# Primary load: LTX-2.3 (Lightricks)
# --------------------------------------------------------------------------
@register_load
class LTX_2_3(LoadModel):
    """LTX-2.3 (Lightricks) video DiT -- the primary modeled load.

    LTX-Video is a compact 2B-parameter video diffusion transformer designed for
    realtime generation. Its defining feature is an extremely high-compression
    3D VAE (8x temporal, 32x32 spatial = 8192x total compression) that produces
    very few DiT tokens even at 1216x704 resolution, enabling ~30fps realtime
    generation at low resolutions and fast generation at higher resolutions.

    This is the model the user has prepared and battle-tested on Apple Silicon
    (MPS). The MPS black-video / NaN fix (MPS_BLACK_VIDEO_FIX.md) targets this
    model's AdaLN modulation (scale_msa/shift_msa/gate_msa + scale_mlp/...) and
    GELU-tanh fused kernel -- the exact three-cast fp32 repair encoded as the
    VDG numerical-robustness skill.

    I2V (image-to-video) is the primary task; T2V is also supported. The
    robustness report notes I2V is more numerically stable than T2V (T2V's text
    conditioning path has wider modulation-parameter dynamic range, more prone
    to the AdaLN scale-approx-negative-1 catastrophic cancellation).

    Grounding (huggingface.co/Lightricks/LTX-Video):
      - Transformer config.json: num_layers=28, num_attention_heads=32,
        attention_head_dim=64 -> hidden_dim=2048, patch_size=1 (spatial),
        patch_size_t=1 (temporal), cross_attention_dim=2048,
        caption_channels=4096, activation_fn=gelu-approximate.
      - diffusers transformer_ltx.py: FeedForward(dim, activation_fn=...) uses
        default mult=4.0 -> d_ff=8192, ffn_expansion=4.0. Each block has
        self-attn + cross-attn + FFN with an AdaLN scale_shift_table of shape
        (6, dim) -- the exact (scale_msa, shift_msa, gate_msa, scale_mlp,
        shift_mlp, gate_mlp) pattern from the MPS fix.
      - VAE config.json: block_out_channels=[128,256,512,512],
        spatio_temporal_scaling=[true,true,true,false] (3 downsample stages ->
        2^3=8x temporal + 2^3=8x spatial), patch_size=4 (VAE patchify ->
        additional 4x spatial), patch_size_t=1. Total: (8, 32, 32).
      - safetensors metadata (original LTX-Video v1): transformer F32 params
        = 1,923,385,472 (~1.92B). HOWEVER, the user's actual model is LTX-2.3
        (a.k.a. LTX-2, file ltx-2-19b-dev.safetensors) which is ~19B params
        (43 GB fp16, 22 GB Q8 GGUF). End-to-end ComfyUI benchmark on M4 Max
        (2026-08-17) confirmed 9.69 s/step for the 19B Q8 model, matching a
        19B roofline estimate -- NOT the 1.92B original. params_b updated to
        19.0 to reflect the real deployed model.
      - Text encoder: LTX-2.3 uses Google Gemma-3-12B (not T5-XXL). The 12B
        Gemma fp8 checkpoint is ~12 GB; te_params_b updated to 12.0.
      - VAE: LTX23_video_vae_bf16 (1.4 GB), separate from LTX2_video_vae.
      - Default ~30-step flow-matching scheduler; distilled to ~4-8 steps
        (ltxv-*-distilled checkpoints) for realtime. scenario.py uses 30 steps.
    """

    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="LTX-2.3",
            params_b=19.0,  # LTX-2.3 (LTX-2) = 19B, NOT 1.92B (that was v1)
            vae_compress=(8, 32, 32),
            patch_size=1,
            te_params_b=12.0,  # Gemma-3-12B (not T5-XXL 4.76B)
            layers=28,
            hidden_dim=2048,
            heads=32,
            default_steps=30,
            supported_tasks=["i2v", "t2v"],
            vae_params_m=419.2,
            ffn_expansion=4.0,
        )


# --------------------------------------------------------------------------
# Wan 2.1 loads
# --------------------------------------------------------------------------
@register_load
class Wan21_T2V_1_3B(LoadModel):
    """Wan 2.1 Text-to-Video 1.3B (Alibaba).

    The compact Wan variant, commonly deployed on Apple Silicon (MLX official
    port: M4 Max ~90 s/it for 480p/81f, ~75 min at 50 steps, ~6 min at 4-step
    distillation) and consumer NVIDIA GPUs.

    Grounding (huggingface.co/Wan-AI/Wan2.1-T2V-1.3B config.json + API):
      dim=1536, num_layers=30, num_heads=12, ffn_dim=8960
      (ffn_expansion=8960/1536=5.833), in_dim=16, text_len=512, model_type=t2v.
      safetensors metadata: F32 params = 1,418,996,800 (~1.42B; the '1.3B' name
      is rounded down).
      VAE: Wan-VAE 3D causal, compress (4, 8, 8), transformer patch [1,2,2].
      Text encoder: UMT5-XXL (multilingual), ~5.68B params (11.36 GB bf16).
      VAE: ~126.9M params (Wan2.1_VAE.pth 507.6 MB fp32).
    """

    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="Wan2.1-T2V-1.3B",
            params_b=1.419,
            vae_compress=(4, 8, 8),
            patch_size=2,
            te_params_b=5.68,
            layers=30,
            hidden_dim=1536,
            heads=12,
            default_steps=50,
            supported_tasks=["t2v"],
            vae_params_m=126.9,
            ffn_expansion=8960 / 1536,
        )


@register_load
class Wan21_T2V_14B(LoadModel):
    """Wan 2.1 Text-to-Video 14B (Alibaba).

    The flagship Wan 2.1 dense model. Widely deployed on consumer NVIDIA (4090:
    LightX2V ~20.26 s/it cfg for 480p/81f/40-step; ~13.5 min at 40 steps, ~1 min
    at 4-step distillation) and datacenter (H100x8 ~30s). GGUF Q4_K_M brings
    weights from 29.1 GB to 10.1 GB for low-VRAM fit.

    Grounding (huggingface.co/Wan-AI/Wan2.1-T2V-14B config.json + API):
      dim=5120, num_layers=40, num_heads=40, ffn_dim=13824
      (ffn_expansion=13824/5120=2.7), in_dim=16, text_len=512, model_type=t2v.
      safetensors metadata: F32 params = 14,288,491,584 (~14.29B).
      VAE + text encoder: same family as Wan21_T2V_1_3B.
    """

    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="Wan2.1-T2V-14B",
            params_b=14.288,
            vae_compress=(4, 8, 8),
            patch_size=2,
            te_params_b=5.68,
            layers=40,
            hidden_dim=5120,
            heads=40,
            default_steps=50,
            supported_tasks=["t2v"],
            vae_params_m=126.9,
            ffn_expansion=13824 / 5120,
        )


@register_load
class Wan21_I2V_14B(LoadModel):
    """Wan 2.1 Image-to-Video 14B (Alibaba).

    The I2V variant of Wan 2.1 14B. Same DiT backbone as T2V-14B but with
    in_dim=36 (16 latent + 20 image-condition channels) for the image-to-video
    conditioning path. TeaCache reports the highest speedup on Wan2.1 I2V
    (up to 2.9x) vs T2V (1.4-2.0x) in the acceleration report.

    Grounding (huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P config.json):
      dim=5120, num_layers=40, num_heads=40, ffn_dim=13824, in_dim=36,
      model_type=i2v. Architecture otherwise identical to Wan21_T2V_14B; the
      extra 20 input channels add a negligible ~0.1M to the patch-embed params.
    """

    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="Wan2.1-I2V-14B",
            params_b=14.288,
            vae_compress=(4, 8, 8),
            patch_size=2,
            te_params_b=5.68,
            layers=40,
            hidden_dim=5120,
            heads=40,
            default_steps=50,
            supported_tasks=["i2v"],
            vae_params_m=126.9,
            ffn_expansion=13824 / 5120,
        )


# --------------------------------------------------------------------------
# Wan 2.2 loads
# --------------------------------------------------------------------------
@register_load
class Wan22_A14B_MoE(LoadModel):
    """Wan 2.2 A14B Mixture-of-Experts (Alibaba).

    Wan 2.2 introduces a two-expert MoE into the video diffusion process: a
    high-noise expert (early denoising, layout) and a low-noise expert (late
    denoising, detail refinement), switched at an SNR threshold. Each expert
    has ~14B parameters; total is 27B but only 14B are active per denoise step,
    so inference compute and per-step FLOPs are nearly identical to Wan2.1-14B.

    MoE modeling note (IMPORTANT):
      'params_b' is set to 14.0 (the ACTIVE parameter count). The base-class
      'per_step_flops' does not use 'params_b' directly -- it uses
      'hidden_dim', 'layers', 'd_ff' -- and those reflect the single
      active expert (ffn_dim=13824, same as Wan2.1-14B). So per-step FLOPs are
      correctly modeled as the active-expert cost.

      The simulator uses 'params_b' in two distinct ways:
        (1) bytes_moved (per-step weight traffic) reads 'chars.params_b'
            directly = 14B -> CORRECT (only the active expert is streamed per
            denoise step; the other expert stays resident).
        (2) base_memory (resident weight memory for OOM) calls
            'load.memory_footprint()' -> must report 27B because BOTH experts
            must be resident in memory simultaneously.
      To keep (1) correct while fixing (2), this subclass OVERRIDES
      'memory_footprint' so that the resident weight memory uses the full 27B
      total (both experts), while 'params_b' stays 14.0 for per-step traffic.
      See the Wan2.2 README: "Each expert model has about 14B parameters,
      resulting in a total of 27B parameters but only 14B active parameters
      per step."

    Grounding (huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers transformer
    config.json + github.com/Wan-Video/Wan2.2 README):
      attention_head_dim=128, num_attention_heads=40 -> hidden_dim=5120,
      num_layers=40, ffn_dim=13824, patch_size=[1,2,2], text_dim=4096.
      The Wan2.2 README lists both T2V-A14B and I2V-A14B as separate MoE
      model variants sharing the same architecture; supported_tasks covers
      both. VAE: Wan2.2-VAE compress (4, 16, 16), patch=2 -> effective 4x32x32
      (README: "T x H x W compression ratio of 4x16x16 ... with an additional
      patchification layer, the total compression ratio reaches 4x32x32").
      VAE params: ~200M (estimate -- Wan2.2-VAE not separately downloadable via
      the gated A14B repo; estimated from the higher-compression Wan2.2-VAE
      structure relative to the 127M Wan2.1-VAE).
    """

    # Total resident parameters (both experts loaded). params_b (14.0) is the
    # per-step active count used for per-step traffic; this is the full resident
    # count used by the overridden memory_footprint for OOM planning.
    _MOE_TOTAL_PARAMS_B: float = 27.0

    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="Wan2.2-A14B-MoE",
            params_b=14.0,
            vae_compress=(4, 16, 16),
            patch_size=2,
            te_params_b=5.68,
            layers=40,
            hidden_dim=5120,
            heads=40,
            default_steps=50,
            supported_tasks=["t2v", "i2v"],
            vae_params_m=200.0,
            ffn_expansion=13824 / 5120,
        )

    def memory_footprint(self, precision: str, tokens: int) -> dict[str, float]:
        """Resident memory footprint: weights use the FULL 27B (both experts).

        The base-class implementation uses 'params_b' (14B active) for weights,
        which underestimates resident memory because both MoE experts must be
        loaded simultaneously. This override uses the total 27B for the weight
        component while keeping KV and activation memory unchanged (those scale
        with the active computation, not the total expert count).

        The simulator's per-step 'bytes_moved' reads 'chars.params_b' directly
        (14B, correct for one active expert per step) and does NOT call this
        method, so per-step traffic remains accurate.
        """
        c = self.characteristics()
        bpe = bytes_per_element(precision)
        weights = self._MOE_TOTAL_PARAMS_B * 1e9 * bpe / GB
        kv = tokens * c.hidden_dim * c.layers * 2 * bpe / GB
        activations = tokens * c.hidden_dim * c.layers * bpe / GB
        total = weights + kv + activations
        return {
            "weights": weights,
            "kv": kv,
            "activations": activations,
            "total_gb": total,
        }


@register_load
class Wan22_TI2V_5B_Dense(LoadModel):
    """Wan 2.2 TI2V-5B dense (Alibaba).

    A 5B dense model using the high-compression Wan2.2-VAE (4x16x16 + patchify =
    4x32x32 effective), supporting both text-to-video and image-to-video in a
    single unified framework. It is one of the fastest 720P@24fps models and can
    run on consumer-grade GPUs (RTX 4090, 24 GB) with offload: the Wan2.2 README
    reports a 5-second 720P video in under 9 minutes on a single consumer GPU.

    Grounding (huggingface.co/Wan-AI/Wan2.2-TI2V-5B config.json):
      dim=3072, num_layers=30, num_heads=24, ffn_dim=14336
      (ffn_expansion=14336/3072=4.667=14/3), in_dim=48, model_type=ti2v.
      VAE: Wan2.2-VAE (4, 16, 16), patch=2 (same as A14B MoE).
      VAE params: ~200M (estimate, same rationale as Wan22_A14B_MoE).
    """

    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="Wan2.2-TI2V-5B",
            params_b=5.0,
            vae_compress=(4, 16, 16),
            patch_size=2,
            te_params_b=5.68,
            layers=30,
            hidden_dim=3072,
            heads=24,
            default_steps=50,
            supported_tasks=["t2v", "i2v"],
            vae_params_m=200.0,
            ffn_expansion=14336 / 3072,
        )


# --------------------------------------------------------------------------
# HunyuanVideo
# --------------------------------------------------------------------------
@register_load
class HunyuanVideo_13B(LoadModel):
    """HunyuanVideo 13B (Tencent).

    A 13B "dual-stream to single-stream" video DiT with full 3D attention. The
    dual-stream phase processes video and text tokens independently; the
    single-stream phase concatenates them for multimodal fusion. Uses an MLLM
    (decoder-only LLM) as the text encoder rather than T5, plus a bidirectional
    token refiner.

    Architecture note: the model has 20 dual-stream transformer blocks + 40
    single-stream blocks = 60 total blocks. The 'layers' field is set to 60.
    The per-step FLOP model treats each block as one attention + one FFN pass
    over the video tokens; the dual-stream blocks' extra text-stream FFN acts on
    only ~256 text tokens (negligible vs ~115K video tokens at 720p/129f), so
    this is a close approximation.

    VAE compression note: the HunyuanVideo model card states "compression
    ratios of video length, space, and channel to 4, 8, and 16 respectively."
    The report table's "(4x8x16)" therefore means (temporal=4, spatial=8,
    latent_channels=16) -- the "16" is the channel count, NOT a width
    compression. The actual spatial VAE compression is 8x8 (symmetric), so
    vae_compress=(4, 8, 8). This is confirmed by the token count: 720p/129f
    with (4,8,8) + patch 2 yields ~115K tokens, matching the report's cited
    figure exactly. (4,8,16) would not reproduce this count for any integer
    patch size.

    Grounding (huggingface.co/tencent/HunyuanVideo: vae/config.json, README,
    checkpoint sizes; diffusers transformer_hunyuan_video.py):
      Transformer: num_attention_heads=24, attention_head_dim=128 ->
      hidden_dim=3072, num_layers=20 (dual), num_single_layers=40 (single),
      mlp_ratio=4.0 -> d_ff=12288, patch_size=2 (spatial), patch_size_t=1.
      fp8 checkpoint = 13.19 GB (1 byte/param) -> ~13B params confirmed.
      VAE: AutoencoderKLCausal3D, block_out_channels=[128,256,512,512],
      time_compression_ratio=4, latent_channels=16, spatial compression 8.
      VAE checkpoint = 986 MB fp32 -> ~246.5M params.
      Text encoder: MLLM (decoder-only, LLaMA-based, ~8B params, 4096-dim
      features). The prompt-rewrite model is a separate Hunyuan-Large.
      Latency anchor: 1280x720/129f/50-step = 1904s (1 GPU) -> 337s (8 GPU).
    """

    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="HunyuanVideo-13B",
            params_b=13.0,
            vae_compress=(4, 8, 8),
            patch_size=2,
            te_params_b=8.0,
            layers=60,
            hidden_dim=3072,
            heads=24,
            default_steps=50,
            supported_tasks=["t2v"],
            vae_params_m=246.5,
            ffn_expansion=4.0,
        )


# --------------------------------------------------------------------------
# CogVideoX
# --------------------------------------------------------------------------
@register_load
class CogVideoX_5B(LoadModel):
    """CogVideoX 5B (Zhipu/THUDM).

    A 5B "Expert Transformer" video DiT with 3D RoPE and a 3D causal VAE.
    Notable for its 226-token text cap (max_text_seq_length=226 in the config;
    the report references this as "~224 token cap"), which limits the text
    conditioning sequence to control memory/compute. TeaCache speedups of
    1.3-2.1x are reported on CogVideoX1.5.

    Grounding (huggingface.co/THUDM/CogVideoX-5b transformer/config.json):
      num_attention_heads=48, attention_head_dim=64 -> hidden_dim=3072,
      num_layers=42, patch_size=2, in_channels=16, text_embed_dim=4096
      (T5-XXL), max_text_seq_length=226, temporal_compression_ratio=4,
      activation_fn=gelu-approximate, use_rotary_positional_embeddings=true.
      FFN: standard FeedForward with default mult=4.0 -> d_ff=12288
      (param-count check: (4*3072^2 + 2*3072*12288)*42 ~ 4.75B + embeds ~ 5B).
      VAE: 3D causal, compress (4, 8, 8) (temporal 4, spatial 8x8).
      VAE params: ~150M (estimate -- from the AutoencoderKLCogVideoX 3D causal
      structure with moderate channel counts; not separately verified by file
      size).
    """

    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="CogVideoX-5B",
            params_b=5.0,
            vae_compress=(4, 8, 8),
            patch_size=2,
            te_params_b=4.76,
            layers=42,
            hidden_dim=3072,
            heads=48,
            default_steps=50,
            supported_tasks=["t2v", "i2v"],
            vae_params_m=150.0,
            ffn_expansion=4.0,
        )


# --------------------------------------------------------------------------
# Open-Sora 2.0
# --------------------------------------------------------------------------
@register_load
class OpenSora2_11B(LoadModel):
    """Open-Sora 2.0 11B (HPC-AI Tech / ColossalAI).

    An 11B MMDiT (dual-stream + single-stream) initialized from FLUX, using a
    Video DC-AE (Deep Compression AutoEncoder) with 4x32x32 compression --
    the highest compression in this library alongside LTX. The high compression
    cuts 768p/5s tokens from ~76K to ~19K (4x), yielding 5.2x training throughput
    and 10x+ inference speedup per the Open-Sora 2.0 paper (arXiv 2503.09642).
    Open-Sora 2.0 is also the only model with a fully public training cost
    breakdown: 4160 H200-days / $199.6k.

    Architecture note: Open-Sora 2.0 is explicitly FLUX-initialized (arXiv
    2503.09642). The layer count (19 dual-stream + 38 single-stream = 57 total)
    is estimated from the FLUX-dev base architecture, which produces ~12B params
    at hidden_dim=3072 with 4x FFN expansion; Open-Sora 2.0 is reported as 11B
    (the small difference arises from video-specific embeddings / VAE latent
    channels). This is the one model whose exact layer count could not be
    fetched directly (the HuggingFace repo is gated and the GitHub config path
    was not locatable); it is marked as an estimate.

    Grounding (arXiv 2503.09642, training report cross-references):
      params_b=11.0 (paper). VAE: Video DC-AE (4, 32, 32), patch_size=1
      (token-count validation: 768p/128f with (4,32,32) + patch 1 -> ~18.4K
      tokens, matching the paper's ~19K figure). Text encoder: T5-XXL
      (~4.76B) + CLIP-Large (~0.4B, pooled features); te_params_b reflects the
      token-level T5-XXL. hidden_dim=3072 (24 heads x 128, FLUX base),
      ffn_expansion=4.0 (d_ff=12288, FLUX FeedForward default).
      VAE params: ~100M (estimate -- DC-AE is architecturally efficient;
      not separately verified).
    """

    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="Open-Sora-2.0",
            params_b=11.0,
            vae_compress=(4, 32, 32),
            patch_size=1,
            te_params_b=4.76,
            layers=57,
            hidden_dim=3072,
            heads=24,
            default_steps=50,
            supported_tasks=["t2v", "i2v"],
            vae_params_m=100.0,
            ffn_expansion=4.0,
        )


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def list_all_loads() -> dict[str, LoadModel]:
    """Return a mapping of registered load name -> LoadModel instance.

    Instantiates every load class currently registered in the VDG registry
    under the 'load' kind. Useful for CLI listing, batch simulation sweeps,
    and governance-agent enumeration of candidate loads.
    """
    result: dict[str, LoadModel] = {}
    for name, cls in REGISTRY.all("load").items():
        instance = cls()
        result[name] = instance
    return result


def recommended_model_for(device_spec: DeviceSpec) -> LoadModel:
    """Pick a recommended load for a device based on its memory and category.

    Selection logic (grounded in the edge-deployment report's decision tree):

      * edge_npu  -> LTX_2_3 (compact 2B, high-compression VAE = few tokens =
        bandwidth-friendly; use distilled ~4-step for realtime. Jetson Thor
        273 GB/s bandwidth is the bottleneck, so low-token-count models win.)
      * apple_silicon -> LTX_2_3 for <=64 GB (bandwidth-limited, compact wins);
        Wan21_T2V_14B for >=128 GB unified memory (can resident 14B bf16).
      * consumer_nv -> LTX_2_3 for ~24 GB (fits comfortably, realtime-capable);
        Wan21_T2V_14B for >=48 GB (14B with FP8/GGUF fits, 720p capable).
      * datacenter -> Wan22_A14B_MoE (highest-capacity MoE for quality).
      * fallback   -> LTX_2_3 (the hero load, broadly deployable).

    The caller can override steps (e.g. set 4 for distilled LTX on NPU) via the
    simulator config; this function only selects the model architecture.
    """
    category = device_spec.category
    mem = device_spec.memory_gb

    if category == DeviceCategory.EDGE_NPU:
        return LTX_2_3()

    if category == DeviceCategory.APPLE_SILICON:
        if mem >= 128:
            return Wan21_T2V_14B()
        return LTX_2_3()

    if category == DeviceCategory.CONSUMER_NV:
        if mem >= 48:
            return Wan21_T2V_14B()
        return LTX_2_3()

    if category == DeviceCategory.DATACENTER:
        return Wan22_A14B_MoE()

    # Fallback for unknown categories.
    return LTX_2_3()
