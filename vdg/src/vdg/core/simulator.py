"""Performance-energy simulator for heterogeneous video DiT deployment.

``PerformanceEnergySimulator.simulate`` combines:
  * the roofline compute model (``core.roofline``) for per-step / VAE / TE time,
  * an energy model (``core.energy_model``) for joules,
  * skill impacts (``core.contracts.SkillImpact``) composed sub-multiplicatively,
  * attention-backend -> precision peak mapping,
  * a memory model (weights + KV + activations).

Modeling constants are documented inline and in ``CONTRACTS.md``. They are
calibratable estimates grounded in the research reports, not bit-exact numbers;
the acceleration report explicitly flags per-video VAE FLOPs and combination
multipliers as data gaps, so conservative, documented constants are used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    DeviceProfile,
    LoadModel,
    Skill,
    SkillImpact,
)
from .energy_model import EnergyModel, TDPEnergyModel
from .roofline import (
    GB,
    bytes_per_element,
    per_step_flops,
    predict_step_time,
    roofline,
    token_count,
    vae_decode_flops,
    text_encoder_flops,
)
from .scenario import Scenario

__all__ = ["SimulationResult", "PerformanceEnergySimulator", "AgentContext"]


# --------------------------------------------------------------------------
# Attention backend -> precision used for the attention matmuls.
# Grounded in the acceleration report:
#   sage3  -> FP4 tensor cores (Blackwell)         [5x over FA on 5090]
#   sage2  -> INT4 QK + FP8 PV ~ approximated fp8   [3x FA2 on 4090]
#   sage1  -> INT8 QK + FP8/FP16 PV ~ int8          [2-5x over FA]
#   flash / sdpa / triton / mlx_sdpa -> step precision
#   math   -> non-fused math path (MPS default), ~0.5x effective peak
# --------------------------------------------------------------------------
ATTENTION_BACKEND_PRECISION: dict[str, str] = {
    "sage3": "fp4",
    "sage2": "fp8",
    "sage1": "int8",
    "flash": "",       # sentinel: use step precision
    "sdpa": "",
    "triton": "",
    "mlx_sdpa": "",
    "math": "",        # effective peak penalized separately
}

# Quality deltas (VBench-proxy points) for attention backends, grounded in the
# report ("FP4 precision lower; SA2 negligible; STA finetune -0.09").
ATTENTION_BACKEND_QUALITY_DELTA: dict[str, float] = {
    "sage3": -1.5,   # FP4 lower precision
    "sage2": -0.1,   # negligible
    "sage1": -0.2,
    "flash": 0.0,
    "sdpa": 0.0,
    "triton": 0.0,
    "mlx_sdpa": 0.0,
    "math": 0.0,
}

# Non-fused math attention (MPS default) achieves a fraction of peak.
MATH_ATTN_PEAK_RATIO: float = 0.5

# Precision -> VBench-proxy quality delta vs bf16 baseline. Grounded in the
# robustness report (fp8 attention artifacts; fp4/int4 lower; fp8 VAE visible
# artifacts -> VAE kept high precision separately).
PRECISION_QUALITY_DELTA: dict[str, float] = {
    "fp32": 1.0,
    "tf32": 0.5,
    "bf16": 0.0,
    "fp16": -0.5,
    "fp8": -2.0,
    "nvfp4": -4.0,
    "fp4": -5.0,
    "int8": -3.0,
    "int4": -6.0,
}

# Compose multiple skills sub-multiplicatively: product(speedup) ** EXPONENT.
# < 1 models diminishing returns from bottleneck transfer and scope overlap
# (e.g. distillation cuts steps so attention's share drops, reducing the
# relative gain of SageAttention; STA and SA3 both touch attention).
COMBINATION_EXPONENT: float = 0.85

# Misc launch / framework overhead fraction added to base latency.
OVERHEAD_FRACTION: float = 0.05

# CPU fallback peak for text-encoder offload (typical ~50 GFLOPS, ~50 GB/s).
CPU_PEAK_TFLOPS: float = 0.05
CPU_MEM_BW_GBPS: float = 50.0


@dataclass
class SimulationResult:
    """Outcome of a single performance-energy simulation."""

    latency_s: float
    energy_j: float
    peak_memory_gb: float
    quality_score: float
    throughput_tokens_s: float
    breakdown: dict[str, float]
    pareto_tag: str
    warnings: list[str] = field(default_factory=list)
    # Extra provenance for governance agents.
    tokens: int = 0
    steps: int = 0
    precision: str = ""
    attention_backend: str = ""

    def meets_slo(self, scenario: Scenario) -> bool:
        return self.latency_s <= scenario.latency_slo_s

    def meets_budget(self, scenario: Scenario) -> bool:
        return self.energy_j <= scenario.energy_budget_j

    def is_feasible(self, scenario: Scenario) -> bool:
        return self.meets_slo(scenario) and self.meets_budget(scenario)


@dataclass
class AgentContext:
    """Shared context passed to governance agents.

    Holds the inputs and the simulation results accumulated so far so an agent
    can reason over alternatives (e.g. "with TeaCache the SLO is met").
    """

    device: DeviceProfile
    load: LoadModel
    scenario: Scenario
    config: dict[str, Any] = field(default_factory=dict)
    skills_applied: list[Skill] = field(default_factory=list)
    results: list[SimulationResult] = field(default_factory=list)


class PerformanceEnergySimulator:
    """Core simulator combining roofline + energy + skill impacts."""

    def __init__(self, energy_model: EnergyModel | None = None) -> None:
        self.energy_model = energy_model or TDPEnergyModel()

    # -- public API ------------------------------------------------------
    def simulate(
        self,
        device: DeviceProfile,
        load: LoadModel,
        skills_applied: list[Skill] | None = None,
        config: dict[str, Any] | None = None,
        scenario: Scenario | None = None,
    ) -> SimulationResult:
        skills_applied = skills_applied or []
        config = config or {}
        spec = device.spec()
        chars = load.characteristics()
        warnings: list[str] = []

        # --- workload ---------------------------------------------------
        if scenario is not None:
            resolution = scenario.resolution
            frames = scenario.frames
            steps = config.get("steps", scenario.steps)
            task = scenario.task
        else:
            resolution = tuple(config.get("resolution", (854, 480)))
            frames = int(config.get("frames", 81))
            steps = int(config.get("steps", chars.default_steps))
            task = config.get("task", "t2v")
        steps = max(1, int(steps))

        precision = str(config.get("precision", "bf16")).lower()
        if not spec.supports(precision):
            warnings.append(
                "Precision " + precision + " not in supported_precisions for "
                + spec.name + "; simulating anyway."
            )

        attention_backend = str(config.get("attention_backend", "math")).lower()
        if spec.attention_backends and attention_backend not in spec.attention_backends:
            warnings.append(
                "Attention backend " + attention_backend + " not listed for "
                + spec.name + "."
            )

        text_tokens = int(config.get("text_tokens", 256 if task == "t2v" else 0))

        # --- tokens & FLOPs --------------------------------------------
        width, height = resolution[0], resolution[1]
        tokens = token_count(frames, height, width, chars.vae_compress, chars.patch_size)
        flops = per_step_flops(
            tokens,
            hidden_dim=chars.hidden_dim,
            layers=chars.layers,
            d_ff=chars.d_ff,
            heads=chars.heads,
            text_tokens=text_tokens,
        )
        attn_flops = flops["attention"]
        ffn_flops = flops["ffn"]
        total_flops = flops["total"]

        # --- memory traffic per step (weights + KV + activations) -----
        bpe = bytes_per_element(precision)
        weights_bytes = chars.params_b * 1e9 * bpe
        kv_bytes = tokens * chars.hidden_dim * chars.layers * 2 * bpe
        act_bytes = tokens * chars.hidden_dim * chars.layers * bpe
        bytes_moved = weights_bytes + kv_bytes + act_bytes

        # --- effective peaks -------------------------------------------
        # Resolve a peak for the step precision, falling back gracefully if the
        # device lacks that precision in compute_tflops (warn, do not crash).
        if precision in spec.compute_tflops:
            step_peak = spec.peak_flops(precision)
        else:
            fb = "bf16" if "bf16" in spec.compute_tflops else next(iter(spec.compute_tflops))
            step_peak = spec.peak_flops(fb)
            warnings.append(
                "Precision " + precision + " not in compute_tflops for "
                + spec.name + "; falling back to " + fb + "."
            )
        attn_precision = ATTENTION_BACKEND_PRECISION.get(attention_backend, "")
        if attn_precision == "":
            attn_peak = step_peak
        elif attn_precision in spec.compute_tflops:
            attn_peak = spec.peak_flops(attn_precision)
        else:
            attn_peak = step_peak
            warnings.append(
                "Backend " + attention_backend + " needs " + attn_precision
                + " which " + spec.name + " lacks; using step precision."
            )
        if attention_backend == "math":
            attn_peak = attn_peak * MATH_ATTN_PEAK_RATIO
        mem_bw = spec.mem_bw_bytes()

        # --- per-step time (split attention/ffn by their own roofline) -
        attn_ai = attn_flops / bytes_moved if bytes_moved > 0 else float("inf")
        ffn_ai = ffn_flops / bytes_moved if bytes_moved > 0 else float("inf")
        attn_achievable = roofline(attn_ai, attn_peak, mem_bw)
        ffn_achievable = roofline(ffn_ai, step_peak, mem_bw)
        attn_time_per_step = attn_flops / attn_achievable
        ffn_time_per_step = ffn_flops / ffn_achievable
        step_time = attn_time_per_step + ffn_time_per_step
        denoise_time = step_time * steps

        # --- VAE decode (separate roofline, high precision) -----------
        vae_precision = str(config.get("vae_precision", "fp16")).lower()
        vae_bpe = bytes_per_element(vae_precision)
        vae_peak = spec.peak_flops(vae_precision) if vae_precision in spec.compute_tflops \
            else step_peak
        vae_peak = vae_peak * 1.0  # VAE_EFFICIENCY is applied to achievable below
        from .roofline import VAE_EFFICIENCY
        vae_flops = vae_decode_flops(frames, height, width, chars.vae_params_m, chars.vae_compress)
        vae_bytes = chars.vae_params_m * 1e6 * vae_bpe + (frames * height * width * 3) * vae_bpe
        vae_ai = vae_flops / vae_bytes if vae_bytes > 0 else float("inf")
        vae_achievable = roofline(vae_ai, vae_peak, mem_bw) * VAE_EFFICIENCY
        vae_decode_time = vae_flops / vae_achievable if vae_achievable > 0 else 0.0

        # --- text encoder (may be CPU-offloaded) ----------------------
        te_offload = bool(config.get("te_offload", False))
        if te_offload:
            te_peak = CPU_PEAK_TFLOPS * 1e12
            te_bw = CPU_MEM_BW_GBPS * 1e9
            warnings.append("Text encoder offloaded to CPU (slow but fits memory).")
        else:
            te_peak = step_peak
            te_bw = mem_bw
        te_flops = text_encoder_flops(chars.te_params_b, text_tokens)
        te_bytes = chars.te_params_b * 1e9 * bpe + text_tokens * chars.hidden_dim * bpe
        te_time = predict_step_time(te_flops, te_peak, te_bw, te_bytes) if te_flops > 0 else 0.0

        # --- base latency / breakdown ----------------------------------
        base_denoise = denoise_time
        breakdown = {
            "denoise": base_denoise,
            "attention": attn_time_per_step * steps,
            "ffn": ffn_time_per_step * steps,
            "vae_decode": vae_decode_time,
            "te_encode": te_time,
        }
        base_latency = (denoise_time + vae_decode_time + te_time) * (1.0 + OVERHEAD_FRACTION)

        # --- base memory & energy & quality ---------------------------
        base_memory = load.memory_footprint(precision, tokens)["total_gb"]
        utilization = float(config.get("utilization", 0.75))
        base_energy = self.energy_model.energy(spec, base_latency, utilization)

        quality_base = float(config.get("quality_base", 84.0))
        quality_base += PRECISION_QUALITY_DELTA.get(precision, 0.0)
        quality_base += ATTENTION_BACKEND_QUALITY_DELTA.get(attention_backend, 0.0)
        # Fewer steps than the reference slightly lowers quality unless distilled.
        distilled = bool(config.get("distilled", steps <= 8))
        if not distilled and steps < 30:
            quality_base -= 0.3 * (30 - steps)

        # --- skill impacts (sub-multiplicative composition) -----------
        impacts: list[SkillImpact] = []
        for skill in skills_applied:
            if not skill.applicable(device, load):
                warnings.append(
                    "Skill " + skill.registry_name() + " not applicable; skipped."
                )
                continue
            cfg = skill.default_config()
            cfg.update(config.get("skill_configs", {}).get(skill.registry_name(), {}))
            impact = skill.predict(device, load, cfg)
            impacts.append(impact)

        speedup_product = 1.0
        memory_ratio = 1.0
        energy_ratio = 1.0
        quality_delta_sum = 0.0
        for imp in impacts:
            speedup_product *= max(imp.speedup, 1e-6)
            memory_ratio *= imp.memory_ratio
            energy_ratio *= imp.energy_ratio
            quality_delta_sum += imp.quality_delta

        effective_speedup = speedup_product ** COMBINATION_EXPONENT if speedup_product > 0 else 1.0
        latency = base_latency / effective_speedup if effective_speedup > 0 else base_latency
        peak_memory = base_memory * memory_ratio
        energy = (base_energy / effective_speedup) * energy_ratio
        quality = quality_base + quality_delta_sum
        quality = max(0.0, min(100.0, quality))

        # --- OOM warning -----------------------------------------------
        if peak_memory > spec.memory_gb:
            warnings.append(
                "OOM risk: peak memory " + format(peak_memory, ".2f")
                + " GB exceeds device " + format(spec.memory_gb, ".2f") + " GB."
            )

        # --- Pareto tag ------------------------------------------------
        pareto_tag = self._pareto_tag(latency, energy, quality, scenario)

        # --- breakdown after skills (scale proportionally) ------------
        scaled_breakdown = {
            k: v / effective_speedup for k, v in breakdown.items()
        } if effective_speedup > 0 else dict(breakdown)

        throughput = tokens / latency if latency > 0 else 0.0

        return SimulationResult(
            latency_s=latency,
            energy_j=energy,
            peak_memory_gb=peak_memory,
            quality_score=quality,
            throughput_tokens_s=throughput,
            breakdown=scaled_breakdown,
            pareto_tag=pareto_tag,
            warnings=warnings,
            tokens=tokens,
            steps=steps,
            precision=precision,
            attention_backend=attention_backend,
        )

    # -- helpers ---------------------------------------------------------
    def _pareto_tag(
        self,
        latency: float,
        energy: float,
        quality: float,
        scenario: Scenario | None,
    ) -> str:
        if scenario is None:
            if quality >= 83.0:
                return "quality"
            if latency < 60.0:
                return "fast"
            return "balanced"
        slo_ok = latency <= scenario.latency_slo_s
        budget_ok = energy <= scenario.energy_budget_j
        if slo_ok and budget_ok:
            return "efficient" if quality >= scenario.quality_target else "feasible_lowq"
        if slo_ok and not budget_ok:
            return "fast_energy_heavy"
        if not slo_ok and budget_ok:
            return "slow_efficient"
        return "infeasible"
