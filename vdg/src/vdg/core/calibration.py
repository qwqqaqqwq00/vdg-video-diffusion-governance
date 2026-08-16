"""Anchor-based simulation calibration for VDG.

This module resolves INTEGRATION_REPORT.md known limitation #1 -- the simulator
is an engineering-grade approximation (roofline + TDP energy), not bit-exact.
Where a REAL measured latency exists for the same (device, load, resolution)
triple, the roofline prediction can be corrected by a multiplicative scale:

    calibration_scale = measured_latency_s / predicted_latency_s

The scale is computed at the ANCHOR's own operating point (anchor resolution /
frames / steps, so measured and predicted describe the same workload), then
applied to the current simulation's latency, energy and phase breakdown. This
preserves the relative shape of the model (skills still compose the same way)
while removing the systematic per-(device, load) bias. Energy scales linearly
with latency under the TDP model (energy = power x time), so the same factor
is applied to both.

Anchors are REAL measured data points with one-handed citations (the VDG
grounding reports + upstream READMEs). ``ANCHORS`` is a plain tuple so users can
extend it with their own measured points; ``find_anchor`` does name-normalized
matching (registry names and display names both work) and prefers a "latency"
anchor whose pixel count is within 20% of the queried resolution.

Multi-GPU and skill-level anchors (kind "speedup" / "memory") are included as
documented data points for the calibration REPORT but never calibrate a
single-device latency prediction -- a 8-GPU measurement must not silently
correct a single-GPU simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .contracts import DeviceProfile, LoadModel
from .scenario import Scenario
from .simulator import PerformanceEnergySimulator, SimulationResult

__all__ = [
    "CalibrationAnchor",
    "CalibrationRow",
    "CalibrationReport",
    "CalibratedSimulator",
    "ANCHORS",
    "find_anchor",
]

# Resolution match tolerance: an anchor matches a queried resolution when the
# pixel counts (width x height) are within this relative error.
RESOLUTION_TOLERANCE: float = 0.20

# Maximum source-line width in the rendered calibration table (keeps tables
# readable without external dependencies).
_TABLE_WIDTH: int = 96


# --------------------------------------------------------------------------
# Anchor data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationAnchor:
    """A real measured latency/energy data point for a (device, load) pair.

    ``measured_latency_s`` is the measured END-TO-END generation latency at the
    anchor's own ``resolution`` / ``frames`` / ``steps`` (not per-iteration;
    per-iteration figures are stated in ``notes`` for provenance). Fields that
    are genuinely unknown for a benchmark (e.g. the TeaCache Open-Sora-Plan
    eval does not publish its GPU) stay ``None`` -- we never fabricate
    hardware. ``kind`` classifies the anchor:

    * "latency" -- absolute latency measurement; calibrates the simulator.
    * "speedup" -- relative (with/without a skill) measurement; report only.
    * "memory"  -- peak-memory measurement; report only (e.g. VAE tiling).
    """

    device_name: str
    load_name: str
    resolution: tuple[int, int] | None = None  # (width, height)
    frames: int | None = None
    steps: int | None = None
    measured_latency_s: float | None = None
    measured_energy_j: float | None = None
    source: str = ""  # URL or VDG grounding-report reference
    notes: str = ""
    kind: str = "latency"  # "latency" | "speedup" | "memory"
    task: str | None = None  # "t2v" / "i2v" used to match the measurement


ANCHORS: tuple[CalibrationAnchor, ...] = (
    # -- MLX Wan2.1-T2V-1.3B on M4 Max (Apple Silicon) --------------------
    # 480p, 81 frames, ~90 s/it measured (MLX Wan2.1 README via the VDG
    # edge-deployment report section A.5). The README benchmark runs 50 steps:
    # 90 s/it x 50 = 4500 s ~ 75 min, matching the report's "~75 min" column.
    CalibrationAnchor(
        device_name="M4_Max",
        load_name="Wan21_T2V_1_3B",
        resolution=(832, 480),
        frames=81,
        steps=50,
        measured_latency_s=4500.0,  # ~90 s/it x 50 steps (per DiT step)
        measured_energy_j=None,
        source="https://github.com/ml-explore/mlx-examples/blob/main/video/wan2.1/README.md "
               "(via edge-video-dit-deployment-2026.md A.5)",
        notes="MLX native (Metal) Wan2.1-T2V-1.3B on M4 Max, 81 frames 480p. "
              "~90 s/it measured; 50-step estimate 4500 s (~75 min). Per-DiT-step "
              "figure, VAE decode excluded (small at 480p/81f vs 4500 s).",
        kind="latency",
        task="t2v",
    ),
    # -- LightX2V Wan2.1-I2V-14B-480P, 40 steps, 81 frames -----------------
    # Measured benchmark table (2025-12-01) via the VDG edge-deployment report
    # section B.6. 20.26 s/it x 40 = 810.4 s ~ 13.5 min (report column matches).
    CalibrationAnchor(
        device_name="RTX4090",
        load_name="Wan21_I2V_14B",
        resolution=(832, 480),
        frames=81,
        steps=40,
        measured_latency_s=810.4,  # 20.26 s/it x 40 steps (cfg)
        measured_energy_j=None,
        source="https://github.com/ModelTC/LightX2V "
               "(via edge-video-dit-deployment-2026.md B.6)",
        notes="LightX2V Wan2.1-I2V-14B-480P, 40 steps, 81 frames, RTX 4090D x1, "
              "cfg. 20.26 s/it; Diffusers reference 30.5 s/it; xDiT/SGL OOM.",
        kind="latency",
        task="i2v",
    ),
    CalibrationAnchor(
        device_name="H100",
        load_name="Wan21_I2V_14B",
        resolution=(832, 480),
        frames=81,
        steps=40,
        measured_latency_s=207.2,  # 5.18 s/it x 40 steps (cfg)
        measured_energy_j=None,
        source="https://github.com/ModelTC/LightX2V "
               "(via edge-video-dit-deployment-2026.md B.6)",
        notes="LightX2V Wan2.1-I2V-14B-480P, 40 steps, 81 frames, H100 x1, cfg. "
              "5.18 s/it; Diffusers reference 9.77 s/it.",
        kind="latency",
        task="i2v",
    ),
    CalibrationAnchor(
        device_name="H100_x8",
        load_name="Wan21_I2V_14B",
        resolution=(832, 480),
        frames=81,
        steps=40,
        measured_latency_s=30.0,  # 0.75 s/it x 40 steps (cfg, multi-GPU)
        measured_energy_j=None,
        source="https://github.com/ModelTC/LightX2V "
               "(via edge-video-dit-deployment-2026.md B.6)",
        notes="LightX2V Wan2.1-I2V-14B-480P, 40 steps, 81 frames, H100 x8. "
              "0.75 s/it (no-cfg 0.39; fp8 0.35). MULTI-GPU measurement: never "
              "matched against a single-device simulation (device token H100_x8 "
              "is not a registered device).",
        kind="latency",
        task="i2v",
    ),
    # -- HunyuanVideo 720p / 129 frames / 50 steps (official README) -------
    # "Latency (Sec) for 1280x720 (129 frames 50 steps)": 1904.08 s (1 GPU),
    # 934.09 (2), 514.08 (4), 337.58 (8). Tested on a single 80G GPU (H100-class).
    CalibrationAnchor(
        device_name="H100",
        load_name="HunyuanVideo_13B",
        resolution=(1280, 720),
        frames=129,
        steps=50,
        measured_latency_s=1904.08,
        measured_energy_j=None,
        source="https://github.com/Tencent-Hunyuan/HunyuanVideo "
               "(README 'Latency (Sec) for 1280x720 (129 frames 50 steps)' table)",
        notes="Official HunyuanVideo README benchmark, 1280x720, 129 frames, "
              "50 steps, 1 GPU (80G H100-class): 1904.08 s. 2/4/8 GPU: 934.09 / "
              "514.08 / 337.58 s. Device attribution: single 80G GPU per README.",
        kind="latency",
        task="t2v",
    ),
    CalibrationAnchor(
        device_name="H100_x8",
        load_name="HunyuanVideo_13B",
        resolution=(1280, 720),
        frames=129,
        steps=50,
        measured_latency_s=337.58,
        measured_energy_j=None,
        source="https://github.com/Tencent-Hunyuan/HunyuanVideo "
               "(README 'Latency (Sec) for 1280x720 (129 frames 50 steps)' table)",
        notes="Official HunyuanVideo README benchmark via xDiT/USP parallel "
              "inference, 8 GPUs: 337.58 s (5.64x vs 1 GPU). MULTI-GPU "
              "measurement: never matched against a single-device simulation.",
        kind="latency",
        task="t2v",
    ),
    # -- TeaCache Open-Sora-Plan: 99.65 s -> 22.62 s (4.41x) ---------------
    # Skill-level SPEEDUP anchor. The TeaCache repo does not publish the eval
    # GPU / resolution, so those fields are None -- the acceleration report
    # (section 2 / 8.4) records the speedup with VBench -0.07% at 4.41x.
    CalibrationAnchor(
        device_name="TeaCache-eval",
        load_name="Open-Sora-Plan",
        resolution=None,
        frames=None,
        steps=None,
        measured_latency_s=22.62,  # cached generation; baseline 99.65 s
        measured_energy_j=None,
        source="https://github.com/LiewFeng/TeaCache "
               "(via video-dit-inference-acceleration-report.md 2 / 8.4)",
        notes="TeaCache on Open-Sora-Plan: 99.65 s -> 22.62 s = 4.41x measured "
              "(up to 4.91x), VBench -0.07% at 4.41x (negligible). Eval GPU and "
              "resolution not published -> device/resolution fields left None; "
              "this is a skill-impact anchor, not a device-latency calibration.",
        kind="speedup",
        task=None,
    ),
    # -- VAE temporal tiling, HunyuanVideo: 32 GB -> 8 GB (4x) -------------
    # Memory anchor from ComfyUI v0.3.10 (VAEDecode Tiled, tile_size=128,
    # temporal_size=32). Reports VAE decode peak memory, not latency.
    CalibrationAnchor(
        device_name="H100",
        load_name="HunyuanVideo_13B",
        resolution=(1280, 720),
        frames=129,
        steps=1,  # VAE decode phase only; steps irrelevant to decode peak
        measured_latency_s=None,
        measured_energy_j=None,
        source="https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/ "
               "(via video-dit-inference-acceleration-report.md 4.2)",
        notes="ComfyUI v0.3.10 HunyuanVideo VAE temporal tiling: 32 GB -> 8 GB "
              "(4x) at tile_size=128, overlap=32, temporal_size=32, "
              "temporal_overlap=4. Peak-memory anchor (kind=memory): no latency "
              "measured, so it never calibrates latency; listed for the report.",
        kind="memory",
        task=None,
    ),
    # -- REAL END-TO-END MEASUREMENT: LTX-2.3 on M4 Max via ComfyUI ---------
    # Measured 2026-08-17 on M4 Max (MPS, --fp16-unet) via ComfyUI API.
    # Model: DasiwaLTX23_goldenLaceV3.gguf (22 GB Q8, ~19B params).
    # VAE: LTX23_video_vae_bf16. Text encoder: Gemma-3-12B fp8.
    # Warm run (model cached): 10-step sampling + VAE decode = 111.2 s.
    # Cold run (full load): 155.5 s (44.3 s model load overhead).
    # 1-step warm run: 24.0 s -> per-step = (111.2-24.0)/(10-1) = 9.69 s/step.
    # Output verified non-black (all 41 frames, mean pixel ~118, std ~28).
    # Q8 GGUF dequantizes to fp32 internally, avoiding the bf16 GELU kernel
    # NaN that --fp16-unet triggers on safetensors bf16 checkpoints.
    CalibrationAnchor(
        device_name="M4_Max",
        load_name="LTX_2_3",
        resolution=(768, 512),
        frames=41,
        steps=10,
        measured_latency_s=111.2,  # warm: 10 steps + VAE decode, model cached
        measured_energy_j=None,
        source="VDG end-to-end ComfyUI benchmark, 2026-08-17 "
               "(scripts/comfyui_ltx_benchmark.py)",
        notes="REAL MEASUREMENT via ComfyUI API on M4 Max MPS. Q8 GGUF 19B, "
              "768x512, 41f, 10 steps, euler, cfg=3.0. Warm run (model cached) "
              "111.2 s; cold (full load) 155.5 s. Per-step 9.69 s (derived from "
              "1-step vs 10-step differential). Non-black output verified. "
              "Q8 dequant avoids bf16 GELU NaN (--fp16-unet safe with GGUF).",
        kind="latency",
        task="t2v",
    ),
    # Per-step anchor (derived): 9.69 s/step for LTX-2.3 Q8 on M4 Max.
    CalibrationAnchor(
        device_name="M4_Max",
        load_name="LTX_2_3",
        resolution=(768, 512),
        frames=41,
        steps=1,
        measured_latency_s=9.69,  # per-step (differential: (111.2-24.0)/9)
        measured_energy_j=None,
        source="VDG end-to-end ComfyUI benchmark, 2026-08-17 (derived)",
        notes="Per-step latency derived from 10-step (111.2 s) vs 1-step "
              "(24.0 s) differential: (111.2-24.0)/(10-1) = 9.69 s/step. "
              "Excludes VAE decode + text encode overhead (~14.3 s).",
        kind="latency",
        task="t2v",
    ),
)


# --------------------------------------------------------------------------
# Matching helpers
# --------------------------------------------------------------------------
def _norm(name: str) -> str:
    """Normalize a device/load name for case/separator-insensitive matching.

    "M4_Max" and "M4 Max" both normalize to "m4max"; "Wan21_T2V_1_3B" and
    "Wan2.1-T2V-1.3B" both normalize to "wan21t2v13b".
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolution_pixel_error(query: tuple[int, int], anchor: tuple[int, int]) -> float:
    """Relative pixel-count error between two (width, height) resolutions."""
    q_px = max(1, int(query[0]) * int(query[1]))
    a_px = max(1, int(anchor[0]) * int(anchor[1]))
    return abs(q_px - a_px) / a_px


def find_anchor(
    device_name: str,
    load_name: str,
    resolution: tuple[int, int],
) -> CalibrationAnchor | None:
    """Return the best-matching LATENCY anchor, or None.

    Matching rules:
    * device + load must match after name normalization (registry names and
      display names both work);
    * only ``kind == "latency"`` anchors with a measured latency are considered
      (a speedup/memory anchor must never correct an absolute prediction);
    * the anchor resolution must be within ``RESOLUTION_TOLERANCE`` (20%)
      pixel-count relative error of the queried resolution;
    * among several candidates the closest pixel count wins.
    """
    dn, ln = _norm(device_name), _norm(load_name)
    candidates = [
        a for a in ANCHORS
        if a.kind == "latency"
        and a.measured_latency_s is not None
        and a.resolution is not None
        and _norm(a.device_name) == dn
        and _norm(a.load_name) == ln
    ]
    best: CalibrationAnchor | None = None
    best_err = float("inf")
    for a in candidates:
        err = _resolution_pixel_error(resolution, a.resolution)  # type: ignore[arg-type]
        if err <= RESOLUTION_TOLERANCE and err < best_err:
            best, best_err = a, err
    return best


def _default_attention_backend(device: DeviceProfile) -> str:
    """Mirror the governance pipeline's backend preference for report runs."""
    backends = device.spec().attention_backends
    if not backends:
        return "math"
    for preferred in ("flash", "sdpa", "mlx_sdpa", "sage2", "triton"):
        if preferred in backends:
            return preferred
    return backends[0]


def _anchor_run_config(
    anchor: CalibrationAnchor,
    base_config: dict[str, Any],
) -> dict[str, Any]:
    """Config that re-runs the base simulator AT the anchor's operating point.

    Resolution/frames/steps/task are forced to the anchor's measured values so
    measured and predicted describe the same workload; precision / attention
    backend / utilization carry over from the current simulation.
    """
    cfg = dict(base_config)
    cfg["resolution"] = tuple(anchor.resolution)  # type: ignore[arg-type]
    cfg["frames"] = int(anchor.frames)  # type: ignore[arg-type]
    cfg["steps"] = int(anchor.steps)  # type: ignore[arg-type]
    if anchor.task:
        cfg["task"] = anchor.task
    return cfg


# --------------------------------------------------------------------------
# Calibrated simulator
# --------------------------------------------------------------------------
class CalibratedSimulator(PerformanceEnergySimulator):
    """Drop-in ``PerformanceEnergySimulator`` with anchor-based calibration.

    ``simulate`` first runs the base roofline simulator, then -- when a
    CalibrationAnchor matches the (device, load, resolution) triple -- rescales
    latency, energy and the phase breakdown by

        anchor_scale = anchor.measured_latency_s / predicted_at_anchor

    where ``predicted_at_anchor`` is the base prediction re-run at the anchor's
    own operating point (resolution / frames / steps). The optional
    ``calibration_scale`` constructor knob is a manual global multiplier
    applied on top of any anchor scale (default 1.0 = no manual override).

    No matching anchor -> scale stays 1.0 and the result carries the warning
    "no anchor, engineering estimate". Skill composition is unaffected: the
    scale corrects the systematic (device, load) bias, and skills still compose
    multiplicatively on top. Energy scales with latency (TDP: E = P x t).
    """

    def __init__(
        self,
        calibration_scale: float = 1.0,
        base: PerformanceEnergySimulator | None = None,
    ) -> None:
        self.calibration_scale = float(calibration_scale)
        self.base = base or PerformanceEnergySimulator()
        # Keep the energy model visible so this class is a transparent drop-in
        # for code that reads simulator.energy_model.
        super().__init__(energy_model=self.base.energy_model)
        # Pre-calibration result of the most recent ``simulate`` call (same
        # config/backend the wrapper actually used) -- lets callers show the
        # uncalibrated prediction and the applied scale consistently.
        self.last_base_result: SimulationResult | None = None

    def simulate(
        self,
        device: DeviceProfile,
        load: LoadModel,
        skills_applied: list[Any] | None = None,
        config: dict[str, Any] | None = None,
        scenario: Scenario | None = None,
    ) -> SimulationResult:
        config = dict(config or {})
        # The anchors were measured with the kernel the benchmark actually
        # shipped (MLX fast SDPA, FlashAttention...), so unless the caller
        # explicitly chose a backend, use the device's best available one for
        # BOTH the current run and the anchor re-run -- inheriting the base
        # simulator's accidental ``math`` default (0.5x peak) would distort the
        # scale. The base simulator's own default is untouched for other
        # callers.
        if "attention_backend" not in config:
            config["attention_backend"] = _default_attention_backend(device)
        base_result = self.base.simulate(
            device, load,
            skills_applied=skills_applied,
            config=config,
            scenario=scenario,
        )
        self.last_base_result = base_result

        # Workload resolution used for anchor matching (same rules as the base
        # simulator: scenario wins, config keys are the fallback).
        if scenario is not None:
            resolution = tuple(scenario.resolution)
        else:
            resolution = tuple(config.get("resolution", (854, 480)))

        anchor = find_anchor(device.registry_name(), load.registry_name(), resolution)
        if anchor is None:
            return self._apply_scale(
                base_result,
                self.calibration_scale,
                warning=(
                    "no anchor, engineering estimate"
                    if self.calibration_scale == 1.0
                    else "no anchor; applied manual calibration_scale "
                         + format(self.calibration_scale, ".4f")
                ),
            )

        # Re-run the base model at the anchor's operating point so measured and
        # predicted describe the same workload before computing the scale.
        anchor_config = _anchor_run_config(anchor, config)
        predicted_at_anchor = self.base.simulate(
            device, load,
            skills_applied=[],  # anchors are unskilled baseline measurements
            config=anchor_config,
            scenario=None,
        )
        anchor_latency = float(anchor.measured_latency_s)  # type: ignore[arg-type]
        if anchor_latency <= 0 or predicted_at_anchor.latency_s <= 0:
            return self._apply_scale(
                base_result,
                self.calibration_scale,
                warning=(
                    "calibration anchor " + anchor.device_name + "/" + anchor.load_name
                    + " unusable (non-positive measured or predicted latency); "
                    "engineering estimate"
                ),
            )

        anchor_scale = anchor_latency / predicted_at_anchor.latency_s
        scale = self.calibration_scale * anchor_scale
        warning = (
            "calibration: anchor " + anchor.device_name + "/" + anchor.load_name
            + " @ " + str(anchor.resolution) + " " + str(anchor.frames) + "f "
            + str(anchor.steps) + " steps (kind=" + anchor.kind + "): scale "
            + format(scale, ".4f") + " (measured " + format(anchor_latency, ".2f")
            + "s vs predicted " + format(predicted_at_anchor.latency_s, ".2f")
            + "s at anchor operating point; manual " 
            + format(self.calibration_scale, ".4f") + ")"
        )
        return self._apply_scale(base_result, scale, warning=warning)

    def _apply_scale(
        self,
        result: SimulationResult,
        scale: float,
        warning: str,
    ) -> SimulationResult:
        """Return ``result`` with latency/energy/breakdown scaled by ``scale``."""
        scale = max(scale, 1e-6)
        latency = result.latency_s * scale
        warnings = list(result.warnings) + [warning]
        return replace(
            result,
            latency_s=latency,
            energy_j=result.energy_j * scale,
            throughput_tokens_s=(result.tokens / latency) if latency > 0 else 0.0,
            breakdown={k: v * scale for k, v in result.breakdown.items()},
            warnings=warnings,
        )


# --------------------------------------------------------------------------
# Calibration report
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationRow:
    """One predicted-vs-measured comparison for an anchor."""

    anchor: CalibrationAnchor
    predicted_latency_s: float | None  # None when the anchor has no latency
    relative_error_pct: float | None  # (predicted - measured) / measured * 100


class CalibrationReport:
    """Table of predicted vs measured for every anchor of a (device, load).

    Rows cover ALL anchors matching the pair (latency, speedup and memory
    kinds). Anchors without a measured latency render as N/A; latency anchors
    additionally carry the relative error of the roofline prediction against
    the real measurement -- the number the calibrated simulator removes.
    """

    def __init__(self, rows: list[CalibrationRow]) -> None:
        self.rows = list(rows)

    @classmethod
    def compare(
        cls,
        device: DeviceProfile,
        load: LoadModel,
        scenario: Scenario | None,
        base: PerformanceEnergySimulator | None = None,
    ) -> "CalibrationReport":
        """Build the predicted-vs-measured table for ``device`` + ``load``.

        ``scenario`` is reserved for future reference-config derivation; the
        predictions are made at each anchor's own resolution / frames / steps
        via the base (uncalibrated) simulator with the device's best attention
        backend, so the scenario's step count is deliberately NOT used here.
        """
        sim = base or PerformanceEnergySimulator()
        base_config = {
            "precision": "bf16",
            "attention_backend": _default_attention_backend(device),
            "utilization": 0.75,
        }
        dn, ln = _norm(device.registry_name()), _norm(load.registry_name())
        rows: list[CalibrationRow] = []
        for anchor in ANCHORS:
            if _norm(anchor.device_name) != dn or _norm(anchor.load_name) != ln:
                continue
            if (
                anchor.measured_latency_s is None
                or anchor.resolution is None
                or anchor.frames is None
                or anchor.steps is None
            ):
                rows.append(CalibrationRow(anchor, None, None))
                continue
            cfg = _anchor_run_config(anchor, base_config)
            predicted = sim.simulate(
                device, load, skills_applied=[], config=cfg, scenario=None,
            ).latency_s
            err = (
                (predicted - anchor.measured_latency_s)
                / anchor.measured_latency_s * 100.0
                if anchor.measured_latency_s > 0 else None
            )
            rows.append(CalibrationRow(anchor, predicted, err))
        return cls(rows)

    def render(self) -> str:
        """Render the table as a fixed-width text report (no dependencies)."""
        if not self.rows:
            return "Calibration report: no anchors for this (device, load)."
        header = "Calibration report: predicted (roofline) vs measured (anchors)"
        lines = [header, "=" * _TABLE_WIDTH]
        lines.append(
            "  device/load/resolution           kind     measured(s)  predicted(s)  rel.err"
        )
        lines.append("-" * _TABLE_WIDTH)
        for row in self.rows:
            a = row.anchor
            loc = (
                a.device_name + "/" + a.load_name + " "
                + (str(a.resolution) if a.resolution is not None else "?")
                + " " + (str(a.frames) + "f" if a.frames is not None else "?")
                + " " + (str(a.steps) + "st" if a.steps is not None else "?")
            )
            if row.predicted_latency_s is None or row.relative_error_pct is None:
                measured = (
                    format(a.measured_latency_s, ".2f")
                    if a.measured_latency_s is not None else "n/a"
                )
                lines.append(
                    "  " + loc.ljust(45) + " " + a.kind.ljust(9)
                    + " " + measured.rjust(11) + "  " + "n/a".rjust(11)
                    + "  " + "n/a".rjust(8)
                )
                continue
            lines.append(
                "  " + loc.ljust(45) + " " + a.kind.ljust(9)
                + " " + format(a.measured_latency_s, ".2f").rjust(11)  # type: ignore[arg-type]
                + "  " + format(row.predicted_latency_s, ".2f").rjust(11)
                + "  " + format(row.relative_error_pct, "+.1f").rjust(8) + "%"
            )
        lines.append("-" * _TABLE_WIDTH)
        lines.append(
            "relative error = (predicted - measured) / measured; the calibrated"
        )
        lines.append("simulator applies the inverse scale at the anchor operating point.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()
