"""Scenario library for VDG simulations.

A ``Scenario`` fixes a generation workload (task, resolution, frames, fps,
denoise steps, quality target, energy budget, latency SLO). The built-in
library centers on LTX-2.3 (the primary model) and spans the regimes reported in
the grounding research: short 480p clips, 720p, long video, and edge NPU short
clips.

Grounding anchors (edge-deployment / acceleration reports):
  * 480p/81f (5s@16fps) is the standard reported test clip across Wan/Hunyuan/
    LTX (MLX Wan2.1 M4 Max ~90 s/it 1.3B; LightX2V 4090D ~20.26 s/it 40-step).
  * LTX-Video realtime target is 1216x704 @ 30fps; LTX-2 default ~30-step flow
    matching, distilled to ~4-8 steps (lightx2v 4-step).
  * Kijai context-windowing: 1025 frames -> window 81 + overlap 16 on a 5090.
  * Jetson Thor 480p short clip is "multi-minute, bandwidth-limited" (no public
    benchmark -> SLO is qualitative, flagged in the scenario).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Scenario", "ScenarioLibrary", "BUILTIN_SCENARIOS"]


@dataclass(frozen=True)
class Scenario:
    """A fixed generation workload to simulate."""

    name: str
    task: str  # "t2v" or "i2v"
    resolution: tuple[int, int]  # (width, height)
    frames: int
    fps: int
    steps: int
    quality_target: float  # VBench-proxy target, 0-100
    energy_budget_j: float  # joules
    latency_slo_s: float  # seconds
    notes: str = ""

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps > 0 else 0.0


@dataclass
class ScenarioLibrary:
    """A collection of named scenarios with lookup.

    ``aliases`` maps a short alias to a canonical scenario name (e.g.
    ``ltx_t2v_480p`` -> ``ltx_t2v_480p_81f``). Aliases are resolved by ``get``
    but are NOT returned by ``names``/``all`` so the canonical scenario set stays
    stable (a UI/CLI shorthand does not inflate the documented library).
    """

    scenarios: dict[str, Scenario] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    def add(self, scenario: Scenario) -> None:
        self.scenarios[scenario.name] = scenario

    def add_alias(self, alias: str, target: str) -> None:
        """Register a shorthand alias resolving to an existing scenario name."""
        self.aliases[alias] = target

    def get(self, name: str) -> Scenario:
        resolved = self.aliases.get(name, name)
        if resolved not in self.scenarios:
            raise KeyError(
                "Unknown scenario: " + repr(name)
                + ". Available: " + ", ".join(sorted(self.scenarios))
            )
        return self.scenarios[resolved]

    def names(self) -> list[str]:
        return sorted(self.scenarios)

    def all(self) -> list[Scenario]:
        return [self.scenarios[n] for n in self.names()]


def _build_builtins() -> ScenarioLibrary:
    lib = ScenarioLibrary()
    # LTX-2.3 text-to-video, 480p, 81 frames (5s@16fps), ~30-step baseline.
    # SLO/budget grounded in 4090 ~20 s/it * 30 ~ 600s full; 4-step distilled
    # ~1 min. Energy: 450W * ~110s distilled ~ 50kJ budget target.
    # quality_target 82.0 (2 VBench points of headroom under the ~84 bf16
    # baseline) so quality-costing acceleration (TeaCache, distillation) is
    # *feasible* under the policy -- a target equal to the baseline quality
    # would make every accel combo infeasible by construction, defeating the
    # governance demo. NOTE: 82 is a demo/governance calibration (accept up to
    # 2 VBench points of quality loss for acceleration), not a VBench-grounded
    # number; override with --quality-floor for a stricter bar.
    lib.add(Scenario(
        name="ltx_t2v_480p_81f",
        task="t2v",
        resolution=(854, 480),
        frames=81,
        fps=16,
        steps=30,
        quality_target=82.0,
        energy_budget_j=50_000.0,
        latency_slo_s=120.0,
        notes="LTX-2.3 T2V 480p/81f standard clip; 30-step baseline, "
              "4-step distilled target ~1min on RTX 4090. quality_target 82 "
              "leaves headroom for quality-costing accel under the policy.",
    ))
    # CLI/documentation shorthand alias for the headline 480p scenario.
    lib.add_alias("ltx_t2v_480p", "ltx_t2v_480p_81f")
    # LTX-2.3 T2V 720p, 129 frames (~5.4s@24fps). Higher-res regime.
    # Grounded in 768p being ~40x costlier than 256p for training; inference
    # 720p needs H100/5090 class to hit minutes. SLO 600s, budget 300kJ.
    lib.add(Scenario(
        name="ltx_t2v_720p_129f",
        task="t2v",
        resolution=(1280, 720),
        frames=129,
        fps=24,
        steps=30,
        quality_target=82.0,
        energy_budget_j=300_000.0,
        latency_slo_s=600.0,
        notes="LTX-2.3 T2V 720p/129f; bandwidth/compute heavy, targets "
              "Blackwell 5090 / H100 class. quality_target 82 is a governance "
              "calibration (headroom for quality-costing accel), not VBench-grounded.",
    ))
    # LTX-2.3 image-to-video, 480p, 81 frames. I2V is the primary LTX task.
    lib.add(Scenario(
        name="ltx_i2v_480p",
        task="i2v",
        resolution=(854, 480),
        frames=81,
        fps=16,
        steps=30,
        quality_target=82.0,
        energy_budget_j=50_000.0,
        latency_slo_s=120.0,
        notes="LTX-2.3 I2V 480p/81f; I2V is LTX primary task, more numerically "
              "stable than T2V per the robustness report. quality_target 82 is a "
              "governance calibration (headroom for accel), not VBench-grounded.",
    ))
    # Long video: 1025 frames at 480p/24fps (~43s). Grounded in Kijai
    # context-windowing (1025f -> window 81 + overlap 16 -> <5GB on 5090).
    lib.add(Scenario(
        name="long_video_1025f",
        task="t2v",
        resolution=(832, 480),
        frames=1025,
        fps=24,
        steps=30,
        quality_target=82.0,
        energy_budget_j=1_500_000.0,
        latency_slo_s=1800.0,
        notes="Long video 1025f; requires context-windowing / chunked denoise + "
              "chunked VAE per the long-video report (Kijai window 81+16).",
    ))
    # Edge NPU short clip: Jetson Thor, low-res, 4-step distilled.
    # Grounded in Thor 40-130W, 273 GB/s bandwidth-limited, multi-minute
    # (no public benchmark -> SLO is a planning target, flagged).
    lib.add(Scenario(
        name="edge_npu_shortclip",
        task="t2v",
        resolution=(480, 320),
        frames=49,
        fps=16,
        steps=4,
        quality_target=80.0,
        energy_budget_j=20_000.0,
        latency_slo_s=300.0,
        notes="Edge NPU short clip (Jetson Thor class); 4-step distilled, "
              "low-res; no public benchmark -> SLO is a planning target.",
    ))
    return lib


BUILTIN_SCENARIOS: ScenarioLibrary = _build_builtins()
