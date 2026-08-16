"""AccelSelectorAgent: enumerate skill combos, simulate, rank by Pareto.

The agent enumerates inference-acceleration skill combinations (singles,
config-variants of the configurable skills, the explicit combos named in the
task, and the 5 named recipe presets), simulates each via
PerformanceEnergySimulator, and ranks them on a Pareto frontier of
(latency, energy, quality) under the governance policy constraints
(energy_budget / latency_slo / quality_floor / max_memory).

Skills are resolved from the registry by canonical name (or an explicit
context.config["skills"] pool). Configurable skills carry their variant in
a config key, so e.g. sage_attention is enumerated at v2 and v3, and
quantization at gguf_q4 / nvfp4 / int8 -- each as a distinct candidate.
Rule-engine directives (disabled/preferred skills, config overrides such as the
R3 quantization-method allowlist) passed via context.config steer
selection. Recipes that reference a skill not yet registered (STA, linear
attention) degrade to a grounded documented-estimate candidate.

Recipe presets (synthesis report section 4.3 -- grounded end-to-end estimates):

  R1 distill_cache_sage_vae   CoDMD 4-step + TeaCache + SageAttention2 + VAE
                              tiling                      (consumer NV)  30-60x
  R2 sta_fp4_compile          STA + SageAttention3 + torch.compile + FP8
                              (Blackwell 5090)                           5-7x
  R3 gguf_offload_tile_cache  GGUF Q4 + offload + VAE tiling + TeaCache
                              (low-VRAM consumer NV + MPS)              ~2x (fit)
  R4 linear_nvfp4_compile     SANA-Video linear attn + NVFP4 + compile DiT
                              (5090 / H100)                              ~38x
  R5 trt_compile_graphs_vae   TensorRT FP8 + compile + CUDA Graphs + INT8 VAE
                              (consumer NV + Jetson)                     ~3.5x
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ..core.contracts import (
    DeviceCategory,
    DeviceProfile,
    GovernanceDecision,
    LoadModel,
    Skill,
    SkillImpact,
)
from ..core.registry import REGISTRY
from ..core.scenario import Scenario
from ..core.simulator import (
    AgentContext,
    PerformanceEnergySimulator,
    SimulationResult,
)
from .base import GovernanceAgent
from ..governance.policy import Policy
from ..governance.rules import (
    SKILL_TEACACHE,
    SKILL_SAGE,
    SKILL_DISTILL,
    SKILL_QUANT,
    SKILL_VAE_TILING,
    SKILL_COMPILE,
    SKILL_OFFLOAD,
    SKILL_STA,
    SKILL_LINEAR_ATTN,
)

__all__ = ["RecipePreset", "RECIPES", "AccelSelectorAgent"]

# Core accel skills used for single/pair enumeration.
_CORE_ACCEL_SKILLS = (
    SKILL_TEACACHE,
    SKILL_SAGE,
    SKILL_DISTILL,
    SKILL_QUANT,
    SKILL_VAE_TILING,
    SKILL_COMPILE,
    SKILL_OFFLOAD,
)

# Config-variant expansions for the configurable skills. Each entry is
# (skill_name, config_override, combo_suffix, attention_backend, steps).
# attention_backend "" and steps None mean "derive from device/scenario".
_VARIANTS: list[tuple[str, dict[str, Any], str, str, int | None]] = [
    (SKILL_SAGE, {"version": "v2"}, "v2", "sage2", None),
    (SKILL_SAGE, {"version": "v3"}, "v3", "sage3", None),
    (SKILL_QUANT, {"method": "gguf_q4"}, "gguf_q4", "", None),
    (SKILL_QUANT, {"method": "nvfp4"}, "nvfp4", "", None),
    (SKILL_QUANT, {"method": "int8"}, "int8", "", None),
    (SKILL_DISTILL, {"steps": 4}, "4step", "", 4),
    (SKILL_DISTILL, {"steps": 8}, "8step", "", 8),
    (SKILL_COMPILE, {"backend": "torch_compile"}, "torch_compile", "", None),
    (SKILL_COMPILE, {"backend": "trt"}, "trt", "", None),
    (SKILL_TEACACHE, {"threshold": 0.1}, "thr0.1", "", None),
    (SKILL_TEACACHE, {"threshold": 0.2}, "thr0.2", "", None),
]


@dataclass(frozen=True)
class RecipePreset:
    """A named, grounded acceleration recipe from the synthesis report."""

    name: str
    skills: tuple[tuple[str, dict[str, Any]], ...]  # (skill_name, config_override)
    device_categories: tuple[str, ...]  # empty = all categories
    steps: int
    attention_backend: str  # "" = device default
    precision: str  # "" = bf16
    grounded_speedup: str
    notes: str


RECIPES: tuple[RecipePreset, ...] = (
    RecipePreset(
        name="distill_cache_sage_vae",
        skills=(
            (SKILL_DISTILL, {"steps": 4}),
            (SKILL_TEACACHE, {"threshold": 0.1}),
            (SKILL_SAGE, {"version": "v2"}),
            (SKILL_VAE_TILING, {}),
        ),
        device_categories=(DeviceCategory.CONSUMER_NV,),
        steps=4,
        attention_backend="sage2",
        precision="bf16",
        grounded_speedup="30-60x",
        notes="CoDMD 4-step + TeaCache + SageAttention2 + VAE temporal tiling. "
              "Consumer NV (4090/5090). Synthesis report 4.3 recipe 1.",
    ),
    RecipePreset(
        name="sta_fp4_compile",
        skills=(
            (SKILL_STA, {}),
            (SKILL_SAGE, {"version": "v3"}),
            (SKILL_COMPILE, {"backend": "torch_compile"}),
        ),
        device_categories=(DeviceCategory.CONSUMER_NV,),
        steps=30,
        attention_backend="sage3",
        precision="fp8",
        grounded_speedup="5-7x",
        notes="Sliding Tile Attention + SageAttention3 + region torch.compile + FP8. "
              "Blackwell 5090 (Hunyuan 945s->~150-250s). Recipe 2.",
    ),
    RecipePreset(
        name="gguf_offload_tile_cache",
        skills=(
            (SKILL_QUANT, {"method": "gguf_q4"}),
            (SKILL_OFFLOAD, {"block_swap_ratio": 0.5}),
            (SKILL_VAE_TILING, {}),
            (SKILL_TEACACHE, {"threshold": 0.1}),
        ),
        device_categories=(DeviceCategory.CONSUMER_NV, DeviceCategory.APPLE_SILICON),
        steps=30,
        attention_backend="",
        precision="bf16",
        grounded_speedup="~2x (core value: fit)",
        notes="GGUF Q4 + block-swap offload + VAE temporal tiling + TeaCache. "
              "Low-VRAM consumer NV + MPS: from unrunnable to runnable. Recipe 3.",
    ),
    RecipePreset(
        name="linear_nvfp4_compile",
        skills=(
            (SKILL_LINEAR_ATTN, {}),
            (SKILL_QUANT, {"method": "nvfp4"}),
            (SKILL_COMPILE, {"backend": "torch_compile"}),
        ),
        device_categories=(DeviceCategory.CONSUMER_NV, DeviceCategory.DATACENTER),
        steps=30,
        attention_backend="",
        precision="nvfp4",
        grounded_speedup="~38x",
        notes="SANA-Video 2.0 linear attention + NVFP4 + compiled DiT. "
              "5090 / H100 (120x vs Wan2.2-A14B on single H100). Recipe 4.",
    ),
    RecipePreset(
        name="trt_compile_graphs_vae",
        skills=(
            (SKILL_COMPILE, {"backend": "trt"}),
            (SKILL_VAE_TILING, {}),
        ),
        device_categories=(DeviceCategory.CONSUMER_NV, DeviceCategory.EDGE_NPU),
        steps=30,
        attention_backend="",
        precision="fp8",
        grounded_speedup="~3.5x",
        notes="TensorRT FP8 + torch.compile + CUDA Graphs + INT8 VAE. "
              "Consumer NV + Jetson (Firefly video: 60% latency cut). Recipe 5.",
    ),
)


@dataclass
class Candidate:
    """A simulated skill combination."""

    name: str
    skills: list[Skill]
    skill_configs: dict[str, dict[str, Any]]
    result: SimulationResult
    feasible: bool
    pareto_rank: int = 0
    is_recipe: bool = False
    estimate_only: bool = False
    # The operating-point config used to simulate this candidate, forwarded to
    # the final SimulatorAgent so the final run matches the selected combo.
    attention_backend: str = ""
    steps: int | None = None
    precision: str = "bf16"


class AccelSelectorAgent(GovernanceAgent):
    """Enumerates skill combos, simulates them, and ranks by Pareto."""

    name = "accel_selector"
    role = "select_accel"

    def __init__(
        self,
        name: str | None = None,
        role: str | None = None,
        simulator: PerformanceEnergySimulator | None = None,
    ) -> None:
        super().__init__(name, role)
        self.simulator = simulator or PerformanceEnergySimulator()
        self.last_candidates: list[Candidate] = []

    def run(self, context: AgentContext) -> dict[str, Any]:
        device = context.device
        load = context.load
        scenario = context.scenario
        cfg = dict(context.config or {})

        disabled = set(cfg.get("disabled_skills", set()))
        preferred = set(cfg.get("preferred_skills", set()))
        overrides = dict(cfg.get("config_overrides", {}))
        policy: Policy = cfg.get("policy") or Policy.from_scenario(scenario, device)
        quant_allow = overrides.get("quant_methods_allowed")

        pool = self._resolve_pool(device, load, cfg, disabled)
        baseline = self._simulate_baseline(device, load, scenario, overrides)
        candidates = self._enumerate(
            device, load, scenario, pool, overrides, policy, baseline, quant_allow,
        )

        ranked = self._rank(candidates, preferred, policy)
        self.last_candidates = ranked

        top = ranked[0] if ranked else None
        decisions = self._decisions_for(top, device, load, policy) if top else []
        alternatives = [
            {
                "combo": c.name,
                "skills": [s.registry_name() for s in c.skills],
                "latency_s": c.result.latency_s,
                "energy_j": c.result.energy_j,
                "quality": c.result.quality_score,
                "peak_memory_gb": c.result.peak_memory_gb,
                "pareto_tag": c.result.pareto_tag,
                "feasible": c.feasible,
                "pareto_rank": c.pareto_rank,
                "estimate_only": c.estimate_only,
            }
            for c in ranked
        ]
        notes = self._summarize(ranked, baseline, top, policy, pool)
        return {
            "agent": self.name,
            "role": self.role,
            "decisions": decisions,
            "notes": notes,
            "extra": {
                "alternatives": alternatives,
                "baseline": baseline,
                "pareto_front": [a for a in alternatives if a["pareto_rank"] == 0],
                "top_combo": top.name if top else None,
                "top_skills": list(top.skills) if top else [],
                "top_skill_configs": dict(top.skill_configs) if top else {},
                "top_config": (
                    {
                        "attention_backend": top.attention_backend,
                        "steps": top.steps,
                        "precision": top.precision,
                        "skill_configs": dict(top.skill_configs),
                    }
                    if top is not None else {}
                ),
                "pool": [s.registry_name() for s in pool],
            },
        }

    # -- skill pool --------------------------------------------------------
    def _resolve_pool(
        self, device: DeviceProfile, load: LoadModel, cfg: dict[str, Any],
        disabled: set[str],
    ) -> list[Skill]:
        explicit = cfg.get("skills")
        if explicit:
            skills = list(explicit)
        else:
            skills = []
            for name, cls in REGISTRY.all("skill").items():
                try:
                    inst = cls()
                except Exception:
                    continue
                if getattr(inst, "kind", "accel") != "accel":
                    continue
                skills.append(inst)
        pool: list[Skill] = []
        seen: set[str] = set()
        for s in skills:
            rn = s.registry_name()
            if rn in disabled or rn in seen:
                continue
            if not s.applicable(device, load):
                continue
            pool.append(s)
            seen.add(rn)
        return pool

    # -- enumeration -------------------------------------------------------
    def _simulate_baseline(self, device, load, scenario, overrides) -> SimulationResult:
        config = self._base_config(device, scenario, overrides)
        return self.simulator.simulate(device, load, skills_applied=[], config=config, scenario=scenario)

    def _enumerate(
        self, device, load, scenario, pool, overrides, policy, baseline, quant_allow,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        by_name = {s.registry_name(): s for s in pool}
        spec = device.spec()

        def add_combo(
            name: str, skills: list[Skill], sconfigs: dict[str, dict[str, Any]],
            is_recipe: bool, steps: int | None, backend: str, precision: str,
            estimate: bool,
        ) -> None:
            if estimate:
                est = self._estimate_result(baseline, name)
                candidates.append(Candidate(
                    name=name, skills=skills, skill_configs=sconfigs, result=est,
                    feasible=policy.is_feasible(est), is_recipe=is_recipe,
                    estimate_only=True, attention_backend=backend,
                    steps=steps, precision=precision or "bf16",
                ))
                return
            config = self._base_config(device, scenario, overrides)
            has_distill = any(s.registry_name() == SKILL_DISTILL for s in skills)
            # Inject the actual baseline step count into the distill skill's
            # config so its speedup is MARGINAL (baseline->distilled) instead of
            # relative to the model default. Without this, an already-distilled
            # scenario (e.g. edge_npu 4-step) would get a spurious ~7.5x
            # multiplier on top of its already-reduced step count.
            if has_distill:
                sconfigs = dict(sconfigs or {})
                dc = dict(sconfigs.get(SKILL_DISTILL, {}))
                dc.setdefault("baseline_steps", scenario.steps)
                sconfigs[SKILL_DISTILL] = dc
            if sconfigs:
                config["skill_configs"] = dict(sconfigs)
            if has_distill:
                # The step reduction is modeled by the skill's end-to-end
                # speedup (relative to baseline_steps); do NOT also lower
                # config['steps'], which would double-count the reduction
                # (denoise reduced by fewer steps AND latency /speedup).
                config["distilled"] = True
            elif steps is not None:
                config["steps"] = steps
                if steps <= 8:
                    config["distilled"] = True
            if backend:
                config["attention_backend"] = backend
            eff_precision = "bf16"
            if precision and precision != "bf16" and spec.supports(precision):
                config["precision"] = precision
                eff_precision = precision
            res = self.simulator.simulate(
                device, load, skills_applied=skills, config=config, scenario=scenario,
            )
            candidates.append(Candidate(
                name=name, skills=skills, skill_configs=sconfigs, result=res,
                feasible=policy.is_feasible(res), is_recipe=is_recipe,
                attention_backend=backend or config.get("attention_backend", ""),
                steps=steps, precision=eff_precision,
            ))

        # Baseline.
        candidates.append(Candidate(
            name="baseline", skills=[], skill_configs={}, result=baseline,
            feasible=policy.is_feasible(baseline),
            attention_backend=self._default_backend(device),
            steps=scenario.steps, precision="bf16",
        ))

        # Config-variant singles (these subsume the default-config singles for
        # the configurable skills, with richer coverage).
        covered_single = set()
        for sname, override, suffix, backend, steps in _VARIANTS:
            if sname not in by_name:
                continue
            if sname == SKILL_QUANT and quant_allow is not None:
                if override.get("method") not in quant_allow:
                    continue
            if sname == SKILL_SAGE and override.get("version") == "v3" and not spec.supports("fp4"):
                continue
            if sname == SKILL_QUANT and override.get("method") == "nvfp4" and not spec.supports("fp4"):
                continue
            s = by_name[sname]
            add_combo(sname + ":" + suffix, [s], {sname: override}, False, steps, backend, "", False)
            covered_single.add(sname)
        # Default-config singles for skills without variants.
        for s in pool:
            if s.registry_name() in covered_single:
                continue
            add_combo(s.registry_name(), [s], {}, False, None, "", "", False)

        # Explicit pairs + triples from the task spec.
        explicit_pairs = [
            (SKILL_TEACACHE, SKILL_SAGE),
            (SKILL_TEACACHE, SKILL_DISTILL),
            (SKILL_DISTILL, SKILL_QUANT),
            (SKILL_TEACACHE, SKILL_COMPILE),
            (SKILL_SAGE, SKILL_COMPILE),
            (SKILL_DISTILL, SKILL_VAE_TILING),
            (SKILL_QUANT, SKILL_VAE_TILING),
            (SKILL_TEACACHE, SKILL_QUANT),
            (SKILL_DISTILL, SKILL_OFFLOAD),
        ]
        for a, b in explicit_pairs:
            if a in by_name and b in by_name:
                sa, sb = by_name[a], by_name[b]
                add_combo(a + "+" + b, [sa, sb], {}, False, None, "", "", False)
        # Explicit triple: distill + teacache + sage.
        if all(n in by_name for n in (SKILL_DISTILL, SKILL_TEACACHE, SKILL_SAGE)):
            add_combo(
                SKILL_DISTILL + "+" + SKILL_TEACACHE + "+" + SKILL_SAGE,
                [by_name[SKILL_DISTILL], by_name[SKILL_TEACACHE], by_name[SKILL_SAGE]],
                {SKILL_DISTILL: {"steps": 4}, SKILL_SAGE: {"version": "v2"}, SKILL_TEACACHE: {"threshold": 0.1}},
                False, 4, "sage2", "", False,
            )
        # Generic pairs among core skills (fills out the Pareto surface).
        core = [by_name[n] for n in _CORE_ACCEL_SKILLS if n in by_name]
        for a, b in combinations(core, 2):
            sig = a.registry_name() + "+" + b.registry_name()
            if any(c.name == sig for c in candidates):
                continue
            add_combo(sig, [a, b], {}, False, None, "", "", False)

        # Named recipes.
        for recipe in RECIPES:
            if recipe.device_categories and spec.category not in recipe.device_categories:
                continue
            # R3: respect the quantization-method allowlist. Skip a recipe whose
            # quantization method is disallowed by the quality-floor rule.
            if quant_allow is not None:
                for sname, override in recipe.skills:
                    if sname == SKILL_QUANT and override.get("method") not in quant_allow:
                        skip_recipe = True
                        break
                else:
                    skip_recipe = False
                if skip_recipe:
                    continue
            rskills: list[Skill] = []
            rconfigs: dict[str, dict[str, Any]] = {}
            missing: list[str] = []
            for sname, override in recipe.skills:
                if sname in by_name:
                    rskills.append(by_name[sname])
                    if override:
                        rconfigs[sname] = dict(override)
                else:
                    missing.append(sname)
            estimate = bool(missing)
            backend = recipe.attention_backend
            add_combo(
                "recipe:" + recipe.name, rskills, rconfigs, True,
                recipe.steps, backend, recipe.precision, estimate,
            )

        # Deduplicate by candidate name. Each candidate already carries a
        # unique name (variant singles use "skill:suffix" like
        # "quantization:nvfp4"; pairs use "a+b"; recipes use "recipe:<name>"),
        # so name-based dedup preserves all config-variants of a configurable
        # skill while removing only exact duplicates. (The earlier skill-name
        # signature dedup collapsed gguf_q4/nvfp4/int8 into one candidate.)
        # Recipe-vs-ad-hoc preference is handled downstream in the sort key
        # (is_recipe tie-break), not here.
        seen_names: set[str] = set()
        unique: list[Candidate] = []
        for c in candidates:
            if c.name in seen_names:
                continue
            seen_names.add(c.name)
            unique.append(c)
        return unique

    # -- config ------------------------------------------------------------
    def _base_config(self, device, scenario, overrides) -> dict[str, Any]:
        config: dict[str, Any] = {
            "precision": "bf16",
            "attention_backend": self._default_backend(device),
            "steps": scenario.steps,
            "utilization": 0.75,
        }
        config.update(overrides)
        # quant_methods_allowed is a selector directive, not a simulator key.
        config.pop("quant_methods_allowed", None)
        config.pop("boundary_first_blocks", None)
        config.pop("boundary_last_blocks", None)
        return config

    def _default_backend(self, device: DeviceProfile) -> str:
        backends = device.spec().attention_backends
        if backends:
            for preferred in ("flash", "sdpa", "mlx_sdpa", "sage2", "triton"):
                if preferred in backends:
                    return preferred
            return backends[0]
        return "math"

    # -- estimate fallback -------------------------------------------------
    def _estimate_result(self, baseline: SimulationResult, recipe_name: str) -> SimulationResult:
        recipe = next((r for r in RECIPES if "recipe:" + r.name == recipe_name), None)
        speedup_mid = 2.0
        if recipe is not None:
            spd = recipe.grounded_speedup
            if spd.startswith("~"):
                spd = spd[1:]
            if "x" in spd:
                lo_hi = spd.split("x")[0]
                if "-" in lo_hi:
                    parts = lo_hi.split("-")
                    try:
                        speedup_mid = (float(parts[0]) + float(parts[1])) / 2.0
                    except ValueError:
                        speedup_mid = 2.0
                else:
                    try:
                        speedup_mid = float(lo_hi)
                    except ValueError:
                        speedup_mid = 2.0
        speedup_mid = max(1.0, speedup_mid)
        scaled = SimulationResult(
            latency_s=baseline.latency_s / speedup_mid,
            energy_j=baseline.energy_j / speedup_mid,
            peak_memory_gb=baseline.peak_memory_gb,
            quality_score=max(0.0, baseline.quality_score - 1.0),
            throughput_tokens_s=baseline.throughput_tokens_s * speedup_mid,
            breakdown=dict(baseline.breakdown),
            pareto_tag="estimate",
            warnings=list(baseline.warnings) + [
                "Estimate only: skill(s) not registered; using grounded speedup midpoint "
                + format(speedup_mid, ".1f") + "x."
            ],
            tokens=baseline.tokens,
            steps=baseline.steps,
            precision=baseline.precision,
            attention_backend=baseline.attention_backend,
        )
        return scaled

    # -- Pareto ranking ----------------------------------------------------
    def _rank(
        self, candidates: list[Candidate], preferred: set[str], policy: Policy,
    ) -> list[Candidate]:
        sim = [c for c in candidates if not c.estimate_only]
        for i, ci in enumerate(sim):
            dominated = False
            for j, cj in enumerate(sim):
                if i == j:
                    continue
                if self._dominates(cj.result, ci.result):
                    dominated = True
                    break
            ci.pareto_rank = 0 if not dominated else 1
        for c in candidates:
            if c.estimate_only:
                c.pareto_rank = 2
        # Prefer: Pareto front, feasible, more preferred-skill hits, lower
        # latency, lower energy. Recipes get a tiny tie-break nudge so a named
        # recipe wins a tie against an ad-hoc combo of the same skills.
        def sort_key(c: Candidate):
            preferred_hit = sum(1 for s in c.skills if s.registry_name() in preferred)
            return (
                c.pareto_rank,
                0 if c.feasible else 1,
                -preferred_hit,
                0 if c.is_recipe else 1,
                c.result.latency_s,
                c.result.energy_j,
            )
        return sorted(candidates, key=sort_key)

    def _dominates(self, a: SimulationResult, b: SimulationResult) -> bool:
        ge_all = (
            a.latency_s <= b.latency_s
            and a.energy_j <= b.energy_j
            and a.quality_score >= b.quality_score
        )
        strict = (
            a.latency_s < b.latency_s
            or a.energy_j < b.energy_j
            or a.quality_score > b.quality_score
        )
        return ge_all and strict

    # -- decisions ---------------------------------------------------------
    def _decisions_for(
        self, top: Candidate, device: DeviceProfile, load: LoadModel, policy: Policy,
    ) -> list[GovernanceDecision]:
        decisions: list[GovernanceDecision] = []
        for s in top.skills:
            cfg = dict(top.skill_configs.get(s.registry_name(), s.default_config()))
            impact = s.predict(device, load, cfg)
            decisions.append(GovernanceDecision(
                skill_name=s.registry_name(),
                config=cfg,
                predicted_impact=impact,
                rationale=(
                    "Selected in top combo '" + top.name + "' (Pareto rank "
                    + str(top.pareto_rank) + ", " + ("feasible" if top.feasible else "infeasible")
                    + "): latency " + format(top.result.latency_s, ".2f") + "s, energy "
                    + format(top.result.energy_j, ".0f") + "J, quality "
                    + format(top.result.quality_score, ".2f") + "."
                ),
            ))
        if not top.skills and top.name != "baseline":
            decisions.append(GovernanceDecision(
                skill_name=top.name,
                config={},
                predicted_impact=SkillImpact(speedup=1.0),
                rationale="Top candidate is a recipe estimate (skills not registered).",
            ))
        return decisions

    # -- summary -----------------------------------------------------------
    def _summarize(
        self, ranked: list[Candidate], baseline: SimulationResult,
        top: Candidate | None, policy: Policy, pool: list[Skill],
    ) -> str:
        lines = [
            "Evaluated " + str(len(ranked)) + " combos (skill pool: "
            + (", ".join(s.registry_name() for s in pool) if pool else "empty -- no accel skills registered")
            + ").",
            "Baseline: latency " + format(baseline.latency_s, ".2f") + "s, energy "
            + format(baseline.energy_j, ".0f") + "J, quality "
            + format(baseline.quality_score, ".2f") + ", tag=" + baseline.pareto_tag + ".",
        ]
        if top is not None:
            speedup = baseline.latency_s / top.result.latency_s if top.result.latency_s > 0 else 0.0
            lines.append(
                "Top: '" + top.name + "' -> latency " + format(top.result.latency_s, ".2f")
                + "s (" + format(speedup, ".1f") + "x), energy "
                + format(top.result.energy_j, ".0f") + "J, quality "
                + format(top.result.quality_score, ".2f") + ", "
                + ("policy-feasible" if top.feasible else "policy-INFEASIBLE")
                + (", estimate-only" if top.estimate_only else "") + "."
            )
        feasible = [c for c in ranked if c.feasible and not c.estimate_only]
        lines.append(
            str(len(feasible)) + " policy-feasible simulated combo(s); "
            + str(sum(1 for c in ranked if c.pareto_rank == 0)) + " on Pareto front."
        )
        return "\n".join(lines)
