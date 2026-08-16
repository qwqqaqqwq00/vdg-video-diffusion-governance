"""VDG command-line interface.

Subcommands:

  vdg devices                                 list registered device plugins
  vdg models                                  list registered load plugins
  vdg simulate --device X --model Y --scenario S [--skills a,b]
                                              run a single performance-energy
                                              simulation
  vdg calibrate --device X --model Y --scenario S
                                              run the simulation with
                                              anchor-based calibration and
                                              print calibrated prediction vs
                                              the measured anchor (rel. error)
  vdg govern  --device X --model Y --scenario S [--energy-budget N]
                                              run the full governance pipeline
  vdg probe   --device X [--model Y] [--precision P]
                                              run NumericalProbe on a device
  vdg report                                  print a system status report
  vdg -h | --help                             show usage

The no-argument form prints the version banner and registered-plugin
inventory (preserving the foundation CLI contract), so vdg with no args is
a quick health check that the registry and energy models are wired.
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__
from .core.registry import REGISTRY
from .core.scenario import BUILTIN_SCENARIOS
from .core.simulator import PerformanceEnergySimulator

__all__ = ["main"]


# --------------------------------------------------------------------------
# Info / usage banners (preserve the foundation CLI contract: main([]) prints
# "VDG" + version + plugin inventory including energy_model/tdp; main(["-h"])
# prints a block containing "Usage").
# --------------------------------------------------------------------------
def _format_plugins() -> str:
    REGISTRY.discover()
    kinds = REGISTRY.kinds()
    if not kinds:
        return "No plugins registered yet (phase-2 devices/loads/skills not installed)."
    lines = []
    for kind in sorted(kinds):
        names = REGISTRY.names(kind)
        lines.append(kind + " (" + str(len(names)) + "): " + ", ".join(sorted(names)))
    return "\n".join(lines)


def _print_info() -> None:
    print("VDG (Video Diffusion Governance) " + __version__)
    print("-" * 48)
    print(_format_plugins())


def _print_usage() -> None:
    print("")
    print("Usage: vdg <command> [options]")
    print("")
    print("Commands:")
    print("  devices                                 list registered device plugins")
    print("  models                                  list registered load plugins")
    print("  simulate --device X --model Y --scenario S [--skills a,b]")
    print("                                          run a single simulation")
    print("  calibrate --device X --model Y --scenario S")
    print("                                          calibrated simulation vs")
    print("                                          measured anchor (rel. error)")
    print("  govern  --device X --model Y --scenario S [--energy-budget N]")
    print("                                          run the full governance pipeline")
    print("  probe   --device X [--model Y] [--precision P]")
    print("                                          run NumericalProbe on a device")
    print("  runtime --device X --model Y --scenario S --runtime comfyui|diffusers|torch")
    print("          |lightx2v|mlx [--out FILE] [--comfy-* MODEL PATHS]")
    print("                                          run governance, then emit the runtime")
    print("                                          artifact (ComfyUI workflow JSON / patch")
    print("                                          script / launch command)")
    print("  report                                  print a system status report")
    print("  -h | --help                             show this help")
    print("")
    print("Scenarios: " + ", ".join(BUILTIN_SCENARIOS.names()))


# --------------------------------------------------------------------------
# Resolution helpers
# --------------------------------------------------------------------------
def _resolve_device(name: str):
    cls = REGISTRY.get("device", name)
    if cls is None:
        raise KeyError(
            "Unknown device: " + repr(name)
            + ". Registered: " + ", ".join(sorted(REGISTRY.names("device")))
            + "."
        )
    return cls()


def _resolve_load(name: str):
    cls = REGISTRY.get("load", name)
    if cls is None:
        raise KeyError(
            "Unknown load: " + repr(name)
            + ". Registered: " + ", ".join(sorted(REGISTRY.names("load")))
            + "."
        )
    return cls()


def _resolve_scenario(name: str):
    return BUILTIN_SCENARIOS.get(name)


def _resolve_skills(names_csv: str, device, load):
    if not names_csv:
        return []
    skills = []
    for raw in names_csv.split(","):
        nm = raw.strip()
        if not nm:
            continue
        cls = REGISTRY.get("skill", nm)
        if cls is None:
            raise KeyError("Unknown skill: " + repr(nm) + ".")
        inst = cls()
        if not inst.applicable(device, load):
            print("warning: skill " + repr(nm) + " not applicable; skipping.", file=sys.stderr)
            continue
        skills.append(inst)
    return skills


# --------------------------------------------------------------------------
# Output formatters
# --------------------------------------------------------------------------
def _print_result(result) -> None:
    print("Simulation result")
    print("  latency:        " + format(result.latency_s, ".2f") + " s")
    print("  energy:         " + format(result.energy_j, ".0f") + " J")
    print("  quality:        " + format(result.quality_score, ".2f"))
    print("  peak memory:    " + format(result.peak_memory_gb, ".2f") + " GB")
    print("  throughput:     " + format(result.throughput_tokens_s, ".0f") + " tokens/s")
    print("  tokens/steps:   " + str(result.tokens) + " / " + str(result.steps))
    print("  precision/attn: " + result.precision + " / " + result.attention_backend)
    print("  pareto tag:     " + result.pareto_tag)
    if result.breakdown:
        print("  breakdown (s):")
        for k in ("denoise", "attention", "ffn", "vae_decode", "te_encode"):
            if k in result.breakdown:
                print("    " + k + ": " + format(result.breakdown[k], ".3f"))
    if result.warnings:
        print("  warnings:")
        for w in result.warnings:
            print("    - " + w)


def _print_probe(report) -> None:
    print("NumericalProbe report")
    print("  device:    " + report.device_name)
    print("  precision: " + report.precision
          + " (" + ("simulated" if report.simulated else "measured") + ")")
    print("  divergence: " + ("DETECTED" if report.has_failure else "none"))
    print("  results (" + str(len(report.results)) + "):")
    for r in report.results:
        print("    [" + r.status + "] " + r.op + "  max_diff=" + format(r.max_diff, ".3g")
              + "  nan_count=" + str(r.nan_count))
        print("        input: " + r.input_desc)
    suggested = report.repair_skills_suggested
    print("  repair skills suggested: " + (", ".join(suggested) if suggested else "none"))
    print("")
    print(report.summary)


# --------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------
def _cmd_devices(args) -> int:
    REGISTRY.discover()
    names = REGISTRY.names("device")
    if not names:
        print("No devices registered. Install device plugins in vdg/devices.")
        return 0
    print("Registered devices (" + str(len(names)) + "):")
    for n in sorted(names):
        cls = REGISTRY.get("device", n)
        try:
            spec = cls().spec()
            print("  " + n + "  (" + spec.category + ", " + format(spec.memory_gb, ".0f")
                  + "GB, " + ", ".join(spec.supported_precisions) + ")")
        except Exception as exc:
            print("  " + n + "  (spec unavailable: " + repr(exc) + ")")
    return 0


def _cmd_models(args) -> int:
    REGISTRY.discover()
    names = REGISTRY.names("load")
    if not names:
        print("No loads registered. Install load plugins in vdg/loads.")
        return 0
    print("Registered loads (" + str(len(names)) + "):")
    for n in sorted(names):
        cls = REGISTRY.get("load", n)
        try:
            c = cls().characteristics()
            print("  " + n + "  (" + c.model_name + ", " + format(c.params_b, ".1f")
                  + "B params, " + str(c.layers) + " layers, default "
                  + str(c.default_steps) + " steps)")
        except Exception as exc:
            print("  " + n + "  (characteristics unavailable: " + repr(exc) + ")")
    return 0


def _cmd_simulate(args) -> int:
    try:
        device = _resolve_device(args.device)
        load = _resolve_load(args.model)
        scenario = _resolve_scenario(args.scenario)
        skills = _resolve_skills(args.skills or "", device, load)
    except KeyError as exc:
        print("error: " + str(exc))
        return 1
    config = {
        "precision": args.precision,
        "steps": args.steps if args.steps is not None else scenario.steps,
        "utilization": 0.75,
    }
    if args.attention:
        config["attention_backend"] = args.attention
    # Step distillation's speedup must be marginal relative to the step count
    # actually being simulated (config['steps'] = --steps or scenario.steps),
    # not the model default -- otherwise an already-distilled scenario (e.g.
    # edge_npu 4-step) gets a spurious multiplier, and an explicit '--steps 4'
    # plus the skill speedup would double-count the reduction.
    if any(s.registry_name() == "step_distill" for s in skills):
        sc = dict(config.get("skill_configs") or {})
        dc = dict(sc.get("step_distill", {}))
        dc.setdefault("baseline_steps", config["steps"])
        sc["step_distill"] = dc
        config["skill_configs"] = sc
        config["distilled"] = True
    sim = PerformanceEnergySimulator()
    result = sim.simulate(device, load, skills_applied=skills, config=config, scenario=scenario)
    _print_result(result)
    return 0


def _cmd_calibrate(args) -> int:
    """Run the simulation with anchor-based calibration and compare to measured.

    Prints the uncalibrated (roofline) prediction, the matched anchor's real
    measured latency, the calibration scale applied, the calibrated prediction
    and the relative error vs the measurement -- then the full predicted-vs-
    measured table for every anchor of this (device, load) pair.
    """
    from .core.calibration import (
        ANCHORS,
        CalibrationReport,
        CalibratedSimulator,
        find_anchor,
    )
    try:
        device = _resolve_device(args.device)
        load = _resolve_load(args.model)
        scenario = _resolve_scenario(args.scenario)
    except KeyError as exc:
        print("error: " + str(exc))
        return 1
    config = {
        "precision": args.precision,
        "steps": args.steps if args.steps is not None else scenario.steps,
        "utilization": 0.75,
    }
    if args.attention:
        config["attention_backend"] = args.attention

    cal_sim = CalibratedSimulator()
    cal = cal_sim.simulate(device, load, skills_applied=[], config=config, scenario=scenario)
    base = cal_sim.last_base_result
    anchor = find_anchor(device.registry_name(), load.registry_name(), scenario.resolution)
    report = CalibrationReport.compare(device, load, scenario)

    print("Calibration for " + device.spec().name + " / "
          + load.characteristics().model_name + " @ " + scenario.name
          + " (" + str(scenario.resolution) + " " + str(scenario.frames) + "f, "
          + str(cal.steps) + " steps)")
    print("  base prediction:   " + format(base.latency_s, ".2f") + " s (roofline, uncalibrated)")
    if anchor is not None:
        measured = anchor.measured_latency_s
        scale = cal.latency_s / base.latency_s if base.latency_s > 0 else 0.0
        # The roofline error at the ANCHOR operating point (same workload as the
        # measurement) is the error the calibration removes -- the calibrated
        # run is at the scenario's step count, so comparing it directly against
        # the anchor's step count would conflate steps and model error.
        row_err = next(
            (r.relative_error_pct for r in report.rows if r.anchor is anchor),
            None,
        )
        print("  anchor:            " + anchor.device_name + "/" + anchor.load_name
              + " @ " + str(anchor.resolution) + " " + str(anchor.frames) + "f "
              + str(anchor.steps) + " steps (kind=" + anchor.kind + ")")
        print("  measured:          " + format(measured, ".2f") + " s  [" + anchor.source + "]")
        print("  calibration scale: " + format(scale, ".4f"))
        print("  calibrated:        " + format(cal.latency_s, ".2f") + " s, energy "
              + format(cal.energy_j, ".0f") + " J")
        print("  roofline error @ anchor op point: "
              + (format(row_err, "+.1f") + "% (removed by calibration)" if row_err is not None else "n/a"))
    else:
        print("  anchor:            none -- engineering estimate (no measured data")
        print("                     point within 20% resolution for this device/load)")
        print("  calibrated:        " + format(cal.latency_s, ".2f") + " s, energy "
              + format(cal.energy_j, ".0f") + " J (scale 1.0)")
    if cal.warnings:
        print("  warnings:")
        for w in cal.warnings:
            print("    - " + w)
    print("")
    print("Anchors available: " + str(len(ANCHORS)))
    print("")
    print(report.render())
    return 0


def _cmd_govern(args) -> int:
    from .governance.pipeline import GovernancePipeline
    try:
        scenario = BUILTIN_SCENARIOS.get(args.scenario)
        device_cls = REGISTRY.get("device", args.device)
        load_cls = REGISTRY.get("load", args.model)
        if device_cls is None:
            raise KeyError(
                "Unknown device: " + repr(args.device)
                + ". Registered: " + ", ".join(sorted(REGISTRY.names("device"))) + "."
            )
        if load_cls is None:
            raise KeyError(
                "Unknown load: " + repr(args.model)
                + ". Registered: " + ", ".join(sorted(REGISTRY.names("load"))) + "."
            )
    except KeyError as exc:
        print("error: " + str(exc))
        return 1
    pipeline = GovernancePipeline()
    try:
        report = pipeline.run(
            args.device, args.model, scenario,
            energy_budget_j=args.energy_budget,
            latency_slo_s=args.latency_slo,
            quality_floor=args.quality_floor,
            max_memory_gb=args.max_memory,
            sim_probe=not args.real_probe,
        )
    except Exception as exc:
        print("error: governance run failed: " + repr(exc))
        return 1
    print(report.summary())
    print("")
    print("Rationale: " + report.rationale)
    if report.alternatives:
        print("")
        print("Alternatives (ranked):")
        for a in report.alternatives[:8]:
            tag = "estimate" if a.get("estimate_only") else ("feasible" if a.get("feasible") else "infeasible")
            print("  " + a["combo"] + "  latency=" + format(a["latency_s"], ".2f")
                  + "s energy=" + format(a["energy_j"], ".0f") + "J q=" + format(a["quality"], ".2f")
                  + " [" + tag + "]")
    if report.patch_instructions:
        print("")
        print("Patch instructions:")
        for block in report.patch_instructions:
            print(block)
            print("")
    if report.probe_summary:
        print("Probe: " + report.probe_summary.replace("\n", " | "))
    return 0


def _diffusers_script(report, args) -> str:
    """Emit a ready-to-run diffusers script for the governance decisions."""
    lines = [
        '"""VDG-generated diffusers LTX-Video binding.',
        '',
        'Builds the pipeline, applies the repair decisions in-process (via',
        'TorchRuntime on pipe.transformer / pipe.vae) and the accel decisions',
        'via the diffusers API surface (enable_teacache / tiling / offload /',
        'compile / set_num_inference_steps).',
        '"""',
        '',
        'from vdg.runtime.diffusers_runtime import DiffusersRuntime',
        '',
        'MODEL_ID = ' + repr(args.model_id or "Lightricks/LTX-Video"),
        'rt = DiffusersRuntime()',
        'pipe = rt.build_pipeline(MODEL_ID)',
        'if pipe is None:',
        '    raise SystemExit(rt.last_error or "pipeline load failed")',
        '',
        '# Repair skills: in-process fp32 guards on transformer + vae.',
        'repairs = rt.apply_repairs(pipe, ' + repr(_decision_pairs(report)) + ')',
        'print("repairs applied:", repairs["applied"])',
        '',
        '# Accel skills: diffusers API mappings (best-effort).',
    ]
    for skill_name, config in _decision_pairs(report):
        if skill_name in _REPAIR_SKILL_SET:
            continue
        lines.append('acc = rt.apply_accel(pipe, ' + repr(skill_name) + ', ' + repr(config) + ')')
        lines.append('print(' + repr(skill_name) + ', "applied=", acc["applied"], "|", acc["notes"])')
    lines.append('')
    lines.append('# Generate (distilled 4-step checkpoint recommended when step_distill is active).')
    lines.append('# video = pipe(prompt=..., num_frames=..., height=..., width=...).frames[0]')
    return '\n'.join(lines)


_REPAIR_SKILL_SET = frozenset({
    "gelu_fp32", "adaln_fp32", "rmsnorm_fp32", "softmax_fp32", "vae_fp32",
})


def _decision_pairs(report) -> list[tuple[str, dict]]:
    """Flatten GovernanceReport.decisions to (skill_name, config) pairs."""
    return [(d.skill_name, dict(d.config)) for d in report.decisions]


def _cmd_runtime(args) -> int:
    """Run governance, then emit the artifact for the selected runtime."""
    from .governance.pipeline import GovernancePipeline
    try:
        scenario = BUILTIN_SCENARIOS.get(args.scenario)
        device_cls = REGISTRY.get("device", args.device)
        load_cls = REGISTRY.get("load", args.model)
        if device_cls is None:
            raise KeyError(
                "Unknown device: " + repr(args.device)
                + ". Registered: " + ", ".join(sorted(REGISTRY.names("device"))) + "."
            )
        if load_cls is None:
            raise KeyError(
                "Unknown load: " + repr(args.model)
                + ". Registered: " + ", ".join(sorted(REGISTRY.names("load"))) + "."
            )
    except KeyError as exc:
        print("error: " + str(exc))
        return 1

    pipeline = GovernancePipeline()
    try:
        report = pipeline.run(
            args.device, args.model, scenario,
            energy_budget_j=args.energy_budget,
            latency_slo_s=args.latency_slo,
            quality_floor=args.quality_floor,
            max_memory_gb=args.max_memory,
            sim_probe=not args.real_probe,
        )
    except Exception as exc:
        print("error: governance run failed: " + repr(exc))
        return 1

    print(report.summary())
    print("")

    # Import the runtime package (auto-imported by vdg/__init__, but be
    # explicit so this works even when the package was imported lazily).
    from . import runtime as runtime_pkg

    artifact: str
    if args.runtime == "comfyui":
        model_paths = {
            key: val for key, val in (
                ("checkpoint", args.comfy_checkpoint),
                ("unet", args.comfy_unet),
                ("vae", args.comfy_vae),
                ("clip", args.comfy_clip),
            ) if val
        }
        wf = runtime_pkg.build_workflow(
            report.decisions, args.model, scenario, model_paths,
        )
        if args.prompt_positive or args.prompt_negative:
            wf = _with_prompts(wf, args.prompt_positive, args.prompt_negative)
        artifact = runtime_pkg.render_markdown(wf)
    elif args.runtime == "torch":
        artifact = runtime_pkg.render_patch_script(report.decisions)
    elif args.runtime == "diffusers":
        artifact = _diffusers_script(report, args)
    elif args.runtime == "lightx2v":
        artifact = runtime_pkg.render_lightx2v_command(
            report.decisions, args.model, scenario.resolution, scenario.frames,
        )
    elif args.runtime == "mlx":
        artifact = runtime_pkg.render_mlx_command(
            report.decisions, args.model, scenario.resolution, scenario.frames,
        )
    else:  # pragma: no cover - argparse constrains the choices
        print("error: unknown runtime " + repr(args.runtime))
        return 1

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(artifact + "\n")
        print("artifact written to " + args.out)
    else:
        print(artifact)
    return 0


def _with_prompts(wf, positive: str | None = None, negative: str | None = None) -> dict:
    """Replace CLIPTextEncode texts: first encode node = positive, second = negative."""
    import copy
    wf = copy.deepcopy(wf)
    encode_nodes = [
        nid for nid in sorted(wf.get("nodes", {}), key=int)
        if wf["nodes"][nid].get("class_type") == "CLIPTextEncode"
    ]
    if positive is not None and encode_nodes:
        wf["nodes"][encode_nodes[0]]["inputs"]["text"] = positive
    if negative is not None and len(encode_nodes) > 1:
        wf["nodes"][encode_nodes[1]]["inputs"]["text"] = negative
    return wf


def _cmd_probe(args) -> int:
    from .agents.diagnostic import NumericalProbe
    from .core.contracts import DeviceCategory
    try:
        device = _resolve_device(args.device)
    except KeyError as exc:
        print("error: " + str(exc))
        return 1
    cat = device.spec().category
    if cat == DeviceCategory.APPLE_SILICON:
        probe_name = "mps"
    elif cat in (DeviceCategory.CONSUMER_NV, DeviceCategory.DATACENTER):
        probe_name = "cuda"
    else:
        probe_name = device.spec().name
    probe = NumericalProbe()
    if args.sim_probe:
        # Reproducible documented-thresholds path (host-independent).
        report = probe._probe_simulated(probe_name, args.precision)
    else:
        report = probe.probe_ops(probe_name, args.precision)
    _print_probe(report)
    return 0


def _cmd_report(args) -> int:
    REGISTRY.discover()
    print("VDG (Video Diffusion Governance) " + __version__ + " -- status report")
    print("=" * 56)
    print("")
    print("Registered plugins:")
    inv = _format_plugins()
    print("  " + inv.replace("\n", "\n  "))
    print("")
    print("Built-in scenarios (" + str(len(BUILTIN_SCENARIOS.names())) + "):")
    for s in BUILTIN_SCENARIOS.all():
        print("  " + s.name + "  " + str(s.resolution) + " " + str(s.frames) + "f/"
              + str(s.fps) + "fps, " + str(s.steps) + " steps, SLO "
              + format(s.latency_slo_s, ".0f") + "s, budget "
              + format(s.energy_budget_j, ".0f") + "J")
    print("")
    try:
        from .agents.accel_selector import RECIPES
        print("Acceleration recipe presets (" + str(len(RECIPES)) + "):")
        for r in RECIPES:
            print("  " + r.name + "  -> " + r.grounded_speedup
                  + "  [" + ", ".join(s[0] for s in r.skills) + "]")
    except Exception:
        pass
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vdg",
        description="VDG (Video Diffusion Governance): simulate and govern "
                    "heterogeneous video DiT deployment.",
        add_help=True,
    )
    sub = parser.add_subparsers(dest="command")

    p_dev = sub.add_parser("devices", help="list registered device plugins")
    p_dev.set_defaults(func=_cmd_devices)

    p_mod = sub.add_parser("models", help="list registered load plugins")
    p_mod.set_defaults(func=_cmd_models)

    p_sim = sub.add_parser("simulate", help="run a single simulation")
    p_sim.add_argument("--device", required=True)
    p_sim.add_argument("--model", required=True)
    p_sim.add_argument("--scenario", required=True)
    p_sim.add_argument("--skills", default="", help="comma-separated skill names")
    p_sim.add_argument("--precision", default="bf16")
    p_sim.add_argument("--steps", type=int, default=None)
    p_sim.add_argument("--attention", default=None, help="attention backend")
    p_sim.set_defaults(func=_cmd_simulate)

    p_cal = sub.add_parser(
        "calibrate",
        help="calibrated simulation vs measured anchor",
    )
    p_cal.add_argument("--device", required=True)
    p_cal.add_argument("--model", required=True)
    p_cal.add_argument("--scenario", required=True)
    p_cal.add_argument("--precision", default="bf16")
    p_cal.add_argument("--steps", type=int, default=None)
    p_cal.add_argument("--attention", default=None, help="attention backend")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_gov = sub.add_parser("govern", help="run the full governance pipeline")
    p_gov.add_argument("--device", required=True)
    p_gov.add_argument("--model", required=True)
    p_gov.add_argument("--scenario", required=True)
    p_gov.add_argument("--energy-budget", type=float, default=None, dest="energy_budget")
    p_gov.add_argument("--latency-slo", type=float, default=None, dest="latency_slo")
    p_gov.add_argument("--quality-floor", type=float, default=None, dest="quality_floor")
    p_gov.add_argument("--max-memory", type=float, default=None, dest="max_memory")
    p_gov.add_argument("--real-probe", action="store_true", default=False,
                       dest="real_probe",
                       help="use the live NumericalProbe (measured) instead of "
                            "the default simulated (documented-thresholds) path")
    p_gov.set_defaults(func=_cmd_govern)

    p_probe = sub.add_parser("probe", help="run NumericalProbe on a device")
    p_probe.add_argument("--device", required=True)
    p_probe.add_argument("--precision", default="bf16")
    p_probe.add_argument("--sim-probe", action="store_true", default=False,
                         dest="sim_probe",
                         help="force the simulated (documented-thresholds) probe "
                              "instead of the live measured path")
    p_probe.set_defaults(func=_cmd_probe)

    p_rt = sub.add_parser(
        "runtime",
        help="run governance, then emit the artifact for a target runtime",
    )
    p_rt.add_argument("--device", required=True)
    p_rt.add_argument("--model", required=True)
    p_rt.add_argument("--scenario", required=True)
    p_rt.add_argument(
        "--runtime", required=True,
        choices=["comfyui", "diffusers", "torch", "lightx2v", "mlx"],
        help="target runtime for the emitted artifact",
    )
    p_rt.add_argument("--energy-budget", type=float, default=None, dest="energy_budget")
    p_rt.add_argument("--latency-slo", type=float, default=None, dest="latency_slo")
    p_rt.add_argument("--quality-floor", type=float, default=None, dest="quality_floor")
    p_rt.add_argument("--max-memory", type=float, default=None, dest="max_memory")
    p_rt.add_argument("--real-probe", action="store_true", default=False,
                      dest="real_probe",
                      help="use the live NumericalProbe instead of the simulated path")
    p_rt.add_argument("--out", default=None, help="write the artifact to a file")
    p_rt.add_argument("--comfy-checkpoint", default=None, dest="comfy_checkpoint",
                      help="ComfyUI checkpoint file path (CheckpointLoaderSimple)")
    p_rt.add_argument("--comfy-unet", default=None, dest="comfy_unet",
                      help="ComfyUI diffusion-model file path (UNETLoader / GGUF)")
    p_rt.add_argument("--comfy-vae", default=None, dest="comfy_vae",
                      help="ComfyUI VAE file path (VAELoader)")
    p_rt.add_argument("--comfy-clip", default=None, dest="comfy_clip",
                      help="ComfyUI CLIP file path (CLIPLoader)")
    p_rt.add_argument("--prompt-positive", default=None, dest="prompt_positive",
                      help="override the positive CLIPTextEncode prompt")
    p_rt.add_argument("--prompt-negative", default=None, dest="prompt_negative",
                      help="override the negative CLIPTextEncode prompt")
    p_rt.add_argument("--model-id", default=None, dest="model_id",
                      help="HuggingFace model id for the diffusers runtime")
    p_rt.set_defaults(func=_cmd_runtime)

    p_rep = sub.add_parser("report", help="print a system status report")
    p_rep.set_defaults(func=_cmd_report)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # No arguments -> banner + plugin inventory (foundation CLI contract).
    if not raw:
        _print_info()
        return 0
    # Global help / version handled manually so the output carries the exact
    # "Usage:" / version banner the foundation tests expect.
    if raw[0] in ("-h", "--help", "help"):
        _print_info()
        _print_usage()
        return 0
    if raw[0] in ("-v", "--version"):
        print("vdg " + __version__)
        return 0
    parser = _build_parser()
    try:
        ns = parser.parse_args(raw)
    except SystemExit as exc:
        # argparse prints usage/errors itself and exits; surface its code.
        return int(exc.code) if exc.code is not None else 2
    func = getattr(ns, "func", None)
    if func is None:
        _print_info()
        _print_usage()
        return 0
    try:
        return func(ns)
    except BrokenPipeError:
        return 0
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        print("error: " + repr(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
