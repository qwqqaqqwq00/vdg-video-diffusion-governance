"""Governance agent + CLI tests."""
from __future__ import annotations

import pytest

from vdg import AgentContext, BUILTIN_SCENARIOS
from vdg.agents.base import AgentResult, GovernanceAgent
import vdg.cli as cli_module


@pytest.fixture(autouse=True)
def _force_simulated_probe(monkeypatch):
    """Force the NumericalProbe SIMULATED path so the probe/govern CLI tests are
    host-independent (no dependence on torch / MPS / CUDA availability)."""
    from vdg.skills.repair.numerical_probe import NumericalProbe
    monkeypatch.setattr(
        NumericalProbe, "probe_ops",
        lambda self, device_name="mps", precision="bf16":
            self._probe_simulated(device_name, precision),
    )


def test_governance_agent_defaults():
    a = GovernanceAgent()
    assert a.name == "base"
    assert a.role == "governance"


def test_governance_agent_custom_name_role():
    a = GovernanceAgent(name="repair_governor", role="repair")
    assert a.name == "repair_governor"
    assert a.role == "repair"


def test_governance_agent_run_default(rtx4090, ltx23):
    ctx = AgentContext(
        device=rtx4090, load=ltx23, scenario=BUILTIN_SCENARIOS.get("ltx_t2v_480p_81f"),
    )
    out = GovernanceAgent().run(ctx)
    assert out["agent"] == "base"
    assert out["decisions"] == []


def test_agent_result_dataclass():
    r = AgentResult(agent="x", role="y")
    assert r.agent == "x"
    assert r.role == "y"
    assert r.decisions == []
    assert r.notes == ""
    assert r.extra == {}


def test_cli_main_returns_zero(capsys):
    rc = cli_module.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "VDG" in out
    assert "0.1.0" in out


def test_cli_help(capsys):
    rc = cli_module.main(["-h"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Usage" in out


def test_cli_lists_energy_models(capsys):
    cli_module.main([])
    out = capsys.readouterr().out
    # tdp + measured energy models register on import.
    assert "energy_model" in out
    assert "tdp" in out


# ---------------------------------------------------------------------------
# Subcommand smoke tests (exit 0 + key output present)
# ---------------------------------------------------------------------------
def test_cli_devices(capsys):
    rc = cli_module.main(["devices"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Registered devices" in out
    # A shipped device name appears (registry name == class name).
    assert "RTX4090" in out or "RTX 4090" in out


def test_cli_models(capsys):
    rc = cli_module.main(["models"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Registered loads" in out
    assert "LTX-2.3" in out


def test_cli_simulate(capsys):
    rc = cli_module.main([
        "simulate", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Simulation result" in out
    assert "latency" in out
    assert "energy" in out
    assert "pareto tag" in out


def test_cli_simulate_with_skills(capsys):
    rc = cli_module.main([
        "simulate", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--skills", "teacache,step_distill",
        "--steps", "4",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Simulation result" in out


def test_cli_simulate_unknown_device_returns_1(capsys):
    rc = cli_module.main([
        "simulate", "--device", "nope", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f",
    ])
    assert rc == 1
    err = capsys.readouterr().out
    assert "error" in err.lower()


def test_cli_govern(capsys):
    rc = cli_module.main([
        "govern", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Governance Report" in out
    assert "baseline" in out
    assert "top combo" in out
    assert "Rationale" in out


def test_cli_govern_with_energy_budget(capsys):
    rc = cli_module.main([
        "govern", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--energy-budget", "1000",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Governance Report" in out


def test_cli_probe(capsys):
    rc = cli_module.main(["probe", "--device", "M4_Max"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NumericalProbe report" in out
    # The probe flags the AdaLN bf16 divergence on Apple Silicon.
    assert "DETECTED" in out or "divergence" in out.lower()


def test_cli_probe_cuda_device(capsys):
    rc = cli_module.main(["probe", "--device", "RTX4090", "--precision", "bf16"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NumericalProbe report" in out


def test_cli_report(capsys):
    rc = cli_module.main(["report"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "status report" in out
    assert "Registered plugins" in out
    assert "Built-in scenarios" in out


def test_cli_runtime_lightx2v(capsys):
    rc = cli_module.main([
        "runtime", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--runtime", "lightx2v",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Governance Report" in out
    assert "lightx2v.infer" in out


def test_cli_runtime_mlx(capsys):
    rc = cli_module.main([
        "runtime", "--device", "M4_Max", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--runtime", "mlx",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "mlx_video_generate" in out


def test_cli_runtime_torch(capsys):
    rc = cli_module.main([
        "runtime", "--device", "M4_Max", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--runtime", "torch",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "TorchRuntime" in out


def test_cli_runtime_comfyui(capsys):
    rc = cli_module.main([
        "runtime", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--runtime", "comfyui",
        "--comfy-checkpoint", "models/checkpoints/ltx.safetensors",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ComfyUI workflow" in out
    assert "CheckpointLoaderSimple" in out


def test_cli_runtime_diffusers(capsys, tmp_path):
    out_file = tmp_path / "bind.py"
    rc = cli_module.main([
        "runtime", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--runtime", "diffusers",
        "--model-id", "Lightricks/LTX-Video", "--out", str(out_file),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "artifact written to" in out
    text = out_file.read_text()
    assert "DiffusersRuntime" in text


def test_cli_runtime_unknown_runtime_rejected(capsys):
    rc = cli_module.main([
        "runtime", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--runtime", "bogus",
    ])
    captured = capsys.readouterr()
    assert rc != 0
    assert "error" in (captured.out + captured.err).lower()
