"""CLI tests for the runtime + calibration layers (vdg runtime / vdg calibrate).

Covers the two subcommands added for the runtime phase:

* ``vdg runtime --runtime comfyui`` -- runs the governance pipeline, then
  renders the decisions into a ComfyUI /prompt API-format workflow whose JSON
  payload is embedded in the markdown output. The JSON must parse and be a
  valid API-format prompt (contiguous "1".."N" node keys, class_type +
  inputs on every node) containing the expected node classes.

* ``vdg calibrate`` -- runs the anchor-calibrated simulation and prints the
  uncalibrated (roofline) prediction, the matched measured anchor, the applied
  scale, the calibrated prediction, and the predicted-vs-measured table.

These are pure-sim commands: the governance pipeline defaults to the
SIMULATED NumericalProbe path (sim_probe=True), so no torch / MPS / CUDA host
is required, and calibration is pure roofline + anchors.
"""
from __future__ import annotations

import json

import pytest

import vdg.cli as cli_module


# ---------------------------------------------------------------------------
# vdg runtime --runtime comfyui -> JSON
# ---------------------------------------------------------------------------
def _extract_json_payload(out: str) -> dict:
    """Return the API-format JSON dict embedded after the markdown marker."""
    marker = "## JSON payload (paste into /prompt)"
    assert marker in out, "markdown must embed the JSON payload section"
    payload = out.split(marker, 1)[1].strip()
    return json.loads(payload)


def test_cli_runtime_comfyui_emits_valid_api_json(capsys):
    rc = cli_module.main([
        "runtime", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--runtime", "comfyui",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    nodes = _extract_json_payload(out)
    # ComfyUI /prompt API format: contiguous "1".."N" keys with class_type +
    # inputs on every node.
    assert sorted(nodes, key=int) == [str(i) for i in range(1, len(nodes) + 1)]
    for node in nodes.values():
        assert "class_type" in node
        assert "inputs" in node
    classes = {n["class_type"] for n in nodes.values()}
    assert "KSampler" in classes
    assert "VAEDecode" in classes
    assert "TeaCache" in classes


def test_cli_runtime_comfyui_json_wires_teacache_into_sampler(capsys):
    rc = cli_module.main([
        "runtime", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--runtime", "comfyui",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    nodes = _extract_json_payload(out)
    tea = next(nid for nid, n in nodes.items() if n["class_type"] == "TeaCache")
    sampler = next(nid for nid, n in nodes.items() if n["class_type"] == "KSampler")
    assert nodes[sampler]["inputs"]["model"][0] == tea
    assert nodes[sampler]["inputs"]["seed"] == 42


def test_cli_runtime_comfyui_out_file_contains_json(capsys, tmp_path):
    out_file = tmp_path / "workflow.md"
    rc = cli_module.main([
        "runtime", "--device", "RTX4090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f", "--runtime", "comfyui",
        "--out", str(out_file),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "artifact written to" in out
    nodes = _extract_json_payload(out_file.read_text())
    assert any(n["class_type"] == "KSampler" for n in nodes.values())


# ---------------------------------------------------------------------------
# vdg calibrate -> prediction + measured anchor
# ---------------------------------------------------------------------------
def test_cli_calibrate_prints_prediction(capsys):
    rc = cli_module.main([
        "calibrate", "--device", "M4_Max", "--model", "Wan21_T2V_1_3B",
        "--scenario", "ltx_t2v_480p_81f",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Calibration for" in out
    assert "base prediction" in out          # roofline, uncalibrated
    assert "calibrated" in out               # scaled prediction
    assert "calibration scale" in out


def test_cli_calibrate_reports_measured_anchor(capsys):
    rc = cli_module.main([
        "calibrate", "--device", "M4_Max", "--model", "Wan21_T2V_1_3B",
        "--scenario", "ltx_t2v_480p_81f",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    # The M4 Max Wan2.1-T2V-1.3B anchor (480p/81f) is matched and cited.
    assert "M4_Max/Wan21_T2V_1_3B" in out
    assert "measured:" in out
    assert "4500.00" in out                 # anchor measured latency (~90 s/it)
    assert "roofline error" in out


def test_cli_calibrate_prints_anchor_table(capsys):
    rc = cli_module.main([
        "calibrate", "--device", "M4_Max", "--model", "Wan21_T2V_1_3B",
        "--scenario", "ltx_t2v_480p_81f",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Calibration report" in out
    assert "measured(s)" in out
    assert "rel.err" in out
    assert "M4_Max/Wan21_T2V_1_3B" in out


def test_cli_calibrate_no_anchor_prints_engineering_estimate(capsys):
    """A (device, load) pair with no anchor degrades to scale 1.0 + warning."""
    rc = cli_module.main([
        "calibrate", "--device", "RTX5090", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "base prediction" in out
    assert "anchor:            none" in out
    assert "scale 1.0" in out
    assert "engineering estimate" in out


def test_cli_calibrate_unknown_device_returns_1(capsys):
    rc = cli_module.main([
        "calibrate", "--device", "nope", "--model", "LTX_2_3",
        "--scenario", "ltx_t2v_480p_81f",
    ])
    assert rc == 1
    assert "error" in capsys.readouterr().out.lower()
