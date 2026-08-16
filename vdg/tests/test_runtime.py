"""Tests for the runtime bindings package (vdg.runtime).

Covers the RuntimeEnvelope upgrade, the real torch patch application
(TorchRuntime.apply / apply_all / unpatch / find_sensitive_modules), the
diffusers runtime bindings, and the ComfyUI / LightX2V / MLX emitters.

Torch-dependent behavior (in-process patching) is gated behind a torch
availability skip so the suite stays host-independent; the emitters and
envelope are pure Python and always run.
"""
from __future__ import annotations

import json

import pytest

from vdg import REGISTRY, Scenario
from vdg.core.contracts import GovernanceDecision, SkillImpact
from vdg.runtime.envelope import RuntimeEnvelope
from vdg.runtime.torch_runtime import TorchRuntime


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# RuntimeEnvelope
# ---------------------------------------------------------------------------
def test_envelope_from_legacy_stub_dict():
    # Exactly the dict shape the accel skills' runtime_envelope() emits.
    env = RuntimeEnvelope.from_dict({
        "skill": "teacache",
        "runtime": "comfyui",
        "config": {"rel_l1_thresh": 0.2, "start_step": 0, "end_step": None},
        "applied": False,
        "notes": "stub",
    })
    assert env.kind == "config"          # applied=False -> config payload
    assert env.target_runtime == "comfyui"
    assert env.validate() == []


def test_envelope_applied_dict_kind_is_patch():
    env = RuntimeEnvelope.from_dict({
        "skill": "gelu_fp32", "runtime": "torch", "config": {},
        "applied": True, "notes": "",
    })
    assert env.kind == "patch"


def test_envelope_validate_detects_missing_required_keys():
    env = RuntimeEnvelope.from_dict({
        "skill": "vae_tiling", "runtime": "comfyui", "config": {},
        "applied": False, "notes": "",
    })
    problems = env.validate()
    assert any("tile_size" in p for p in problems)


def test_envelope_validate_rejects_bad_runtime_and_kind():
    env = RuntimeEnvelope(
        skill="teacache", runtime="comfyui", config={"rel_l1_thresh": 0.1},
        kind="bogus", target_runtime="nonsense",
    )
    problems = env.validate()
    assert any("kind" in p and "bogus" in p for p in problems)
    assert any("target_runtime" in p and "nonsense" in p for p in problems)


def test_envelope_validate_or_raise():
    good = RuntimeEnvelope.from_dict({
        "skill": "teacache", "runtime": "comfyui",
        "config": {"rel_l1_thresh": 0.1}, "applied": False, "notes": "",
    })
    good.validate_or_raise()  # must not raise
    bad = RuntimeEnvelope.from_dict({
        "skill": "teacache", "runtime": "comfyui", "config": {},
        "applied": False, "notes": "",
    })
    with pytest.raises(ValueError):
        bad.validate_or_raise()


def test_envelope_roundtrip_to_dict():
    env = RuntimeEnvelope.from_dict({
        "skill": "teacache", "runtime": "comfyui",
        "config": {"rel_l1_thresh": 0.15}, "applied": False, "notes": "n",
    })
    out = env.to_dict()
    assert out["skill"] == "teacache"
    assert out["config"]["rel_l1_thresh"] == 0.15
    assert out["kind"] == "config"
    assert out["target_runtime"] == "comfyui"


# ---------------------------------------------------------------------------
# TorchRuntime -- real in-process patching
# ---------------------------------------------------------------------------
def _build_mini_dit():
    """A small DiT-shaped module carrying the sensitive sites."""
    import torch
    import torch.nn as nn

    class FakeAdaLN(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale_shift_table = nn.Parameter(torch.zeros(1))
            self.attn1 = nn.Linear(8, 8)
            self.ff = nn.Linear(8, 8)

        def forward(self, x, **kwargs):
            return self.attn1(x) + self.ff(x)

    class MiniDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.gelu1 = nn.GELU(approximate="tanh")
            self.norm1 = nn.LayerNorm(8)
            self.adaln_block = FakeAdaLN()
            self.softmax1 = nn.Softmax(dim=-1)
            self.conv = nn.Conv3d(3, 3, 1)

    return MiniDiT()


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_runtime_finds_sensitive_modules():
    rt = TorchRuntime()
    sites = rt.find_sensitive_modules(_build_mini_dit())
    names = {k: [n for n, _m in v] for k, v in sites.items()}
    assert "gelu1" in names["gelu"]
    assert "adaln_block" in names["adaln"]
    # adaln children must NOT be flagged (leaf-segment matching).
    assert not any(n.startswith("adaln_block.") for n in names["adaln"])
    assert "norm1" in names["rmsnorm"]
    assert "softmax1" in names["softmax"]


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_runtime_apply_and_unpatch_roundtrip():
    rt = TorchRuntime()
    mod = _build_mini_dit()
    results = rt.apply_all(mod, [
        ("gelu_fp32", {}),
        ("adaln_fp32", {}),
        ("rmsnorm_fp32", {}),
        ("softmax_fp32", {}),
        ("vae_fp32", {}),
    ])
    assert len(results) == 5
    assert all(r["applied"] >= 1 for r in results)

    marked = [m for _n, m in mod.named_modules() if getattr(m, "_vdg_patched", None)]
    assert len(marked) == 5

    # The patched model still runs.
    import torch
    out = mod.adaln_block(torch.randn(2, 8))
    assert tuple(out.shape) == (2, 8)

    n = rt.unpatch(mod)
    assert n == 5
    assert all(
        not getattr(m, "_vdg_patched", None) for _n, m in mod.named_modules()
    )


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_runtime_accepts_governance_decision_objects():
    import torch.nn as nn
    rt = TorchRuntime()
    decision = GovernanceDecision(
        skill_name="gelu_fp32", config={"approximate": "tanh"},
        predicted_impact=SkillImpact(), rationale="test",
    )
    mod = nn.Sequential(nn.GELU())
    results = rt.apply_all(mod, [decision])
    assert results[0]["applied"] == 1
    assert results[0]["targets"] == ["0"]


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_runtime_unknown_skill_reports_not_raises():
    rt = TorchRuntime()
    mod = _build_mini_dit()
    res = rt.apply(mod, "not_a_repair_skill", {})
    assert res["applied"] == 0
    assert "no torch patch site" in res["notes"]


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_runtime_leaf_module_apply():
    import torch.nn as nn
    rt = TorchRuntime()
    gelu = nn.GELU(approximate="tanh")
    res = rt.apply(gelu, "gelu_fp32", {})
    assert res["targets"] == ["self"]
    assert res["applied"] == 1
    assert gelu._vdg_patched == "gelu_fp32"
    rt.unpatch(gelu)
    assert not hasattr(gelu, "_vdg_patched")


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_runtime_apply_is_idempotent():
    """Double-apply must not corrupt unpatch: the original forward is restored."""
    import torch
    import torch.nn as nn
    rt = TorchRuntime()
    gelu = nn.GELU(approximate="tanh")
    orig = gelu.forward

    first = rt.apply(gelu, "gelu_fp32", {})
    assert first["applied"] == 1
    second = rt.apply(gelu, "gelu_fp32", {})
    assert second["applied"] == 0  # already patched, skipped
    assert "skipped" in second["notes"]

    rt.unpatch(gelu)
    # Descriptor access creates a fresh bound-method object, so compare by
    # equality + behavior, not identity.
    assert gelu.forward == orig
    x = torch.randn(4)
    assert torch.allclose(gelu(x), orig(x))


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_runtime_vae_noop_reports_not_applied():
    """A module with no callable decode/forward is a no-op, not a patch."""
    class Plain:
        pass
    rt = TorchRuntime()
    res = rt.apply(Plain(), "vae_fp32", {})
    assert res["applied"] == 0
    assert "nothing patched" in res["notes"]


def test_torch_runtime_error_message_without_torch(monkeypatch):
    """The guard raises the canonical message when torch is unavailable."""
    import builtins
    import vdg.runtime.torch_runtime as tr

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError) as excinfo:
        tr._require_torch()
    assert "torch required for runtime application" in str(excinfo.value)


# ---------------------------------------------------------------------------
# DiffusersRuntime
# ---------------------------------------------------------------------------
def test_diffusers_runtime_build_pipeline_graceful_without_diffusers():
    from vdg.runtime.diffusers_runtime import DiffusersRuntime
    rt = DiffusersRuntime()
    pipe = rt.build_pipeline("Lightricks/LTX-Video")
    assert pipe is None
    assert rt.last_error is not None


def test_diffusers_runtime_apply_repairs_and_accel():
    from vdg.runtime.diffusers_runtime import DiffusersRuntime
    if not _has_torch():
        pytest.skip("torch not installed")
    import torch.nn as nn

    class FakeVae(nn.Module):
        def decode(self, x):
            return x

        def enable_tiling(self, **kwargs):
            return None

        def enable_temporal_tiling(self):
            return None

    class FakePipe:
        def __init__(self):
            self.transformer = nn.Sequential(nn.LayerNorm(8), nn.GELU())
            self.vae = FakeVae()

        def enable_teacache(self, threshold):
            self._tea = threshold

        def enable_model_cpu_offload(self):
            self._offloaded = True

        def set_num_inference_steps(self, n):
            self._steps = n

    rt = DiffusersRuntime()
    pipe = FakePipe()
    repairs = rt.apply_repairs(pipe, [
        ("gelu_fp32", {}),
        ("vae_fp32", {}),
        ("teacache", {"threshold": 0.1}),
    ])
    assert repairs["applied"] == 2
    assert repairs["skipped"] == ["teacache"]
    # Per-module summaries accumulate across skills (not last-wins).
    assert repairs["transformer"]["applied"] == 1
    assert len(repairs["transformer"]["results"]) == 1
    assert repairs["vae"]["applied"] == 1

    tea = rt.apply_accel(pipe, "teacache", {"rel_l1_thresh": 0.2})
    assert tea["applied"] is True
    assert pipe._tea == 0.2

    tile = rt.apply_accel(pipe, "vae_tiling", {})
    assert tile["applied"] is True

    off = rt.apply_accel(pipe, "offload", {})
    assert off["applied"] is True

    steps = rt.apply_accel(pipe, "step_distill", {"steps": 4})
    assert steps["applied"] is True
    assert pipe._steps == 4

    quant = rt.apply_accel(pipe, "quantization", {"method": "gguf_q4"})
    assert quant["applied"] is False

    compile_res = rt.apply_accel(pipe, "compile_graph", {"backend": "torch_compile"})
    assert compile_res["applied"] is True


# ---------------------------------------------------------------------------
# ComfyUI emitter
# ---------------------------------------------------------------------------
def _scenario() -> Scenario:
    return Scenario(
        name="test_480p", task="t2v", resolution=(854, 480), frames=81, fps=16,
        steps=30, quality_target=85.0, energy_budget_j=20000.0,
        latency_slo_s=60.0,
    )


def _decisions() -> list[GovernanceDecision]:
    def d(skill, config):
        return GovernanceDecision(
            skill_name=skill, config=config,
            predicted_impact=SkillImpact(), rationale="test",
        )
    return [
        d("teacache", {"threshold": 0.2, "rel_l1_thresh": 0.2}),
        d("step_distill", {"steps": 4}),
        d("vae_tiling", {"tile_size": 512, "overlap": 64,
          "temporal_size": 64, "temporal_overlap": 8}),
        d("sage_attention", {"version": "v2"}),
        d("gelu_fp32", {}),
    ]


def test_comfyui_workflow_is_valid_api_format():
    from vdg.runtime.comfyui_emitter import build_workflow
    wf = build_workflow(_decisions(), "LTX_2_3", _scenario(), {})
    nodes = wf["nodes"]
    # Keys are "1", "2", ... with class_type + inputs (ComfyUI /prompt format).
    assert sorted(nodes, key=int) == [str(i) for i in range(1, len(nodes) + 1)]
    for node in nodes.values():
        assert "class_type" in node
        assert "inputs" in node
    # TeaCache node with rel_l1_thresh feeding the sampler.
    tea = next(n for n in nodes.values() if n["class_type"] == "TeaCache")
    assert tea["inputs"]["rel_l1_thresh"] == 0.2
    sampler = next(n for n in nodes.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["steps"] == 4      # step_distill -> KSampler steps
    assert sampler["inputs"]["model"][0] == next(
        k for k, v in nodes.items() if v["class_type"] == "TeaCache"
    )
    # vae_tiling -> VAEDecodeTiled with tile params.
    tiled = next(n for n in nodes.values() if n["class_type"] == "VAEDecodeTiled")
    assert tiled["inputs"]["tile_size"] == 512
    assert "temporal_size" in tiled["inputs"]
    # sage_attention surfaces as a launch-flag note, not a node.
    assert not any(n["class_type"] == "SageAttention" for n in nodes.values())
    assert any("--use-sage-attention" in note for note in wf["notes"])
    # Output node present.
    assert any(n["class_type"] == "VHS_VideoCombine" for n in nodes.values())


def test_comfyui_workflow_gguf_loader():
    from vdg.runtime.comfyui_emitter import build_workflow
    wf = build_workflow(
        [("quantization", {"method": "gguf_q4"})],
        "Wan21_T2V_14B", _scenario(),
        {"unet": "models/diffusion_models/wan14b.gguf"},
    )
    classes = [n["class_type"] for n in wf["nodes"].values()]
    assert "UnetLoaderGGUF" in classes
    assert "VAELoader" in classes
    assert "CheckpointLoaderSimple" not in classes


def test_comfyui_render_markdown_mentions_paste_target_and_nodes():
    from vdg.runtime.comfyui_emitter import build_workflow, render_markdown
    wf = build_workflow(_decisions(), "LTX_2_3", _scenario(), {})
    md = render_markdown(wf)
    assert "/prompt" in md
    assert "KSampler" in md
    # Raw JSON payload section present (whitespace-insensitive check).
    assert '"class_type": "KSampler"' in md


def test_comfyui_render_patch_script_is_executable_python():
    from vdg.runtime.comfyui_emitter import render_patch_script
    script = render_patch_script(_decisions())
    # Must compile as python (no syntax errors), with the repair decisions.
    compile(script, "<patch_script>", "exec")
    assert "TorchRuntime" in script
    assert "gelu_fp32" in script
    assert "apply_all" in script
    # Non-repair decisions are surfaced as comments.
    assert "teacache" in script


def test_comfyui_render_patch_script_executes(monkeypatch):
    """The emitted script runs end-to-end against a fake TorchRuntime."""
    from vdg.runtime.comfyui_emitter import render_patch_script
    import vdg.runtime.torch_runtime as tr_mod

    class FakeRuntime:
        def apply_all(self, model, decisions):
            assert model is ...  # the 'model = ...' placeholder executed
            return [
                {"skill": s, "applied": 1, "targets": ["self"]}
                for s, _c in decisions
            ]

    monkeypatch.setattr(tr_mod, "TorchRuntime", FakeRuntime)
    script = render_patch_script(_decisions())
    ns: dict = {}
    exec(compile(script, "<patch_script>", "exec"), ns)  # must not raise


def test_comfyui_workflow_plain_decode_uses_vaedecode():
    """Without a vae_tiling decision the decode node is VAEDecode (not Tiled)."""
    from vdg.runtime.comfyui_emitter import build_workflow
    decisions = [d for d in _decisions() if d.skill_name != "vae_tiling"]
    wf = build_workflow(decisions, "LTX_2_3", _scenario(), {})
    nodes = wf["nodes"]
    classes = {n["class_type"] for n in nodes.values()}
    assert "VAEDecode" in classes
    assert "VAEDecodeTiled" not in classes
    decode = next(n for n in nodes.values() if n["class_type"] == "VAEDecode")
    assert decode["inputs"]["samples"][0] == next(
        k for k, v in nodes.items() if v["class_type"] == "KSampler"
    )
    # The required class_types (KSampler, VAEDecode, TeaCache) are all present.
    assert {"KSampler", "VAEDecode", "TeaCache"} <= classes


# ---------------------------------------------------------------------------
# LightX2V / MLX emitters
# ---------------------------------------------------------------------------
def test_lightx2v_command_maps_skills():
    from vdg.runtime.lightx2v_emitter import render_command
    cmd = render_command(
        [
            ("step_distill", {"steps": 4}),
            ("quantization", {"method": "nvfp4"}),
            ("sage_attention", {"version": "v2"}),
            ("teacache", {"threshold": 0.2}),
            ("vae_tiling", {}),
            ("offload", {}),
        ],
        "Lightricks/LTX-Video", (854, 480), 81,
    )
    assert "lightx2v.infer" in cmd
    assert "--steps 4" in cmd
    assert "--quant nvfp4" in cmd
    assert "--attn sageattn" in cmd
    assert "--use_teacache" in cmd
    assert "--teacache_threshold 0.2" in cmd
    assert "--vae_tiling" in cmd
    assert "--offload" in cmd


def test_lightx2v_command_empty_decisions():
    from vdg.runtime.lightx2v_emitter import render_command
    cmd = render_command([], "m", (100, 50), 10)
    assert cmd.startswith("python -m lightx2v.infer")
    assert "--width 100" in cmd and "--height 50" in cmd and "--frames 10" in cmd
    assert "--steps" not in cmd


def test_mlx_command_maps_skills():
    from vdg.runtime.mlx_emitter import render_command
    cmd = render_command(
        [
            ("step_distill", {"steps": 4}),
            ("quantization", {"method": "gguf_q4"}),
            ("teacache", {}),
            ("vae_tiling", {}),
        ],
        "mlx-community/LTX-Video-4bit", (854, 480), 81,
    )
    assert "mlx_video_generate" in cmd
    assert "--sampler euler" in cmd
    assert "--quantize 4" in cmd
    assert "--steps 4" in cmd
    assert "--use_teacache" in cmd
    assert "--tiling" in cmd


def test_runtime_registered_in_vdg_subpackages():
    import vdg
    from vdg import runtime as runtime_pkg
    assert hasattr(runtime_pkg, "TorchRuntime")
    assert hasattr(runtime_pkg, "DiffusersRuntime")
    assert hasattr(runtime_pkg, "build_workflow")
    assert hasattr(runtime_pkg, "render_patch_script")
    assert hasattr(runtime_pkg, "render_lightx2v_command")
    assert hasattr(runtime_pkg, "render_mlx_command")
