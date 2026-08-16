"""Tests for the shipped video DiT load plugins (vdg.loads).

LTX-2.3 is the primary modeled load. These tests verify:
  * LTX_2_3 is registered and is the hero architecture (28 layers, 2048 dim).
  * tokens_for matches the roofline token_count formula exactly.
  * memory_footprint is monotonic in precision (more bytes/element -> more GB).
  * the Wan 2.2 A14B MoE is documented: 27B resident / 14B active, and its
    memory_footprint override diverges from the base class to prevent silent
    OOM under-estimation.

Counts use subset checks because test_registry.py registers a throwaway load
into the global REGISTRY at collection time.
"""
from __future__ import annotations

import pytest

import vdg
from vdg import DeviceCategory, DeviceSpec, LoadModel
from vdg.core.registry import REGISTRY
from vdg.core.roofline import GB, bytes_per_element, token_count
from vdg.loads import (
    CogVideoX_5B,
    HunyuanVideo_13B,
    LTX_2_3,
    OpenSora2_11B,
    Wan21_I2V_14B,
    Wan21_T2V_1_3B,
    Wan21_T2V_14B,
    Wan22_A14B_MoE,
    Wan22_TI2V_5B_Dense,
    list_all_loads,
    recommended_model_for,
)

ALL_LOADS = [
    LTX_2_3, Wan21_T2V_1_3B, Wan21_T2V_14B, Wan21_I2V_14B, Wan22_A14B_MoE,
    Wan22_TI2V_5B_Dense, HunyuanVideo_13B, CogVideoX_5B, OpenSora2_11B,
]


# ---------------------------------------------------------------------------
# Registration + LTX-2.3 as primary
# ---------------------------------------------------------------------------
def test_all_nine_loads_registered():
    names = set(REGISTRY.names("load"))
    for cls in ALL_LOADS:
        assert cls.__name__ in names, cls.__name__ + " not registered"


def test_list_all_loads_count():
    shipped = {n: l for n, l in list_all_loads().items() if n in {c.__name__ for c in ALL_LOADS}}
    assert len(shipped) == 9


def test_ltx_2_3_is_primary():
    """LTX-2.3 is the hero load: registered, compact 2B, high-compression VAE."""
    assert "LTX_2_3" in REGISTRY.names("load")
    c = LTX_2_3().characteristics()
    assert c.model_name == "LTX-2.3"
    assert c.params_b == pytest.approx(19.0, abs=0.1)
    assert c.layers == 28
    assert c.hidden_dim == 2048
    assert c.heads == 32
    assert c.patch_size == 1
    # High-compression 3D VAE (8x temporal, 32x32 spatial = 8192x total).
    assert c.vae_compress == (8, 32, 32)
    assert c.d_ff == 8192
    assert "i2v" in c.supported_tasks
    assert "t2v" in c.supported_tasks


# ---------------------------------------------------------------------------
# tokens_for matches the roofline formula
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALL_LOADS, ids=[c.__name__ for c in ALL_LOADS])
def test_tokens_for_matches_formula(cls):
    """tokens_for must equal token_count(frames, H, W, vae_compress, patch)."""
    load = cls()
    c = load.characteristics()
    for resolution, frames in [((854, 480), 81), ((1280, 720), 129), ((768, 768), 128)]:
        width, height = resolution
        expected = token_count(frames, height, width, c.vae_compress, c.patch_size)
        got = load.tokens_for(resolution, frames)
        assert got == expected, (
            cls.__name__ + " tokens_for" + str(resolution) + "/" + str(frames)
            + " = " + str(got) + " != formula " + str(expected)
        )


@pytest.mark.parametrize("cls", ALL_LOADS, ids=[c.__name__ for c in ALL_LOADS])
def test_tokens_for_not_overridden(cls):
    """Subclasses rely on the base-class tokens_for (only characteristics is set)."""
    assert "tokens_for" not in cls.__dict__


def test_ltx_high_compression_low_tokens():
    """LTX-2.3 1216x704/97f yields very few tokens (high-compression VAE)."""
    tokens = LTX_2_3().tokens_for((1216, 704), 97)
    assert tokens < 12_000


def test_hunyuanvideo_720p_token_anchor():
    """HunyuanVideo 720p/129f -> ~115K tokens (validates (4,8,8)+patch2)."""
    tokens = HunyuanVideo_13B().tokens_for((1280, 720), 129)
    assert 110_000 <= tokens <= 120_000


# ---------------------------------------------------------------------------
# memory_footprint monotonic in precision
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALL_LOADS, ids=[c.__name__ for c in ALL_LOADS])
def test_memory_footprint_monotonic_in_precision(cls):
    """More bytes/element -> more memory: fp32 > bf16 > fp8 > fp4."""
    load = cls()
    tokens = load.tokens_for((854, 480), 81)
    gb = {p: load.memory_footprint(p, tokens)["total_gb"] for p in ("fp32", "bf16", "fp8", "fp4")}
    assert gb["fp32"] > gb["bf16"], cls.__name__ + " fp32 <= bf16"
    assert gb["bf16"] > gb["fp8"], cls.__name__ + " bf16 <= fp8"
    assert gb["fp8"] > gb["fp4"], cls.__name__ + " fp8 <= fp4"
    # And the exact byte ratio holds for weights. The MoE overrides
    # memory_footprint to use the full resident param count (27B, both
    # experts) rather than params_b (14B active); account for that.
    c = load.characteristics()
    total_params_b = getattr(cls, "_MOE_TOTAL_PARAMS_B", None)
    if total_params_b is None:
        total_params_b = c.params_b
    assert load.memory_footprint("fp32", tokens)["weights"] == pytest.approx(
        total_params_b * 1e9 * bytes_per_element("fp32") / GB
    )


@pytest.mark.parametrize("cls", ALL_LOADS, ids=[c.__name__ for c in ALL_LOADS])
def test_memory_footprint_keys(cls):
    mem = cls().memory_footprint("bf16", 4096)
    assert set(mem.keys()) == {"weights", "kv", "activations", "total_gb"}
    assert mem["total_gb"] == pytest.approx(mem["weights"] + mem["kv"] + mem["activations"])


# ---------------------------------------------------------------------------
# Wan 2.2 A14B MoE documentation (27B resident / 14B active)
# ---------------------------------------------------------------------------
def test_wan22_moe_total_params_documented():
    assert Wan22_A14B_MoE._MOE_TOTAL_PARAMS_B == 27.0


def test_wan22_moe_active_params_b():
    """params_b is the per-step ACTIVE count (14B), not the total 27B."""
    c = Wan22_A14B_MoE().characteristics()
    assert c.params_b == 14.0
    assert c.model_name == "Wan2.2-A14B-MoE"


def test_wan22_moe_resident_weights_use_27b():
    """memory_footprint must report 27B resident weights (both experts)."""
    mem = Wan22_A14B_MoE().memory_footprint("bf16", 8106)
    # 27B * 2 bytes = 54 GB.
    assert mem["weights"] == pytest.approx(54.0, abs=0.1)


def test_wan22_moe_override_diverges_from_base():
    """The override must diverge from the base class (14B -> 28 GB) to prevent
    silent OOM under-estimation when both experts are resident."""
    moe = Wan22_A14B_MoE()
    base = vdg.LoadModel.memory_footprint(moe, "bf16", 8106)
    override = moe.memory_footprint("bf16", 8106)
    assert base["weights"] == pytest.approx(28.0, abs=0.1)
    assert override["weights"] > base["weights"]
    # Only the MoE overrides memory_footprint; all others use the base class.
    assert "memory_footprint" in Wan22_A14B_MoE.__dict__


def test_wan22_moe_per_step_flops_matches_dense_14b():
    """Per-step FLOPs are the active-expert cost (same as Wan2.1-14B)."""
    moe = Wan22_A14B_MoE().per_step_flops(8106, text_tokens=256)
    dense = Wan21_T2V_14B().per_step_flops(8106, text_tokens=256)
    assert moe["total"] == dense["total"]


def test_only_moe_overrides_memory_footprint():
    for cls in ALL_LOADS:
        if cls is Wan22_A14B_MoE:
            assert "memory_footprint" in cls.__dict__
        else:
            assert "memory_footprint" not in cls.__dict__


# ---------------------------------------------------------------------------
# recommended_model_for selection logic
# ---------------------------------------------------------------------------
def _spec(category, mem_gb):
    return DeviceSpec(
        name="test", category=category, memory_gb=mem_gb,
        memory_bandwidth_gbps=500, compute_tflops={"bf16": 100},
        tdp_w=200, idle_power_w=10, supported_precisions=["bf16"],
        attention_backends=["flash"],
    )


def test_recommended_model_for_edge_npu_gets_ltx():
    assert isinstance(recommended_model_for(_spec(DeviceCategory.EDGE_NPU, 128)), LTX_2_3)


def test_recommended_model_for_consumer_nv_24gb_gets_ltx():
    assert isinstance(recommended_model_for(_spec(DeviceCategory.CONSUMER_NV, 24)), LTX_2_3)


def test_recommended_model_for_consumer_nv_48gb_gets_wan14b():
    assert isinstance(recommended_model_for(_spec(DeviceCategory.CONSUMER_NV, 48)), Wan21_T2V_14B)


def test_recommended_model_for_datacenter_gets_moe():
    assert isinstance(recommended_model_for(_spec(DeviceCategory.DATACENTER, 80)), Wan22_A14B_MoE)


def test_recommended_model_for_apple_silicon_large_gets_wan14b():
    assert isinstance(recommended_model_for(_spec(DeviceCategory.APPLE_SILICON, 192)), Wan21_T2V_14B)


# ---------------------------------------------------------------------------
# per_step_flops sanity (base-class method, not overridden)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cls", ALL_LOADS, ids=[c.__name__ for c in ALL_LOADS])
def test_per_step_flops_not_overridden(cls):
    assert "per_step_flops" not in cls.__dict__


def test_ltx_per_step_flops_breakdown():
    flops = LTX_2_3().per_step_flops(4096, text_tokens=256)
    assert flops["attention"] > 0
    assert flops["ffn"] > 0
    assert flops["total"] == flops["attention"] + flops["ffn"]
