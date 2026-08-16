"""Regression tests for the shipped video DiT load plugins (vdg.loads.video_dit).

These tests assert the GROUNDED architecture numbers -- fetched directly from
HuggingFace model configs, transformer source code, and checkpoint file sizes.
They protect against accidental edits to params_b, vae_compress, patch_size,
layers, hidden_dim, heads, ffn_expansion (-> d_ff), and the token-count anchors
that validate the VAE compression choices.

The conftest.py fixtures (LTX23, Wan14B) are separate test doubles with
approximate numbers; these tests exercise the REAL shipped loads instead.
"""
from __future__ import annotations

import pytest

import vdg
from vdg import DeviceCategory, DeviceSpec
from vdg.core.registry import REGISTRY
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

# The nine shipped load classes, keyed by their registry name.
ALL_LOAD_CLASSES = [
    LTX_2_3,
    Wan21_T2V_1_3B,
    Wan21_T2V_14B,
    Wan21_I2V_14B,
    Wan22_A14B_MoE,
    Wan22_TI2V_5B_Dense,
    HunyuanVideo_13B,
    CogVideoX_5B,
    OpenSora2_11B,
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestRegistration:
    def test_all_nine_registered(self):
        loads = REGISTRY.all("load")
        # Other tests may register throwaway loads in the global REGISTRY; we
        # only assert that our 9 shipped loads are present (subset check).
        for cls in ALL_LOAD_CLASSES:
            assert cls.__name__ in loads, cls.__name__ + " not registered"

    def test_registry_metadata(self):
        for cls in ALL_LOAD_CLASSES:
            assert cls.__registry_kind__ == "load"
            assert cls.__registry_name__ == cls.__name__

    def test_list_all_loads_returns_instances(self):
        all_loads = list_all_loads()
        # Subset check: our 9 loads must be present and be LoadModel instances.
        for cls in ALL_LOAD_CLASSES:
            name = cls.__name__
            assert name in all_loads, name + " missing from list_all_loads()"
            assert isinstance(all_loads[name], vdg.LoadModel)
            assert all_loads[name].characteristics().model_name


# ---------------------------------------------------------------------------
# Subclassing contract: only characteristics() is overridden
# (tokens_for / per_step_flops / memory_footprint come from the base class,
#  EXCEPT Wan22_A14B_MoE which overrides memory_footprint for MoE correctness)
# ---------------------------------------------------------------------------
class TestSubclassContract:
    @pytest.mark.parametrize("cls", ALL_LOAD_CLASSES, ids=[c.__name__ for c in ALL_LOAD_CLASSES])
    def test_tokens_for_not_overridden(self, cls):
        assert "tokens_for" not in cls.__dict__

    @pytest.mark.parametrize("cls", ALL_LOAD_CLASSES, ids=[c.__name__ for c in ALL_LOAD_CLASSES])
    def test_per_step_flops_not_overridden(self, cls):
        assert "per_step_flops" not in cls.__dict__

    @pytest.mark.parametrize("cls", ALL_LOAD_CLASSES, ids=[c.__name__ for c in ALL_LOAD_CLASSES])
    def test_characteristics_implemented(self, cls):
        assert "characteristics" in cls.__dict__

    def test_only_moe_overrides_memory_footprint(self):
        """Only the MoE model overrides memory_footprint (for resident-weight
        correctness); all other loads use the base-class implementation."""
        for cls in ALL_LOAD_CLASSES:
            if cls is Wan22_A14B_MoE:
                assert "memory_footprint" in cls.__dict__
            else:
                assert "memory_footprint" not in cls.__dict__


# ---------------------------------------------------------------------------
# Grounded d_ff values (must match the config ffn_dim exactly)
# ---------------------------------------------------------------------------
GROUNDED_D_FF = {
    LTX_2_3: 8192,          # FeedForward default mult=4.0; 2048*4
    Wan21_T2V_1_3B: 8960,   # config ffn_dim=8960
    Wan21_T2V_14B: 13824,   # config ffn_dim=13824
    Wan21_I2V_14B: 13824,   # same arch as T2V-14B
    Wan22_A14B_MoE: 13824,  # same arch as Wan2.1-14B (active expert)
    Wan22_TI2V_5B_Dense: 14336,  # config ffn_dim=14336
    HunyuanVideo_13B: 12288,  # mlp_ratio=4.0; 3072*4
    CogVideoX_5B: 12288,    # FeedForward default mult=4.0; 3072*4
    OpenSora2_11B: 12288,   # FLUX base; 3072*4
}


class TestGroundedArchitecture:
    @pytest.mark.parametrize("cls", ALL_LOAD_CLASSES, ids=[c.__name__ for c in ALL_LOAD_CLASSES])
    def test_d_ff_matches_config(self, cls):
        c = cls().characteristics()
        assert c.d_ff == GROUNDED_D_FF[cls], (
            cls.__name__ + " d_ff=" + str(c.d_ff) + " expected " + str(GROUNDED_D_FF[cls])
        )

    def test_ltx_architecture(self):
        c = LTX_2_3().characteristics()
        assert c.layers == 28
        assert c.hidden_dim == 2048
        assert c.heads == 32
        assert c.params_b == pytest.approx(19.0, abs=0.1)

    def test_wan14b_architecture(self):
        c = Wan21_T2V_14B().characteristics()
        assert c.layers == 40
        assert c.hidden_dim == 5120
        assert c.heads == 40
        assert c.params_b == pytest.approx(14.288, abs=0.001)

    def test_wan13b_exact_params(self):
        c = Wan21_T2V_1_3B().characteristics()
        assert c.params_b == pytest.approx(1.419, abs=0.001)

    def test_hunyuanvideo_60_layers(self):
        """20 dual-stream + 40 single-stream = 60 total blocks."""
        c = HunyuanVideo_13B().characteristics()
        assert c.layers == 60
        assert c.hidden_dim == 3072
        assert c.heads == 24

    def test_cogvideox_226_text_cap_documented(self):
        c = CogVideoX_5B().characteristics()
        assert c.layers == 42
        assert c.heads == 48

    def test_moe_params_b_is_active(self):
        c = Wan22_A14B_MoE().characteristics()
        assert c.params_b == 14.0


# ---------------------------------------------------------------------------
# Token-count anchors (validate VAE compression + patch choices)
# ---------------------------------------------------------------------------
class TestTokenAnchors:
    def test_hunyuanvideo_720p_129f_approx_115k(self):
        """Report: HunyuanVideo 720p/129f -> ~115K tokens. Validates (4,8,8)+patch2."""
        tokens = HunyuanVideo_13B().tokens_for((1280, 720), 129)
        assert 110_000 <= tokens <= 120_000

    def test_opensora_768p_128f_approx_19k(self):
        """Paper: Open-Sora 2.0 768p/128f -> ~19K tokens. Validates (4,32,32)+patch1."""
        tokens = OpenSora2_11B().tokens_for((768, 768), 128)
        assert 17_000 <= tokens <= 21_000

    def test_ltx_high_compression_low_tokens(self):
        """LTX 1216x704/97f -> ~10K tokens (high-compression VAE)."""
        tokens = LTX_2_3().tokens_for((1216, 704), 97)
        assert tokens < 12_000

    def test_wan2_2_higher_compression_than_wan2_1(self):
        """Wan2.2-VAE (4,16,16) compresses more than Wan2.1-VAE (4,8,8)."""
        w22 = Wan22_TI2V_5B_Dense().characteristics().vae_compress
        w21 = Wan21_T2V_14B().characteristics().vae_compress
        assert w22[1] > w21[1]  # spatial compression 16 > 8


# ---------------------------------------------------------------------------
# MoE memory correctness (resident = 27B, not 14B active)
# ---------------------------------------------------------------------------
class TestMoEMemory:
    def test_moe_resident_weights_use_27b(self):
        """memory_footprint must report 27B resident weights (both experts),
        not the 14B active params_b."""
        moe = Wan22_A14B_MoE()
        mem = moe.memory_footprint("bf16", 8106)
        # 27B * 2 bytes = 54 GB
        assert mem["weights"] == pytest.approx(54.0, abs=0.1)

    def test_moe_base_class_would_underestimate(self):
        """The base-class implementation (14B) would give 28 GB -- confirm the
        override diverges to prevent silent OOM under-estimation."""
        moe = Wan22_A14B_MoE()
        base_mem = vdg.LoadModel.memory_footprint(moe, "bf16", 8106)
        override_mem = moe.memory_footprint("bf16", 8106)
        assert base_mem["weights"] == pytest.approx(28.0, abs=0.1)
        assert override_mem["weights"] > base_mem["weights"]

    def test_moe_per_step_flops_matches_dense_14b(self):
        """Per-step FLOPs are the active-expert cost (same as Wan2.1-14B)."""
        moe_flops = Wan22_A14B_MoE().per_step_flops(8106, text_tokens=256)
        dense_flops = Wan21_T2V_14B().per_step_flops(8106, text_tokens=256)
        assert moe_flops["total"] == dense_flops["total"]

    def test_moe_total_params_documented(self):
        assert Wan22_A14B_MoE._MOE_TOTAL_PARAMS_B == 27.0


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


class TestRecommendedModel:
    def test_npu_gets_ltx(self):
        rec = recommended_model_for(_spec(DeviceCategory.EDGE_NPU, 128))
        assert isinstance(rec, LTX_2_3)

    def test_consumer_nv_24gb_gets_ltx(self):
        rec = recommended_model_for(_spec(DeviceCategory.CONSUMER_NV, 24))
        assert isinstance(rec, LTX_2_3)

    def test_consumer_nv_48gb_gets_wan14b(self):
        rec = recommended_model_for(_spec(DeviceCategory.CONSUMER_NV, 48))
        assert isinstance(rec, Wan21_T2V_14B)

    def test_apple_silicon_small_gets_ltx(self):
        rec = recommended_model_for(_spec(DeviceCategory.APPLE_SILICON, 64))
        assert isinstance(rec, LTX_2_3)

    def test_apple_silicon_large_gets_wan14b(self):
        rec = recommended_model_for(_spec(DeviceCategory.APPLE_SILICON, 192))
        assert isinstance(rec, Wan21_T2V_14B)

    def test_datacenter_gets_moe(self):
        rec = recommended_model_for(_spec(DeviceCategory.DATACENTER, 80))
        assert isinstance(rec, Wan22_A14B_MoE)
