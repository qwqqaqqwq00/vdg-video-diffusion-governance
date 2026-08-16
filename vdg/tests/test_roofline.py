"""Roofline + FLOP model tests."""
from __future__ import annotations

import math

import pytest

from vdg.core.roofline import (
    GB,
    attention_flops,
    bytes_per_element,
    ffn_flops,
    operational_intensity,
    per_step_flops,
    predict_step_time,
    roofline,
    text_encoder_flops,
    token_count,
    vae_decode_flops,
)


def test_roofline_compute_bound():
    # Very high arithmetic intensity -> limited by peak.
    assert roofline(1e6, 1e14, 1e12) == 1e14


def test_roofline_memory_bound():
    # ai=1 < peak/bw=100 -> memory bound; time = bytes/bw = 1e12/1e12 = 1.0.
    assert roofline(1.0, 1e14, 1e12) == 1e12


def test_roofline_boundary_equal():
    # At the ridge point ai*mem_bw == peak.
    assert roofline(100.0, 1e14, 1e12) == 1e14


def test_roofline_invalid_inputs():
    with pytest.raises(ValueError):
        roofline(-1, 1e14, 1e12)
    with pytest.raises(ValueError):
        roofline(1, 0, 1e12)
    with pytest.raises(ValueError):
        roofline(1, 1e14, 0)


def test_bytes_per_element():
    assert bytes_per_element("fp32") == 4.0
    assert bytes_per_element("BF16") == 2.0
    assert bytes_per_element("fp8") == 1.0
    assert bytes_per_element("fp4") == 0.5
    assert bytes_per_element("int8") == 1.0
    assert bytes_per_element("nvfp4") == 0.5
    with pytest.raises(ValueError):
        bytes_per_element("fp7")


def test_token_count_ltx_480p_81f():
    tokens = token_count(81, 480, 854, (8, 32, 32), 1)
    assert tokens == (81 * 480 * 854) // (8 * 32 * 32)
    assert tokens == 4053


def test_token_count_patch_reduces_tokens():
    no_patch = token_count(81, 480, 854, (8, 32, 32), 1)
    patch2 = token_count(81, 480, 854, (8, 32, 32), 2)
    assert patch2 == no_patch // 4


def test_token_count_minimum_one():
    assert token_count(1, 1, 1, (8, 32, 32), 1) == 1


def test_token_count_invalid():
    with pytest.raises(ValueError):
        token_count(0, 480, 854, (8, 32, 32), 1)
    with pytest.raises(ValueError):
        token_count(81, 480, 854, (8, 32, 32), 0)


def test_attention_flops_formula():
    N, d, L = 1000, 512, 2
    expected = (4 * N * N * d + 8 * N * d * d) * L
    assert attention_flops(N, d, L) == expected


def test_attention_flops_with_cross():
    N, d, L, Mt = 1000, 512, 2, 256
    no_cross = attention_flops(N, d, L, text_tokens=0)
    with_cross = attention_flops(N, d, L, text_tokens=Mt)
    assert with_cross - no_cross == 4 * N * Mt * d * L


def test_ffn_flops_formula():
    N, d, dff, L = 1000, 512, 2048, 2
    assert ffn_flops(N, d, dff, L) == 4 * N * d * dff * L


def test_per_step_flops_total_is_sum():
    out = per_step_flops(4053, 1536, 48, 6144, heads=24, text_tokens=256)
    assert out["total"] == out["attention"] + out["ffn"]
    assert out["attention"] > 0 and out["ffn"] > 0


def test_per_step_flops_attention_dominates_for_large_tokens():
    out = per_step_flops(4053, 1536, 48, 6144)
    assert out["attention"] > out["ffn"]


def test_operational_intensity():
    assert operational_intensity(1e12, 1e9) == 1e3
    assert operational_intensity(1e12, 0) == math.inf


def test_predict_step_time_compute_bound():
    # Compute-bound: time ~= flops / peak.
    t = predict_step_time(1e15, 1e14, 1e12, 1e6)
    assert abs(t - 10.0) < 1e-6


def test_predict_step_time_memory_bound():
    # ai=1 (< ridge 100) -> memory bound; time = bytes/bw = 1e12/1e12 = 1.0.
    t = predict_step_time(1e12, 1e14, 1e12, 1e12)
    assert abs(t - 1.0) < 1e-6


def test_vae_decode_flops_geometric_mean():
    frames, H, W, params_m = 81, 480, 854, 175.0
    compress = (8, 32, 32)
    R = 8 * 32 * 32
    eff = (frames * H * W) / (R ** 0.5)
    expected = int(2.0 * params_m * 1e6 * eff * 1.5)
    assert vae_decode_flops(frames, H, W, params_m, compress) == expected


def test_vae_decode_flops_monotonic_in_params():
    a = vae_decode_flops(81, 480, 854, 100.0, (8, 32, 32))
    b = vae_decode_flops(81, 480, 854, 200.0, (8, 32, 32))
    assert b > a


def test_vae_decode_flops_no_compress_uses_output_positions():
    val = vae_decode_flops(81, 480, 854, 175.0, None)
    expected = int(2.0 * 175.0 * 1e6 * (81 * 480 * 854) * 1.5)
    assert val == expected


def test_text_encoder_flops():
    assert text_encoder_flops(5.0, 256) == int(2.0 * 5.0 * 1e9 * 256)
    assert text_encoder_flops(0.0, 256) == 0
    assert text_encoder_flops(5.0, 0) == 0


def test_gb_constant():
    assert GB == 1e9
