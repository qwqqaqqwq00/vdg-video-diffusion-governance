"""Shared test fixtures: grounded mini device/load/skill plugins.

These are NOT part of the shipped package (they live under ``tests/``) -- they
exist to prove the foundation contracts work end-to-end with real, report-
grounded numbers. Phase-2 plugins in ``vdg/devices`` etc. will provide the full
set.
"""
from __future__ import annotations

import pytest

from vdg import (
    DeviceCategory,
    DeviceProfile,
    DeviceSpec,
    LoadModel,
    Skill,
    SkillImpact,
    VideoDiTLoad,
)


# --- devices (grounded in the edge-deployment report) ---------------------
class RTX4090(DeviceProfile):
    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="RTX 4090",
            category=DeviceCategory.CONSUMER_NV,
            memory_gb=24.0,
            memory_bandwidth_gbps=1008.0,
            compute_tflops={
                "fp32": 82.6, "bf16": 165.0, "fp16": 165.0,
                "fp8": 330.0, "int8": 660.0,
            },
            tdp_w=450.0,
            idle_power_w=45.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8", "int8"],
            attention_backends=["flash", "sdpa", "sage2", "math"],
            unified_memory=False,
            cost_per_hour_usd=0.5,
        )


class M4Max(DeviceProfile):
    def spec(self) -> DeviceSpec:
        return DeviceSpec(
            name="M4 Max",
            category=DeviceCategory.APPLE_SILICON,
            memory_gb=64.0,
            memory_bandwidth_gbps=546.0,
            # Apple does not publish a clean TFLOPS figure; use a conservative
            # GPU estimate. No FP4 hardware (report: Apple has no FP4 tensor core).
            compute_tflops={"fp32": 27.0, "bf16": 54.0, "fp16": 54.0, "fp8": 108.0},
            tdp_w=480.0,
            idle_power_w=10.0,
            supported_precisions=["fp32", "bf16", "fp16", "fp8"],
            attention_backends=["mlx_sdpa", "math"],
            unified_memory=True,
        )


# --- loads (grounded in the training-side report) -------------------------
class LTX23(LoadModel):
    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="LTX-2.3",
            params_b=2.0,
            vae_compress=(8, 32, 32),
            patch_size=1,
            te_params_b=5.0,
            layers=48,
            hidden_dim=1536,
            heads=24,
            default_steps=30,
            supported_tasks=["t2v", "i2v"],
            vae_params_m=175.0,
            ffn_expansion=4.0,
        )


class Wan14B(LoadModel):
    def characteristics(self) -> VideoDiTLoad:
        return VideoDiTLoad(
            model_name="Wan-2.1-T2V-14B",
            params_b=14.0,
            vae_compress=(4, 8, 8),
            patch_size=2,
            te_params_b=5.0,
            layers=40,
            hidden_dim=5120,
            heads=40,
            default_steps=40,
            supported_tasks=["t2v", "i2v"],
            vae_params_m=450.0,
            ffn_expansion=4.0,
        )


# --- skills (grounded in the acceleration report) -------------------------
class TeaCacheSkill(Skill):
    kind = "accel"

    def applicable(self, device, load) -> bool:
        return device.spec().category in (
            DeviceCategory.CONSUMER_NV, DeviceCategory.APPLE_SILICON,
        )

    def default_config(self) -> dict:
        return {"threshold": 0.1}

    def predict(self, device, load, config=None) -> SkillImpact:
        return SkillImpact(
            speedup=2.0,
            memory_ratio=1.0,
            quality_delta=-0.07,
            energy_ratio=1.0,
            applies_to=[DeviceCategory.CONSUMER_NV, DeviceCategory.APPLE_SILICON],
            notes="TeaCache 1.4-4.4x; -0.07% VBench (CVPR 2025).",
        )


class SageAttention2Skill(Skill):
    kind = "accel"

    def applicable(self, device, load) -> bool:
        return device.spec().category == DeviceCategory.CONSUMER_NV

    def predict(self, device, load, config=None) -> SkillImpact:
        return SkillImpact(
            speedup=1.8,
            memory_ratio=1.0,
            quality_delta=-0.1,
            energy_ratio=0.9,
            applies_to=[DeviceCategory.CONSUMER_NV],
            notes="SageAttention2 3x FA2 on 4090; negligible loss.",
        )


class DistillationSkill(Skill):
    kind = "accel"

    def predict(self, device, load, config=None) -> SkillImpact:
        return SkillImpact(
            speedup=6.0,
            memory_ratio=1.0,
            quality_delta=-0.5,
            energy_ratio=1.0,
            applies_to=[],
            notes="CoDMD 50->4 step distillation (~25x); VBench 84.5-84.9.",
        )


class QuantizeFP8Skill(Skill):
    kind = "accel"

    def predict(self, device, load, config=None) -> SkillImpact:
        return SkillImpact(
            speedup=1.2,
            memory_ratio=0.5,
            quality_delta=-2.0,
            energy_ratio=0.85,
            applies_to=[DeviceCategory.CONSUMER_NV],
            notes="FP8 weights: ~half memory, small speedup.",
        )


# --- pytest fixtures ------------------------------------------------------
@pytest.fixture
def rtx4090() -> RTX4090:
    return RTX4090()


@pytest.fixture
def m4max() -> M4Max:
    return M4Max()


@pytest.fixture
def ltx23() -> LTX23:
    return LTX23()


@pytest.fixture
def wan14b() -> Wan14B:
    return Wan14B()


@pytest.fixture
def teacache() -> TeaCacheSkill:
    return TeaCacheSkill()


@pytest.fixture
def sage2() -> SageAttention2Skill:
    return SageAttention2Skill()
