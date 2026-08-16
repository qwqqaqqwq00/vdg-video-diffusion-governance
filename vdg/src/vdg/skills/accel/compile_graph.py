"""Compile / graph-optimization acceleration skill (torch.compile / TensorRT).

Compile-time graph capture accelerates steady-state inference by fusing kernels,
eliminating Python dispatch overhead, and selecting low-precision tensor-core
kernels. The shared tradeoff is cold-start compile cost versus steady-state
speedup, plus dynamic-shape fragility (shape changes recompile). Model-agnostic;
applied to any server-trained DiT/UNet.

Grounding (video-dit-inference-acceleration-report.md, Section 7 / 8):
  * torch.compile (Inductor): FLUX.1-Dev ~1.5x (6.7s -> 4.5s, H100, bf16, 28
    steps), no quality regression; ComfyUI-KJNodes video models 20-40%, VAE
    15-25%, standard SD 10-30%. Region compile ('compile_repeated_blocks') cuts
    cold start 67.4s -> 9.6s cold / 2.4s warm (7x cheaper), same 1.5x steady.
    max-autotune picks best kernel impl. Community band 1.5-2x.
  * CUDA Graphs ('reduce-overhead'): subsumed into the 20-40% video figure;
    bit-exact (no quality change); needs static shapes.
  * TensorRT (FP8/BF16 engine): Adobe Firefly video (Hopper H100, TRT + FP8)
    60% latency cut, ~40% TCO cut, backbone up to 2.5x faster than PyTorch
    baseline (SDPA is the profiling bottleneck). No FID/FVD/VBench reported.
  * Apple Core ML / MLX mx.compile are MPS-native paths, not modeled here.

VDG model (config key 'backend' in {torch_compile, trt}):
  * torch_compile: speedup 1.5, quality_delta 0.0, energy_ratio 1.0.
  * trt: speedup 2.5, quality_delta -0.5 (FP8 calibration), energy_ratio 0.9.
  * memory_ratio 1.0 (compile does not materially change footprint).
  * applies_to [consumer_nv, edge_npu] (CUDA / Inductor stack; Jetson runs
    CUDA). Apple Silicon uses Core ML / MLX instead (out of this skill's scope).
"""
from __future__ import annotations

from typing import Any

from ...core.contracts import DeviceCategory, DeviceProfile, LoadModel, Skill, SkillImpact
from ...core.registry import register_skill
from ._common import runtime_envelope

__all__ = ["CompileGraph"]

# Per-backend impact.
_BACKENDS: dict[str, dict[str, Any]] = {
    "torch_compile": {
        "speedup": 1.5,
        "quality_delta": 0.0,
        "energy_ratio": 1.0,
        "runtime": "diffusers",
        "notes": "torch.compile (Inductor): FLUX 1.5x, video 20-40% "
                 "(ComfyUI-KJNodes); region compile cold 9.6s/2.4s warm; no "
                 "quality regression.",
    },
    "trt": {
        "speedup": 2.5,
        "quality_delta": -0.5,
        "energy_ratio": 0.9,
        "runtime": "tensorrt",
        "notes": "TensorRT FP8 engine: Adobe Firefly video 60% latency cut, "
                 "backbone up to 2.5x (SDPA bottleneck); FP8 calibration; no "
                 "FID/FVD reported.",
    },
}

_APPLIES_TO = [DeviceCategory.CONSUMER_NV, DeviceCategory.EDGE_NPU]


@register_skill("compile_graph")
class CompileGraph(Skill):
    """Graph compilation (torch.compile / TensorRT). Consumer NV + edge NPU."""

    kind = "accel"

    def applicable(self, device: DeviceProfile, load: LoadModel) -> bool:
        # Inductor / TRT run on the CUDA stack (consumer NV + Jetson edge NPU).
        # Apple Silicon uses Core ML / MLX, not modeled here.
        return device.spec().category in _APPLIES_TO

    def default_config(self) -> dict[str, Any]:
        return {"backend": "torch_compile", "mode": "default"}

    def _backend(self, config: dict[str, Any]) -> str:
        backend = str(config.get("backend", "torch_compile")).lower()
        # Tolerate aliases.
        if backend in ("inductor", "torch", "compile"):
            backend = "torch_compile"
        elif backend in ("tensorrt", "trt"):
            backend = "trt"
        return backend if backend in _BACKENDS else "torch_compile"

    def predict(
        self,
        device: DeviceProfile,
        load: LoadModel,
        config: dict[str, Any] | None = None,
    ) -> SkillImpact:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        backend = self._backend(cfg)
        b = _BACKENDS[backend]
        return SkillImpact(
            speedup=float(b["speedup"]),
            memory_ratio=1.0,
            quality_delta=float(b["quality_delta"]),
            energy_ratio=float(b["energy_ratio"]),
            applies_to=list(_APPLIES_TO),
            notes=str(b["notes"]),
        )

    def apply(self, model_or_pipeline: Any, config: dict[str, Any] | None = None) -> Any:
        cfg = self.default_config()
        if config:
            cfg.update(config)
        backend = self._backend(cfg)
        mode = str(cfg.get("mode", "default"))
        runtime_cfg: dict[str, Any] = {
            "backend": backend,
            "enable": True,
            "fullgraph": False,
            "dynamic": True,
        }
        if backend == "torch_compile":
            runtime_cfg["mode"] = mode
            runtime_cfg["compile_repeated_blocks"] = True  # region compile
        else:  # trt
            runtime_cfg["precision"] = "fp8"
            runtime_cfg["opt_profiles"] = True
        applied = False
        hook = getattr(model_or_pipeline, "enable_torch_compile", None)
        if callable(hook):
            try:
                hook(mode=mode)
                applied = True
            except Exception:
                applied = False
        return runtime_envelope(
            skill="compile_graph",
            runtime=str(_BACKENDS[backend]["runtime"]),
            config=runtime_cfg,
            applied=applied,
            notes=_BACKENDS[backend]["notes"]
                  + " Use dynamic=True for variable video time-dim; LoRA hot-swap "
                  "triggers recompile (pre-declare max rank). Stub: runtime "
                  "applies config.",
        )
