"""Numerical divergence probe (the repair diagnostic).

Implements NumericalProbe.probe_ops(device_name, precision), which tests each
sensitive op on boundary inputs known to trigger cross-device divergence,
comparing the cpu-fp32 reference (correct) against the target device/precision.
It returns a DiagnosticReport of OpResult{op, input_desc, max_diff, nan_count,
status}.

CRITICAL: this MUST run in PURE-SIM mode on any machine. If torch is missing or
the target device is unavailable, probe_ops falls back to a SIMULATED report
computed from the known divergence thresholds in the cross-device robustness
report section 8, using a numpy bf16/fp16 emulation so the numbers are computed
rather than hardcoded:

  * GELU-tanh: MPS bf16 fused-kernel bug -> NaN for |x| >= 15 (kernel defect,
    not overflow; CPU bf16 is correct). NOTE: this is PyTorch/Metal-version
    dependent -- verified fixed on M4 Max / torch 2.12 (GELU(15)=15.0, no NaN);
    the simulated path encodes the documented historical threshold so governance
    planning stays worst-case-biased and reproducible. fp16 |x| > 40 overflows 65504.
  * AdaLN (1+scale): |1+scale| < 2^-7 ~= 0.0078 -> catastrophic cancellation;
    scale = -0.999 rounds to -1.0 in bf16 so (1+scale) becomes 0.0.
  * RMSNorm: fp16 square-sum overflows 65504 for |x| > 256 -> zero/NaN norm.
  * Softmax: large-sequence exp underflow / sum precision loss.

Boundary inputs (task-specified, grounded in report section 8 thresholds):
  gelu x=15.0 (bf16), adaln scale=-0.999, rmsnorm |x|=300.0, softmax seq=4096.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["OpResult", "DiagnosticReport", "NumericalProbe"]

# --------------------------------------------------------------------------
# Boundary inputs (report section 8 thresholds; task-specified).
# --------------------------------------------------------------------------
GELU_X: float = 15.0          # bf16 |x|>=15 -> NaN on MPS (kernel defect).
ADALN_SCALE: float = -0.999   # |1+scale|<2^-7 -> catastrophic cancellation.
RMSNORM_ABS: float = 300.0    # fp16 |x|>256 -> x^2 overflow 65504.
SOFTMAX_SEQ: int = 4096       # large-sequence softmax divergence.

# Status classification thresholds.
_OK_MAX_DIFF: float = 1e-2        # normal low-precision noise floor (~bf16 ULP).
_DIVERGENCE_MAX_DIFF: float = 0.5 # clearly wrong output magnitude.
_ADALN_RELATIVE_FACTOR_ERR: float = 0.5  # (1+scale) relative error -> divergence.

# Fixed seed so simulated reports are deterministic and reproducible.
_SEED: int = 20260814


@dataclass(frozen=True)
class OpResult:
    """One sensitive-op probe outcome.

    * max_diff: max |cpu_fp32_ref - target_result| over the probed tensor. For
      the adaln op this is the (1+scale) factor absolute error (small in
      absolute terms because the factor is near zero; see status / input_desc).
      NaN-producing ops report float("inf").
    * nan_count: number of NaN elements in the target result.
    * status: "ok" | "divergence" | "nan".
    """

    op: str
    input_desc: str
    max_diff: float
    nan_count: int
    status: str


@dataclass
class DiagnosticReport:
    """Aggregate probe report for one (device, precision) pair."""

    device_name: str
    precision: str
    simulated: bool
    results: list[OpResult] = field(default_factory=list)
    summary: str = ""

    @property
    def has_failure(self) -> bool:
        return any(r.status in ("divergence", "nan") for r in self.results)

    @property
    def repair_skills_suggested(self) -> list[str]:
        """Map failing ops to the repair skill names that fix them."""
        mapping = {
            "gelu": "gelu_fp32",
            "gelu_tanh": "gelu_fp32",
            "adln_modulate": "adaln_fp32",
            "rmsnorm": "rmsnorm_fp32",
            "softmax": "softmax_fp32",
            "vae_decode": "vae_fp32",
        }
        suggested: list[str] = []
        for r in self.results:
            skill = mapping.get(r.op)
            if skill and r.status in ("divergence", "nan") and skill not in suggested:
                suggested.append(skill)
        return suggested


# --------------------------------------------------------------------------
# Low-precision emulation (numpy, pure-sim).
# --------------------------------------------------------------------------
def _to_bf16(x: np.ndarray) -> np.ndarray:
    """Emulate bfloat16 by truncating fp32 to its top 16 bits (round-to-nearest-even).

    bf16 = the high 16 bits of fp32 (same 8-bit exponent, 7-bit mantissa). We
    round-to-nearest-even before truncation. numpy has no portable native bf16,
    so this int32 manipulation is the standard emulation.
    """
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    bf16_bits = rounded & np.uint32(0xFFFF0000)
    return bf16_bits.view(np.float32)


def _to_precision(x: np.ndarray, precision: str) -> np.ndarray:
    """Quantize an fp32 array to the target precision (pure-sim)."""
    p = precision.lower()
    if p == "fp32" or p == "tf32":
        return np.asarray(x, dtype=np.float32)
    if p == "fp16":
        return np.asarray(x, dtype=np.float16).astype(np.float32)
    if p == "bf16":
        return _to_bf16(x).astype(np.float32)
    # int8/int4/fp8/nvfp4/fp4: emulate via bf16-grade activation precision (the
    # report keeps activations in bf16 for quantized backends; weights are
    # int8/int4). For the probe, bf16 emulation surfaces the same sensitive-op
    # risks the report section 7 predicts for these backends.
    return _to_bf16(x).astype(np.float32)


def _gelu_tanh_np(x: np.ndarray) -> np.ndarray:
    """GELU tanh approximation in numpy (matches torch approximate='tanh')."""
    x = np.asarray(x, dtype=np.float32)
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def _rms_norm_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """No-affine RMSNorm over the last dimension (matches comfy common_dit.rms_norm)."""
    x = np.asarray(x, dtype=np.float32)
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps)


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (max-subtract) in numpy."""
    x = np.asarray(x, dtype=np.float32)
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _is_mps_kernel_bf16_bug(device_name: str, precision: str, abs_x: float) -> bool:
    """MPS fused bf16 GELU kernel -> NaN at |x|>=15 (report section 8 / section 4 trap 1)."""
    return (
        device_name.lower() == "mps"
        and precision.lower() == "bf16"
        and abs(abs_x) >= 15.0
    )


def _classify(diff: float, nan_count: int, op: str, relative_factor_err: float = 0.0) -> str:
    if nan_count > 0:
        return "nan"
    if op == "adln_modulate":
        # Absolute diff is small (factor is near zero); classify by relative error.
        return "divergence" if relative_factor_err > _ADALN_RELATIVE_FACTOR_ERR else "ok"
    if diff >= _DIVERGENCE_MAX_DIFF:
        return "divergence"
    if diff > _OK_MAX_DIFF:
        return "divergence"
    return "ok"


class NumericalProbe:
    """Diagnose cross-device numerical divergence on boundary inputs.

    Usage:
        report = NumericalProbe().probe_ops("mps", "bf16")
        for r in report.results:
            print(r.op, r.status, r.max_diff, r.nan_count)
        report.has_failure            # bool
        report.repair_skills_suggested  # ["gelu_fp32", "adaln_fp32", ...]

    Falls back to a simulated (numpy) report when torch or the target device is
    unavailable, so the platform runs on any machine without a GPU.
    """

    def probe_ops(self, device_name: str = "mps", precision: str = "bf16") -> DiagnosticReport:
        """Probe all sensitive ops; real torch comparison when available, else simulated."""
        real = self._probe_real(device_name, precision)
        if real is not None:
            return real
        return self._probe_simulated(device_name, precision)

    # ------------------------------------------------------------------
    # Real path (torch + target device available)
    # ------------------------------------------------------------------
    def _probe_real(self, device_name: str, precision: str) -> DiagnosticReport | None:
        try:
            import torch
            import torch.nn.functional as F
        except Exception:
            return None
        dev = self._resolve_device(device_name, torch)
        if dev is None:
            return None
        dtype = self._precision_dtype(precision, torch)
        if dtype is None:
            # int8/int4/coreml cannot form a real float tensor -> simulate.
            return None

        torch.manual_seed(_SEED)
        results: list[OpResult] = []
        results.append(self._gelu_real(torch, F, dev, dtype, precision))
        results.append(self._adaln_real(torch, dev, dtype))
        results.append(self._rmsnorm_real(torch, dev, dtype, precision))
        results.append(self._softmax_real(torch, F, dev, dtype))
        return DiagnosticReport(
            device_name=device_name,
            precision=precision,
            simulated=False,
            results=results,
            summary=self._summarize(results, device_name, precision, simulated=False),
        )

    def _resolve_device(self, device_name: str, torch: Any) -> Any:
        name = device_name.lower()
        if name == "cpu":
            return torch.device("cpu")
        if name == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                return None
            return torch.device("mps")
        if name in ("cuda", "gpu"):
            if not torch.cuda.is_available():
                return None
            return torch.device("cuda")
        return None

    def _precision_dtype(self, precision: str, torch: Any) -> Any:
        p = precision.lower()
        if p == "fp32" or p == "tf32":
            return torch.float32
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None  # int8/int4/fp8/coreml -> simulate

    def _gelu_real(self, torch: Any, F: Any, dev: Any, dtype: Any, precision: str) -> OpResult:
        x_val = GELU_X
        ref = F.gelu(torch.tensor([x_val], dtype=torch.float32), approximate="tanh")
        xt = torch.tensor([x_val], dtype=dtype, device=dev)
        try:
            out = F.gelu(xt, approximate="tanh").to(torch.float32).cpu()
        except Exception:
            return OpResult("gelu_tanh", "x=" + str(x_val) + " " + precision,
                            float("inf"), 1, "nan")
        nan_count = int(torch.isnan(out).sum().item())
        diff = float("inf") if nan_count else float((out - ref).abs().max().item())
        status = _classify(diff, nan_count, "gelu_tanh")
        return OpResult("gelu_tanh", "x=" + str(x_val) + " " + precision + " on " + str(dev),
                        diff, nan_count, status)

    def _adaln_real(self, torch: Any, dev: Any, dtype: Any) -> OpResult:
        d = 1024
        # True fp32 inputs (the cpu-fp32 reference); quantized copies feed the
        # target device so the reference is the real -0.999, not bf16(-1.0).
        x_true = (torch.randn(d) * 50.0).to(dtype=torch.float32)
        scale_true = torch.tensor([ADALN_SCALE], dtype=torch.float32)
        shift_true = torch.zeros(1, dtype=torch.float32)
        x = x_true.to(dtype=dtype, device=dev)
        scale = scale_true.to(dtype=dtype, device=dev)
        shift = shift_true.to(dtype=dtype, device=dev)
        eps = 1e-6

        def rms(t: Any) -> Any:
            return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps)

        # fp32 reference (true values) on cpu: (1 + -0.999) == 0.001.
        ref_factor = 1.0 + scale_true
        ref = rms(x_true) * ref_factor + shift_true
        # target precision on device: (1 + bf16(-0.999)) == (1 + -1.0) == 0.0
        # because -0.999 rounds to -1.0 in bf16 -> catastrophic cancellation.
        out_factor = 1.0 + scale
        out = (rms(x) * out_factor + shift).to(torch.float32).cpu()
        nan_count = int(torch.isnan(out).sum().item())
        diff = float("inf") if nan_count else float((out - ref).abs().max().item())
        rel = float((out_factor.to(torch.float32).cpu() - ref_factor).abs().max().item())
        rel /= max(float(ref_factor.abs().max().item()), 1e-12)
        status = _classify(diff, nan_count, "adln_modulate", relative_factor_err=rel)
        return OpResult("adln_modulate", "scale=" + str(ADALN_SCALE) + " " + str(dtype) +
                        " on " + str(dev) + "; (1+scale) rel_err=" + format(rel, ".4f"),
                        diff, nan_count, status)

    def _rmsnorm_real(self, torch: Any, dev: Any, dtype: Any, precision: str) -> OpResult:
        d = 1024
        x_val = RMSNORM_ABS
        x = torch.full((d,), x_val, dtype=dtype, device=dev)
        eps = 1e-6

        def rms(t: Any) -> Any:
            return t * torch.rsqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps)

        ref = rms(x.to(torch.float32).cpu())
        try:
            out = rms(x).to(torch.float32).cpu()
        except Exception:
            return OpResult("rmsnorm", "|x|=" + str(x_val) + " " + precision,
                            float("inf"), 1, "nan")
        nan_count = int(torch.isnan(out).sum().item())
        inf_count = int(torch.isinf(out).sum().item())
        diff = float("inf") if (nan_count or inf_count) else float((out - ref).abs().max().item())
        status = _classify(diff, nan_count + inf_count, "rmsnorm")
        return OpResult("rmsnorm", "|x|=" + str(x_val) + " " + precision + " on " + str(dev),
                        diff, nan_count, status)

    def _softmax_real(self, torch: Any, F: Any, dev: Any, dtype: Any) -> OpResult:
        n = SOFTMAX_SEQ
        # True fp32 logits; quantize a copy for the target device.
        logits_true = (torch.randn(n) * 5.0).to(dtype=torch.float32)
        logits_true[0] = 8.0  # large score gap to stress exp precision.
        logits = logits_true.to(dtype=dtype, device=dev)
        ref = F.softmax(logits_true, dim=-1)
        out = F.softmax(logits, dim=-1).to(torch.float32).cpu()
        nan_count = int(torch.isnan(out).sum().item())
        diff = float("inf") if nan_count else float((out - ref).abs().max().item())
        status = _classify(diff, nan_count, "softmax")
        return OpResult("softmax", "seq=" + str(n) + " " + str(dtype) + " on " + str(dev),
                        diff, nan_count, status)

    # ------------------------------------------------------------------
    # Simulated path (no torch / no device) -- pure numpy
    # ------------------------------------------------------------------
    def _probe_simulated(self, device_name: str, precision: str) -> DiagnosticReport:
        rng = np.random.RandomState(_SEED)
        results: list[OpResult] = [
            self._gelu_sim(device_name, precision),
            self._adaln_sim(device_name, precision, rng),
            self._rmsnorm_sim(device_name, precision),
            self._softmax_sim(device_name, precision, rng),
        ]
        return DiagnosticReport(
            device_name=device_name,
            precision=precision,
            simulated=True,
            results=results,
            summary=self._summarize(results, device_name, precision, simulated=True),
        )

    def _gelu_sim(self, device_name: str, precision: str) -> OpResult:
        x_val = GELU_X
        ref = _gelu_tanh_np(np.array([x_val], dtype=np.float32))
        # MPS bf16 fused-kernel bug: NaN at |x|>=15 (not a pure-precision effect).
        if _is_mps_kernel_bf16_bug(device_name, precision, x_val):
            return OpResult("gelu_tanh", "x=" + str(x_val) + " " + precision + " (MPS kernel bug)",
                            float("inf"), 1, "nan")
        xq = _to_precision(np.array([x_val], dtype=np.float32), precision)
        out = _to_precision(_gelu_tanh_np(xq), precision)
        nan_count = int(np.isnan(out).sum())
        diff = float("inf") if nan_count else float(np.max(np.abs(out - ref)))
        status = _classify(diff, nan_count, "gelu_tanh")
        return OpResult("gelu_tanh", "x=" + str(x_val) + " " + precision,
                        diff, nan_count, status)

    def _adaln_sim(self, device_name: str, precision: str, rng: np.random.RandomState) -> OpResult:
        d = 1024
        x = rng.randn(d).astype(np.float32) * 50.0
        scale = np.array([ADALN_SCALE], dtype=np.float32)
        shift = np.zeros(1, dtype=np.float32)

        ref_factor = 1.0 + scale
        ref = _rms_norm_np(x) * ref_factor + shift

        scale_q = _to_precision(scale, precision)
        x_q = _to_precision(x, precision)
        out_factor = 1.0 + scale_q
        out = _to_precision(_rms_norm_np(x_q) * out_factor + shift, precision)

        nan_count = int(np.isnan(out).sum())
        diff = float("inf") if nan_count else float(np.max(np.abs(out - ref)))
        rel = float(np.max(np.abs(out_factor - ref_factor)))
        rel /= max(float(np.max(np.abs(ref_factor))), 1e-12)
        status = _classify(diff, nan_count, "adln_modulate", relative_factor_err=rel)
        return OpResult("adln_modulate", "scale=" + str(ADALN_SCALE) + " " + precision +
                        "; (1+scale)=" + format(float(ref_factor[0]), ".6f") +
                        " -> " + format(float(out_factor[0]), ".6f") +
                        " (rel_err=" + format(rel, ".4f") + ")",
                        diff, nan_count, status)

    def _rmsnorm_sim(self, device_name: str, precision: str) -> OpResult:
        x_val = RMSNORM_ABS
        x = np.array([x_val] * 8, dtype=np.float32)
        ref = _rms_norm_np(x)  # fp32 -> 1.0 for constant |x|.
        # fp16 square-sum overflow: |x|>256 -> x^2 > 65504 -> inf -> zero/NaN norm.
        p = precision.lower()
        if p == "fp16":
            with np.errstate(over="ignore"):
                sq = np.array([x_val ** 2], dtype=np.float16)
            if not np.isfinite(sq)[0]:
                return OpResult("rmsnorm", "|x|=" + str(x_val) + " " + precision +
                                " (fp16 x^2 overflow)", float("inf"), 1, "nan")
        xq = _to_precision(x, precision)
        out = _to_precision(_rms_norm_np(xq), precision)
        nan_count = int(np.isnan(out).sum())
        inf_count = int(np.isinf(out).sum())
        diff = float("inf") if (nan_count or inf_count) else float(np.max(np.abs(out - ref)))
        status = _classify(diff, nan_count + inf_count, "rmsnorm")
        return OpResult("rmsnorm", "|x|=" + str(x_val) + " " + precision,
                        diff, nan_count, status)

    def _softmax_sim(self, device_name: str, precision: str, rng: np.random.RandomState) -> OpResult:
        n = SOFTMAX_SEQ
        logits = (rng.randn(n).astype(np.float32) * 5.0)
        logits[0] = 8.0
        ref = _softmax_np(logits)
        lq = _to_precision(logits, precision)
        out = _to_precision(_softmax_np(lq), precision)
        nan_count = int(np.isnan(out).sum())
        diff = float("inf") if nan_count else float(np.max(np.abs(out - ref)))
        status = _classify(diff, nan_count, "softmax")
        return OpResult("softmax", "seq=" + str(n) + " " + precision,
                        diff, nan_count, status)

    # ------------------------------------------------------------------
    def _summarize(
        self,
        results: list[OpResult],
        device_name: str,
        precision: str,
        simulated: bool,
    ) -> str:
        mode = "SIMULATED" if simulated else "measured"
        fails = [r for r in results if r.status in ("divergence", "nan")]
        if not fails:
            return (
                "[" + mode + "] " + device_name + "/" + precision + ": all sensitive ops "
                "within tolerance; no repair skill required."
            )
        names = ", ".join(r.op + "=" + r.status for r in fails)
        skills = ", ".join(DiagnosticReport(
            device_name=device_name, precision=precision, simulated=simulated, results=fails,
        ).repair_skills_suggested) or "none"
        return (
            "[" + mode + "] " + device_name + "/" + precision + ": " + str(len(fails)) + "/"
            + str(len(results)) + " sensitive ops divergent (" + names + "). Suggested repair: "
            + skills + "."
        )


# Module-level convenience for quick one-shot probing.
def probe(device_name: str = "mps", precision: str = "bf16") -> DiagnosticReport:
    """Run a NumericalProbe and return its DiagnosticReport."""
    return NumericalProbe().probe_ops(device_name, precision)
