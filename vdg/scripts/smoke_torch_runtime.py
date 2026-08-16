#!/usr/bin/env python3
"""Real-host smoke test: VDG TorchRuntime repair patches on a toy video DiT.

Builds a tiny DiT-shaped module (nn.Linear stack + nn.GELU + an LTX-style
AdaLN block carrying scale_shift_table / attn1 / ff + RMSNorm + Softmax),
applies the governance repair decisions via ``TorchRuntime.apply_all`` on
whatever device torch exposes (MPS on Apple Silicon, CUDA, or CPU), asserts
that:

  * forward still runs (same output shape, finite values) after patching,
  * ``_vdg_patched`` marks exist on every patched site,
  * ``unpatch()`` restores the original forwards and clears all marks.

This is the "runtime is real, not an emitter stub" proof: the exact patch
functions VDG's governance decisions point at run in-process against a real
module graph. Pure stdlib + torch; no diffusers required. Exits nonzero on
any assertion failure.

Usage:
    python scripts/smoke_torch_runtime.py
"""
from __future__ import annotations

import sys


def _device_name() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_toy_dit():
    """A small DiT-shaped module carrying every sensitive patch site.

    * ``gelu1``          -- nn.GELU(tanh)                      -> gelu_fp32
    * ``norm1``          -- nn.LayerNorm (RMSNorm bucket)      -> rmsnorm_fp32
    * ``adaln_block``    -- LTX-style AdaLN (scale_shift_table,
                            attn1, ff)                         -> adaln_fp32
    * ``softmax1``       -- nn.Softmax                         -> softmax_fp32
    * ``proj_out``       -- nn.Linear                          -> plain (no patch)

    The AdaLN block mirrors the real LTX BasicTransformerBlock interface the
    patch guards on (scale_shift_table + attn1 + ff attributes).
    """
    import torch
    import torch.nn as nn

    class Attn1(nn.Module):
        """Mini self-attention honoring the LTX call interface the patch uses
        (pe / mask / transformer_options kwargs)."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x, pe=None, mask=None, transformer_options=None):
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            attn = (q @ k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
            if mask is not None:
                attn = attn + mask
            w = torch.softmax(attn, dim=-1)
            return self.proj(w @ v)

    class Attn2(nn.Module):
        """Mini cross-attention honoring the kwargs the patched forward passes."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.q = nn.Linear(dim, dim)
            self.k = nn.Linear(dim, dim)
            self.v = nn.Linear(dim, dim)

        def forward(self, x, context=None, mask=None, transformer_options=None):
            if context is None:
                context = x
            q, k, v = self.q(x), self.k(context), self.v(context)
            attn = (q @ k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
            if mask is not None:
                attn = attn + mask
            w = torch.softmax(attn, dim=-1)
            return w @ v

    class AdaLNBlock(nn.Module):
        """LTX-style AdaLN block: scale_shift_table + attn1 + ff interface that
        the adaln_fp32 patch guards on (the six modulation tensors are derived
        from scale_shift_table + timestep)."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) * 0.02)
            self.attn1 = Attn1(dim)
            self.attn2 = Attn2(dim)
            self.ff = nn.Linear(dim, dim)

        def forward(self, x, context=None, attention_mask=None, timestep=None,
                    pe=None, transformer_options=None, self_attention_mask=None,
                    prompt_timestep=None, **kwargs):
            # Minimal modulated forward: x -> attn1 -> residual -> ff.
            h = self.attn1(x)
            x = x + h
            return x + self.ff(x)

    class ToyDiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dim = 16
            self.proj_in = nn.Linear(dim, dim)
            self.gelu1 = nn.GELU(approximate="tanh")
            self.norm1 = nn.LayerNorm(dim)
            self.adaln_block = AdaLNBlock(dim)
            self.softmax1 = nn.Softmax(dim=-1)
            self.proj_out = nn.Linear(dim, dim)

        def forward(self, x):
            x = self.proj_in(x)
            x = self.gelu1(x)
            x = self.norm1(x)
            # Pass a real timestep so the patched AdaLN forward takes its
            # fp32-MPS modulation branch (not the passthrough path). The patch
            # reshapes timestep to [B, T, 6, dim] against the 6-row
            # scale_shift_table, so the embedding carries 6*dim per frame.
            timestep = torch.randn(x.shape[0], 1, 6 * self.proj_in.out_features,
                                   device=x.device)
            x = self.adaln_block(x, timestep=timestep)
            x = self.softmax1(x)
            return self.proj_out(x)

    return ToyDiT()


def main() -> int:
    import torch

    from vdg.runtime.torch_runtime import TorchRuntime

    print("VDG real-host TorchRuntime smoke test")
    print("  torch: " + torch.__version__ + "  device: " + _device_name())

    torch.manual_seed(0)
    device = torch.device(_device_name())
    mod = build_toy_dit().to(device)
    # 3D token-sequence input (B=2 batch, L=8 spatial/temporal tokens, dim=16),
    # faithful to a real DiT: the AdaLN modulation broadcasts [B, 1, dim] over
    # the token axis.
    x = torch.randn(2, 8, 16, device=device)

    # --- baseline ----------------------------------------------------------
    with torch.no_grad():
        ref = mod(x)
    print("  baseline forward ok: shape=" + str(tuple(ref.shape))
          + " finite=" + str(bool(torch.isfinite(ref).all())))

    # --- apply the governance repair decisions -----------------------------
    rt = TorchRuntime()
    decisions = [
        ("gelu_fp32", {"approximate": "tanh"}),
        ("adaln_fp32", {}),
        ("rmsnorm_fp32", {}),
        ("softmax_fp32", {}),
    ]
    results = rt.apply_all(mod, decisions)
    for r in results:
        print("  applied %-14s count=%d targets=%s"
              % (r["skill"], r["applied"], r["targets"][:4]))
    total = sum(r["applied"] for r in results)
    assert total >= 4, "expected >=4 patch sites, got %d" % total

    # --- _vdg_patched marks -------------------------------------------------
    marked = [
        name for name, m in mod.named_modules() if getattr(m, "_vdg_patched", None)
    ]
    print("  _vdg_patched sites: " + ", ".join(marked))
    assert len(marked) == total
    assert any("gelu1" in n for n in marked)
    assert any("adaln_block" in n for n in marked)
    assert any("norm1" in n for n in marked)
    assert any("softmax1" in n for n in marked)

    # --- forward still runs after patching (on the real device) ------------
    with torch.no_grad():
        out = mod(x)
    assert tuple(out.shape) == tuple(ref.shape)
    assert bool(torch.isfinite(out).all()), "patched forward produced non-finite output"
    print("  patched forward ok: shape=" + str(tuple(out.shape))
          + " finite=" + str(bool(torch.isfinite(out).all())))

    # --- idempotency guard: re-apply must skip already-patched sites --------
    again = rt.apply_all(mod, decisions)
    assert all(r["applied"] == 0 for r in again), "re-apply must be a no-op"
    print("  re-apply idempotent: skipped all %d already-patched sites" % len(again))

    # --- unpatch restores originals ----------------------------------------
    n = rt.unpatch(mod)
    leftovers = [
        name for name, m in mod.named_modules() if getattr(m, "_vdg_patched", None)
    ]
    assert n == total, "unpatch count %d != patched %d" % (n, total)
    assert not leftovers, "leftover _vdg_patched marks: %r" % (leftovers,)
    with torch.no_grad():
        out2 = mod(x)
    assert bool(torch.isfinite(out2).all())
    print("  unpatch ok: restored %d original forwards, 0 marks left" % n)

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - smoke test must print the failure
        print("SMOKE TEST FAILED: " + repr(exc), file=sys.stderr)
        raise SystemExit(1)
