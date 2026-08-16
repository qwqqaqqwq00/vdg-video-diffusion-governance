"""End-to-end real-host smoke test for the TorchRuntime repair patches.

Runs the standalone smoke script (scripts/smoke_torch_runtime.py) which builds
a tiny DiT-shaped module (nn.Linear stack + nn.GELU + LTX-style AdaLN block +
RMSNorm + Softmax), applies the governance repair decisions via
TorchRuntime.apply_all on the real torch device, asserts forward still runs
and _vdg_patched marks exist, then unpatch()es and asserts a clean restore.

Gated behind torch availability so the suite stays host-independent; when
torch is present this is the "runtime is real, not an emitter stub" proof.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "smoke_torch_runtime.py")


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_runtime_smoke_script_passes():
    """The standalone real-host smoke test exits 0 and reports PASSED."""
    result = subprocess.run(
        [sys.executable, _SCRIPT],
        capture_output=True, text=True, timeout=300,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, (
        "smoke_torch_runtime.py failed:\nstdout:\n" + result.stdout
        + "\nstderr:\n" + result.stderr
    )
    assert "SMOKE TEST PASSED" in result.stdout
    assert "_vdg_patched sites" in result.stdout


@pytest.mark.skipif(not _has_torch(), reason="torch not installed")
def test_torch_runtime_smoke_marks_every_site():
    """The smoke run reports marks for all four sensitive site classes."""
    result = subprocess.run(
        [sys.executable, _SCRIPT],
        capture_output=True, text=True, timeout=300,
        cwd=_REPO_ROOT,
    )
    for site in ("gelu1", "adaln_block", "norm1", "softmax1"):
        assert site in result.stdout, "missing patched site " + site
    assert "unpatch ok: restored 4 original forwards, 0 marks left" in result.stdout
