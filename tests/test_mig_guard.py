"""Worker-side MIG-slice fail-fast: a partitioned GPU must be detected up front and surfaced as a
RETRIABLE infra error so the runner re-provisions a fresh FULL GPU — instead of the run dying with
an opaque CUDA-allocator assert mid-setup ("won't randomly die")."""

from __future__ import annotations

import types

import pytest

from flash.engine.worker.perf import RetriableInfraError, detect_mig_slice, wait_for_gpu


def _fake_run(outputs):
    """A subprocess.run stub returning per-command stdout (keyed by '-L' vs 'mig.mode')."""

    def run(cmd, capture_output=True, text=True, timeout=None):
        key = "-L" if "-L" in cmd else "mig.mode"
        return types.SimpleNamespace(stdout=outputs.get(key, ""))

    return run


_FULL_A100 = "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-abc123)"
_MIG_LIST = (
    "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-abc123)\n"
    "  MIG 1g.10gb     Device  0: (UUID: MIG-deadbeef-0000-0000-0000-000000000000)"
)


def test_detect_mig_via_device_list(monkeypatch):
    monkeypatch.setattr("subprocess.run", _fake_run({"-L": _MIG_LIST, "mig.mode": "Disabled"}))
    reason = detect_mig_slice()
    assert reason is not None
    assert "MIG slice detected" in reason


def test_detect_mig_via_mig_mode(monkeypatch):
    # -L doesn't show a MIG device, but mig.mode reports Enabled (partitioned GPU).
    monkeypatch.setattr("subprocess.run", _fake_run({"-L": _FULL_A100, "mig.mode": "Enabled"}))
    reason = detect_mig_slice()
    assert reason is not None
    assert "MIG mode enabled" in reason


def test_no_false_positive_on_full_gpu(monkeypatch):
    # A normal full GPU: no MIG device line, mig.mode Disabled / Not Supported.
    for mode in ("Disabled", "[N/A]", "[Not Supported]", ""):
        monkeypatch.setattr(
            "subprocess.run", _fake_run({"-L": "GPU 0: NVIDIA L40 (UUID: GPU-x)", "mig.mode": mode})
        )
        assert detect_mig_slice() is None


def test_detect_never_raises_on_nvidia_smi_failure(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("subprocess.run", boom)
    assert detect_mig_slice() is None  # best-effort; absence of nvidia-smi != MIG


def test_wait_for_gpu_fails_fast_retriable_on_mig(monkeypatch):
    # wait_for_gpu checks MIG BEFORE any CUDA op, so this raises without needing a GPU/torch.
    monkeypatch.setattr("subprocess.run", _fake_run({"-L": _MIG_LIST, "mig.mode": "Enabled"}))
    with pytest.raises(RetriableInfraError) as exc:
        wait_for_gpu()
    # The runner classifies this off the heartbeat 'retriable' flag; the marker is for human logs.
    assert "RETRIABLE_INFRA_GPU" in str(exc.value)
    assert "fresh full" in str(exc.value)
