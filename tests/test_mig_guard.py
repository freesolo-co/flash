"""Worker-side GPU verification + CUDA-readiness fail-fast (perf.lifecycle).

``verify_gpu`` is the identity / CUDA-floor check that runs on EVERY provider's rented box, so a
wrong/smaller GPU or a too-old driver fails over fast (a RETRIABLE infra error) instead of crashing
mid-setup with an opaque "no kernel image" / "PTX unsupported toolchain" / OOM. ``wait_for_gpu`` wraps
it in the CUDA-readiness poll, fast-failing a dead host (NVML can't init) instead of burning the full
patient loop. (The MIG-slice guard was removed with the MIG-capable GPUs it protected against.)
"""

from __future__ import annotations

import types

import pytest

# verify_gpu + its helpers live in perf.lifecycle (dev split perf.py into the perf/ package); import
# the module too so monkeypatches target the names wait_for_gpu/verify_gpu resolve internally.
from flash.engine.worker.perf import lifecycle
from flash.engine.worker.perf.lifecycle import (
    RetriableInfraError,
    _gpu_mismatch_reason,
    _sm_major,
    verify_gpu,
    wait_for_gpu,
)


# ---------------------------------------------------------------------------
# Standardized GPU-identity / CUDA-floor verification (verify_gpu): the one check that runs on EVERY
# provider's rented box, so a wrong/smaller GPU or a too-old driver fails over fast instead of
# crashing mid-setup with an opaque "no kernel image" / "PTX unsupported toolchain" / OOM.
# ---------------------------------------------------------------------------
def test_sm_major_parses_compute_capability():
    # 'sm89' = compute 8.9 (major 8); 'sm120' = 12.0 (major 12) — major is all-but-last-digit.
    assert _sm_major("sm80") == 8
    assert _sm_major("sm89") == 8
    assert _sm_major("sm90") == 9
    assert _sm_major("sm120") == 12
    assert _sm_major("sm100") == 10
    assert _sm_major("") is None
    assert _sm_major("ampere") is None


def test_mismatch_reason_none_when_gpu_matches():
    # Correct L40 (sm89 / 48 GB / CUDA 12.8): live values all satisfy -> no mismatch.
    assert _gpu_mismatch_reason("L40", (8, 9), 47.5, 12.8) is None


def test_mismatch_reason_flags_smaller_vram():
    # Asked H100 (80 GB) but handed a 48 GB card -> VRAM substitution caught.
    r = _gpu_mismatch_reason("H100", (9, 0), 48.0, 12.8)
    assert r is not None
    assert "VRAM" in r
    assert "H100" in r


def test_mismatch_reason_flags_old_driver_for_blackwell():
    # RTX 5090 (sm120) needs CUDA 13.0 to JIT the wheels' PTX; a 12.8 driver is below floor.
    r = _gpu_mismatch_reason("RTX 5090", (12, 0), 32.0, 12.8)
    assert r is not None
    assert "driver CUDA" in r
    assert "13" in r


def test_mismatch_reason_flags_generational_downgrade():
    # Asked H100 (sm90, major 9) but handed an sm80 card (same 80 GB so VRAM passes) -> caught on
    # compute-capability major.
    r = _gpu_mismatch_reason("H100", (8, 0), 80.0, 12.8)
    assert r is not None
    assert "compute capability" in r


def test_mismatch_reason_does_not_penalize_higher_minor():
    # Minor is NOT capability-ordered: an A100 (sm80, major 8) substituted for an A6000 (sm86) must
    # NOT be flagged on capability (major 8 >= 8); the bigger A100 also clears the A6000 VRAM bar.
    assert _gpu_mismatch_reason("RTX A6000", (8, 0), 79.0, 12.8) is None


def test_mismatch_reason_best_effort_on_unknowns():
    # Unknown/unset class -> nothing to verify against; None live inputs -> each dimension skipped.
    assert _gpu_mismatch_reason("totally-made-up-gpu", (8, 0), 1.0, 1.0) is None
    assert _gpu_mismatch_reason("L40", None, None, None) is None
    assert _gpu_mismatch_reason(None, (8, 0), 48.0, 12.8) is None


def test_mismatch_reason_joins_multiple():
    # Both VRAM and driver wrong for an RTX 5090 -> both reasons reported.
    r = _gpu_mismatch_reason("RTX 5090", (12, 0), 16.0, 12.0)
    assert r is not None
    assert "VRAM" in r
    assert "driver CUDA" in r


def test_host_driver_cuda_parses_nvidia_smi_header(monkeypatch):
    # pynvml is absent in CI (and nvmlInit fails on a GPU-less box), so it falls to the nvidia-smi
    # header parse. CUDA 12.8 in the table header -> 12.8.
    header = (
        "| NVIDIA-SMI 570.195.03    Driver Version: 570.195.03    CUDA Version: 12.8 |\n"
    )
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(stdout=header),
    )
    assert lifecycle._host_driver_cuda() == 12.8


def test_host_driver_cuda_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi"))
    )
    assert lifecycle._host_driver_cuda() is None


def test_verify_gpu_noop_when_unset():
    # Falsy requested class -> no-op (returns before any torch/NVML access).
    verify_gpu(None)
    verify_gpu("")


def test_verify_gpu_raises_retriable_on_mismatch(monkeypatch):
    torch = pytest.importorskip("torch")  # offline CI has no torch -> skip the live-read tests

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0), raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a: types.SimpleNamespace(total_memory=int(40e9)),
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA A100-40GB", raising=False)
    monkeypatch.setattr(lifecycle, "_host_driver_cuda", lambda: 12.8)
    # Requested H100 (sm90 / 80 GB) but the live box is a 40 GB sm80 card -> VRAM + capability fail.
    with pytest.raises(RetriableInfraError) as exc:
        verify_gpu("H100")
    assert "does not match requested" in str(exc.value)
    assert "RETRIABLE_INFRA_GPU" in str(exc.value)


def test_verify_gpu_passes_on_correct_gpu(monkeypatch):
    torch = pytest.importorskip("torch")  # offline CI has no torch -> skip the live-read tests

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 9), raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a: types.SimpleNamespace(total_memory=int(47.5e9)),
        raising=False,
    )
    monkeypatch.setattr(lifecycle, "_host_driver_cuda", lambda: 12.8)
    verify_gpu("L40")  # sm89 / 48 GB / 12.8 -> matches -> no raise


def test_wait_for_gpu_propagates_verify_mismatch(monkeypatch):
    # A verify mismatch must FAIL OVER, not be swallowed by the readiness retry loop and masked as
    # "never became ready". Drive past the CUDA-live probe, then have verify_gpu reject.
    torch = pytest.importorskip("torch")  # offline CI has no torch -> skip the live-read tests

    class _Live:  # the result of torch.zeros(...) must support `+ 1`
        def __add__(self, other):
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: _Live(), raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "wrong-gpu", raising=False)
    monkeypatch.setattr(
        lifecycle,
        "verify_gpu",
        lambda g: (_ for _ in ()).throw(RetriableInfraError("WRONG GPU substituted")),
    )
    with pytest.raises(RetriableInfraError) as exc:
        wait_for_gpu("H100")
    assert "WRONG GPU substituted" in str(exc.value)
    assert "never became ready" not in str(exc.value)


def test_wait_for_gpu_fast_fails_on_dead_nvml(monkeypatch):
    """A persistent NVML-init failure (broken host / GPU off the bus) fails over FAST (after ~30s of
    ruling out a transient), NOT after the full 12-try patient loop — emits the NVML-fault message."""
    torch = pytest.importorskip("torch")

    def boom(*a, **k):
        raise RuntimeError("CUDA error: cudaErrorDevicesUnavailable")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch, "zeros", boom, raising=False)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_nvml_alive", lambda: False)  # broken host: NVML can't init
    with pytest.raises(RetriableInfraError) as exc:
        wait_for_gpu()
    assert "NVML init failed" in str(exc.value)
    assert "never became ready" not in str(exc.value)  # fast-failed, didn't exhaust the patient loop


def test_wait_for_gpu_fast_fails_on_dead_nvml_when_cuda_unavailable(monkeypatch):
    """Regression: a dead driver / GPU off the bus makes ``torch.cuda.is_available()`` return False
    WITHOUT raising, so the NVML fast-fail must also run on that no-exception path — otherwise the
    broken host burns the full ~120s patient loop instead of failing over fast (~30s)."""
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)  # no exception
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_nvml_alive", lambda: False)  # broken host: NVML can't init
    with pytest.raises(RetriableInfraError) as exc:
        wait_for_gpu()
    assert "NVML init failed" in str(exc.value)
    assert "never became ready" not in str(exc.value)  # fast-failed, didn't exhaust the patient loop


def test_wait_for_gpu_stays_patient_when_nvml_alive(monkeypatch):
    """A merely BUSY device (NVML still alive) keeps the patient loop — it must NOT be treated as a
    dead host; it only gives up with 'never became ready' after the full window."""
    torch = pytest.importorskip("torch")

    def boom(*a, **k):
        raise RuntimeError("CUDA device busy")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch, "zeros", boom, raising=False)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_nvml_alive", lambda: True)  # busy but NVML alive -> stay patient
    with pytest.raises(RetriableInfraError) as exc:
        wait_for_gpu()
    assert "never became ready" in str(exc.value)
    assert "NVML init failed" not in str(exc.value)
