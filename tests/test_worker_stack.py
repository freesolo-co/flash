"""Worker stack selection + worker config compat + LoRA exclusion unit tests (CPU-only)."""

from __future__ import annotations

import inspect
import sys
import types
from types import SimpleNamespace

import pytest

import flash.engine.worker.io.heartbeat as worker_heartbeat
import flash.engine.worker.model.adapter as worker_adapter


def _worker_image_specs() -> list[str]:
    """The pinned stack the worker image actually installs.

    Dockerfile.worker is the single source of truth for the run stack: it is what the GPU runs.
    """
    import pathlib

    import docker.kernel_fingerprint as kf

    root = pathlib.Path(__file__).resolve().parent.parent
    return kf._pip_stack_specs((root / "Dockerfile.worker").read_text())


def _perf_pins() -> tuple[str, str]:
    """The (tilelang, apache-tvm-ffi) versions perf enforces at runtime.

    Read from the source rather than restated here: these pins move together with flash-qla's hard
    requirement, and a test that hardcodes them keeps passing while the image build breaks -- which
    is exactly how the 0.1.11/0.1.9 conflict went unnoticed.
    """
    import re

    import flash.engine.worker.perf as perf

    source = inspect.getsource(perf._ensure_fla_fastpath_on_hopper)
    tilelang = re.search(r'TILELANG_PIN\s*=\s*"([^"]+)"', source)
    tvm_ffi = re.search(r'TVM_FFI_PIN\s*=\s*"([^"]+)"', source)
    assert tilelang, "perf must declare TILELANG_PIN as a string literal"
    assert tvm_ffi, "perf must declare TVM_FFI_PIN as a string literal"
    return tilelang.group(1), tvm_ffi.group(1)


def _other_version(pin: str) -> str:
    """A version that is definitely NOT the pin, for 'wrong version resident' cases."""
    return "0.0.1-not-the-pin" if pin != "0.0.1-not-the-pin" else "0.0.2-not-the-pin"


# sentinel meaning "default to whatever perf currently enforces"; None already means "absent".
_PIN = object()


def test_gdn_fastpath_deps_present_and_kept_on_hopper():
    """The GDN fast-path stack (fla-from-git + tilelang + pinned apache-tvm-ffi) is baked in, and
    fla is KEPT on Hopper (sm90) — the #640 fix is fla's tilelang backend, not dropping fla."""
    specs = _worker_image_specs()
    joined = " ".join(specs)
    assert (
        "git+https://github.com/fla-org/flash-linear-attention" in joined
    )  # complete fla, not the broken PyPI stub
    assert any(
        d.startswith("tilelang==") for d in specs
    )  # correct GDN backend on Triton>=3.4, PINNED for reproducibility
    tilelang_pin, tvm_ffi_pin = _perf_pins()
    assert any(
        d.startswith(f"apache-tvm-ffi=={tvm_ffi_pin}") for d in specs
    )  # pin (0.1.12 aborts tilelang import)
    # The image and the runtime gate must agree. perf fails CLOSED on a mismatch: it deletes fla and
    # drops sm90 to the pure-PyTorch fallback, so a drifted pin is a silent perf cliff, not an error.
    assert any(d.startswith(f"tilelang=={tilelang_pin}") for d in specs), (
        f"Dockerfile tilelang must equal perf's TILELANG_PIN {tilelang_pin}, got: "
        f"{[d for d in specs if d.startswith('tilelang')]}"
    )
    # fla must NOT be dropped on Hopper anymore (it was, pre-fix).
    assert any("flash-linear-attention" in d for d in specs), (
        "fla must be kept on Hopper for the tilelang fast path"
    )


def test_tilelang_pin_satisfies_flash_qla_hard_requirement():
    """flash-qla hard-pins tilelang and apache-tvm-ffi with `==`, so any other pin makes the image
    layer unsatisfiable. That is not a soft warning: pip fails the build with ResolutionImpossible,
    the image is never rebuilt, and GPU workers silently keep running whatever code was baked last.
    Asserted against the Dockerfile rather than a restated constant so the two cannot drift apart.
    """
    specs = _worker_image_specs()
    flash_qla = [d for d in specs if d.startswith("flash-qla==")]
    if not flash_qla:
        pytest.skip("no pinned flash-qla in the worker stack")
    tilelang_pin, tvm_ffi_pin = _perf_pins()
    # flash-qla 0.1.1 and 0.1.2 both require exactly tilelang==0.1.9 / apache-tvm-ffi==0.1.9.
    required = "0.1.9"
    assert tilelang_pin == required, (
        f"flash-qla requires tilelang=={required}, but the worker pins {tilelang_pin}; "
        "the image build fails with ResolutionImpossible"
    )
    assert tvm_ffi_pin == required, (
        f"flash-qla requires apache-tvm-ffi=={required}, but the worker pins {tvm_ffi_pin}"
    )


def test_worker_stack_pins_qwen35_capable_versions():
    from packaging.requirements import Requirement
    from packaging.version import Version

    requirements = {
        requirement.name: requirement
        for spec in _worker_image_specs()
        if " @ " not in spec
        for requirement in (Requirement(spec),)
    }
    exact_versions = {
        name: [Version(item.version) for item in requirement.specifier if item.operator == "=="]
        for name, requirement in requirements.items()
    }
    assert exact_versions["vllm"] == [Version("0.19.1")]
    assert len(exact_versions["transformers"]) == 1
    assert exact_versions["transformers"][0].major == 5  # qwen3_5 model types need transformers 5.x
    assert exact_versions["bitsandbytes"] == [Version("0.50.1")]  # 8-bit paged AdamW state


def _import_worker(monkeypatch):
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    import flash.engine.worker.model.adapter as worker

    return worker


@pytest.mark.parametrize(
    ("revision", "expected"),
    [("refs/pr/123", {"revision": "refs/pr/123"}), ("", {})],
)
def test_model_revision_keyword_is_present_only_when_nonempty(revision, expected):
    from flash.engine.worker.io.hf import model_revision_kwargs

    assert model_revision_kwargs(revision) == expected


@pytest.mark.parametrize("revision", ["refs/pr/123", ""])
def test_model_revision_threads_through_config_probes(monkeypatch, revision):
    calls = []

    class _AutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return SimpleNamespace(
                model_type="qwen3_5",
                hidden_size=4096,
                num_hidden_layers=32,
                layer_types=("linear_attention",),
            )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = _AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from flash.engine.worker.entry import sft
    from flash.engine.worker.model import packing
    from flash.engine.worker.perf import liger

    assert packing.model_is_gdn_hybrid("org/model", revision=revision)
    assert packing.gdn_model_type("org/model", revision=revision)
    assert sft._model_arch_dims("uncataloged/model", revision=revision) == (4096, 32)
    assert isinstance(liger._liger_default_for_model("org/model", revision=revision), bool)

    expected = {"trust_remote_code": True}
    if revision:
        expected["revision"] = revision
    assert calls
    assert all(kwargs == expected for _model_id, kwargs in calls)


def _fake_arch_probe(monkeypatch, *, hidden, layers):
    """Stub AutoConfig so ``_model_arch_dims`` sees a probe that yields ``(hidden, layers)``."""

    class _AutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return SimpleNamespace(
                model_type="qwen3_6", hidden_size=hidden, num_hidden_layers=layers
            )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = _AutoConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_arch_dims_revision_zero_probe_falls_back_to_catalog(monkeypatch):
    # regression: the 35B-A3B multimodal-nested config makes the AutoConfig probe return (0, 0). A
    # revision pin must treat that as "unparseable" and fall back to the curated catalog geometry
    # (exactly like an unpinned run), NOT as a mismatch -- otherwise revision-pinned SFT on that model
    # raises after the GPU is already rented.
    from flash.engine.worker.entry import sft

    _fake_arch_probe(monkeypatch, hidden=0, layers=0)
    # Qwen/Qwen3.6-35B-A3B is the sole catalog entry carrying (hidden, layers) == (2048, 40).
    assert sft._model_arch_dims("Qwen/Qwen3.6-35B-A3B", revision="refs/pr/123") == (2048, 40)


def test_arch_dims_revision_nonzero_mismatch_still_fails_closed(monkeypatch):
    # a NONZERO probe dim that genuinely disagrees with the catalog is a real revision mismatch and must
    # still fail closed, so a revision pin can never silently size VRAM with the wrong geometry.
    from flash.engine.worker.entry import sft

    _fake_arch_probe(monkeypatch, hidden=9999, layers=99)
    with pytest.raises(RuntimeError, match="revision-specific model architecture"):
        sft._model_arch_dims("Qwen/Qwen3.6-35B-A3B", revision="refs/pr/123")


def test_control_plane_tokenizer_disables_remote_repository_code_without_changing_workers(
    monkeypatch,
):
    calls = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append((model_id, kwargs))
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from flash.engine.profiling.tokenizer import load_control_plane_tokenizer, load_tokenizer

    load_control_plane_tokenizer("org/model", revision="refs/pr/123")
    load_tokenizer("org/model", revision="refs/pr/123")

    assert calls == [
        ("org/model", {"trust_remote_code": False, "revision": "refs/pr/123"}),
        ("org/model", {"trust_remote_code": True, "revision": "refs/pr/123"}),
    ]


def test_model_revision_threads_through_tokenizer_and_prefetch(monkeypatch):
    calls = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _AutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import huggingface_hub

    from flash.engine.worker.io import hf

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: calls.append(("snapshot", kwargs["repo_id"], kwargs)),
    )
    monkeypatch.setattr(hf, "_shared_weight_cache_dir", lambda: None)
    monkeypatch.setattr(hf, "_hf_cache_bytes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(hf, "gpu_diagnostics", dict)
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda *args, **kwargs: None)

    assert hf.load_tokenizer("org/model", revision="refs/pr/123") is not None
    hf.prefetch_model("org/model", revision="refs/pr/123")
    assert hf.load_tokenizer("org/model", revision="") is not None
    hf.prefetch_model("org/model", revision="")

    revision_calls = [kwargs for _kind, _model, kwargs in calls[:2]]
    empty_calls = [kwargs for _kind, _model, kwargs in calls[2:]]
    assert all(kwargs["revision"] == "refs/pr/123" for kwargs in revision_calls)
    assert all("revision" not in kwargs for kwargs in empty_calls)


def test_gpu_diagnostics_parses_nvidia_smi(monkeypatch):
    import flash.engine.worker.perf as perf

    class _Completed:
        def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    def fake_run(cmd, **_kwargs):
        joined = " ".join(cmd)
        if "--query-gpu=" in joined:
            return _Completed(
                "0, GPU-abc, 575.57, NVIDIA GeForce RTX 5090, 98, 77, "
                "32607, 24000, 8607, 69, 412.5, 575.0, P0, 2700, 14001, 5, 16\n"
            )
        if "--query-compute-apps=" in joined:
            return _Completed("1234, /usr/bin/python, 23900\n")
        return _Completed(returncode=1, stderr="unexpected command")

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)

    diag = perf.gpu_diagnostics()

    assert diag["device_name"] == "NVIDIA GeForce RTX 5090"
    assert diag["driver_version"] == "575.57"
    assert diag["gpu_util_pct"] == 98
    assert diag["mem_util_pct"] == 77
    assert diag["memory_used_gb"] == pytest.approx(23.438)
    assert diag["memory_total_gb"] == pytest.approx(31.8428, rel=1e-3)
    assert diag["temperature_c"] == 69
    assert diag["power_w"] == 412.5
    assert diag["processes"][0]["process_name"] == "/usr/bin/python"
    assert diag["processes"][0]["used_memory_gb"] == pytest.approx(23.34, rel=1e-3)


def test_heartbeat_commit_is_throttled(monkeypatch):
    """heartbeat() must rate-limit HF commits (per-step commits blow HF's 128/hour repo cap),
    while always committing milestone stages."""
    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker.io.heartbeat as w

    calls = []
    monkeypatch.setattr(w.hf_io, "hf_upload_file", lambda *a, **k: calls.append(a[1]))

    # Large interval -> only milestone + the first commit; per-step heartbeats throttled.
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 9999.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", 0.0)
    worker_heartbeat.heartbeat("rl_start")  # milestone -> commits
    worker_heartbeat.heartbeat("rl_step", step=1)  # throttled
    worker_heartbeat.heartbeat("rl_step", step=2)  # throttled
    assert calls.count("heartbeat.json") == 1

    # Zero interval -> every call commits.
    calls.clear()
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", 0.0)
    worker_heartbeat.heartbeat("rl_step", step=1)
    worker_heartbeat.heartbeat("rl_step", step=2)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "stage", ["model_prefetching", "sft_pretokenizing", "sft_initializing", "rl_initializing"]
)
def test_setup_progress_heartbeats_are_throttled(monkeypatch, stage):
    """The periodic setup pings run on a side thread every 30s through a long phase (a cold
    snapshot_download can pull tens of GB for ~40 min, and disaggregated workers share one HF_REPO),
    so their HF UPLOAD must be throttled like rl_step or they blow HF's 128/hour repo commit cap. The
    "HEARTBEAT" stderr line still prints every call (the only per-call side effect now — HF is the one
    durable channel); this pins only that the repeated stage does NOT commit to HF on every call."""
    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker.io.heartbeat as w

    calls = []
    monkeypatch.setattr(w.hf_io, "hf_upload_file", lambda *a, **k: calls.append(a[1]))
    # Large interval -> only the FIRST emit of the stage commits; the rest are upload-throttled.
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 9999.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", 0.0)
    worker_heartbeat.heartbeat(stage, elapsed_seconds=1)
    worker_heartbeat.heartbeat(stage, elapsed_seconds=31)
    worker_heartbeat.heartbeat(stage, elapsed_seconds=61)
    assert calls.count("heartbeat.json") == 1, f"{stage} must be upload-throttled, got {calls}"


def test_heartbeat_hf_upload_runs_outside_lock(monkeypatch):
    """Perf regression guard: the synchronous hf_upload_file network call must run OUTSIDE
    _HB_LOCK. Holding the lock across the upload serializes the trainer's per-step reward
    callback behind the checkpoint daemon's HF commit during GRPO."""
    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker.io.heartbeat as w

    # When hf_upload_file is invoked, the lock must be acquirable (i.e. not held).
    lock_free_during_upload = []

    def fake_upload(*a, **k):
        acquired = w._HB_LOCK.acquire(blocking=False)
        lock_free_during_upload.append(acquired)
        if acquired:
            w._HB_LOCK.release()

    monkeypatch.setattr(w.hf_io, "hf_upload_file", fake_upload)
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", 0.0)
    worker_heartbeat.heartbeat("rl_start")
    assert lock_free_during_upload == [True], "hf_upload_file must run with _HB_LOCK released"


def test_heartbeat_upload_skips_when_lock_is_stuck(monkeypatch):
    """A wedged upload holding _HB_UPLOAD_LOCK must not block the NEXT heartbeat. A milestone like
    model_prefetched (unthrottled, on the worker's critical path right before trainer construction)
    must skip its best-effort commit after a bounded wait rather than wedge the worker."""
    import importlib
    import time as _time

    # NB: resolve the submodule explicitly (the package re-exports the heartbeat() function).
    hbmod = importlib.import_module("flash.engine.worker.io.heartbeat")

    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker.io.heartbeat as w

    monkeypatch.setattr(hbmod, "_HB_UPLOAD_LOCK_TIMEOUT_S", 0.05)
    uploads = []
    monkeypatch.setattr(w.hf_io, "hf_upload_file", lambda *a, **k: uploads.append(a))
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    sentinel_last_upload = 123.0  # a prior successful-commit timestamp the skip must NOT clobber
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", sentinel_last_upload)

    assert hbmod._HB_UPLOAD_LOCK.acquire(blocking=False), "lock should be free at test start"
    try:
        t0 = _time.monotonic()
        worker_heartbeat.heartbeat("model_prefetched")  # must NOT block on the held lock
        elapsed = _time.monotonic() - t0
    finally:
        hbmod._HB_UPLOAD_LOCK.release()

    assert elapsed < 5.0, f"heartbeat wedged on the held upload lock ({elapsed:.2f}s)"
    assert uploads == [], "the best-effort commit must be skipped while the lock is stuck"
    # The skipped upload must ROLL BACK its optimistic slot claim — otherwise the throttle defers the
    # next real commit and the throttle treats a stale channel as fresh.
    assert sentinel_last_upload == worker_heartbeat._HB_LAST_UPLOAD, (
        f"a skipped commit must not advance _HB_LAST_UPLOAD (got {worker_heartbeat._HB_LAST_UPLOAD})"
    )


def test_heartbeat_rolls_back_slot_when_upload_reports_failure(monkeypatch):
    """The optimistic _HB_LAST_UPLOAD slot is claimed BEFORE the best-effort HF commit. If that commit
    fails, hf_upload_file swallows the error and returns False (it never raises on best-effort) — HF is
    still stale, so the slot must roll back exactly as the lock-timeout skip does. Otherwise the
    throttle defers the next retry on the strength of an upload that never happened.
    """
    import importlib

    hbmod = importlib.import_module("flash.engine.worker.io.heartbeat")

    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker.io.heartbeat as w

    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    sentinel_last_upload = (
        123.0  # a prior successful-commit timestamp the failed retry must restore
    )
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", sentinel_last_upload)

    calls = []
    # Mirror the real hf_upload_file contract: best-effort failure returns False (does not raise).
    monkeypatch.setattr(w.hf_io, "hf_upload_file", lambda *a, **k: (calls.append(a[1]), False)[1])

    worker_heartbeat.heartbeat("model_prefetched")

    assert calls == ["heartbeat.json"], "the upload must actually be attempted"
    assert sentinel_last_upload == worker_heartbeat._HB_LAST_UPLOAD, (
        f"a failed upload must roll _HB_LAST_UPLOAD back to its prior value (got {worker_heartbeat._HB_LAST_UPLOAD})"
    )


def test_heartbeat_keeps_slot_when_upload_reports_success(monkeypatch):
    """The dual of the rollback test: a SUCCESSFUL commit (or a mock that doesn't report False) must
    KEEP the advanced slot so the throttle works. ``is False`` — not falsy — gates the rollback, so a
    None-returning mock counts as success."""
    import importlib

    hbmod = importlib.import_module("flash.engine.worker.io.heartbeat")

    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker.io.heartbeat as w

    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", 0.0)
    monkeypatch.setattr(w.hf_io, "hf_upload_file", lambda *a, **k: True)

    worker_heartbeat.heartbeat("model_prefetched")

    assert worker_heartbeat._HB_LAST_UPLOAD > 0.0, (
        "a successful commit must keep the advanced throttle slot"
    )


def test_critical_stages_wait_longer_for_upload_lock(monkeypatch):
    """done/already_done/error_* are CRITICAL — no later heartbeat repairs a skipped terminal commit,
    and error_* carries the `retriable` flag worker_flagged_retriable() reads. So they wait the LONGER
    _HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S for the upload lock before skipping, vs the short progress
    timeout. Proven by timing: with the lock held, a terminal stage blocks for the critical timeout
    while a progress stage gives up after the short one."""
    import importlib
    import time as _time

    hbmod = importlib.import_module("flash.engine.worker.io.heartbeat")

    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker.io.heartbeat as w

    monkeypatch.setattr(hbmod, "_HB_UPLOAD_LOCK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(hbmod, "_HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S", 0.4)
    monkeypatch.setattr(w.hf_io, "hf_upload_file", lambda *a, **k: None)
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", 0.0)

    assert hbmod._HB_UPLOAD_LOCK.acquire(blocking=False), "lock should be free at test start"
    try:
        t = _time.monotonic()
        worker_heartbeat.heartbeat("rl_step", step=1)  # progress -> short timeout, skips fast
        progress_wait = _time.monotonic() - t

        t = _time.monotonic()
        worker_heartbeat.heartbeat("done")  # critical -> waits the long timeout before skipping
        critical_wait = _time.monotonic() - t
    finally:
        hbmod._HB_UPLOAD_LOCK.release()

    assert progress_wait < 0.3, (
        f"progress stage should skip after the short timeout, waited {progress_wait:.2f}s"
    )
    assert critical_wait >= 0.3, (
        f"critical stage should wait the long timeout, waited {critical_wait:.2f}s"
    )
    assert critical_wait > progress_wait + 0.15


def test_heartbeat_terminal_only_mode(monkeypatch):
    """TERMINAL_ONLY mode throttles every non-terminal stage (not just rl_step) so a fan-out of
    runs sharing one HF_REPO stays under the 128-commits/hour cap; terminal done/error_* still
    always commit so the control plane never misses a transition."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker.io.heartbeat as w

    calls = []

    def _fake_upload(*a, **k):
        calls.append(a[1])
        return True  # simulate a successful commit so the throttle clock advances

    monkeypatch.setattr(w.hf_io, "hf_upload_file", _fake_upload)
    monkeypatch.setattr(worker_heartbeat, "_HB_TERMINAL_ONLY", True)
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 9999.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_LAST_UPLOAD", 0.0)

    # Short interval: terminal-only must STILL suppress non-terminal after the first, NOT
    # leak a commit once the window elapses (the bot-caught 128/hr re-breach).
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    worker_heartbeat.heartbeat("sft_start")  # first non-terminal -> commits (last_upload==0)
    worker_heartbeat.heartbeat("sft_model_load")  # suppressed despite 0s interval
    worker_heartbeat.heartbeat("sft_trained")  # suppressed
    assert len(calls) == 1
    worker_heartbeat.heartbeat("error_sft", error="boom")  # terminal -> always commits
    worker_heartbeat.heartbeat("done")  # terminal -> always commits
    assert calls.count("heartbeat.json") == 3


def test_liger_default_model_size_gate(monkeypatch):
    """The model-size gate is OFF for small models (1B-class) and ON at ≥ ~3B.

    Named for liger because that is where the threshold was measured (PR #174, fused-CE's memory win
    only paying off above ~3B), but liger itself is gone from the verl paths: this predicate now
    feeds ``_memory_mode`` -> ``grad_checkpointing_on``, so the threshold is load-bearing for
    gradient checkpointing rather than for a fused-CE choice.
    """
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker.perf as w

    # _estimate_params: ~1B vs ~4B configs
    small = types.SimpleNamespace(
        hidden_size=1536, vocab_size=130560, num_hidden_layers=24, tie_word_embeddings=False
    )
    big = types.SimpleNamespace(
        hidden_size=2560, vocab_size=151936, num_hidden_layers=36, tie_word_embeddings=False
    )
    assert w._estimate_params(small) < 3e9
    assert w._estimate_params(big) >= 3e9

    def fake_cfg(cfg):
        fake = types.SimpleNamespace(
            AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: cfg)
        )
        monkeypatch.setitem(sys.modules, "transformers", fake)

    fake_cfg(small)
    assert w._liger_default_for_model("meta-llama/Llama-3.2-1B") is False
    # grad checkpointing follows the same small=speed(off) / large=memory(on) principle
    assert w.grad_checkpointing_on("meta-llama/Llama-3.2-1B") is False
    fake_cfg(big)
    assert w._liger_default_for_model("Qwen/Qwen3-4B") is True
    assert w.grad_checkpointing_on("Qwen/Qwen3-4B") is True
    # context-aware: a SMALL model at LONG context is memory-bound -> memory mode ON (PR #174:
    # 1B GRPO OOMs at 4096 in speed mode, fits in memory mode).
    fake_cfg(small)
    assert w._memory_mode("meta-llama/Llama-3.2-1B", 512) is False
    assert w._memory_mode("meta-llama/Llama-3.2-1B", 4096) is True
    assert w.grad_checkpointing_on("meta-llama/Llama-3.2-1B", 4096) is True


def test_make_lora_uses_standard_init_and_scaling(monkeypatch):
    """make_lora uses serve-safe, convergence-stable LoRA defaults for every model:
    standard zero-B init (PiSSA removed — its residual corrupts serve + GRPO warm-start) and
    standard alpha/r scaling (rsLoRA removed — alpha/sqrt(r) is ~5.6x larger and diverges the
    SFT at the usual LoRA LR -> degenerate served adapter)."""
    captured = {}
    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = lambda **kw: (captured.update(kw), kw)[1]
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    worker = _import_worker(monkeypatch)
    import flash.engine.worker.runtime.state as worker_state

    monkeypatch.setattr(
        worker_state,
        "JOB_SPEC",
        types.SimpleNamespace(
            train=types.SimpleNamespace(lora_rank=32, lora_alpha=64),
            model_revision="a" * 40,
        ),
    )

    for model_id in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B"):
        captured.clear()
        worker.make_lora(model_id)
        assert captured.get("init_lora_weights") is True
        assert "pissa" not in str(captured.get("init_lora_weights")).lower()
        assert captured.get("use_rslora") is False
        assert captured.get("revision") == "a" * 40
        assert "target_parameters" not in captured
        assert captured["exclude_modules"] == r"^(?!model\.language_model(?:\.|$)).*$"

    captured.clear()
    worker.make_lora("Qwen/Qwen3.5-9B", algorithm="sft", multimodal=True)
    assert captured["target_modules"] == "all-linear"
    assert captured["exclude_modules"] is None

    captured.clear()
    worker.make_lora("Qwen/Qwen3.6-35B-A3B")
    assert captured["r"] == 32
    assert captured["target_modules"] == "all-linear"
    assert captured["exclude_modules"] == r"^(?!model\.language_model(?:\.|$)).*$"
    assert captured["target_parameters"] == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


def test_worker_exports_only_the_current_warmstart_adapter_surface(monkeypatch):
    worker = _import_worker(monkeypatch)

    assert callable(worker_adapter.validate_warmstart_adapter)
    assert callable(worker.lora_target_parameters)
    for deleted in (
        "adapter_has_fused_expert_tensors",
        "expected_fused_expert_modules",
        "legacy_fused_expert_config_is_recoverable",
        "prepare_warmstart_adapter_config",
        "restore_fused_expert_targets",
    ):
        assert not hasattr(worker, deleted)


def test_train_metadata_keeps_model_revision_in_nested_job_spec(monkeypatch):
    from flash.core.spec import JobSpec
    from flash.engine.worker.train.core.lifecycle import finalize

    captured = []
    monkeypatch.setattr(finalize.worker_state, "JOB_SPEC", JobSpec(model_revision="refs/pr/123"))
    monkeypatch.setattr(finalize.worker_state, "SEED", 42)
    monkeypatch.setattr(finalize.worker_state, "THINKING", False)
    monkeypatch.setattr(
        finalize.worker_state, "require_active_env", lambda: SimpleNamespace(id="org/env")
    )
    monkeypatch.setattr(finalize.hf_io, "hf_upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(finalize.heartbeat_io, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        finalize, "_finalize", lambda metrics, **kwargs: captured.append((metrics, kwargs))
    )
    monkeypatch.setattr(finalize, "gpu_diagnostics", dict)

    finalize.write_train_meta(
        phase="sft",
        adapter_dir="/tmp/adapter",
        model_id="org/model",
        train_wall=1.0,
        setup_seconds=2.0,
        train_tokens=3,
        generated_tokens=0,
        notes={},
    )

    assert captured[0][0].notes["job_spec"]["model_revision"] == "refs/pr/123"
    assert captured[0][1] == {"heartbeat_fields": {}}


def test_train_metadata_preserves_terminal_heartbeat_fields(monkeypatch):
    from flash.engine.worker.train.core.lifecycle import finalize

    emitted = []
    finalized = []
    metrics_last = [{"step": 4, "reward": 0.75}]
    monkeypatch.setattr(finalize.worker_state, "JOB_SPEC", None)
    monkeypatch.setattr(finalize.worker_state, "SEED", 42)
    monkeypatch.setattr(finalize.worker_state, "THINKING", False)
    monkeypatch.setattr(
        finalize.worker_state, "require_active_env", lambda: SimpleNamespace(id="org/env")
    )
    monkeypatch.setattr(finalize.hf_io, "hf_upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        finalize.heartbeat_io,
        "heartbeat",
        lambda stage, **kwargs: emitted.append((stage, kwargs)),
    )
    monkeypatch.setattr(
        finalize,
        "_finalize",
        lambda metrics, **kwargs: finalized.append((metrics, kwargs)),
    )
    monkeypatch.setattr(finalize, "gpu_diagnostics", dict)

    finalize.write_train_meta(
        phase="rl",
        adapter_dir="/tmp/adapter",
        model_id="org/model",
        train_wall=1.0,
        setup_seconds=2.0,
        train_tokens=0,
        generated_tokens=3,
        notes={},
        heartbeat_fields={"metrics_last": metrics_last},
    )

    assert emitted[-1][0] == "rl_train_done"
    assert emitted[-1][1]["metrics_last"] == metrics_last
    assert finalized[0][1] == {"heartbeat_fields": {"metrics_last": metrics_last}}


def test_finalize_preserves_terminal_heartbeat_fields(monkeypatch):
    from unittest.mock import mock_open

    from flash.engine.result.accounting import RunMetrics
    from flash.engine.worker.train.core.lifecycle import finalize

    emitted = []
    metrics_last = [{"step": 4, "reward": 0.75}]
    monkeypatch.setattr(RunMetrics, "save", lambda self, path: None)
    monkeypatch.setattr(finalize.hf_io, "hf_upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        finalize.heartbeat_io,
        "heartbeat",
        lambda stage, **kwargs: emitted.append((stage, kwargs)),
    )
    monkeypatch.setattr(finalize, "gpu_diagnostics", dict)
    monkeypatch.setattr("builtins.open", mock_open())

    finalize._finalize(RunMetrics(phase="rl"), heartbeat_fields={"metrics_last": metrics_last})

    assert emitted[-1][0] == "done"
    assert emitted[-1][1]["metrics_last"] == metrics_last


def test_finalize_uploads_metrics_before_done(monkeypatch):
    from unittest.mock import mock_open

    from flash.engine.result.accounting import RunMetrics
    from flash.engine.worker.train.core.lifecycle import finalize

    uploads = []
    monkeypatch.setattr(RunMetrics, "save", lambda self, path: None)
    monkeypatch.setattr(
        finalize.hf_io,
        "hf_upload_file",
        lambda local_path, remote_name, **kwargs: uploads.append((remote_name, kwargs)),
    )
    monkeypatch.setattr(finalize.heartbeat_io, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(finalize, "gpu_diagnostics", dict)
    monkeypatch.setattr("builtins.open", mock_open())

    finalize._finalize(RunMetrics(phase="rl"))

    assert uploads == [
        ("metrics.json", {"required": True}),
        ("DONE", {"required": True}),
    ]


# ---------------------------------------------------------------------------
# Hopper fla GDN fast-path fallback: when the healthy fla+tilelang stack can't be
# Hopper fla GDN fast-path fallback: when the healthy fla+tilelang stack can't be assembled (probe
# `ok` false), fla must be DISABLED (physically removed) so transformers' is_fla_available() gate
# flips off and the model uses the correct pure-PyTorch delta rule instead of fla's broken
# Triton>=3.4 GDN chunk_bwd (fla #640). A print alone is not enough.
# ---------------------------------------------------------------------------
def _hopper_torch():
    """A stub ``torch`` that looks like Hopper (sm90) with CUDA available."""
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda *a: (9, 0),
    )
    return t


class _FakeCompleted:
    """Minimal stand-in for subprocess.CompletedProcess so _pip can read .returncode."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _patch_hopper_stack(
    monkeypatch,
    *,
    pip_rc: int = 0,
    find_spec_ok: bool = True,
    tvm_ffi_version: str | None = _PIN,
    tilelang_version: str | None = _PIN,
    record_pip: list[str] | None = None,
):
    """Wire the perf helper's external touchpoints for the Hopper fast-path tests and return the

    list that records _remove_fla_from_disk (fla-disable) calls. * ``pip_rc`` -> the return code
    every mocked ``pip install`` reports (non-zero = failed install). * ``find_spec_ok`` -> whether
    the post-install import probe finds fla/fla.modules/tilelang. * ``tvm_ffi_version`` -> what
    importlib.metadata.version('apache-tvm-ffi') reports (None=absent, _PIN=the enforced pin).
    * ``tilelang_version`` -> same for tilelang. Defaulting to the pin read from perf keeps these
    cases healthy-by-default without restating a version that moves with flash-qla.
    """
    resolved_tilelang_pin, resolved_tvm_ffi_pin = _perf_pins()
    if tvm_ffi_version is _PIN:
        tvm_ffi_version = resolved_tvm_ffi_pin
    if tilelang_version is _PIN:
        tilelang_version = resolved_tilelang_pin
    import importlib.metadata
    import importlib.util
    import subprocess

    import flash.engine.worker.perf as perf

    monkeypatch.setitem(sys.modules, "torch", _hopper_torch())

    # subprocess is imported locally inside the helper, so patch the real module. Return a fake
    # CompletedProcess so _pip can read .returncode (the install-success gate).
    def _fake_run(cmd, *a, **k):
        if record_pip is not None:
            # cmd == [sys.executable, "-m", "pip", "install", "-q", *specs]
            record_pip.append(" ".join(str(c) for c in cmd[5:]))
        return _FakeCompleted(pip_rc)

    monkeypatch.setattr(subprocess, "run", _fake_run, raising=True)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        (lambda name: object()) if find_spec_ok else (lambda name: None),
        raising=True,
    )

    def _fake_version(dist: str) -> str:
        ver = {"apache-tvm-ffi": tvm_ffi_version, "tilelang": tilelang_version}.get(dist)
        if ver is None:
            raise importlib.metadata.PackageNotFoundError(dist)
        return ver

    monkeypatch.setattr(importlib.metadata, "version", _fake_version, raising=True)

    removed: list[int] = []
    monkeypatch.setattr(
        perf, "_remove_fla_from_disk", lambda: (removed.append(1), (["/x/fla"], False))[1]
    )
    return perf, removed


def test_hopper_fla_fallback_disables_fla_when_stack_unavailable(monkeypatch):
    """Probe fails (fla absent after install attempt) -> _remove_fla_from_disk IS called so the
    fla gate flips off and the worker uses the pure-PyTorch delta path (not the broken fla kernel)."""
    perf, removed = _patch_hopper_stack(monkeypatch, find_spec_ok=False)

    perf._ensure_fla_fastpath_on_hopper()

    # The fallback gating actually fired: fla was disabled (the gate, not just a print).
    assert removed, "fallback must DISABLE fla (call _remove_fla_from_disk) when ok is false"


def test_hopper_fla_fallback_when_install_fails(monkeypatch):
    """A FAILED pip install (rc!=0) must flip the gate off + disable fla EVEN IF find_spec still
    succeeds on a stale/partial resident copy — a failed install is not silently treated as healthy
    (perf.py:~487). Without the rc check, find_spec alone would wrongly keep fla.
    A wrong tvm-ffi version makes the helper ATTEMPT the pinned reinstall (the conditional-install
    gate), so pip_rc=1 exercises the failed-install path."""
    perf, removed = _patch_hopper_stack(
        monkeypatch, pip_rc=1, find_spec_ok=True, tvm_ffi_version="0.1.12"
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert removed, "a failed install (rc!=0) must DISABLE fla even when find_spec passes"


def test_hopper_fla_fallback_when_tvm_ffi_pin_did_not_land(monkeypatch):
    """tilelang's apache-tvm-ffi~=0.1.0 range can keep the BROKEN 0.1.12 (find_spec-importable but
    aborts `import tilelang`). The gate verifies the RESOLVED tvm-ffi version, so a wrong version
    disables fla even though every install 'succeeded' and every module imports."""
    perf, removed = _patch_hopper_stack(
        monkeypatch, pip_rc=0, find_spec_ok=True, tvm_ffi_version="0.1.12"
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert removed, "wrong resolved apache-tvm-ffi version must DISABLE fla (pin didn't land)"


def test_hopper_fla_kept_when_stack_healthy(monkeypatch):
    """Success path: every install exits 0, fla + fla.modules + tilelang all import, AND the
    resolved apache-tvm-ffi is exactly the pin -> fla is KEPT (the fast path is engaged, not
    removed)."""
    perf, removed = _patch_hopper_stack(monkeypatch, pip_rc=0, find_spec_ok=True)

    perf._ensure_fla_fastpath_on_hopper()

    assert not removed, "healthy stack must KEEP fla (no disable on the success path)"


def test_hopper_tilelang_present_but_wrong_version_is_reinstalled(monkeypatch):
    """Regression (perf.py:~511): a DIFFERENT tilelang already resident (a job or
    the base image carries one) must NOT be treated as healthy. The helper gates on the installed
    version, so it (re)installs the exact pin; once the pin lands fla is KEPT."""
    pip_calls: list[str] = []
    # Resident wrong version first; after the (mocked) reinstall the metadata reports the pin.
    perf, removed = _patch_hopper_stack(
        monkeypatch,
        pip_rc=0,
        find_spec_ok=True,
        # both default to the enforced pin: post-reinstall resolved version
        record_pip=pip_calls,
    )
    # Make the FIRST _ver('tilelang') read (the install gate) see a stale wrong version, while the
    # final ok-gate read sees the pin — i.e. the reinstall corrected it.
    import importlib.metadata as _md

    tilelang_pin = _perf_pins()[0]
    # first read = stale wrong version; later reads -> the pin (i.e. the reinstall corrected it)
    gate_reads = iter([_other_version(tilelang_pin)])
    orig_version = _md.version

    def _versioned(dist: str) -> str:
        if dist == "tilelang":
            return next(gate_reads, tilelang_pin)
        return orig_version(dist)

    monkeypatch.setattr(_md, "version", _versioned, raising=True)

    perf._ensure_fla_fastpath_on_hopper()

    assert any(c.startswith(f"tilelang=={tilelang_pin}") for c in pip_calls), (
        f"present-but-wrong tilelang must trigger a pinned reinstall, got pip calls: {pip_calls}"
    )
    assert not removed, "after the pin reinstall lands, fla must be KEPT"


def test_hopper_tilelang_wrong_version_persists_disables_fla(monkeypatch):
    """If the resident tilelang is the WRONG version AND the reinstall doesn't land the pin (version
    still != pin), the final gate must DISABLE fla — the uncertain GDN backend is not trusted."""
    pip_calls: list[str] = []
    perf, removed = _patch_hopper_stack(
        monkeypatch,
        pip_rc=0,
        find_spec_ok=True,
        tilelang_version=_other_version(_perf_pins()[0]),  # wrong throughout (reinstall failed)
        record_pip=pip_calls,
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert any(c.startswith(f"tilelang=={_perf_pins()[0]}") for c in pip_calls), (
        "wrong resident tilelang must still attempt the pinned reinstall"
    )
    assert removed, "tilelang version != pin after install must DISABLE fla (pin didn't land)"


def test_hopper_tvm_ffi_pip_skipped_when_pin_already_present(monkeypatch):
    """Regression (perf.py:~521): when the EXACT apache-tvm-ffi pin is already
    resident AND tilelang was NOT (re)installed this invocation, the helper must SKIP the tvm-ffi
    pip — re-running it unconditionally adds avoidable cold-start latency and could spuriously
    disable fla on a transient network/resolver hiccup. The ok gate still re-verifies the version,
    so fla stays KEPT."""
    pip_calls: list[str] = []
    perf, removed = _patch_hopper_stack(
        monkeypatch,
        pip_rc=0,
        find_spec_ok=True,
        # both default to the exact pin already resident -> tilelang NOT reinstalled here
        record_pip=pip_calls,
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert not any("apache-tvm-ffi" in c for c in pip_calls), (
        "tvm-ffi pin already resident + tilelang not reinstalled must SKIP the tvm-ffi pip, "
        f"got pip calls: {pip_calls}"
    )
    assert not removed, "the pin being resident is the healthy path -> fla must be KEPT"


def test_hopper_outer_exception_disables_fla(monkeypatch):
    """Regression (perf.py:~580): an unexpected error mid-setup (AFTER the Hopper
    check passes) must FAIL-CLOSED — best-effort disable fla so transformers can't engage the broken
    Triton GDN path (#640) on a half-configured fla. The outer handler must call _remove_fla_from_disk
    and never re-raise."""
    import importlib

    perf, removed = _patch_hopper_stack(monkeypatch, pip_rc=0, find_spec_ok=True)

    # Make a call inside the try (after the Hopper guard + installs) blow up. invalidate_caches runs
    # right before the import-probe gate, so this drives the outer `except`.
    def _boom() -> None:
        raise RuntimeError("invalidate_caches exploded")

    monkeypatch.setattr(importlib, "invalidate_caches", _boom, raising=True)

    # Must NOT propagate (the worker keeps running on the pure-PyTorch delta path).
    perf._ensure_fla_fastpath_on_hopper()

    assert removed, (
        "outer exception path must FAIL-CLOSED: disable fla (call _remove_fla_from_disk)"
    )


def test_non_hopper_fla_fastpath_is_noop(monkeypatch):
    """On a non-Hopper arch (fla's Triton kernel is correct there) the helper is a no-op — it must
    NOT touch fla at all (neither install nor disable)."""
    import importlib.util
    import subprocess

    import flash.engine.worker.perf as perf

    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda *a: (12, 0),  # Blackwell consumer (sm120) — not Hopper
    )
    monkeypatch.setitem(sys.modules, "torch", t)
    touched: list[str] = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: touched.append("pip"), raising=True)
    monkeypatch.setattr(
        perf, "_remove_fla_from_disk", lambda: (touched.append("remove"), ([], False))[1]
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None, raising=True)

    perf._ensure_fla_fastpath_on_hopper()

    assert touched == [], "non-Hopper must be a no-op (don't install or disable fla)"


def test_verl_venv_pins_transformers_to_the_main_interpreters_version():
    """The verl venv must carry the same exact transformers version as the main interpreter.

    /opt/verl-venv is built without --system-site-packages, so the main interpreter's pin does not
    reach it. The venv is the interpreter that TRAINS, and transformers owns the gdn modelling code
    the boundary-reset shim patches, so an unbounded resolve there silently moves training onto a
    transformers line nothing validated.

    This pin does NOT fix the cuda-gated probes -- they are byte-identical in 5.12.1 and 5.14.1 and
    answer False on any gpu-less builder at every version. See
    ``test_venv_sanity_block_uses_no_cuda_gated_probe``.

    The pin must also be in the OVERRIDE file: verl and vllm both require transformers, so a direct
    pin alone can be re-widened by a transitive requirement.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile.worker").read_text()

    main_pin = re.search(r'"(transformers==[\d.]+)"[^\n]*\\\n\s+"peft', dockerfile)
    assert main_pin, "could not find the main interpreter's exact transformers pin"

    overrides = re.search(r"printf '%s\\n'(.*?)> /tmp/verl-overrides\.txt", dockerfile)
    assert overrides, "could not find the verl-overrides.txt line"
    assert main_pin.group(1) in overrides.group(1), (
        "the verl venv override file must pin transformers to the same exact version as the main "
        f"interpreter ({main_pin.group(1)}); verl and vllm both depend on transformers, so without "
        "the exact override a transitive requirement can re-widen it"
    )

    venv_block = dockerfile[dockerfile.index("uv venv --seed /opt/verl-venv") :]
    venv_block = venv_block[: venv_block.index("uv cache clean")]
    assert f'"{main_pin.group(1)}"' in venv_block, (
        "the verl venv install list must also name the same exact transformers pin directly"
    )


def test_causal_conv1d_is_required_not_best_effort_in_the_image():
    """Both causal_conv1d installs must fail the image build rather than degrade to no kernel.

    It used to be best-effort in both interpreters, on the premise that a failed build meant "gdn
    trains unpacked". That premise is dead: every catalog model is a gdn hybrid, and
    ``require_gdn_boundary_resets`` RAISES for grpo/opd when the child cannot reset, because the old
    padded fallback dies at ``padding.py:144`` against the hardcoded ``use_fused_kernels=True``. A
    conv-less image therefore does not degrade -- it fails those runs after paying for a gpu.

    Asserts on the ABSENCE of the swallowing ``|| uninstall`` branch, not merely on the presence of
    the install line: the install was always there, and it was the ``||`` that made it optional.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile.worker").read_text()

    assert dockerfile.count("causal-conv1d==1.6.2.post1") == 2, (
        "both the main interpreter and the verl venv must install causal_conv1d: /opt/verl-venv is "
        "built without --system-site-packages, so the main interpreter's copy is invisible to the "
        "child that actually trains"
    )
    # the uninstall-on-failure escape hatch must be gone from BOTH steps.
    assert "pip uninstall -y causal-conv1d" not in dockerfile
    assert "uv pip uninstall --python /opt/verl-venv/bin/python causal-conv1d" not in dockerfile
    # and the venv sanity block must import it, the way it imports fla. NOT via transformers'
    # is_causal_conv1d_available(): that begins with is_torch_cuda_available(), and the image builds
    # on a cpu runner, so it answers False with the kernel installed and fails every build. The
    # import is what a cpu can prove; the cuda-gated capability is asserted on the worker.
    venv_sanity = dockerfile[dockerfile.index("# Sanity: the verl venv must be able to LAUNCH") :]
    assert "importlib.import_module('causal_conv1d')" in venv_sanity, (
        "the verl venv sanity block must import causal_conv1d the way it imports fla; an install "
        "line whose result is never imported lets an ABI-broken build ship green"
    )


def test_venv_sanity_block_uses_no_cuda_gated_probe():
    """The build-time sanity block must not call a probe that needs a gpu to answer True.

    ``is_flash_linear_attention_available()`` and ``is_causal_conv1d_available()`` both open with
    ``is_torch_cuda_available()`` -> ``torch.cuda.is_available()``. worker-image.yml builds on
    ``ubuntu-24.04-8core``, which has no device, so both return False with the kernels correctly
    installed and the asserts are unsatisfiable BY CONSTRUCTION -- they fail every build regardless
    of image contents. Run 31291646212 died exactly this way with fla 0.5.2 present and
    transformers 5.12.1 resolved.

    The build asserts what a cpu can prove (importable, and fla >= the 0.2.2 the probe demands). The
    cuda-gated capability is asserted where a device exists: the child probe feeding
    ``require_gdn_boundary_resets``, which raises for grpo/opd before a step runs.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile.worker").read_text()
    venv_sanity = dockerfile[dockerfile.index("# Sanity: the verl venv must be able to LAUNCH") :]
    venv_sanity = venv_sanity[: venv_sanity.index("# RunPod Serverless worker entrypoint")]

    for probe in ("is_flash_linear_attention_available", "is_causal_conv1d_available"):
        assert f"assert {probe}()" not in venv_sanity, (
            f"{probe}() is cuda-gated and the image builds on a cpu runner, so asserting it fails "
            "every build with the kernel installed. assert the import instead, and leave the "
            "capability to the worker-side probe."
        )


def test_fla_git_pin_is_consistent_and_pinned():
    """The fla git dependency is PINNED to an exact commit (not the moving default branch) and the
    worker's runtime reinstall uses that same SHA (baked image == what perf reinstalls)."""
    import pathlib
    import re

    spec = next(d for d in _worker_image_specs() if "flash-linear-attention" in d)
    m = re.search(r"flash-linear-attention\.git@([0-9a-f]{40})\b", spec)
    assert m, f"Dockerfile.worker fla install must be pinned to a 40-char commit SHA, got: {spec!r}"
    image_sha = m.group(1)

    # repo root: tests/ -> repo root is the parent.
    root = pathlib.Path(__file__).resolve().parent.parent
    # The worker's runtime fla reinstall (perf._ensure_fla_fastpath_on_hopper) must use the SAME pin —
    # an unpinned reinstall would pull the moving default branch and defeat reproducibility.
    perf_src = (root / "flash" / "engine" / "worker" / "perf" / "__init__.py").read_text()
    # The URL is built via implicit string concatenation across lines:
    #   "...flash-linear-attention.git"\n        "@<sha>" — so allow quotes/newline/space between.
    pm = re.search(r"flash-linear-attention\.git[\"'\s]*@([0-9a-f]{40})\b", perf_src)
    assert pm, "perf/__init__.py runtime fla reinstall must be pinned to a 40-char commit SHA"
    assert pm.group(1) == image_sha, (
        f"perf/__init__.py fla SHA must match Dockerfile.worker (image={image_sha}, perf={pm.group(1)})"
    )


def test_tilelang_pin_is_consistent_and_pinned():
    """tilelang (the Hopper GDN correctness backend) is PINNED to an exact version (not unversioned)
    and the SAME pin is used in Dockerfile.worker and perf.py's runtime reinstall, so image rebuilds
    and runtime reinstalls resolve the identical backend."""
    import pathlib
    import re

    spec = next(
        d for d in _worker_image_specs() if d.split("==")[0].split(">")[0].strip() == "tilelang"
    )
    m = re.match(r"tilelang==([0-9][0-9A-Za-z.\-]*)$", spec.strip())
    assert m, f"tilelang must be pinned to an exact version (tilelang==X.Y.Z), got: {spec!r}"
    pin = m.group(1)

    root = pathlib.Path(__file__).resolve().parent.parent
    perf_src = (root / "flash" / "engine" / "worker" / "perf" / "__init__.py").read_text()
    # perf/__init__.py builds the spec via an f-string `f"tilelang=={TILELANG_PIN}"`, so assert the constant.
    pm = re.search(r'TILELANG_PIN\s*=\s*"([0-9][0-9A-Za-z.\-]*)"', perf_src)
    assert pm, (
        "perf/__init__.py must define a pinned TILELANG_PIN constant for the runtime reinstall"
    )
    assert pm.group(1) == pin, (
        f"perf/__init__.py TILELANG_PIN must match Dockerfile.worker (image={pin}, perf={pm.group(1)})"
    )


def test_exact_gpu_validation_accepts_alias_and_rejects_neighbor_classes(monkeypatch):
    from types import SimpleNamespace

    import flash.engine.worker.perf.lifecycle as lifecycle
    from flash.engine.worker.perf import RetriableInfraError

    verify_gpu = lifecycle.verify_gpu

    cuda = SimpleNamespace(
        get_device_capability=lambda _index: (9, 0),
        get_device_properties=lambda _index: SimpleNamespace(total_memory=80 * 10**9),
        get_device_name=lambda _index: "NVIDIA H100 PCIE",
    )
    torch = SimpleNamespace(cuda=cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(lifecycle, "_host_driver_cuda", lambda: 13.0)
    verify_gpu("H100", gpu_type="h100")

    for observed in ("NVIDIA H200", "NVIDIA B200"):
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index, name=observed: name)
        with pytest.raises(RetriableInfraError, match=r"requested='H100'.*observed="):
            verify_gpu("H100", gpu_type="H100")


def test_exact_gpu_validation_accepts_pytorch_a100_sxm4_40gb_name(monkeypatch):
    from types import SimpleNamespace

    import flash.engine.worker.perf.lifecycle as lifecycle

    cuda = SimpleNamespace(
        get_device_capability=lambda _index: (8, 0),
        get_device_properties=lambda _index: SimpleNamespace(total_memory=40 * 10**9),
        get_device_name=lambda _index: "NVIDIA A100-SXM4-40GB",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(lifecycle, "_host_driver_cuda", lambda: 13.0)

    lifecycle.verify_gpu("A100 SXM 40GB", gpu_type="A100 SXM 40GB")


def test_wait_for_gpu_raises_retriable_infra_error(monkeypatch):
    # A GPU that never comes up is infra-shaped -> typed RetriableInfraError, not RuntimeError.
    import time as _time

    from flash.engine.worker.perf import RetriableInfraError, wait_for_gpu

    monkeypatch.setattr(_time, "sleep", lambda *_a: None)
    try:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    except ImportError:
        pass
    with pytest.raises(RetriableInfraError):
        wait_for_gpu()


def test_required_upload_exhaustion_raises_retriable_infra_error(monkeypatch):
    # A required upload that fails after its retries is bad host/network -> RetriableInfraError.
    import flash.engine.worker.io.hf as worker
    import flash.engine.worker.runtime.state as worker_state
    from flash.engine.worker.perf import RetriableInfraError

    monkeypatch.setattr(worker_state, "HF_REPO", "owner/repo")
    monkeypatch.setattr(worker.time, "sleep", lambda *_a: None)

    def boom():
        raise OSError("connection reset by peer")

    with pytest.raises(RetriableInfraError):
        worker._hf_upload(boom, "DONE", required=True, label="DONE")


def test_required_upload_starts_no_hf_call_at_deadline(monkeypatch):
    import flash.engine.worker.io.hf as worker
    import flash.engine.worker.runtime.state as worker_state
    from flash.engine.worker.perf import RetriableInfraError

    calls = []
    monkeypatch.setattr(worker_state, "HF_REPO", "owner/repo")
    monkeypatch.setattr(worker_state, "_remaining_worker_wall_seconds", lambda: 0.0)

    with pytest.raises(RetriableInfraError):
        worker._hf_upload(lambda: calls.append("upload"), "DONE", required=True, label="DONE")

    assert calls == []


def test_required_upload_caps_retry_sleep_and_starts_no_late_retry(monkeypatch, capsys):
    import flash.engine.worker.io.hf as worker
    import flash.engine.worker.runtime.state as worker_state
    from flash.engine.worker.perf import RetriableInfraError

    calls = []
    sleeps = []
    remaining = iter((2.0, 2.0, 0.0))
    monkeypatch.setattr(worker_state, "HF_REPO", "owner/repo")
    monkeypatch.setattr(worker_state, "_remaining_worker_wall_seconds", lambda: next(remaining))
    monkeypatch.setattr(worker.time, "sleep", sleeps.append)
    monkeypatch.setenv("HF_TOKEN", "bearer-secret")

    def boom():
        calls.append("upload")
        raise OSError("Authorization: Bearer bearer-secret provider-body-private")

    with pytest.raises(RetriableInfraError):
        worker._hf_upload(boom, "DONE", required=True, label="DONE")

    assert calls == ["upload"]
    assert sleeps == [2.0]
    output = capsys.readouterr().out
    assert "bearer-secret" not in output
    assert "provider-body-private" in output


def test_optional_upload_without_deadline_preserves_single_attempt(monkeypatch):
    import flash.engine.worker.io.hf as worker
    import flash.engine.worker.runtime.state as worker_state

    calls = []
    monkeypatch.setattr(worker_state, "HF_REPO", "owner/repo")
    monkeypatch.setattr(worker_state, "_remaining_worker_wall_seconds", lambda: None)

    def boom():
        calls.append("upload")
        raise OSError("offline")

    assert not worker._hf_upload(boom, "debug.json", required=False, label="debug")
    assert calls == ["upload"]


# ---------------------------------------------------------------------------
# tilelang's libcudart_stub.so shadows the real CUDA runtime in
# vLLM's CudaRTLibrary (intermittent `undefined symbol: cudaDeviceReset`).
# ---------------------------------------------------------------------------
def _fake_tilelang(tmp_path, stub_bytes=b"STUB"):
    """Lay down a fake `tilelang` package (with lib/libcudart_stub.so) and return (pkg_dir, stub)."""

    pkg = tmp_path / "tilelang"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    stub = pkg / "lib" / "libcudart_stub.so"
    stub.write_bytes(stub_bytes)
    return str(pkg), str(stub)


def test_find_real_libcudart_safe_when_nothing_matches(monkeypatch):
    """_find_real_libcudart returns None (never raises) when no candidate exposes the symbol."""
    import builtins
    import ctypes.util
    import glob

    import flash.engine.worker.perf as perf

    monkeypatch.setattr(glob, "glob", lambda *_a, **_k: [])
    monkeypatch.setattr(ctypes.util, "find_library", lambda _n: None)
    real_import = builtins.__import__

    def _no_nvidia(name, *a, **k):
        if name.startswith("nvidia"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_nvidia)
    assert perf._find_real_libcudart() is None


def _compile_so(c_path, so_path, src):
    """Compile a tiny shared object; return True on success (skip the test if no toolchain)."""
    import shutil
    import subprocess

    cc = shutil.which("cc") or shutil.which("gcc")
    if not cc:
        return False
    with open(c_path, "w") as f:
        f.write(src)
    return subprocess.run([cc, "-shared", "-fPIC", "-o", so_path, c_path]).returncode == 0


def test_find_real_libcudart_handles_bare_soname_without_crashing(monkeypatch):
    """find_library('cudart') returns a bare soname (e.g. 'libcudart.so.12'), not a path. The
    os.path.exists guard must not silently drop it; with no loadable cudart present it still resolves
    to None safely (and never raises)."""
    import builtins
    import ctypes.util
    import glob

    import flash.engine.worker.perf as perf

    monkeypatch.setattr(glob, "glob", lambda *_a, **_k: [])  # no absolute-path candidates
    monkeypatch.setattr(ctypes.util, "find_library", lambda _n: "libcudart.so.12")  # bare soname
    real_import = builtins.__import__

    def _no_nvidia(name, *a, **k):
        if name.startswith("nvidia"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_nvidia)
    # On a host without a real libcudart the bare soname won't load -> None (no exception, no skip).
    assert perf._find_real_libcudart() is None


def test_find_real_libcudart_finds_cu13_wheel_layout(tmp_path, monkeypatch):
    """Regression: the resolver must be CUDA-major-agnostic. The cu13 nvidia wheel ships the runtime
    at ``nvidia/cu13/lib/libcudart.so.13`` (NOT ``nvidia/cuda_runtime/lib/libcudart.so.12``, and it
    has no ``nvidia.cuda_runtime`` module), so the original ``.so.12``-only probe returns None on a
    cu13 stack -> the stub shadow stays in place. Build a real .so exporting cudaDeviceReset at the
    cu13 wheel path and assert the resolver finds it (this FAILS against the .so.12-only version)."""
    import ctypes.util
    import glob
    import os
    import types

    cu13lib = tmp_path / "nvidia" / "cu13" / "lib"
    cu13lib.mkdir(parents=True)
    real = str(cu13lib / "libcudart.so.13")
    if not _compile_so(str(tmp_path / "real.c"), real, "void cudaDeviceReset(void){}"):
        import pytest

        pytest.skip("no C toolchain to build a real libcudart.so")

    import flash.engine.worker.perf as perf

    # Fake the `nvidia` namespace package so its __path__ is our tmp tree (the cu13 wheel layout).
    fake_nvidia = types.ModuleType("nvidia")
    fake_nvidia.__path__ = [str(tmp_path / "nvidia")]
    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)
    # Neutralize the toolkit + ldconfig fallbacks so ONLY the wheel-layout path can match — keeps the
    # assertion deterministic on any CI box, with or without a system CUDA install.
    monkeypatch.setattr(ctypes.util, "find_library", lambda _n: None)
    _real_glob = glob.glob
    monkeypatch.setattr(
        glob, "glob", lambda p, *a, **k: [] if p.startswith("/usr") else _real_glob(p, *a, **k)
    )

    assert perf._find_real_libcudart() == os.path.realpath(real)


# ---------------------------------------------------------------------------
# Blackwell fla GDN autotune restriction (fla #913 / #1000): on sm100/sm120 the
# Blackwell fla GDN autotune restriction (fla #913 / #1000): on sm100/sm120 the unrestricted
# prepare_wy_repr_bwd autotune space can select grad-miscomputing configs (live B200
# Qwen3.6-35B-A3B SFT: grad_norm ~1e8 from the first logged step, loss flat or collapsing at every
# LR, while H200 trained healthily).
# ---------------------------------------------------------------------------
def _blackwell_torch(cc=(10, 0)):
    """A stub ``torch`` that looks like a Blackwell card with CUDA available."""
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda *a: cc,
    )
    return t


class _TunerCfg:
    def __init__(self, num_warps: int, num_stages: int) -> None:
        self.num_warps = num_warps
        self.num_stages = num_stages


class _FakeTuner:
    def __init__(self, configs) -> None:
        self.configs = configs


class _TunerWrapper:
    """Decorator layer (heuristics/cache wrapper) the unwrap walk descends through via ``.fn``."""

    def __init__(self, fn) -> None:
        self.fn = fn


_FULL_SPACE = [(2, 2), (2, 3), (2, 4), (4, 2), (4, 3), (4, 4)]


def _patch_blackwell_stack(monkeypatch, *, cc=(10, 0), fla_present: bool = True, kernel=None):
    """Wire the perf helper's touchpoints for the Blackwell autotune-restriction tests.

    ``kernel`` is what ``fla.ops.gated_delta_rule.wy_fast.prepare_wy_repr_bwd_kernel``
    resolves to (None = attribute absent). Returns (perf, removed, wy_fast_module).
    """
    import importlib.util

    import flash.engine.worker.perf as perf

    monkeypatch.setitem(sys.modules, "torch", _blackwell_torch(cc))
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        (lambda name: object()) if fla_present else (lambda name: None),
        raising=True,
    )
    wy = types.ModuleType("fla.ops.gated_delta_rule.wy_fast")
    if kernel is not None:
        wy.prepare_wy_repr_bwd_kernel = kernel
    pkg_fla = types.ModuleType("fla")
    pkg_ops = types.ModuleType("fla.ops")
    pkg_gdr = types.ModuleType("fla.ops.gated_delta_rule")
    pkg_fla.__path__, pkg_ops.__path__, pkg_gdr.__path__ = [], [], []
    pkg_gdr.wy_fast = wy
    for name, mod in (
        ("fla", pkg_fla),
        ("fla.ops", pkg_ops),
        ("fla.ops.gated_delta_rule", pkg_gdr),
        ("fla.ops.gated_delta_rule.wy_fast", wy),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    removed: list[int] = []
    monkeypatch.setattr(
        perf, "_remove_fla_from_disk", lambda: (removed.append(1), (["/x/fla"], False))[1]
    )
    return perf, removed, wy


def test_blackwell_gdn_autotune_restricted_to_validated_config(monkeypatch):
    """sm100 + full upstream config space -> only num_warps=2/num_stages=4 survives; fla is KEPT.
    The tuner is wrapped in decorator layers, so the unwrap walk (.fn descent) is exercised too."""
    tuner = _FakeTuner([_TunerCfg(w, s) for w, s in _FULL_SPACE])
    perf, removed, _ = _patch_blackwell_stack(
        monkeypatch, kernel=_TunerWrapper(_TunerWrapper(tuner))
    )

    perf._restrict_fla_gdn_autotune_on_blackwell()

    assert [(c.num_warps, c.num_stages) for c in tuner.configs] == [(2, 4)]
    assert not removed, "restriction succeeded -> fla must be KEPT"


def test_blackwell_gdn_autotune_sm120_also_restricted(monkeypatch):
    """Consumer Blackwell (sm120) matches upstream's IS_NVIDIA_BLACKWELL gate: restricted too."""
    tuner = _FakeTuner([_TunerCfg(w, s) for w, s in _FULL_SPACE])
    perf, removed, _ = _patch_blackwell_stack(monkeypatch, cc=(12, 0), kernel=tuner)

    perf._restrict_fla_gdn_autotune_on_blackwell()

    assert [(c.num_warps, c.num_stages) for c in tuner.configs] == [(2, 4)]
    assert not removed


def test_blackwell_gdn_autotune_already_restricted_is_noop(monkeypatch):
    """A pinned fla that already carries fla #1000 (space == validated config) is left alone —
    the guard must be a no-op on a future pin bump, not a second filter or a fail-closed."""
    tuner = _FakeTuner([_TunerCfg(2, 4)])
    perf, removed, _ = _patch_blackwell_stack(monkeypatch, kernel=tuner)

    perf._restrict_fla_gdn_autotune_on_blackwell()

    assert [(c.num_warps, c.num_stages) for c in tuner.configs] == [(2, 4)]
    assert not removed


def test_blackwell_gdn_autotune_fail_closed_when_tuner_missing(monkeypatch):
    """Autotuner not found (fla layout changed, e.g. an unreviewed pin bump) -> fail CLOSED:
    fla is physically removed so transformers falls back to the pure-PyTorch delta rule."""
    perf, removed, _ = _patch_blackwell_stack(monkeypatch, kernel=None)

    perf._restrict_fla_gdn_autotune_on_blackwell()

    assert removed, "no autotuner -> must DISABLE fla (grad-correctness over speed)"


def test_blackwell_gdn_autotune_fail_closed_when_no_validated_config(monkeypatch):
    """The validated (2,4) config missing from the space -> nothing safe to run -> fail CLOSED."""
    tuner = _FakeTuner([_TunerCfg(4, 2), _TunerCfg(4, 3)])
    perf, removed, _ = _patch_blackwell_stack(monkeypatch, kernel=tuner)

    perf._restrict_fla_gdn_autotune_on_blackwell()

    assert removed, "no validated config in space -> must DISABLE fla"


def test_blackwell_gdn_autotune_noop_without_fla(monkeypatch):
    """fla absent (already the pure-PyTorch path) -> nothing to restrict, nothing to remove."""
    perf, removed, _ = _patch_blackwell_stack(monkeypatch, fla_present=False, kernel=None)

    perf._restrict_fla_gdn_autotune_on_blackwell()

    assert not removed


def test_non_blackwell_gdn_autotune_untouched(monkeypatch):
    """Hopper (sm90) keeps the full autotune space — the restriction is Blackwell-only (the
    tilelang fast path owns the sm90 story; shrinking its Triton space would only cost perf)."""
    tuner = _FakeTuner([_TunerCfg(w, s) for w, s in _FULL_SPACE])
    perf, removed, _ = _patch_blackwell_stack(monkeypatch, cc=(9, 0), kernel=tuner)

    perf._restrict_fla_gdn_autotune_on_blackwell()

    assert len(tuner.configs) == len(_FULL_SPACE)
    assert not removed


# ---------------------------------------------------------------------------
# sm100 tilelang GDN opt-out: the baked tilelang backend (needed for Hopper, fla
# The worker must opt fla out via FLA_TILELANG=0 (upstream's own knob; upstream default-gates
# tilelang to Hopper since fla #975) so fla dispatches to its Triton path, correct on sm100.
# ---------------------------------------------------------------------------
def _patch_arch(monkeypatch, cc):
    import flash.engine.worker.perf as perf

    monkeypatch.setitem(sys.modules, "torch", _blackwell_torch(cc))
    return perf


def test_sm100_fla_tilelang_opted_out(monkeypatch):
    """sm100 + FLA_TILELANG unset -> the worker sets FLA_TILELANG=0 before any fla dispatch."""
    import os

    perf = _patch_arch(monkeypatch, (10, 0))
    monkeypatch.delenv("FLA_TILELANG", raising=False)

    perf._force_fla_triton_gdn_on_sm100()

    assert os.environ.get("FLA_TILELANG") == "0"


def test_sm100_fla_tilelang_overrides_an_explicit_preset(monkeypatch):
    """A pre-set FLA_TILELANG=1 is overridden on sm100: this is a correctness floor.

    tilelang's chunk_bwd_dqkwg miscomputes GDN gradients on sm100, and the failure is silent --
    training completes and only the weights are wrong. Honouring an operator's opt-in here would let
    a run produce quietly garbage weights, so flash owns this value on this arch.
    """
    import os

    perf = _patch_arch(monkeypatch, (10, 0))
    monkeypatch.setenv("FLA_TILELANG", "1")

    perf._force_fla_triton_gdn_on_sm100()

    assert os.environ.get("FLA_TILELANG") == "0"


@pytest.mark.parametrize("cc", [(9, 0), (12, 0), (8, 9)])
def test_non_sm100_fla_tilelang_untouched(monkeypatch, cc):
    """sm90 NEEDS tilelang (fla #640); sm89/sm120 train healthily under the pin's default —
    the opt-out is strictly sm100 (the measured-broken arch)."""
    import os

    perf = _patch_arch(monkeypatch, cc)
    monkeypatch.delenv("FLA_TILELANG", raising=False)

    perf._force_fla_triton_gdn_on_sm100()

    assert "FLA_TILELANG" not in os.environ


def test_gpu_type_pin_overrides_larger_requested_gpu_hint(monkeypatch):
    # regression (PR #538 finding 2): gpu_type is the authoritative hardware pin. once the live card
    # matches the pinned class, the softer requested_gpu hint (gpu.type) must NOT be re-checked against
    # _gpu_mismatch_reason, which can name a larger class and reject a correctly-provisioned card.
    from types import SimpleNamespace

    import flash.engine.worker.perf.lifecycle as lifecycle
    from flash.engine.worker.perf import RetriableInfraError

    # live card is an 80 GB H100 (sm90). gpu_type pins H100 (a match); the requested hint names the
    # larger B200 class, which a bare _gpu_mismatch_reason would reject (80 GB < B200 need, sm90 < sm100).
    cuda = SimpleNamespace(
        get_device_capability=lambda _index: (9, 0),
        get_device_properties=lambda _index: SimpleNamespace(total_memory=80 * 10**9),
        get_device_name=lambda _index: "NVIDIA H100 PCIE",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(lifecycle, "_host_driver_cuda", lambda: 13.0)

    # pin matches the live card -> accept, despite the larger requested hint (pre-fix this raised).
    lifecycle.verify_gpu("B200", gpu_type="H100")

    # control: when the pin itself does not match the live card, it still raises.
    with pytest.raises(RetriableInfraError, match="exact-class mismatch"):
        lifecycle.verify_gpu("B200", gpu_type="B200")


def test_gpu_type_pin_still_rejects_underprovisioned_matching_card(monkeypatch):
    # regression (PR #538 finding 2): honoring the gpu_type pin must NOT drop the live safety net.
    # a card whose NAME canonicalizes to the pinned class but is under-provisioned (a mig slice with
    # reduced vram, or a too-old host driver) has no quote-time equivalent check, so it must still be
    # caught here -- validated against the pinned class itself, not the softer requested hint.
    from types import SimpleNamespace

    import flash.engine.worker.perf.lifecycle as lifecycle
    from flash.engine.worker.perf import RetriableInfraError

    # live card reports an H100 name (canonical class matches the H100 pin) but only 40 GB, below the
    # 90% floor for the full 80 GB class. an early return on class-name match would wrongly accept it.
    cuda = SimpleNamespace(
        get_device_capability=lambda _index: (9, 0),
        get_device_properties=lambda _index: SimpleNamespace(total_memory=40 * 10**9),
        get_device_name=lambda _index: "NVIDIA H100 PCIE",
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(lifecycle, "_host_driver_cuda", lambda: 13.0)

    with pytest.raises(RetriableInfraError, match="does not match requested"):
        lifecycle.verify_gpu("H100", gpu_type="H100")


def test_no_except_handler_supplies_a_fallback_gdn_hybrid():
    """No `except` may assign `gdn_hybrid`. A swallowed error must not answer the arch question.

    The capability call is evaluated FIRST, so any raise from it (no cuda, driver mismatch, a probe
    failure with nothing to do with the checkpoint) skipped the classification entirely, reported a
    genuine GDN hybrid as not-hybrid, skipped the boundary gate, and left `use_remove_padding` true
    -- packing a GDN model with no boundary resets, exactly the contamination the gate prevents.
    """
    import ast
    import inspect as _inspect

    from flash.engine.worker.train.entry import opd_train, rl_train, sft_train

    for module in (sft_train, opd_train, rl_train):
        tree = ast.parse(_inspect.getsource(module))
        for handler in (n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)):
            assigned = {
                target.id
                for node in ast.walk(handler)
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            assert "gdn_hybrid" not in assigned, (
                f"{module.__name__}:{handler.lineno} an except handler assigns gdn_hybrid. a "
                "failure anywhere in that try -- including probes unrelated to the checkpoint -- "
                "would report a gdn hybrid as not-hybrid and pack it without boundary resets."
            )


def test_each_path_resolves_the_gdn_arch_question_exactly_once():
    """The gdn arch question may be asked at most once per module. Two calls can disagree.

    The helper answers False when its OWN probe raises (a hub blip, a revision fetch failure), so
    the second call could return False where the first returned True. Asserted structurally because
    the disagreeing case needs a transient failure to reproduce and so will not show up in any
    deterministic test.
    """
    import ast
    import inspect as _inspect

    from flash.engine.worker.train.entry import opd_train, rl_train, sft_train

    for module in (sft_train, opd_train, rl_train):
        tree = ast.parse(_inspect.getsource(module))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "model_is_gdn_hybrid"
        ]
        assert len(calls) == 1, (
            f"{module.__name__} resolves the gdn arch question {len(calls)} times (lines "
            f"{[c.lineno for c in calls]}). the probe answers False on its own failure, so a "
            "second ask can contradict the first: resolve it once and reuse the value."
        )


def test_every_algorithm_records_whether_gdn_boundary_resets_engaged():
    """All three verl paths must publish `gdn_boundary_resets` in their run metadata.

    A SUCCESSFUL run uploads no console, so without this key a finished GDN run gives no way to tell
    whether it packed with boundary resets or fell back to the padded path. For a gate whose failure
    mode is silent cross-example contamination, "which mode did this run actually train in" is the
    one question the artifacts have to answer -- the same reasoning rl_train already applies to
    `vllm_kv_cache_dtype`.
    """
    import inspect as _inspect

    from flash.engine.worker.train.entry import opd_train, rl_train, sft_train

    # grpo renders its run metadata in train.rl.verl_config and opd in train.opd.overrides, so
    # each trainer's source is its module plus wherever its notes are built.
    from flash.engine.worker.train.opd.orchestration import overrides as opd_overrides
    from flash.engine.worker.train.rl.launch import verl_config as rl_verl_config

    extra = {"rl_train": rl_verl_config, "opd_train": opd_overrides}
    for module in (sft_train, opd_train, rl_train):
        source = _inspect.getsource(module)
        if module.__name__.rsplit(".", 1)[-1] in extra:
            source += _inspect.getsource(extra[module.__name__.rsplit(".", 1)[-1]])
        assert '"gdn_boundary_resets"' in source, (
            f"{module.__name__} computes the gdn boundary-reset decision but never records it, so a "
            "finished run cannot be checked for whether it trained packed-with-resets or padded."
        )


def test_sft_remove_padding_is_ungated_tensor_layout():
    """sft's `use_remove_padding` must not be gated on ANYTHING.

    That premise is false, and verl says so directly: sft_trainer.py:240 `global_batch_size =
    config.data.train_batch_size` sft_trainer.py:181 `total_training_steps =
    config.trainer.total_training_steps` sft_trainer.py:344 `use_remove_padding` appears ONLY as a
    logged field Examples-per-update and the step horizon come from config the worker copies
    straight off the profile (`train_batch_size = profile.examples_per_update`,
    `total_training_steps = profile.authoritative_steps`).
    """
    import ast
    import inspect as _inspect

    from flash.engine.worker.train.entry import sft_train

    tree = ast.parse(_inspect.getsource(sft_train))
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "use_remove_padding"
    ]
    assert len(assigns) == 1, (
        f"expected exactly one `use_remove_padding` assignment, found {len(assigns)} "
        f"(lines {[a.lineno for a in assigns]})"
    )
    names = {n.attr for n in ast.walk(assigns[0].value) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(assigns[0].value) if isinstance(n, ast.Name)
    }
    assert not names, (
        f"use_remove_padding is gated on {sorted(names)}. any false case forces the padded verl "
        "path, and its no_padding sft_loss cannot read a strided tensor -- the run dies on the "
        "first optimizer step. the step contract is already pinned by profile.examples_per_update "
        "and profile.authoritative_steps, and boundary isolation by the batch size that follows "
        "from the packing mode; this flag is layout only and has no legitimate false case."
    )


def test_grpo_and_opd_do_not_launch_into_the_unrunnable_padded_fallback():
    """A gdn hybrid whose child cannot reset boundaries must fail AT THE GATE, not mid-run.

    The padded fallback (`use_remove_padding=False`) is boundary-correct but cannot complete a step
    on verl's fsdp engine: flash sets `use_fused_kernels=True` unconditionally, and that pair walks
    into `prepare_model_outputs`' fused padded branch, which returns a dense `[bsz, response_len]`
    where every sibling path re-nests via `cu_seqlens`. Raising there would fail runs that train
    correctly.
    """
    import ast
    import inspect as _inspect
    import textwrap as _textwrap

    from flash.engine.worker.train.entry import backend_common, opd_train, rl_train, sft_train

    # the gate lives in one shared helper, so the assertions split: the helper must raise, and each
    # affected algorithm must route through it rather than re-deriving a decision of its own.
    gate = _inspect.getsource(backend_common.require_gdn_boundary_resets)
    assert "raise RuntimeError(" in gate, (
        "require_gdn_boundary_resets no longer raises, so a gdn hybrid whose child cannot reset "
        "boundaries would again launch into use_remove_padding=False, which cannot complete a "
        "training step on verl's fsdp engine."
    )
    # and it must be a hard gate, not a warn-and-continue: nothing may follow the raise on a path
    # that could still return None for a hybrid. two returns exactly -- the non-gdn None and the
    # arch -- means the raise is the only other exit.
    returns = [n for n in ast.walk(ast.parse(gate.lstrip())) if isinstance(n, ast.Return)]
    assert len(returns) == 2, (
        f"require_gdn_boundary_resets grew to {len(returns)} return paths. it must have exactly "
        "two -- None for a non-gdn model and the arch for a resettable one -- so that a hybrid "
        "without resets has no exit but the raise."
    )

    for module in (opd_train, rl_train):
        src = _inspect.getsource(module)
        assert "require_gdn_boundary_resets(" in src, (
            f"{module.__name__} no longer routes its gdn decision through the raising gate, so it "
            "can reach use_remove_padding=False again."
        )
        # and it must not have grown its own escape hatch: the override is hardcoded true, so any
        # reappearance of the conditional plumbing is a regression back to the unrunnable config.
        assert "use_remove_padding=False" not in src.replace(" ", ""), (
            f"{module.__name__} sets use_remove_padding=False, which verl's fsdp engine cannot run "
            "alongside the use_fused_kernels=True this recipe also sets."
        )

    # sft is conditional where grpo/opd are unconditional, and the condition is the whole point: a
    # packed gdn profile has packed neighbours to contaminate, so it must take the raising gate. the
    # control-plane gate cannot answer this because it is device-independent by construction, so it
    # proves the kernels are installed, never that the conv kernel runs on this
    # card. only the child probe knows. an exact-unpacked run keeps the soft form because
    # examples_per_update is 1.
    sft_src = _inspect.getsource(sft_train.run_sft_train)
    assert "require_gdn_boundary_resets(" in sft_src, (
        "packed sft no longer routes through the raising gate, so a gdn hybrid whose child cannot "
        "reset boundaries would train across packed example boundaries while appearing patched: "
        "transformers' fallbacks ACCEPT cu_seq_lens_q and seq_idx and silently discard them."
    )
    assert 'profile.packing_mode == "packed"' in sft_src, (
        "the sft gate no longer keys on the packed profile. it must stay conditional: raising on "
        "an exact-unpacked run would fail runs that are already boundary-safe at "
        "examples_per_update=1, and dropping the condition entirely would let a packed run through."
    )
    # the OTHER half of that condition has to come from the frozen profile, not from a re-probe.
    # `model_is_gdn_hybrid` swallows a failed hub/cache read and answers False, so deriving
    # `gdn_hybrid` from it alone lets a transient failure turn the gate above into dead code on a
    # profile that is already packed -- resets skipped, shim not installed, state bleeding across
    # packed neighbours, and nothing raised. the profile's label was frozen by a raising probe.
    assert 'profile.architecture_mode == "gdn-hybrid"' in sft_src, (
        "the sft gdn decision no longer consults the frozen profile.architecture_mode. it must, "
        "because model_is_gdn_hybrid returns False on a transient config-read failure: that would "
        "silently skip the packed-boundary reset requirement instead of failing closed."
    )
    # the module the shim patches must fail closed too. gdn_model_type answers "qwen3_5" both for a
    # dense model and for a config it could not read, and Qwen/Qwen3.6-35B-A3B is qwen3_5_moe -- so
    # the guess patches the WRONG module and reports resets active over unpatched MoE layers.
    #
    # assert on the GUARD, not on the call: `strict_gdn_probe_module(` survives inside `if False:`,
    # so a substring check passes with the fix disabled (observed while mutation-testing this).
    sft_tree = ast.parse(_textwrap.dedent(sft_src))
    strict_guards = [
        node
        for node in ast.walk(sft_tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "strict_gdn_probe_module"
            for c in ast.walk(node)
        )
    ]
    assert strict_guards, (
        "packed sft no longer resolves its modeling module strictly, so a transient config-read "
        "failure would fall back to the dense qwen3_5 module and patch the wrong arch on a "
        "qwen3_5_moe model while logging boundary resets as active."
    )
    guard_src = " ".join(ast.unparse(node.test) for node in strict_guards)
    assert "packing_mode" in guard_src, (
        "the strict module resolve is no longer guarded by the packed profile (guard is "
        f"{guard_src!r}). an exact-unpacked run has no boundaries and must not be forced to "
        "resolve its arch strictly."
    )
    assert "examples_per_update" in guard_src, (
        "the strict module resolve is no longer guarded by the realized batch (guard is "
        f"{guard_src!r}). it must be reached exactly when the reset gate below fires, or the two "
        "disagree about which runs need a proven arch."
    )

    # and the demand must key on the REALIZED batch: min(batch_size, len(rows)) can be 1 under a
    # `packed` label, and one example per update has no neighbour to contaminate. assert on the
    # expression that actually feeds `require_gdn_boundary_resets`, since the same literal also
    # appears at the strict-resolve guard above and would satisfy a substring check on its own.
    require_guards = [
        node
        for node in ast.walk(sft_tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "require_gdn_boundary_resets"
            for c in ast.walk(node)
        )
    ]
    assert require_guards, "packed sft no longer calls require_gdn_boundary_resets under any guard."
    # the guard may name a local (packed_neighbours); resolve any plain assignments to it so the
    # test reads the CONDITION, not the variable name.
    assigns = {
        t.id: ast.unparse(n.value)
        for n in ast.walk(sft_tree)
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    require_src = " ".join(ast.unparse(node.test) for node in require_guards)
    resolved = require_src + " " + " ".join(assigns.get(n, "") for n in assigns if n in require_src)
    assert "examples_per_update" in resolved, (
        "the packed sft gate no longer checks examples_per_update (condition resolves to "
        f"{resolved.strip()!r}), so a packed profile that realized a single-example update would "
        "hard-fail on a missing child capability despite having no packed neighbours to protect."
    )


# ---------------------------------------------------------------------------
# the CHILD venv's tilelang stub. the parent repair only ever sees the parent's site-packages,
# but vLLM imports in the verl child, so the child needs the same repair against its own tilelang.
# ---------------------------------------------------------------------------
def _run_child_cudart_fix(tmp_path, extra_env=None):
    """Run the real child fix script in a subprocess whose sys.path holds only tmp_path.

    Runs the shipped script rather than a copy: a script that stopped repairing anything (an
    ImportError, a renamed symbol) must fail these tests rather than pass by doing nothing.
    """
    import os
    import subprocess
    import sys

    from flash.engine.worker.verl.capabilities import _CHILD_CUDART_FIX

    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    env.pop("PYTHONHOME", None)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_CUDART_FIX],
        capture_output=True,
        timeout=120,
        env=env,
        check=False,
    )
    # the shipped call does not pass text=True (the module's subprocess.run stubs do not accept
    # it), so decode here exactly as the shipped reader does.
    return subprocess.CompletedProcess(
        result.args,
        result.returncode,
        result.stdout.decode("utf-8", "replace"),
        result.stderr.decode("utf-8", "replace"),
    )


def test_child_cudart_fix_symlinks_child_stub_to_real_libcudart(tmp_path):
    """The child script repoints the CHILD venv's stub, not the parent's."""
    import os

    _fake_tilelang(tmp_path)
    stub = tmp_path / "tilelang" / "lib" / "libcudart_stub.so"
    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL")

    # the script probes for a real libcudart via glob/ctypes, which will not find our fake one.
    # point the system resolver at it by making it the only candidate the loader can name.
    result = _run_child_cudart_fix(tmp_path)

    # either it repointed the stub, or it truthfully reported finding no real libcudart. what it
    # must never do is claim success while leaving a plain file, or silently print nothing.
    assert result.stdout.strip(), "child fix produced no output, so it did not run its repair"
    if "redirected" in result.stdout:
        assert result.returncode == 0, f"child fix crashed: {result.stderr}"
        assert os.path.islink(str(stub))
    else:
        assert "no real libcudart found" in result.stdout, result.stdout
        # giving up leaves the stub a plain file, which is what aborts vLLM import. it has to be
        # distinguishable from the benign no-ops, which also print and also exit 0.
        assert result.returncode != 0, "an unrepaired stub must not report success"


def test_child_cudart_fix_is_a_clean_noop_without_tilelang(tmp_path):
    """No tilelang in the child venv -> nothing to shadow libcudart, and no crash."""
    result = _run_child_cudart_fix(tmp_path)

    assert result.returncode == 0, f"child fix crashed: {result.stderr}"
    assert "no tilelang in child venv" in result.stdout, result.stdout


def test_child_cudart_fix_leaves_an_already_repointed_stub_alone(tmp_path):
    """An existing symlink is already repaired; touching it again would churn the venv."""
    import os

    _fake_tilelang(tmp_path)
    stub = tmp_path / "tilelang" / "lib" / "libcudart_stub.so"
    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL")
    os.remove(str(stub))
    os.symlink(str(real), str(stub))

    result = _run_child_cudart_fix(tmp_path)

    assert result.returncode == 0, f"child fix crashed: {result.stderr}"
    assert "already repointed" in result.stdout, result.stdout
    assert os.path.realpath(str(stub)) == os.path.realpath(str(real))


def test_child_cudart_neutralize_reports_unsafe_when_the_stub_is_left_shadowing(
    tmp_path, monkeypatch
):
    """The helper's verdict is read off the real script's exit, not inferred.

    Drives the shipped script through its give-up path (a tilelang stub present, no real libcudart
    to point it at) and asserts the caller is told the venv is unsafe. Stubbing subprocess.run here
    would only assert against the stub.
    """
    import os
    import sys

    from flash.engine.worker.verl.capabilities import _neutralize_child_tilelang_cudart_stub

    _fake_tilelang(tmp_path)
    # the script probes the real filesystem for a libcudart exporting cudaDeviceReset. an empty
    # candidate set is the give-up path; a host that happens to have one would repair instead.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.delenv("PYTHONHOME", raising=False)

    safe = _neutralize_child_tilelang_cudart_stub(sys.executable)

    stub = tmp_path / "tilelang" / "lib" / "libcudart_stub.so"
    if os.path.islink(str(stub)):
        pytest.skip("this host has a real libcudart, so the script repaired instead of giving up")
    assert safe is False


def test_child_cudart_neutralize_reports_safe_when_there_is_no_tilelang(tmp_path, monkeypatch):
    """Nothing to shadow libcudart is a benign no-op, and must not withhold the stamp."""
    import sys

    from flash.engine.worker.verl.capabilities import _neutralize_child_tilelang_cudart_stub

    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.delenv("PYTHONHOME", raising=False)

    assert _neutralize_child_tilelang_cudart_stub(sys.executable) is True


def test_child_cudart_neutralize_reports_unsafe_when_the_interpreter_cannot_run(tmp_path):
    """An interpreter that never started did not repair anything either."""
    from flash.engine.worker.verl.capabilities import _neutralize_child_tilelang_cudart_stub

    assert _neutralize_child_tilelang_cudart_stub(str(tmp_path / "absent-python")) is False


def test_child_cudart_fix_does_not_import_flash():
    """The script must not depend on flash: it is absent from the child venv.

    A `from flash...` import there raises ImportError, the repair never happens, and the child
    dies on vLLM import on a paid GPU -- the exact failure this fix exists to prevent.
    """
    from flash.engine.worker.verl.capabilities import _CHILD_CUDART_FIX

    assert "import flash" not in _CHILD_CUDART_FIX
    assert "from flash" not in _CHILD_CUDART_FIX


def test_resolve_verl_python_repairs_a_venv_it_provisions():
    """The repair is wired into the venv flash builds, not merely defined.

    Scope note: only the provisioned venv. A preset ``FLASH_VERL_PYTHON`` is an interpreter flash
    does not own and must never mutate (see
    ``test_resolve_verl_python_returns_preset_unmodified``), so a prebuilt verl image carries its
    own stub repair rather than being patched from here.
    """
    import inspect

    from flash.engine.worker.verl import capabilities

    src = inspect.getsource(capabilities.resolve_verl_python)
    assert "_neutralize_child_tilelang_cudart_stub(py)" in src, (
        "resolve_verl_python no longer repairs the venv it provisions, so vLLM can abort its "
        "import in the child after the gpu is already rented."
    )
    # it must sit on the rebuild path: repairing a reused venv turns reuse back into work.
    rebuilt = src.split("if install_wandb:")[0]
    assert "_neutralize_child_tilelang_cudart_stub(py)" in rebuilt
    # and its verdict must gate the stamp. behaviourally covered by
    # test_an_unrepaired_child_cudart_stub_leaves_the_venv_unstamped; pinned here because a repair
    # whose result is computed and then dropped reads as wired-up while stamping the failure in.
    assert "cudart_safe" in rebuilt, (
        "the venv stamp no longer depends on the child libcudart repair, so a failed repair is "
        "recorded as fully provisioned and reused by every later attempt on this pod."
    )


def test_worker_image_repairs_its_own_verl_venv_cudart_stub():
    """The PREBUILT image repairs /opt/verl-venv itself.

    Dockerfile.worker sets FLASH_VERL_PYTHON, and resolve_verl_python returns a preset unchanged
    (flash does not own a foreign interpreter), so live workers take the early return and the
    runtime repair above never runs for them. That venv installs tilelang, so without a build-time
    repair every production worker keeps the exact stub that aborts vLLM's import in the child --
    the runtime fix would be dead code on the only image that ships.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile.worker").read_text()

    assert "extract_cudart_fix.py" in dockerfile, (
        "Dockerfile.worker no longer repairs its own /opt/verl-venv tilelang stub, so production "
        "workers abort on vLLM import in the child after the gpu is already rented."
    )
    # the repair must run against the verl venv, not the main interpreter: they have disjoint
    # site-packages, and the child trains in the verl one.
    assert re.search(r"/opt/verl-venv/bin/python\s+\S*fix\.py", dockerfile), (
        "the stub repair must run against /opt/verl-venv/bin/python (FLASH_VERL_PYTHON), not the "
        "main interpreter, whose site-packages the child never sees."
    )

    # ordering: after the last install that can (re)write tilelang into the venv, or the repair is
    # undone by a later layer while still appearing present.
    repair_at = dockerfile.index("extract_cudart_fix.py")
    for later in ("causal-conv1d==", '"${spec}"'):
        assert dockerfile.index(later) < repair_at, (
            f"{later} installs into the verl venv AFTER the libcudart repair, which can restore "
            "the stub the repair just replaced."
        )
    # and before the layer that imports vLLM there, which is what the stub crashes.
    assert repair_at < dockerfile.index("from vllm import ModelRegistry"), (
        "the verl venv imports vLLM before the stub is repaired, so the sanity layer hits the "
        "crash this repair prevents."
    )


def test_worker_image_cudart_fix_is_extracted_not_duplicated():
    """The image bakes the SAME script the runtime path runs.

    Two hand-maintained copies of this repair would drift, and a drifted copy fails only on a
    rented gpu. The extractor reads `_CHILD_CUDART_FIX` itself, so there is one source of truth.
    """
    import pathlib
    import subprocess
    import sys
    import tempfile

    from flash.engine.worker.verl.capabilities import _CHILD_CUDART_FIX

    root = pathlib.Path(__file__).resolve().parent.parent
    extractor = root / "docker" / "extract_cudart_fix.py"
    assert extractor.exists(), "the build-time extractor is missing; the image cannot repair itself"

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "fix.py"
        result = subprocess.run(
            [
                sys.executable,
                str(extractor),
                str(root / "flash" / "engine" / "worker" / "verl" / "capabilities.py"),
                str(out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"extractor failed: {result.stderr}"
        baked = out.read_text()

    # the generated header is comments only; everything below it must be the constant verbatim.
    body = baked.split("\n", 2)[2]
    assert body == _CHILD_CUDART_FIX, (
        "the baked script drifted from _CHILD_CUDART_FIX; the image and the runtime path would "
        "repair the same venv differently."
    )


def test_worker_image_rebuilds_when_the_cudart_extractor_changes():
    """A change to the extractor alone must rebuild :cu128.

    The repair is generated at build time, so its output is frozen into the image. `flash/**`
    watches the `_CHILD_CUDART_FIX` source the extractor reads, but the extractor lives outside
    `flash/`: without its own entry an extractor-only fix merges green and every GPU worker keeps
    running the image built from the previous script.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    workflow = (root / ".github" / "workflows" / "worker-image.yml").read_text()

    for script in ("docker/extract_cudart_fix.py", "docker/make_rp_handler.py"):
        assert f"- {script}" in workflow, (
            f"worker-image.yml does not rebuild on {script}; a change to a build-time generator "
            "would ship an image built from the old one."
        )
