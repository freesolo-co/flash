"""Worker stack selection + worker config compat + LoRA exclusion unit tests (CPU-only)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from flash.providers.runpod.train import (
    WORKER_DEPS,
    resolve_worker_deps,
)


def test_resolve_worker_deps_default():
    # The single pinned stack is the validated default (bench/results/phase1 matrix); fully
    # managed, no per-run override.
    assert resolve_worker_deps() == WORKER_DEPS


def test_gdn_fastpath_deps_present_and_kept_on_hopper():
    """The GDN fast-path stack (fla-from-git + tilelang + pinned apache-tvm-ffi) is baked in, and
    fla is KEPT on Hopper (sm90) — the #640 fix is fla's tilelang backend, not dropping fla."""
    joined = " ".join(WORKER_DEPS)
    assert (
        "git+https://github.com/fla-org/flash-linear-attention" in joined
    )  # complete fla, not the broken PyPI stub
    assert any(
        d.startswith("tilelang==") for d in WORKER_DEPS
    )  # correct GDN backend on Triton>=3.4, PINNED for reproducibility
    assert any(
        d.startswith("apache-tvm-ffi==0.1.11") for d in WORKER_DEPS
    )  # pin (0.1.12 aborts tilelang import)
    # fla must NOT be dropped on Hopper anymore (it was, pre-fix).
    deps = resolve_worker_deps()
    assert any("flash-linear-attention" in d for d in deps), (
        "fla must be kept on Hopper for the tilelang fast path"
    )


def test_worker_stack_pins_qwen35_capable_versions():
    joined = " ".join(WORKER_DEPS)
    assert "vllm==0.19" in joined  # first transformers-5-compatible vllm line
    assert "transformers>=5" in joined  # qwen3_5 model types need transformers 5.x
    assert "bitsandbytes" in joined  # 8-bit paged AdamW optimizer state (LoRA+ coexists)


# ---------------------------------------------------------------------------
# is_vl_checkpoint: qwen3_5* are VL; text models are not
# ---------------------------------------------------------------------------
def _import_worker(monkeypatch):
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as worker

    return worker


def _fake_transformers(monkeypatch, model_type: str):
    fake_cfg = types.SimpleNamespace(model_type=model_type)
    fake_auto = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: fake_cfg,
    )
    fake_mod = types.ModuleType("transformers")
    fake_mod.AutoConfig = fake_auto
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)


def test_is_vl_checkpoint_qwen35(monkeypatch):
    # qwen3_5* stay VL checkpoints WITHOUT a LoRA module exclusion: this flag must not be coupled to
    # any deleted exclusion list.
    worker = _import_worker(monkeypatch)
    for model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_6"):
        _fake_transformers(monkeypatch, model_type)
        assert worker.is_vl_checkpoint("Qwen/Qwen3.5-4B") is True


def test_is_vl_checkpoint_text_model(monkeypatch):
    worker = _import_worker(monkeypatch)
    _fake_transformers(monkeypatch, "llama")
    assert worker.is_vl_checkpoint("meta-llama/Llama-3.2-1B") is False


@pytest.mark.parametrize(
    ("revision", "expected"),
    [("refs/pr/123", {"revision": "refs/pr/123"}), ("", {})],
)
def test_model_revision_keyword_is_present_only_when_nonempty(revision, expected):
    from flash.engine.worker.hf import model_revision_kwargs

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

    from flash.engine.worker import lora, packing, sft
    from flash.engine.worker.perf import liger

    assert lora.is_vl_checkpoint("org/model", revision=revision)
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
    from flash.engine.worker import sft

    _fake_arch_probe(monkeypatch, hidden=0, layers=0)
    # Qwen/Qwen3.6-35B-A3B is the sole catalog entry carrying (hidden, layers) == (2048, 40).
    assert sft._model_arch_dims("Qwen/Qwen3.6-35B-A3B", revision="refs/pr/123") == (2048, 40)


def test_arch_dims_revision_nonzero_mismatch_still_fails_closed(monkeypatch):
    # a NONZERO probe dim that genuinely disagrees with the catalog is a real revision mismatch and must
    # still fail closed, so a revision pin can never silently size VRAM with the wrong geometry.
    from flash.engine.worker import sft

    _fake_arch_probe(monkeypatch, hidden=9999, layers=99)
    with pytest.raises(RuntimeError, match="revision-specific model architecture"):
        sft._model_arch_dims("Qwen/Qwen3.6-35B-A3B", revision="refs/pr/123")


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

    import flash.engine.worker as worker
    from flash.engine.worker import hf

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: calls.append(("snapshot", kwargs["repo_id"], kwargs)),
    )
    monkeypatch.setattr(hf, "_shared_weight_cache_dir", lambda: None)
    monkeypatch.setattr(hf, "_hf_cache_bytes", lambda *args, **kwargs: 0)
    monkeypatch.setattr(hf, "gpu_diagnostics", dict)
    monkeypatch.setattr(worker, "heartbeat", lambda *args, **kwargs: None)

    assert hf.load_tokenizer("org/model", revision="refs/pr/123") is not None
    hf.prefetch_model("org/model", revision="refs/pr/123")
    assert hf.load_tokenizer("org/model", revision="") is not None
    hf.prefetch_model("org/model", revision="")

    revision_calls = [kwargs for _kind, _model, kwargs in calls[:2]]
    empty_calls = [kwargs for _kind, _model, kwargs in calls[2:]]
    assert all(kwargs["revision"] == "refs/pr/123" for kwargs in revision_calls)
    assert all("revision" not in kwargs for kwargs in empty_calls)


def _fake_torch(monkeypatch):
    """Inject a stub ``torch`` (the CPU/server venv has no torch) exposing optim.AdamW."""

    class _AdamW:  # marker class; identity is all the tests check
        pass

    fake = types.ModuleType("torch")
    fake.optim = types.SimpleNamespace(AdamW=_AdamW)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return _AdamW


def _fake_bitsandbytes(monkeypatch):
    """Inject a stub ``bitsandbytes`` so loraplus_optimizer_cls can resolve the 8-bit class
    without a CUDA build of bnb installed."""

    class _PagedAdamW8bit:  # marker class; identity is all the test checks
        pass

    fake = types.ModuleType("bitsandbytes")
    fake.optim = types.SimpleNamespace(PagedAdamW8bit=_PagedAdamW8bit)
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)
    return _PagedAdamW8bit


def test_gpu_diagnostics_parses_nvidia_smi(monkeypatch):
    from flash.engine.worker import perf

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


def test_loraplus_optimizer_mirrors_8bit_optim(monkeypatch):
    """An `8bit` optim string -> bnb PagedAdamW8bit (LoRA+ and 8-bit state coexist), always-on."""
    worker = _import_worker(monkeypatch)
    _fake_torch(monkeypatch)
    paged = _fake_bitsandbytes(monkeypatch)
    cls, extra = worker.loraplus_optimizer_cls("paged_adamw_8bit")
    assert cls is paged
    assert extra == {}


def test_loraplus_optimizer_fp_optim_uses_adamw(monkeypatch):
    """A non-8-bit optim string keeps full-precision torch AdamW (mirrors the configured optim)."""
    worker = _import_worker(monkeypatch)
    adamw = _fake_torch(monkeypatch)
    cls, _extra = worker.loraplus_optimizer_cls("adamw_torch")
    assert cls is adamw


def test_loraplus_optimizer_bnb_missing_falls_back(monkeypatch):
    """If bitsandbytes can't be imported, fall back to fp32 AdamW (never block training)."""
    import builtins

    worker = _import_worker(monkeypatch)
    adamw = _fake_torch(monkeypatch)
    real_import = builtins.__import__

    def _no_bnb(name, *a, **k):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            raise ImportError("no bitsandbytes")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_bnb)
    assert worker.loraplus_optimizer_cls("paged_adamw_8bit")[0] is adamw


def test_heartbeat_commit_is_throttled(monkeypatch):
    """heartbeat() must rate-limit HF commits (per-step commits blow HF's 128/hour repo cap),
    while always committing milestone stages."""
    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    calls = []
    monkeypatch.setattr(w, "hf_upload_file", lambda *a, **k: calls.append(a[1]))

    # Large interval -> only milestone + the first commit; per-step heartbeats throttled.
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 9999.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    w.heartbeat("rl_start")  # milestone -> commits
    w.heartbeat("rl_step", step=1)  # throttled
    w.heartbeat("rl_step", step=2)  # throttled
    assert calls.count("heartbeat.json") == 1

    # Zero interval -> every call commits.
    calls.clear()
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    w.heartbeat("rl_step", step=1)
    w.heartbeat("rl_step", step=2)
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
    import flash.engine.worker as w

    calls = []
    monkeypatch.setattr(w, "hf_upload_file", lambda *a, **k: calls.append(a[1]))
    # Large interval -> only the FIRST emit of the stage commits; the rest are upload-throttled.
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 9999.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    w.heartbeat(stage, elapsed_seconds=1)
    w.heartbeat(stage, elapsed_seconds=31)
    w.heartbeat(stage, elapsed_seconds=61)
    assert calls.count("heartbeat.json") == 1, f"{stage} must be upload-throttled, got {calls}"


def test_heartbeat_rollback_guards_on_claim_seq_not_coarse_ts(monkeypatch):
    """A failed/timed-out commit rolls its slot claim back, but the rollback is gated on a monotonic
    claim SEQ, not on wall-clock-ts equality. So if a NEWER heartbeat claims the slot (which on a
    coarse clock can share the same _HB_LAST_UPLOAD ts) before our older commit fails, our rollback
    must NOT wipe that fresher claim — doing so would let the throttle / quiet_gate read the channel
    as stale right after a real upload."""
    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    hbmod = sys.modules[w.heartbeat.__module__]
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)

    seen = []

    def fake_upload(path, name):
        seen.append(name)
        if len(seen) == 1:
            # A concurrent NEWER heartbeat claims the slot (higher claim seq) while our older commit
            # is in flight; our commit then fails, so we attempt to roll our claim back.
            hbmod._HB_CLAIM_SEQ += 1
            return False
        return True

    monkeypatch.setattr(w, "hf_upload_file", fake_upload)

    w.heartbeat("rl_step", step=1)
    # The newer claim owns the slot now, so our failed older commit must NOT restore _HB_LAST_UPLOAD
    # to its pre-claim 0.0. A ts-equality guard (now == _HB_LAST_UPLOAD) would have wrongly fired and
    # wiped the fresh claim; the claim-seq guard does not.
    assert w._HB_LAST_UPLOAD != 0.0


def test_heartbeat_hf_upload_runs_outside_lock(monkeypatch):
    """Perf regression guard: the synchronous hf_upload_file network call must run OUTSIDE
    _HB_LOCK. Holding the lock across the upload serializes the trainer's per-step reward
    callback behind the checkpoint daemon's HF commit during GRPO."""
    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    # When hf_upload_file is invoked, the lock must be acquirable (i.e. not held).
    lock_free_during_upload = []

    def fake_upload(*a, **k):
        acquired = w._HB_LOCK.acquire(blocking=False)
        lock_free_during_upload.append(acquired)
        if acquired:
            w._HB_LOCK.release()

    monkeypatch.setattr(w, "hf_upload_file", fake_upload)
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    w.heartbeat("rl_start")
    assert lock_free_during_upload == [True], "hf_upload_file must run with _HB_LOCK released"


def test_heartbeat_upload_skips_when_lock_is_stuck(monkeypatch):
    """A wedged upload holding _HB_UPLOAD_LOCK must not block the NEXT heartbeat. A milestone like
    model_prefetched (unthrottled, on the worker's critical path right before trainer construction)
    must skip its best-effort commit after a bounded wait rather than wedge the worker."""
    import importlib
    import time as _time

    # NB: resolve the submodule explicitly (the package re-exports the heartbeat() function).
    hbmod = importlib.import_module("flash.engine.worker.heartbeat")

    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    monkeypatch.setattr(hbmod, "_HB_UPLOAD_LOCK_TIMEOUT_S", 0.05)
    uploads = []
    monkeypatch.setattr(w, "hf_upload_file", lambda *a, **k: uploads.append(a))
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    sentinel_last_upload = 123.0  # a prior successful-commit timestamp the skip must NOT clobber
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", sentinel_last_upload)

    assert hbmod._HB_UPLOAD_LOCK.acquire(blocking=False), "lock should be free at test start"
    try:
        t0 = _time.monotonic()
        w.heartbeat("model_prefetched")  # must NOT block on the held lock
        elapsed = _time.monotonic() - t0
    finally:
        hbmod._HB_UPLOAD_LOCK.release()

    assert elapsed < 5.0, f"heartbeat wedged on the held upload lock ({elapsed:.2f}s)"
    assert uploads == [], "the best-effort commit must be skipped while the lock is stuck"
    # The skipped upload must ROLL BACK its optimistic slot claim — otherwise the throttle defers the
    # next real commit and the throttle treats a stale channel as fresh.
    assert sentinel_last_upload == w._HB_LAST_UPLOAD, (
        f"a skipped commit must not advance _HB_LAST_UPLOAD (got {w._HB_LAST_UPLOAD})"
    )


def test_heartbeat_rolls_back_slot_when_upload_reports_failure(monkeypatch):
    """The optimistic _HB_LAST_UPLOAD slot is claimed BEFORE the best-effort HF commit. If that commit
    fails, hf_upload_file swallows the error and returns False (it never raises on best-effort) — HF is
    still stale, so the slot must roll back exactly as the lock-timeout skip does. Otherwise the
    throttle defers the next retry on the strength of an upload that never happened.
    """
    import importlib

    hbmod = importlib.import_module("flash.engine.worker.heartbeat")

    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    sentinel_last_upload = (
        123.0  # a prior successful-commit timestamp the failed retry must restore
    )
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", sentinel_last_upload)

    calls = []
    # Mirror the real hf_upload_file contract: best-effort failure returns False (does not raise).
    monkeypatch.setattr(w, "hf_upload_file", lambda *a, **k: (calls.append(a[1]), False)[1])

    w.heartbeat("model_prefetched")

    assert calls == ["heartbeat.json"], "the upload must actually be attempted"
    assert sentinel_last_upload == w._HB_LAST_UPLOAD, (
        f"a failed upload must roll _HB_LAST_UPLOAD back to its prior value (got {w._HB_LAST_UPLOAD})"
    )


def test_heartbeat_keeps_slot_when_upload_reports_success(monkeypatch):
    """The dual of the rollback test: a SUCCESSFUL commit (or a mock that doesn't report False) must
    KEEP the advanced slot so the throttle works. ``is False`` — not falsy — gates the rollback, so a
    None-returning mock counts as success."""
    import importlib

    hbmod = importlib.import_module("flash.engine.worker.heartbeat")

    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    monkeypatch.setattr(w, "hf_upload_file", lambda *a, **k: True)

    w.heartbeat("model_prefetched")

    assert w._HB_LAST_UPLOAD > 0.0, "a successful commit must keep the advanced throttle slot"


def test_critical_stages_wait_longer_for_upload_lock(monkeypatch):
    """done/already_done/error_* are CRITICAL — no later heartbeat repairs a skipped terminal commit,
    and error_* carries the `retriable` flag worker_flagged_retriable() reads. So they wait the LONGER
    _HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S for the upload lock before skipping, vs the short progress
    timeout. Proven by timing: with the lock held, a terminal stage blocks for the critical timeout
    while a progress stage gives up after the short one."""
    import importlib
    import time as _time

    hbmod = importlib.import_module("flash.engine.worker.heartbeat")

    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    monkeypatch.setattr(hbmod, "_HB_UPLOAD_LOCK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(hbmod, "_HB_CRITICAL_UPLOAD_LOCK_TIMEOUT_S", 0.4)
    monkeypatch.setattr(w, "hf_upload_file", lambda *a, **k: None)
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)

    assert hbmod._HB_UPLOAD_LOCK.acquire(blocking=False), "lock should be free at test start"
    try:
        t = _time.monotonic()
        w.heartbeat("rl_step", step=1)  # progress -> short timeout, skips fast
        progress_wait = _time.monotonic() - t

        t = _time.monotonic()
        w.heartbeat("done")  # critical -> waits the long timeout before skipping
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
    import flash.engine.worker as w

    calls = []

    def _fake_upload(*a, **k):
        calls.append(a[1])
        return True  # simulate a successful commit so the throttle clock advances

    monkeypatch.setattr(w, "hf_upload_file", _fake_upload)
    monkeypatch.setattr(w, "_HB_TERMINAL_ONLY", True)
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 9999.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)

    # Short interval: terminal-only must STILL suppress non-terminal after the first, NOT
    # leak a commit once the window elapses (the bot-caught 128/hr re-breach).
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    w.heartbeat("sft_start")  # first non-terminal -> commits (last_upload==0)
    w.heartbeat("sft_model_load")  # suppressed despite 0s interval
    w.heartbeat("sft_trained")  # suppressed
    assert len(calls) == 1
    w.heartbeat("error_sft", error="boom")  # terminal -> always commits
    w.heartbeat("done")  # terminal -> always commits
    assert calls.count("heartbeat.json") == 3


def test_optimal_attn_impl_no_cuda_is_none(monkeypatch):
    """optimal_attn_impl picks the arch-best backend for the live GPU; with no CUDA (CI) it
    leaves transformers' default (None). There is no env override."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    assert w.optimal_attn_impl() is None


def test_attn_impl_for_capability_per_arch(monkeypatch):
    """Pure capability -> best-per-arch flash policy (no CUDA needed): flash on every arch EXCEPT
    Blackwell (datacenter sm100 + consumer sm120). Hopper(sm90): FA3, else a UNIFORM fall back to
    plain SDPA (NO FA3->FA2 chain). Ampere(sm80/86)+Ada(sm89): FA2. sm100/sm120: cuDNN SDPA."""
    w = _import_worker(monkeypatch)
    f = w._attn_impl_for_capability
    # Hopper sm90: FA3 is the arch's best flash; absent -> plain SDPA (uniform fallback, NOT FA2).
    assert f(9, 0, fa3_available=True, fa2_available=True) == "flash_attention_3"
    assert f(9, 0, fa3_available=False, fa2_available=True) is None  # uniform: -> SDPA, not FA2
    assert f(9, 0, fa3_available=False, fa2_available=False) is None
    # Ampere (8.0/8.6) + Ada (8.9): FA2 when the wheel is present, else SDPA. FA3 never applies.
    assert f(8, 0, fa2_available=True) == "flash_attention_2"  # A100
    assert f(8, 6, fa2_available=True) == "flash_attention_2"  # 3090/A10
    assert f(8, 9, fa2_available=True) == "flash_attention_2"  # Ada 4090
    assert f(8, 7, fa2_available=True) is None  # sm87 Jetson Orin: NOT a validated FA2 arch -> SDPA
    assert f(8, 0, fa2_available=False) is None
    # consumer Blackwell sm120: cuDNN SDPA regardless of flash availability.
    assert f(12, 0, fa3_available=True, fa2_available=True) == "sdpa"
    # datacenter Blackwell sm100 (B200): cuDNN SDPA too — NOT None (a bare None would let run_sft's
    # FA2 packing fallback force a possibly-missing sm100 FA2 kernel). Holds even when fa2 imports.
    assert f(10, 0, fa2_available=True) == "sdpa"
    assert f(10, 0, fa2_available=False) == "sdpa"


def test_flash_attn_probes_false_in_ci(monkeypatch):
    """The FA2/FA3 probes report False in offline CI (neither transformers/flash_attn nor the FA3
    ``flash_attn_interface`` is present). FA is used whenever importable — there is no disable
    hatch, so the result is purely 'is the package available'."""
    w = _import_worker(monkeypatch)
    assert w._flash_attn_3_available() is False  # flash_attn_interface / transformers absent in CI
    assert w._flash_attn_available() is False  # flash_attn wheel absent in CI


def test_liger_on_requires_default_and_gpu(monkeypatch):
    """liger_on(False) is always off; liger_on(True) still needs a CUDA GPU + importable
    liger_kernel (both absent in CI), so it's off here too."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    assert w.liger_on(False) is False
    assert w.liger_on(True) is False  # no CUDA / liger_kernel in CI


def test_liger_default_model_size_gate(monkeypatch):
    """Liger default is OFF for small models (1B-class, measured net loss PR #174) and ON only
    for models ≥ ~3B where fused-CE's memory win pays off."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

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
    worker.JOB_SPEC = types.SimpleNamespace(
        train=types.SimpleNamespace(lora_rank=32, lora_alpha=64),
        model_revision="a" * 40,
    )

    for model_id in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-0.8B"):
        captured.clear()
        worker.make_lora(model_id)
        assert captured.get("init_lora_weights") is True
        assert "pissa" not in str(captured.get("init_lora_weights")).lower()
        assert captured.get("use_rslora") is False
        assert captured.get("revision") == "a" * 40
        assert "target_parameters" not in captured

    captured.clear()
    worker.make_lora("Qwen/Qwen3.6-35B-A3B")
    assert captured["r"] == 32
    assert captured["target_modules"] == "all-linear"
    assert captured["target_parameters"] == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]


def test_35b_warmstart_requires_fused_expert_targets(monkeypatch):
    worker = _import_worker(monkeypatch)
    model_id = "Qwen/Qwen3.6-35B-A3B"

    with pytest.raises(ValueError, match="omits required expert targets"):
        worker.validate_lora_target_parameters({"target_modules": "all-linear"}, model_id)

    worker.validate_lora_target_parameters(
        {
            "target_parameters": [
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
            ]
        },
        model_id,
    )
    worker.validate_lora_target_parameters({}, "Qwen/Qwen3.5-9B")


def test_prepare_fresh_lora_base_uses_multimodal_loader_for_vl(monkeypatch):
    """Fresh LoRA on a VL checkpoint must wrap the full image-text tree, not TRL's default loader."""
    import flash.engine.worker.adapter as adapter_mod

    calls = []

    class _ImageText:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return {"loader": "vl", "args": args, "kwargs": kwargs}

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _ImageText
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(adapter_mod, "is_vl_checkpoint", lambda model_id, revision="": True)

    out = adapter_mod.prepare_fresh_lora_base(
        "/tmp/flash_sft_merged_x",
        "Qwen/Qwen3.5-4B",
        {"dtype": "bfloat16", "attn_implementation": "sdpa"},
        phase="sft",
    )

    assert out["loader"] == "vl"
    assert calls == [
        (
            ("/tmp/flash_sft_merged_x",),
            {"trust_remote_code": True, "dtype": "bfloat16", "attn_implementation": "sdpa"},
        )
    ]


def test_prepare_fresh_lora_base_keeps_non_vl_path(monkeypatch):
    import flash.engine.worker.adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "is_vl_checkpoint", lambda model_id, revision="": False)

    assert (
        adapter_mod.prepare_fresh_lora_base(
            "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B", {}, phase="sft"
        )
        == "meta-llama/Llama-3.2-1B"
    )


def test_prepare_fresh_lora_base_forwards_revision_to_probe_and_loader(monkeypatch):
    import flash.engine.worker.adapter as adapter_mod

    probes = []
    loads = []

    class _ImageText:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            loads.append((args, kwargs))
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = _ImageText
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        adapter_mod,
        "is_vl_checkpoint",
        lambda model_id, revision="": probes.append((model_id, revision)) or True,
    )

    adapter_mod.prepare_fresh_lora_base(
        "org/model",
        "org/model",
        {"dtype": "bfloat16", "revision": "refs/pr/123"},
        phase="sft",
        model_revision="refs/pr/123",
    )

    assert probes == [("org/model", "refs/pr/123")]
    assert loads[0][1]["revision"] == "refs/pr/123"


def test_prepare_fresh_lora_base_rejects_revision_authority_conflict(monkeypatch):
    import flash.engine.worker.adapter as adapter_mod

    with pytest.raises(ValueError, match="probe revision must match"):
        adapter_mod.prepare_fresh_lora_base(
            "org/model",
            "org/model",
            {"revision": "a" * 40},
            model_revision="b" * 40,
        )


def test_warmstart_base_loader_forwards_model_revision(monkeypatch, tmp_path):
    import flash.engine.worker.adapter as adapter_mod

    loads = []

    class _Base:
        _checkpoint_conversion_mapping = None

    class _Causal:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            loads.append((args, kwargs))
            return _Base()

    class _ImageText:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("unexpected vl loader")

    class _Peft:
        @classmethod
        def from_pretrained(cls, base, adapter_dir, is_trainable):
            return cls()

        def load_adapter(self, *args, **kwargs):
            return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = _Causal
    fake_transformers.AutoModelForImageTextToText = _ImageText
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = _Peft
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(
        adapter_mod,
        "_w",
        SimpleNamespace(
            JOB_SPEC=SimpleNamespace(
                model_revision="refs/pr/123",
                train=SimpleNamespace(init_from_adapter="owner/repo:sft/source"),
            )
        ),
    )
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(adapter_mod, "_download_adapter", lambda prefix: str(tmp_path))
    monkeypatch.setattr(adapter_mod, "adapter_is_vl_warmstart", lambda *args, **kwargs: False)
    monkeypatch.setattr(adapter_mod, "optimal_attn_impl", lambda: None)
    monkeypatch.setattr(adapter_mod, "_assert_warmstart_adapter_applied", lambda *args: None)

    model, peft_config = adapter_mod._init_adapter_model("org/model")

    assert isinstance(model, _Peft)
    assert peft_config is None
    assert loads == [
        (
            ("org/model",),
            {
                "dtype": "bfloat16",
                "trust_remote_code": True,
                "revision": "refs/pr/123",
            },
        )
    ]


def test_train_metadata_keeps_model_revision_in_nested_job_spec(monkeypatch):
    import flash.engine.worker as worker
    from flash.engine.worker import finalize
    from flash.spec import JobSpec

    captured = []
    monkeypatch.setattr(worker, "JOB_SPEC", JobSpec(model_revision="refs/pr/123"))
    monkeypatch.setattr(worker, "SEED", 42)
    monkeypatch.setattr(worker, "THINKING", False)
    monkeypatch.setattr(worker, "require_active_env", lambda: SimpleNamespace(id="org/env"))
    monkeypatch.setattr(worker, "hf_upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker, "_finalize", lambda metrics, **kwargs: captured.append((metrics, kwargs))
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
    import flash.engine.worker as worker
    from flash.engine.worker import finalize

    emitted = []
    finalized = []
    metrics_last = [{"step": 4, "reward": 0.75}]
    monkeypatch.setattr(worker, "JOB_SPEC", None)
    monkeypatch.setattr(worker, "SEED", 42)
    monkeypatch.setattr(worker, "THINKING", False)
    monkeypatch.setattr(worker, "require_active_env", lambda: SimpleNamespace(id="org/env"))
    monkeypatch.setattr(worker, "hf_upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker, "heartbeat", lambda stage, **kwargs: emitted.append((stage, kwargs))
    )
    monkeypatch.setattr(
        worker,
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

    import flash.engine.worker as worker
    from flash.engine.accounting import RunMetrics

    emitted = []
    metrics_last = [{"step": 4, "reward": 0.75}]
    monkeypatch.setattr(RunMetrics, "save", lambda self, path: None)
    monkeypatch.setattr(worker, "hf_upload_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker, "heartbeat", lambda stage, **kwargs: emitted.append((stage, kwargs))
    )
    monkeypatch.setattr(worker, "gpu_diagnostics", dict)
    monkeypatch.setattr("builtins.open", mock_open())

    worker._finalize(RunMetrics(phase="rl"), heartbeat_fields={"metrics_last": metrics_last})

    assert emitted[-1][0] == "done"
    assert emitted[-1][1]["metrics_last"] == metrics_last


# ---------------------------------------------------------------------------
# Hopper fla GDN fast-path fallback: when the healthy fla+tilelang stack can't be
# assembled (probe `ok` false), fla must be DISABLED (physically removed) so transformers'
# is_fla_available() gate flips off and the model uses the correct pure-PyTorch delta rule
# instead of fla's broken Triton>=3.4 GDN chunk_bwd (fla #640). A print alone is not enough.
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
    tvm_ffi_version: str | None = "0.1.11",
    tilelang_version: str | None = "0.1.11",
    record_pip: list[str] | None = None,
):
    """Wire the perf helper's external touchpoints for the Hopper fast-path tests and return the
    list that records _remove_fla_from_disk (fla-disable) calls.

    * ``pip_rc`` -> the return code every mocked ``pip install`` reports (non-zero = failed install).
    * ``find_spec_ok`` -> whether the post-install import probe finds fla/fla.modules/tilelang.
    * ``tvm_ffi_version`` -> what importlib.metadata.version('apache-tvm-ffi') reports (None=absent).
    * ``tilelang_version`` -> what importlib.metadata.version('tilelang') reports (None=absent). The
      helper gates the tilelang (re)install AND the final ``ok`` on this exact version, so a value
      != the pin models a present-but-wrong-version stack.
    * ``record_pip`` -> if given, every mocked ``pip install`` appends its joined spec args here (so
      a test can assert WHICH packages were reinstalled).
    """
    import importlib.metadata
    import importlib.util
    import subprocess

    from flash.engine.worker import perf

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
    (Copilot review on perf.py:~487). Without the rc check, find_spec alone would wrongly keep fla.
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
    perf, removed = _patch_hopper_stack(
        monkeypatch, pip_rc=0, find_spec_ok=True, tvm_ffi_version="0.1.11"
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert not removed, "healthy stack must KEEP fla (no disable on the success path)"


def test_hopper_tilelang_present_but_wrong_version_is_reinstalled(monkeypatch):
    """Regression (Copilot review on perf.py:~511): a DIFFERENT tilelang already resident (a job or
    the base image carries one) must NOT be treated as healthy. The helper gates on the installed
    version, so it (re)installs the exact pin; once the pin lands fla is KEPT."""
    pip_calls: list[str] = []
    # Resident wrong version first; after the (mocked) reinstall the metadata reports the pin.
    perf, removed = _patch_hopper_stack(
        monkeypatch,
        pip_rc=0,
        find_spec_ok=True,
        tvm_ffi_version="0.1.11",
        tilelang_version="0.1.11",  # post-reinstall resolved version
        record_pip=pip_calls,
    )
    # Make the FIRST _ver('tilelang') read (the install gate) see a stale wrong version, while the
    # final ok-gate read sees the pin — i.e. the reinstall corrected it.
    import importlib.metadata as _md

    gate_reads = iter(["0.1.9"])  # first read = stale; later reads -> pin (reinstall corrected it)
    orig_version = _md.version

    def _versioned(dist: str) -> str:
        if dist == "tilelang":
            return next(gate_reads, "0.1.11")
        return orig_version(dist)

    monkeypatch.setattr(_md, "version", _versioned, raising=True)

    perf._ensure_fla_fastpath_on_hopper()

    assert any(c.startswith("tilelang==0.1.11") for c in pip_calls), (
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
        tvm_ffi_version="0.1.11",
        tilelang_version="0.1.9",  # wrong version throughout (reinstall didn't correct it)
        record_pip=pip_calls,
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert any(c.startswith("tilelang==0.1.11") for c in pip_calls), (
        "wrong resident tilelang must still attempt the pinned reinstall"
    )
    assert removed, "tilelang version != pin after install must DISABLE fla (pin didn't land)"


def test_hopper_tvm_ffi_pip_skipped_when_pin_already_present(monkeypatch):
    """Regression (Copilot review on perf.py:~521): when the EXACT apache-tvm-ffi pin is already
    resident AND tilelang was NOT (re)installed this invocation, the helper must SKIP the tvm-ffi
    pip — re-running it unconditionally adds avoidable cold-start latency and could spuriously
    disable fla on a transient network/resolver hiccup. The ok gate still re-verifies the version,
    so fla stays KEPT."""
    pip_calls: list[str] = []
    perf, removed = _patch_hopper_stack(
        monkeypatch,
        pip_rc=0,
        find_spec_ok=True,
        tvm_ffi_version="0.1.11",  # exact pin already resident
        tilelang_version="0.1.11",  # exact pin -> tilelang NOT reinstalled this invocation
        record_pip=pip_calls,
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert not any("apache-tvm-ffi" in c for c in pip_calls), (
        "tvm-ffi pin already resident + tilelang not reinstalled must SKIP the tvm-ffi pip, "
        f"got pip calls: {pip_calls}"
    )
    assert not removed, "the pin being resident is the healthy path -> fla must be KEPT"


def test_hopper_outer_exception_disables_fla(monkeypatch):
    """Regression (Copilot review on perf.py:~580): an unexpected error mid-setup (AFTER the Hopper
    check passes) must FAIL-CLOSED — best-effort disable fla so transformers can't engage the broken
    Triton GDN path (#640) on a half-configured fla. The outer handler must call _remove_fla_from_disk
    and never re-raise."""
    import importlib

    perf, removed = _patch_hopper_stack(
        monkeypatch, pip_rc=0, find_spec_ok=True, tvm_ffi_version="0.1.11"
    )

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

    from flash.engine.worker import perf

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


def test_fla_git_pin_is_consistent_and_pinned():
    """The fla git dependency is PINNED to an exact commit (not the moving default branch) and the
    SAME SHA is used in both WORKER_DEPS and Dockerfile.worker (worker venv == baked image)."""
    import pathlib
    import re

    spec = next(d for d in WORKER_DEPS if "flash-linear-attention" in d)
    m = re.search(r"flash-linear-attention\.git@([0-9a-f]{40})\b", spec)
    assert m, f"fla dep must be pinned to a 40-char commit SHA, got: {spec!r}"
    deps_sha = m.group(1)

    # repo root: tests/ -> repo root is the parent.
    root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile.worker").read_text()
    dm = re.search(r"flash-linear-attention\.git@([0-9a-f]{40})\b", dockerfile)
    assert dm, "Dockerfile.worker fla install must be pinned to a 40-char commit SHA"
    assert dm.group(1) == deps_sha, (
        "Dockerfile.worker fla SHA must match WORKER_DEPS so the baked image and the worker "
        f"venv agree (deps={deps_sha}, dockerfile={dm.group(1)})"
    )

    # The worker's runtime fla reinstall (perf._ensure_fla_fastpath_on_hopper) must use the SAME pin —
    # an unpinned reinstall would pull the moving default branch and defeat reproducibility.
    perf_src = (root / "flash" / "engine" / "worker" / "perf" / "__init__.py").read_text()
    # The URL is built via implicit string concatenation across lines:
    #   "...flash-linear-attention.git"\n        "@<sha>" — so allow quotes/newline/space between.
    pm = re.search(r"flash-linear-attention\.git[\"'\s]*@([0-9a-f]{40})\b", perf_src)
    assert pm, "perf/__init__.py runtime fla reinstall must be pinned to a 40-char commit SHA"
    assert pm.group(1) == deps_sha, (
        f"perf/__init__.py fla SHA must match WORKER_DEPS (deps={deps_sha}, perf={pm.group(1)})"
    )


def test_tilelang_pin_is_consistent_and_pinned():
    """tilelang (the Hopper GDN correctness backend) is PINNED to an exact version (not unversioned)
    and the SAME pin is used in WORKER_DEPS, Dockerfile.worker, and perf.py's runtime reinstall, so
    cold-start installs / image rebuilds / runtime reinstalls all resolve the identical backend
    (Copilot review on flash/providers/_worker.py)."""
    import pathlib
    import re

    spec = next(d for d in WORKER_DEPS if d.split("==")[0].split(">")[0].strip() == "tilelang")
    m = re.match(r"tilelang==([0-9][0-9A-Za-z.\-]*)$", spec.strip())
    assert m, f"tilelang must be pinned to an exact version (tilelang==X.Y.Z), got: {spec!r}"
    pin = m.group(1)

    root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile.worker").read_text()
    dm = re.search(r'"tilelang==([0-9][0-9A-Za-z.\-]*)"', dockerfile)
    assert dm, "Dockerfile.worker must install a PINNED tilelang==X.Y.Z (not unversioned)"
    assert dm.group(1) == pin, (
        f"Dockerfile.worker tilelang pin must match WORKER_DEPS (deps={pin}, dockerfile={dm.group(1)})"
    )

    perf_src = (root / "flash" / "engine" / "worker" / "perf" / "__init__.py").read_text()
    # perf/__init__.py builds the spec via an f-string `f"tilelang=={TILELANG_PIN}"`, so assert the constant.
    pm = re.search(r'TILELANG_PIN\s*=\s*"([0-9][0-9A-Za-z.\-]*)"', perf_src)
    assert pm, (
        "perf/__init__.py must define a pinned TILELANG_PIN constant for the runtime reinstall"
    )
    assert pm.group(1) == pin, (
        f"perf/__init__.py TILELANG_PIN must match WORKER_DEPS (deps={pin}, perf={pm.group(1)})"
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
    from flash.engine import worker
    from flash.engine.worker.perf import RetriableInfraError

    monkeypatch.setattr(worker, "HF_REPO", "owner/repo")
    monkeypatch.setattr(worker.time, "sleep", lambda *_a: None)

    def boom():
        raise OSError("connection reset by peer")

    with pytest.raises(RetriableInfraError):
        worker._hf_upload(boom, "DONE", required=True, label="DONE")


def test_required_upload_starts_no_hf_call_at_deadline(monkeypatch):
    from flash.engine import worker
    from flash.engine.worker.perf import RetriableInfraError

    calls = []
    monkeypatch.setattr(worker, "HF_REPO", "owner/repo")
    monkeypatch.setattr(worker, "_remaining_worker_wall_seconds", lambda: 0.0)

    with pytest.raises(RetriableInfraError):
        worker._hf_upload(lambda: calls.append("upload"), "DONE", required=True, label="DONE")

    assert calls == []


def test_required_upload_caps_retry_sleep_and_starts_no_late_retry(monkeypatch, capsys):
    from flash.engine import worker
    from flash.engine.worker.perf import RetriableInfraError

    calls = []
    sleeps = []
    remaining = iter((2.0, 2.0, 0.0))
    monkeypatch.setattr(worker, "HF_REPO", "owner/repo")
    monkeypatch.setattr(worker, "_remaining_worker_wall_seconds", lambda: next(remaining))
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
    from flash.engine import worker

    calls = []
    monkeypatch.setattr(worker, "HF_REPO", "owner/repo")
    monkeypatch.setattr(worker, "_remaining_worker_wall_seconds", lambda: None)

    def boom():
        calls.append("upload")
        raise OSError("offline")

    assert not worker._hf_upload(boom, "debug.json", required=False, label="debug")
    assert calls == ["upload"]


# ---------------------------------------------------------------------------
# flash #184: tilelang's libcudart_stub.so shadows the real CUDA runtime in
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


def test_neutralize_tilelang_stub_symlinks_to_real_libcudart(tmp_path, monkeypatch):
    import importlib
    import os

    from flash.engine.worker import perf

    _pkg, stub = _fake_tilelang(tmp_path)
    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL-CUDART")  # stands in for the real runtime (has cudaDeviceReset)

    # Make find_spec("tilelang") resolve to our fake package, and skip the real-libcudart probe.
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    monkeypatch.setattr(perf, "_find_real_libcudart", lambda: str(real))

    perf._neutralize_tilelang_cudart_stub()

    # The stub path now points at the real runtime; the original stub is backed up verbatim.
    assert os.path.islink(stub), "stub should be replaced by a symlink to the real libcudart"
    assert os.path.realpath(stub) == os.path.realpath(str(real))
    assert os.path.exists(stub + ".orig")
    with open(stub + ".orig", "rb") as f:
        assert f.read() == b"STUB", "the original stub must be preserved for reversibility"

    # Idempotent: a second pass keeps the symlink and never clobbers the saved original.
    perf._neutralize_tilelang_cudart_stub()
    assert os.path.islink(stub)
    assert os.path.realpath(stub) == os.path.realpath(str(real))
    with open(stub + ".orig", "rb") as f:
        assert f.read() == b"STUB"


def test_neutralize_tilelang_stub_noop_without_real_libcudart(tmp_path, monkeypatch):
    """No discoverable real runtime -> leave the stub untouched (never break tilelang)."""
    import importlib
    import os

    from flash.engine.worker import perf

    _pkg, stub = _fake_tilelang(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    monkeypatch.setattr(perf, "_find_real_libcudart", lambda: None)

    perf._neutralize_tilelang_cudart_stub()

    assert not os.path.islink(stub)
    assert not os.path.exists(stub + ".orig")
    with open(stub, "rb") as f:
        assert f.read() == b"STUB"


def test_neutralize_tilelang_stub_noop_when_tilelang_absent(monkeypatch):
    """tilelang not installed -> clean no-op (must not even probe for a real libcudart)."""
    import importlib.util

    from flash.engine.worker import perf

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    def _boom():
        raise AssertionError("_find_real_libcudart must not run when tilelang is absent")

    monkeypatch.setattr(perf, "_find_real_libcudart", _boom)
    perf._neutralize_tilelang_cudart_stub()  # no exception


def test_find_real_libcudart_safe_when_nothing_matches(monkeypatch):
    """_find_real_libcudart returns None (never raises) when no candidate exposes the symbol."""
    import builtins
    import ctypes.util
    import glob

    from flash.engine.worker import perf

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


def _maps_has(name):
    with open("/proc/self/maps") as f:
        return any(name in line for line in f)


def test_neutralize_never_loads_the_stub_into_the_process(tmp_path, monkeypatch):
    """The crux of #184: the neutralize step must NEVER dlopen the stub. Loading it (even just to
    inspect it) maps `libcudart_stub.so` into /proc/self/maps, which is the exact line vLLM's
    CudaRTLibrary scan would then resolve -> the crash we're preventing. Compile a REAL stub .so
    missing cudaDeviceReset, run neutralize, and assert it was never mapped."""
    import os

    if not os.path.exists("/proc/self/maps"):
        import pytest

        pytest.skip(
            "/proc/self/maps unavailable (non-Linux); the loaded-mapping assertion needs it"
        )

    pkg = tmp_path / "tilelang"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    stub = str(pkg / "lib" / "libcudart_stub.so")
    real = str(tmp_path / "libcudart.so.12")
    if not _compile_so(str(tmp_path / "stub.c"), stub, "void cudaOther(void){}"):
        import pytest

        pytest.skip("no C toolchain to build a real stub .so")
    assert _compile_so(str(tmp_path / "real.c"), real, "void cudaDeviceReset(void){}")

    import importlib

    from flash.engine.worker import perf

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    monkeypatch.setattr(perf, "_find_real_libcudart", lambda: real)

    assert not _maps_has("libcudart_stub.so"), "precondition: stub not yet loaded"
    perf._neutralize_tilelang_cudart_stub()
    # THE regression assertion: the stub was redirected on disk WITHOUT ever being dlopen'd.
    assert not _maps_has("libcudart_stub.so"), "neutralize must not load the stub into the process"
    assert os.path.islink(stub)
    assert os.path.realpath(stub) == os.path.realpath(real)

    # And the redirected path now resolves a libcudart that DOES export cudaDeviceReset.
    import ctypes

    assert hasattr(ctypes.CDLL(stub), "cudaDeviceReset")


def test_neutralize_repoints_a_dangling_symlink(tmp_path, monkeypatch):
    """A DANGLING stub symlink (a prior pass's target moved/was removed) is NOT 'already done' — it
    leaves tilelang with a broken libcudart_stub.so. Neutralize must re-point it at a real lib."""
    import importlib
    import os

    pkg = tmp_path / "tilelang"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    stub = str(pkg / "lib" / "libcudart_stub.so")
    os.symlink(str(tmp_path / "gone-libcudart.so.12"), stub)  # dangling: target does not exist
    assert os.path.islink(stub)
    assert not os.path.exists(stub)

    real = tmp_path / "libcudart.so.12"
    real.write_bytes(b"REAL-CUDART")

    from flash.engine.worker import perf

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    monkeypatch.setattr(perf, "_find_real_libcudart", lambda: str(real))

    perf._neutralize_tilelang_cudart_stub()

    assert os.path.islink(stub)
    assert os.path.exists(stub)  # now RESOLVES
    assert os.path.realpath(stub) == os.path.realpath(str(real))


def test_find_real_libcudart_handles_bare_soname_without_crashing(monkeypatch):
    """find_library('cudart') returns a bare soname (e.g. 'libcudart.so.12'), not a path. The
    os.path.exists guard must not silently drop it; with no loadable cudart present it still resolves
    to None safely (and never raises)."""
    import builtins
    import ctypes.util
    import glob

    from flash.engine.worker import perf

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

    from flash.engine.worker import perf

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
# unrestricted prepare_wy_repr_bwd autotune space can select grad-miscomputing
# configs (live B200 Qwen3.6-35B-A3B SFT: grad_norm ~1e8 from the first logged
# step, loss flat or collapsing at every LR, while H200 trained healthily). The
# worker restricts the space in-process to the B200-validated config, and fails
# CLOSED (disables fla -> pure-PyTorch delta) when it cannot.
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

    from flash.engine.worker import perf

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
# #640) computes WRONG GRADIENTS on B200/sm100 (measured: dq/dk ~0.72, dg ~1.28
# rel-err at the production H==HV call shapes, deterministic, bf16 AND fp32) —
# the root cause of the B200 35B-A3B SFT incident. The worker must opt fla out
# via FLA_TILELANG=0 (upstream's own knob; upstream default-gates tilelang to
# Hopper since fla #975) so fla dispatches to its Triton path, correct on sm100.
# ---------------------------------------------------------------------------
def _patch_arch(monkeypatch, cc):
    from flash.engine.worker import perf

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

    tilelang's chunk_bwd_dqkwg miscomputes GDN gradients on sm100, and the failure is silent —
    training completes and only the weights are wrong. Honouring an operator's opt-in here would
    let a run produce quietly garbage weights, so flash owns this value on this arch.
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

    THE regression for a real defect: opd computed `gdn_hybrid` in the SAME try block as its fp8-KV
    `cuda.get_device_capability()` check, and that block's `except` set `gdn_hybrid = False`. The
    capability call is evaluated FIRST, so any raise from it (no cuda, driver mismatch, a probe
    failure with nothing to do with the checkpoint) skipped the classification entirely, reported a
    genuine GDN hybrid as not-hybrid, skipped the boundary gate, and left `use_remove_padding` true
    -- packing a GDN model with no boundary resets, exactly the contamination the gate prevents.

    The hazard is specifically an `except` that SUPPLIES a value, because that is what converts an
    unrelated failure into a confident wrong answer. A bare `try/finally` (grpo wraps its whole
    training block in one) is fine: an exception propagates and the run dies rather than reaching the
    packing decision with a fabricated `gdn_hybrid`. So this asserts on handler bodies, not on what
    else shares the `try`.

    `model_is_gdn_hybrid` already returns False on its own probe failure, so it needs no outer guard.
    Asserted structurally, across all three algorithms, because the failure is invisible at runtime:
    it fails toward "pack anyway", which logs nothing and moves no metric.
    """
    import ast
    import inspect as _inspect

    from flash.engine.worker import opd_train, rl_train, sft_train

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
