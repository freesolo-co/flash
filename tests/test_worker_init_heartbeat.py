"""The init-phase heartbeat must never block on CUDA telemetry.

Regression for the consumer-GPU warm-start "hang": ``run_rl``/``run_sft`` start a daemon thread
that heartbeats ``rl_initializing``/``sft_initializing`` every 30s while the MAIN thread is blocked
inside ``GRPOTrainer.__init__`` / ``SFTTrainer.__init__`` (a long, CUDA- and allocator-busy section
— vLLM colocate engine build + weight load + cold kernel JIT). The old code called
``gpu_diagnostics()`` (``include_torch=True``) on that side thread, which issues ``torch.cuda``
queries (``mem_get_info`` / ``memory_allocated`` / ``memory_reserved`` / ``get_device_name``). Those
serialize on the CUDA driver lock and PyTorch's caching-allocator mutex — both held by the init
thread — so the heartbeat thread could FREEZE for the whole init. The control plane then saw no
heartbeat and false-flagged a HANG on a run that was merely doing a slow (but live) cold init. The
fix: the init heartbeats use ``gpu_diagnostics(include_torch=False)`` (nvidia-smi only, out of
process, 8s timeout, GIL released during the wait).

These tests reproduce the freeze deterministically on CPU (no GPU needed) by injecting a fake
``torch`` whose CUDA query blocks, then prove the ``include_torch=False`` path stays responsive.
A source/AST wiring check pins the call sites so the regression can't silently return.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import threading
import time
import types

import pytest

from flash.engine.worker.perf import diagnostics


@pytest.fixture
def fast_nvidia(monkeypatch):
    """A fast, GPU-free nvidia-smi stand-in so the ``include_torch=False`` path returns instantly
    regardless of whether nvidia-smi exists on the test host."""
    monkeypatch.setattr(
        diagnostics, "_query_nvidia_gpu", lambda: {"gpu_util_pct": 0, "device_name": "FAKE-GPU"}
    )
    monkeypatch.setattr(diagnostics, "_query_nvidia_processes", lambda: [])


def _install_blocking_torch(monkeypatch, gate: threading.Event) -> None:
    """Inject a fake ``torch`` whose CUDA memory queries block until ``gate`` is set — standing in
    for the CUDA driver / allocator lock held by the main thread inside ``*Trainer.__init__``."""

    def _blocking_mem_get_info():
        gate.wait(timeout=30.0)  # held "by the init thread" until the test releases it
        return (1 << 30, 1 << 34)

    torch = types.ModuleType("torch")
    torch.__version__ = "2.10.0-fake"
    torch.version = types.SimpleNamespace(cuda="12.8")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda i=0: "FAKE-GPU",
        mem_get_info=_blocking_mem_get_info,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def _run_async(fn):
    state: dict = {}

    def _target():
        state["value"] = fn()
        state["done"] = True

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return th, state


def test_include_torch_true_freezes_when_cuda_is_locked(monkeypatch, fast_nvidia):
    """Repro: with the torch path on, a side-thread diag wedges while CUDA is "locked"."""
    gate = threading.Event()
    _install_blocking_torch(monkeypatch, gate)

    th, state = _run_async(lambda: diagnostics.gpu_diagnostics(include_torch=True))
    th.join(timeout=1.5)
    assert not state.get("done"), (
        "expected gpu_diagnostics(include_torch=True) to block on the CUDA query "
        "(this is the heartbeat freeze the fix removes)"
    )

    gate.set()  # main thread released the CUDA lock
    th.join(timeout=5.0)
    assert state.get("done"), "diag should complete once the CUDA lock is released"


def test_include_torch_false_stays_responsive_while_cuda_locked(monkeypatch, fast_nvidia):
    """Fix: with the torch path off, the diag returns promptly even while CUDA is wedged."""
    gate = threading.Event()  # NEVER set -> CUDA stays "locked" for the whole test
    _install_blocking_torch(monkeypatch, gate)

    th, state = _run_async(lambda: diagnostics.gpu_diagnostics(include_torch=False))
    th.join(timeout=2.0)

    assert state.get("done"), (
        "gpu_diagnostics(include_torch=False) must NOT touch torch.cuda and must return even while "
        "the init thread holds the CUDA lock"
    )
    diag = state["value"]
    # The nvidia-smi-only payload — and crucially NO torch keys that would have required a CUDA call.
    assert diag.get("gpu_util_pct") == 0
    assert "torch" not in diag
    assert "torch_memory_free_gb" not in diag


# --------------------------------------------------------------------------------------------
# liveness_heartbeat: the SINGLE daemon that keeps a stage alive while a long blocking call runs on
# the main thread (cold *Trainer.__init__, model prefetch, first GRPO step). Behaviour is tested once
# here on the helper; the call sites are pinned by thin wiring checks at the end.


def _liveness_env(monkeypatch, *, tick=0.01):
    """Patch heartbeat's module globals for a fast, side-effect-free liveness run.

    Returns (hb_module, worker_pkg, diag_include_torch_calls)."""
    hb = importlib.import_module("flash.engine.worker.heartbeat")
    import flash.engine.worker as w

    monkeypatch.setattr(hb, "_LIVENESS_TICK_S", tick)
    diag: list = []
    monkeypatch.setattr(
        hb, "gpu_diagnostics", lambda include_torch=True: (diag.append(include_torch), {})[1]
    )
    monkeypatch.setattr(hb, "_dump_thread_stacks", lambda reason: None)  # don't dump real stacks
    return hb, w, diag


def test_liveness_heartbeat_emits_liveness_pings_nvidia_smi_only(monkeypatch):
    hb, w, diag = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append(k.get("liveness")))
    with hb.liveness_heartbeat("init_stage"):
        time.sleep(0.2)
    assert emitted, "must emit while alive"
    assert all(v is True for v in emitted), "bare liveness_heartbeat emits LIVENESS pings (liveness=True)"
    assert diag, "diagnostics collected"
    assert all(it is False for it in diag), "must use gpu_diagnostics(include_torch=False)"


def test_liveness_heartbeat_reports_progress_advance_as_real_heartbeat(monkeypatch):
    hb, w, _ = _liveness_env(monkeypatch)
    seen: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: seen.append(bool(k.get("liveness"))))
    monkeypatch.setattr(w, "_HB_LAST_PROGRESS_TS", time.time())
    vals = iter([1, 2, 2, 2, 2, 2, 2, 2])  # advances, then stalls
    with hb.liveness_heartbeat("model_prefetching", progress=lambda: next(vals, 2)):
        time.sleep(0.2)
    assert False in seen, "a progress advance must emit a REAL (non-liveness) heartbeat"
    assert True in seen, "no advance must emit a liveness ping"


def test_liveness_heartbeat_dumps_stacks_once_when_progress_stale(monkeypatch):
    """No REAL progress for _STALL_DUMP_S -> dump every thread's stack ONCE (operator trace); the
    provider does the kill+retry off the same stale-progress signal."""
    hb, w, _ = _liveness_env(monkeypatch)
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: None)
    monkeypatch.setattr(hb, "_STALL_DUMP_S", 0.05)
    monkeypatch.setattr(w, "_HB_LAST_PROGRESS_TS", time.time() - 100)  # already stale
    dumped: list = []
    monkeypatch.setattr(hb, "_dump_thread_stacks", lambda reason: dumped.append(reason))
    with hb.liveness_heartbeat("rl_step"):
        time.sleep(0.2)
    assert len(dumped) == 1, f"must dump exactly once on a stall, got {len(dumped)}"


def test_liveness_heartbeat_join_is_bounded_even_if_emit_wedges(monkeypatch):
    """The exit join must be BOUNDED: a wedged heartbeat() upload can't hang the worker at block exit."""
    hb, w, _ = _liveness_env(monkeypatch)
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: time.sleep(30))
    monkeypatch.setattr(hb, "_HB_UPLOAD_LOCK_TIMEOUT_S", 0.2)
    t0 = time.time()
    with hb.liveness_heartbeat("init_stage"):
        time.sleep(0.1)
    assert time.time() - t0 < 5, "exit must be bounded by the join timeout, not wait on a wedged emit"


def test_liveness_heartbeat_rechecks_done_after_diagnostics():
    """gpu_diagnostics shells out to nvidia-smi (seconds); the wrapped call can finish during it. The
    daemon must re-check done BETWEEN diagnostics and the emit, so no stale stage lands afterward."""
    hb = importlib.import_module("flash.engine.worker.heartbeat")
    src = inspect.getsource(inspect.unwrap(hb.liveness_heartbeat))
    between = src[src.index("gpu_diagnostics(include_torch=False)") : src.index("_w.heartbeat(stage")]
    assert "done.is_set()" in between, "must re-check done.is_set() between diagnostics and emit"


def test_heartbeat_marks_progress_only_for_real_heartbeats(monkeypatch):
    """heartbeat() bumps _HB_LAST_PROGRESS_TS for a real heartbeat but NOT a liveness ping, and stamps
    liveness=True on the liveness payload so the provider can skip it."""
    import json

    import flash.engine.worker as ne

    seen: list = []
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 0.0)

    def _capture(local, *a, **k):
        with open(local) as f:
            seen.append(json.load(f))

    monkeypatch.setattr(ne, "hf_upload_file", _capture)

    ne._HB_LAST_PROGRESS_TS = 0.0
    ne.heartbeat("rl_step", step=1)  # real progress
    after_real = ne._HB_LAST_PROGRESS_TS
    assert after_real > 0, "a real heartbeat must mark progress"
    assert seen[-1].get("liveness") is None, "a real heartbeat carries no liveness flag"

    ne.heartbeat("rl_step", liveness=True, step=1)  # liveness ping
    assert after_real == ne._HB_LAST_PROGRESS_TS, "a liveness ping must NOT advance progress"
    assert seen[-1].get("liveness") is True, "a liveness ping is stamped liveness=True"


def test_per_step_training_stages_are_throttled():
    """Both per-step training stages must be in _HB_THROTTLED_STAGES so their HF upload is capped at
    _HB_MIN_INTERVAL_S. The reward/SFT log callbacks AND the train-loop liveness daemon re-emit the
    SAME stage frequently; without throttling sft_step (liveness ~every 30s ~= 120 commits/hr plus the
    log callback) would blow the 128/hr repo commit cap — exactly the regression rl_step's throttle
    already prevents."""
    import flash.engine.worker as ne

    assert "rl_step" in ne._HB_THROTTLED_STAGES
    assert "sft_step" in ne._HB_THROTTLED_STAGES, (
        "sft_step must be throttled like rl_step or the SFT liveness daemon blows the commit cap"
    )


def test_sft_step_liveness_upload_is_throttled(monkeypatch):
    """A burst of sft_step liveness pings within _HB_MIN_INTERVAL_S commits only ONCE — the throttle
    (not just rl_step) covers sft_step, so a slow SFT run's 30s liveness ticks stay under the cap."""
    import flash.engine.worker as ne

    uploads: list = []
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 60.0)
    monkeypatch.setattr(ne, "hf_upload_file", lambda local, *a, **k: uploads.append(local))
    ne._HB_LAST_UPLOAD = 0.0
    # First sft_step claims the slot; the next two (well within 60s) must be throttled out.
    for _ in range(3):
        ne.heartbeat("sft_step", liveness=True, step=0)
    assert len(uploads) == 1, "sft_step uploads must be throttled to one per _HB_MIN_INTERVAL_S"


# --------------------------------------------------------------------------------------------
# _hf_cache_bytes feeds the prefetch progress signal: bytes downloaded, or None when the cache dir
# doesn't exist yet (unmeasurable) — a growth is reported by liveness_heartbeat as REAL progress.
def test_hf_cache_bytes_counts_blobs_and_reports_unmeasurable_as_none(tmp_path, monkeypatch):
    import huggingface_hub.constants as hconst

    from flash.engine.worker import hf

    monkeypatch.setattr(hconst, "HF_HUB_CACHE", str(tmp_path))
    assert hf._hf_cache_bytes("org/model") is None  # no repo dir yet -> unmeasurable
    repo = tmp_path / "models--org--model"
    repo.mkdir(parents=True)
    assert hf._hf_cache_bytes("org/model") == 0  # repo dir, no blobs -> 0 (measurable)
    blobs = repo / "blobs"
    blobs.mkdir()
    (blobs / "complete").write_bytes(b"x" * 100)
    (blobs / "partial.incomplete").write_bytes(b"y" * 50)
    assert hf._hf_cache_bytes("org/model") == 150


# --------------------------------------------------------------------------------------------
# Provider: liveness pings must NOT count as progress (else a wedged worker pinging "alive" masks the
# stall). surface_heartbeat — shared by every provider — returns no-advance for a liveness heartbeat.
def test_is_training_heartbeat_gates_setup_vs_training():
    from flash.providers._poll import is_training_heartbeat

    # Setup stages (and a missing stage) never tighten — still the cold start.
    assert is_training_heartbeat("rl_train_start", None) is False
    assert is_training_heartbeat("sft_initializing", 5) is False
    assert is_training_heartbeat(None, 9) is False
    # The per-step training stages tighten ONLY at a COMPLETED step (>= 1); a step=0 gap-fill during
    # the silent cold first step keeps setup grace.
    assert is_training_heartbeat("rl_step", 0) is False
    assert is_training_heartbeat("sft_step", 0) is False
    assert is_training_heartbeat("rl_step", 1) is True
    assert is_training_heartbeat("sft_step", 3) is True
    # A malformed/missing step on a per-step stage is treated as 0 (must not raise) -> stays setup.
    assert is_training_heartbeat("rl_step", None) is False
    assert is_training_heartbeat("sft_step", "not-a-number") is False
    # POST-training stages carry NO step but mean training is DONE -> tighten so a hung teardown/DONE
    # upload falls under the tight window, not the wide setup grace (the MqsTh fix).
    assert is_training_heartbeat("rl_trained", None) is True
    assert is_training_heartbeat("sft_trained", None) is True
    assert is_training_heartbeat("rl_train_done", None) is True
    assert is_training_heartbeat("sft_train_done", None) is True


def test_provider_surface_heartbeat_skips_liveness_pings():
    from flash.providers._poll import surface_heartbeat

    real = {"stage": "rl_initializing", "step": 0, "ts": 100.0, "attempt": "1"}
    key, stage = surface_heartbeat(lambda: real, None, lambda _m: None)
    assert key is not None
    assert stage == "rl_initializing"
    live = {"stage": "rl_initializing", "step": 0, "ts": 200.0, "attempt": "1", "liveness": True}
    key2, stage2 = surface_heartbeat(lambda: live, key, lambda _m: None)
    assert key2 == key, "a liveness ping must not advance the stall key"
    assert stage2 is None, "a liveness ping is not surfaced as progress"


# --------------------------------------------------------------------------------------------
# Wiring: each long blocking phase runs under the shared liveness_heartbeat helper (behaviour covered
# above; these pin the call sites so coverage can't silently regress).
def test_rl_init_wraps_trainer_build_in_liveness_heartbeat():
    from flash.engine.worker import rl

    assert 'liveness_heartbeat("rl_initializing")' in inspect.getsource(rl.run_rl)


def test_sft_init_wraps_trainer_build_in_liveness_heartbeat():
    from flash.engine.worker import sft

    assert 'liveness_heartbeat("sft_initializing")' in inspect.getsource(sft.run_sft)


@pytest.mark.parametrize(
    ("modname", "outer", "stage"),
    [
        ("flash.engine.worker.rl", "run_rl", "rl_step"),
        ("flash.engine.worker.sft", "run_sft", "sft_step"),
    ],
)
def test_train_phase_wraps_train_in_liveness_heartbeat(modname, outer, stage):
    mod = importlib.import_module(modname)
    src = inspect.getsource(getattr(mod, outer))
    assert f'liveness_heartbeat("{stage}")' in src, (
        f"{outer} must wrap trainer.train() in liveness_heartbeat({stage!r}) — without it the cold "
        "first step emits no real heartbeat and looks like a hang"
    )


def test_prefetch_wraps_download_in_liveness_heartbeat_gated_on_bytes():
    from flash.engine.worker import hf

    src = inspect.getsource(hf.prefetch_model)
    assert "liveness_heartbeat(" in src
    assert '"model_prefetching"' in src
    assert "_hf_cache_bytes(" in src, "prefetch must report downloaded-byte growth as real progress"


def test_no_worker_side_stall_watchdog():
    """The worker has no separate stall watchdog: the provider owns kill+retry, and the dump fires on
    liveness give-up. Guard against re-adding the env-tunable faulthandler timer."""
    import importlib

    hb = importlib.import_module("flash.engine.worker.heartbeat")
    assert not hasattr(hb, "_rearm_stall_faulthandler")
    assert not hasattr(hb, "_STALL_WATCHDOG_S")
