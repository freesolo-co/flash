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
    return hb, w, diag


def test_liveness_heartbeat_emits_while_alive_with_nvidia_smi_only(monkeypatch):
    hb, w, diag = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append(s))
    with hb.liveness_heartbeat("init_stage"):
        time.sleep(0.2)
    assert emitted, "must emit the stage while alive"
    assert all(s == "init_stage" for s in emitted)
    assert diag, "diagnostics must be collected"
    assert all(it is False for it in diag), "must use gpu_diagnostics(include_torch=False)"


def test_liveness_heartbeat_join_is_bounded_even_if_emit_wedges(monkeypatch):
    """The context manager's exit join must be BOUNDED: a wedged heartbeat() upload can never hang
    the worker at the end of the wrapped block."""
    hb, w, _ = _liveness_env(monkeypatch)
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: time.sleep(30))  # a wedged upload
    t0 = time.time()
    with hb.liveness_heartbeat("init_stage", join_timeout=0.2):
        time.sleep(0.1)  # let the daemon enter the wedged emit
    assert time.time() - t0 < 5, "exit must be bounded by join_timeout, not wait on a wedged emit"


def test_liveness_heartbeat_quiet_gate_skips_when_channel_fresh(monkeypatch):
    hb, w, _ = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append(s))
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", time.time())  # fresh -> nothing to gap-fill
    with hb.liveness_heartbeat("rl_step", quiet_gate_s=1000.0):
        time.sleep(0.2)
    assert not emitted, "quiet_gate must suppress liveness while another heartbeat keeps it fresh"


def test_liveness_heartbeat_quiet_gate_fills_when_channel_stale(monkeypatch):
    hb, w, _ = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append(s))
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)  # ancient -> channel quiet -> gap-fill
    with hb.liveness_heartbeat("rl_step", quiet_gate_s=1.0):
        time.sleep(0.2)
    assert emitted, "quiet_gate must gap-fill once the channel has gone quiet"


def test_liveness_heartbeat_stops_when_progress_stalls(monkeypatch):
    """progress + max_silence_s: a liveness ping re-arms the watchdog, so it must STOP once the
    progress counter stalls — handing a genuinely wedged call back to the stall path (anti-mask)."""
    hb, w, _ = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append(time.time()))
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    t0 = time.time()
    with hb.liveness_heartbeat("rl_step", progress=lambda: 5, max_silence_s=0.15):
        time.sleep(0.8)
    assert emitted, "should cover the stall briefly before giving up"
    assert max(emitted) - t0 < 0.6, "must STOP emitting after max_silence (not run to block end)"


def test_liveness_heartbeat_stops_after_max_duration(monkeypatch):
    """max_duration_s (no progress counter, e.g. cold *Trainer.__init__): the ping re-arms the watchdog
    AND the provider setup-grace, so a stuck-but-GIL-releasing init would mask the hang forever. The
    daemon must STOP pinging once total lifetime exceeds the cap, handing off to the stall path."""
    hb, w, _ = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append(time.time()))
    t0 = time.time()
    with hb.liveness_heartbeat("rl_initializing", max_duration_s=0.15):
        time.sleep(0.8)
    assert emitted, "should cover the init briefly before giving up"
    assert max(emitted) - t0 < 0.6, "must STOP emitting after max_duration (not run to block end)"


def test_liveness_heartbeat_keeps_covering_while_progress_advances(monkeypatch):
    hb, w, _ = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append(time.time()))
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    counter = {"n": 0}

    def advancing():
        counter["n"] += 1
        return counter["n"]

    t0 = time.time()
    with hb.liveness_heartbeat("rl_step", progress=advancing, max_silence_s=0.1):
        time.sleep(0.5)
    assert emitted, "advancing progress must keep liveness alive"
    assert max(emitted) - t0 > 0.3, "advancing progress must keep liveness alive"


def test_liveness_heartbeat_rechecks_done_after_diagnostics(monkeypatch):
    """gpu_diagnostics shells out to nvidia-smi (seconds); the wrapped call can finish during it. The
    daemon must re-check done BETWEEN diagnostics and the emit, so no stale stage lands after the
    phase's terminal stage (e.g. a model_prefetching after model_prefetched)."""
    hb = importlib.import_module("flash.engine.worker.heartbeat")
    src = inspect.getsource(inspect.unwrap(hb.liveness_heartbeat))
    between = src[src.index("gpu_diagnostics(include_torch=False)") : src.index("_w.heartbeat(stage")]
    assert "done.is_set()" in between, "must re-check done.is_set() between diagnostics and emit"


def test_liveness_silence_uses_monotonic_clock_not_wall_clock():
    """The no-progress silence (max_silence_s) measures an ELAPSED interval, so it must use the
    monotonic clock — a wall-clock jump (NTP step, VM suspend/resume) must not trip it early/late,
    since it decides when liveness stops and hands off to the stall watchdog. The quiet_gate compares
    against _HB_LAST_UPLOAD (a time.time() stamp), so that one stays on wall clock."""
    hb = importlib.import_module("flash.engine.worker.heartbeat")
    src = inspect.getsource(inspect.unwrap(hb.liveness_heartbeat))
    # The silence window between an advance and the max_silence_s check is the monotonic span.
    silence = src[src.index("advanced_at =") : src.index("if quiet_gate_s is not None")]
    assert "time.monotonic()" in silence, "silence/progress interval must use time.monotonic()"
    assert "time.time()" not in silence, "silence/progress interval must NOT use wall-clock time.time()"
    # The quiet-gate still pairs with the wall-clock _HB_LAST_UPLOAD stamp.
    quiet = src[src.index("if quiet_gate_s is not None") : src.index("gpu = gpu_diagnostics")]
    assert "time.time()" in quiet, "quiet_gate must stay on wall clock to match _HB_LAST_UPLOAD"


def test_train_liveness_heartbeat_gap_fills_with_step(monkeypatch):
    """train_liveness_heartbeat composes liveness_heartbeat for the train phase: gap-fill the per-step
    stage when quiet, carrying the live global_step."""
    hb, w, _ = _liveness_env(monkeypatch)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)  # quiet -> gap-fill fires
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append((s, k.get("step"))))
    counter = {"n": 0}

    def step():
        counter["n"] += 1
        return counter["n"]

    with hb.train_liveness_heartbeat("sft_step", step):
        time.sleep(0.2)
    assert emitted
    assert all(s == "sft_step" for s, _ in emitted)
    assert any(st is not None for _, st in emitted), "train liveness must carry the step in the payload"


def test_liveness_heartbeat_survives_raising_fields_callback(monkeypatch):
    """The ``fields`` callback can re-read live state (train_liveness_heartbeat's lambda calls
    get_step() again), so a raise there must NOT kill the gap-filler thread for the rest of the wrapped
    block — same defensive contract as progress(). The daemon falls back to no extra fields and keeps
    emitting the bare heartbeat (which still re-arms the watchdog)."""
    hb, w, _ = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append((s, k)))

    def boom():
        raise RuntimeError("get_step exploded mid-train")

    with hb.liveness_heartbeat("rl_step", fields=boom):
        time.sleep(0.2)
    assert emitted, "a raising fields callback must not kill the daemon — it must keep emitting"
    assert all(s == "rl_step" for s, _ in emitted)
    assert all("step" not in k for _, k in emitted), "failed fields must fall back to no extra fields"


# --------------------------------------------------------------------------------------------
# _hf_cache_bytes feeds the prefetch progress gate: bytes downloaded, or None when the cache dir
# doesn't exist yet (unmeasurable). liveness_heartbeat treats None as "no advancement", so the
# unmeasurable pre-structure window is itself bounded by max_silence_s.
def test_hf_cache_bytes_counts_blobs_and_reports_unmeasurable_as_none(tmp_path, monkeypatch):
    import huggingface_hub.constants as hconst

    from flash.engine.worker import hf

    monkeypatch.setattr(hconst, "HF_HUB_CACHE", str(tmp_path))
    # No repo cache dir yet -> None (can't measure) -> never counts as progress (bounded by silence).
    assert hf._hf_cache_bytes("org/model") is None
    repo = tmp_path / "models--org--model"
    repo.mkdir(parents=True)
    # Repo dir exists but blobs/ not written yet -> 0 (a real "0 bytes" measurement), NOT None: a
    # download wedged before writing any blob must still let the silence timer trip.
    assert hf._hf_cache_bytes("org/model") == 0
    blobs = repo / "blobs"
    blobs.mkdir()
    (blobs / "complete").write_bytes(b"x" * 100)
    (blobs / "partial.incomplete").write_bytes(b"y" * 50)  # an in-flight download's growing partial
    assert hf._hf_cache_bytes("org/model") == 150


# --------------------------------------------------------------------------------------------
# Wiring: every long blocking phase must run under the shared liveness_heartbeat helper with the
# right stage / progress gate. (Behaviour is covered above; these pin the call sites so the coverage
# can't silently regress.)
def test_rl_init_wraps_trainer_build_in_init_liveness_heartbeat():
    from flash.engine.worker import rl

    # init_liveness_heartbeat (NOT the bare liveness_heartbeat) so the cold init ping is bounded by
    # max_duration and can't mask a stuck-but-GIL-releasing init to the wall-clock timeout.
    assert 'init_liveness_heartbeat("rl_initializing")' in inspect.getsource(rl.run_rl)


def test_sft_init_wraps_trainer_build_in_init_liveness_heartbeat():
    from flash.engine.worker import sft

    assert 'init_liveness_heartbeat("sft_initializing")' in inspect.getsource(sft.run_sft)


def test_init_liveness_heartbeat_is_bounded_by_max_duration():
    """init_liveness_heartbeat must pass max_duration_s (init has no incremental progress counter, so
    the bound is a duration cap) — pins the anti-mask wiring so it can't silently regress."""
    hb = importlib.import_module("flash.engine.worker.heartbeat")
    src = inspect.getsource(hb.init_liveness_heartbeat)
    assert "max_duration_s=" in src, "init liveness must be bounded by a max_duration cap"
    assert hb._INIT_LIVENESS_MAX_S > 0


@pytest.mark.parametrize(
    ("modname", "outer", "stage"),
    [
        ("flash.engine.worker.rl", "run_rl", "rl_step"),
        ("flash.engine.worker.sft", "run_sft", "sft_step"),
    ],
)
def test_train_phase_wraps_train_in_train_liveness_heartbeat(modname, outer, stage):
    mod = importlib.import_module(modname)
    src = inspect.getsource(getattr(mod, outer))
    assert "train_liveness_heartbeat(" in src, (
        f"{outer} must wrap trainer.train() in train_liveness_heartbeat — without a gap filler the "
        "cold first step emits no heartbeat and looks like a hang"
    )
    assert f'"{stage}"' in src, f"{outer} must pass stage {stage!r}"


def test_prefetch_wraps_download_in_liveness_heartbeat_gated_on_bytes():
    from flash.engine.worker import hf

    src = inspect.getsource(hf.prefetch_model)
    assert "liveness_heartbeat(" in src
    assert '"model_prefetching"' in src
    # The ping MUST be gated on downloaded-byte growth + a silence bound. Without these a NON-raising
    # wedge (stuck cache filelock / NFS I/O stall on the shared mount / endless retry) never returns
    # from snapshot_download, so the ping would re-arm the watchdog AND — model_prefetching is a setup
    # stage — the provider setup-grace forever, masking the stall until the wall-clock timeout.
    assert "progress=" in src, "prefetch ping must gate on a progress counter"
    assert "_hf_cache_bytes(" in src, "prefetch ping must gate on downloaded-byte growth"
    assert "max_silence_s=" in src, "prefetch ping must stop after a wedge silence bound"


# --------------------------------------------------------------------------------------------
# Stall watchdog default. A true hang (no heartbeat for the window) must self-dump every thread's
# stack and fail the run instead of wedging silently until the control-plane kill (the
# "process wedged, no console upload" gap). This is safe-by-default because every heartbeat re-arms
# the timer and the init/training heartbeats now keep ticking through slow-but-live phases.


def test_stall_watchdog_always_on_and_outlasts_provider_grace():
    import importlib

    hb = importlib.import_module("flash.engine.worker.heartbeat")
    assert hb._STALL_WATCHDOG_S > 0, "stall watchdog is always on (no env, no disable)"
    # Longer than the providers' setup grace so a stuck SETUP trips the provider's RETRIABLE path
    # before this exit=True watchdog hard-fails the run. Compare to the canonical provider value.
    from flash.providers.runpod import jobs as runpod_jobs

    assert runpod_jobs.stall_kwargs()["setup_grace_s"] < hb._STALL_WATCHDOG_S


def test_rearm_arms_the_single_window(monkeypatch):
    import importlib

    hb = importlib.import_module("flash.engine.worker.heartbeat")
    windows: list = []
    monkeypatch.setattr(
        hb.faulthandler, "dump_traceback_later", lambda w, exit=False: windows.append(w)
    )
    monkeypatch.setattr(hb.faulthandler, "cancel_dump_traceback_later", lambda: None)
    hb._rearm_stall_faulthandler()
    assert windows == [hb._STALL_WATCHDOG_S], "every arm uses the single always-on window"
