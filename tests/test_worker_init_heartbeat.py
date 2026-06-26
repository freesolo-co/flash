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

import ast
import inspect
import sys
import threading
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
# Wiring guards: the two init-heartbeat closures must keep calling gpu_diagnostics(include_torch=
# False). A future edit that drops the kwarg (or flips it) reintroduces the freeze, so assert it at
# the source/AST level (the closures are locals, so parse the enclosing function's source).


def _inner_func_node(outer_func, inner_name: str) -> ast.FunctionDef | None:
    tree = ast.parse(inspect.getsource(outer_func))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == inner_name:
            return node
    return None


def _assert_gpu_diag_disables_torch(node: ast.FunctionDef, where: str) -> None:
    calls = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "gpu_diagnostics"
    ]
    assert calls, f"{where}: no longer calls gpu_diagnostics — check this guard is still valid"
    for call in calls:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        val = kwargs.get("include_torch")
        msg = f"{where}: init heartbeat must call gpu_diagnostics(include_torch=False) (got {ast.dump(call)})"
        assert isinstance(val, ast.Constant), msg
        assert val.value is False, msg


def test_rl_init_heartbeat_disables_torch_telemetry():
    from flash.engine.worker import rl

    node = _inner_func_node(rl.run_rl, "_rl_init_heartbeat")
    assert node is not None, "run_rl no longer defines _rl_init_heartbeat"
    _assert_gpu_diag_disables_torch(node, "run_rl._rl_init_heartbeat")


def test_sft_init_heartbeat_disables_torch_telemetry():
    from flash.engine.worker import sft

    node = _inner_func_node(sft.run_sft, "_sft_init_heartbeat")
    assert node is not None, "run_sft no longer defines _sft_init_heartbeat"
    _assert_gpu_diag_disables_torch(node, "run_sft._sft_init_heartbeat")


# --------------------------------------------------------------------------------------------
# prefetch_model: snapshot_download blocks with NO heartbeat until it returns. A cold cache can pull
# tens of GB for many minutes — longer than the stall watchdog AND the provider setup grace — so the
# download must ping a progress heartbeat (re-arming the watchdog) or a HEALTHY long fetch self-kills.
def test_prefetch_model_heartbeats_during_download():
    from flash.engine.worker import hf

    node = _inner_func_node(hf.prefetch_model, "_prefetch_heartbeat")
    assert node is not None, (
        "prefetch_model no longer pings a heartbeat during snapshot_download — a long cold-cache "
        "download would look like a hang and trip the stall watchdog / provider setup grace"
    )
    # nvidia-smi only (the GPU isn't in use yet, and torch telemetry could block).
    _assert_gpu_diag_disables_torch(node, "prefetch_model._prefetch_heartbeat")
    # ...and it must emit the progress stage that proves liveness + re-arms the watchdog.
    stages = [
        n.args[0].value
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "heartbeat"
        and n.args
        and isinstance(n.args[0], ast.Constant)
    ]
    assert "model_prefetching" in stages, f"expected a model_prefetching heartbeat, got {stages}"


def test_prefetch_model_joins_heartbeat_with_a_bounded_timeout():
    """The prefetch heartbeat thread must be joined with a BOUNDED timeout, never unbounded.

    An unbounded ``join()`` would wedge the worker indefinitely if the side thread were stuck inside
    ``heartbeat()``'s HF upload (``hf_upload_file`` -> ``huggingface_hub.upload_file`` has no hard
    timeout): the model download would have completed, yet the main thread would hang here forever.
    Ordering of ``model_prefetched`` after the last ``model_prefetching`` does NOT depend on this
    join — it is guaranteed by the ``is_set()`` guard in the loop (asserted below) plus heartbeat()'s
    ``_HB_UPLOAD_LOCK`` (which serializes any in-flight upload before ``model_prefetched``'s) — so the
    join only reaps the daemon thread and must stay bounded."""
    from flash.engine.worker import hf

    tree = ast.parse(inspect.getsource(hf.prefetch_model))
    joins = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "join"
    ]
    assert joins, "prefetch_model no longer joins its heartbeat thread"
    for call in joins:
        # A bounded join is expressed as a positional timeout (join(10.0)) or join(timeout=...).
        positional = call.args and isinstance(call.args[0], ast.Constant)
        keyworded = any(kw.arg == "timeout" for kw in call.keywords)
        assert positional or keyworded, (
            "prefetch_model must join() its heartbeat thread with a BOUNDED timeout — an unbounded "
            "join can wedge the worker if the side thread is stuck inside an HF upload"
        )


@pytest.mark.parametrize(
    ("modname", "outer", "inner", "done"),
    [
        ("flash.engine.worker.hf", "prefetch_model", "_prefetch_heartbeat", "_prefetch_done"),
        ("flash.engine.worker.rl", "run_rl", "_rl_init_heartbeat", "_rl_init_done"),
        ("flash.engine.worker.sft", "run_sft", "_sft_init_heartbeat", "_sft_init_done"),
    ],
)
def test_periodic_heartbeat_threads_guard_on_done(modname, outer, inner, done):
    """Every periodic side-thread heartbeat must re-check its done Event AFTER wait() (the event can
    be set in the gap between wait() timing out and the emit) and return without emitting, so it can't
    publish a stale stage once the phase it covers has finished — the prefetch race, generalized to
    the init threads. (An emit after gpu_diagnostics is additionally guarded in the source; here we
    just pin that the closure consults <done>.is_set() before emitting. The shared train-liveness loop
    has its own guard, covered by test_train_liveness_loop_*.)"""
    import importlib

    mod = importlib.import_module(modname)
    node = _inner_func_node(getattr(mod, outer), inner)
    assert node is not None, f"{outer} no longer defines {inner}"

    guards = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "is_set"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == done
    ]
    assert guards, (
        f"{inner} must re-check {done}.is_set() after waking, then return without emitting — "
        "otherwise a stale stage can race past the terminal stage of the phase it covers"
    )


# --------------------------------------------------------------------------------------------
# A synthetic model_prefetching ping re-arms the watchdog + refreshes the provider timestamp, so it
# must NOT mask a WEDGED snapshot_download (no bytes). The liveness is gated on real download progress
# (HF cache byte growth); after _MAX_PREFETCH_SILENCE_S of zero growth it stops so the stall path fires.
def test_hf_cache_bytes_counts_blobs_and_fails_safe(tmp_path, monkeypatch):
    import huggingface_hub.constants as hconst

    from flash.engine.worker import hf

    monkeypatch.setattr(hconst, "HF_HUB_CACHE", str(tmp_path))
    # No repo cache dir yet -> -1 (can't measure), so the caller assumes progress (no false-stop).
    assert hf._hf_cache_bytes("org/model") == -1
    repo = tmp_path / "models--org--model"
    repo.mkdir(parents=True)
    # Repo dir exists but blobs/ not written yet -> 0 (a real "0 bytes" measurement), NOT -1: a
    # download wedged before writing any blob must still let the silence timer trip.
    assert hf._hf_cache_bytes("org/model") == 0
    blobs = repo / "blobs"
    blobs.mkdir()
    (blobs / "complete").write_bytes(b"x" * 100)
    (blobs / "partial.incomplete").write_bytes(b"y" * 50)  # an in-flight download's growing partial
    assert hf._hf_cache_bytes("org/model") == 150


def test_prefetch_heartbeat_gates_on_download_progress():
    from flash.engine.worker import hf

    node = _inner_func_node(hf.prefetch_model, "_prefetch_heartbeat")
    assert node is not None, "prefetch_model no longer defines _prefetch_heartbeat"
    calls_cache = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_hf_cache_bytes"
    ]
    assert calls_cache, "prefetch liveness must gate on real download progress via _hf_cache_bytes"
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    assert "_MAX_PREFETCH_SILENCE_S" in names, (
        "prefetch liveness must STOP after _MAX_PREFETCH_SILENCE_S of no byte progress so a wedged "
        "download trips the stall path instead of being masked to the job timeout"
    )
    # The silence timer must reset ONLY on measured byte GROWTH, never on an unmeasurable -1: else a
    # download wedged before it ever creates the cache dir (pre-structure Hub-metadata phase) would
    # reset the timer every tick and emit forever. So the unmeasurable window is itself bounded.
    src = inspect.getsource(hf.prefetch_model)
    assert "cur >= 0 and cur > last_bytes" in src, (
        "prefetch silence must reset only on measured growth (cur >= 0 and cur > last_bytes)"
    )
    assert "cur < 0" not in src, "an unmeasurable -1 must NOT reset the prefetch silence timer"


# --------------------------------------------------------------------------------------------
# Training-phase gap-filler. The per-step rl_step/sft_step heartbeat fires on on_log AFTER a step, so
# the FIRST step (cold vLLM rollout warmup + backward — measured ~17 min on a consumer GPU; SFT logs
# only every N steps) emits NO heartbeat while it runs. That stale-heartbeat window was the actual
# escalation symptom (and a slow-enough step would trip the stall watchdog). BOTH run_rl and run_sft
# must run the shared liveness daemon (heartbeat.train_liveness_loop) around trainer.train().


@pytest.mark.parametrize(
    ("modname", "outer", "stage"),
    [
        ("flash.engine.worker.rl", "run_rl", "rl_step"),
        ("flash.engine.worker.sft", "run_sft", "sft_step"),
    ],
)
def test_train_phase_runs_shared_liveness_loop(modname, outer, stage):
    import importlib

    mod = importlib.import_module(modname)
    src = inspect.getsource(getattr(mod, outer))
    tree = ast.parse(src)
    # It must spawn the SHARED train_liveness_loop (not a bespoke inline closure) with this stage.
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "train_liveness_loop"
    ]
    target_refs = "train_liveness_loop" in src and "target=train_liveness_loop" in src
    assert calls or target_refs, (
        f"{outer} must run heartbeat.train_liveness_loop around trainer.train() — without a gap "
        "filler the cold first step emits no heartbeat and looks like a hang"
    )
    assert f'"{stage}"' in src, f"{outer} must pass stage {stage!r} to train_liveness_loop"


def test_train_liveness_loop_emits_when_quiet_and_advancing(monkeypatch):
    """The gap-filler emits the per-step stage while the channel is quiet and the step is advancing."""
    import importlib
    import threading
    import time

    hb = importlib.import_module("flash.engine.worker.heartbeat")
    import flash.engine.worker as w

    monkeypatch.setattr(hb, "_TRAIN_LIVENESS_TICK_S", 0.01)
    monkeypatch.setattr(hb, "_TRAIN_LIVENESS_QUIET_S", 0.0)  # always "quiet" -> emit each tick
    monkeypatch.setattr(hb, "_TRAIN_LIVENESS_MAX_STEP_S", 60.0)  # generous -> never the stop reason
    monkeypatch.setattr(hb, "gpu_diagnostics", lambda include_torch=True: {})
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    emitted = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append((s, k.get("step"))))

    counter = {"n": 0}

    def advancing():
        counter["n"] += 1
        return counter["n"]

    done = threading.Event()
    th = threading.Thread(target=hb.train_liveness_loop, args=(done, advancing, "sft_step"), daemon=True)
    th.start()
    time.sleep(0.3)
    done.set()
    th.join(timeout=1.0)
    assert not th.is_alive()
    assert emitted, "expected liveness heartbeats while quiet + advancing"
    assert all(s == "sft_step" for s, _ in emitted)


def test_train_liveness_loop_stops_when_step_is_stuck(monkeypatch):
    """A synthetic per-step heartbeat re-arms the watchdog + refreshes the provider timestamp, so the
    gap-filler must NOT emit forever when train() is genuinely stuck (no global_step progress). After
    _TRAIN_LIVENESS_MAX_STEP_S of no advancement it must STOP so the stall path can fire."""
    import importlib
    import threading
    import time

    hb = importlib.import_module("flash.engine.worker.heartbeat")
    import flash.engine.worker as w

    monkeypatch.setattr(hb, "_TRAIN_LIVENESS_TICK_S", 0.01)
    monkeypatch.setattr(hb, "_TRAIN_LIVENESS_QUIET_S", 0.0)
    monkeypatch.setattr(hb, "_TRAIN_LIVENESS_MAX_STEP_S", 0.2)  # stuck after 0.2s of no advance
    monkeypatch.setattr(hb, "gpu_diagnostics", lambda include_torch=True: {})
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    emitted = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append(s))

    done = threading.Event()
    th = threading.Thread(target=hb.train_liveness_loop, args=(done, lambda: 0, "rl_step"), daemon=True)
    th.start()
    time.sleep(1.0)  # well past _TRAIN_LIVENESS_MAX_STEP_S
    stopped = not th.is_alive()
    done.set()
    th.join(timeout=1.0)
    assert stopped, "train_liveness_loop must STOP covering a stuck (no-progress) step, not emit forever"
    # It covered the step briefly (a few emits) then stopped — bounded, not unbounded.
    assert len(emitted) <= 40, f"expected a bounded number of synthetic emits before stopping, got {len(emitted)}"


# --------------------------------------------------------------------------------------------
# Stall watchdog default. A true hang (no heartbeat for the window) must self-dump every thread's
# stack and fail the run instead of wedging silently until the control-plane kill (the
# "process wedged, no console upload" gap). This is safe-by-default because every heartbeat re-arms
# the timer and the init/training heartbeats now keep ticking through slow-but-live phases.


@pytest.mark.skipif(
    bool(
        {"FLASH_STALL_FAULTHANDLER_S", "FLASH_STALL_FAULTHANDLER_STARTUP_S"}
        & set(__import__("os").environ)
    ),
    # Both knobs are asserted below; either one set in the env overrides the defaults under test.
    reason="env overrides the watchdog default(s) under test",
)
def test_stall_watchdog_enabled_by_default():
    import importlib

    # Note: ``from flash.engine.worker import heartbeat`` resolves to the re-exported heartbeat()
    # FUNCTION, not the submodule — import the module path explicitly. Reload so the module-level
    # defaults reflect the env AT TEST TIME, not whatever env was present when it was first imported
    # (another test may have reloaded it under a patched env).
    hb = importlib.reload(importlib.import_module("flash.engine.worker.heartbeat"))

    assert hb._STALL_FAULTHANDLER_S >= 600, "stall watchdog must be ON by default so hangs self-dump"
    # The setup window must be at least as wide as the steady-state window...
    assert hb._STALL_STARTUP_GRACE_S >= hb._STALL_FAULTHANDLER_S
    # ...and STRICTLY LONGER than the providers' setup grace, so a stuck setup trips the provider's
    # RETRIABLE setup-stall path before this (exit=True) faulthandler hard-fails the run. Compare to
    # the canonical provider value rather than a magic number so the two can't silently drift apart.
    from flash.providers.runpod import jobs as runpod_jobs

    provider_setup_grace = runpod_jobs.stall_kwargs()["setup_grace_s"]
    assert provider_setup_grace < hb._STALL_STARTUP_GRACE_S, (
        "setup faulthandler window must be strictly longer than the provider setup_grace_s "
        f"({hb._STALL_STARTUP_GRACE_S} vs {provider_setup_grace})"
    )


def test_stall_watchdog_disabled_via_worker_env(monkeypatch):
    import importlib

    hb = importlib.import_module("flash.engine.worker.heartbeat")

    monkeypatch.setenv("FLASH_STALL_FAULTHANDLER_S", "0")
    try:
        importlib.reload(hb)
        assert hb._STALL_FAULTHANDLER_S == 0, "FLASH_STALL_FAULTHANDLER_S=0 must disable the watchdog"
    finally:
        monkeypatch.delenv("FLASH_STALL_FAULTHANDLER_S", raising=False)
        importlib.reload(hb)  # restore to env-current state for the rest of the suite


@pytest.mark.skipif(
    bool(
        {"FLASH_STALL_FAULTHANDLER_S", "FLASH_STALL_FAULTHANDLER_STARTUP_S"}
        & set(__import__("os").environ)
    ),
    reason="env overrides the watchdog windows under test",
)
def test_stall_watchdog_stays_wide_until_first_training_step(monkeypatch):
    """The watchdog must keep the WIDE setup grace for EVERY arm until the first per-step TRAINING
    heartbeat (rl_step/sft_step) — not just the first arm. The whole cold start (prefetch, weight
    load, vLLM/trainer build, and the silent full-dataset render/tokenize over a possibly-uncapped
    dataset) runs before any rl_step/sft_step, so tightening on the first (setup) heartbeat would
    false-kill a healthy-but-slow setup. The providers' SETUP_HEARTBEAT_STAGES draws the same line."""
    import importlib

    hb = importlib.reload(importlib.import_module("flash.engine.worker.heartbeat"))
    if hb._STALL_FAULTHANDLER_S <= 0:
        pytest.skip("watchdog disabled")

    windows: list = []
    monkeypatch.setattr(
        hb.faulthandler, "dump_traceback_later", lambda w, exit=False: windows.append(w)
    )
    monkeypatch.setattr(hb.faulthandler, "cancel_dump_traceback_later", lambda: None)

    setup_window = max(hb._STALL_FAULTHANDLER_S, hb._STALL_STARTUP_GRACE_S)
    assert setup_window > hb._STALL_FAULTHANDLER_S, (
        "setup grace must be strictly wider than the training window (>= provider SETUP_GRACE_S), "
        "else a slow silent setup phase gets only the tight training window"
    )

    # Every SETUP arm (start ping, prefetch ping, model load, init ping) keeps the wide setup grace.
    for stage in ("rl_start", "model_prefetching", "sft_model_load", "rl_initializing"):
        hb._rearm_stall_faulthandler(stage)
        assert windows[-1] == setup_window, f"{stage} must keep the wide setup grace"

    # The first per-step TRAINING heartbeat tightens to the steady-state window — and stays tight
    # even if a late setup-shaped ping arrives afterward.
    hb._rearm_stall_faulthandler("sft_step")
    assert windows[-1] == hb._STALL_FAULTHANDLER_S, "first training step must tighten the watchdog"
    hb._rearm_stall_faulthandler("rl_initializing")
    assert windows[-1] == hb._STALL_FAULTHANDLER_S, "watchdog must stay tight once training started"
