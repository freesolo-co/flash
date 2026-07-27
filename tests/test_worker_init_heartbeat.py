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
import math
import re
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


def test_liveness_heartbeat_progress_step_stamps_step(monkeypatch):
    """progress_step=True stamps the progress counter as ``step`` on every emit, so the poller's
    step gate and cancel billing see the true step even when the daemon wins the upload slot."""
    hb, w, _ = _liveness_env(monkeypatch)
    seen: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: seen.append((k.get("liveness"), k.get("step"))))
    vals = iter([3, 7])  # advances once, then stalls at 7
    with hb.liveness_heartbeat("sft_step", progress=lambda: next(vals, 7), progress_step=True):
        time.sleep(0.2)
    assert seen
    assert all(step in (3, 7) for _, step in seen), "every emit must stamp the last seen step"
    assert any(step == 7 and liveness for liveness, step in seen), (
        "liveness pings must carry the step too"
    )


def test_liveness_heartbeat_first_progress_sample_is_baseline_not_progress(monkeypatch):
    """A resumed run's restored global_step (or a constant finalize counter) must NOT count as
    progress on the daemon's first sample — it would emit a real step>=1 heartbeat seconds into
    train() and prematurely tighten the provider's stall window, defeating the per-attempt
    setup-grace re-arm. Only an ADVANCE past the first-seen value is progress."""
    hb, w, _ = _liveness_env(monkeypatch)
    seen: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: seen.append((k.get("liveness"), k.get("step"))))
    with hb.liveness_heartbeat("rl_step", progress=lambda: 57, progress_step=True):
        time.sleep(0.2)
    assert seen
    assert all(liveness is True for liveness, _ in seen), (
        "a never-advancing counter must emit only liveness pings"
    )
    assert all(step == 57 for _, step in seen), "pings still stamp the baseline step"


def test_liveness_heartbeat_keepalive_forces_real_heartbeats_on_constant_progress(monkeypatch):
    """keepalive=True: a synchronous checkpoint/finalize upload freezes global_step (a CONSTANT
    counter), which — per the test above — would emit ONLY liveness pings, and those do NOT advance
    the provider's stall clock (surface_heartbeat returns stage=None for them). A healthy upload that
    outlasts STALL_AFTER_S would then be killed mid-save. keepalive forces a REAL (liveness=False)
    heartbeat every tick, still stamped with the step so a cancel landing here still bills it."""
    hb, w, _ = _liveness_env(monkeypatch)
    seen: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: seen.append((k.get("liveness"), k.get("step"))))
    with hb.liveness_heartbeat(
        "checkpoint_uploading", progress=lambda: 42, progress_step=True, keepalive=True
    ):
        time.sleep(0.2)
    assert seen, "must emit while alive"
    assert all(liveness is False for liveness, _ in seen), (
        "keepalive must emit REAL (stall-clock-advancing) heartbeats even when progress never advances"
    )
    assert all(step == 42 for _, step in seen), "keepalive still stamps the step for cancel billing"


def test_checkpoint_uploading_keepalive_stage_is_throttled_on_tight_cadence():
    """The checkpoint-upload keepalive daemon re-emits a REAL heartbeat every 30s, so it MUST be
    throttled (else ~120/hr blows the HF commit cap) AND ride the tighter setup-liveness interval, so
    the provider stall clock is refreshed well inside STALL_AFTER_S rather than every _HB_MIN_INTERVAL_S."""
    import flash.engine.worker as ne
    from flash.providers._poll import STALL_AFTER_S

    assert "checkpoint_uploading" in ne._HB_UPLOAD_LIVENESS_STAGES
    assert "checkpoint_uploading" in ne._HB_TIGHT_LIVENESS_STAGES
    assert "checkpoint_uploading" in ne._HB_THROTTLED_STAGES
    assert ne._HB_UPLOAD_LIVENESS_STAGES <= ne._HB_THROTTLED_STAGES
    # tight interval must leave margin under the training stall window (a single refresh must land).
    assert ne._HB_SETUP_LIVENESS_INTERVAL_S < STALL_AFTER_S


def test_liveness_heartbeat_survives_inline_thread_stub(monkeypatch):
    """Several suites stub threading.Thread to run targets INLINE on .start() (to make the
    checkpoint-upload daemon synchronous). The liveness daemon must detect it is running on the
    spawning thread and bail, or the inlined loop spins forever and hangs the whole test run
    (bit test_resume_on_retry via the checkpoint_prefetching wrap in hf_resume_checkpoint)."""
    hb, w, _ = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(w, "heartbeat", lambda s, **k: emitted.append(s))

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            self._target()

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(hb.threading, "Thread", _SyncThread)
    t0 = time.time()
    with hb.liveness_heartbeat("checkpoint_prefetching"):
        pass
    assert time.time() - t0 < 5, "inlined daemon must return immediately, not spin on done.wait"
    assert emitted == [], "an inlined daemon emits nothing"


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


def test_reward_heartbeat_projects_bounded_per_step_metrics(monkeypatch):
    hb = importlib.import_module("flash.engine.worker.heartbeat")
    import flash.engine.worker as w

    transformers = types.ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(hb, "_maybe_attach_gpu_diag", lambda payload, last, now: last)
    emitted = []
    monkeypatch.setattr(w, "heartbeat", lambda stage, **payload: emitted.append((stage, payload)))

    callback = hb.make_reward_heartbeat_callback()
    args = types.SimpleNamespace(max_completion_length=256)
    state = types.SimpleNamespace(global_step=1)
    callback.on_log(
        args,
        state,
        None,
        logs={
            "reward": 0.75,
            "reward_std": 0.12,
            "grad_norm": 1.5,
            "kl": 0.03,
            "entropy": 0.82,
            "frac_reward_zero_std": 0.25,
            "completions/mean_length": 48.5,
            "completions/clipped_ratio": 0.125,
        },
    )

    stage, payload = emitted[-1]
    assert stage == "rl_step"
    expected = {
        "step": 1,
        "reward": 0.75,
        "reward_std": 0.12,
        "grad_norm": 1.5,
        "kl": 0.03,
        "entropy": 0.82,
        "frac_reward_zero_std": 0.25,
        "mean_completion_tokens": 48.5,
        "truncation_rate": 0.125,
        "max_completion_tokens": 256,
    }
    for key, value in expected.items():
        assert payload[key] == value
    assert payload["metrics_last"] == [expected]

    state.global_step = 2
    callback.on_log(
        args,
        state,
        None,
        logs={
            "reward": 0.8,
            "reward_std": float("nan"),
            "grad_norm": float("inf"),
            "entropy": 0.79,
        },
    )
    payload = emitted[-1][1]
    assert payload["reward"] == 0.8
    assert payload["entropy"] == 0.79
    for key in ("reward_std", "grad_norm", "kl"):
        assert key not in payload
        assert key not in payload["metrics_last"][-1]

    for step in range(3, 1027):
        state.global_step = step
        callback.on_log(args, state, None, logs={"reward": step / 1027})
    metrics_last = emitted[-1][1]["metrics_last"]
    assert len(metrics_last) == 1024
    assert [item["step"] for item in metrics_last] == list(range(3, 1027))


def test_heartbeat_console_summarizes_metric_backlog():
    import json

    from flash.engine.worker.heartbeat import _console_heartbeat_snapshot

    console = json.loads(
        _console_heartbeat_snapshot(
            {
                "stage": "rl_step",
                "step": 1024,
                "metrics_last": [{"step": step} for step in range(1024)],
            }
        )
    )

    assert "metrics_last" not in console
    assert console["metrics_last_count"] == 1024
    assert console["step"] == 1024


def test_rl_lifecycle_heartbeats_carry_latest_metrics():
    import ast
    import textwrap

    from flash.engine.worker import rl

    tree = ast.parse(textwrap.dedent(inspect.getsource(rl.run_rl)))
    terminal_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "heartbeat"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "rl_trained"
    ]
    assert len(terminal_calls) == 1
    terminal_keywords = {keyword.arg: keyword.value for keyword in terminal_calls[0].keywords}
    assert "metrics_last" in terminal_keywords
    assert "hb_cb" in ast.unparse(terminal_keywords["metrics_last"])

    liveness_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "liveness_heartbeat"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in {"rl_step", "rl_finalizing"}
    ]
    assert len(liveness_calls) == 2
    for call in liveness_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "fields" in keywords
        assert "metrics_last" in ast.unparse(keywords["fields"])
        assert "hb_cb" in ast.unparse(keywords["fields"])

    write_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "write_train_meta"
    ]
    assert len(write_calls) == 1
    write_keywords = {keyword.arg: keyword.value for keyword in write_calls[0].keywords}
    assert "heartbeat_fields" in write_keywords
    assert "metrics_last" in ast.unparse(write_keywords["heartbeat_fields"])
    assert "hb_cb" in ast.unparse(write_keywords["heartbeat_fields"])


def test_error_heartbeat_fallback_preserves_metric_backlog():
    # regression (#591): the error_{RUN_MODE} heartbeat exists to surface the bounded metric
    # backlog (metrics_last) for short failing RL runs. main() emits it twice -- a primary call
    # (with gpu diagnostics) and an except-fallback used if that primary call raises. BOTH must
    # splat **_err_metrics, or a failure inside the primary call (e.g. gpu_diagnostics() or the
    # heartbeat upload itself) re-emits an error snapshot that drops the very backlog this path
    # was added to preserve.
    import ast
    import textwrap

    import flash.engine.worker as ne

    tree = ast.parse(textwrap.dedent(inspect.getsource(ne.main)))
    error_hb_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "heartbeat"
        and node.args
        and isinstance(node.args[0], ast.JoinedStr)
        and ast.unparse(node.args[0]).startswith(("f'error_", 'f"error_'))
    ]
    assert len(error_hb_calls) == 2, "expected a primary and a fallback error heartbeat in main()"
    for call in error_hb_calls:
        assert any(
            keyword.arg is None and ast.unparse(keyword.value) == "_err_metrics"
            for keyword in call.keywords
        ), "both the primary and fallback error heartbeats must splat **_err_metrics (metrics_last)"


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


def test_opd_step_post_update_heartbeat_forces_through_throttle(monkeypatch):
    """Regression (codex[bot], heartbeat.py/opd.py): a mid-step opd_step progress ping (carrying the
    PREVIOUS opt_steps) can claim the throttle slot immediately before the post-update ping (the
    incremented step). Without force the stepped commit is throttled out, so a cancellation is billed
    from the STALE step even though the update landed. heartbeat(force=True) must upload within the
    throttle interval; a NON-forced opd_step in the same window is still throttled."""
    import flash.engine.worker as ne

    uploads: list = []
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 60.0)
    monkeypatch.setattr(ne, "hf_upload_file", lambda local, *a, **k: uploads.append(local))
    ne._HB_LAST_UPLOAD = 0.0
    ne._HB_LAST_FORCED_UPLOAD = 0.0
    ne._HB_LAST_COMMITTED_STEP = 0
    ne.heartbeat("opd_step", step=5, samples_done=1)  # mid-step ping claims the slot (stale step 5)
    assert len(uploads) == 1
    ne.heartbeat("opd_step", step=6, samples_done=2)  # normal ping within 60s -> throttled out
    assert len(uploads) == 1, "a non-forced opd_step within the interval must be throttled"
    ne.heartbeat("opd_step", step=6, loss=0.1, coverage=1.0, force=True)  # post-update forces through
    assert len(uploads) == 2, "force=True must commit the stepped post-update ping despite the throttle"


@pytest.mark.parametrize("stage", ["rl_step", "opd_step"])
def test_forced_sample_payload_commits_after_same_step_liveness(monkeypatch, stage):
    """a liveness commit for a step must not throttle the first sample payload for that step."""
    import flash.engine.worker as ne

    uploads: list = []
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(ne, "_HB_FORCE_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(ne, "hf_upload_file", lambda local, *a, **k: uploads.append(local))
    ne._HB_LAST_UPLOAD = 0.0
    ne._HB_LAST_FORCED_UPLOAD = 0.0
    ne._HB_LAST_COMMITTED_STEP = 0

    ne.heartbeat(stage, step=1)
    assert len(uploads) == 1

    committed = ne.heartbeat(
        stage,
        step=1,
        force=True,
        sampled_completions=[
            {
                "prompt_tail": "prompt",
                "completion": "completion",
                "reward" if stage == "rl_step" else "loss": 1.0,
                "generated_at_step": 0,
            }
        ],
    )

    assert committed is True
    assert len(uploads) == 2


def test_initial_rl_step_persists_through_throttle_and_bills_cancel(monkeypatch):
    import json

    import flash.engine.worker as ne
    import flash.runner as runner

    persisted = []

    def upload(local, *args, **kwargs):
        with open(local) as f:
            persisted.append(json.load(f))
        return True

    monkeypatch.setattr(ne, "hf_upload_file", upload)
    ne._HB_LAST_UPLOAD = time.time()
    ne._HB_LAST_COMMITTED_STEP = 0

    committed = ne.heartbeat("rl_step", step=0, initial=True)

    assert committed is True
    assert persisted[-1]["stage"] == "rl_step"
    assert persisted[-1]["step"] == 0
    status = runner.RunStatus(
        run_id="r",
        state="cancelled",
        spec={},
        last_heartbeat=persisted[-1],
    )
    assert runner.actual_steps_run(status) == 1


def test_forced_opd_step_commits_each_distinct_step_advance(monkeypatch):
    """Regression (cursor[bot], heartbeat.py): when optimizer steps land FARTHER apart than the force
    floor (the normal teacher-round-trip-gated regime), every DISTINCT completed step still commits
    exactly once within the 900s throttle window, so a cancel always bills the true latest step — none
    is dropped. (A sub-floor BURST is instead coalesced to protect the HF commit cap; see the burst test
    below.) Modelled with a zero force floor: each advancing step exceeds it, so none is floored out."""
    import flash.engine.worker as ne

    uploads: list = []
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(ne, "_HB_FORCE_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(ne, "hf_upload_file", lambda local, *a, **k: uploads.append(local))
    ne._HB_LAST_UPLOAD = 0.0
    ne._HB_LAST_FORCED_UPLOAD = 0.0
    ne._HB_LAST_COMMITTED_STEP = 0
    for stepv in (1, 2, 3, 4):
        ne.heartbeat("opd_step", step=stepv, loss=0.1, force=True)
    assert len(uploads) == 4, "each distinct forced step advance beyond the floor must commit"
    assert ne._HB_LAST_COMMITTED_STEP == 4


def test_forced_opd_step_burst_within_floor_coalesces_to_protect_commit_cap(monkeypatch):
    """Regression (codex[bot], heartbeat.py): a tiny/fast OPD config (batch=1, group=1, small student,
    cached teacher) can land optimizer updates many times per minute. Unthrottled, force=True would turn
    every post-step ping into an HF commit and blow the per-repo commit cap before the final adapter/DONE
    upload. Forced commits are floored: the FIRST advance in a sub-floor burst commits, the rest within
    _HB_FORCE_MIN_INTERVAL_S coalesce (the persisted step then lags by at most one floor window — a
    bounded, customer-favouring cancel under-bill)."""
    import flash.engine.worker as ne

    uploads: list = []
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(ne, "_HB_FORCE_MIN_INTERVAL_S", 60.0)
    monkeypatch.setattr(ne, "hf_upload_file", lambda local, *a, **k: uploads.append(local))
    ne._HB_LAST_UPLOAD = time.time()  # a recent upload -> the 900s regular throttle blocks every ping
    ne._HB_LAST_FORCED_UPLOAD = 0.0  # forced clock cold -> only the first advance punches through
    ne._HB_LAST_COMMITTED_STEP = 0
    for stepv in (1, 2, 3, 4, 5):
        ne.heartbeat("opd_step", step=stepv, loss=0.1, force=True)
    assert len(uploads) == 1, "a sub-floor burst of forced step-advances must commit once, not per step"
    assert ne._HB_LAST_COMMITTED_STEP == 1


def test_force_commit_via_regular_throttle_arms_the_floor(monkeypatch):
    """Regression (cursor[bot], heartbeat.py): a force=True heartbeat that commits because the regular
    throttle was ALREADY due (not because the force branch bypassed it) must STILL arm the forced-commit
    clock — else the clock stays stale and a following sub-floor forced ping punches through, defeating
    the burst coalescing that protects the HF commit cap."""
    import flash.engine.worker as ne

    uploads: list = []
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(ne, "_HB_FORCE_MIN_INTERVAL_S", 60.0)
    monkeypatch.setattr(ne, "hf_upload_file", lambda local, *a, **k: uploads.append(local))
    ne._HB_LAST_UPLOAD = 0.0  # regular throttle is DUE -> the first force commits via it, not the bypass
    ne._HB_LAST_FORCED_UPLOAD = 0.0
    ne._HB_LAST_COMMITTED_STEP = 0
    ne.heartbeat("opd_step", step=1, loss=0.1, force=True)  # commits via the elapsed regular throttle
    assert len(uploads) == 1
    ne.heartbeat("opd_step", step=2, loss=0.1, force=True)  # sub-floor advance -> must be COALESCED now
    assert len(uploads) == 1, "the regular-path force commit must arm the floor so the next is coalesced"
    assert ne._HB_LAST_COMMITTED_STEP == 1


def test_forced_opd_step_repeated_same_step_does_not_recommit(monkeypatch):
    """Self-limiting counterpart: a forced ping whose step does NOT advance past the last committed step
    (a redundant post-update carrying the same opt_steps, or a retry) stays throttled. So forcing can't
    inflate commits beyond the actual optimizer-step rate and blow the HF per-repo commit cap."""
    import flash.engine.worker as ne

    uploads: list = []
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(ne, "hf_upload_file", lambda local, *a, **k: uploads.append(local))
    ne._HB_LAST_UPLOAD = 0.0
    ne._HB_LAST_FORCED_UPLOAD = 0.0
    ne._HB_LAST_COMMITTED_STEP = 0
    ne.heartbeat("opd_step", step=7, loss=0.1, force=True)  # first commit at step 7
    assert len(uploads) == 1
    for _ in range(3):  # same step, within throttle -> force must NOT re-commit
        ne.heartbeat("opd_step", step=7, loss=0.1, force=True)
    assert len(uploads) == 1, "forced pings that don't advance the step must not re-commit"
    assert ne._HB_LAST_COMMITTED_STEP == 7


def test_forced_opd_step_commit_failure_rolls_back_committed_step(monkeypatch):
    """If a forced commit's upload FAILS, the committed-step marker AND the forced-commit clock roll back
    with the throttle slot — so the retry at the same step still forces through (fstep must again exceed
    the last committed step, and the floor must not treat the failed attempt as a landed forced commit
    that delays the retry). Otherwise the failed step would be recorded as committed / the retry would be
    floored out, and a cancel would bill the stale prior step. (Force floor zeroed so the two forced
    attempts here aren't coalesced — that path is the burst test.)"""
    import flash.engine.worker as ne

    attempts: list = []
    fail = {"on": False}

    def _upload(local, *a, **k):
        attempts.append(1)
        return not fail["on"]

    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(ne, "_HB_FORCE_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(ne, "hf_upload_file", _upload)
    ne._HB_LAST_UPLOAD = 0.0
    ne._HB_LAST_FORCED_UPLOAD = 0.0
    ne._HB_LAST_COMMITTED_STEP = 0
    ne.heartbeat("opd_step", step=2, force=True)  # succeeds -> committed step = 2, slot claimed
    assert ne._HB_LAST_COMMITTED_STEP == 2
    forced_clock_after_2 = ne._HB_LAST_FORCED_UPLOAD  # any force=True commit arms the forced clock
    fail["on"] = True
    ne.heartbeat("opd_step", step=3, force=True)  # forces (3>2) within throttle, but upload FAILS
    assert ne._HB_LAST_COMMITTED_STEP == 2, "a failed forced commit must roll back the committed step"
    assert forced_clock_after_2 == ne._HB_LAST_FORCED_UPLOAD, "and roll back the forced clock"
    fail["on"] = False
    ne.heartbeat("opd_step", step=3, force=True)  # retry: still throttled by time, must force on 3>2
    assert ne._HB_LAST_COMMITTED_STEP == 3
    assert len(attempts) == 3, "the retry must re-attempt the upload, not be throttled/blocked out"


def test_setup_liveness_upload_uses_shorter_interval(monkeypatch):
    """Setup liveness must refresh public status before a 300s external frozen-heartbeat watchdog,
    while training-step liveness stays under the normal shared-repo throttle."""
    import json

    import flash.engine.worker as ne

    hbmod = importlib.import_module("flash.engine.worker.heartbeat")

    now = {"t": 1000.0}
    uploads: list[dict] = []

    def _capture(local, *a, **k):
        with open(local) as f:
            uploads.append(json.load(f))

    monkeypatch.setattr(hbmod.time, "time", lambda: now["t"])
    monkeypatch.setattr(ne, "hf_upload_file", _capture)
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(ne, "_HB_SETUP_LIVENESS_INTERVAL_S", 240.0)

    ne._HB_LAST_UPLOAD = 1000.0
    now["t"] = 1239.0
    ne.heartbeat("sft_initializing", liveness=True)
    assert uploads == []

    now["t"] = 1241.0
    ne.heartbeat("sft_initializing", liveness=True)
    assert uploads[-1]["stage"] == "sft_initializing"
    assert uploads[-1]["liveness"] is True

    uploads.clear()
    ne._HB_LAST_UPLOAD = 1000.0
    now["t"] = 1241.0
    ne.heartbeat("sft_step", liveness=True, step=0)
    assert uploads == [], "training-step liveness must stay on _HB_MIN_INTERVAL_S"


def test_default_heartbeat_interval_fits_shared_environment_repos():
    import flash.engine.worker as ne

    assert ne._HB_MIN_INTERVAL_S >= 900.0
    assert ne._HB_MIN_INTERVAL_S < 1200.0
    assert 180.0 <= ne._HB_SETUP_LIVENESS_INTERVAL_S < 300.0


def test_every_setup_liveness_stage_is_throttled():
    """The liveness daemon re-emits its stage every 30s; a setup-liveness stage missing from
    _HB_THROTTLED_STAGES would commit unthrottled (~120/hr) and blow the shared repo commit cap."""
    import flash.engine.worker as ne

    assert ne._HB_SETUP_LIVENESS_STAGES <= ne._HB_THROTTLED_STAGES


def test_quiet_phases_upload_at_setup_liveness_interval():
    """Every liveness-wrapped quiet phase must refresh public status at the faster setup cadence,
    or run status looks frozen for up to 15 min during a healthy phase."""
    import flash.engine.worker as ne

    for stage in (
        "model_prefetching",
        "checkpoint_prefetching",
        "sft_data_loading",
        "rl_data_loading",
        "rl_adapter_loading",
        "sft_pretokenizing",
        "sft_initializing",
        "rl_initializing",
        "sft_finalizing",
        "rl_finalizing",
    ):
        assert stage in ne._HB_SETUP_LIVENESS_STAGES, stage


# --------------------------------------------------------------------------------------------
# Progress-carry latch: a real heartbeat that never reached HF (throttled away or failed upload)
# upgrades the NEXT committed liveness ping to a real heartbeat. Without it, the train-loop liveness
# daemon can win the shared 900s upload slot with a bare ping, deferring the real per-step heartbeat
# to T+1800s > the provider's 1500s stall window — a healthy training run killed as "stalled".
def _reset_hb_state(monkeypatch, ne, *, last_upload=0.0):
    # monkeypatch (not direct assignment) so the latch state is restored after each test — a
    # leaked pending latch would make unrelated later tests order-dependent.
    monkeypatch.setattr(ne, "_HB_LAST_UPLOAD", last_upload)
    monkeypatch.setattr(ne, "_HB_LAST_PROGRESS_TS", 0.0)
    monkeypatch.setattr(ne, "_HB_PROGRESS_SEQ", 0)
    monkeypatch.setattr(ne, "_HB_PROGRESS_UPLOADED_SEQ", 0)


def test_progress_carry_upgrades_ping_after_throttled_real_heartbeat(monkeypatch):
    import json

    import flash.engine.worker as ne

    hbmod = importlib.import_module("flash.engine.worker.heartbeat")
    now = {"t": 1000.0}
    uploads: list[dict] = []

    def _capture(local, *a, **k):
        with open(local) as f:
            uploads.append(json.load(f))

    monkeypatch.setattr(hbmod.time, "time", lambda: now["t"])
    monkeypatch.setattr(ne, "hf_upload_file", _capture)
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 900.0)
    _reset_hb_state(monkeypatch, ne, last_upload=1000.0)

    ne.heartbeat("sft_step", step=5)  # real progress, throttled away (slot busy for 900s)
    assert uploads == []

    now["t"] = 1901.0  # slot open; a bare liveness ping wins the race
    ne.heartbeat("sft_step", liveness=True, step=5)
    assert uploads, "the ping must commit once the slot opens"
    assert uploads[-1].get("liveness") is None, (
        "the ping must be upgraded to a real heartbeat: it carries progress HF never saw"
    )

    uploads.clear()
    now["t"] = 2802.0  # next slot; no real heartbeat since the carried one
    ne.heartbeat("sft_step", liveness=True, step=5)
    assert uploads[-1].get("liveness") is True, (
        "once carried progress is committed, later pings must stay liveness — a wedged worker "
        "pinging alive must not mask a stall"
    )


def test_progress_carry_survives_failed_upload(monkeypatch):
    import json

    import flash.engine.worker as ne

    uploads: list[dict] = []
    outcome = {"ok": False}

    def _capture(local, *a, **k):
        with open(local) as f:
            uploads.append(json.load(f))
        return outcome["ok"]

    monkeypatch.setattr(ne, "hf_upload_file", _capture)
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 0.0)
    _reset_hb_state(monkeypatch, ne)

    ne.heartbeat("sft_step", step=9)  # real, upload FAILS -> progress still pending
    assert uploads[-1].get("liveness") is None

    outcome["ok"] = True
    ne.heartbeat("sft_step", liveness=True, step=9)
    assert uploads[-1].get("liveness") is None, (
        "a failed real upload keeps the latch pending; the next committed ping must carry it"
    )

    ne.heartbeat("sft_step", liveness=True, step=9)
    assert uploads[-1].get("liveness") is True, "settled after the successful carried commit"


def test_progress_carry_does_not_mark_new_progress(monkeypatch):
    """An upgraded ping carries OLD progress; it must not advance the worker's own stall-dump
    reference (_HB_LAST_PROGRESS_TS)."""
    import flash.engine.worker as ne

    monkeypatch.setattr(ne, "hf_upload_file", lambda *a, **k: False)  # keep the latch pending
    monkeypatch.setattr(ne, "_HB_MIN_INTERVAL_S", 0.0)
    _reset_hb_state(monkeypatch, ne)

    ne.heartbeat("sft_step", step=1)  # real; upload fails -> pending
    after_real = ne._HB_LAST_PROGRESS_TS
    assert after_real > 0
    ne.heartbeat("sft_step", liveness=True, step=1)  # upgraded ping
    assert after_real == ne._HB_LAST_PROGRESS_TS


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
    assert is_training_heartbeat("opd_step", 0) is False  # opd first-step in-progress ping (opt_steps==0)
    assert is_training_heartbeat("rl_step", 1) is True
    assert is_training_heartbeat("sft_step", 3) is True
    assert is_training_heartbeat("opd_step", 1) is True  # tightens once a real optimizer update lands
    # A malformed/missing step on a per-step stage is treated as 0 (must not raise) -> stays setup.
    assert is_training_heartbeat("rl_step", None) is False
    assert is_training_heartbeat("sft_step", "not-a-number") is False
    assert is_training_heartbeat("opd_step", None) is False
    # POST-training stages carry NO step but mean training is DONE -> tighten so a hung teardown/DONE
    # upload falls under the tight window, not the wide setup grace.
    assert is_training_heartbeat("rl_trained", None) is True
    assert is_training_heartbeat("sft_trained", None) is True
    assert is_training_heartbeat("rl_train_done", None) is True
    assert is_training_heartbeat("sft_train_done", None) is True
    # The finalize phases run after training -> tighten, same as sft_trained/rl_trained.
    assert is_training_heartbeat("sft_finalizing", None) is True
    assert is_training_heartbeat("rl_finalizing", None) is True


def test_setup_heartbeat_stages_cover_every_pre_training_liveness_stage():
    """The worker's progress-carry latch can upgrade any liveness ping to a REAL heartbeat. Every
    pre-training liveness stage must therefore be in SETUP_HEARTBEAT_STAGES, or a carried setup
    heartbeat would prematurely flip stall detection to the tight training window."""
    from flash.providers._poll import SETUP_HEARTBEAT_STAGES

    for stage in (
        "model_prefetching",
        "checkpoint_prefetching",
        "sft_data_loading",
        "rl_data_loading",
        "rl_adapter_loading",
        "sft_pretokenizing",
        "sft_initializing",
        "rl_initializing",
    ):
        assert stage in SETUP_HEARTBEAT_STAGES, stage
    # The post-training finalize phases are NOT setup: they must tighten, not re-buy setup grace.
    assert "sft_finalizing" not in SETUP_HEARTBEAT_STAGES
    assert "rl_finalizing" not in SETUP_HEARTBEAT_STAGES


def test_provider_surface_heartbeat_records_liveness_without_progress(monkeypatch):
    from flash.providers import _poll

    real = {"stage": "rl_initializing", "step": 0, "ts": 100.0, "attempt": "1"}
    key, stage = _poll.surface_heartbeat(lambda: real, None, lambda _m: None)
    assert key is not None
    assert stage == "rl_initializing"
    live = {"stage": "rl_initializing", "step": 0, "ts": 200.0, "attempt": "1", "liveness": True}
    recorded = []
    lines = []
    monkeypatch.setattr(_poll, "_record_heartbeat", recorded.append)
    key2, stage2 = _poll.surface_heartbeat(lambda: live, key, lines.append)
    assert key2 != key, "a liveness ping should advance the dedupe key"
    assert stage2 is None, "a liveness ping is not surfaced as progress"
    assert recorded == [live], "a liveness ping must still refresh visible run status"
    assert lines
    assert "liveness=true" in lines[-1]
    key3, stage3 = _poll.surface_heartbeat(lambda: live, key2, lines.append)
    assert key3 == key2
    assert stage3 is None
    assert recorded == [live], "duplicate liveness JSON must not be recorded repeatedly"


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
    assert f'liveness_heartbeat(\n            "{stage}",\n            progress=' in src, (
        f"{outer} must wrap trainer.train() in liveness_heartbeat({stage!r}, progress=...) — "
        "without the wrap the cold first step emits no real heartbeat and looks like a hang, and "
        "without progress= the daemon can win the throttled upload slot with a bare liveness ping "
        "and starve the provider's stall clock while training is healthy"
    )
    assert "progress_step=True" in src, (
        f"{outer} must stamp the trainer global step on daemon heartbeats (progress_step=True) so "
        "the poller's step gate and cancel billing see the true step"
    )
    assert "global_step" in src


def test_prefetch_wraps_download_in_liveness_heartbeat_gated_on_bytes():
    from flash.engine.worker import hf

    src = inspect.getsource(hf.prefetch_model)
    assert "liveness_heartbeat(" in src
    assert '"model_prefetching"' in src
    assert "_hf_cache_bytes(" in src, "prefetch must report downloaded-byte growth as real progress"


@pytest.mark.parametrize(
    ("modname", "outer", "stages"),
    [
        (
            "flash.engine.worker.sft",
            "run_sft",
            ("sft_data_loading", "sft_finalizing"),
        ),
        (
            "flash.engine.worker.rl",
            "run_rl",
            ("rl_data_loading", "rl_adapter_loading", "rl_finalizing"),
        ),
    ],
)
def test_quiet_phases_are_wrapped_in_liveness_heartbeat(modname, outer, stages):
    """Dataset load + template render, adapter warm-start, and the finalize save/upload each run
    for minutes with no other heartbeat; each must run under a liveness wrap."""
    mod = importlib.import_module(modname)
    src = inspect.getsource(getattr(mod, outer))
    for stage in stages:
        # Formatting-robust: the stage string may sit on the next line when the call wraps to fit the
        # line length (e.g. once keepalive=True is added to the finalize wrap).
        assert re.search(rf'liveness_heartbeat\(\s*"{re.escape(stage)}"', src), (
            f"{outer} must wrap the {stage} phase"
        )


def test_resume_checkpoint_download_is_wrapped_in_liveness_heartbeat():
    from flash.engine.worker import hf

    src = inspect.getsource(hf.hf_resume_checkpoint)
    assert 'liveness_heartbeat("checkpoint_prefetching")' in src, (
        "the multi-GB resume checkpoint download must keep the heartbeat fresh"
    )


@pytest.mark.parametrize(
    ("modname", "outer"),
    [("flash.engine.worker.sft", "run_sft"), ("flash.engine.worker.rl", "run_rl")],
)
def test_chalk_kernel_install_runs_inside_init_liveness_wrap(modname, outer):
    """install_chalk_kernels can JIT-compile for minutes right after trainer init; it must run
    INSIDE the *_initializing liveness wrap (checked structurally via the AST, not indentation)."""
    import ast
    import textwrap

    def _call_name(call):
        return getattr(call.func, "id", None) or getattr(call.func, "attr", None)

    mod = importlib.import_module(modname)
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(mod, outer))))
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not (isinstance(call, ast.Call) and _call_name(call) == "liveness_heartbeat"):
                continue
            stage = call.args[0].value if call.args and isinstance(call.args[0], ast.Constant) else ""
            if not str(stage).endswith("_initializing"):
                continue
            if any(
                isinstance(n, ast.Call) and _call_name(n) == "install_chalk_kernels"
                for n in ast.walk(node)
            ):
                return
    raise AssertionError(
        f"{outer}: install_chalk_kernels must run inside the *_initializing liveness wrap"
    )


def test_no_worker_side_stall_watchdog():
    """The worker has no separate stall watchdog: the provider owns kill+retry, and the dump fires on
    liveness give-up. Guard against re-adding the env-tunable faulthandler timer."""
    import importlib

    hb = importlib.import_module("flash.engine.worker.heartbeat")
    assert not hasattr(hb, "_rearm_stall_faulthandler")
    assert not hasattr(hb, "_STALL_WATCHDOG_S")


def test_bounded_reward_metrics_sanitizes_and_bounds_names() -> None:
    hb = importlib.import_module("flash.engine.worker.heartbeat")
    long_name = "x" * 100_000

    bounded = hb._bounded_reward_metrics(
        {
            long_name: 1.0,
            "line\nbreak": 2.0,
            "reward": 3.0,
            "step": 4.0,
        }
    )

    assert "x" * 64 in bounded
    assert all(len(name) <= 64 for name in bounded)
    assert "linebreak" in bounded
    assert all("\n" not in name for name in bounded)
    assert "reward" not in bounded
    assert "step" not in bounded


def test_reward_heartbeat_carries_bounded_finite_named_metrics(monkeypatch):
    hb = importlib.import_module("flash.engine.worker.heartbeat")
    worker = importlib.import_module("flash.engine.worker")
    emitted = []
    monkeypatch.setattr(worker, "heartbeat", lambda stage, **payload: emitted.append((stage, payload)))
    monkeypatch.setattr(hb, "_maybe_attach_gpu_diag", lambda payload, last, now: last)

    transformers = types.ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    metrics = {
        "nan_metric": float("nan"),
        "inf_metric": float("inf"),
        **{f"metric_{index:02d}": float(index) for index in reversed(range(14))},
    }
    callback = hb.make_reward_heartbeat_callback(lambda: metrics)
    state = types.SimpleNamespace(global_step=3)
    callback.on_log(None, state, None, logs={"reward": 0.65})

    assert emitted[0][0] == "rl_step"
    reward_metrics = emitted[0][1]["reward_metrics"]
    assert list(reward_metrics) == [f"metric_{index:02d}" for index in range(12)]
    assert all(math.isfinite(value) for value in reward_metrics.values())
    assert callback.latest_fields() == {"reward_metrics": reward_metrics}
