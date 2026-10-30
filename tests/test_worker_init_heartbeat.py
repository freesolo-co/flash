"""The init-phase heartbeat must never block on CUDA telemetry.

Regression for the consumer-GPU warm-start "hang": ``run_rl``/``run_sft`` start a daemon thread that
heartbeats ``rl_initializing``/``sft_initializing`` every 30s while the MAIN thread is blocked
inside ``GRPOTrainer.__init__`` / ``SFTTrainer.__init__`` (a long, CUDA- and allocator-busy section
-- vLLM colocate engine build + weight load + cold kernel JIT).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
import textwrap
import threading
import time
import types

import pytest

import flash.engine.worker.io.heartbeat as worker_heartbeat
import flash.engine.worker.io.hf as worker_hf
import flash.engine.worker.perf as worker_perf
import flash.engine.worker.train.core.lifecycle.finalize as worker_finalize
import flash.engine.worker.train.rl.launch.inputs as rl_inputs
import flash.runner.accounting.costs as runner_costs
import flash.runner.lifecycle.state as runner_state
from flash.engine.worker.perf import diagnostics


@pytest.fixture
def fast_nvidia(monkeypatch):
    """A fast, GPU-free nvidia-smi stand-in so the diagnostics path returns instantly
    regardless of whether nvidia-smi exists on the test host."""
    monkeypatch.setattr(
        diagnostics, "_query_nvidia_gpu", lambda: {"gpu_util_pct": 0, "device_name": "FAKE-GPU"}
    )
    monkeypatch.setattr(diagnostics, "_query_nvidia_processes", list)


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


def test_diagnostics_stay_responsive_while_cuda_locked(monkeypatch, fast_nvidia):
    """Fix: with the torch path off, the diag returns promptly even while CUDA is wedged."""
    gate = threading.Event()  # NEVER set -> CUDA stays "locked" for the whole test
    _install_blocking_torch(monkeypatch, gate)

    th, state = _run_async(lambda: diagnostics.gpu_diagnostics())
    th.join(timeout=2.0)

    assert state.get("done"), (
        "gpu_diagnostics() must NOT touch torch.cuda and must return even while "
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

    Returns (hb_module, diagnostic_calls)."""
    hb = importlib.import_module("flash.engine.worker.io.heartbeat")
    monkeypatch.setattr(hb, "_LIVENESS_TICK_S", tick)
    diag: list = []
    monkeypatch.setattr(
        worker_perf,
        "gpu_diagnostics",
        lambda: (diag.append(True), {})[1],
    )
    monkeypatch.setattr(hb, "_dump_thread_stacks", lambda reason: None)  # don't dump real stacks
    return hb, diag


def test_liveness_heartbeat_emits_liveness_pings_nvidia_smi_only(worker_context, monkeypatch):
    hb, diag = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(
        worker_heartbeat,
        "heartbeat",
        lambda s, context=None, **k: emitted.append(k.get("liveness")),
    )
    with hb.liveness_heartbeat("init_stage", context=worker_context):
        time.sleep(0.2)
    assert emitted, "must emit while alive"
    assert all(v is True for v in emitted), (
        "bare liveness_heartbeat emits LIVENESS pings (liveness=True)"
    )
    assert diag, "diagnostics collected"
    assert all(diag)


def test_liveness_heartbeat_reports_progress_advance_as_real_heartbeat(worker_context, monkeypatch):
    hb, _ = _liveness_env(monkeypatch)
    seen: list = []
    monkeypatch.setattr(
        worker_heartbeat,
        "heartbeat",
        lambda s, context=None, **k: seen.append(bool(k.get("liveness"))),
    )
    monkeypatch.setattr(worker_context.heartbeat_state, "last_progress_ts", time.time())
    vals = iter([1, 2, 2, 2, 2, 2, 2, 2])  # advances, then stalls
    with hb.liveness_heartbeat(
        "model_prefetching", progress=lambda: next(vals, 2), context=worker_context
    ):
        time.sleep(0.2)
    assert False in seen, "a progress advance must emit a REAL (non-liveness) heartbeat"
    assert True in seen, "no advance must emit a liveness ping"


def test_liveness_heartbeat_progress_step_stamps_step(worker_context, monkeypatch):
    """progress_step=True stamps the progress counter as ``step`` on every emit, so the poller's
    step gate and cancel billing see the true step even when the daemon wins the upload slot."""
    hb, _ = _liveness_env(monkeypatch)
    seen: list = []
    monkeypatch.setattr(
        worker_heartbeat,
        "heartbeat",
        lambda s, context=None, **k: seen.append((k.get("liveness"), k.get("step"))),
    )
    vals = iter([3, 7])  # advances once, then stalls at 7
    with hb.liveness_heartbeat(
        "sft_step", progress=lambda: next(vals, 7), progress_step=True, context=worker_context
    ):
        time.sleep(0.2)
    assert seen
    assert all(step in (3, 7) for _, step in seen), "every emit must stamp the last seen step"
    assert any(step == 7 and liveness for liveness, step in seen), (
        "liveness pings must carry the step too"
    )


def test_liveness_heartbeat_first_progress_sample_is_baseline_not_progress(
    worker_context, monkeypatch
):
    """A resumed run's restored global_step (or a constant finalize counter) must NOT count as
    progress on the daemon's first sample — it would emit a real step>=1 heartbeat seconds into
    train() and prematurely tighten the provider's stall window, defeating the per-attempt
    setup-grace re-arm. Only an ADVANCE past the first-seen value is progress."""
    hb, _ = _liveness_env(monkeypatch)
    seen: list = []
    monkeypatch.setattr(
        worker_heartbeat,
        "heartbeat",
        lambda s, context=None, **k: seen.append((k.get("liveness"), k.get("step"))),
    )
    with hb.liveness_heartbeat(
        "rl_step", progress=lambda: 57, progress_step=True, context=worker_context
    ):
        time.sleep(0.2)
    assert seen
    assert all(liveness is True for liveness, _ in seen), (
        "a never-advancing counter must emit only liveness pings"
    )
    assert all(step == 57 for _, step in seen), "pings still stamp the baseline step"


def test_liveness_heartbeat_keepalive_forces_real_heartbeats_on_constant_progress(
    worker_context, monkeypatch
):
    """keepalive=True: a synchronous checkpoint/finalize upload freezes global_step (a CONSTANT
    counter), which — per the test above — would emit ONLY liveness pings, and those do NOT advance
    the provider's stall clock (surface_heartbeat returns stage=None for them). A healthy upload that
    outlasts STALL_AFTER_S would then be killed mid-save. keepalive forces a REAL (liveness=False)
    heartbeat every tick, still stamped with the step so a cancel landing here still bills it."""
    hb, _ = _liveness_env(monkeypatch)
    seen: list = []
    monkeypatch.setattr(
        worker_heartbeat,
        "heartbeat",
        lambda s, context=None, **k: seen.append((k.get("liveness"), k.get("step"))),
    )
    with hb.liveness_heartbeat(
        "checkpoint_uploading",
        progress=lambda: 42,
        progress_step=True,
        keepalive=True,
        context=worker_context,
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
    import flash.engine.worker.io.heartbeat as ne
    from flash.providers._lifecycle.instances.poll import STALL_AFTER_S

    assert "checkpoint_uploading" in ne._HB_UPLOAD_LIVENESS_STAGES
    assert "checkpoint_uploading" in ne._HB_TIGHT_LIVENESS_STAGES
    assert "checkpoint_uploading" in ne._HB_THROTTLED_STAGES
    assert ne._HB_UPLOAD_LIVENESS_STAGES <= ne._HB_THROTTLED_STAGES
    # tight interval must leave margin under the training stall window (a single refresh must land).
    assert ne._HB_SETUP_LIVENESS_INTERVAL_S < STALL_AFTER_S


def test_liveness_heartbeat_survives_inline_thread_stub(worker_context, monkeypatch):
    """Several suites stub threading.Thread to run targets INLINE on .start() (to make the
    checkpoint-upload daemon synchronous). The liveness daemon must detect it is running on the
    spawning thread and bail, or the inlined loop spins forever and hangs the whole test run
    (bit test_resume_on_retry via the checkpoint_prefetching wrap in hf_resume_checkpoint)."""
    hb, _ = _liveness_env(monkeypatch)
    emitted: list = []
    monkeypatch.setattr(
        worker_heartbeat, "heartbeat", lambda s, context=None, **k: emitted.append(s)
    )

    class _SyncThread:
        def __init__(self, target=None, daemon=None, **kw):
            self._target = target

        def start(self):
            self._target()

        def join(self, timeout=None):
            return None

    monkeypatch.setattr(hb.threading, "Thread", _SyncThread)
    t0 = time.time()
    with hb.liveness_heartbeat("checkpoint_prefetching", context=worker_context):
        pass
    assert time.time() - t0 < 5, "inlined daemon must return immediately, not spin on done.wait"
    assert emitted == [], "an inlined daemon emits nothing"


def test_liveness_heartbeat_dumps_stacks_once_when_progress_stale(worker_context, monkeypatch):
    """No REAL progress for _STALL_DUMP_S -> dump every thread's stack ONCE (operator trace); the
    provider does the kill+retry off the same stale-progress signal."""
    hb, _ = _liveness_env(monkeypatch)
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda s, context=None, **k: None)
    monkeypatch.setattr(hb, "_STALL_DUMP_S", 0.05)
    monkeypatch.setattr(
        worker_context.heartbeat_state, "last_progress_ts", time.time() - 100
    )  # already stale
    dumped: list = []
    monkeypatch.setattr(hb, "_dump_thread_stacks", lambda reason: dumped.append(reason))
    with hb.liveness_heartbeat("rl_step", context=worker_context):
        time.sleep(0.2)
    assert len(dumped) == 1, f"must dump exactly once on a stall, got {len(dumped)}"


def test_liveness_heartbeat_join_is_bounded_even_if_emit_wedges(worker_context, monkeypatch):
    """The exit join must be BOUNDED: a wedged heartbeat() upload can't hang the worker at block exit."""
    hb, _ = _liveness_env(monkeypatch)
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda s, context=None, **k: time.sleep(30))
    monkeypatch.setattr(hb, "_HB_UPLOAD_LOCK_TIMEOUT_S", 0.2)
    t0 = time.time()
    with hb.liveness_heartbeat("init_stage", context=worker_context):
        time.sleep(0.1)
    assert time.time() - t0 < 5, (
        "exit must be bounded by the join timeout, not wait on a wedged emit"
    )


def test_liveness_heartbeat_rechecks_done_after_diagnostics():
    """gpu_diagnostics shells out to nvidia-smi (seconds); the wrapped call can finish during it. The
    daemon must re-check done BETWEEN diagnostics and the emit, so no stale stage lands afterward."""
    hb = importlib.import_module("flash.engine.worker.io.heartbeat")
    src = inspect.getsource(inspect.unwrap(hb.liveness_heartbeat))
    between = src[
        src.index("gpu_diagnostics()") : src.index("heartbeat(", src.index("gpu_diagnostics()"))
    ]
    assert "done.is_set()" in between, "must re-check done.is_set() between diagnostics and emit"


def test_heartbeat_publishes_canonical_progress_age(worker_context, monkeypatch):
    """real progress has age zero; liveness pings age the latest known progress."""
    import json

    hbmod = importlib.import_module("flash.engine.worker.io.heartbeat")
    now = {"t": 1000.0}
    seen: list[dict] = []
    monkeypatch.setattr(hbmod.time, "time", lambda: now["t"])
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_upload", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "progress_seq", 0)
    monkeypatch.setattr(worker_context.heartbeat_state, "progress_uploaded_seq", 0)

    def _capture(local, *a, context=None, **k):
        with open(local) as f:
            seen.append(json.load(f))

    monkeypatch.setattr(worker_hf, "hf_upload_file", _capture)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_progress_ts", 900.0)

    worker_heartbeat.heartbeat("rl_step", step=1, context=worker_context)
    assert seen[-1]["progress_age_s"] == 0.0
    assert seen[-1].get("liveness") is None
    assert worker_context.heartbeat_state.last_progress_ts == 1000.0

    now["t"] = 1012.3
    worker_heartbeat.heartbeat("rl_step", liveness=True, step=1, context=worker_context)
    assert seen[-1]["progress_age_s"] == 12.3
    assert seen[-1].get("liveness") is True
    assert worker_context.heartbeat_state.last_progress_ts == 1000.0

    now["t"] = 1045.6
    worker_heartbeat.heartbeat("rl_step", liveness=True, step=1, context=worker_context)
    assert seen[-1]["progress_age_s"] == 45.6
    assert seen[-1].get("liveness") is True
    assert worker_context.heartbeat_state.last_progress_ts == 1000.0


def test_heartbeat_omits_progress_age_before_first_progress(worker_context, monkeypatch):
    import json

    hbmod = importlib.import_module("flash.engine.worker.io.heartbeat")
    seen: list[dict] = []
    monkeypatch.setattr(hbmod.time, "time", lambda: 1000.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_upload", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_progress_ts", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "progress_seq", 0)
    monkeypatch.setattr(worker_context.heartbeat_state, "progress_uploaded_seq", 0)

    def _capture(local, *a, context=None, **k):
        with open(local) as f:
            seen.append(json.load(f))

    monkeypatch.setattr(worker_hf, "hf_upload_file", _capture)

    worker_heartbeat.heartbeat("rl_step", liveness=True, step=0, context=worker_context)

    assert "progress_age_s" not in seen[-1]
    assert worker_context.heartbeat_state.last_progress_ts == 0.0


def test_heartbeat_console_marks_commit_state_and_bounds_payload():
    import json

    from flash.engine.worker.io.heartbeat import _console_heartbeat_snapshot

    payload = {
        "stage": "rl_step",
        "step": 3,
        "metrics_last": [{}] * 8,
        "sampled_completions": ["private"] * 4,
    }
    committed = json.loads(_console_heartbeat_snapshot(payload))
    pending = json.loads(_console_heartbeat_snapshot(payload, False, True))
    throttled = json.loads(_console_heartbeat_snapshot(payload, False, False))

    for snapshot in (committed, pending, throttled):
        assert "metrics_last" not in snapshot
        assert "sampled_completions" not in snapshot
        assert snapshot["metrics_last_count"] == 8
        assert snapshot["samples_count"] == 4
    assert pending["pending"] is True
    assert throttled["throttled"] is True
    assert "pending" not in committed
    assert "throttled" not in committed


def test_concurrent_heartbeat_commits_once_and_skips_the_duplicate(worker_context, monkeypatch):
    """Two throttled heartbeats racing one HF commit produce exactly one upload.

    Previously the second thread proved this by BLOCKING on the upload lock and re-checking
    eligibility after the first commit landed. That wait is the contention this no longer pays: the
    in-flight marker makes the second heartbeat skip immediately. The outcome contract is unchanged
    and is what this asserts -- one upload attempt, one committed result, one skipped result, and
    the committed step recorded from the commit that actually ran.
    """

    first_started = threading.Event()
    release_first = threading.Event()
    attempts: list[int] = []
    results: list[bool] = []

    def upload(*_args, context=None, **_kwargs):
        attempts.append(1)
        first_started.set()
        assert release_first.wait(5.0)
        return True

    monkeypatch.setattr(worker_hf, "hf_upload_file", upload)
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_upload", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_committed_step", 0)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_forced_upload", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "progress_uploaded_seq", 0)

    threads = [
        threading.Thread(
            target=lambda step=step: results.append(
                worker_heartbeat.heartbeat("rl_step", step=step, context=worker_context)
            )
        )
        for step in (1, 2)
    ]
    threads[0].start()
    assert first_started.wait(5.0)
    threads[1].start()
    # the second thread must not need the first to finish: it skips instead of queueing.
    threads[1].join(timeout=5.0)
    assert not threads[1].is_alive(), "the second heartbeat blocked behind the in-flight upload"
    release_first.set()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert attempts == [1]
    assert sorted(results) == [False, True]
    assert worker_context.heartbeat_state.last_committed_step == 1


def test_heartbeat_console_summarizes_metric_backlog():
    import json

    from flash.engine.worker.io.heartbeat import _console_heartbeat_snapshot

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


def _last_console_heartbeat(capsys) -> dict:
    import json

    lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("HEARTBEAT {")
    ]
    assert lines
    return json.loads(lines[-1].removeprefix("HEARTBEAT "))


def _reset_console_heartbeat_state(monkeypatch, worker, *, worker_context) -> None:
    monkeypatch.setattr(worker_context.heartbeat_state, "last_upload", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_committed_step", -1)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_forced_upload", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_progress_ts", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "progress_seq", 0)
    monkeypatch.setattr(worker_context.heartbeat_state, "progress_uploaded_seq", 0)
    monkeypatch.setattr(worker_context.heartbeat_state, "pending_checkpoint_failure", None)


def _capture_heartbeat_payloads(monkeypatch, worker, *, worker_context) -> list[dict]:
    import json

    committed: list[dict] = []

    def capture(path, destination, context=None, **_kwargs):
        if destination == "heartbeat.json":
            with open(path) as file:
                committed.append(json.load(file))
        return True

    _reset_console_heartbeat_state(monkeypatch, worker, worker_context=worker_context)
    monkeypatch.setattr(worker_hf, "hf_upload_file", capture)
    return committed


def test_fatal_heartbeat_preserves_checkpoint_failure_and_terminal_error(
    worker_context, monkeypatch
):
    import flash.engine.worker.io.heartbeat as worker

    committed = _capture_heartbeat_payloads(monkeypatch, worker, worker_context=worker_context)
    failure = {"step": 50, "operation": "resume", "error": "quota denied"}

    worker_heartbeat.heartbeat(
        "checkpoint_upload_failed", step=50, checkpoint_failure=failure, context=worker_context
    )
    worker_heartbeat.heartbeat("error_sft", error="watcher failed", context=worker_context)

    assert committed[-1]["stage"] == "error_sft"
    assert committed[-1]["error"] == "watcher failed"
    assert committed[-1]["checkpoint_failure"] == failure


def test_finalize_preserves_failed_checkpoint_identity(worker_context, monkeypatch):
    import flash.engine.worker.io.heartbeat as worker
    from flash.engine.result.accounting import RunMetrics

    committed = _capture_heartbeat_payloads(monkeypatch, worker, worker_context=worker_context)
    monkeypatch.setattr(worker_perf, "gpu_diagnostics", dict)
    failure = {"step": 50, "operation": "resume", "error": "quota denied"}

    worker_heartbeat.heartbeat(
        "checkpoint_upload_failed", step=50, checkpoint_failure=failure, context=worker_context
    )
    worker_finalize._publish_training_completion(
        RunMetrics(phase="sft", step=100), context=worker_context
    )

    assert committed[-1]["stage"] == "done"
    assert committed[-1]["step"] == 100
    assert committed[-1]["checkpoint_failure"] == failure


@pytest.mark.parametrize("terminal_stage", ["done", "error_sft"])
def test_successful_checkpoint_clears_failure_before_any_terminal(
    worker_context, monkeypatch, terminal_stage
):
    import flash.engine.worker.io.heartbeat as worker
    from flash.engine.result.accounting import RunMetrics

    committed = _capture_heartbeat_payloads(monkeypatch, worker, worker_context=worker_context)
    failure = {"step": 50, "operation": "resume", "error": "quota denied"}

    worker_heartbeat.heartbeat(
        "checkpoint_upload_failed", step=50, checkpoint_failure=failure, context=worker_context
    )
    worker_heartbeat.heartbeat("checkpoint_uploaded", step=75, context=worker_context)
    if terminal_stage == "done":
        monkeypatch.setattr(worker_perf, "gpu_diagnostics", dict)
        worker_finalize._publish_training_completion(
            RunMetrics(phase="sft", step=100), context=worker_context
        )
    else:
        worker_heartbeat.heartbeat(
            terminal_stage, step=100, error="later fatal", context=worker_context
        )

    assert committed[-1]["stage"] == terminal_stage
    assert "checkpoint_failure" not in committed[-1]


@pytest.mark.parametrize(
    ("scenario", "upload_result", "expected_result", "marker"),
    [
        pytest.param("success", True, True, None, id="success"),
        pytest.param("upload-false", False, False, "pending", id="upload-false"),
        pytest.param("lock-skip", None, False, "pending", id="noninitial-lock-skip"),
        pytest.param("local-throttle", None, False, "throttled", id="local-throttle"),
    ],
)
def test_heartbeat_console_and_upload_marker_matrix(
    worker_context, monkeypatch, capsys, scenario, upload_result, expected_result, marker
):
    import json

    import flash.engine.worker.io.heartbeat as worker

    heartbeat_module = importlib.import_module("flash.engine.worker.io.heartbeat")
    uploaded: list[dict] = []
    samples = [{"completion": "visible only in the uploaded payload"}]

    def _upload(local, *args, context=None, **kwargs):
        with open(local) as handle:
            uploaded.append(json.load(handle))
        return upload_result

    _reset_console_heartbeat_state(monkeypatch, worker, worker_context=worker_context)
    held = False
    if scenario in {"success", "upload-false"}:
        monkeypatch.setattr(worker_hf, "hf_upload_file", _upload)
    elif scenario == "lock-skip":
        monkeypatch.setattr(heartbeat_module, "_HB_UPLOAD_LOCK_TIMEOUT_S", 0.01)
        monkeypatch.setattr(
            worker_hf,
            "hf_upload_file",
            lambda *args, context=None, **kwargs: pytest.fail("a skipped upload must not call hf"),
        )
        assert worker_context.heartbeat_state.upload_lock.acquire(timeout=1.0)
        held = True
    else:
        monkeypatch.setattr(heartbeat_module.time, "time", lambda: 1000.0)
        monkeypatch.setattr(worker_context.heartbeat_state, "last_upload", 999.0)
        monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 900.0)
        monkeypatch.setattr(
            worker_hf,
            "hf_upload_file",
            lambda *args, context=None, **kwargs: pytest.fail(
                "a throttled heartbeat must not call hf"
            ),
        )

    stage = "sft_step" if scenario == "local-throttle" else "rl_train_start"
    kwargs = {"step": 1} if scenario == "local-throttle" else {}
    try:
        result = worker_heartbeat.heartbeat(
            stage, sampled_completions=samples, **kwargs, context=worker_context
        )
    finally:
        if held:
            worker_context.heartbeat_state.upload_lock.release()

    assert result is expected_result
    console = _last_console_heartbeat(capsys)
    assert "sampled_completions" not in console
    assert console["samples_count"] == 1
    assert ("pending" in console) is (marker == "pending")
    assert ("throttled" in console) is (marker == "throttled")

    if uploaded:
        assert len(uploaded) == 1
        assert uploaded[0]["sampled_completions"] == samples
        assert "pending" not in uploaded[0]
        assert "throttled" not in uploaded[0]
    else:
        assert scenario in {"lock-skip", "local-throttle"}


def test_rl_lifecycle_heartbeats_carry_latest_metrics():
    import ast
    import textwrap

    from flash.engine.worker.train.entry import rl_train

    source = inspect.getsource(rl_train.run_rl_train) + inspect.getsource(
        rl_train._write_terminal_metadata
    )
    source += inspect.getsource(rl_train._publish_rl_result)
    tree = ast.parse(textwrap.dedent(source))
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
    # the value must be a COPY of the live accumulator, not the bare name: run_rl_train keeps appending
    # to metrics_last while heartbeats are in flight, and the heartbeat payload is serialized
    # asynchronously, so passing the list itself would let the snapshot mutate after it was taken.
    assert ast.unparse(terminal_keywords["metrics_last"]) == "list(metrics_last)"

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
        # same copy requirement, and `fields` must be a callable so the daemon re-reads the
        # accumulator on every tick rather than freezing whatever it held at context entry.
        assert "list(metrics_last)" in ast.unparse(keywords["fields"])
        assert isinstance(keywords["fields"], ast.Lambda)

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
    assert "list(metrics_last)" in ast.unparse(write_keywords["heartbeat_fields"])


def test_error_heartbeat_fallback_preserves_metric_backlog():
    # regression (#591): the error_{RUN_MODE} heartbeat exists to surface the bounded metric
    # backlog (metrics_last) for short failing RL runs. main() emits it twice -- a primary
    # call (with gpu diagnostics) and an except-fallback used if that primary call raises.
    import ast
    import textwrap

    import flash.engine.worker.entry.worker as ne

    tree = ast.parse(textwrap.dedent(inspect.getsource(ne.main)))
    error_hb_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "heartbeat_io"
        and node.func.attr == "heartbeat"
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
    import flash.engine.worker.io.heartbeat as ne

    assert "rl_step" in ne._HB_THROTTLED_STAGES
    assert "sft_step" in ne._HB_THROTTLED_STAGES, (
        "sft_step must be throttled like rl_step or the SFT liveness daemon blows the commit cap"
    )


def test_sft_step_liveness_upload_is_throttled(worker_context, monkeypatch):
    """A burst of sft_step liveness pings within _HB_MIN_INTERVAL_S commits only ONCE — the throttle
    (not just rl_step) covers sft_step, so a slow SFT run's 30s liveness ticks stay under the cap."""

    uploads: list = []
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 60.0)
    monkeypatch.setattr(
        worker_hf, "hf_upload_file", lambda local, *a, context=None, **k: uploads.append(local)
    )
    worker_context.heartbeat_state.last_upload = 0.0
    # First sft_step claims the slot; the next two (well within 60s) must be throttled out.
    for _ in range(3):
        worker_heartbeat.heartbeat("sft_step", liveness=True, step=0, context=worker_context)
    assert len(uploads) == 1, "sft_step uploads must be throttled to one per _HB_MIN_INTERVAL_S"


@pytest.mark.parametrize("stage", ["rl_step"])
def test_forced_sample_payload_commits_after_same_step_liveness(worker_context, monkeypatch, stage):
    """a liveness commit for a step must not throttle the first sample payload for that step."""

    uploads: list = []
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_FORCE_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(
        worker_hf, "hf_upload_file", lambda local, *a, context=None, **k: uploads.append(local)
    )
    worker_context.heartbeat_state.last_upload = 0.0
    worker_context.heartbeat_state.last_forced_upload = 0.0
    worker_context.heartbeat_state.last_committed_step = 0

    worker_heartbeat.heartbeat(stage, step=1, context=worker_context)
    assert len(uploads) == 1

    committed = worker_heartbeat.heartbeat(
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
        context=worker_context,
    )

    assert committed is True
    assert len(uploads) == 2


def test_initial_rl_step_persists_through_throttle_and_bills_cancel(worker_context, monkeypatch):
    import json

    persisted = []
    required_flags = []

    def upload(local, *args, context=None, **kwargs):
        required_flags.append(kwargs.get("required"))
        with open(local) as f:
            persisted.append(json.load(f))
        return True

    monkeypatch.setattr(worker_hf, "hf_upload_file", upload)
    worker_context.heartbeat_state.last_upload = time.time()
    worker_context.heartbeat_state.last_committed_step = 0

    committed = worker_heartbeat.heartbeat("rl_step", step=0, initial=True, context=worker_context)

    assert committed is True
    assert required_flags == [True]
    assert persisted[-1]["stage"] == "rl_step"
    assert persisted[-1]["step"] == 0
    status = runner_state.RunStatus(
        run_id="r",
        state="cancelled",
        spec={},
        last_heartbeat=persisted[-1],
    )
    assert runner_costs.actual_steps_run(status) == 1


def test_initial_rl_step_lock_timeout_is_retriable(worker_context, monkeypatch):
    hb = importlib.import_module("flash.engine.worker.io.heartbeat")

    monkeypatch.setattr(hb, "_HB_UPLOAD_LOCK_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        worker_hf,
        "hf_upload_file",
        lambda *args, context=None, **kwargs: pytest.fail(
            "lock timeout must not attempt an upload"
        ),
    )
    worker_context.heartbeat_state.last_upload = 17.0
    worker_context.heartbeat_state.last_committed_step = 4
    worker_context.heartbeat_state.last_forced_upload = 9.0

    assert worker_context.heartbeat_state.upload_lock.acquire(timeout=1.0)
    try:
        with pytest.raises(worker_perf.RetriableInfraError, match="initial heartbeat upload lock"):
            worker_heartbeat.heartbeat("rl_step", step=0, initial=True, context=worker_context)
        assert worker_context.heartbeat_state.last_upload == 17.0
        assert worker_context.heartbeat_state.last_committed_step == 4
        assert worker_context.heartbeat_state.last_forced_upload == 9.0

        assert worker_heartbeat.heartbeat("rl_step", step=5, context=worker_context) is False
        assert worker_context.heartbeat_state.last_upload == 17.0
        assert worker_context.heartbeat_state.last_committed_step == 4
        assert worker_context.heartbeat_state.last_forced_upload == 9.0
    finally:
        worker_context.heartbeat_state.upload_lock.release()


def test_force_commit_via_regular_throttle_arms_the_floor(worker_context, monkeypatch):
    """Regression (heartbeat.py): a force=True heartbeat that commits because the regular
    throttle was ALREADY due (not because the force branch bypassed it) must STILL arm the forced-commit
    clock — else the clock stays stale and a following sub-floor forced ping punches through, defeating
    the burst coalescing that protects the HF commit cap."""

    uploads: list = []
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(worker_heartbeat, "_HB_FORCE_MIN_INTERVAL_S", 60.0)
    monkeypatch.setattr(
        worker_hf, "hf_upload_file", lambda local, *a, context=None, **k: uploads.append(local)
    )
    worker_context.heartbeat_state.last_upload = (
        0.0  # regular throttle is DUE -> the first force commits via it, not the bypass
    )
    worker_context.heartbeat_state.last_forced_upload = 0.0
    worker_context.heartbeat_state.last_committed_step = 0
    worker_heartbeat.heartbeat(
        "rl_step", step=1, loss=0.1, force=True, context=worker_context
    )  # commits via the elapsed regular throttle
    assert len(uploads) == 1
    worker_heartbeat.heartbeat(
        "rl_step", step=2, loss=0.1, force=True, context=worker_context
    )  # sub-floor advance -> must be COALESCED now
    assert len(uploads) == 1, (
        "the regular-path force commit must arm the floor so the next is coalesced"
    )
    assert worker_context.heartbeat_state.last_committed_step == 1


def test_setup_liveness_upload_uses_shorter_interval(worker_context, monkeypatch):
    """Setup liveness must refresh public status before a 300s external frozen-heartbeat watchdog,
    while training-step liveness stays under the normal shared-repo throttle."""
    import json

    import flash.engine.worker.io.heartbeat as ne

    hbmod = importlib.import_module("flash.engine.worker.io.heartbeat")

    now = {"t": 1000.0}
    uploads: list[dict] = []

    def _capture(local, *a, context=None, **k):
        with open(local) as f:
            uploads.append(json.load(f))

    monkeypatch.setattr(hbmod.time, "time", lambda: now["t"])
    monkeypatch.setattr(worker_hf, "hf_upload_file", _capture)
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 900.0)
    monkeypatch.setattr(ne, "_HB_SETUP_LIVENESS_INTERVAL_S", 240.0)

    worker_context.heartbeat_state.last_upload = 1000.0
    now["t"] = 1239.0
    worker_heartbeat.heartbeat("sft_initializing", liveness=True, context=worker_context)
    assert uploads == []

    now["t"] = 1241.0
    worker_heartbeat.heartbeat("sft_initializing", liveness=True, context=worker_context)
    assert uploads[-1]["stage"] == "sft_initializing"
    assert uploads[-1]["liveness"] is True

    uploads.clear()
    worker_context.heartbeat_state.last_upload = 1000.0
    now["t"] = 1241.0
    worker_heartbeat.heartbeat("sft_step", liveness=True, step=0, context=worker_context)
    assert uploads == [], "training-step liveness must stay on _HB_MIN_INTERVAL_S"


def test_default_heartbeat_interval_fits_shared_environment_repos():
    import flash.engine.worker.io.heartbeat as ne

    assert worker_heartbeat._HB_MIN_INTERVAL_S >= 900.0
    assert worker_heartbeat._HB_MIN_INTERVAL_S < 1200.0
    assert 180.0 <= ne._HB_SETUP_LIVENESS_INTERVAL_S < 300.0


def test_every_setup_liveness_stage_is_throttled():
    """The liveness daemon re-emits its stage every 30s; a setup-liveness stage missing from
    _HB_THROTTLED_STAGES would commit unthrottled (~120/hr) and blow the shared repo commit cap."""
    import flash.engine.worker.io.heartbeat as ne

    assert ne._HB_SETUP_LIVENESS_STAGES <= ne._HB_THROTTLED_STAGES


def test_quiet_phases_upload_at_setup_liveness_interval():
    """Every liveness-wrapped quiet phase must refresh public status at the faster setup cadence,
    or run status looks frozen for up to 15 min during a healthy phase."""
    import flash.engine.worker.io.heartbeat as ne

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
def _reset_hb_state(monkeypatch, ne, *, last_upload=0.0, worker_context):
    # monkeypatch (not direct assignment) so the latch state is restored after each test — a
    # leaked pending latch would make unrelated later tests order-dependent.
    monkeypatch.setattr(worker_context.heartbeat_state, "last_upload", last_upload)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_progress_ts", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "progress_seq", 0)
    monkeypatch.setattr(worker_context.heartbeat_state, "progress_uploaded_seq", 0)


def test_progress_carry_upgrades_ping_after_throttled_real_heartbeat(worker_context, monkeypatch):
    import json

    import flash.engine.worker.io.heartbeat as ne

    hbmod = importlib.import_module("flash.engine.worker.io.heartbeat")
    now = {"t": 1000.0}
    uploads: list[dict] = []

    def _capture(local, *a, context=None, **k):
        with open(local) as f:
            uploads.append(json.load(f))

    monkeypatch.setattr(hbmod.time, "time", lambda: now["t"])
    monkeypatch.setattr(worker_hf, "hf_upload_file", _capture)
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 900.0)
    _reset_hb_state(monkeypatch, ne, last_upload=1000.0, worker_context=worker_context)

    worker_heartbeat.heartbeat(
        "sft_step", step=5, context=worker_context
    )  # real progress, throttled away (slot busy for 900s)
    assert uploads == []

    now["t"] = 1901.0  # slot open; a bare liveness ping wins the race
    worker_heartbeat.heartbeat("sft_step", liveness=True, step=5, context=worker_context)
    assert uploads, "the ping must commit once the slot opens"
    assert uploads[-1].get("liveness") is None, (
        "the ping must be upgraded to a real heartbeat: it carries progress HF never saw"
    )
    assert uploads[-1]["progress_age_s"] == 901.0, (
        "carried progress keeps the age of the original real heartbeat; the upgrade is not new progress"
    )

    # feed the real carried payload through the cli path. progress predates its ts, so ten seconds
    # after upload the conservative age bound is 10 + 901, not the upload age alone.
    now["t"] = 1911.0
    from flash.cli.ui.heartbeat import _heartbeat_pairs

    pairs = _heartbeat_pairs({"state": "running", "last_heartbeat": uploads[-1]})
    progress = dict(pairs)["progress"]
    assert "last known progress can be as old as 911.0s" in progress
    assert "the upload is 10.0s old versus" not in progress

    uploads.clear()
    now["t"] = 2802.0  # next slot; no real heartbeat since the carried one
    worker_heartbeat.heartbeat("sft_step", liveness=True, step=5, context=worker_context)
    assert uploads[-1].get("liveness") is True, (
        "once carried progress is committed, later pings must stay liveness — a wedged worker "
        "pinging alive must not mask a stall"
    )


def test_progress_carry_survives_failed_upload(worker_context, monkeypatch):
    import json

    import flash.engine.worker.io.heartbeat as ne

    hbmod = importlib.import_module("flash.engine.worker.io.heartbeat")
    now = {"t": 1000.0}
    uploads: list[dict] = []
    outcome = {"ok": False}

    def _capture(local, *a, context=None, **k):
        with open(local) as f:
            uploads.append(json.load(f))
        return outcome["ok"]

    monkeypatch.setattr(hbmod.time, "time", lambda: now["t"])
    monkeypatch.setattr(worker_hf, "hf_upload_file", _capture)
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    _reset_hb_state(monkeypatch, ne, worker_context=worker_context)

    worker_heartbeat.heartbeat(
        "sft_step", step=9, context=worker_context
    )  # real, upload fails, so progress remains pending
    assert uploads[-1].get("liveness") is None
    assert uploads[-1]["progress_age_s"] == 0.0

    outcome["ok"] = True
    now["t"] = 1015.0
    worker_heartbeat.heartbeat("sft_step", liveness=True, step=9, context=worker_context)
    assert uploads[-1].get("liveness") is None, (
        "a failed real upload keeps the latch pending; the next committed ping must carry it"
    )
    assert uploads[-1]["progress_age_s"] == 15.0

    now["t"] = 1020.0
    worker_heartbeat.heartbeat("sft_step", liveness=True, step=9, context=worker_context)
    assert uploads[-1].get("liveness") is True, "settled after the successful carried commit"
    assert uploads[-1]["progress_age_s"] == 20.0


def test_progress_carry_does_not_mark_new_progress(worker_context, monkeypatch):
    """An upgraded ping carries OLD progress; it must not advance the worker's own stall-dump
    reference (_HB_LAST_PROGRESS_TS)."""
    import flash.engine.worker.io.heartbeat as ne

    monkeypatch.setattr(
        worker_hf, "hf_upload_file", lambda *a, context=None, **k: False
    )  # keep the latch pending
    monkeypatch.setattr(worker_heartbeat, "_HB_MIN_INTERVAL_S", 0.0)
    _reset_hb_state(monkeypatch, ne, worker_context=worker_context)

    worker_heartbeat.heartbeat(
        "sft_step", step=1, context=worker_context
    )  # real; upload fails -> pending
    after_real = worker_context.heartbeat_state.last_progress_ts
    assert after_real > 0
    worker_heartbeat.heartbeat(
        "sft_step", liveness=True, step=1, context=worker_context
    )  # upgraded ping
    assert after_real == worker_context.heartbeat_state.last_progress_ts


# --------------------------------------------------------------------------------------------
# _hf_cache_bytes feeds the prefetch progress signal: bytes downloaded, or None when the cache dir
# doesn't exist yet (unmeasurable) — a growth is reported by liveness_heartbeat as REAL progress.
def test_hf_cache_bytes_counts_blobs_and_reports_unmeasurable_as_none(tmp_path, monkeypatch):
    import huggingface_hub.constants as hconst

    from flash.engine.worker.io import hf

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
    from flash.providers._lifecycle.instances.poll import is_training_heartbeat

    # Setup stages (and a missing stage) never tighten — still the cold start.
    assert is_training_heartbeat("rl_train_start", None) is False
    assert is_training_heartbeat("sft_initializing", 5) is False
    assert is_training_heartbeat(None, 9) is False
    # The per-step training stages tighten ONLY at a COMPLETED step (>= 1); a step=0 gap-fill during
    # the silent cold first step keeps setup grace.
    assert is_training_heartbeat("rl_step", 0) is False
    assert is_training_heartbeat("sft_step", 0) is False
    assert (
        is_training_heartbeat("rl_step", 0) is False
    )  # first-step in-progress ping (opt_steps==0)
    assert is_training_heartbeat("rl_step", 1) is True
    assert is_training_heartbeat("sft_step", 3) is True
    assert (
        is_training_heartbeat("rl_step", 1) is True
    )  # tightens once a real optimizer update lands
    # A malformed/missing step on a per-step stage is treated as 0 (must not raise) -> stays setup.
    assert is_training_heartbeat("rl_step", None) is False
    assert is_training_heartbeat("sft_step", "not-a-number") is False
    assert is_training_heartbeat("rl_step", None) is False
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
    from flash.providers._lifecycle.instances.poll import SETUP_HEARTBEAT_STAGES

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
    from flash.providers._lifecycle.instances import poll as _poll

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
# Wiring: each long blocking phase runs under the shared liveness_heartbeat helper (behaviour
# covered above; these pin the call sites so coverage can't silently regress). rl_initializing is
# intentionally absent, for the same reason SFT is absent from the warmup test below: verl builds
# its trainer in a subprocess, so there is no in-process build window to wrap.
@pytest.mark.parametrize(
    ("modname", "outer", "stage"),
    [
        ("flash.engine.worker.train.entry.rl_train", "run_rl_train", "rl_step"),
        ("flash.engine.worker.train.entry.sft_train_runner", "_run_sft_child", "sft_step"),
    ],
)
def test_train_phase_wraps_train_in_liveness_heartbeat(modname, outer, stage):
    mod = importlib.import_module(modname)
    src = inspect.getsource(getattr(mod, outer))
    # formatting-robust: the wrap sits at different nesting depths across backends, so match the
    # call shape rather than a fixed indent (the sibling quiet-phase test does the same).
    assert re.search(rf'liveness_heartbeat\(\s*"{re.escape(stage)}",\s*progress=', src), (
        f"{outer} must wrap trainer.train() in liveness_heartbeat({stage!r}, progress=...) — "
        "without the wrap the cold first step emits no real heartbeat and looks like a hang, and "
        "without progress= the daemon can win the throttled upload slot with a bare liveness ping "
        "and starve the provider's stall clock while training is healthy"
    )
    assert "progress_step=True" in src, (
        f"{outer} must stamp the trainer global step on daemon heartbeats (progress_step=True) so "
        "the poller's step gate and cancel billing see the true step"
    )
    # the step has to come from the trainer's own counter, not a local tick. trl exposed it in-process
    # as global_step; verl trains in a subprocess and parses it back out into progress["step"].
    assert any(
        value in src for value in ("global_step", 'progress["step"]', 'progress.values["step"]')
    )


def test_prefetch_wraps_download_in_liveness_heartbeat_gated_on_bytes():
    from flash.engine.worker.io import hf

    src = inspect.getsource(hf.prefetch_model)
    assert "liveness_heartbeat(" in src
    assert '"model_prefetching"' in src
    assert "_hf_cache_bytes(" in src, "prefetch must report downloaded-byte growth as real progress"


@pytest.mark.parametrize(
    ("modname", "outer", "stages"),
    [
        (
            "flash.engine.worker.train.entry.sft_train",
            "run_sft_train",
            ("sft_data_loading", "sft_finalizing"),
        ),
        (
            "flash.engine.worker.train.entry.rl_train",
            "run_rl_train",
            ("rl_data_loading", "rl_configuring", "rl_finalizing"),
        ),
    ],
)
def test_quiet_phases_are_wrapped_in_liveness_heartbeat(modname, outer, stages):
    """Dataset load + template render, adapter warm-start, and the finalize save/upload each run
    for minutes with no other heartbeat; each must run under a liveness wrap."""
    mod = importlib.import_module(modname)
    src = inspect.getsource(getattr(mod, outer))
    publish_phase = "_publish_sft_result" if outer == "run_sft_train" else "_publish_rl_result"
    assert publish_phase + "(" in src
    src += inspect.getsource(getattr(mod, publish_phase))
    for stage in stages:
        # Formatting-robust: the stage string may sit on the next line when the call wraps to fit the
        # line length (e.g. once keepalive=True is added to the finalize wrap).
        assert re.search(rf'liveness_heartbeat\(\s*"{re.escape(stage)}"', src), (
            f"{outer} must wrap the {stage} phase"
        )


@pytest.mark.parametrize(
    ("modname", "outer", "stage"),
    [
        ("flash.engine.worker.train.entry.sft_train", "run_sft_train", "sft_configuring"),
        ("flash.engine.worker.train.entry.rl_train", "run_rl_train", "rl_configuring"),
    ],
)
def test_venv_provisioning_and_the_capability_probe_run_under_one_wrap(modname, outer, stage):
    """All THREE trainers must wrap the verl-interpreter setup, not just sft.

    That is minutes of silence with no training step to report and no liveness thread otherwise
    running -- long enough for the stall watchdog to fail a healthy run on a paid GPU. sft
    have wrapped it since dev #442; grpo called it bare, which is the gap this pins shut.
    """
    mod = importlib.import_module(modname)
    src = inspect.getsource(getattr(mod, outer))
    tree = ast.parse(textwrap.dedent(src))
    wraps = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "liveness_heartbeat"
            and item.context_expr.args
            and isinstance(item.context_expr.args[0], ast.Constant)
            and item.context_expr.args[0].value == stage
            for item in node.items
        )
    ]
    assert len(wraps) == 1, (
        f"{outer} must wrap its cold verl setup in liveness_heartbeat({stage!r})"
    )
    called = {
        node.func.id
        for node in ast.walk(wraps[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "resolve_verl_python" in called, (
        f"{outer}: the venv build is the silent span -- it must be INSIDE the {stage} wrap"
    )
    assert "probe_verl_capabilities" in called, (
        f"{outer}: the child probe pays a cold torch import and must share the {stage} wrap"
    )


def test_rl_warmstart_adapter_download_is_wrapped_in_liveness_heartbeat():
    """The warm-start adapter pull is multi-GB and lives in the input resolver rather than the
    entry point, so it carries its own wrap; the sibling test above cannot see it."""

    src = inspect.getsource(rl_inputs._resolve_grpo_inputs)
    assert 'liveness_heartbeat("rl_adapter_loading", context=context)' in src, (
        "the multi-GB warm-start adapter download must keep the heartbeat fresh"
    )


def test_sft_configuring_is_a_setup_stage_on_the_tight_liveness_cadence():
    """The config span's stage must behave like every other pre-training liveness stage: it keeps the
    wide setup grace (it has not even loaded the model yet), refreshes status on the faster setup
    cadence, and is throttled so its 30s re-emit can't blow the HF commit cap."""
    import flash.engine.worker.io.heartbeat as ne
    from flash.providers._lifecycle.instances.poll import (
        SETUP_HEARTBEAT_STAGES,
        is_training_heartbeat,
    )

    assert "sft_configuring" in SETUP_HEARTBEAT_STAGES
    assert is_training_heartbeat("sft_configuring", 0) is False
    assert "sft_configuring" in ne._HB_SETUP_LIVENESS_STAGES
    assert "sft_configuring" in ne._HB_THROTTLED_STAGES


def test_resume_checkpoint_download_is_wrapped_in_liveness_heartbeat():
    from flash.engine.worker.io import hf

    src = inspect.getsource(hf.hf_resume_checkpoint)
    assert 'liveness_heartbeat("checkpoint_prefetching", context=context)' in src, (
        "the multi-GB resume checkpoint download must keep the heartbeat fresh"
    )


def test_post_download_model_setup_runs_under_a_liveness_wrap():
    """Post-download SFT setup must continue reporting liveness during cold-cache reads."""
    from flash.engine.worker.train.entry import sft_train_runner

    sft_src = inspect.getsource(sft_train_runner._prepare_sft_model)
    assert re.search(r'liveness_heartbeat\(\s*"sft_model_load"', sft_src), (
        "the post-download sft model setup must keep the heartbeat fresh"
    )
    # the reads that actually pay the cold-cache cost have to be INSIDE the wrap, not beside it.
    tree = ast.parse(textwrap.dedent(sft_src))
    wraps = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "liveness_heartbeat"
            and item.context_expr.args
            and isinstance(item.context_expr.args[0], ast.Constant)
            and item.context_expr.args[0].value == "sft_model_load"
            for item in node.items
        )
    ]
    assert len(wraps) == 1
    wrapped = textwrap.dedent(ast.unparse(wraps[0]))
    for call in ("make_lora", "_warmstart_adapter_path", "_resolve_sft_vocab_size"):
        assert call in wrapped, f"{call} is part of the silent span and must share the wrap"


@pytest.mark.parametrize("stage", ["sft_model_load"])
def test_model_load_is_a_throttled_setup_stage(stage):
    """Now that these hold a liveness thread, they re-emit every 30s -- so they must be throttled or
    a slow cold mount spends the HF commit budget on them, and must keep the WIDE setup grace since
    no training has started."""
    import flash.engine.worker.io.heartbeat as ne
    from flash.providers._lifecycle.instances.poll import (
        SETUP_HEARTBEAT_STAGES,
        is_training_heartbeat,
    )

    assert stage in SETUP_HEARTBEAT_STAGES
    assert is_training_heartbeat(stage, 0) is False
    assert stage in ne._HB_SETUP_LIVENESS_STAGES
    assert stage in ne._HB_THROTTLED_STAGES


@pytest.mark.parametrize("stage", ["sft_model_load"])
def test_model_load_transition_commits_even_behind_a_fresh_setup_ping(
    worker_context, stage, monkeypatch
):
    """The one-shot transition must land; only the liveness ticks after it may be coalesced.

    Throttling the stage as a whole drops the transition whenever a setup ping committed inside the
    240s interval -- and the ping right before it is `model_prefetching`, which pings all through the
    download. So the common case loses the only heartbeat that says the run reached this stage, which
    is precisely what adding the stage was meant to make visible. Membership assertions cannot catch
    this; it has to be exercised through `heartbeat()`.
    """
    import json

    import flash.engine.worker.io.heartbeat as hb_mod

    committed: list[str] = []

    def _fake_upload(local, remote, required=False, context=None, **_context):
        with open(local) as f:
            committed.append(json.load(f)["stage"])
        return True

    monkeypatch.setattr(worker_hf, "hf_upload_file", _fake_upload)
    monkeypatch.setattr(worker_perf, "gpu_diagnostics", lambda **k: {})
    monkeypatch.setattr(worker_context.heartbeat_state, "last_upload", 0.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_committed_step", -1)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_forced_upload", 0.0)

    # the download's own liveness ping commits first and arms the throttle window.
    hb_mod.heartbeat("model_prefetching", liveness=True, context=worker_context)
    hb_mod.heartbeat(stage, download_seconds=12.0, context=worker_context)
    assert stage in committed, "the stage transition was swallowed by the liveness throttle"

    # and the wrap that follows must still be coalesced -- that is what the throttle is for.
    before = len(committed)
    for _ in range(6):
        hb_mod.heartbeat(stage, liveness=True, context=worker_context)
    assert len(committed) == before, "liveness ticks must stay throttled"


def test_status_panel_knows_every_setup_liveness_stage():
    """The panel's hint set must track the worker's, or a new stage silently loses its diagnosis.

    `_stale_setup_hint` fires only for stages it lists. The worker owns the real set, so a stage
    added there and not here would fall back to the generic "quiet is not dead" reassurance at
    exactly the ages this PR exists to stop reassuring at -- with nothing failing to say so.
    """
    import flash.engine.worker.io.heartbeat as ne
    from flash.cli.ui.heartbeat import _LIVENESS_SETUP_STAGES

    assert _LIVENESS_SETUP_STAGES == ne._HB_SETUP_LIVENESS_STAGES, (
        "the status panel's liveness-setup stages drifted from the worker's"
    )


def test_no_worker_side_stall_watchdog():
    """The worker has no separate stall watchdog: the provider owns kill+retry, and the dump fires on
    liveness give-up. Guard against re-adding the env-tunable faulthandler timer."""
    import importlib

    hb = importlib.import_module("flash.engine.worker.io.heartbeat")
    assert not hasattr(hb, "_rearm_stall_faulthandler")
    assert not hasattr(hb, "_STALL_WATCHDOG_S")


def test_bounded_reward_metrics_sanitizes_and_bounds_names() -> None:
    hb = importlib.import_module("flash.engine.worker.io.heartbeat")
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


def test_throttled_heartbeat_does_not_block_behind_an_in_flight_upload(worker_context, monkeypatch):
    """A slow HF commit must not stall every other throttled heartbeat on the upload lock.

    The throttle clock only advances AFTER a commit lands, so while one is in flight every other
    throttled caller still computes ``upload_due`` and blocks in ``_HB_UPLOAD_LOCK.acquire()`` for
    up to the acquire timeout (30s, or 120s on a critical stage). That window stalls the liveness
    daemon, and ``liveness_heartbeat``'s join delays leaving the wrapped stage by the same amount.
    """
    hb = importlib.import_module("flash.engine.worker.io.heartbeat")

    monkeypatch.setattr(hb, "_HB_UPLOAD_LOCK_TIMEOUT_S", 5.0)
    monkeypatch.setattr(worker_context.heartbeat_state, "terminal_only", False)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_upload", 0.0)  # never uploaded -> due

    upload_started = threading.Event()
    release_upload = threading.Event()

    def slow_upload(*args, context=None, **kwargs):
        upload_started.set()
        release_upload.wait(10)
        return True

    monkeypatch.setattr(worker_hf, "hf_upload_file", slow_upload)

    committer = threading.Thread(
        target=lambda: worker_heartbeat.heartbeat("rl_step", step=1, context=worker_context),
        daemon=True,
    )
    committer.start()
    try:
        assert upload_started.wait(5), "the first heartbeat never reached its upload"

        t0 = time.time()
        assert worker_heartbeat.heartbeat("rl_step", liveness=True, context=worker_context) is False
        blocked = time.time() - t0
        assert blocked < 1.0, (
            f"a throttled heartbeat waited {blocked:.2f}s behind an in-flight upload; it must skip "
            "rather than queue on the upload lock"
        )
    finally:
        release_upload.set()
        committer.join(10)

    # the marker is transient: once the commit lands, later heartbeats are eligible again.
    assert worker_context.heartbeat_state.upload_in_flight is False


def test_terminal_heartbeat_still_queues_behind_an_in_flight_upload(worker_context, monkeypatch):
    """Skipping is only for throttled pings. A terminal/error snapshot has no later heartbeat to
    repair it, so it must still wait for the lock and commit."""
    hb = importlib.import_module("flash.engine.worker.io.heartbeat")

    monkeypatch.setattr(worker_context.heartbeat_state, "terminal_only", False)
    with worker_context.heartbeat_state.lock:
        hb._set_upload_in_flight(True, context=worker_context)
    try:
        with worker_context.heartbeat_state.lock:
            assert (
                hb._heartbeat_upload_due(
                    "done",
                    liveness=False,
                    force=False,
                    initial=False,
                    first_timing=False,
                    fields={},
                    now=time.time(),
                    context=worker_context,
                )
                is True
            ), "a terminal stage must not be skipped by the in-flight marker"
            assert (
                hb._heartbeat_upload_due(
                    "rl_step",
                    liveness=False,
                    force=False,
                    initial=True,
                    first_timing=False,
                    fields={},
                    now=time.time(),
                    context=worker_context,
                )
                is True
            ), "the initial heartbeat must not be skipped by the in-flight marker"
    finally:
        with worker_context.heartbeat_state.lock:
            hb._set_upload_in_flight(False, context=worker_context)


def test_failed_upload_clears_the_in_flight_marker(worker_context, monkeypatch):
    """A raising upload must not strand the marker, which would skip every later heartbeat."""
    hb = importlib.import_module("flash.engine.worker.io.heartbeat")

    monkeypatch.setattr(worker_context.heartbeat_state, "terminal_only", False)
    monkeypatch.setattr(worker_context.heartbeat_state, "last_upload", 0.0)

    def boom(*args, context=None, **kwargs):
        raise RuntimeError("hf unreachable")

    monkeypatch.setattr(worker_hf, "hf_upload_file", boom)
    with pytest.raises(RuntimeError, match="hf unreachable"):
        worker_heartbeat.heartbeat("rl_step", step=1, context=worker_context)

    assert worker_context.heartbeat_state.upload_in_flight is False
    with worker_context.heartbeat_state.lock:
        assert (
            hb._heartbeat_upload_due(
                "rl_step",
                liveness=False,
                force=False,
                initial=False,
                first_timing=False,
                fields={},
                now=time.time(),
                context=worker_context,
            )
            is True
        ), "a failed upload must leave later heartbeats eligible"
