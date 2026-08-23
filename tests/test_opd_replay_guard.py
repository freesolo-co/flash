from __future__ import annotations

import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.opd.child import bridge, replay_guard


class _Buffer:
    def __init__(self, sample):
        self.lock = threading.Lock()
        self.partitions = defaultdict(dict)
        self.poll_interval = 0.001
        self.sample = sample


class _Upstream:
    def __init__(self):
        self.calls = 0
        self.release = threading.Event()
        self.value = object()
        self.error: BaseException | None = None

    def __call__(self, *, partition_id, global_steps=None, batch_size=None):
        self.calls += 1
        self.release.wait(timeout=2)
        if self.error is not None:
            raise self.error
        return self.value


def _trainer(buffer):
    return SimpleNamespace(
        replay_buffer=buffer,
        async_rollout_manager=SimpleNamespace(agent_loop_workers=[]),
    )


def _run_sample(buffer, state):
    try:
        state["value"] = buffer.sample(partition_id="train", global_steps=7)
    except BaseException as error:
        state["error"] = error
    finally:
        state["done"].set()


@pytest.fixture(autouse=True)
def _fast_guard(monkeypatch):
    monkeypatch.setattr(replay_guard, "_REPLAY_POLL_SECONDS", 0.002)
    monkeypatch.setattr(replay_guard, "_REPLAY_STALL_SECONDS", 0.08)
    monkeypatch.setattr(replay_guard, "_actor_probe_state", lambda _trainer: "unknown")


def test_failed_prompt_beats_still_running_siblings_and_upstream_result():
    upstream = _Upstream()
    buffer = _Buffer(upstream)
    buffer.partitions["train"] = {
        "failed": {"global_steps": 7, "status": "failure"},
        "sibling": {"global_steps": 7, "status": "running"},
    }
    replay_guard.install_bounded_replay_buffer_sample(_trainer(buffer))
    upstream.release.set()

    with pytest.raises(RuntimeError, match=r"failed prompt.*failed"):
        buffer.sample(partition_id="train", global_steps=7)
    assert upstream.calls == 1


def test_recorded_rollout_cause_beats_generic_failed_prompt(monkeypatch, tmp_path):
    rollout_path = tmp_path / "rollout-failure"
    monkeypatch.setenv("FLASH_OPD_ROLLOUT_FAILURE_PATH", str(rollout_path))
    try:
        raise LookupError("replay-original-sentinel")
    except LookupError as error:
        bridge._write_rollout_failure_fallback(error, "permanent")

    upstream = _Upstream()
    buffer = _Buffer(upstream)
    buffer.partitions["train"]["failed"] = {"global_steps": 7, "status": "failure"}
    replay_guard.install_bounded_replay_buffer_sample(_trainer(buffer))
    upstream.release.set()

    with pytest.raises(RuntimeError) as excinfo:
        buffer.sample(partition_id="train", global_steps=7)
    detail = str(excinfo.value)
    assert "multi-turn OPD rollout failed" in detail
    assert "LookupError" in detail
    assert "replay-original-sentinel" in detail
    assert "opd replay observed failed prompt" not in detail


def test_initially_empty_step_stays_guarded_until_target_step_is_observed():
    upstream = _Upstream()
    buffer = _Buffer(upstream)
    trainer = _trainer(buffer)
    original_identity = trainer.replay_buffer
    replay_guard.install_bounded_replay_buffer_sample(trainer)
    upstream.release.set()
    state = {"done": threading.Event()}
    thread = threading.Thread(target=_run_sample, args=(buffer, state), daemon=True)
    thread.start()

    assert not state["done"].wait(0.03)
    with buffer.lock:
        buffer.partitions["train"]["prompt"] = {"global_steps": 7, "status": "finished"}
    assert state["done"].wait(1)
    assert state["value"] is upstream.value
    assert trainer.replay_buffer is original_identity


def test_failure_before_during_and_after_upstream_sample_is_rejected():
    for timing in ("before", "during", "after"):
        upstream = _Upstream()
        buffer = _Buffer(upstream)
        if timing == "before":
            buffer.partitions["train"]["prompt"] = {"global_steps": 7, "status": "failure"}
        else:
            buffer.partitions["train"]["prompt"] = {"global_steps": 7, "status": "running"}
        replay_guard.install_bounded_replay_buffer_sample(_trainer(buffer))
        state = {"done": threading.Event()}
        thread = threading.Thread(target=_run_sample, args=(buffer, state), daemon=True)
        thread.start()
        if timing == "during":
            with buffer.lock:
                buffer.partitions["train"]["prompt"]["status"] = "failure"
            upstream.release.set()
        elif timing == "after":
            upstream.release.set()
            time.sleep(0.01)
            with buffer.lock:
                buffer.partitions["train"]["prompt"]["status"] = "failure"
        else:
            upstream.release.set()
        assert state["done"].wait(1), timing
        assert isinstance(state.get("error"), RuntimeError), timing
        assert "failed prompt" in str(state["error"]), timing


def test_survivor_only_result_is_rechecked_before_return():
    upstream = _Upstream()
    buffer = _Buffer(upstream)
    buffer.partitions["train"] = {
        "prompt-a": {"global_steps": 7, "status": "finished"},
        "prompt-b": {"global_steps": 7, "status": "running"},
        "prompt-a_0_0": {"global_steps": 7, "status": "success"},
    }
    replay_guard.install_bounded_replay_buffer_sample(_trainer(buffer))
    upstream.release.set()
    state = {"done": threading.Event()}
    threading.Thread(target=_run_sample, args=(buffer, state), daemon=True).start()
    time.sleep(0.01)
    with buffer.lock:
        buffer.partitions["train"]["prompt-b"]["status"] = "failure"

    assert state["done"].wait(1)
    assert isinstance(state.get("error"), RuntimeError)


def test_a_prompt_that_fails_while_upstream_returns_cannot_reach_training(monkeypatch):
    """the recheck after upstream returns, isolated from the top-of-loop check.

    a prompt can flip to failure in the window between the loop's status read and the return of a
    survivor-only batch. without the second read that batch trains on fewer rollouts than were
    collected, so drive the two reads apart deterministically rather than by sleeping.
    """
    upstream = _Upstream()
    buffer = _Buffer(upstream)
    replay_guard.install_bounded_replay_buffer_sample(_trainer(buffer))
    upstream.release.set()

    terminal = (("prompt-a", "finished"), ("prompt-b", "finished"))
    failed = (("prompt-a", "finished"), ("prompt-b", "failure"))
    reads = {"count": 0}

    def _staged(_buffer, _partition_id, _global_steps):
        # every read the loop takes is terminal-and-clean; only the post-upstream read sees the
        # failure, so a passing assertion can come from no other branch.
        reads["count"] += 1
        return failed if reads["count"] > 1 else terminal

    monkeypatch.setattr(replay_guard, "_target_statuses", _staged)

    with pytest.raises(RuntimeError, match=r"failed prompt.*prompt-b"):
        buffer.sample(partition_id="train", global_steps=7)


def test_actor_death_beats_upstream_but_probe_error_is_unknown(monkeypatch):
    upstream = _Upstream()
    buffer = _Buffer(upstream)
    buffer.partitions["train"]["prompt"] = {"global_steps": 7, "status": "running"}
    replay_guard.install_bounded_replay_buffer_sample(_trainer(buffer))
    upstream.release.set()
    monkeypatch.setattr(replay_guard, "_actor_probe_state", lambda _trainer: "dead")
    with pytest.raises(RuntimeError, match="actor died"):
        buffer.sample(partition_id="train", global_steps=7)

    upstream = _Upstream()
    upstream.error = ValueError("upstream")
    buffer = _Buffer(upstream)
    buffer.partitions["train"]["prompt"] = {"global_steps": 7, "status": "finished"}
    replay_guard.install_bounded_replay_buffer_sample(_trainer(buffer))
    upstream.release.set()
    monkeypatch.setattr(replay_guard, "_actor_probe_state", lambda _trainer: "unknown")
    with pytest.raises(ValueError, match="upstream"):
        buffer.sample(partition_id="train", global_steps=7)


def test_status_change_resets_stall_and_success_object_is_preserved():
    upstream = _Upstream()
    buffer = _Buffer(upstream)
    buffer.partitions["train"]["prompt"] = {"global_steps": 7, "status": "running"}
    replay_guard.install_bounded_replay_buffer_sample(_trainer(buffer))
    state = {"done": threading.Event()}
    threading.Thread(target=_run_sample, args=(buffer, state), daemon=True).start()
    time.sleep(0.05)
    with buffer.lock:
        buffer.partitions["train"]["prompt"]["status"] = "finished"
    upstream.release.set()

    assert state["done"].wait(1)
    assert state["value"] is upstream.value


def test_installer_is_idempotent_and_stall_is_named():
    upstream = _Upstream()
    buffer = _Buffer(upstream)
    buffer.partitions["train"]["prompt"] = {"global_steps": 7, "status": "running"}
    trainer = _trainer(buffer)
    replay_guard.install_bounded_replay_buffer_sample(trainer)
    wrapped = buffer.sample
    replay_guard.install_bounded_replay_buffer_sample(trainer)
    assert buffer.sample is wrapped

    upstream.release.set()
    with pytest.raises(RuntimeError, match="opd_replay_stall"):
        buffer.sample(partition_id="train", global_steps=7)
    assert upstream.calls == 1
