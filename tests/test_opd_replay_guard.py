"""Replay ownership: failure precedence, races, progress, and bounded waits."""

from __future__ import annotations

import importlib
import sys
import threading
import types
from collections import defaultdict
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.opd.child import bridge


@pytest.fixture
def recipe(monkeypatch):
    class ReplayBuffer:
        def __init__(self):
            self.lock = threading.Lock()
            self.partitions = defaultdict(dict)
            self._stop_event = threading.Event()

    dependency = types.ModuleType("verl.trainer.main_ppo_sync")
    dependency.ReplayBuffer = ReplayBuffer
    dependency.KVBatchMeta = SimpleNamespace
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    name = "flash.engine.worker.train.opd.child.replay"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules.pop(name)
    module = importlib.import_module(name)
    clock = SimpleNamespace(now=0.0, advance=lambda: None)

    def sleep(seconds):
        clock.now += seconds
        clock.advance()

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    monkeypatch.setattr(module, "_REPLAY_POLL_SECONDS", 0.1)
    monkeypatch.setattr(module, "_REPLAY_STALL_SECONDS", 0.3)
    monkeypatch.setattr(module, "_actor_probe_state", lambda _: "unknown")
    monkeypatch.delenv("FLASH_OPD_ROLLOUT_FAILURE_PATH", raising=False)
    buffer = module.BoundedReplayBuffer(SimpleNamespace())
    return module, buffer, clock


def test_failed_prompt_beats_running_siblings_and_successful_rollouts(recipe):
    _, buffer, _ = recipe
    buffer.partitions["train"] = {
        "failed": {"global_steps": 7, "status": "failure"},
        "sibling": {"global_steps": 7, "status": "running"},
        "success": {"global_steps": 7, "status": "success"},
    }
    with pytest.raises(RuntimeError, match=r"failed prompt.*failed"):
        buffer.sample("train", global_steps=7)


def test_recorded_rollout_cause_beats_generic_failed_prompt(recipe, monkeypatch, tmp_path):
    _, buffer, _ = recipe
    monkeypatch.setenv("FLASH_OPD_ROLLOUT_FAILURE_PATH", str(tmp_path / "failure"))
    try:
        raise LookupError("replay-original-sentinel")
    except LookupError as error:
        bridge._write_rollout_failure_fallback(error, "permanent")
    buffer.partitions["train"]["failed"] = {"global_steps": 7, "status": "failure"}
    with pytest.raises(
        RuntimeError, match=r"multi-turn OPD rollout failed.*LookupError.*replay-original-sentinel"
    ):
        buffer.sample("train", global_steps=7)


def test_initially_empty_step_waits_until_target_step_is_observed(recipe):
    _, buffer, clock = recipe
    buffer.partitions["train"]["old"] = {"global_steps": 6, "status": "success"}

    def arrive():
        assert clock.now > 0
        buffer.partitions["train"]["new"] = {"global_steps": 7, "status": "success"}

    clock.advance = arrive
    result = buffer.sample("train", global_steps=7)
    assert result.keys == ["new"]
    assert result.tags == [{"global_steps": 7, "status": "success"}]


def test_survivor_only_result_waits_for_siblings_and_rejects_later_failure(recipe):
    _, buffer, clock = recipe
    buffer.partitions["train"] = {
        "prompt-a": {"global_steps": 7, "status": "finished"},
        "prompt-b": {"global_steps": 7, "status": "running"},
        "sample-a": {"global_steps": 7, "status": "success"},
    }
    clock.advance = lambda: buffer.partitions["train"]["prompt-b"].update(status="failure")
    with pytest.raises(RuntimeError, match=r"failed prompt.*prompt-b"):
        buffer.sample("train", global_steps=7)


def test_failure_between_observation_and_selection_cannot_reach_training(recipe, monkeypatch):
    module, buffer, _ = recipe
    buffer.partitions["train"]["sample"] = {"global_steps": 7, "status": "success"}

    def fail_after_read(_trainer):
        buffer.partitions["train"]["sample"]["status"] = "failure"
        return "live"

    monkeypatch.setattr(module, "_actor_probe_state", fail_after_read)
    with pytest.raises(RuntimeError, match=r"failed prompt.*sample"):
        buffer.sample("train", global_steps=7)


def test_actor_death_is_fatal_but_unknown_probe_does_not_discard_success(recipe, monkeypatch):
    module, buffer, _ = recipe
    buffer.partitions["train"]["sample"] = {"global_steps": 7, "status": "success"}
    monkeypatch.setattr(module, "_actor_probe_state", lambda _: "dead")
    with pytest.raises(RuntimeError, match="actor died"):
        buffer.sample("train", global_steps=7)
    monkeypatch.setattr(module, "_actor_probe_state", lambda _: "unknown")
    assert buffer.sample("train", global_steps=7).keys == ["sample"]


def test_status_changes_reset_stall_budget(recipe):
    _, buffer, clock = recipe
    buffer.partitions["train"]["prompt"] = {"global_steps": 7, "status": "running"}

    def progress():
        # Total wait exceeds the stall budget, but every observed change resets that budget.
        if clock.now >= 0.5:
            buffer.partitions["train"]["prompt"]["status"] = "success"
        elif clock.now >= 0.2:
            buffer.partitions["train"]["peer"] = {"global_steps": 7, "status": "finished"}

    clock.advance = progress
    assert buffer.sample("train", global_steps=7).keys == ["prompt"]
    assert clock.now >= 0.5


def test_stalled_replay_is_named_and_does_not_spawn_a_sampling_thread(recipe):
    _, buffer, _ = recipe
    buffer.partitions["train"]["prompt"] = {"global_steps": 7, "status": "running"}
    before = set(threading.enumerate())
    with pytest.raises(RuntimeError, match="opd_replay_stall"):
        buffer.sample("train", global_steps=7)
    assert set(threading.enumerate()) == before
