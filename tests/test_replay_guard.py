"""Replay failure precedence and bounded polling through the upstream sampler API."""

from __future__ import annotations

import importlib
import sys
import threading
import types
from collections import defaultdict
from types import SimpleNamespace

import pytest


@pytest.fixture
def recipe(monkeypatch):
    clock = SimpleNamespace(now=0.0, advance=lambda: None)

    class ReplayBuffer:
        def __init__(self, **kwargs):
            self.partitions = defaultdict(dict)
            self.pending_keys = defaultdict(set)
            self.running_keys = defaultdict(set)
            self.failure_keys = defaultdict(set)
            self.poll_interval = kwargs["poll_interval"]
            self.snapshots = []

        def _sync_metadata_from_transfer_queue(self):
            pass

        def sample(self, global_steps, partition_id, batch_size):
            while True:
                self._sync_metadata_from_transfer_queue()
                self.snapshots.append((global_steps, partition_id, batch_size))
                rows = self.partitions[partition_id]
                if (
                    rows
                    and not self.pending_keys[partition_id]
                    and not self.running_keys[partition_id]
                ):
                    return list(rows), {}
                clock.now += self.poll_interval
                clock.advance()

    dependency = types.ModuleType("verl.trainer.ppo.v1.replay_buffer")
    dependency.ReplayBuffer = ReplayBuffer
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    name = "flash.engine.worker.train.core.child.replay"
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules.pop(name)
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(module, "_REPLAY_POLL_SECONDS", 0.1)
    monkeypatch.setattr(module, "_REPLAY_STALL_SECONDS", 0.3)
    monkeypatch.setattr(module, "_actor_probe_state", lambda _: "unknown")
    return module, module.BoundedReplayBuffer(SimpleNamespace()), clock


def test_failed_prompt_beats_running_siblings_and_successful_rollouts(recipe):
    _, buffer, _ = recipe
    buffer.failure_keys["train"].add("failed")
    buffer.running_keys["train"].add("sibling")
    buffer.partitions["train"]["success"] = {"status": "success"}
    with pytest.raises(RuntimeError, match=r"failed prompt.*failed"):
        buffer.sample(7, "train", 2)
    assert not buffer.snapshots


def test_empty_queue_waits_then_delegates_selection_to_upstream(recipe):
    _, buffer, clock = recipe
    clock.advance = lambda: buffer.partitions["train"].update(new={"status": "success"})
    assert buffer.sample(7, "train", 3) == (["new"], {})
    assert buffer.snapshots == [(7, "train", 3)] * 2


def test_running_sibling_failure_prevents_survivor_only_training(recipe):
    _, buffer, clock = recipe
    buffer.running_keys["train"].add("sibling")
    buffer.partitions["train"]["success"] = {"status": "success"}
    clock.advance = lambda: buffer.failure_keys["train"].add("sibling")
    with pytest.raises(RuntimeError, match=r"failed prompt.*sibling"):
        buffer.sample(7, "train", 2)


def test_actor_death_is_fatal_but_unknown_probe_allows_selection(recipe, monkeypatch):
    module, buffer, _ = recipe
    buffer.partitions["train"]["sample"] = {"status": "success"}
    monkeypatch.setattr(module, "_actor_probe_state", lambda _: "dead")
    with pytest.raises(RuntimeError, match="actor died"):
        buffer.sample(7, "train", 1)
    monkeypatch.setattr(module, "_actor_probe_state", lambda _: "unknown")
    assert buffer.sample(7, "train", 1) == (["sample"], {})


def test_status_changes_reset_stall_budget(recipe):
    _, buffer, clock = recipe
    buffer.pending_keys["train"].add("prompt")

    def progress():
        if clock.now >= 0.5:
            buffer.running_keys["train"].clear()
            buffer.partitions["train"]["sample"] = {"status": "success"}
        elif clock.now >= 0.2:
            buffer.pending_keys["train"].clear()
            buffer.running_keys["train"].add("prompt")

    clock.advance = progress
    assert buffer.sample(7, "train", 1) == (["sample"], {})
    assert clock.now >= 0.5


def test_stalled_replay_is_named_without_a_sampling_thread(recipe):
    _, buffer, _ = recipe
    buffer.running_keys["train"].add("prompt")
    before = set(threading.enumerate())
    with pytest.raises(RuntimeError, match="GRPO_replay_stall"):
        buffer.sample(7, "train", 1)
    assert set(threading.enumerate()) == before
