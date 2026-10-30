"""Bound upstream replay polling by progress and decisive rollout actor failures."""

import time

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

from flash.engine.worker.train.core.child.replay_guard import (
    _ACTOR_PROBE_SECONDS,
    _REPLAY_POLL_SECONDS,
    _REPLAY_STALL_SECONDS,
    _actor_probe_state,
    _failure_key,
    _raise_actor_death,
    _raise_prompt_failure,
    _raise_stall,
)


class BoundedReplayBuffer(ReplayBuffer):
    def __init__(self, trainer, **kwargs):
        self.trainer = trainer
        super().__init__(poll_interval=_REPLAY_POLL_SECONDS, **kwargs)

    def sample(self, global_steps, partition_id, batch_size):
        self._sampling_step = global_steps
        self._sampling_partition = partition_id
        self._fingerprint = None
        self._unchanged_since = time.monotonic()
        self._next_probe = 0.0
        return super().sample(global_steps, partition_id, batch_size)

    def _sync_metadata_from_transfer_queue(self):
        super()._sync_metadata_from_transfer_queue()
        statuses = tuple(
            sorted(
                (str(key), str(tag.get("status", "")))
                for key, tag in self.partitions.get(self._sampling_partition, {}).items()
            )
        )
        # upstream stores prompt statuses separately from trajectory rows.
        failures = self.failure_keys[self._sampling_partition]
        if failures:
            _raise_prompt_failure(sorted(failures)[0], self._sampling_step)
        failure = _failure_key(statuses)
        if failure is not None:
            _raise_prompt_failure(failure, self._sampling_step)
        now = time.monotonic()
        fingerprint = (
            statuses,
            tuple(sorted(self.pending_keys[self._sampling_partition])),
            tuple(sorted(self.running_keys[self._sampling_partition])),
        )
        if fingerprint != self._fingerprint:
            self._fingerprint, self._unchanged_since = fingerprint, now
        if now >= self._next_probe:
            if _actor_probe_state(self.trainer) == "dead":
                _raise_actor_death(self._sampling_step)
            self._next_probe = now + _ACTOR_PROBE_SECONDS
        if now - self._unchanged_since >= _REPLAY_STALL_SECONDS:
            _raise_stall(self._sampling_step, statuses)
