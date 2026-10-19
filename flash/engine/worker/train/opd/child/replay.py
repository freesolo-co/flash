"""bounded replay ownership without replacing a live buffer's methods."""

import time

from verl.trainer.main_ppo_sync import KVBatchMeta, ReplayBuffer

from flash.engine.worker.train.opd.child.replay_guard import (
    _ACTOR_PROBE_SECONDS,
    _REPLAY_POLL_SECONDS,
    _REPLAY_STALL_SECONDS,
    _actor_probe_state,
    _failure_key,
    _raise_actor_death,
    _raise_prompt_failure,
    _raise_stall,
    _status_is_terminal,
    _target_statuses,
)


class BoundedReplayBuffer(ReplayBuffer):
    def __init__(self, trainer):
        self.trainer = trainer
        super().__init__()

    def sample(self, partition_id, global_steps=None, batch_size=None):
        if (global_steps is None) == (batch_size is None):
            raise ValueError("exactly one of global_steps or batch_size is required")
        if global_steps is None:
            return super().sample(partition_id, global_steps, batch_size)
        fingerprint = None
        unchanged_since = time.monotonic()
        next_probe = 0.0
        while not self._stop_event.is_set():
            statuses = _target_statuses(self, partition_id, global_steps)
            failure = _failure_key(statuses)
            if failure is not None:
                _raise_prompt_failure(failure, global_steps)
            now = time.monotonic()
            if statuses != fingerprint:
                fingerprint, unchanged_since = statuses, now
            if now >= next_probe:
                if _actor_probe_state(self.trainer) == "dead":
                    _raise_actor_death(global_steps)
                next_probe = now + _ACTOR_PROBE_SECONDS
            if _status_is_terminal(statuses):
                with self.lock:
                    current = self.partitions.get(partition_id, {})
                    latest = tuple(
                        sorted(
                            (str(key), str(tag.get("status", "")))
                            for key, tag in current.items()
                            if tag.get("global_steps") == global_steps
                        )
                    )
                    if latest != statuses:
                        continue
                    selected = [
                        (key, dict(tag))
                        for key, tag in current.items()
                        if tag.get("global_steps") == global_steps
                        and tag.get("status") == "success"
                    ]
                return KVBatchMeta(
                    partition_id=partition_id,
                    keys=[key for key, _ in selected],
                    tags=[tag for _, tag in selected],
                )
            if now - unchanged_since >= _REPLAY_STALL_SECONDS:
                _raise_stall(global_steps, statuses)
            time.sleep(_REPLAY_POLL_SECONDS)
        raise RuntimeError("OPD replay buffer closed while waiting for rollout")
