"""bounded replay sampling guard for the isolated opd verl child."""

from __future__ import annotations

import os
import threading
import time
import types

if __name__ == "flash_opd_replay_guard":
    from flash_child_diagnostics import sanitize_diagnostic
    from flash_opd_bridge import _read_rollout_failure_fallback, _render_rollout_failure
else:
    from flash._internal.diagnostics import sanitize_diagnostic
    from flash.engine.worker.train.opd.child.bridge import (
        _read_rollout_failure_fallback,
        _render_rollout_failure,
    )

_REPLAY_STALL_SECONDS = 1200.0
_REPLAY_POLL_SECONDS = 0.1
_ACTOR_PROBE_SECONDS = 5.0
_INSTALLED_ATTR = "_flash_bounded_replay_sample_installed"


def _target_statuses(buffer, partition_id: str, global_steps: int) -> tuple[tuple[str, str], ...]:
    with buffer.lock:
        partition = buffer.partitions.get(partition_id, {})
        return tuple(
            sorted(
                (str(key), str(tag.get("status", "")))
                for key, tag in partition.items()
                if tag.get("global_steps") == global_steps
            )
        )


def _failure_key(statuses: tuple[tuple[str, str], ...]) -> str | None:
    return next((key for key, status in statuses if status == "failure"), None)


def _status_is_terminal(statuses: tuple[tuple[str, str], ...]) -> bool:
    return bool(statuses) and all(status != "running" for _key, status in statuses)


def _actor_probe_state(trainer) -> str:
    """return dead only for ray's decisive actor-death exception."""
    try:
        import ray
        from ray.exceptions import RayActorError
    except Exception:
        return "unknown"
    workers = getattr(getattr(trainer, "async_rollout_manager", None), "agent_loop_workers", ())
    saw_live = False
    for worker in workers:
        try:
            reference = worker.__ray_ready__.remote()
            ready, _pending = ray.wait([reference], timeout=0)
            if not ready:
                continue
            ray.get(ready[0])
            saw_live = True
        except RayActorError:
            return "dead"
        except BaseException:
            continue
    return "live" if saw_live else "unknown"


def _raise_prompt_failure(key: str, global_steps: int) -> None:
    recorded = _read_rollout_failure_fallback(os.environ.get("FLASH_OPD_ROLLOUT_FAILURE_PATH", ""))
    if recorded is not None:
        detail = _render_rollout_failure(recorded)
        raise RuntimeError(
            f"multi-turn OPD rollout failed at global_steps={global_steps}: {detail}"
        )
    detail = sanitize_diagnostic(
        f"opd replay observed failed prompt {key!r} at global_steps={global_steps}",
        limit=2000,
    )
    raise RuntimeError(detail)


def _raise_actor_death(global_steps: int) -> None:
    raise RuntimeError(f"opd rollout actor died at global_steps={global_steps}")


def _raise_stall(global_steps: int, statuses: tuple[tuple[str, str], ...]) -> None:
    detail = sanitize_diagnostic(
        f"opd_replay_stall: current-step statuses did not change at global_steps={global_steps}: "
        f"{statuses!r}",
        limit=2000,
    )
    raise RuntimeError(detail)


def _sample_with_guard(trainer, original_sample, buffer, partition_id, global_steps, batch_size):
    if global_steps is None:
        return original_sample(
            partition_id=partition_id,
            global_steps=global_steps,
            batch_size=batch_size,
        )
    completed = threading.Event()
    outcome: dict[str, object] = {}

    def run_upstream() -> None:
        try:
            outcome["value"] = original_sample(
                partition_id=partition_id,
                global_steps=global_steps,
                batch_size=batch_size,
            )
        except BaseException as error:
            outcome["error"] = error
        finally:
            completed.set()

    threading.Thread(target=run_upstream, name="opd-replay-sample", daemon=True).start()
    seen_target_step = False
    fingerprint: tuple[tuple[str, str], ...] | None = None
    unchanged_since = time.monotonic()
    next_actor_probe = 0.0
    while True:
        statuses = _target_statuses(buffer, partition_id, global_steps)
        failure = _failure_key(statuses)
        if failure is not None:
            _raise_prompt_failure(failure, global_steps)
        now = time.monotonic()
        if statuses:
            seen_target_step = True
        if statuses != fingerprint:
            fingerprint = statuses
            unchanged_since = now
        if now >= next_actor_probe:
            if _actor_probe_state(trainer) == "dead":
                _raise_actor_death(global_steps)
            next_actor_probe = now + _ACTOR_PROBE_SECONDS
        if seen_target_step and completed.is_set() and _status_is_terminal(statuses):
            final_statuses = _target_statuses(buffer, partition_id, global_steps)
            failure = _failure_key(final_statuses)
            if failure is not None:
                _raise_prompt_failure(failure, global_steps)
            if not _status_is_terminal(final_statuses):
                fingerprint = final_statuses
                unchanged_since = time.monotonic()
                continue
            if "error" in outcome:
                raise outcome["error"]  # type: ignore[misc]
            return outcome["value"]
        if now - unchanged_since >= _REPLAY_STALL_SECONDS:
            _raise_stall(global_steps, statuses)
        time.sleep(_REPLAY_POLL_SECONDS)


def install_bounded_replay_buffer_sample(trainer):
    """wrap the trainer's bound replay sample without replacing its buffer."""
    buffer = trainer.replay_buffer
    if getattr(buffer, _INSTALLED_ATTR, False):
        return buffer
    original_sample = buffer.sample

    def guarded(self, partition_id: str, global_steps=None, batch_size=None):
        return _sample_with_guard(
            trainer,
            original_sample,
            self,
            partition_id,
            global_steps,
            batch_size,
        )

    buffer.sample = types.MethodType(guarded, buffer)
    setattr(buffer, _INSTALLED_ATTR, True)
    return buffer
