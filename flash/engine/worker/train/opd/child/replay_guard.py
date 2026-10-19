"""bounded replay sampling guard for the isolated opd verl child."""

from __future__ import annotations

import os

from flash._internal.diagnostics import sanitize_diagnostic
from flash.engine.worker.train.opd.child.bridge import (
    _read_rollout_failure_fallback,
    _render_rollout_failure,
)

_REPLAY_STALL_SECONDS = 1200.0
_REPLAY_POLL_SECONDS = 0.1
_ACTOR_PROBE_SECONDS = 5.0


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
