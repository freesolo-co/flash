"""bounded replay sampling guard for the isolated GRPO verl child."""

from __future__ import annotations

from flash._internal.diagnostics import sanitize_diagnostic

_REPLAY_STALL_SECONDS = 1200.0
_REPLAY_POLL_SECONDS = 0.1
_ACTOR_PROBE_SECONDS = 5.0


def _failure_key(statuses: tuple[tuple[str, str], ...]) -> str | None:
    return next((key for key, status in statuses if status == "failure"), None)


def _actor_probe_state(trainer) -> str:
    """return dead only for ray's decisive actor-death exception."""
    try:
        import ray
        from ray.exceptions import RayActorError
    except Exception:
        return "unknown"
    workers = getattr(getattr(trainer, "agent_loop_manager", None), "agent_loop_workers", ())
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
    detail = sanitize_diagnostic(
        f"GRPO replay observed failed prompt {key!r} at global_steps={global_steps}",
        limit=2000,
    )
    raise RuntimeError(detail)


def _raise_actor_death(global_steps: int) -> None:
    raise RuntimeError(f"GRPO rollout actor died at global_steps={global_steps}")


def _raise_stall(global_steps: int, statuses: tuple[tuple[str, str], ...]) -> None:
    detail = sanitize_diagnostic(
        f"GRPO_replay_stall: current-step statuses did not change at global_steps={global_steps}: "
        f"{statuses!r}",
        limit=2000,
    )
    raise RuntimeError(detail)
