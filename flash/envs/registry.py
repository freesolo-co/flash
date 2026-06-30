"""Environment registry used by specs, worker, CLI, and server."""

from __future__ import annotations

from .base import Environment

FREESOLO_WORKER_SPEC = "freesolo>=0.2.52"


def worker_pip_for_env(env_id: str) -> list[str]:
    """Pip deps the GPU worker needs to run a Freesolo environment."""
    return [FREESOLO_WORKER_SPEC]


def load_environment(
    env_id: str, params: dict | None = None, resolved_sha: str | None = None
) -> Environment:
    """Load a Freesolo SDK environment and wrap it in Flash's protocol."""
    params = params or {}
    from .adapter import load_freesolo_environment

    if not env_id:
        raise ValueError(
            "no environment specified: set [environment] id to the id returned by "
            "`flash env push --name <name>` (for example 'your-name/your-env')"
        )
    # resolved_sha is positional-only so a user param named "resolved_sha" can't shadow it.
    return load_freesolo_environment(env_id, resolved_sha or None, **params)
