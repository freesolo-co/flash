"""Environment registry used by specs, worker, CLI, and server."""

from __future__ import annotations

from .base import Environment

FREESOLO_WORKER_SPEC = "freesolo>=0.2.60"


def worker_pip_for_env(env_id: str) -> list[str]:
    """Pip deps the GPU worker needs to run a Freesolo environment."""
    return [FREESOLO_WORKER_SPEC]


def load_environment(
    env_id: str,
    params: dict | None = None,
    resolved_sha: str | None = None,
    *,
    source_kind: str = "",
    package_base64: str = "",
    package_sha256: str = "",
) -> Environment:
    """Load a Freesolo SDK environment and wrap it in Flash's protocol."""
    params = params or {}
    from .adapter import load_freesolo_environment

    if not env_id:
        raise ValueError(
            "no environment specified: set [environment] id to the id returned by "
            "`flash env push --project <project-uuid> --name <name>` "
            "(for example 'your-name/your-env')"
        )
    # managed source values are positional-only so user params cannot shadow them.
    return load_freesolo_environment(
        env_id,
        resolved_sha or None,
        source_kind,
        package_base64,
        package_sha256,
        **params,
    )
