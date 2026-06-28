"""Environment registry used by specs, worker, CLI, and server.

Every managed run names a Freesolo SDK environment by Hub slug.
The canonical generated environment entrypoint is ``environment.py:load_environment``.
"""

from __future__ import annotations

from .base import Environment


def worker_pip_for_env(env_id: str) -> list[str]:
    """Pip deps the GPU worker needs to run a Freesolo environment."""
    return ["freesolo"]


def load_environment(
    env_id: str, params: dict | None = None, resolved_sha: str | None = None
) -> Environment:
    """Load a Freesolo SDK environment and wrap it in Flash's protocol.

    ``resolved_sha`` is the optional resolve-once hint (the control-plane-pinned commit sha for the
    env's GitHub ref). None/"" preserves today's behavior — the adapter resolves the ref itself.
    """
    params = params or {}
    from .adapter import load_freesolo_environment

    if not env_id:
        raise ValueError(
            "no environment specified: set [environment] id to the id returned by "
            "`flash env push --name <name>` (for example 'your-name/your-env')"
        )
    # User [environment.params] are freeform and forwarded verbatim to the SDK loader. The
    # control-plane resolve-once pin is passed out-of-band as a POSITIONAL-ONLY argument, so a user
    # param of ANY name (even "pinned_sha"/"resolved_sha") lands in **params and reaches the SDK
    # unchanged — it can never bind to or disable the pin. None/"" keeps today's behavior.
    return load_freesolo_environment(env_id, resolved_sha or None, **params)
