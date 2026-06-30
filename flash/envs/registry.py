"""Environment registry used by specs, worker, CLI, and server."""

from __future__ import annotations

from typing import Any

from .base import Environment

_FLASH_TRAIN_MAX_EXAMPLES = "__flash_train_max_examples"


def worker_pip_for_env(env_id: str) -> list[str]:
    """Pip deps the GPU worker needs to run a Freesolo environment."""
    return ["freesolo"]


def training_env_params(spec) -> dict[str, Any]:
    """Environment-loader params for a training run.

    ``[train].max_examples`` caps Flash training, but some environments also have a
    loader-side cap (commonly ``max_examples`` or ``limit``) with a small default. Carry
    the training intent through an internal marker so the adapter can disable or raise that
    loader cap before the worker calls ``env.dataset()``.
    """
    params = dict(getattr(getattr(spec, "environment", None), "params", None) or {})
    train = getattr(spec, "train", None)
    if train is None:
        return params
    if "max_examples" in params or "limit" in params:
        return params

    max_examples = getattr(train, "max_examples", None)
    algorithm = str(getattr(spec, "algorithm", "") or "").lower()
    if algorithm != "sft" and max_examples is None:
        return params

    params[_FLASH_TRAIN_MAX_EXAMPLES] = (
        int(max_examples) if max_examples is not None and int(max_examples) > 0 else None
    )
    return params


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
