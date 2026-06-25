"""Lambda Cloud credential handling (operator-side), mirroring the RunPod auth module.

The Lambda REST client authenticates via the ``LAMBDA_API_KEY`` environment variable, set by
the **operator** on the control-plane host. Env-only by design, exactly like ``RUNPOD_API_KEY``:
it is never written to config files. (Unlike Vast/RunPod, Lambda has no instance-scoped key, so
the operator key is also NOT shipped to the box — teardown is control-plane-side only.)
"""

from __future__ import annotations

from .._auth import load_provider_key

_ENV_VAR = "LAMBDA_API_KEY"


def load_api_key() -> str | None:
    """API key from the environment (operator configuration)."""
    return load_provider_key(_ENV_VAR)
