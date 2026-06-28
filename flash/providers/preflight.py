"""Control-plane startup preflight — fail fast on missing operator config."""

from __future__ import annotations

import os

from flash.providers.runpod.preflight import PreflightError

__all__ = [
    "PreflightError",
    "check_run_preflight",
]

# >= 2 distinct keys required so reap/failover can cross accounts; single-account pools leak idle endpoints.
_REQUIRED_RUNPOD_ACCOUNTS = 2


def _present(var: str) -> bool:
    """True iff env var is set to a non-empty, non-whitespace value."""
    return bool((os.environ.get(var) or "").strip())


def check_run_preflight() -> None:
    """Raise PreflightError if any required operator config is missing."""
    from flash.providers.runpod import keys as runpod_keys

    problems: list[str] = []

    pool = runpod_keys.keys()
    distinct = len(set(pool))
    if not pool:
        problems.append(
            f"  - RUNPOD_API_KEY: the operator's RunPod API key "
            f"(>= {_REQUIRED_RUNPOD_ACCOUNTS} comma-separated account keys)"
        )
    elif distinct < _REQUIRED_RUNPOD_ACCOUNTS:
        problems.append(
            f"  - RUNPOD_API_KEY: needs >= {_REQUIRED_RUNPOD_ACCOUNTS} DISTINCT comma-separated "
            f"account keys (found {distinct}) — duplicate keys hit the SAME account, and a "
            f"single-account pool can't reap or fail over across accounts"
        )

    if not _present("LAMBDA_API_KEY"):
        problems.append("  - LAMBDA_API_KEY: the operator's Lambda Cloud API key")
    if not _present("FREESOLO_INTERNAL_KEY"):
        problems.append(
            "  - FREESOLO_INTERNAL_KEY: the control-plane <-> Freesolo backend internal auth key"
        )

    if not _present("GITHUB_TOKEN"):
        problems.append(
            "  - GITHUB_TOKEN: server token with access to managed Freesolo environments"
        )
    if not _present("HF_TOKEN"):
        problems.append(
            "  - HF_TOKEN: a token with write access to each run's "
            "`[train] hf_repo`, e.g. `export HF_TOKEN=hf_...`"
        )

    if problems:
        raise PreflightError(
            "the Flash control plane is missing required operator configuration:\n"
            + "\n".join(problems)
            + "\n\nSet these on the control-plane host."
        )
