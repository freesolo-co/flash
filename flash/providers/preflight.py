"""Control-plane startup preflight.

``check_run_preflight`` aggregates every missing piece of REQUIRED operator configuration — the
RunPod multi-account pool, the Lambda provider key, the Freesolo backend internal key,
and the shared GitHub + HF tokens — into one startup error, so a half-configured plane fails fast at
deploy instead of degrading silently in production.
"""

from __future__ import annotations

import os

from flash.providers.runpod.preflight import PreflightError

__all__ = [
    "PreflightError",
    "check_run_preflight",
]

# A managed control plane provisions across BOTH GPU substrates (RunPod multi-account pool +
# Lambda) and authenticates to the Freesolo backend, so a complete operator config is
# mandatory. Requiring >= 2 RunPod account keys guards the leak that motivated this gate: a pool
# launched with a SINGLE key never reaps (or fails over) across the second account, so that
# account's idle endpoints pile up unseen.
_REQUIRED_RUNPOD_ACCOUNTS = 2


def _present(var: str) -> bool:
    """True iff env var ``var`` holds a NON-EMPTY value after stripping surrounding whitespace.

    A whitespace-only secret (e.g. ``LAMBDA_API_KEY='   '``) is treated as MISSING here rather than
    silently passing preflight and then failing later when the provider/auth layer actually tries to
    use it. (RunPod is validated via ``keys()``, which already strips and drops empty entries.)
    """
    return bool((os.environ.get(var) or "").strip())


def check_run_preflight() -> None:
    """Validate the FULL operator config for a managed control-plane deployment; raise on missing.

    One self-contained check: a RunPod pool of >= 2 account keys, the Lambda provider
    key, the Freesolo backend internal key, and the shared GitHub + HF tokens. Every missing piece
    is aggregated into a single, actionable startup error.
    """
    from flash.providers.runpod import keys as runpod_keys

    problems: list[str] = []

    # RunPod: a comma-separated pool of >= 2 DISTINCT account keys. keys() is already stripped and
    # drops empties, so a blank / whitespace / ","-only value yields an EMPTY pool (caught as
    # "missing", not "too few"), and counting DISTINCT keys rejects a pool padded with the same key
    # twice (e.g. "rp-a,rp-a"): both entries authenticate to ONE account, so reap/failover still
    # can't cross to a second — exactly the single-account leak this gate exists to block.
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

    # The other GPU substrate and the control-plane <-> backend auth key.
    if not _present("LAMBDA_API_KEY"):
        problems.append("  - LAMBDA_API_KEY: the operator's Lambda Cloud API key")
    if not _present("FREESOLO_INTERNAL_KEY"):
        problems.append(
            "  - FREESOLO_INTERNAL_KEY: the control-plane <-> Freesolo backend internal auth key"
        )

    # Shared run infra (the HF dataset repo itself is per-run, ``[train] hf_repo``, not checked here).
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
