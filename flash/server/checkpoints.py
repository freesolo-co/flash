"""Mirror a run's deployable RL checkpoints to the freesolo backend (best-effort)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from flash.runner.checkpoints import list_checkpoints
from flash.spec import JobSpec

from ._internal_client import DEFAULT_TIMEOUT_S, build_internal_request, internal_key, org_id_of

_RECORD_PATH = "/api/runs/internal/checkpoints"


def _post_checkpoints(*, token: str, body: dict) -> dict:
    """POST the checkpoint batch to the backend; raise on any non-2xx/unreachable."""
    req = build_internal_request(_RECORD_PATH, body, token=token)
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
        raw = resp.read()
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        return {}


def register_run_checkpoints(*, internal_key: str, status, checkpoints: list[dict]) -> dict:
    """Upsert checkpoints for one run into the backend store (idempotent by run_id+step)."""
    if not checkpoints:
        raise ValueError("no checkpoints to record")
    context = status.billing_context if isinstance(status.billing_context, dict) else {}
    org_id = org_id_of(context)
    if not org_id:
        raise ValueError("missing org id for run checkpoints")
    spec = status.spec or {}
    first = checkpoints[0]
    body = {
        "orgId": org_id,
        "runId": status.run_id,
        "baseModel": spec.get("model"),
        "repoId": first.get("repo_id"),
        "repoType": first.get("repo_type", "dataset"),
        "checkpoints": [
            {"step": c["step"], "subfolder": c["subfolder"]} for c in checkpoints
        ],
    }
    return _post_checkpoints(token=internal_key, body=body)


def register_checkpoints_best_effort(status, *, log=None) -> int:
    """Mirror deployable checkpoints to the backend; returns count submitted, never raises."""

    def _log(msg: str) -> None:
        print(msg, file=log, flush=True) if log is not None else print(msg)

    key = (internal_key() or "").strip()
    if not key:
        return 0  # local/dev control plane: HF still has the checkpoints
    try:
        spec = JobSpec.from_dict(status.spec)
    except Exception as exc:
        _log(f"[ckpt] register skipped ({status.run_id}): bad spec: {exc}")
        return 0
    checkpoints = list_checkpoints(spec)
    if not checkpoints:
        return 0
    try:
        register_run_checkpoints(
            internal_key=key, status=status, checkpoints=checkpoints
        )
    except (ValueError, urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        _log(f"[ckpt] backend register warn ({status.run_id}): {exc}")
        return 0
    _log(f"[ckpt] registered {len(checkpoints)} checkpoint(s) for {status.run_id}")
    return len(checkpoints)
