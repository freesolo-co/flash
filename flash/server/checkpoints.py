"""Mirror a run's deployable RL checkpoints to the freesolo backend.

The worker streams each step's LoRA adapter to the run's HF repo; HF is the source of truth
for what's deployable. This module persists that list to the backend's ``run_checkpoints``
store so the dashboard/SDK can enumerate a run's checkpoints without crawling HF, and so a
cancelled run's checkpoints survive in one queryable place.

Like ``flash.server.billing``, the POST is authenticated with the operator INTERNAL key (the
control plane never persists a user's freesolo key) and carries the org id from the run's
non-secret billing context. Unlike billing, checkpoint persistence is STRICTLY best-effort:
a failure here must never disturb a run or a deploy, so the public entry point swallows
everything."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from flash.runner.checkpoints import list_checkpoints
from flash.spec import JobSpec

from .auth import INTERNAL_KEY_ENV, freesolo_base_url

_TIMEOUT_S = 10.0
_RECORD_PATH = "/api/runs/internal/checkpoints"


def _post_checkpoints(*, token: str, body: dict) -> dict:
    """POST the checkpoint batch to the backend; raise on any non-2xx/unreachable.

    Callers in this module always wrap this in a best-effort guard — the raise exists so the
    one network boundary is easy for tests to stub/assert."""
    url = f"{freesolo_base_url()}{_RECORD_PATH}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        raw = resp.read()
    try:
        return json.loads(raw or b"{}")
    except ValueError:
        return {}


def register_run_checkpoints(*, internal_key: str, status, checkpoints: list[dict]) -> dict:
    """Upsert ``checkpoints`` for one run into the backend store (idempotent by run_id+step).

    Pulls the org id from the run's persisted billing context (same source as billing). Raises
    ``ValueError`` when there's nothing to record or no org id; raises ``urllib`` errors through
    on a backend failure — ``register_checkpoints_best_effort`` is the guarded wrapper most
    callers use."""
    if not checkpoints:
        raise ValueError("no checkpoints to record")
    context = status.billing_context if isinstance(status.billing_context, dict) else {}
    org_id = str(context.get("org_id") or "").strip()
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
    """List ``status``'s deployable checkpoints from HF and mirror them to the backend.

    Returns the number of checkpoints submitted (0 if none, or if persistence was skipped /
    failed). Never raises: the HF copy remains the source of truth, so a persistence miss only
    costs the convenience of a DB-backed listing — not correctness."""

    def _log(msg: str) -> None:
        print(msg, file=log, flush=True) if log is not None else print(msg)

    internal_key = os.environ.get(INTERNAL_KEY_ENV, "").strip()
    if not internal_key:
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
            internal_key=internal_key, status=status, checkpoints=checkpoints
        )
    except (ValueError, urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        _log(f"[ckpt] backend register warn ({status.run_id}): {exc}")
        return 0
    _log(f"[ckpt] registered {len(checkpoints)} checkpoint(s) for {status.run_id}")
    return len(checkpoints)
