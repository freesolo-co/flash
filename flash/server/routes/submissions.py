"""Durable, owner-scoped training submissions, addressed by a caller's idempotency key.

PUT /v1/run-submissions/{id} accepts the POST /v1/runs payload (paid runs only).
Persist a random ID before sending and reuse that ID and payload on every retry.
GET resolves the current job; DELETE cancels it or tombstones a not-yet-arrived intent.
Both return submission_id, run_id and state; run_id is null before admission.
A concurrent operation returns 409/Retry-After, payload conflicts 422, cancellations 410.

Bindings live in the control plane's durable SQLite store, alongside run ownership,
and never expire or disappear on job cancellation. IDs are scoped to the authenticated
key, not globally. Runtime secret values may rotate and are never stored in the mapping.
This contract depends on preserving the plane's existing shared durable state/lock volume;
it cannot recover a lost database. Legacy POST /v1/runs remains non-idempotent.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from contextlib import contextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from flash.runner.lifecycle import state, status
from flash.server.platform import db
from flash.server.platform.deps import require_key
from flash.server.routes import runs

router = APIRouter()


def _validate_id(submission_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", submission_id):
        raise HTTPException(400, "submission ID must contain 16-128 letters, digits, '_' or '-'")


@contextmanager
def _submission_lock(run_id: str):
    # Unlike a timeout-based lease, a live but slow submitter cannot lose ownership.
    # Separate descriptors contend even within a process; the kernel releases on crash.
    os.makedirs(state.RUNS_DIR, exist_ok=True)
    fd = os.open(state.runs_file_path(run_id, ".submission.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HTTPException(
                409, "submission is in progress; retry the same ID", headers={"Retry-After": "2"}
            ) from None
        yield
    finally:
        os.close(fd)


def _existing(row: dict) -> dict | None:
    try:
        result = status.get_status(row["run_id"]).to_dict()
    except FileNotFoundError:
        if row["accepted"]:
            # A missing/corrupt status store is not permission to recreate paid work.
            raise HTTPException(
                503, "accepted submission status unavailable; retry the same ID"
            ) from None
        return None
    db.accept_submission(row["run_id"])
    owner = db.run_owner(row["run_id"])
    if owner is None:
        # Ownership may have been deliberately removed to prevent unsafe recovery.
        raise HTTPException(503, "accepted submission ownership unavailable")
    if owner != row["key_id"]:
        raise HTTPException(409, "submission ownership conflict")
    return {**result, "submission_id": row["submission_id"]}


def _cancelled(row: dict) -> dict:
    return {"submission_id": row["submission_id"], "state": "cancelled", "run_id": None}


@router.put("/v1/run-submissions/{submission_id}")
def submit(
    submission_id: str,
    payload: dict,
    request: Request,
    key: Annotated[dict, Depends(require_key)],
    authorization: Annotated[str | None, Header()] = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
):
    _validate_id(submission_id)
    if payload.get("dry_run", False) is not False:
        raise HTTPException(
            400, "idempotent submissions are for paid runs; preview with POST /v1/runs"
        )
    # Credentials can rotate without changing the logical job. Never persist secret values
    # or hashes of low-entropy secrets. The spec still binds the requested secret names.
    identity = {name: value for name, value in payload.items() if name != "runtime_secrets"}
    identity.setdefault("dry_run", False)
    identity["org_id"] = key.get("org_id") or x_freesolo_org_id
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    row = db.reserve_submission(key["id"], submission_id, digest, state.new_run_id())
    with _submission_lock(row["run_id"]):
        row = db.get_submission(key["id"], submission_id)
        if row["request_hash"] not in (None, digest):
            raise HTTPException(422, "submission ID is already bound to a different request")
        if row["cancelled"]:
            raise HTTPException(410, "submission was cancelled; it will never create another run")
        if result := _existing(row):
            return result

        def accepted(run_id):
            db.accept_submission(run_id)
            if db.get_submission(key["id"], submission_id)["cancelled"]:
                status._update(run_id, "cancelled")
                raise HTTPException(410, "submission was cancelled before dispatch")

        result = runs._create_run(
            payload,
            request,
            key,
            authorization,
            x_freesolo_org_id,
            run_id=row["run_id"],
            on_persisted=accepted,
        )
        db.accept_submission(row["run_id"])
        return {**result, "submission_id": submission_id}


@router.get("/v1/run-submissions/{submission_id}")
def lookup(submission_id: str, key: Annotated[dict, Depends(require_key)]):
    _validate_id(submission_id)
    row = db.get_submission(key["id"], submission_id)
    if row is None:
        raise HTTPException(404, "unknown submission ID")
    with _submission_lock(row["run_id"]):
        row = db.get_submission(key["id"], submission_id)
        if result := _existing(row):
            return result
        if row["cancelled"]:
            return _cancelled(row)
        return {"submission_id": submission_id, "run_id": None, "state": "submission_pending"}


@router.delete("/v1/run-submissions/{submission_id}")
def cancel(submission_id: str, request: Request, key: Annotated[dict, Depends(require_key)]):
    _validate_id(submission_id)
    # A cancellation arriving before a delayed PUT leaves a permanent tombstone.
    row = db.reserve_submission(key["id"], submission_id, None, state.new_run_id(), cancelled=True)
    with _submission_lock(row["run_id"]):
        row = db.get_submission(key["id"], submission_id)
        if _existing(row) is None:
            return _cancelled(row)
        result = runs.cancel(row["run_id"], request, key)
        return {**result, "submission_id": submission_id}
