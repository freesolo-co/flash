"""Durable Hugging Face intent artifacts for short-lived RunPod Pod controllers."""

from __future__ import annotations

import hashlib
import io
import json
import math
import secrets
import threading
import time
from dataclasses import dataclass

from flash.providers.runpod.pod_identity import RunpodPodHandle

_INTENT_VERSION = 1
INTENT_LEASE_S = 600.0
_CAS_ATTEMPTS = 4
_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


class IntentLeaseHeld(RuntimeError):
    """Another live controller owns this remote lifecycle identity."""


def new_intent_owner(kind: str) -> str:
    """Return an unguessable per-invocation owner token without embedding caller metadata."""
    if not kind:
        raise ValueError("RunPod intent owner kind is required")
    return f"{kind}-{secrets.token_hex(16)}"


def intent_lock(repo: str, path: str) -> threading.Lock:
    """Serialize same-process owners; the remote CAS lease remains authoritative."""
    key = (repo, path)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def intent_path(kind: str, identity: str) -> str:
    """Return one stable non-secret artifact path for a controller identity."""
    if not kind or not identity:
        raise ValueError("RunPod intent kind and identity are required")
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f".flash/runpod-intents/{kind}/{digest}.json"


def _commit_oid(value) -> str | None:
    for field in ("oid", "commit_oid", "sha"):
        candidate = getattr(value, field, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _is_cas_conflict(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in {409, 412}


@dataclass
class HfRunpodIntentStore:
    """CAS-backed lease and lifecycle phases without payload bytes or provider credentials."""

    api: object
    repo: str
    path: str
    token: str | None
    kind: str
    identity: str
    owner: str
    lease_s: float = INTENT_LEASE_S
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.owner:
            raise ValueError("RunPod intent owner is required")
        if not math.isfinite(self.lease_s) or self.lease_s <= 120.0:
            raise ValueError("RunPod intent lease must exceed the reconciliation window")

    def _repo_revision(self) -> str:
        info = self.api.repo_info(repo_id=self.repo, repo_type="dataset")
        revision = str(getattr(info, "sha", "") or "").strip()
        if not revision:
            raise RuntimeError("RunPod intent repository revision is missing")
        return revision

    def _read_record(self) -> dict | None:
        self.revision = self._repo_revision()
        files = self.api.list_repo_files(
            repo_id=self.repo,
            repo_type="dataset",
            revision=self.revision,
        )
        if self.path not in files:
            return None
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=self.repo,
            repo_type="dataset",
            filename=self.path,
            revision=self.revision,
            token=self.token,
            force_download=True,
        )
        with open(local, encoding="utf-8") as stream:
            record = json.load(stream)
        if type(record) is not dict or record.get("version") != _INTENT_VERSION:
            raise RuntimeError("persisted RunPod controller intent is malformed")
        if record.get("kind") != self.kind or record.get("identity") != self.identity:
            raise RuntimeError("persisted RunPod controller intent identity conflicts")
        if type(record.get("owner")) is not str or not record["owner"]:
            raise RuntimeError("persisted RunPod controller owner is invalid")
        state = record.get("state")
        if state == "cleared":
            return record
        if state != "active":
            raise RuntimeError("persisted RunPod controller intent state is invalid")
        lease_expires_at = record.get("lease_expires_at")
        if (
            isinstance(lease_expires_at, bool)
            or not isinstance(lease_expires_at, (int, float))
            or not math.isfinite(float(lease_expires_at))
            or lease_expires_at <= 0
        ):
            raise RuntimeError("persisted RunPod controller lease is invalid")
        if type(record.get("run_id")) is not str or not record["run_id"]:
            raise RuntimeError("persisted RunPod controller run identity is invalid")
        if type(record.get("seed")) is not int or record["seed"] < 0:
            raise RuntimeError("persisted RunPod controller seed is invalid")
        if type(record.get("handle")) is not dict:
            raise RuntimeError("persisted RunPod controller handle is missing")
        RunpodPodHandle.from_dict(record["handle"])
        return record

    def load(self) -> dict | None:
        """Return a validated active intent, treating a cleared marker as absent."""
        record = self._read_record()
        return record if record is not None and record.get("state") == "active" else None

    def _cas_upload(self, record: dict, message: str, expected_revision: str) -> None:
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        result = self.api.upload_file(
            path_or_fileobj=io.BytesIO(encoded),
            path_in_repo=self.path,
            repo_id=self.repo,
            repo_type="dataset",
            commit_message=message,
            parent_commit=expected_revision,
        )
        self.revision = _commit_oid(result) or self._repo_revision()

    def _active_record(self, run_id: str, seed: int, handle: dict, now: float) -> dict:
        strict = RunpodPodHandle.from_dict(handle)
        return {
            "version": _INTENT_VERSION,
            "kind": self.kind,
            "identity": self.identity,
            "owner": self.owner,
            "state": "active",
            "lease_expires_at": now + self.lease_s,
            "run_id": run_id,
            "seed": seed,
            "handle": strict.to_dict(),
        }

    def publish_active(self, run_id: str, seed: int, handle: dict) -> None:
        """Publish a phase only when this controller owns the current CAS lease."""
        for attempt in range(_CAS_ATTEMPTS):
            existing = self._read_record()
            if (
                existing is not None
                and existing.get("state") == "active"
                and existing["owner"] != self.owner
            ):
                raise IntentLeaseHeld("RunPod controller intent is owned by another controller")
            desired = self._active_record(run_id, seed, handle, time.time())
            try:
                self._cas_upload(
                    desired,
                    f"persist {self.kind} RunPod lifecycle phase",
                    self.revision,
                )
                return
            except Exception as exc:
                if not _is_cas_conflict(exc) or attempt + 1 == _CAS_ATTEMPTS:
                    raise
        raise AssertionError("unreachable")

    def renew(self) -> dict:
        """Extend only this controller's active lease using the current parent commit."""
        for attempt in range(_CAS_ATTEMPTS):
            existing = self._read_record()
            if existing is None or existing.get("state") != "active":
                raise RuntimeError("RunPod controller intent is not active")
            if existing["owner"] != self.owner:
                raise IntentLeaseHeld("RunPod controller lease renewal owner mismatch")
            desired = {**existing, "lease_expires_at": time.time() + self.lease_s}
            try:
                self._cas_upload(
                    desired,
                    f"renew {self.kind} RunPod lifecycle lease",
                    self.revision,
                )
                return desired
            except Exception as exc:
                if not _is_cas_conflict(exc) or attempt + 1 == _CAS_ATTEMPTS:
                    raise
        raise AssertionError("unreachable")

    def claim_expired(self) -> dict | None:
        """Atomically claim an expired intent; a live foreign lease fails closed."""
        for attempt in range(_CAS_ATTEMPTS):
            existing = self._read_record()
            if existing is None or existing.get("state") == "cleared":
                return None
            now = time.time()
            if float(existing["lease_expires_at"]) > now:
                raise IntentLeaseHeld("RunPod controller intent has a live remote owner")
            desired = {
                **existing,
                "owner": self.owner,
                "lease_expires_at": now + self.lease_s,
            }
            try:
                self._cas_upload(
                    desired,
                    f"claim expired {self.kind} RunPod lifecycle intent",
                    self.revision,
                )
                return desired
            except Exception as exc:
                if not _is_cas_conflict(exc) or attempt + 1 == _CAS_ATTEMPTS:
                    raise
        raise AssertionError("unreachable")

    def clear(self) -> None:
        """Mark reusable only while this controller owns the active remote lease."""
        for attempt in range(_CAS_ATTEMPTS):
            existing = self._read_record()
            if existing is None or existing.get("state") == "cleared":
                return
            if existing["owner"] != self.owner:
                raise IntentLeaseHeld("RunPod controller cleanup owner mismatch")
            desired = {
                "version": _INTENT_VERSION,
                "kind": self.kind,
                "identity": self.identity,
                "owner": self.owner,
                "state": "cleared",
            }
            try:
                self._cas_upload(
                    desired,
                    f"clear {self.kind} RunPod lifecycle intent",
                    self.revision,
                )
                return
            except Exception as exc:
                if not _is_cas_conflict(exc) or attempt + 1 == _CAS_ATTEMPTS:
                    raise
        raise AssertionError("unreachable")
