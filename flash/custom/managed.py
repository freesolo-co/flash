"""Session credentials and backend-specific resume validation for managed custom runs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import replace
from pathlib import Path


def _attempt_secret(label, run_id, attempt):
    secret = os.environ.get("FREESOLO_INTERNAL_KEY", "")
    if not secret:
        raise RuntimeError(
            "managed custom sessions require FREESOLO_INTERNAL_KEY on the controller"
        )
    message = json.dumps([label, run_id, attempt]).encode()
    return hmac.new(secret.encode(), message, hashlib.sha256)


def session_token(run_id, attempt):
    return _attempt_secret("flash-custom-session-v1", run_id, attempt).hexdigest()


SESSION_TLS_NAME = "flash-custom-session"


def session_tls(run_id, attempt):
    """A self-signed TLS identity for a directly reachable (Spot) custom worker.

    Derived like the session token, so the control plane can hand the controller the exact
    certificate to pin without storing anything, and a reconciled relaunch of the same attempt
    serves the same identity. Ed25519 signatures and fixed validity make the bytes deterministic.
    The certificate is the controller's only trust anchor; it names no IP because the VM's
    address is unknown when the attempt is built.
    """
    import datetime as dt

    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.x509.oid import NameOID

    seed = _attempt_secret("flash-custom-session-tls-v1", run_id, attempt).digest()
    key = Ed25519PrivateKey.from_private_bytes(seed)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, SESSION_TLS_NAME)])
    serial = int.from_bytes(hashlib.sha256(seed).digest()[:16], "big") >> 1 or 1
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(serial)
        .not_valid_before(dt.datetime(2026, 1, 1, tzinfo=dt.UTC))
        .not_valid_after(dt.datetime(2036, 1, 1, tzinfo=dt.UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(SESSION_TLS_NAME)]), critical=False
        )
        .sign(key, None)
    )
    return {
        "cert": certificate.public_bytes(serialization.Encoding.PEM).decode(),
        "key": key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode(),
    }


def _custom_checkpoint(source, step, *, files=None, revision=None):
    """``{repo, revision, path}`` for a complete flash-custom-v1 checkpoint of ``source``."""
    from flash.artifacts.api import ArtifactStore, download_file
    from flash.engine.support.checkpoint_transport import MANIFEST, parse_manifest
    from flash.runner.results.checkpoints import (
        _adapter_folder_has_required_files,
        checkpoint_adapter_prefix,
    )

    repo = source.train.hf_repo
    if files is None or revision is None:
        api = ArtifactStore(token=None)
        revision = api.repo_info(repo, repo_type="dataset").sha
        files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    path = checkpoint_adapter_prefix(source, step)
    required = {f"{path}/{name}" for name in ("metadata.json", "algorithm.json")}
    native_state = f"{path}/state.pt" in files
    if not native_state and f"{path}/{MANIFEST}" in files:
        manifest = download_file(
            repo,
            f"{path}/{MANIFEST}",
            repo_type="dataset",
            revision=revision,
            token=None,
        )
        entry = parse_manifest(Path(manifest).read_bytes()).get("state.pt")
        native_state = entry is not None and all(
            f"{path}/{part['path']}" in files for part in entry["parts"]
        )
    if (
        not required <= set(files)
        or not native_state
        or not _adapter_folder_has_required_files(files, path)
    ):
        raise ValueError("custom checkpoint is incomplete")
    metadata_file = download_file(
        repo,
        f"{path}/metadata.json",
        repo_type="dataset",
        revision=revision,
        token=None,
    )
    metadata = json.loads(Path(metadata_file).read_text())
    if metadata.get("format") != "flash-custom-v1" or metadata.get("step") != step:
        raise ValueError("checkpoint has no compatible custom training state")
    if metadata.get("model_revision") != source.model_revision:
        raise ValueError("checkpoint base revision does not match the source run")
    return {"repo": repo, "revision": revision, "path": path}


def latest_run_checkpoint(spec):
    """The newest complete custom checkpoint this run published, or None.

    A Spot attempt replacing a preempted VM continues from here instead of the run's
    starting point. Incomplete saves (a VM stopped mid-upload) are skipped.
    """
    import re

    from flash.artifacts.api import ArtifactStore
    from flash.runner.lifecycle.state import adapter_prefix

    if not spec.train.hf_repo:
        return None
    api = ArtifactStore(token=None)
    revision = api.repo_info(spec.train.hf_repo, repo_type="dataset").sha
    files = api.list_repo_files(spec.train.hf_repo, repo_type="dataset", revision=revision)
    pattern = re.compile(
        re.escape(adapter_prefix(spec)) + r"/checkpoints/step-([1-9][0-9]*)/metadata\.json"
    )
    steps = sorted({int(m.group(1)) for f in files if (m := pattern.fullmatch(f))}, reverse=True)
    for step in steps:
        try:
            return {**_custom_checkpoint(spec, step, files=files, revision=revision), "step": step}
        except ValueError:
            continue
    return None


def resolve_resume(spec, *, owner_org_id, owner_key_id):
    from flash.runner.lifecycle import preparation, status
    from flash.schema import parse_checkpoint_ref

    source_id, step = parse_checkpoint_ref(spec.custom["resume_from"])
    source_status = status.get_status(source_id)
    if not preparation._warmstart_source_is_authorized(
        source_status,
        source_id,
        owner_org_id=owner_org_id,
        owner_key_id=owner_key_id,
    ):
        raise ValueError("custom resume checkpoint is not accessible")
    source = status.effective_spec_from_status(source_status)
    if source.algorithm != "custom" or source.model != spec.model:
        raise ValueError("custom resume requires a custom checkpoint of the same base model")
    if (source.train.lora_rank, source.train.lora_alpha) != (
        spec.train.lora_rank,
        spec.train.lora_alpha,
    ):
        raise ValueError("custom resume requires the source LoRA rank and alpha")
    resolved = _custom_checkpoint(source, step)
    if spec.train.max_steps <= step:
        raise ValueError("train.max_steps must exceed the resumed optimizer step")
    return replace(
        spec,
        model_revision=source.model_revision,
        model_revision_auto=True,
        custom={
            **spec.custom,
            "resolved_resume": resolved,
        },
    )
