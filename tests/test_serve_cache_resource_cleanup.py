"""Cache ownership with shared artifact bytes and failed destination creation."""

import errno
import fcntl
import hashlib
import os
from contextlib import suppress

import pytest

from flash.serve.app import materialize
from flash.serve.app.manifest import load_serving_manifest
from flash.serve.control._canonical import canonical_json
from tests.test_serve_app_materialize import (
    _artifact_bytes,
    _download_stub,
    _manifest,
    _token_fd,
)


def test_checkpoints_with_identical_weights_share_one_digest_lock(tmp_path, monkeypatch):
    config, weights = _artifact_bytes(tmp_path)
    payload = _manifest(config, weights).payload(False)
    payload["adapters"].append({**payload["adapters"][0], "checkpoint_id": "run-1/step-1"})
    payload["manifest_id"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    manifest = load_serving_manifest(payload)
    cache = tmp_path / "cache"
    materialize.hydrate_manifest(
        manifest,
        cache,
        token_fd=_token_fd(),
        snapshot_download_fn=_download_stub(config, weights, []),
    )
    real_flock = fcntl.flock

    def nonblocking_lock(fd, operation):
        # A duplicate open would wait on this process's own lock forever. Fail immediately.
        return real_flock(fd, operation | fcntl.LOCK_NB)

    monkeypatch.setattr(fcntl, "flock", nonblocking_lock)
    with materialize.locked_manifest_cache(manifest, cache) as paths:
        assert set(paths) == {"run-1/final", "run-1/step-1"}
        assert len(set(paths.values())) == 1


def test_copy_closes_source_if_destination_cannot_be_created(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"adapter")
    destination = tmp_path / "existing"
    destination.write_bytes(b"keep")
    real_open = os.open
    descriptors = []

    def capture_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if path == source:
            descriptors.append(fd)
        return fd

    monkeypatch.setattr(os, "open", capture_open)
    try:
        with pytest.raises(FileExistsError):
            materialize._copy_regular_file(source, destination)
        assert destination.read_bytes() == b"keep"
        assert len(descriptors) == 1
        with pytest.raises(OSError, match="Bad file descriptor") as caught:
            os.fstat(descriptors[0])
        assert caught.value.errno == errno.EBADF
    finally:
        for fd in descriptors:
            with suppress(OSError):
                os.close(fd)
