"""Stream large checkpoint exports to a private temporary file without following redirects."""

from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from pathlib import Path

from flash._internal.http import _urlopen_no_redirect
from flash.client.http import ClientError
from flash.client.streaming import _cap_socket_timeout
from flash.custom.types import checkpoint_name


def download_checkpoint(http, name, destination, *, timeout=900, max_bytes=4 * 1024**3):
    checkpoint_name(name)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"{http.api_url}/v1/custom/checkpoints/{name}",
        headers={"Accept": "application/zip", **http._auth_headers()},
    )
    deadline = time.monotonic() + timeout
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".flash-checkpoint-")
    try:
        with (
            os.fdopen(fd, "wb") as handle,
            http._translate_http_errors(deadline=deadline),
            _urlopen_no_redirect(request, timeout=timeout, deadline=deadline) as response,
        ):
            declared = response.headers.get("Content-Length")
            expected = int(declared) if declared is not None else None
            if expected is not None and (expected < 0 or expected > max_bytes):
                raise ClientError("checkpoint exceeds export size limit")
            total = 0
            read = getattr(response, "read1", response.read)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ClientError("checkpoint download exceeded its deadline")
                _cap_socket_timeout(response, remaining)
                chunk = read(1024 * 1024 if hasattr(response, "read1") else 1)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ClientError("checkpoint exceeds export size limit")
                handle.write(chunk)
            if expected is not None and total != expected:
                raise ClientError("incomplete checkpoint download; retry the immutable export")
        # no overwrites, including a destination created while the download was in progress.
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
    return destination
