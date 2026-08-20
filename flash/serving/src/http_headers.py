"""Request-header parsing for the serving front door: bearer tokens, the trusted-internal-key
compare, and checkpoint response headers.

Split out of router.py's app builder. These read only the request/response headers -- no router,
pool or settings state -- so they are testable without building the app.
"""

from typing import Any

from fastapi import Request


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("Authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = value.strip()
    return token or None


def _checkpoint_headers(active_checkpoint: Any) -> dict[str, str]:
    checkpoint = str(active_checkpoint or "").strip()
    return {"X-Freesolo-Checkpoint": checkpoint} if checkpoint else {}

