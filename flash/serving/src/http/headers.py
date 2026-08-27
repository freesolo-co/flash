"""Request-header parsing for the serving front door: bearer tokens, the trusted-internal-key
compare, and checkpoint response headers.

Split out of router.py's app builder. These read only the request/response headers plus the
configured key -- no router, pool or settings state -- so they are testable without building
the app.
"""

import hmac
from typing import Any

from fastapi import HTTPException, Request, status


def internal_org_id(request: Request) -> str:
    """return the mandatory tenant scope for an internal checkpoint operation."""

    org_id = (request.headers.get("X-Freesolo-Org-Id") or "").strip()
    if not org_id:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "X-Freesolo-Org-Id is required for internal checkpoint operations",
        )
    return org_id


def optional_internal_org_id(request: Request) -> str | None:
    """read the tenant scope an internal caller supplied, without requiring one."""

    return (request.headers.get("X-Freesolo-Org-Id") or "").strip() or None


def assert_internal(request: Request, internal_key: str | None) -> None:
    if not internal_key:
        # No internal key configured -> the control plane can't be authenticated. Fail closed
        # rather than serve an open /adapters surface (register/teardown). Production always sets
        # FREESOLO_INTERNAL_KEY; an unset key is a misconfiguration, not "auth disabled".
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "serving internal key is not configured"
        )
    presented = request.headers.get("X-Freesolo-Internal-Key") or ""
    # Constant-time compare on a secret header (consistent with is_trusted_internal) so a
    # rejected key can't be recovered via response timing.
    if not hmac.compare_digest(presented, internal_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid serving internal key")


def is_trusted_internal(request: Request, trusted_keys: tuple[str, ...]) -> bool:
    presented = request.headers.get("X-Freesolo-Internal-Key")
    if not presented:
        return False
    # Evaluate every compare_digest (list comp, not a generator) so the per-key constant-time
    # compare always runs -- a short-circuiting any() would leak which key matched via timing.
    # C419 (prefer a generator) is exactly the rewrite the comment above forbids: a
    # generator short-circuits, which is the timing leak. The list is load-bearing.
    return any([hmac.compare_digest(presented, k) for k in trusted_keys])  # noqa: C419


# The job-scope headers a training container sends with a catalog-base sample. The backend
# re-checks all five against the live training job row before authorizing, so they have to
# survive the hop through this service -- it forwards the caller's key but is otherwise the
# only thing standing between the container and /api/serving/authorize.
TRAINING_SCOPE_HEADERS = (
    "x-freesolo-org-id",
    "x-freesolo-training-job-id",
    "x-freesolo-project-id",
    "x-freesolo-worker-id",
    "x-freesolo-training-job-attempt",
)


def training_scope_headers(request: Request) -> dict[str, str]:
    """The training job-scope headers present on this request, lowercased.

    Absent headers are omitted rather than sent empty: a normal customer chat request
    carries none of these, and forwarding five blanks would make every such request look
    like a malformed training call to the backend.
    """
    found = {}
    for name in TRAINING_SCOPE_HEADERS:
        value = (request.headers.get(name) or "").strip()
        if value:
            found[name] = value
    return found


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
