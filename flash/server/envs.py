"""Managed environment publishing.

`POST /v1/envs` accepts a packaged verifiers env — built client-side, so the user needs NO
Prime Intellect account — and publishes it to FreeSolo's Prime Hub account using the control
plane's ``PRIME_API_KEY``. Every env is therefore managed under one account: namespaced per
freesolo identity (so users can't collide) and published PRIVATE.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

_PUSH_MAX_ATTEMPTS = 8
_PUSH_CONFLICT_MARKERS = ("already exists", "version already", "duplicate", "conflict", "409")


class EnvPublishError(Exception):
    """A managed env publish failed. ``status`` is the HTTP status the route should return:
    400 for a bad client package, 503 when the control plane itself isn't configured to publish
    (no PRIME_API_KEY / `prime` CLI)."""

    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.status = status


def namespace_for(key: dict) -> str:
    """Stable, Hub-safe prefix isolating one identity's envs under the shared FreeSolo Prime account.

    It MUST be per-key distinct: the server DB stores the SAME placeholder email for every external
    key ("freesolo-user") and the internal key ("freesolo-internal") — see ``db.ensure_external_key``
    / ``ensure_internal_key`` — so namespacing on ``email`` would collapse every external user into
    one namespace and let them collide on env names. So we only trust ``email`` when it's a real,
    user-specific address (contains ``@``); otherwise we fall back to the per-key primary-key ``id``
    (each external token gets its own ``api_keys`` row), then a non-placeholder ``key_prefix``. The
    chosen identifier is stable across re-publishes, so a user's re-push bumps the SAME Hub env."""
    email = str(key.get("email") or "")
    if "@" in email:
        raw = email
    elif key.get("id") is not None:
        raw = f"key-{key['id']}"
    else:
        raw = str(key.get("key_prefix") or "user")
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return slug or "user"


def _sanitize_name(name: str) -> str:
    # Hub env names are lowercase, like the identity namespace they're joined to.
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")
    return slug or "env"


def _safe_extract(tar_bytes: bytes, dest: Path) -> None:
    """Extract a .tar.gz into ``dest``, rejecting absolute paths and parent traversal.

    Defence in depth for an uploaded archive: besides path traversal, we explicitly gate member
    TYPES — only regular files and directories are allowed. Symlinks/hardlinks are rejected (a link
    could point outside ``dest``), and so are every other special member ``tarfile`` understands
    (device/block/char nodes, FIFOs): there's no legitimate reason an env package contains them, and
    ``extractall`` would otherwise attempt to materialize them. We don't rely on tarfile's own
    extraction filter alone; the gate here is the contract.
    """
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if target != root and root not in target.parents:
                raise EnvPublishError(f"unsafe path in env package: {member.name!r}")
            if member.islnk() or member.issym():
                raise EnvPublishError(f"links are not allowed in env packages: {member.name!r}")
            if not (member.isreg() or member.isdir()):
                raise EnvPublishError(
                    f"only regular files and directories are allowed in env packages, but "
                    f"{member.name!r} is a special file (device/fifo/etc.)"
                )
        tar.extractall(dest)


def _slug_from(env_dir: Path, output: str, *, pushed_name: str | None = None) -> str | None:
    """The published ``owner/name`` slug, from prime's metadata file or its stdout.

    ``pushed_name`` is the ``--name`` we pushed under: if prime records the env ``owner`` but not a
    usable ``name`` (or only the stdout names the owner), we can still reconstruct the slug from the
    name we know we published, so a successful push isn't reported as a failure for lack of a slug.
    """
    meta = env_dir / ".prime" / ".env-metadata.json"
    if meta.is_file():
        try:
            data = json.loads(meta.read_text())
            owner, name = data.get("owner"), data.get("name") or pushed_name
            if owner and name:
                return f"{owner}/{name}"
        except Exception:
            pass
    # prime's success line, across phrasings ("Successfully pushed", "Pushed", "Published ...").
    match = re.search(
        r"(?:[Ss]uccessfully\s+)?(?:[Pp]ushed|[Pp]ublished)\s+([A-Za-z0-9][\w.-]*)/([\w.-]+)",
        output,
    )
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    # Owner named without an explicit env name (e.g. ".../<owner>/simple/" index URL in output):
    # pair it with the name we pushed, so a clearly-successful push still yields a usable slug.
    if pushed_name:
        owner_match = re.search(r"hub\.primeintellect\.ai/([A-Za-z0-9][\w.-]*)/", output)
        if owner_match:
            return f"{owner_match.group(1)}/{pushed_name}"
    return None


def _is_version_conflict(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _PUSH_CONFLICT_MARKERS)


def _prime_push(env_dir: Path, *, name: str, is_new: bool) -> str:
    """`prime env push` the package under the control plane's Prime account (PRIVATE), climbing
    past version conflicts. Returns the published slug. Raises EnvPublishError on failure."""
    if not os.environ.get("PRIME_API_KEY"):
        raise EnvPublishError(
            "this control plane has no PRIME_API_KEY configured, so it cannot publish to the "
            "managed Environments Hub",
            status=503,
        )
    if not shutil.which("prime"):
        raise EnvPublishError("the `prime` CLI is not installed on this control plane", status=503)
    base = [
        "prime",
        "env",
        "push",
        "--plain",
        "--path",
        str(env_dir),
        "--visibility",
        "PRIVATE",  # managed envs can hold proprietary task data
        "--name",
        name,
    ]
    env = {**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"}
    auto_bump = not is_new  # a re-publish must land on a fresh version
    last = ""
    for _ in range(_PUSH_MAX_ATTEMPTS):
        cmd = [*base, "--auto-bump"] if auto_bump else list(base)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        last = f"{proc.stdout or ''}{proc.stderr or ''}"
        if proc.returncode == 0:
            slug = _slug_from(env_dir, last, pushed_name=name)
            if not slug:
                # A clean exit but no discoverable owner/name: the env is on the Hub but we can't
                # name it. This is degenerate (prime writes .prime/.env-metadata.json on a real
                # success), so surface it as a server-side problem (502) — not a 400 that blames the
                # user's package for what was actually a successful publish.
                raise EnvPublishError(
                    "env published, but the control plane could not determine its owner/name id "
                    f"from prime's output: {last.strip()[:300]}",
                    status=502,
                )
            return slug
        if _is_version_conflict(last):
            auto_bump = True
            continue
        raise EnvPublishError(f"`prime env push` failed: {last.strip()[:500]}")
    raise EnvPublishError(
        f"`prime env push` failed after {_PUSH_MAX_ATTEMPTS} version-conflict retries"
    )


def publish_package(*, package_b64: str, name: str, is_new: bool, key: dict) -> str:
    """Decode + extract a client-built env package and publish it to the managed Prime account.

    Returns the published ``owner/name`` slug. The Hub env name is ``<identity>-<name>`` so each
    freesolo identity's envs are isolated under the one shared account, published PRIVATE.
    """
    if not name:
        raise EnvPublishError("missing env name")
    try:
        tar_bytes = base64.b64decode(package_b64, validate=True)
    except Exception as exc:
        raise EnvPublishError("env package is not valid base64") from exc
    if not tar_bytes:
        raise EnvPublishError("empty env package")
    hub_name = f"{namespace_for(key)}-{_sanitize_name(name)}"
    with tempfile.TemporaryDirectory(prefix="flash-env-publish-") as tmp:
        dest = Path(tmp)
        _safe_extract(tar_bytes, dest)
        if not (dest / "pyproject.toml").is_file():
            raise EnvPublishError("env package is missing a pyproject.toml")
        return _prime_push(dest, name=hub_name, is_new=is_new)
