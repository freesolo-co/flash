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

# Namespace/name join delimiter. A `namespace_for` slug can NEVER contain ``--`` (its regex
# collapses any run of non-alphanumerics to a single ``-`` and strips leading/trailing ``-``), so
# joining with ``--`` makes namespaces NON-prefix-colliding: ``dev--x`` can't be mistaken for a
# member of ``dev-clado-ai``'s namespace. With a single ``-`` join, a short namespace (``dev``) is a
# raw-string prefix of a longer one (``dev-clado-ai``), which let one identity's re-publish hijack
# another's Hub env (Cursor "idempotent hub name prefix hijack"). The double dash is the boundary.
_NS_SEP = "--"


def _limit_bytes(env_var: str, default: int) -> int:
    """An operator-configurable byte cap from ``env_var`` (positive int), else ``default``."""
    raw = os.environ.get(env_var)
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return default


# Resource caps on an uploaded env package (DoS / tar-bomb defence). All operator-overridable.
#   * compressed upload: the .tar.gz we base64-decode into memory, rejected BEFORE a full decode.
#   * uncompressed total: the sum of member sizes a tar-bomb would expand to on disk.
#   * member count: archive complexity (many tiny members also exhaust inodes/CPU).
# Defaults are generous for a real verifiers env (source + a modest bundled dataset) yet bound abuse.
_MAX_UPLOAD_BYTES = _limit_bytes("FLASH_ENV_MAX_UPLOAD_BYTES", 64 * 1024 * 1024)  # 64 MB compressed
_MAX_UNCOMPRESSED_BYTES = _limit_bytes(
    "FLASH_ENV_MAX_UNCOMPRESSED_BYTES", 256 * 1024 * 1024
)  # 256 MB extracted
_MAX_MEMBERS = _limit_bytes("FLASH_ENV_MAX_MEMBERS", 5000)


def _human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


class EnvPublishError(Exception):
    """A managed env publish failed. ``status`` is the HTTP status the route should return:
    400 for a bad client package, 413 when the upload exceeds a size cap, 503 when the control plane
    itself isn't configured to publish (no PRIME_API_KEY / `prime` CLI)."""

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

    We also bound EXPANSION — a small .tar.gz can decompress to enormous data (a "tar bomb") and
    exhaust disk/CPU. So we cap the member COUNT and the cumulative UNCOMPRESSED size, summing
    declared member sizes as we iterate and aborting BEFORE ``extractall`` if either cap is exceeded
    (the declared sizes in the header are what ``extractall`` would write). Both caps are
    operator-configurable.
    """
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        total = 0
        for count, member in enumerate(tar.getmembers(), start=1):
            if count > _MAX_MEMBERS:
                raise EnvPublishError(
                    f"env package has too many members (limit {_MAX_MEMBERS}); refusing to extract a "
                    f"possible archive bomb"
                )
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
            total += max(0, member.size)
            if total > _MAX_UNCOMPRESSED_BYTES:
                raise EnvPublishError(
                    f"env package is too large uncompressed (limit {_human_mb(_MAX_UNCOMPRESSED_BYTES)}); "
                    f"refusing to extract a possible archive bomb"
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
    # The env dir is unpacked from a CLIENT-uploaded tarball, so a preexisting ``.prime/`` (and its
    # ``.env-metadata.json``) is UNTRUSTED — a client could ship one naming another owner/name to
    # spoof the published slug. Remove it so the ONLY metadata we read back is what THIS push writes
    # under the control plane's own Prime account, with the server-derived ``--name`` below. The
    # published owner is the authenticated managed account; the name is server-controlled (caller
    # namespace + sanitized name) — never client metadata.
    shutil.rmtree(env_dir / ".prime", ignore_errors=True)
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

    Returns the published ``owner/name`` slug. The Hub env name is ``<identity>--<name>`` so each
    freesolo identity's envs are isolated under the one shared account, published PRIVATE. Both the
    owner (the managed Prime account) and the namespace are SERVER-controlled — never taken from the
    uploaded package — so one identity cannot publish into another's slug.
    """
    if not name:
        raise EnvPublishError("missing env name")
    # Reject an oversized upload BEFORE decoding it all into memory. base64 expands ~4/3, so the
    # encoded string for an N-byte payload is ~ceil(N/3)*4 chars; cap the encoded length first (a
    # cheap len() on the already-received string — we can't avoid holding the request body, but we
    # avoid allocating the decoded bytes for an abusive payload), then re-check the decoded size.
    max_encoded = ((_MAX_UPLOAD_BYTES + 2) // 3) * 4 + 3
    if len(package_b64) > max_encoded:
        raise EnvPublishError(
            f"env package upload is too large (limit {_human_mb(_MAX_UPLOAD_BYTES)} compressed)",
            status=413,
        )
    try:
        tar_bytes = base64.b64decode(package_b64, validate=True)
    except Exception as exc:
        raise EnvPublishError("env package is not valid base64") from exc
    if not tar_bytes:
        raise EnvPublishError("empty env package")
    if len(tar_bytes) > _MAX_UPLOAD_BYTES:
        raise EnvPublishError(
            f"env package upload is too large (limit {_human_mb(_MAX_UPLOAD_BYTES)} compressed)",
            status=413,
        )
    ns = namespace_for(key)
    clean = _sanitize_name(name)
    # Server-controlled namespacing isolates each identity's envs under the one managed Prime
    # account. We join with ``_NS_SEP`` ("--"), which a namespace slug can never contain, so the
    # prefix test below is a true namespace-BOUNDARY check — a short namespace ("dev") can't match a
    # name that belongs to a longer one ("dev-clado-ai-...") and hijack another identity's env.
    # Idempotent on re-publish: a re-push passes the name part of the already-namespaced slug
    # ("<ns>--<name>", e.g. from a config's [environment] id), so we don't prefix it a second time
    # into "<ns>--<ns>--<name>" (a NEW env) — but ONLY when that prefix is THIS caller's namespace.
    prefix = f"{ns}{_NS_SEP}"
    hub_name = clean if clean == ns or clean.startswith(prefix) else f"{prefix}{clean}"
    with tempfile.TemporaryDirectory(prefix="flash-env-publish-") as tmp:
        dest = Path(tmp)
        _safe_extract(tar_bytes, dest)
        if not (dest / "pyproject.toml").is_file():
            raise EnvPublishError("env package is missing a pyproject.toml")
        return _prime_push(dest, name=hub_name, is_new=is_new)
