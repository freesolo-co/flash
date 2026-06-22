"""Managed environment publishing.

`POST /v1/envs` accepts a packaged verifiers env — built client-side, so the user needs NO
separate environment account — and publishes it to FreeSolo's managed environment account.
Every env is therefore managed under one account: namespaced per freesolo identity (so users
can't collide) and published PRIVATE.
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
import tomllib
from pathlib import Path

_PUSH_MAX_ATTEMPTS = 8

# Namespace/name join delimiter. A `namespace_for` slug can NEVER contain ``--`` (its regex
# collapses any run of non-alphanumerics to a single ``-`` and strips leading/trailing ``-``), so
# joining with ``--`` makes namespaces NON-prefix-colliding: ``dev--x`` can't be mistaken for a
# member of ``dev-clado-ai``'s namespace. With a single ``-`` join, a short namespace (``dev``) is a
# raw-string prefix of a longer one (``dev-clado-ai``), which let one identity's re-publish hijack
# another's env (Cursor "idempotent env name prefix hijack"). The double dash is the boundary.
_NS_SEP = "--"


# Resource caps on an uploaded env package (DoS / tar-bomb defence). Generous for a real verifiers
# env (source + a modest bundled dataset), yet bound abuse.
#   * compressed upload: the .tar.gz we base64-decode into memory, rejected BEFORE a full decode.
#   * uncompressed total: the sum of member sizes a tar-bomb would expand to on disk.
#   * member count: archive complexity (many tiny members also exhaust inodes/CPU).
_MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # 64 MB compressed
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024  # 256 MB extracted
_MAX_MEMBERS = 5000

# Per-attempt wall clock for a single environment-publish subprocess: it shells out to a build
# frontend + several network round-trips, so a network stall / publisher outage / CLI deadlock would
# otherwise block the request handler thread forever. Each retry gets its OWN timeout and the attempt
# count is bounded (_PUSH_MAX_ATTEMPTS), so total wall time is bounded too. A timeout surfaces as a
# 504.
_PUSH_TIMEOUT_S = 180

# Allowlisted PEP 517 build backends. The publisher BUILDS A WHEEL LOCALLY on this control plane
# (`uv build --wheel` / `python -m build --wheel`, cwd = the extracted upload), which imports and runs
# the project's declared ``build-system.build-backend``. An untrusted ``pyproject.toml`` could name an
# arbitrary backend — or ship one in-tree via ``backend-path`` — giving ARBITRARY CODE EXECUTION on
# the control plane (with access to publisher credentials/filesystem) the moment the frontend imports
# it. So
# we reject any package whose build backend isn't one of the standard, well-known backends, and reject
# ``backend-path`` outright (it loads code FROM the untrusted upload). This is the standard set; flash's
# own client emits ``hatchling.build`` (see cli/main/envpush.py:_ENV_PUSH_PYPROJECT).
_ALLOWED_BUILD_BACKENDS = frozenset({
    "hatchling.build",
    "setuptools.build_meta",
    "setuptools.build_meta:__legacy__",
    "flit_core.buildapi",
    "poetry.core.masonry.api",
    "pdm.backend",
    "pdm.pep517.api",
    "maturin",
    "scikit_build_core.build",
})


def _human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f} MB"


class EnvPublishError(Exception):
    """A managed env publish failed. ``status`` is the HTTP status the route should return:
    400 for a bad client package, 413 when the upload exceeds a size cap, 502 when the publisher
    fails for a reason retrying can't fix, 503 when the control plane itself isn't configured to
    publish, 504 when the publisher times out."""

    def __init__(self, message: str, *, status: int = 400):
        super().__init__(message)
        self.status = status


def namespace_for(key: dict) -> str:
    """Stable prefix isolating one identity's envs under the shared FreeSolo managed account.

    It MUST be per-key distinct: the server DB stores the SAME placeholder email for every external
    key ("freesolo-user") and the internal key ("freesolo-internal") — see ``db.ensure_external_key``
    / ``ensure_internal_key`` — so namespacing on ``email`` would collapse every external user into
    one namespace and let them collide on env names. So we only trust ``email`` when it's a real,
    user-specific address (contains ``@``); otherwise we fall back to the per-key primary-key ``id``
    (each external token gets its own ``api_keys`` row), then a non-placeholder ``key_prefix``. The
    chosen identifier is stable across re-publishes, so a user's re-push bumps the SAME env."""
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
    # Published env names are lowercase, like the identity namespace they're joined to.
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-")
    return slug or "env"


def _safe_extract(tar_bytes: bytes, dest: Path) -> None:
    """Extract a .tar.gz into ``dest``, rejecting absolute paths and parent traversal.

    Defence in depth for an uploaded archive: besides path traversal, we explicitly gate member
    TYPES — only regular files and directories are allowed. Symlinks/hardlinks are rejected (a link
    could point outside ``dest``), and so are every other special member ``tarfile`` understands
    (device/block/char nodes, FIFOs): there's no legitimate reason an env package contains them, and
    extracting them would otherwise attempt to materialize them. We don't rely on tarfile's own
    extraction filter alone; the gate here is the contract.

    We also bound EXPANSION — a small .tar.gz can decompress to enormous data (a "tar bomb") and
    exhaust disk/CPU. We do NOT call ``getmembers()``/``extractall()`` (which fully scan and
    decompress the whole archive up front, before any cap could trip). Instead we STREAM members one
    at a time — iterating the ``TarFile`` reads one header at a time lazily — accumulating the member
    COUNT and cumulative UNCOMPRESSED size and aborting the moment either cap is exceeded, so we stop
    decompressing a bomb early. Each member is validated then extracted inline (its payload read
    exactly once, in order).
    """
    root = dest.resolve()
    # The bytes are an untrusted upload: non-tar / non-gzip / truncated input makes tarfile raise
    # TarError (incl. ReadError). Convert that to a 400 EnvPublishError so a malformed package is a
    # client error, not an uncaught 500. Our own EnvPublishError (raised in the loop below) is NOT a
    # TarError, so it propagates past this handler unchanged.
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            total = 0
            # enumerate over the TarFile object (NOT getmembers()): it yields one member at a time,
            # lazily reading a single header per step, so the caps below can trip before the whole
            # archive is scanned/decompressed.
            for count, member in enumerate(tar, start=1):
                if count > _MAX_MEMBERS:
                    raise EnvPublishError(
                        f"env package has too many members (limit {_MAX_MEMBERS}); refusing to "
                        f"extract a possible archive bomb"
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
                        f"env package is too large uncompressed (limit "
                        f"{_human_mb(_MAX_UNCOMPRESSED_BYTES)}); refusing to extract a possible "
                        f"archive bomb"
                    )
                tar.extract(member, dest)  # validated -> extract this member's payload now
    except tarfile.TarError as exc:
        raise EnvPublishError(f"env package is not a valid .tar.gz archive: {exc}") from exc
    except OSError as exc:
        # A malformed archive can still fail extraction at the filesystem layer — e.g. a file/dir
        # name collision (member "a" as a file, then "a/b") or an unwritable path. That's bad client
        # input, not a server fault, so surface it as a 400 rather than letting it bubble as a 500.
        raise EnvPublishError(f"env package could not be extracted: {exc}") from exc


def _slug_from(env_dir: Path, output: str, *, pushed_name: str | None = None) -> str | None:
    """The published ``owner/name`` slug, from publisher metadata or stdout.

    ``pushed_name`` is the ``--name`` we pushed under: if the publisher records the env ``owner`` but not a
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
    # Publisher success line, across phrasings ("Successfully pushed", "Pushed", "Published ...").
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


def _is_permanent_push_failure(output: str) -> bool:
    """Does this publish failure clearly NOT come from a version/content-hash conflict —
    i.e. one ``--auto-bump`` cannot fix (auth, permissions, not-found, build/test errors)?

    We match PERMANENT classes here, not conflict markers, on purpose: an INCOMPLETE list just
    falls back to the bounded retry (safe — at worst we retry a dead-end a few extra times),
    whereas an incomplete *conflict*-marker list would fail-fast a real, resolvable conflict. So
    this never makes a climbable conflict non-retriable; it only short-circuits the obvious
    dead-ends that would otherwise burn all _PUSH_MAX_ATTEMPTS attempts.

    The numeric HTTP codes (401/403/404) are matched only as an ACTUAL HTTP status — i.e. preceded by
    an "http"/"status"/"code" token (``HTTP 401``, ``status: 403``) — not as bare substrings. A
    conflict rejection embeds a content hash / version (e.g. "Wheel version with content hash a401f3…"
    or "version 1.404.0 already exists"), and a bare ``"404" in output`` (or even a ``\b``-bounded one)
    would spuriously flag the digits inside such a hash/version as a permanent failure and fail-fast a
    resolvable conflict. The textual markers (unauthorized/forbidden/not found/…) already cover the
    real auth/not-found cases on their own; the HTTP-status match is just a belt-and-suspenders."""
    o = output.lower()
    text_markers = (
        "unauthorized", "authentication", "invalid api key", "permission", "forbidden",
        "not found", "no such",
        "build failed", "test failed", "tests failed",
    )
    if any(m in o for m in text_markers):
        return True
    # A real HTTP status code, not digits embedded in a content hash / version string: require an
    # http/status/code lead-in and a word boundary after the code.
    return re.search(r"(?:http|status|code)[^\d]{0,4}(?:401|403|404)\b", o) is not None


def _assert_safe_build_backend(env_dir: Path) -> None:
    """Reject an uploaded project whose ``pyproject.toml`` would run UNTRUSTED code at build time.

    The publisher builds a wheel locally on this control plane (``uv build --wheel`` /
    ``python -m build --wheel`` with cwd = the extracted upload), and a
    PEP 517 build IMPORTS AND EXECUTES the project's declared ``build-system.build-backend``. With an
    untrusted ``pyproject.toml`` that's an arbitrary-code-execution surface on the control plane:

      * a custom ``build-backend`` (e.g. an attacker-shipped module) runs on import; and
      * ``backend-path`` makes the frontend load the backend FROM the upload's own files (an "in-tree
        backend"), so even a standard-looking ``build-backend`` name can resolve to attacker code.

    So we (a) reject any ``backend-path`` (it loads code from the untrusted upload), and (b) require
    ``build-backend`` to be one of the standard, well-known backends (``_ALLOWED_BUILD_BACKENDS``). A
    package with NO ``[build-system]`` is allowed: PEP 518 then defaults to setuptools (an allowlisted
    backend) — the same as declaring it explicitly. A malformed ``pyproject.toml`` is a client error.

    This bounds the RCE surface to vetted backends; the per-attempt subprocess timeout
    (``_PUSH_TIMEOUT_S``) separately bounds a HANG but does not, on its own, prevent code execution.
    """
    pyproject = env_dir / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise EnvPublishError(
            f"env package pyproject.toml must be valid UTF-8 text: {exc}"
        ) from exc
    except OSError as exc:
        raise EnvPublishError(f"env package pyproject.toml could not be read: {exc}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise EnvPublishError(f"env package pyproject.toml is not valid TOML: {exc}") from exc
    build_system = data.get("build-system")
    if build_system is None:
        return  # no [build-system] -> PEP 518 default (setuptools), which is allowlisted
    if not isinstance(build_system, dict):
        raise EnvPublishError("env package pyproject.toml has a malformed [build-system] table")
    # ``backend-path`` points the build frontend at a backend shipped INSIDE the untrusted upload, so
    # any value at all is an in-tree-backend RCE vector — refuse it outright.
    if build_system.get("backend-path") is not None:
        raise EnvPublishError(
            "env package declares a [build-system] backend-path (an in-tree build backend); this is "
            "not allowed for managed publishing because it would execute uploaded code on the "
            "control plane during the wheel build"
        )
    backend = build_system.get("build-backend")
    # Per PEP 518, an absent build-backend ALSO defaults to setuptools — allowed.
    if backend is None:
        return
    if not isinstance(backend, str) or backend not in _ALLOWED_BUILD_BACKENDS:
        allowed = ", ".join(sorted(_ALLOWED_BUILD_BACKENDS))
        raise EnvPublishError(
            f"env package declares an unsupported build-backend {backend!r}; managed publishing "
            f"builds the wheel on the control plane, so only standard build backends are allowed "
            f"({allowed})"
        )


def _prime_push(env_dir: Path, *, name: str, is_new: bool) -> str:
    """Publish the package under the control plane's managed account (PRIVATE), climbing
    past version conflicts. Returns the published slug. Raises EnvPublishError on failure.

    The publisher rejects a push whose (version, source-content-hash) already exists. Rather
    than parse its rejection message — it varies by version and phrasing (``Upload failed``,
    ``HTTP 400: Wheel version with content hash …``, ``version already exists`` …), so any literal
    marker list silently rots — we drive the retry off prime's **exit code** and let ``--auto-bump``
    resolve the version: it rewrites ``pyproject``'s version to the next patch *in place*, so each
    retry climbs to a fresh version until one is free. A brand-new env (``is_new``) publishes at its
    declared version on the first attempt; a re-publish climbs from the start. To avoid burning all
    attempts on a failure that bumping can't fix, we short-circuit clearly-PERMANENT failures
    (auth/not-found/build/test — see ``_is_permanent_push_failure``); matching permanent classes
    (rather than conflict markers) means an incomplete list degrades to a bounded retry instead of
    fail-fasting a real conflict. The non-retriable basics (no key / no CLI) are pre-checked below."""
    if not os.environ.get("PRIME_API_KEY"):
        raise EnvPublishError(
            "this control plane is missing managed environment publish credentials",
            status=503,
        )
    if not shutil.which("prime"):
        raise EnvPublishError("the managed environment publisher is not installed", status=503)
    # The env dir is unpacked from a CLIENT-uploaded tarball, so a preexisting ``.prime/`` (and its
    # ``.env-metadata.json``) is UNTRUSTED — a client could ship one naming another owner/name to
    # spoof the published slug. Remove it so the ONLY metadata we read back is what THIS push writes
    # under the control plane's own managed account, with the server-derived ``--name`` below. The
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
    last = ""
    for attempt in range(_PUSH_MAX_ATTEMPTS):
        # is_new starts at the declared version; otherwise (and on every retry) --auto-bump climbs.
        cmd = list(base) if (is_new and attempt == 0) else [*base, "--auto-bump"]
        # Bound each attempt: a hung publish (network stall / publisher outage / CLI deadlock)
        # would otherwise block the request handler thread forever. Each retry gets its own timeout
        # and the attempt count is bounded, so total wall time stays bounded. A timeout is a 504 (we
        # don't retry it — the attempt's --auto-bump may already have rewritten pyproject's version,
        # and a hang isn't a version conflict --auto-bump can resolve).
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=_PUSH_TIMEOUT_S
            )
        except subprocess.TimeoutExpired as exc:
            raise EnvPublishError(
                f"environment publish timed out after {_PUSH_TIMEOUT_S}s; aborting to keep the "
                f"control plane responsive",
                status=504,
            ) from exc
        last = f"{proc.stdout or ''}{proc.stderr or ''}"
        if proc.returncode == 0:
            slug = _slug_from(env_dir, last, pushed_name=name)
            if not slug:
                # A clean exit but no discoverable owner/name: the env is published but we can't
                # name it. This is degenerate (prime writes .prime/.env-metadata.json on a real
                # success), so surface it as a server-side problem (502) — not a 400 that blames the
                # user's package for what was actually a successful publish.
                raise EnvPublishError(
                    "env published, but the control plane could not determine its owner/name id "
                    f"from publisher output: {last.strip()[:300]}",
                    status=502,
                )
            return slug
        # Non-zero exit. Only a version/content-hash CONFLICT is worth retrying — `--auto-bump`
        # climbs to a free version. A clearly-permanent failure (auth, not-found, build/test) will
        # fail identically no matter how we bump, so fail fast with the real error instead of
        # burning every attempt (and pointlessly rewriting pyproject's version each time).
        if _is_permanent_push_failure(last):
            raise EnvPublishError(
                "environment publish failed and retrying will not help (not a version conflict): "
                f"{last.strip()[:500]}",
                status=502,
            )
    # Retry exhaustion is a server-side publish failure, not a bad client package — surface it
    # as 502, not the default 400 that would blame the user's upload.
    raise EnvPublishError(
        f"environment publish failed after {_PUSH_MAX_ATTEMPTS} attempts "
        f"(version conflict unresolved by --auto-bump): {last.strip()[:500]}",
        status=502,
    )


def publish_package(*, package_b64: str, name: str, is_new: bool, key: dict) -> str:
    """Decode + extract a client-built env package and publish it to the managed account.

    Returns the published ``owner/name`` slug. The published env name is ``<identity>--<name>`` so each
    freesolo identity's envs are isolated under the one shared account, published PRIVATE. Both the
    owner (the managed account) and the namespace are SERVER-controlled — never taken from the
    uploaded package — so one identity cannot publish into another's slug.
    """
    # The payload is arbitrary client JSON, so these can arrive as non-strings (e.g. {"name": 1});
    # validate types up front so malformed input is a clean 400, not a 500 from `.lower()`/`len()`.
    if not isinstance(name, str):
        raise EnvPublishError("env name must be a string")
    if not isinstance(package_b64, str):
        raise EnvPublishError("env package must be a base64 string")
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
    # Server-controlled namespacing isolates each identity's envs under the one managed account. We
    # join with ``_NS_SEP`` ("--"), which a namespace slug can never contain, so the
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
        # The publisher builds a wheel on THIS host, importing the project's build backend, so an
        # untrusted pyproject.toml is an RCE surface. Reject non-standard/in-tree build backends
        # before we ever hand the dir to the build frontend.
        _assert_safe_build_backend(dest)
        return _prime_push(dest, name=hub_name, is_new=is_new)
