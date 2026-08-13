"""Credential content scan for `flash env push`.

Filename filters cannot decide this question. `_ENV_PUSH_SECRET_PATTERNS` drops files *named* like
secret stores (`.env`, `*.pem`, `credentials*`), but the common convention of exporting keys from a
sourceable shell file -- `env.sh`, `setenv.sh`, `secrets.sh` -- is named like ordinary tooling, so a
plain `flash env push .` committed a live `FREESOLO_API_KEY` into the shared environment hub. A
published env repo is org-shared and its history is permanent, so the leak survives deleting the
file afterwards. Python source is exempt from the name filter entirely (so helper modules ship
instead of breaking the worker with ModuleNotFoundError), so a key pasted into a helper had nothing
between it and the hub either.

So the authoritative check reads what is about to be published and refuses on credential *shape*.
Patterns require an issuer prefix plus a long key body: `hf_[A-Za-z0-9]{20,}` is a token, while the
`hf_hub_download` in ordinary code is not, so a real environment still publishes untouched.

Split out of `flash.cli.commands.env.push` to keep that module under the file-size limit, and kept
free of any import from it so the dependency runs one way.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import os
import re
import zipfile
from pathlib import Path
from typing import IO

# Bodies are bounded rather than open-ended so a match has a maximum length, which is what lets
# `_SCAN_OVERLAP_BYTES` below be a real guarantee instead of a hope. The cap is far above every
# issued key format; a longer key still matches its first `_MAX_BODY` characters, which is a
# detection either way.
_MAX_BODY = 256

# (kind, pattern) for issued tokens: an issuer prefix plus a long key body, captured as a group.
# The kind names the credential in the refusal so the author knows which key to rotate; the matched
# text is NEVER echoed, since the error is printed and may reach a log.
#
# Patterns are BYTES: members are scanned as raw bytes so a credential stored inside a binary
# container (a sqlite state file, a pickle, an archive) is not skipped. Prefix-anchored patterns
# cannot realistically fire on random bytes -- matching `fslo_` alone is 256**-5 per position.
#
# No AWS entry. An `AKIA...` access key ID is a public identifier -- AWS puts it in signed URLs in
# the clear, so it turns up verbatim in any web-scraped dataset (it does, in the mbti training
# shards here) -- and the matching secret access key is 40 undifferentiated base64 characters with
# no prefix to anchor on. Matching the identifier would block real dataset publishes while still
# not catching the secret that actually matters.
_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("a Freesolo API key", re.compile(rb"fslo_([A-Za-z0-9_-]{16,%d})" % _MAX_BODY)),
    ("a Hugging Face token", re.compile(rb"hf_([A-Za-z0-9]{20,%d})" % _MAX_BODY)),
    (
        "a GitHub token",
        re.compile(
            rb"gh[pousr]_([A-Za-z0-9]{20,%d})|github_pat_([A-Za-z0-9_]{20,%d})"
            % (_MAX_BODY, _MAX_BODY)
        ),
    ),
    ("a Prime Intellect key", re.compile(rb"pit_([A-Za-z0-9]{16,%d})" % _MAX_BODY)),
    ("an Anthropic API key", re.compile(rb"sk-ant-([A-Za-z0-9_-]{20,%d})" % _MAX_BODY)),
    ("an OpenRouter API key", re.compile(rb"sk-or-v1-([A-Za-z0-9]{20,%d})" % _MAX_BODY)),
    # every currently-issued OpenAI family is named explicitly. `sk-svcacct-` and `sk-admin-` keys
    # carry project-wide and organization-wide authority, and neither is reachable through the bare
    # `sk-` branch below: the subtype's own hyphen ends that branch's alphanumeric run early.
    (
        "an OpenAI API key",
        re.compile(
            rb"sk-(?:proj|svcacct|admin)-([A-Za-z0-9_-]{20,%d})"
            # the bare legacy form requires a capital SOMEWHERE in the body, tested by lookahead
            # rather than by position. Demanding 31 more characters *after* the first capital
            # missed real keys: a legacy body carries `T3BlbkFJ` around index 20, leaving too few
            # behind it. The requirement is still what kills the false positive, since a
            # lowercase-hex body of the same length is a content hash, not a key --
            # `.../assets/sk-<32 hex>.js` is an ordinary CDN asset URL.
            rb"|sk-((?=[a-z0-9]*[A-Z])[A-Za-z0-9]{32,%d})" % (_MAX_BODY, _MAX_BODY)
        ),
    ),
    # `xapp-` is Slack's app-level token, a different prefix rather than another `xox` letter.
    ("a Slack token", re.compile(rb"(?:xox[baprs]|xapp)-([A-Za-z0-9-]{10,%d})" % _MAX_BODY)),
)

# A PEM header only means a key is present when the BODY follows it. The header alone is something
# documentation says -- "if you see -----BEGIN RSA PRIVATE KEY----- in a log, redact it" -- and
# refusing on it blocks a legitimate publish over prose about credentials. Requiring a base64 line
# after the header keeps every real key (they all carry one) and drops the mention of one.
_LITERAL_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "a private key block",
        re.compile(
            rb"-----BEGIN [A-Z ]*PRIVATE KEY-----[\r\n\s]*(?:[A-Za-z0-9+/=]{32,}|[A-Za-z-]+:)"
        ),
    ),
)

# Read in bounded chunks so a large dataset member is never held in memory whole. This costs no
# more I/O than the publish already pays: `_tar_b64` reads every one of these bytes to gzip them.
_SCAN_CHUNK_BYTES = 1 << 20
# Carried between chunks so a credential straddling a chunk boundary is still matched. Longer than
# the longest possible match (`_MAX_BODY` plus the longest prefix), so every match is fully visible
# inside some window rather than merely likely to be.
_SCAN_OVERLAP_BYTES = 1024

# Leading bytes of the compressed containers worth expanding. Only *compressed* ones are listed:
# inside a plain `.tar` the member bytes appear literally, so the ordinary scan already reads them,
# whereas a deflated zip member or a gzipped dataset shard does not contain its credential anywhere
# in the file. Detected by magic rather than by extension, since the extension is the publisher's
# choice and a renamed archive is still an archive.
_COMPRESSED_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00")
# Decompressed bytes scanned per file. Expansion is unbounded in principle, so it needs a stop. The
# cap is the publish's own package size limit: no legitimate member of a 256 MB package expands to
# more than the entire package, so nothing real is truncated, while a pathological archive cannot
# expand without end.
_MAX_DECOMPRESSED_BYTES = 256 << 20


def _is_high_entropy(body: bytes) -> bool:
    """Whether a key body looks issued rather than hand-written.

    An issued token is random, so across 16+ characters it is all but certain to carry a digit or a
    capital: for a 45-character Freesolo body the chance of neither is about 1 in 2,000,000. A
    hand-written placeholder is snake_case English -- `fslo_retry_after_close` -- and carries
    neither. So this separates the two without the length or dictionary heuristics that would start
    guessing about real keys.
    """
    return any(char.isdigit() or char.isupper() for char in body.decode("ascii", "ignore"))


def _match(data: bytes) -> str | None:
    """The kind of credential the literal bytes `data` contain, or None."""
    for kind, pattern in _LITERAL_PATTERNS:
        if pattern.search(data):
            return kind
    for kind, pattern in _TOKEN_PATTERNS:
        for match in pattern.finditer(data):
            # the alternations above put the body in whichever group matched; the rest are None.
            body = next((group for group in match.groups() if group), b"")
            if _is_high_entropy(body):
                return kind
    return None


def _credential_kind(data: bytes) -> str | None:
    """The kind of credential `data` contains under any of its plausible text encodings.

    A wide encoding interleaves NUL bytes between ASCII characters, so a UTF-16 `env.ps1` holding
    an otherwise-detected key matches none of the byte patterns and published intact. Rather than
    decode every member (most are not text at all), strip the NUL padding of each wide form and
    re-test: that is exact for the ASCII-range characters every one of these credentials is built
    from, and costs nothing on the ordinary UTF-8 file, which has no NULs to strip.
    """
    if kind := _match(data):
        return kind
    if b"\x00" not in data:
        return None
    for width, keep in ((2, (0, 1)), (4, (0, 3))):
        for offset in keep:
            # take every `width`-th byte: for UTF-16 that is the ASCII half of each code unit, in
            # whichever of the two byte orders the file used.
            narrowed = data[offset::width]
            if kind := _match(narrowed):
                return kind
    return None


def _scan_stream(handle: IO[bytes], *, limit: int) -> str | None:
    """The kind of credential in the first `limit` bytes of `handle`, or None.

    Read in chunks with an overlap so a credential straddling a chunk boundary is still matched.
    """
    carry = b""
    read = 0
    while read < limit and (chunk := handle.read(min(_SCAN_CHUNK_BYTES, limit - read))):
        read += len(chunk)
        window = carry + chunk
        if kind := _credential_kind(window):
            return kind
        carry = window[-_SCAN_OVERLAP_BYTES:]
    return None


def _credential_in_compressed(path: Path) -> str | None:
    """The kind of credential inside a compressed container, or None.

    A deflated zip member or a gzipped shard does not contain its credential anywhere in the file,
    so scanning the container's own bytes cannot see it -- and `flash env push` happily publishes a
    `.zip` holding an `env.sh`. Only one level is expanded, which is what the convention actually
    produces; a container nested inside a container is left to its own bytes.

    A container that will not open is not an error, and neither is one that fails partway through
    reading. An unsupported, truncated or corrupt archive falls back to the literal scan of its own
    bytes, which is the coverage it had before. Refusing to publish it, or crashing on it, would be
    a worse bug than the hole being closed: a half-written shard in a dataset directory is ordinary.
    """
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    with archive.open(info) as member:
                        if kind := _scan_stream(member, limit=_MAX_DECOMPRESSED_BYTES):
                            return kind
            return None
        with path.open("rb") as raw:
            header = raw.read(6)
        opener = {b"BZh": bz2.open, b"\xfd7zXZ\x00": lzma.open}.get(
            next((magic for magic in (b"BZh", b"\xfd7zXZ\x00") if header.startswith(magic)), b""),
            gzip.open,
        )
        with opener(path, "rb") as stream:
            return _scan_stream(stream, limit=_MAX_DECOMPRESSED_BYTES)
    except (OSError, EOFError, ValueError, zipfile.BadZipFile):
        return None


def credential_in_file(path: Path) -> str | None:
    """The kind of credential publishing `path` would leak, or None.

    Scanned as raw bytes, binary members included. Skipping binaries would be a hole rather than a
    saving: a credential sitting in a sqlite state file or a pickle is as published as one in a
    shell script, and the prefixes above cannot realistically collide with random bytes.
    """
    with path.open("rb") as handle:
        compressed = handle.read(6).startswith(_COMPRESSED_MAGIC)
        handle.seek(0)
        if kind := _scan_stream(handle, limit=_MAX_DECOMPRESSED_BYTES):
            return kind
    return _credential_in_compressed(path) if compressed else None


def credential_in_name(name: str) -> str | None:
    """The kind of credential the path `name` itself carries, or None.

    A file whose *name* is the key leaks it through the archive's member list even when its
    contents are empty, and the published repo shows that name in its tree forever.
    """
    return _credential_kind(name.encode("utf-8", "surrogateescape"))


def _redacted(name: str) -> str:
    """`name` with any credential body masked, safe to print.

    The refusal names the member so the author can find it, and when the credential is IN that name
    printing it verbatim re-leaks the key into a terminal and whatever collects its output -- the
    one thing the rest of this module is careful never to do. The issuer prefix and the surrounding
    path survive, which is all that is needed to locate the file.
    """

    def _mask(match: re.Match[bytes]) -> bytes:
        body = next((index for index, group in enumerate(match.groups(), 1) if group), None)
        if body is None:
            return match.group(0)
        return match.group(0)[: match.start(body) - match.start()] + b"***"

    masked = name.encode("utf-8", "surrogateescape")
    for _kind, pattern in _TOKEN_PATTERNS:
        masked = pattern.sub(_mask, masked)
    return masked.decode("utf-8", "replace")


def reject_credential_bearing_package(package_root: Path, *, display: dict[str, str]) -> None:
    """Refuse the publish if any member of the staged package carries a credential.

    Takes the STAGED package rather than the source tree, so what is scanned is exactly what is
    uploaded. Scanning the source instead left three holes: the generated README (which embeds
    `--name`, so a key passed there was published verbatim), the generated entrypoint alias, and
    the window between reading the source and copying it, in which any local process rewriting a
    file put unscanned bytes into the archive.

    Raises ValueError naming the member and the credential kind. Refusing rather than quietly
    dropping the file is deliberate: the author needs to rotate a key that has been sitting in a
    directory they just tried to publish, and a silent drop teaches them nothing.
    """
    # sorted so the member named in the refusal is the same one on every machine.
    for root, dirs, files in os.walk(package_root):
        dirs.sort()
        for name in sorted(dirs) + sorted(files):
            member = Path(root) / name
            relative = member.relative_to(package_root).as_posix()
            # the NAME is checked too, directories included: a file called `fslo_<key>.json`
            # publishes the key in the repository's file tree whatever its contents are.
            kind = credential_in_name(relative) or (
                credential_in_file(member) if member.is_file() else None
            )
            if not kind:
                continue
            raise ValueError(
                f"{_redacted(display.get(relative, relative))} contains what looks like {kind}. "
                "Publishing would commit it to a shared environment repository, permanently in "
                "git history. Remove the credential from the environment directory and rotate it "
                "before publishing."
            )
