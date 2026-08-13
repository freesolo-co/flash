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

import os
import re
from pathlib import Path

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
    ("a Slack token", re.compile(rb"xox[baprs]-([A-Za-z0-9-]{10,%d})" % _MAX_BODY)),
)

# Credentials that are self-evident from their framing and need no key body to judge.
_LITERAL_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("a private key block", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)

# Read in bounded chunks so a large dataset member is never held in memory whole. This costs no
# more I/O than the publish already pays: `_tar_b64` reads every one of these bytes to gzip them.
_SCAN_CHUNK_BYTES = 1 << 20
# Carried between chunks so a credential straddling a chunk boundary is still matched. Longer than
# the longest possible match (`_MAX_BODY` plus the longest prefix), so every match is fully visible
# inside some window rather than merely likely to be.
_SCAN_OVERLAP_BYTES = 1024


def _is_high_entropy(body: bytes) -> bool:
    """Whether a key body looks issued rather than hand-written.

    An issued token is random, so across 16+ characters it is all but certain to carry a digit or a
    capital: for a 45-character Freesolo body the chance of neither is about 1 in 2,000,000. A
    hand-written placeholder is snake_case English -- `fslo_retry_after_close` -- and carries
    neither. So this separates the two without the length or dictionary heuristics that would start
    guessing about real keys.
    """
    return any(char.isdigit() or char.isupper() for char in body.decode("ascii", "ignore"))


def _credential_kind(data: bytes) -> str | None:
    """The kind of credential `data` contains, or None."""
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


def credential_in_file(path: Path) -> str | None:
    """The kind of credential publishing `path` would leak, or None.

    Scanned as raw bytes, binary members included. Skipping binaries would be a hole rather than a
    saving: a credential sitting in a sqlite state file or a pickle is as published as one in a
    shell script, and the prefixes above cannot realistically collide with random bytes.
    """
    carry = b""
    with path.open("rb") as handle:
        while chunk := handle.read(_SCAN_CHUNK_BYTES):
            window = carry + chunk
            if kind := _credential_kind(window):
                return kind
            carry = window[-_SCAN_OVERLAP_BYTES:]
    return None


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
    for root, dirs, files in os.walk(package_root):
        dirs.sort()
        for name in sorted(files):
            member = Path(root) / name
            kind = credential_in_file(member)
            if not kind:
                continue
            relative = member.relative_to(package_root).as_posix()
            raise ValueError(
                f"{display.get(relative, relative)} contains what looks like {kind}. "
                "Publishing would commit it to a shared environment repository, permanently in "
                "git history. Remove the credential from the environment directory and rotate it "
                "before publishing."
            )
