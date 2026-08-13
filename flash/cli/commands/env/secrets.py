"""Credential content scan for `flash env push`.

Filename filters cannot decide this question. `_ENV_PUSH_SECRET_PATTERNS` drops files *named* like
secret stores (`.env`, `*.pem`, `credentials*`), but the common convention of exporting keys from a
sourceable shell file -- `env.sh`, `setenv.sh`, `secrets.sh` -- is named like ordinary tooling, so a
plain `flash env push .` committed a live `FREESOLO_API_KEY` into the shared environment hub. A
published env repo is org-shared and its history is permanent, so the leak survives deleting the
file afterwards.

So the authoritative check reads what is about to be published and refuses on credential *shape*.
Patterns require an issuer prefix plus a long key body: `hf_[A-Za-z0-9]{20,}` is a token, while the
`hf_hub_download` in ordinary code is not, so a real environment still publishes untouched.

Split out of `flash.cli.commands.env.push` to keep that module under the file-size limit, and kept
free of any import from it so the dependency runs one way.
"""

from __future__ import annotations

import re
from pathlib import Path

# (kind, pattern) for issued tokens: an issuer prefix plus a long key body, where group 1 is the
# body. The kind names the credential in the refusal so the author knows which key to rotate; the
# matched text is NEVER echoed, since the error is printed and may reach a log.
#
# No AWS entry. An `AKIA...` access key ID is a public identifier -- AWS puts it in signed URLs in
# the clear, so it turns up verbatim in any web-scraped dataset (it does, in the mbti training
# shards here) -- and the matching secret access key is 40 undifferentiated base64 characters with
# no prefix to anchor on. Matching the identifier would block real dataset publishes while still
# not catching the secret that actually matters.
_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a Freesolo API key", re.compile(r"fslo_([A-Za-z0-9_-]{16,})")),
    ("a Hugging Face token", re.compile(r"hf_([A-Za-z0-9]{20,})")),
    ("a GitHub token", re.compile(r"gh[pousr]_([A-Za-z0-9]{20,})|github_pat_([A-Za-z0-9_]{20,})")),
    ("a Prime Intellect key", re.compile(r"pit_([A-Za-z0-9]{16,})")),
    ("an Anthropic API key", re.compile(r"sk-ant-([A-Za-z0-9_-]{20,})")),
    ("an OpenRouter API key", re.compile(r"sk-or-v1-([A-Za-z0-9]{20,})")),
    ("an OpenAI API key", re.compile(r"sk-proj-([A-Za-z0-9_-]{20,})|sk-([A-Za-z0-9]{32,})")),
    ("a Slack token", re.compile(r"xox[baprs]-([A-Za-z0-9-]{10,})")),
)

# Credentials that are self-evident from their framing and need no key body to judge.
_LITERAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


def _is_high_entropy(body: str) -> bool:
    """Whether a key body looks issued rather than hand-written.

    An issued token is random, so across 16+ characters it is all but certain to carry a digit or a
    capital: for a 45-character Freesolo body the chance of neither is about 1 in 2,000,000. A
    hand-written placeholder is snake_case English -- `fslo_retry_after_close` -- and carries
    neither. So this separates the two without the length or dictionary heuristics that would start
    guessing about real keys.
    """
    return any(char.isdigit() or char.isupper() for char in body)


# Read in bounded chunks so a large dataset member cannot be held in memory whole. This costs no
# more I/O than the publish already pays: `_tar_b64` reads every one of these bytes to gzip them.
_SCAN_CHUNK_BYTES = 1 << 20
# Carried between chunks so a credential straddling a chunk boundary is still matched. Longer than
# any pattern above can match, which is what makes the overlap sufficient rather than merely likely.
_SCAN_OVERLAP_BYTES = 512


def _credential_kind(text: str) -> str | None:
    """The kind of credential `text` contains, or None."""
    for kind, pattern in _LITERAL_PATTERNS:
        if pattern.search(text):
            return kind
    for kind, pattern in _TOKEN_PATTERNS:
        for match in pattern.finditer(text):
            # the alternations above put the body in whichever group matched; the rest are None.
            body = next((group for group in match.groups() if group), "")
            if _is_high_entropy(body):
                return kind
    return None


def credential_in_file(path: Path) -> str | None:
    """The kind of credential published `path` would leak, or None.

    Binary members are skipped on a NUL byte in the first chunk: credentials are not kept in
    compressed dataset shards or images, and scanning random bytes is the one way this check
    could refuse a publish that carries no credential at all.
    """
    carry = ""
    with path.open("rb") as handle:
        first = True
        while chunk := handle.read(_SCAN_CHUNK_BYTES):
            if first:
                if b"\x00" in chunk:
                    return None
                first = False
            # errors="ignore" keeps a mixed-encoding file scannable: an undecodable byte is not a
            # credential character, so dropping it cannot hide one of the ASCII patterns above.
            text = carry + chunk.decode("utf-8", errors="ignore")
            if kind := _credential_kind(text):
                return kind
            carry = text[-_SCAN_OVERLAP_BYTES:]
    return None


def reject_credential_bearing_files(files) -> None:
    """Refuse the publish if any file about to be published carries a credential.

    Raises ValueError naming the file and the credential kind. Refusing rather than quietly
    dropping the file is deliberate: the author needs to rotate a key that has been sitting in a
    directory they just tried to publish, and a silent drop teaches them nothing.
    """
    for path, relative in files:
        kind = credential_in_file(path)
        if kind:
            raise ValueError(
                f"{Path(relative).as_posix()} contains what looks like {kind}. "
                "Publishing would commit it to a shared environment repository, permanently in "
                "git history. Remove the file from the environment directory and rotate the "
                "credential before publishing."
            )
