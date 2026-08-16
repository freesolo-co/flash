"""Safe display of archive names that carry credentials in reversible encodings."""

from __future__ import annotations

import re
from collections.abc import Callable

from flash.envscan.base64 import _match_base64
from flash.envscan.patterns import _ASSIGNED_PATTERNS, _LITERAL_PATTERNS, _TOKEN_PATTERNS

CredentialCheck = Callable[..., str | None]


def redacted_name(
    name: str,
    *,
    credential_check: CredentialCheck,
    deadline: float | None = None,
    refusals: tuple[type[BaseException], ...] = (),
) -> str:
    """`name` with any credential body masked, safe to print.

    the refusal names the member so the author can find it. when the credential is in that name,
    printing it verbatim leaks the key into terminals, http responses, and collected logs. a token
    body can be masked in place; a reversible encoding has to be withheld because it has no plaintext
    body to replace.
    """

    def mask(match: re.Match[bytes]) -> bytes:
        body = next((index for index, group in enumerate(match.groups(), 1) if group), None)
        if body is None:
            return match.group(0)
        return match.group(0)[: match.start(body) - match.start()] + b"***"

    # surrogatepass matches the name scanner. surrogateescape cannot encode a lone surrogate, and a
    # reporting crash would replace the validation message that was supposed to protect the publish.
    masked = name.encode("utf-8", "surrogatepass")
    for _kind, pattern in _TOKEN_PATTERNS + _ASSIGNED_PATTERNS:
        masked = pattern.sub(mask, masked)

    # literal structures and base64 values have no body that can be safely masked. the final check is
    # container-aware, so base64url(gzip(key)) is withheld just as it is refused by the name scanner.
    decoded = masked.decode("utf-8", "replace")
    try:
        encoded = _match_base64(masked) or credential_check(decoded, deadline=deadline) is not None
    except refusals:
        encoded = True
    if any(pattern.search(masked) for _kind, pattern in _LITERAL_PATTERNS) or encoded:
        parent = name.rsplit("/", 1)[0] if "/" in name else ""
        return (
            f"{parent}/<a file whose name encodes a credential>"
            if parent
            else "<a file whose name encodes a credential>"
        )
    return decoded
