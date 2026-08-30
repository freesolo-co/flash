"""native string scans shared by tool declaration validation and the generated-output parser.

both scans run on untrusted request and generation text that reaches the megabytes, so each one
measures a run with a single native operation rather than stepping per character.
"""

from __future__ import annotations

import re

_LEADING_WHITESPACE_RE = re.compile(r"\s*")


def strings_overlap(left: str, right: str) -> bool:
    """whether either string contains the other or their ends and starts share a run."""

    if left in right or right in left:
        return True
    limit = min(len(left), len(right))
    return any(
        left[-size:] == right[:size] or right[-size:] == left[:size] for size in range(1, limit)
    )


def skip_whitespace(text: str, cursor: int) -> int:
    """offset of the first non-whitespace character at or after the cursor.

    a declaration reaches the megabytes and this runs on the request path, so stepping per
    character holds the event loop for seconds on a run of spaces. `\\s` accepts exactly what
    `str.isspace` accepts on every unicode codepoint, so the pattern measures the same run
    natively, and matching from the cursor never copies the tail to weigh it.
    """
    # a cursor past the end matches nothing and must come back unchanged, as stepping did.
    return _LEADING_WHITESPACE_RE.match(text, cursor).end() if cursor < len(text) else cursor
