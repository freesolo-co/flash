"""the tool-call grammar markers and the native string scans over them.

declaration validation and the generated-output parser both read this grammar, and both run on
untrusted text that reaches the megabytes, so each scan measures a run with a single native
operation rather than stepping per character. the markers live here rather than beside either
reader so the two can share them without importing each other.
"""

from __future__ import annotations

import re

TOOL_CALL_START, TOOL_CALL_END = "<tool_call>", "</tool_call>"
FUNCTION_START, FUNCTION_END = "<function=", "</function>"
PARAMETER_START, PARAMETER_END = "<parameter=", "</parameter>"
_LEADING_WHITESPACE_RE = re.compile(r"\s*")
# a ``</parameter>`` that could actually close a value: only one followed, after whitespace, by the
# next parameter or the function end can, and every other one is inert. searching for the pair lets
# the engine skip the inert ones natively instead of stepping to each in python.
VIABLE_PARAMETER_END_RE = re.compile(
    rf"{re.escape(PARAMETER_END)}\s*(?:{re.escape(PARAMETER_START)}|{re.escape(FUNCTION_END)})"
)


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
