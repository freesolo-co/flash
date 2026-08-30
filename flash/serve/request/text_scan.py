"""the tool-call grammar markers and the native string scans over them.

declaration validation and the generated-output parser both read this grammar, and both run on
untrusted text that reaches the megabytes, so each scan measures a run with a single native
operation rather than stepping per character. the markers live here rather than beside either
reader so the two can share them without importing each other.
"""

from __future__ import annotations

import re
from collections.abc import Container, Iterable

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
# the same pairing, capturing the opener's name so the caller can reject one its schema never
# declared. the name charset matches what a declaration may hold, so a name too long or carrying
# any other character cannot be declared and correctly fails to pair here.
_NAMED_PARAMETER_END_RE = re.compile(
    rf"{re.escape(PARAMETER_END)}\s*"
    rf"(?:{re.escape(PARAMETER_START)}([A-Za-z0-9_-]{{1,64}})>|{re.escape(FUNCTION_END)})"
)


def find_viable_parameter_end(text: str, cursor: int, declared: Container[str]) -> int:
    """offset of the first closer at or after ``cursor`` that could end a value, or -1.

    an opener the schema never declared cannot continue a call, so a closer followed by one is as
    inert as a closer followed by ordinary text. skipping those in the engine rather than returning
    each to the caller is what keeps a value quoting an undeclared opener a million times from
    costing a million caller passes.

    the names are filtered here instead of being spelled into the pattern because a declaration may
    carry hundreds of them: compiling that alternation costs more on one wide schema than every
    skip it saves, and the compilation is not work this parser can charge for.
    """
    while (found := _NAMED_PARAMETER_END_RE.search(text, cursor)) is not None:
        if found[1] is None or found[1] in declared:
            return found.start()
        # no closer can begin inside the pairing just rejected: the name charset excludes `<`, so
        # resuming at its end skips only spans that could not have started a match anyway.
        cursor = found.end()
    return -1


def overlaps_any(value: str, candidates: Iterable[str]) -> bool:
    """whether ``value`` contains a candidate, is contained by one, or shares a run with one.

    a per-pair form rebuilds a slice of ``value`` for every size against every candidate, so a
    declaration that contributes one marker per parameter turns a single stop into millions of
    slice allocations on the request path. the runs that can be shared depend only on ``value``,
    so they are enumerated once here and each candidate is then two native comparisons.

    a suffix longer than the candidate cannot prefix it, so passing the whole tuple decides the
    same predicate as bounding each pair by ``min(len(value), len(candidate))``.
    """
    if not value:
        # the empty string is a substring of everything, so any candidate at all overlaps it.
        return any(True for _ in candidates)
    suffixes = tuple(value[-size:] for size in range(1, len(value)))
    prefixes = tuple(value[:size] for size in range(1, len(value)))
    for candidate in candidates:
        if value in candidate or candidate in value:
            return True
        if suffixes and (candidate.startswith(suffixes) or candidate.endswith(prefixes)):
            return True
    return False


def skip_whitespace(text: str, cursor: int) -> int:
    """offset of the first non-whitespace character at or after the cursor.

    a declaration reaches the megabytes and this runs on the request path, so stepping per
    character holds the event loop for seconds on a run of spaces. `\\s` accepts exactly what
    `str.isspace` accepts on every unicode codepoint, so the pattern measures the same run
    natively, and matching from the cursor never copies the tail to weigh it.
    """
    # a cursor past the end matches nothing and must come back unchanged, as stepping did.
    return _LEADING_WHITESPACE_RE.match(text, cursor).end() if cursor < len(text) else cursor
