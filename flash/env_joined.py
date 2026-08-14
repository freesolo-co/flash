"""Rejoining a credential that the file itself stores in pieces.

Two ways a source file holds a key that no contiguous run of its bytes contains: adjacent string
literals, which the language concatenates at parse time, and a backslash-newline continuation,
which the shell removes before the value is ever assigned. Either one splits a token across a seam
that is invisible to a pattern but absent by the time anything reads the value, so a key written
that way published intact while the same key on one line was refused.

Split out to keep `flash.env_secrets` under the file-size limit. The dependency runs one way: this
knows about bytes, nothing about files, packages or the scan.
"""

from __future__ import annotations

import re

# The seam between two adjacent string literals: a closing quote, whitespace that may cross one
# line break, then an opening quote of the SAME kind. Removing it welds the pair into the single
# string the language builds at runtime.
#
# Same quote character on both sides, and the closing one must not be escaped. A `", "` between two
# JSON array elements has the same shape as a concatenation seam, so joining indiscriminately would
# weld unrelated values into runs that decode to credentials nobody wrote. Requiring the quotes to
# match and the separator to be whitespace ONLY is what distinguishes `"a" "b"` -- which is one
# string -- from `"a", "b"`, which is two.
#
# At most one newline, so this joins a wrapped literal without welding two lines of a list that
# happen to sit under each other. Applied only after the ordinary literal pass has found nothing.
_ADJACENT_LITERALS = re.compile(rb"(?<!\\)([\"'])[ \t]*(?:\r?\n[ \t]*)?\1")

# A backslash immediately before a newline, which POSIX sh, make, C and YAML all remove to rejoin
# the line. An EVEN number of preceding backslashes means the backslash is itself escaped and the
# newline stands, so `"C:\\\\"` at the end of a line is not a continuation -- matching it would weld
# two unrelated lines together. The captured pairs are kept, so only the final backslash and the
# newline are removed.
_CONTINUED_LINE = re.compile(rb"(?<!\\)((?:\\\\)*)\\\r?\n")

# What a continuation looks like as a plain substring, for the guard below.
_CONTINUATIONS = (b"\\\n", b"\\\r\n")


def _rejoined(data: bytes) -> bytes:
    """`data` with both kinds of seam closed, or `data` itself when it holds neither.

    Returning the input unchanged is what the caller uses to skip the second match entirely, so an
    ordinary file pays two searches and no rematch.

    The continuation substitution is guarded by a plain substring test. Its pattern has to count
    preceding backslashes to tell a real continuation from an escaped one, and that prefix
    backtracks at every position: measured 350 ms per 8 MiB against 110 ms for the literal join,
    which on a 300 MiB expansion is most of the scan's budget and pushed a real test past its
    deadline. A backslash before a newline is absent from almost every file, and `bytes.__contains__`
    settles it at memchr speed.
    """
    joined = _ADJACENT_LITERALS.sub(b"", data)
    if any(marker in data for marker in _CONTINUATIONS):
        joined = _CONTINUED_LINE.sub(rb"\1", joined)
    return joined
