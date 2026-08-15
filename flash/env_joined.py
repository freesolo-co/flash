"""Rejoining a credential that the file itself stores in pieces.

Three ways a source file holds a key that no contiguous run of its bytes contains: adjacent string
literals, which the language concatenates at parse time; a backslash-newline continuation, which
the shell removes before the value is ever assigned; and a JSON `\\uXXXX` escape, which the parser
resolves to the character it names. Each one splits a token across a seam that is invisible to a
pattern but absent by the time anything reads the value, so a key written that way published intact
while the same key on one line was refused.

Split out to keep `flash.env_secrets` under the file-size limit. The dependency runs one way: this
knows about bytes, nothing about files, packages or the scan.
"""

from __future__ import annotations

import re
import unicodedata

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
#
# The two quotes need not be the SAME character. Python concatenates `'fslo_AbCd'"Ef0123456789"`
# just as it does a matched pair, and requiring a backreference left that spelling split while the
# matched one was refused. What separates a concatenation from two list elements is the absence of
# anything between the quotes, not which quote character each side uses -- `", "` and `', '` are
# already excluded by the whitespace-only separator, and so is `","`.
#
# The second quote may carry a literal PREFIX. Python concatenates `b'fslo_AbCd' b'Ef0123456789'`
# and the `r`, `f`, `u`, `rb`, `br` forms exactly as it does a bare pair, so a seam this could not
# cross left the key split while the unprefixed spelling of the same value was refused. The prefix
# is consumed with the seam, which is what a parser does with it. Bounded to the two-character
# combinations the language defines, so an ordinary identifier ending in a quote cannot be eaten:
# `x = foo["a"]["b"]` has no whitespace-only seam between its quotes and is unaffected either way.
#
# A `#` COMMENT may sit in the seam too. Inside parentheses Python spans the concatenation across
# lines, and a comment before the newline is discarded by the tokenizer exactly as whitespace is:
# `('fslo_AbCd'  # prefix\n 'Ef0123456789')` evaluates to the whole key while the raw bytes are
# split by the comment text. Allowed only immediately before the single permitted newline, which is
# where a comment can legally end a line -- so the separator is still "nothing that survives
# parsing", and a `", "` between two list elements is as excluded as it was.
_ADJACENT_LITERALS = re.compile(
    rb"(?<!\\)[\"'][ \t]*(?:#[^\n]*)?(?:\r?\n[ \t]*)?(?i:[bruf]{0,2})[\"']"
)

# A backslash immediately before a newline, which POSIX sh, make, C and YAML all remove to rejoin
# the line. An EVEN number of preceding backslashes means the backslash is itself escaped and the
# newline stands, so `"C:\\\\"` at the end of a line is not a continuation -- matching it would weld
# two unrelated lines together. The captured pairs are kept, so only the final backslash and the
# newline are removed.
_CONTINUED_LINE = re.compile(rb"(?<!\\)((?:\\\\)*)\\\r?\n")

# What a continuation looks like as a plain substring, for the guard below.
_CONTINUATIONS = (b"\\\n", b"\\\r\n")

# A JSON `\uXXXX` escape naming a character the credential patterns would otherwise match. The
# parser resolves it, so `{"key":"fslo_AbCdEf..."}` carries the SAME key as the plain spelling
# and `json.loads` returns it verbatim -- while the raw bytes the patterns read are split by the
# escape and match nothing. Any single character of a key body can be written this way.
#
# Restricted to the ASCII range these credentials are built from. A general `\uXXXX` would have to
# decide an encoding for characters above 0x7F and would rewrite ordinary prose containing escaped
# accents, which no pattern here can match anyway -- so the narrow form does the whole job and
# cannot invent text. The surrogate range cannot appear alone in valid JSON and is excluded with it.
#
# Three spellings of the same character, because the formats an environment publishes are not only
# JSON. TOML and Python accept `\U000000XX` for the same code point, and Python and most config
# readers also accept `\xXX` -- `tomllib.loads('key="fslo_AbCd\\U00000045..."')` and a Python
# sidecar written with `\x45` both hand the complete credential to the runtime, while the raw bytes
# a pattern reads are split by the escape. All three are anchored to the same ASCII range, so none
# of them can invent a character outside what these patterns already match.
_JSON_ESCAPE = re.compile(rb"\\(?:u00|U000000|x)([0-7][0-9A-Fa-f])")

# The OCTAL spelling of the same character, which Python resolves in an ordinary string literal:
# `"fslo_a1\1052c3D4..."` evaluates to the complete key while the raw bytes a pattern reads are
# split by the escape. Exactly three octal digits, bounded to `[0-1][0-7][0-7]` so the value stays
# in the same 0x00-0x7F ASCII range as the hex forms above and cannot invent a character outside
# what these patterns already match.
#
# Three digits rather than one or two, deliberately. Python accepts `\1` and `\12` as well, but a
# lone `\0`-`\7` is one character of ordinary text away from any regex-quoted string and rewriting
# it would corrupt far more than it recovers -- while a key body written with a SHORT octal escape
# still leaves a long unbroken run on one side for the ordinary pass to match.
_OCTAL_ESCAPE = re.compile(rb"\\([0-1][0-7][0-7])")

# python also accepts a Unicode character NAME. `fslo_AbCd\N{LATIN CAPITAL LETTER E}f...` is the
# same string as its plain spelling, but the raw bytes split the shortest accepted key into runs too
# short to match. Only ASCII results are rejoined because credential bodies are ASCII; an unknown
# name, a non-ASCII result or a malformed name stays byte-for-byte untouched.
_NAMED_ESCAPE = re.compile(rb"\\N\{([^{}\r\n]+)\}")

# What those escapes look like as plain substrings, for the guard in `_rejoined`. The octal form
# has no distinctive prefix -- a backslash alone is its marker -- so it is guarded by the same
# `\\` test rather than a letter.
_ESCAPE_MARKERS = (b"\\u", b"\\U", b"\\x", b"\\N{")


def _named_ascii(match: re.Match[bytes]) -> bytes:
    """The ASCII text named by a Python `\\N{...}` escape, or the original escape unchanged."""
    try:
        resolved = unicodedata.lookup(match.group(1).decode("ascii"))
        return resolved.encode("ascii")
    except (KeyError, UnicodeDecodeError, UnicodeEncodeError):
        return match.group(0)


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
    # Guarded by the same substring test as the continuation, for the same reason: `\u` is absent
    # from almost every file, and settling that at memchr speed keeps the ordinary scan free of a
    # substitution pass over every byte.
    if any(marker in joined for marker in _ESCAPE_MARKERS):
        joined = _JSON_ESCAPE.sub(lambda point: bytes.fromhex(point.group(1).decode()), joined)
        joined = _NAMED_ESCAPE.sub(_named_ascii, joined)
    # Guarded by a bare backslash rather than a two-byte marker, since that is all an octal escape
    # has. Still a memchr-speed test, and still absent from most files.
    if b"\\" in joined:
        joined = _OCTAL_ESCAPE.sub(lambda point: bytes([int(point.group(1), 8)]), joined)
    return joined
