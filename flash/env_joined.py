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
    rb"(?<!\\)[\"'][ \t]*(?:(?:#[^\n]*)?\r?\n[ \t]*)?(?i:[bruf]{0,2})[\"']"
)

# A backslash immediately before a newline, which POSIX sh, make, C and YAML all remove to rejoin
# the line. An EVEN number of preceding backslashes means the backslash is itself escaped and the
# newline stands, so `"C:\\\\"` at the end of a line is not a continuation -- matching it would weld
# two unrelated lines together. The captured pairs are kept, so only the final backslash and the
# newline are removed.
_CONTINUED_LINE = re.compile(rb"(?<!\\)((?:\\\\)*)\\\r?\n")

# What a continuation looks like as a plain substring, for the guard below.
_CONTINUATIONS = (b"\\\n", b"\\\r\n")

# the bytes the two joins need before either can change anything: a seam is made of quotes, and a
# shell assignment word is introduced by `=`. both guards below are plain substring tests for the
# same reason the continuation is one, and they are what keeps a quote-free padding block cheap.
_QUOTES = (b'"', b"'")
_ASSIGNMENT_SIGN = b"="

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

# a shell assignment word begins at a command-word boundary and carries one identifier before `=`.
# restricting quote removal to this context joins `KEY=fslo_ab"cd"` without welding quotes in prose,
# python statements, or adjacent top-level lines that the runtime never combines.
_SHELL_ASSIGNMENT = re.compile(rb"(?<![^\s;|&()])[A-Za-z_][A-Za-z0-9_]*=")

# at bracket depth zero, a bare command word followed by whitespace introduces shell argv rather
# than a source expression. preserving seams on that shape stops `printf "a" "b"` from becoming
# one value, while assignments, bracketed calls, and expression keywords retain literal joining.
_SHELL_COMMAND_START = re.compile(rb"[ \t]*(?P<word>[A-Za-z_][A-Za-z0-9_.-]*)[ \t]+")
_EXPRESSION_WORDS = frozenset((b"assert", b"await", b"raise", b"return", b"yield"))
_SHELL_ASSIGNMENT_COMMANDS = frozenset(
    (b"declare", b"env", b"export", b"local", b"readonly", b"typeset")
)

# python keeps escapes literal when a string prefix contains `r`. finding these spans before the
# adjacent-literal join matters because that join consumes the second prefix; doing it afterwards
# decoded `\\x42` inside a raw string and invented a credential that python never constructs.
_RAW_LITERAL_START = re.compile(
    rb"(?<![A-Za-z0-9_])(?i:(?:r[bf]?|[bf]r))(?P<quote>\"\"\"|'''|\"|')"
)


def _named_ascii(match: re.Match[bytes]) -> bytes:
    """The ASCII text named by a Python `\\N{...}` escape, or the original escape unchanged."""
    try:
        resolved = unicodedata.lookup(match.group(1).decode("ascii"))
        return resolved.encode("ascii")
    except (KeyError, UnicodeDecodeError, UnicodeEncodeError):
        return match.group(0)


def _raw_literal_spans(data: bytes) -> list[tuple[int, int]]:
    """The byte spans of Python string literals whose prefix contains `r`."""
    spans = []
    search_at = 0
    while match := _RAW_LITERAL_START.search(data, search_at):
        quote = match.group("quote")
        at = match.end()
        while True:
            end = data.find(quote, at)
            if end < 0:
                spans.append((match.start(), len(data)))
                return spans
            slash = end - 1
            while slash >= match.end() and data[slash] == 92:
                slash -= 1
            if (end - slash - 1) % 2 == 0:
                literal_end = end + len(quote)
                spans.append((match.start(), literal_end))
                search_at = literal_end
                break
            at = end + 1
    return spans


def _resolved_escapes(data: bytes) -> bytes:
    """Resolve supported runtime escapes in one span known not to be a raw string."""
    resolved = data
    if any(marker in resolved for marker in _ESCAPE_MARKERS):
        resolved = _JSON_ESCAPE.sub(lambda point: bytes.fromhex(point.group(1).decode()), resolved)
        resolved = _NAMED_ESCAPE.sub(_named_ascii, resolved)
    if b"\\" in resolved:
        resolved = _OCTAL_ESCAPE.sub(lambda point: bytes([int(point.group(1), 8)]), resolved)
    return resolved


def _decode_runtime_escapes(data: bytes) -> bytes:
    """Resolve escapes outside Python raw strings while leaving every raw byte scannable."""
    spans = _raw_literal_spans(data)
    if not spans:
        return _resolved_escapes(data)
    out = bytearray()
    at = 0
    for start, end in spans:
        out += _resolved_escapes(data[at:start])
        out += data[start:end]
        at = end
    out += _resolved_escapes(data[at:])
    return bytes(out)


def _seam_depths(data: bytes, starts: set[int]) -> dict[int, int]:
    """The bracket depth at each candidate literal seam, ignoring strings and comments."""
    depths: dict[int, int] = {}
    depth = 0
    quote: int | None = None
    triple = False
    comment = False
    at = 0
    while at < len(data):
        if at in starts:
            depths[at] = depth
        byte = data[at]
        if comment:
            if byte in (10, 13):
                comment = False
            at += 1
            continue
        if quote is not None:
            if triple and data[at : at + 3] == bytes([quote]) * 3:
                quote, triple, at = None, False, at + 3
                continue
            if not triple and byte == quote:
                quote = None
                at += 1
                continue
            # a backslash protects the next quote from ending the literal. skipping the pair also
            # keeps brackets inside an escape from changing the surrounding expression's depth.
            at += 2 if byte == 92 and at + 1 < len(data) else 1
            continue
        if byte == 35:
            comment = True
        elif byte in (34, 39):
            quote = byte
            triple = data[at : at + 3] == bytes([byte]) * 3
            if triple:
                at += 3
                continue
        elif byte in (40, 91, 123):
            depth += 1
        elif byte in (41, 93, 125):
            depth = max(0, depth - 1)
        at += 1
    return depths


def _is_shell_word_seam(data: bytes, at: int, depth: int) -> bool:
    """Whether the seam at `at` separates shell words rather than source literals."""
    if depth:
        return False
    line_start = data.rfind(b"\n", 0, at) + 1
    prefix = data[line_start:at]
    command = _SHELL_COMMAND_START.match(prefix)
    if not command or command.group("word") in _EXPRESSION_WORDS:
        return False
    # an equals sign normally makes this a source assignment. shell declaration commands remain
    # commands when later words assign values, so joining their quoted argv would still invent text.
    return b"=" not in prefix or command.group("word") in _SHELL_ASSIGNMENT_COMMANDS


def _join_adjacent_literals(data: bytes) -> bytes:
    """Close source-literal seams without welding separately quoted shell words."""
    matches = list(_ADJACENT_LITERALS.finditer(data))
    if not matches:
        return data
    depths = _seam_depths(data, {match.start() for match in matches})
    out = bytearray()
    at = 0
    for match in matches:
        out += data[at : match.start()]
        seam = match.group(0)
        depth = depths.get(match.start(), 0)
        if (b"\n" in seam and depth == 0) or _is_shell_word_seam(data, match.start(), depth):
            out += seam
        at = match.end()
    out += data[at:]
    return bytes(out)


def _join_shell_assignments(data: bytes) -> bytes:
    """Remove balanced quote boundaries inside shell assignment words only."""
    out = bytearray()
    copied = 0
    search_at = 0
    while match := _SHELL_ASSIGNMENT.search(data, search_at):
        at = match.end()
        quote: int | None = None
        value = bytearray()
        changed = False
        while at < len(data):
            byte = data[at]
            if quote is None and byte in b" \t\r\n\v\f":
                break
            if byte == 92 and at + 1 < len(data):
                value += data[at : at + 2]
                at += 2
                continue
            if byte in (34, 39) and (quote is None or quote == byte):
                quote = byte if quote is None else None
                changed = True
                at += 1
                continue
            value.append(byte)
            at += 1
        if changed and quote is None:
            out += data[copied : match.end()] + value
            copied = at
        search_at = max(at, match.end())
    if copied == 0:
        return data
    out += data[copied:]
    return bytes(out)


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

    the two joins are guarded the same way, and for the same measured reason. neither can change a
    byte without a quote to close: the literal join costs 101 ms per 8 mib and the assignment join
    114 ms, which over a 300 mib expansion is 3.79s and 4.28s spent proving a padding block holds no
    quote. running them unguarded took that expansion from 41s to 53s against a 60s budget, and under
    `pytest -n 2` in ci the scan missed its deadline and reported a real key as clean. an assignment
    additionally needs its `=`, so it is tested for both.
    """
    joined = data
    if any(marker in joined for marker in _CONTINUATIONS):
        joined = _CONTINUED_LINE.sub(rb"\1", joined)
    # escape decoding precedes literal joining because the join consumes a following `r` prefix.
    # keeping the prefix visible prevents a raw `\\x42` from becoming `B`, while the later join still
    # combines adjacent raw literals that contain a real credential verbatim.
    if b"\\" in joined:
        joined = _decode_runtime_escapes(joined)
    quoted = any(quote in joined for quote in _QUOTES)
    if quoted:
        joined = _join_adjacent_literals(joined)
    if quoted and _ASSIGNMENT_SIGN in joined:
        joined = _join_shell_assignments(joined)
    return joined
