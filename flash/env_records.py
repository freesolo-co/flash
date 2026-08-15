"""Where one record of a multi-record stream ends, and how a two-marker pair is folded into it.

Split out of `flash.env_patterns` to keep that module under the file-size limit. It holds the JSON
record tokenizer and the per-record accumulation of a two-marker detector; the patterns themselves,
and the matching over them, stay there. The dependency runs one way: this module knows nothing
about which credentials exist, only about where a record begins and ends.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flash.env_patterns import _TwoMarkers

# Where one record of a multi-record file ends: a top-level `}` closing back to depth zero. Below
# the top level a `}` closes a nested member and does NOT end the record, which is what keeps a
# real JWK carrying nested extension objects -- or one wrapped in an enclosing document -- whole.
#
# Only the brace, never a newline. A newline reads like the separator of a line-delimited file, but
# it is redundant for JSONL, whose rows already close their own brace, and WRONG for the formats
# whose records legitimately span lines: a `.netrc` entry is `machine`, `login` and `password` on
# three separate lines, and splitting at the newline put its two halves in different records and
# published the key. Braces alone leave a brace-free format as one record, which is the behaviour
# those detectors already had.
#
# Strings are skipped whole so a brace INSIDE a quoted value cannot split a record --
# `{"note":"} ", "kty":"RSA", "d":"..."}` is one object, and treating its quoted `}` as the end
# would separate the halves of a genuine key and publish it. The escape alternative is what makes
# `"\\"` end the string and `"\""` not.
#
# The `*+` is possessive: without it an unterminated string backtracks over every position in the
# rest of the file, which on a megabyte chunk is the same quadratic cost the span between markers
# was removed for. Possessive quantifiers are available from Python 3.11, which is the floor this
# package already declares.
# `.` matches a newline here, so `\\.` consumes an escape whose second byte is one. Without that a
# backslash before a newline ended the string token early and the tokenizer carried on counting the
# braces AFTER it, which are inside the quoted value rather than around a record.
#
# The closing quote is captured rather than merely optional: whether the string was terminated is
# what the chunked scan needs, and a match that ends `\"` -- an ESCAPED quote -- ends with the same
# byte as a terminated one, so testing the last byte reported an open string as closed.
_JSON_RECORD_SPLIT = re.compile(rb'"(?:[^"\\]|\\.)*+(")?|(\{)|(\})', re.DOTALL)

# The remainder of a string that began in an earlier window. The chunked scan hands over one window
# at a time and a quoted value may straddle the cut, so resuming with the pattern above starts the
# quote phase half a string out: it read the filler's closing quote as an OPENING one, matched
# `"}\n{"` as a single string token, and swallowed the brace between two JSONL rows. Those rows
# merged into one record, and a public JWK in one paired with an ordinary high-entropy `d` in the
# other as a private key that neither row held -- on a file that scanned clean in a single buffer.
#
# Consuming the open string first is what puts the phase back: everything up to the terminating
# quote is the string's remainder, and tokenizing resumes from the byte after it.
_JSON_STRING_TAIL = re.compile(rb'(?:[^"\\]|\\.)*+(")?', re.DOTALL)

# Just the strings, for a window with no brace in it. The phase still has to advance across those
# bytes -- a quote left open here is what the NEXT window resumes inside -- but nothing else in
# them can move the depth, so the brace alternatives are left out.
_STRING_ONLY = re.compile(rb'"(?:[^"\\]|\\.)*+(")?', re.DOTALL)

# A string's body without its closing quote. Matched from the body's start to a point INSIDE the
# string, it says whether that point falls between the two bytes of an escape pair: the body is
# possessive and a lone trailing backslash matches neither alternative, so it stops short exactly
# when the byte after it is escaped.
_ESCAPE_BODY = re.compile(rb'(?:[^"\\]|\\.)*+', re.DOTALL)


class _RecordSplitter:
    """Where the top-level records of a stream end, tracked across the chunks it arrives in.

    Separate from the pairing so the tokenizing runs ONCE per window rather than once per detector.
    Each detector asking for its own boundaries cost 18 ms per megabyte per detector, which on the
    300 MiB expansion the padding test scans was 16 seconds of duplicated work and pushed that scan
    past its budget -- the scan then refused a file it had previously read to the end.
    """

    def __init__(self) -> None:
        # Depth is carried between windows so a record split across chunks is not read as two. It
        # starts at zero, which is also the depth of a brace-free format such as a `.netrc`: those
        # produce no boundaries at all and are one record, as they were before this existed.
        self.depth = 0
        # Whether the previous window ended INSIDE a quoted value. Both halves of the tokenizer's
        # state have to cross the boundary, not just the depth: resuming a straddling string with
        # the ordinary pattern read its closing quote as an opening one and mispaired every quote
        # after it, so `"}\n{"` matched as one string token and the brace between two JSONL rows
        # vanished. The rows merged, and a public JWK in one paired with an unrelated high-entropy
        # `d` in the other as a private key that neither row held.
        self.in_string = False
        # Whether the byte the next window STARTS on is the second half of an escape pair. A cut
        # falling between a backslash and the character it escapes left the next window reading
        # `\"` as a closing quote, which mispaired every quote after it exactly as a straddling
        # string did.
        self.escaped = False
        # The state as it stood where the NEXT window begins. Consecutive windows overlap so a
        # credential on a boundary is fully visible in one of them, which means the overlap is
        # tokenized twice -- and carrying the END state into a window that starts BEFORE it counted
        # those braces a second time. The depth never fell back to zero, so no record closed again
        # and every later row merged into one.
        self.resume = (0, False, False)
        self._pending = True
        # Where the currently-open string's BODY starts in the window being tokenized, so a resume
        # point inside it can be tested for a split escape pair.
        self._string_from = -1

    def ends(self, data: bytes, *, overlap: int = 0) -> list[int]:
        """The offsets in `data` just past each top-level record boundary.

        `overlap` is how many trailing bytes of `data` the next call will re-read. The state is
        rewound to that point on entry, so the shared bytes are tokenized twice but counted once.
        Boundaries inside the overlap are reported by both calls, which is what the pairing needs:
        a record ending there has to close in whichever window the caller is pairing over.

        Both halves of the tokenizer's state cross the boundary, not just the depth. A quoted value
        may straddle the cut, and resuming one with the ordinary pattern read its closing quote as
        an OPENING one: `"}\\n{"` matched as a single string token, the brace between two JSONL rows
        vanished, and the rows merged. A public JWK in one then paired with an unrelated
        high-entropy `d` in the other as a private key that neither row held, on a file that
        scanned clean in a single buffer.

        The full tokenizer is skipped when the window holds no brace at all; the quotes are still
        scanned, since an unterminated one is what the next window resumes inside. `bytes.find` is
        a memchr scan and the tokenizer is not, so this keeps the cost off the padding, the binary
        members and the prose that make up almost every byte actually scanned.
        """
        self.depth, self.in_string, self.escaped = self.resume
        resume_at = max(0, len(data) - overlap) if overlap else len(data)
        self._pending = True
        # A window resuming mid-string continues a body that began before it; one that resumes on
        # the second byte of an escape pair skips that byte, so the `"` in a split `\"` cannot be
        # read as the quote that closes the string.
        self._string_from = 1 if self.escaped and self.in_string else 0
        at = 0
        if self.in_string:
            # Consume the straddling string first, so the quote that ENDS it is not read as the
            # quote that starts another one. A window resuming on the second byte of an escape pair
            # skips it, so the `"` in a split `\"` cannot close the string.
            tail = _JSON_STRING_TAIL.match(data, self._string_from)
            self._mark(data, self._string_from, tail.end(), resume_at, in_string=True)
            if not tail.group(1):
                # Never terminated, so the whole window is inside that value and no brace in it can
                # be structural.
                self._settle(data, resume_at)
                return []
            at = tail.end()
            self.in_string = False
        found: list[int] = []
        pattern = _JSON_RECORD_SPLIT if b"{" in data or b"}" in data else _STRING_ONLY
        # The full tokenizer is skipped when the window holds no brace at all, but its quotes are
        # still scanned: one left open here is what the next window resumes inside.
        for token in pattern.finditer(data, at):
            brace = pattern is _JSON_RECORD_SPLIT and (token.group(2) or token.group(3))
            self._mark(data, token.start(), token.end(), resume_at, in_string=not brace)
            if not brace:
                self.in_string = not token.group(1)
                self._string_from = token.start() + 1
            elif token.group(2):
                self.depth += 1
            else:
                # Never below zero: a stray `}` in prose would otherwise leave the depth negative
                # and every later `{` would close a record early, splitting a real key in two.
                self.depth = max(0, self.depth - 1)
                if not self.depth:
                    found.append(token.end())
        self._settle(data, resume_at)
        return found

    def _mark(self, data: bytes, start: int, end: int, resume_at: int, *, in_string: bool) -> None:
        """Record the state at `resume_at`, given a token about to be applied that reaches past it.

        Called before the token is applied, so `self.depth` is still the depth in front of it. A
        token STARTING at or after `resume_at` leaves the state there as it is now; one SPANNING it
        can only be a string, since a brace is a single byte, and `resume_at` then falls inside a
        quoted value.
        """
        if not self._pending or end <= resume_at:
            return
        self._pending = False
        inside = in_string and start < resume_at
        # `start` is the token's start; the string's body begins one byte later. The open-string
        # remainder handled above has no opening quote in this window, so it passes its body start
        # directly and the two agree.
        body_from = start + 1 if end - start > 1 and data[start : start + 1] == b'"' else start
        self.resume = (self.depth, inside, inside and _escape_open(data, body_from, resume_at))

    def _settle(self, data: bytes, resume_at: int) -> None:
        """Take the resume state from the end of the window when no token reached past it.

        Every token ended before `resume_at`, so the state there is the state now -- with one
        exception: an UNTERMINATED string runs to the end of the window, and `resume_at` may fall
        inside its escape pair. `_JSON_STRING_TAIL` matches that remainder without a token of its
        own, which is why it is checked here rather than in `_mark`.
        """
        if not self._pending:
            return
        escaped = self.in_string and _escape_open(data, self._string_from, resume_at)
        self.resume = (self.depth, self.in_string, escaped)


def _escape_open(data: bytes, body_from: int, resume_at: int) -> bool:
    """Whether `resume_at` falls between a backslash and the character it escapes.

    `body_from` is where the open string's BODY starts, one byte past its opening quote. Matching
    the body up to `resume_at` stops short of a lone trailing backslash, since the escape
    alternative needs the byte after it -- so a short match is exactly the split-escape case.
    """
    if body_from < 0 or body_from >= resume_at:
        return False
    body = data[body_from:resume_at]
    return _ESCAPE_BODY.match(body).end() < len(body)


class _RecordHalves:
    """Pairs one detector's halves within a single record rather than across a whole file.

    A stream-wide pairing combined unrelated records: a JSONL dataset holding a PUBLIC JWK in one
    row and an ordinary high-entropy string under a private member name in another -- a build id,
    a timestamped artifact name -- has both halves present in the file and neither row holds a key.
    That refused a legitimate publish, and the entropy test cannot separate the two because a build
    id scores exactly as random as a key body does.

    Scoped by RECORD rather than by distance. A window would reintroduce the bug the stream-wide
    pairing exists to fix: JWK members may sit any distance apart, so a real key with a megabyte of
    metadata between `kty` and `d` must still pair, and it does here because that metadata is
    inside the same object. Only a top-level record boundary separates halves.

    State is carried between calls so the chunked scan can hand over one window at a time. A record
    straddling a chunk boundary keeps its seen halves; a record that ENDS inside the window clears
    them, which is what stops the next record inheriting the last one's markers.
    """

    def __init__(self, detector: _TwoMarkers) -> None:
        self.detector = detector
        self.context = False
        self.payload: re.Match[bytes] | None = None

    def paired(self, data: bytes, ends: list[int]) -> re.Match[bytes] | None:
        """The payload match of the first record in `data` holding BOTH halves, or None."""
        start = 0
        for boundary in ends:
            if found := self._absorb(data[start:boundary]):
                return found
            self.context = False
            self.payload = None
            start = boundary
        return self._absorb(data[start:])

    def _absorb(self, record: bytes) -> re.Match[bytes] | None:
        """Fold one record fragment into the halves seen so far, returning a completed pair."""
        if not record:
            return None
        self.context = self.context or bool(self.detector.context.search(record))
        self.payload = self.payload or self.detector.payload_match(record)
        return self.payload if self.context and self.payload else None
