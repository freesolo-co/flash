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
_JSON_RECORD_SPLIT = re.compile(
    rb'(")(?:[^"\\]|\\.)*+(")?|(\')(?:[^\'\\]|\\.)*+(\')?|(\{)|(\})', re.DOTALL
)

# The remainder of a string that began in an earlier window. The chunked scan hands over one window
# at a time and a quoted value may straddle the cut, so resuming with the pattern above starts the
# quote phase half a string out: it read the filler's closing quote as an OPENING one, matched
# `"}\n{"` as a single string token, and swallowed the brace between two JSONL rows. Those rows
# merged into one record, and a public JWK in one paired with an ordinary high-entropy `d` in the
# other as a private key that neither row held -- on a file that scanned clean in a single buffer.
#
# Consuming the open string first is what puts the phase back: everything up to the terminating
# quote is the string's remainder, and tokenizing resumes from the byte after it. single quotes are
# included because Python-shaped mappings use the same brace syntax, and a brace in one is data too.
_JSON_STRING_TAIL = {
    ord('"'): re.compile(rb'(?:[^"\\]|\\.)*+(")?', re.DOTALL),
    ord("'"): re.compile(rb"(?:[^'\\]|\\.)*+(')?", re.DOTALL),
}

# Just the strings, for a window with no brace in it. The phase still has to advance across those
# bytes -- a quote left open here is what the next window resumes inside -- but nothing else in
# them can move the depth, so the brace alternatives are left out.
_STRING_ONLY = re.compile(rb'(")(?:[^"\\]|\\.)*+(")?|(\')(?:[^\'\\]|\\.)*+(\')?', re.DOTALL)

# A string's body without its closing quote. Matched from the body's start to a point inside the
# string, it says whether that point falls between the two bytes of an escape pair.
_ESCAPE_BODY = {
    ord('"'): re.compile(rb'(?:[^"\\]|\\.)*+', re.DOTALL),
    ord("'"): re.compile(rb"(?:[^'\\]|\\.)*+", re.DOTALL),
}


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
        # the delimiter of the open string. carrying only a boolean made a resumed single-quoted
        # value look for a double quote and exposed every brace behind it as structural.
        self.quote: int | None = None
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
        self.resume = (0, False, False, None)
        self._pending = True
        # Where the currently-open string's BODY starts in the window being tokenized, so a resume
        # point inside it can be tested for a split escape pair.
        self._string_from = -1

    def ends(self, data: bytes, *, overlap: int = 0) -> list[tuple[int, int]]:
        """Each offset in `data` just past a closing brace, with the depth left behind it.

        Depth 0 is a top-level record boundary; a deeper one closes a nested object, which the
        pairing uses to drop halves that belong to a subtree it has left.

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
        self.depth, self.in_string, self.escaped, self.quote = self.resume
        resume_at = max(0, len(data) - overlap) if overlap else len(data)
        self._pending = True
        # a window resuming mid-string continues a body that began before it; one that resumes on
        # the second byte of an escape pair skips that byte, so it cannot close the string.
        self._string_from = 1 if self.escaped and self.in_string else 0
        at = 0
        if self.in_string:
            # consume the straddling string first, using the delimiter that opened it in the prior
            # window. otherwise a single-quoted value waits for the next double quote and exposes
            # braces that are still inside its text.
            tail = _JSON_STRING_TAIL[self.quote].match(data, self._string_from)
            self._mark(
                data, self._string_from, tail.end(), resume_at, in_string=True, quote=self.quote
            )
            if not tail.group(1):
                self._settle(data, resume_at)
                return []
            at = tail.end()
            self.in_string = False
            self.quote = None
        found: list[tuple[int, int]] = []
        has_brace = b"{" in data or b"}" in data
        if not has_brace and b'"' not in data and b"'" not in data:
            # no token can change state in this window. skipping the regex keeps the added single-
            # quote alternative off the 300 mib zero padding path, where it otherwise added seconds.
            self._settle(data, resume_at)
            return found
        pattern = _JSON_RECORD_SPLIT if has_brace else _STRING_ONLY
        # the full tokenizer is skipped when the window holds no brace at all, but its quotes are
        # still scanned: one left open here is what the next window resumes inside.
        for token in pattern.finditer(data, at):
            opener = token.group(1) or token.group(3)
            closer = token.group(2) or token.group(4)
            brace = pattern is _JSON_RECORD_SPLIT and (token.group(5) or token.group(6))
            quote = opener[0] if opener else None
            self._mark(
                data, token.start(), token.end(), resume_at, in_string=not brace, quote=quote
            )
            if not brace:
                self.in_string = not closer
                self.quote = quote if self.in_string else None
                self._string_from = token.start() + 1
            elif token.group(5):
                self.depth += 1
                # an opening brace is reported too, with the depth it opens into. without it a
                # fragment spans several depths at once and cannot be credited to one object: in
                # `{"kty":"RSA","meta":{"x":1}` the `kty` is on the outer object while the brace
                # ending the fragment closes the inner one, and in `{"public":{"kty":"RSA"}` the
                # `kty` is on the inner object. splitting at every brace gives each piece one depth.
                found.append((token.end(), self.depth))
            else:
                # never below zero: a stray `}` in prose would otherwise leave the depth negative
                # and every later `{` would close a record early, splitting a real key in two.
                self.depth = max(0, self.depth - 1)
                # every close is reported, with the depth left behind it, not just the ones that
                # return to zero. a top-level boundary ends a record; a deeper one ends a subtree.
                found.append((token.end(), self.depth))
        self._settle(data, resume_at)
        return found

    def _mark(
        self,
        data: bytes,
        start: int,
        end: int,
        resume_at: int,
        *,
        in_string: bool,
        quote: int | None,
    ) -> None:
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
        body_from = start + 1 if end - start > 1 and data[start] in (34, 39) else start
        escaped = inside and _escape_open(data, body_from, resume_at, quote)
        self.resume = (self.depth, inside, escaped, quote if inside else None)

    def _settle(self, data: bytes, resume_at: int) -> None:
        """Take the resume state from the end of the window when no token reached past it.

        Every token ended before `resume_at`, so the state there is the state now -- with one
        exception: an UNTERMINATED string runs to the end of the window, and `resume_at` may fall
        inside its escape pair. `_JSON_STRING_TAIL` matches that remainder without a token of its
        own, which is why it is checked here rather than in `_mark`.
        """
        if not self._pending:
            return
        escaped = self.in_string and _escape_open(data, self._string_from, resume_at, self.quote)
        self.resume = (self.depth, self.in_string, escaped, self.quote)


def _escape_open(data: bytes, body_from: int, resume_at: int, quote: int | None) -> bool:
    """Whether `resume_at` falls between a backslash and the character it escapes.

    `body_from` is where the open string's BODY starts, one byte past its opening quote. Matching
    the body up to `resume_at` stops short of a lone trailing backslash, since the escape
    alternative needs the byte after it -- so a short match is exactly the split-escape case.
    """
    if body_from < 0 or body_from >= resume_at or quote is None:
        return False
    body = data[body_from:resume_at]
    return _ESCAPE_BODY[quote].match(body).end() < len(body)


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
        # Halves are held per DEPTH, not once for the whole record. A half seen at depth 2 belongs
        # to the object that is open there, and closing that object drops it -- which is what stops
        # two SIBLING objects, both inside one record, from lending each other their halves.
        self.context: set[int] = set()
        self.payload: dict[int, re.Match[bytes]] = {}
        self.depth = 0

    def paired(self, data: bytes, ends: list[tuple[int, int]]) -> re.Match[bytes] | None:
        """The payload match of the first OBJECT in `data` holding both halves, or None."""
        start = 0
        for boundary, depth in ends:
            # `ends` reports every brace, so a fragment lies wholly at one depth -- the one in
            # force before the brace that ends it. Its halves belong to the object open there.
            if found := self._absorb(data[start:boundary], self.depth):
                return found
            if depth < self.depth:
                # A CLOSING brace: everything seen strictly deeper goes out of scope with the
                # object it closed. Depth 0 clears everything, which is the top-level record
                # boundary the JSONL case relies on. An opening brace only descends and drops
                # nothing, since the object being entered is a child of what is already in scope.
                self.context = {at for at in self.context if at <= depth}
                self.payload = {at: hit for at, hit in self.payload.items() if at <= depth}
            self.depth = depth
            start = boundary
        return self._absorb(data[start:], self.depth)

    def _absorb(self, record: bytes, depth: int) -> re.Match[bytes] | None:
        """Fold one fragment into the halves seen so far, returning a completed pair.

        `depth` is the nesting level the fragment's own content sits at. A pair completes when the
        payload is at or below the object carrying the context marker -- so a key written
        `{"kty":"RSA","sub":{"d":"..."}}` still pairs, since an enclosing object is an ancestor of
        the one holding `d`. Two SIBLINGS share no ancestor below their parent, so neither can lend
        the other its half.
        """
        if not record:
            return None
        if self.detector.context.search(record):
            self.context.add(depth)
        if depth not in self.payload and (hit := self.detector.payload_match(record)):
            self.payload[depth] = hit
        if not self.context or not self.payload:
            return None
        # The shallowest context marker has the widest scope, and a payload at or below it is
        # inside the object that marker describes.
        context_at = min(self.context)
        for at in sorted(self.payload):
            if at >= context_at:
                return self.payload[at]
        return None
