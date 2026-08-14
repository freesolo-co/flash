"""Expanding DEFLATE payloads that carry no magic of their own.

Two shapes the magic-based recognition in `flash.env_formats` cannot reach: a headerless RFC 1951
stream, which has no header at all, and the compressed streams inside a PDF, which begin after an
object header rather than at byte zero. Both hold their credential nowhere a pattern can see, so a
file in either shape published intact while the same payload standing alone was expanded.

Split out to keep both modules under the file-size limit. The dependency runs one way: this knows
about bytes and formats, nothing about files, packages or the scan.
"""

from __future__ import annotations

import itertools
import re
import zlib
from collections.abc import Iterator

# Where a PDF keeps its compressed content, and how many of those streams are expanded. The
# `/FlateDecode` filter names the encoding, and the `stream` keyword with its mandatory newline
# marks where the zlib record begins. The gap between them is bounded because a real object
# dictionary is short -- a `/Length`, sometimes a `/DecodeParms`, little else -- and unbounded the
# pattern would pair a filter name with a `stream` keyword arbitrarily far away in a document.
_PDF_STREAM = re.compile(rb"/FlateDecode\b[\s\S]{0,512}?\bstream\r?\n")
_MAX_PDF_STREAMS = 4096


def _raw_deflate_payload(data: bytes, budget: int) -> bytes | None:
    """What ALL of `data` inflates to when it is one complete raw DEFLATE stream (RFC 1951).

    Empty means "not a raw deflate stream"; None means the stream is real but larger than `budget`,
    which is undecided rather than clean and the caller turns into a refusal.

    Raw deflate is deflate with no header at all -- what `zlib.compressobj(wbits=-15)` writes and
    what a `.deflate` sidecar carries. Having no header is exactly why it slipped through:
    recognition here is magic-based, and the zlib rule needs the two header bytes raw deflate does
    not have, so the compressed bytes were scanned as content and the credential inside published.

    With no magic to key on, the decode IS the recognition, and it has to cover the WHOLE input:
    the stream must inflate, reach its end marker, and leave no trailing bytes. That is what keeps
    this off ordinary binaries -- deflate is self-terminating, so arbitrary bytes end early or run
    out mid-symbol. Measured across 2,000 random buffers at four sizes and all 308 real binaries in
    `/usr/bin`: 0 acceptances. The weaker "inflates something" rule accepted 8 of 2,000, which is
    why completeness rather than output is the test.

    Anchored at offset zero and never searched for: a headerless format cannot be located by
    scanning, and trying every offset in a large file is both unbounded and meaningless.
    """
    if len(data) < 2:
        return b""
    try:
        inflate = zlib.decompressobj(-zlib.MAX_WBITS)
        plain = inflate.decompress(data, budget)
    except zlib.error:
        return b""
    if inflate.unconsumed_tail:
        return None
    return plain if plain and inflate.eof and not inflate.unused_data else b""


def _pdf_stream_payloads(data: bytes, budget: int) -> Iterator[bytes | None]:
    """What each `/FlateDecode` stream in a PDF inflates to, or None for one over `budget`.

    A PDF keeps its content in compressed streams whose zlib record begins after the object header
    rather than at byte zero, so the head-anchored zlib check never saw one and the appended-payload
    search covers only gzip, bzip2 and xz. A credential in a document published intact even though
    the same zlib record standing alone is expanded.

    Located by the format's own grammar -- the `%PDF-` signature, the filter name, and the `stream`
    keyword with its mandatory newline -- rather than by searching for zlib headers. Searching is
    what makes this unaffordable: that rule is about eleven bits, so it trips once per 2 KiB of
    arbitrary data, measured 44,197 candidates across 310 MB of real binaries of which 15 inflated.
    Feeding those through the appended-payload machinery would exhaust its bound and refuse every
    large binary. The grammar costs nothing on a non-PDF, which is every other file in a package.

    A stream that will not inflate is skipped rather than refused: `/FlateDecode` may name an
    encoding this cannot read, and the `stream` keyword appears in ordinary PDF text too.
    """
    if not data.startswith(b"%PDF-"):
        return
    for found in itertools.islice(_PDF_STREAM.finditer(data), _MAX_PDF_STREAMS):
        inflate = zlib.decompressobj()
        try:
            plain = inflate.decompress(data[found.end() :], budget)
        except zlib.error:
            continue
        yield None if inflate.unconsumed_tail else plain
