"""Member-by-member scanning of the two archive formats that carry other files.

A zip and a tar both hold members whose bytes are not the archive's own, so each member has to be
handed back to the scanner as if it were a file: a COMPRESSED member holds a credential nowhere the
archive's literal bytes can show it. Both formats share one rule -- an unreadable member is recorded
and raised only after the rest of the archive has been scanned, so a credential found further in
wins over the weaker "cannot read" answer, while an archive nothing could be read from is refused
rather than approved.

Split from `flash.env_secrets` to keep that module under the file-size limit. The scanner and its
refusal are passed in rather than imported, so the dependency still runs one way: this knows about
archive structure, and what a member's bytes MEAN stays with the caller.
"""

from __future__ import annotations

import io
import lzma
import tarfile
import time
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import IO

from flash.env_formats import _read_at, _zip_end_offset, _zip_member_count

# What "this member cannot be read" looks like, across both formats and every codec underneath them.
#
# Shallow corruption is a trap when testing this: truncating near the end of an xz stream raises
# EOFError, which was already caught, so the bug looks absent. The distinct error only appears when
# the damage is deep enough that the decompressor rejects the data rather than running out of it.
_UNREADABLE_MEMBER = (
    OSError,
    EOFError,
    ValueError,
    RuntimeError,
    NotImplementedError,
    zipfile.BadZipFile,
    # `TarError` inherits straight from `Exception`, so nothing else in this tuple covers it -- a
    # truncated tar (`ReadError: unexpected end of data`) crashed the publish outright. A
    # half-written shard in a dataset directory is ordinary, and crashing on it would be a worse
    # bug than the hole being closed.
    tarfile.TarError,
    # The two codec errors that also inherit straight from `Exception`. A member whose deflate or
    # xz stream is damaged deep enough raises one of these, and uncaught it escaped the member loop
    # entirely: the dispatcher read that as "not a zip after all" and moved on, so a corrupt FIRST
    # member hid a perfectly readable second member holding the key. Recorded like every other
    # unreadable member instead, which refuses rather than approves.
    zlib.error,
    lzma.LZMAError,
)

# What the caller hands in: a scanner for one member's bytes, and the refusal it raises. Taking them
# as arguments is what keeps the import one-way -- the alternative is importing the scan module,
# which imports this one.
Scanner = Callable[[IO[bytes], float, int], str | None]
Namer = Callable[[str], str | None]


def credential_in_zip(
    source: Path | bytes,
    *,
    deadline: float,
    depth: int,
    scan: Scanner,
    refusal: type[Exception],
    named: Namer,
    member_limit: int,
) -> str | None:
    """The kind of credential in any readable member of a zip, or None."""
    # The member count is read from the end-of-central-directory record BEFORE `ZipFile` is
    # constructed. `ZipFile.__init__` parses the whole central directory and materializes every
    # `ZipInfo`, so a bound checked after it is charged the cost it exists to avoid -- measured at
    # 1.8 seconds and 239 MB of resident memory for 400,000 empty entries in a 35 MB file, all of
    # it spent before the per-member loop below ran once.
    if _zip_member_count(source, member_limit) > member_limit:
        raise refusal("contains an archive with too many members to inspect")
    unreadable = ""
    with zipfile.ZipFile(source if isinstance(source, Path) else io.BytesIO(source)) as archive:
        for count, info in enumerate(archive.infolist(), 1):
            if count > member_limit:
                raise refusal("contains an archive with too many members to inspect")
            if time.monotonic() > deadline:
                raise refusal("takes too long to decompress")
            # the member NAME is checked too, exactly as the tar walk checks it: a zip entry called
            # `fslo_<key>.json` publishes the key in the archive's listing whatever its contents
            # are, and a name that is itself an encoded container is refused rather than decoded
            # speculatively -- the raw scan over the archive's own bytes swallows that refusal,
            # so a member named with base64 of an OpenSSL-encrypted file returned clean here while
            # the same name passed to the name scanner refused it.
            #
            # Checked BEFORE the directory filter, as in the tar walk. A directory entry has no
            # contents to scan but its name is in the central directory exactly as a file's is, and
            # `ZipInfo.is_dir` is the trailing `/` alone -- which also stops the raw pass from
            # recovering the encoded stream, since the slash lands inside the base64 run. A
            # directory named with base64 of a gzipped key returned clean while the name scanner
            # refused that same string on its own.
            if kind := named(info.filename.rstrip("/")):
                return kind
            if info.is_dir():
                continue
            # Bit 0 of the general-purpose flags marks an encrypted member. Its bytes cannot be
            # read without the password, so treating it as clean approved a package whose only copy
            # of a credential was inside -- `zip -P` around a key file published intact.
            #
            # Noted and raised AFTER the loop rather than here, because the members behind it must
            # still be scanned: refusing on sight would abandon the rest of the archive, and a
            # credential further down would go unreported in favour of a weaker message about an
            # unreadable member. A found credential is the more specific answer, so it wins. The
            # same deferral covers members whose compression this Python cannot decode, below.
            if info.flag_bits & 0x01:
                unreadable = unreadable or "an encrypted archive member this check cannot read"
                continue
            try:
                with archive.open(info) as member:
                    if kind := scan(member, deadline, depth):
                        return kind
            except NotImplementedError:
                # The member is valid but uses a compression method this Python has no decompressor
                # for -- Deflate64 is the common one, and `zip -fd` writes it. Caught by the broad
                # skip below it read as "opaque member, carry on" and the credential in its payload
                # published. Recorded like an encrypted member: unverifiable is not clean.
                #
                # Reported distinctly from encryption because the remedy differs: a password is not
                # what is missing, the archive has to be rewritten with a supported method.
                unreadable = unreadable or (
                    "an archive member compressed in a way this check cannot read"
                )
            except _UNREADABLE_MEMBER:
                # Recorded like the two above rather than skipped silently. A member of a SPLIT
                # archive (`zip -s`) has its directory entry in the final volume and its bytes in
                # an earlier one, so opening it here raises and the member read as clean -- both
                # published parts returned None while joining the volumes recovered the key. The
                # bytes are not in this file, which is exactly the "unverifiable" case, and every
                # other unreadable member reaches the same conclusion for the same reason.
                unreadable = unreadable or "an archive member this check cannot read"
                continue  # the rest of the archive still gets scanned
        # Anything BEFORE the first member is a prefix `zipfile` never enumerates. It reads the
        # LAST end-of-central-directory record, so a second zip appended to a first makes the whole
        # first archive an invisible prefix -- `namelist()` returns only the second zip's members
        # and a key deflated inside the first published intact, since compression also keeps it out
        # of the raw byte pass. `cat a.zip b.zip` is the one-line recipe.
        #
        # The prefix is handed back to the scanner rather than parsed here: it may be another zip,
        # a tar, a compressed stream, or an SFX stub that is genuinely just an executable. Bounded
        # by the lowest local-header offset, so this reads what the archive itself says precedes
        # its first member, and an ordinary zip has no prefix at all and pays nothing.
        prefix = min((info.header_offset for info in archive.infolist()), default=0)
        suffix = _zip_end_offset(source)
        source_size = len(source) if isinstance(source, bytes) else source.stat().st_size
    # A refusal the MEMBERS produced is reported first. The prefix is bytes this archive says it
    # does not own, so a vaguer complaint about them must not displace the specific answer the walk
    # already has -- an encrypted member inside a zip that itself sits inside a tar has a non-zero
    # offset, and letting the prefix scan speak first replaced "an encrypted archive member" with
    # the generic message and named the wrong remedy.
    if unreadable:
        raise refusal(f"contains {unreadable}")
    if prefix > 0:
        head = source[:prefix] if isinstance(source, bytes) else _read_at(source, 0, prefix)
        if head:
            try:
                if kind := scan(io.BytesIO(head), deadline, depth):
                    return kind
            except _UNREADABLE_MEMBER:
                # Only the ordinary unreadable cases are swallowed. A refusal the scanner itself
                # raises is not caught here at all: it carries its own specific message and
                # propagates, which is what a prefix holding real ciphertext should do.
                raise refusal("contains an archive member this check cannot read") from None
    # Bytes AFTER the declared end record are outside the zip just as the prefix is. `ZipFile`
    # accepts that suffix but never enumerates it, so a zlib-compressed key appended after a valid
    # archive returned clean while the same record standing alone was detected.
    if suffix is not None and suffix < source_size:
        tail = (
            source[suffix:]
            if isinstance(source, bytes)
            else _read_at(source, suffix, source_size - suffix)
        )
        if tail:
            try:
                if kind := scan(io.BytesIO(tail), deadline, depth):
                    return kind
            except _UNREADABLE_MEMBER:
                raise refusal("contains an archive member this check cannot read") from None
    return None


def credential_in_tar(
    source: Path | bytes,
    *,
    deadline: float,
    depth: int,
    scan: Scanner,
    refusal: type[Exception],
    named: Namer,
    member_limit: int,
) -> str | None:
    """The kind of credential in any readable member of a tar, or None.

    A tar is not itself compressed, so its members' bytes are read by the ordinary scan already --
    but a COMPRESSED member inside one is not: `tar > shard.gz` holds the credential nowhere a
    pattern can see, exactly like `zip > shard.gz`, and only the zip form was ever expanded.
    Enumerating members hands each one to the scanner, which expands it if it is a container.

    Streamed with `r|*` rather than `r:*`: the streaming reader does not seek back over the archive,
    so a member's data is read once, in order. Members are guarded separately for the same reason
    they are in a zip -- one unreadable entry must not abandon the entries behind it.
    """
    handle = source.open("rb") if isinstance(source, Path) else io.BytesIO(source)
    source_size = len(source) if isinstance(source, bytes) else source.stat().st_size
    saw_header = False
    unreadable = ""
    remainder_at: int | None = None
    try:
        with tarfile.open(fileobj=handle, mode="r|*") as archive:
            try:
                for count, info in enumerate(archive, 1):
                    # A parsed header is what proves these bytes really are a tar: `tarfile`
                    # validates its checksum, so arbitrary content does not produce one.
                    saw_header = True
                    if count > member_limit:
                        raise refusal("contains an archive with too many members to inspect")
                    if time.monotonic() > deadline:
                        raise refusal("takes too long to decompress")
                    # the member NAME is checked too: a tar entry called `fslo_<key>.json` publishes
                    # the key in the archive's listing whatever its contents are.
                    #
                    # Checked BEFORE the entry-type filter. A directory, symlink or device node has
                    # no contents to scan, but its name is in the listing exactly as a file's is --
                    # so skipping non-files first meant a directory named with base64 of an
                    # OpenSSL-encrypted file returned clean here while the name scanner refused that
                    # same string on its own.
                    if kind := named(info.name):
                        return kind
                    # A symlink or hardlink publishes its TARGET in the listing too. Filtering
                    # non-files first left a base64-encoded encrypted target clean even though the
                    # same target passed directly to the name scanner was refused.
                    if info.linkname and (kind := named(info.linkname)):
                        return kind
                    if not info.isfile():
                        continue
                    try:
                        member = archive.extractfile(info)
                        if member is None:
                            continue
                        if kind := scan(member, deadline, depth):
                            return kind
                    except _UNREADABLE_MEMBER:
                        unreadable = unreadable or "an archive member this check cannot read"
                        continue
            except _UNREADABLE_MEMBER:
                # A member declaring more bytes than the file holds fails in the ITERATOR walking to
                # the next header, escaping the per-member guard into the dispatch loop -- which
                # reads this tuple as "not this format", so the file published on its literal bytes.
                # A truncated tar around a zlib-compressed key returned clean while that member
                # alone reported it: the raw scan finds no compressed record at an arbitrary offset.
                #
                # Re-raised without a header, where the same exception is the ordinary "not a tar"
                # every non-tar file gives and refusing would refuse most of a corpus.
                if not saw_header:
                    raise
                unreadable = unreadable or "an archive member this check cannot read"
            else:
                # the streaming iterator stops at the first zero block and may have read ahead into
                # its own buffer, so the underlying handle's position is not the archive boundary.
                # its logical position is the point it actually consumed and must be captured before
                # closing the stream wrapper below.
                remainder_at = archive.fileobj.tell()
    finally:
        handle.close()
    if unreadable:
        raise refusal(f"contains {unreadable}")
    if remainder_at is not None and 0 < remainder_at < source_size:
        tail = (
            source[remainder_at:]
            if isinstance(source, bytes)
            else _read_at(source, remainder_at, source_size - remainder_at)
        )
        # tar writers pad the two zero end blocks to a 10 KiB record. Leaving those blocks in front
        # made a complete second tar invisible to the next structural walk; stripping only whole
        # zero blocks preserves a real suffix whose first header byte happens to be zero.
        while tail.startswith(bytes(512)):
            tail = tail[512:]
        # every recursive pass starts strictly later than its parent. This is both the concatenated
        # tar fix and the recursion guard: a remainder that did not advance is never dispatched.
        if tail and len(tail) < source_size:
            try:
                if kind := scan(io.BytesIO(tail), deadline, depth):
                    return kind
            except _UNREADABLE_MEMBER:
                raise refusal("contains an archive member this check cannot read") from None
    return None


# What an ar archive begins with, and the fixed-width header each member carries: a 16-byte name, 12
# bytes of mtime, 6 each of uid and gid, 8 of mode, 10 of size, then the two-byte magic that ends
# it. Members are padded to an even offset.
_AR_MAGIC = b"!<arch>\n"
_AR_HEADER = 60
_AR_SIZE_FIELD = slice(48, 58)
_AR_NAME_FIELD = slice(0, 16)


def credential_in_ar(
    source: Path | bytes,
    *,
    deadline: float,
    depth: int,
    scan: Scanner,
    refusal: type[Exception],
    named: Namer,
    member_limit: int,
    size_limit: int | None = None,
) -> str | None:
    """The kind of credential in any readable member of an ar archive, or None.

    A Debian package is an ar holding `debian-binary`, `control.tar.*` and `data.tar.*`, and it is
    an ordinary thing to find beside an environment. None of the other handlers enumerates it: the
    members' magic sits at their own offsets rather than at byte zero, so the anchored format check
    could not see it, and the raw pass reads compressed member bytes as opaque content. A `.deb`
    whose `data.tar.zst` this cannot expand therefore published clean, while the same zstd payload
    standing alone was refused.

    Each member's bytes go back to the scanner, exactly as a tar's do, so a container inside one is
    expanded and an uninspectable one is refused rather than passed over.
    """
    # The signature is read from the first bytes rather than by loading the file. An ar archive is
    # rare beside an environment while a large member is ordinary, so reading a 256 MiB shard whole
    # to discover it is not an ar is a cost paid by every file to serve almost none of them -- the
    # same reason the zlib probe reads a bounded prefix rather than the member.
    if isinstance(source, Path):
        with source.open("rb") as head:
            if head.read(len(_AR_MAGIC)) != _AR_MAGIC:
                return None
        # the ar walk needs random access to member bodies, so it holds the archive once recognised.
        # a 200 MiB archive previously drove peak rss up by 448 MiB before any nested-buffer bound
        # applied; oversized containers are undecided rather than clean, as in the pdf walk.
        if size_limit is not None and source.stat().st_size > size_limit:
            raise refusal("contains an archive too large to inspect")
        data = source.read_bytes()
    else:
        data = source
        if not data.startswith(_AR_MAGIC):
            return None
    unreadable = ""
    gnu_names: bytes | None = None
    at = len(_AR_MAGIC)
    for _ in range(member_limit):
        if at + _AR_HEADER > len(data):
            # Fewer bytes left than a header. Exactly zero is the archive's clean end; anything
            # else is a remainder the walk cannot parse and therefore has not read -- and unread is
            # not clean. A 29-byte zlib record after a zero-length member fits in that gap, and
            # returning None here published the key it inflates to while the same record standing
            # alone was refused.
            if at != len(data):
                unreadable = unreadable or "an archive member this check cannot read"
            break
        header = data[at : at + _AR_HEADER]
        try:
            size = int(header[_AR_SIZE_FIELD].decode("ascii", "replace").strip() or "-1")
        except ValueError:
            size = -1
        if size < 0:
            # A header whose size field is unreadable stops the walk: every later member is found by
            # stepping over this one, so a wrong length would resynchronise on arbitrary bytes.
            # Unread is not clean, and the archive is reported as such below.
            unreadable = unreadable or "an archive member this check cannot read"
            break
        if time.monotonic() > deadline:
            raise refusal("takes too long to decompress")
        raw_name = header[_AR_NAME_FIELD].decode("ascii", "replace").strip()
        # The NAME leaks through the archive's listing exactly as a tar's does, and `ar` pads its
        # short names with spaces and ends them with `/`. The placeholder is checked before resolving
        # long-name schemes so neither representation becomes a gap.
        if kind := named(raw_name.rstrip("/")):
            return kind
        body = data[at + _AR_HEADER : at + _AR_HEADER + size]
        if len(body) < size:
            unreadable = unreadable or "an archive member this check cannot read"
            break
        payload = body
        resolved_name: str | None = None
        if raw_name == "//":
            gnu_names = body
        elif raw_name.startswith("#1/"):
            # BSD ar stores the real name at the front of the member body. Scanning only `#1/<len>`
            # left an encoded credential in that resolved name unpublished by the name check.
            try:
                name_size = int(raw_name[3:])
            except ValueError:
                name_size = -1
            if name_size < 0 or name_size > len(body):
                unreadable = unreadable or "an archive member this check cannot read"
            else:
                resolved_name = body[:name_size].decode("ascii", "replace").rstrip("\0")
                payload = body[name_size:]
        elif raw_name.startswith("/") and raw_name[1:].isdigit():
            # GNU ar stores names in the `//` member and writes `/<offset>` in each real header. The
            # placeholder contains no credential, while the resolved newline-terminated name can.
            offset = int(raw_name[1:])
            if gnu_names is None or offset >= len(gnu_names):
                unreadable = unreadable or "an archive member this check cannot read"
            else:
                end = gnu_names.find(b"\n", offset)
                if end < 0:
                    unreadable = unreadable or "an archive member this check cannot read"
                else:
                    resolved_name = gnu_names[offset:end].rstrip(b"/").decode("ascii", "replace")
        if resolved_name and (kind := named(resolved_name)):
            return kind
        try:
            if kind := scan(io.BytesIO(payload), deadline, depth):
                return kind
        except _UNREADABLE_MEMBER:
            unreadable = unreadable or "an archive member this check cannot read"
        at += _AR_HEADER + size + (size % 2)
    else:
        # Reaching the bound is allowed when that member ended the archive. Only bytes still left
        # prove there is a member beyond the configured limit; the old unconditional `else` refused
        # a clean one-member archive when its limit was exactly one.
        if at != len(data):
            raise refusal("contains an archive with too many members to inspect")
    if unreadable:
        raise refusal(f"contains {unreadable}")
    return None
