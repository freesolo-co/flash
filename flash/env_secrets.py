"""Credential content scan for `flash env push`.

Filename filters cannot decide this question. `_ENV_PUSH_SECRET_PATTERNS` drops files *named* like
secret stores (`.env`, `*.pem`, `credentials*`), but the common convention of exporting keys from a
sourceable shell file -- `env.sh`, `setenv.sh`, `secrets.sh` -- is named like ordinary tooling, so a
plain `flash env push .` committed a live `FREESOLO_API_KEY` into the shared environment hub. A
published env repo is org-shared and its history is permanent, so the leak survives deleting the
file afterwards. Python source is exempt from the name filter entirely (so helper modules ship
instead of breaking the worker with ModuleNotFoundError), so a key pasted into a helper had nothing
between it and the hub either.

So the authoritative check reads what is about to be published and refuses on credential *shape*.
Patterns require an issuer prefix plus a long key body: `hf_[A-Za-z0-9]{20,}` is a token, while the
`hf_hub_download` in ordinary code is not, so a real environment still publishes untouched.

Split out of `flash.cli.commands.env.push` to keep that module under the file-size limit, and kept
free of any import from it so the dependency runs one way.
"""

from __future__ import annotations

import base64
import binascii
import bz2
import gzip
import io
import lzma
import os
import re
import tarfile
import time
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import IO, NoReturn, Protocol

from flash.env_formats import (
    _MAX_ARCHIVE_MEMBERS,
    _UNEXPANDABLE_MAGIC,
    _ZIP_TAIL_BYTES,
    _after_skippable_frames,
    _has_zip_end_record,
    _is_openpgp_secret_key,
    _looks_compressed,
    _looks_like_tar,
    _zip_member_count,
)

# Bodies are bounded rather than open-ended so a match has a maximum length, which is what lets
# `_SCAN_OVERLAP_BYTES` below be a real guarantee instead of a hope. The cap is far above every
# issued key format; a longer key still matches its first `_MAX_BODY` characters, which is a
# detection either way.
_MAX_BODY = 256

# (kind, pattern) for issued tokens: an issuer prefix plus a long key body, captured as a group.
# The kind names the credential in the refusal so the author knows which key to rotate; the matched
# text is NEVER echoed, since the error is printed and may reach a log.
#
# Patterns are BYTES: members are scanned as raw bytes so a credential stored inside a binary
# container (a sqlite state file, a pickle, an archive) is not skipped. Prefix-anchored patterns
# cannot realistically fire on random bytes -- matching `fslo_` alone is 256**-5 per position.
#
# No AWS entry. An `AKIA...` access key ID is a public identifier -- AWS puts it in signed URLs in
# the clear, so it turns up verbatim in any web-scraped dataset (it does, in the mbti training
# shards here) -- and the matching secret access key is 40 undifferentiated base64 characters with
# no prefix to anchor on. Matching the identifier would block real dataset publishes while still
# not catching the secret that actually matters.
_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("a Freesolo API key", re.compile(rb"fslo_([A-Za-z0-9_-]{16,%d})" % _MAX_BODY)),
    ("a Hugging Face token", re.compile(rb"hf_([A-Za-z0-9]{20,%d})" % _MAX_BODY)),
    (
        "a GitHub token",
        re.compile(
            rb"gh[pousr]_([A-Za-z0-9]{20,%d})|github_pat_([A-Za-z0-9_]{20,%d})"
            % (_MAX_BODY, _MAX_BODY)
        ),
    ),
    ("a Prime Intellect key", re.compile(rb"pit_([A-Za-z0-9]{16,%d})" % _MAX_BODY)),
    ("an Anthropic API key", re.compile(rb"sk-ant-([A-Za-z0-9_-]{20,%d})" % _MAX_BODY)),
    ("an OpenRouter API key", re.compile(rb"sk-or-v1-([A-Za-z0-9]{20,%d})" % _MAX_BODY)),
    # every currently-issued OpenAI family is named explicitly. `sk-svcacct-` and `sk-admin-` keys
    # carry project-wide and organization-wide authority, and neither is reachable through the bare
    # `sk-` branch below: the subtype's own hyphen ends that branch's alphanumeric run early.
    (
        "an OpenAI API key",
        re.compile(
            rb"sk-(?:proj|svcacct|admin)-([A-Za-z0-9_-]{20,%d})"
            # the bare legacy form requires a capital SOMEWHERE in the body, tested by lookahead
            # rather than by position. Demanding 31 more characters *after* the first capital
            # missed real keys: a legacy body carries `T3BlbkFJ` around index 20, leaving too few
            # behind it. The requirement is still what kills the false positive, since a
            # lowercase-hex body of the same length is a content hash, not a key --
            # `.../assets/sk-<32 hex>.js` is an ordinary CDN asset URL.
            rb"|sk-((?=[a-z0-9]*[A-Z])[A-Za-z0-9]{32,%d})" % (_MAX_BODY, _MAX_BODY)
        ),
    ),
    # `xapp-` is Slack's app-level token, a different prefix rather than another `xox` letter.
    ("a Slack token", re.compile(rb"(?:xox[baprs]|xapp)-([A-Za-z0-9-]{10,%d})" % _MAX_BODY)),
)

# Credentials with no issuer prefix, anchored on the ASSIGNMENT that names them instead. Both are
# names this repository already treats as runtime secrets (`WANDB_API_KEY` is the default in
# `flash/client/runtime_secrets.py`, and `AWS_SECRET_ACCESS_KEY` is its documented example), so a
# training environment is exactly the directory where one sits beside the config.
#
# The NAME matches case-insensitively: the same key sits in an env file as `WANDB_API_KEY` and in a
# yaml or python config as `wandb_api_key`, and it is equally live in both. The BODY is what bounds
# the false positives, not the casing of the name.
#
# Bodies are deliberately not pinned to one length. W&B issued 40-hex keys historically and now
# issues much longer ones (the SDK's own `API key must be 40 characters long, yours was 86` error
# is that migration), and both remain live -- a new-format key does not revoke an existing legacy
# one. Matching only 40-hex would have caught the legacy form and published every currently-issued
# key. AWS secret access keys are 40 characters of base64 alphabet with no prefix at all.
#
# The variable name is what makes these bytes a credential: bare 40-hex is a git sha and bare
# 40-base64 is any digest, and refusing those would block ordinary dataset publishes. So the
# assignment is required, and a `${VAR}` or `$(...)` indirection matches nothing because it is not
# a key-shaped body.
#
# These take `_is_high_entropy` like every other pattern. Exempting them was tenable only while the
# W&B body was pinned to 40 hex characters; once it widened,
# `WANDB_API_KEY=your_wandb_api_key_here...` matched and a scaffolded environment became
# unpublishable. The exemption existed for the all-hex legacy key, which that function now admits
# directly.
_ASSIGNED_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "a Weights & Biases API key",
        re.compile(rb"(?i:wandb_api_key)[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_-]{40,%d})" % _MAX_BODY),
    ),
    (
        "an AWS secret access key",
        re.compile(
            rb"(?i:aws_secret_access_key)[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"
        ),
    ),
)


class _Searchable(Protocol):
    """What this module needs of a pattern: `search(data)` returning a match or None.

    `re.Pattern[bytes]` satisfies it, and so does a detector built from more than one pattern.
    """

    def search(self, data: bytes, /) -> re.Match[bytes] | None: ...


class _JwkPrivateKey:
    """A private JSON Web Key: a `kty` naming a key type, plus at least one private member.

    Presented as a pattern object because `_match` iterates `(kind, pattern)` pairs and calls
    `.search`; the two markers cannot be one regex without reintroducing the span between them.
    """

    _KTY = re.compile(rb"\"kty\"\s*:\s*\"(?:RSA|EC|OKP|oct)\"")
    # `d` is the private exponent or scalar in every key type; for RSA the CRT parameters
    # accompany it. A public JWK carries none of these, which is exactly what separates the two.
    _PRIVATE = re.compile(rb"\"(?:d|dp|dq|qi)\"\s*:\s*\"[A-Za-z0-9+/\-_]{20,}={0,2}\"")

    def search(self, data: bytes) -> re.Match[bytes] | None:
        """The private member's match when a `kty` accompanies it anywhere in `data`, else None."""
        if not (private := self._PRIVATE.search(data)):
            return None
        return private if self._KTY.search(data) else None


# Read in bounded chunks so a large dataset member is never held in memory whole. This costs no
# more I/O than the publish already pays: `_tar_b64` reads every one of these bytes to gzip them.
_SCAN_CHUNK_BYTES = 1 << 20
# Carried between chunks so a credential straddling a chunk boundary is still matched. Longer than
# the longest possible match (`_MAX_BODY` plus the longest prefix), so every match is fully visible
# inside some window rather than merely likely to be.
_SCAN_OVERLAP_BYTES = 1024

# How much of a chunk's head is walked for skippable frames. Generous for the handful a real
# seekable stream carries, and bounded so a chain of crafted frame headers cannot make this walk a
# cost of its own.
_SKIPPABLE_SCAN_BYTES = 64 << 10

# A base64 run long enough to hold the shortest credential a pattern admits. The lower bound makes
# the scan walk past ordinary prose rather than decoding every word it meets. There is deliberately
# no upper bound: capping the run split a long encoded file into adjacent pieces, and a credential
# straddling the cut decoded into neither half -- so base64 of a whole `env.sh` published clean.
# Length is instead bounded by `_decode_windows` below, which slides a window over the run.
#
# The URL-safe alphabet (`-` and `_` for `+` and `/`, RFC 4648 section 5) is admitted too. It is
# what `base64.urlsafe_b64encode`, a JWT, and most token-in-a-URL configs emit, and accepting only
# `+/` split such a value at its first `-` so the fragments decoded to neither the whole token nor
# anything matching. The two alphabets are disjoint apart from the shared 62 characters, so one
# pattern covers both and the decode below translates whichever pair is present.
_BASE64_RUN = re.compile(rb"[A-Za-z0-9+/\-_]{24,}={0,2}")

# Maps the URL-safe alphabet onto the standard one so a single decoder handles both. A run mixing
# the two is not valid base64 either way, and translating it simply fails to decode as before.
_URL_SAFE_ALPHABET = bytes.maketrans(b"-_", b"+/")

# Marks NUL as 1 and everything else as 0, so an unbroken stretch of padding bytes becomes a run
# `_WIDE_RUN` can find. The length floor is the shortest credential body worth decoding; below it
# a chance alignment of NULs cannot carry one anyway.
_NUL_MARKER = bytes(1 if byte == 0 else 0 for byte in range(256))
_WIDE_RUN = re.compile(rb"\x01{24,}")

# A fixed-width wrapped base64 block: one or more full-width lines, then a final line of any
# length. The widths are the two conventions in use -- 76 for MIME (`base64.encodebytes`, mail,
# many `kubectl -o yaml` outputs) and 64 for PEM bodies. Joining ONLY these leaves ordinary
# adjacent lines alone, so no arbitrary pair of values is welded into a run that decodes to
# something neither line contained.
#
# ONE full line is enough to qualify, not two. Requiring two meant the commonest shape of all --
# a blob just over the width, so one full line plus a short tail -- never joined, and a key
# straddling its single break decoded into neither side.
#
# `\r?\n` because a Windows checkout or a YAML export wraps with CRLF, and matching only `\n` left
# every CRLF blob unjoined. The `\r` is dropped along with the `\n` when the block is joined.
#
# `[ \t]*` after each break because a wrapped blob is routinely INDENTED: a YAML block scalar, a
# base64 value under a `data:` key, a heredoc inside a function body. Requiring the next line to
# start in column 0 missed 20 of 60 key offsets on a two-space-indented block, at both widths --
# and an indented blob is the commonest way a key is embedded in a config file.
_WRAPPED_BLOCK = re.compile(
    rb"(?:[A-Za-z0-9+/]{76}\r?\n[ \t]*)+[A-Za-z0-9+/]{1,76}={0,2}"
    rb"|(?:[A-Za-z0-9+/]{64}\r?\n[ \t]*)+[A-Za-z0-9+/]{1,64}={0,2}"
)
# A necessary condition for the block above: a full-width line of base64 followed by a break. Cheap
# to reject, and it fails on essentially every real file, so the expensive alternation only runs
# where a wrapped block could actually be.
_WRAPPED_HINT = re.compile(rb"[A-Za-z0-9+/]{64}\r?\n")
# The break itself, removed when a block is joined -- with any indent that follows it, or the
# joined run would still carry spaces outside the base64 alphabet. Both endings, so a CRLF blob
# joins into a continuous run rather than one still carrying stray `\r` bytes.
_WRAPPED_BREAK = re.compile(rb"\r?\n[ \t]*")

# How much of one base64 run to decode at a time, and how far the windows overlap. The overlap
# exceeds the encoded length of the longest possible match (4/3 of `_MAX_BODY` plus its prefix), so
# a credential anywhere in a run of any length lands whole inside some window.
_BASE64_WINDOW = 8192
_BASE64_WINDOW_OVERLAP = 1024

# How many container layers deep to expand. A zip holding a gzipped shard is an ordinary way to
# ship a dataset, and stopping at one level meant the inner member's bytes were treated as final
# content -- so a key one layer further in published untouched. The limit still exists because
# each layer multiplies the work a hostile archive can demand.
_MAX_CONTAINER_DEPTH = 4
# A nested container is buffered in memory to be reopened, so its size is capped.
#
# BOTH limits refuse rather than pass when they bite. Returning "nothing found" made the cheapest
# bypass of the entire check "make it expensive": pad an archive past the buffer cap, or bury the
# key one layer past the depth cap, and the scan reported clean. No real environment here comes
# close to either bound, so a refusal means something genuinely unusual is being published.
_MAX_NESTED_BUFFER_BYTES = 64 << 20
# Wall-clock budget for expanding one file's archives. Expansion is unbounded in principle, so it
# needs a stop -- but a stop on BYTES is the wrong one: it discards the tail of the stream, and a
# credential placed after the cutoff then publishes. That is not hypothetical, since the package
# limit bounds *compressed* size: gzip does about 1000:1 on padding, so a 294 KB member expands
# past any byte cap you would plausibly set while staying far under the 256 MB package limit.
#
# So the whole stream is scanned, chunk by chunk with bounded memory, and the budget bounds TIME.
# A pathological archive costs a bounded wait rather than an unbounded one, while every real member
# -- the largest here expands in about 3 seconds -- finishes long inside it.
#
# The budget covers a whole PACKAGE, not one file. Per-file it multiplied: a package may hold 5,000
# members (`ARCHIVE_MEMBER_LIMIT`), so a caller could split compression bombs across them and buy
# 5,000 x 60s of expansion from an authenticated `POST /v1/envs` while every individual file stayed
# inside the apparent one-minute limit.
_MAX_DECOMPRESS_SECONDS = 60.0
# How many members of ONE archive to enumerate, imported above. `ZipFile` reads the entire central
# directory up front, so a nested archive of millions of empty entries materialises millions of
# `ZipInfo` objects before any per-member budget is consulted -- and empty members never enter the
# read loop that checks the deadline, so neither bound could stop it. The package extractor counts
# such an archive as a single ordinary file, so this is the only place the inner count is bounded.
#
# Every read of it is in THIS module, and the directory walk takes it as an argument rather than
# reading it from its own, so rebinding it here still governs the whole check.

# An all-hex body of key length. Recognised so `_is_high_entropy` admits a legacy 40-hex W&B key,
# whose lowercase letters would otherwise read as the hand-written-placeholder convention.
_HEX_BODY = re.compile(r"[0-9a-fA-F]{32,}")

# Everything the standard library raises for an archive it cannot read. There is no single base
# class to catch, and most of these are not OSError, so each omission crashed `flash env push` with
# a traceback on an ordinary corrupt shard: an encrypted member raises RuntimeError, an
# unimplemented compression method NotImplementedError, a corrupt deflate stream zlib.error, and a
# corrupt xz lzma.LZMAError (which inherits straight from Exception).
#
# Shallow corruption is a trap when testing this: truncating near the end of an xz stream raises
# EOFError, which was already caught, so the bug looks absent. The distinct error only appears when
# the damage is deep enough that the decompressor rejects the data rather than running out of it.
_UNREADABLE_ARCHIVE = (
    OSError,
    EOFError,
    ValueError,
    RuntimeError,
    NotImplementedError,
    zipfile.BadZipFile,
    zlib.error,
    lzma.LZMAError,
    # `TarError` inherits straight from `Exception`, so nothing else in this tuple covers it -- a
    # truncated tar (`ReadError: unexpected end of data`) crashed the publish outright. A
    # half-written shard in a dataset directory is ordinary, and crashing on it would be a worse
    # bug than the hole being closed.
    tarfile.TarError,
)


# A PEM header only means a key is present when the BODY follows it. The header alone is something
# documentation says -- "if you see -----BEGIN RSA PRIVATE KEY----- in a log, redact it" -- and
# refusing on it blocks a legitimate publish over prose about credentials. Requiring a base64 line
# after the header keeps every real key (they all carry one) and drops the mention of one.
#
# The second alternative is the encrypted form, whose body is preceded by RFC 1421 headers instead
# of starting with base64. It names those two headers exactly: a general `[A-Za-z-]+:` also accepts
# `Warning:` or `Note:`, which is prose about a key rather than a key, and reopens the very false
# positive the base64 requirement exists to close.
_LITERAL_PATTERNS: tuple[tuple[str, _Searchable], ...] = (
    #
    # `(?: BLOCK)?` because OpenPGP armours as `-----BEGIN PGP PRIVATE KEY BLOCK-----`. Without it
    # the trailing word made the header unmatchable and a `gpg --export-secret-keys --armor` file
    # published intact -- `[A-Z ]*` reaches `PGP PRIVATE KEY`, but nothing followed ` BLOCK`.
    #
    # `_ARMOR_HEADERS` skips the RFC 4880 armor headers that sit between the BEGIN line and the
    # body. Requiring base64 IMMEDIATELY after the header caught only the headerless export: an
    # armour carrying `Version:` or `Comment:` -- what most implementations emit, and what a
    # hand-annotated backup carries -- went undetected again.
    #
    # Those five keys are named exactly, for the same reason `Proc-Type:`/`DEK-Info:` are below. A
    # general `[A-Za-z-]+:` would also skip `Warning:` and `Note:`, which is prose about a key
    # rather than a key, and reopens the false positive the base64 requirement exists to close.
    (
        "a private key block",
        re.compile(
            rb"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----[\r\n\s]*"
            rb"(?:(?:Version|Comment|MessageID|Hash|Charset):[^\r\n]*[\r\n\s]*)*"
            rb"(?:[A-Za-z0-9+/=]{32,}|Proc-Type:|DEK-Info:)"
        ),
    ),
    # The same key in DER: the binary encoding a PEM block base64-wraps. `openssl ... -outform DER`
    # writes it, and it carries no text marker at all, so the PEM pattern above cannot see it.
    #
    # Anchored on the ASN.1 that distinguishes a private key from a public one rather than on the
    # algorithm OID alone, which a certificate or public key carries too. In PKCS#8 that is the
    # `INTEGER 0` version field preceding the AlgorithmIdentifier (a SubjectPublicKeyInfo has no
    # version); in PKCS#1 it is the same version INTEGER before the modulus; in SEC1 it is
    # `INTEGER 1` followed by the private scalar as an OCTET STRING of a curve-sized length.
    (
        "a private key",
        re.compile(
            # PKCS#8 PrivateKeyInfo, by algorithm: RSA, the RFC 8410 curves, EC.
            #
            # The RFC 8410 OIDs are `1.3.101.{110,111,112,113}` = X25519, X448, Ed25519, Ed448,
            # whose final byte is 0x6e, 0x6f, 0x70, 0x71. Naming only the 25519 pair let a real
            # Ed448 or X448 key publish intact; the four are one contiguous range.
            # DSA is `1.2.840.10040.4.1` (`2a 86 48 ce 38 04 01`). Its AlgorithmIdentifier is a
            # SEQUENCE rather than the 1-byte length the others use, so the `\x30.` above does not
            # cover it: `openssl pkcs8 -topk8 -nocrypt -outform DER` on a real 1024-bit DSA key
            # produced a file every branch here passed as clean.
            rb"\x02\x01\x00\x30.\x06(?:\x09\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01"
            rb"|\x03\x2b\x65[\x6e-\x71]|\x07\x2a\x86\x48\xce\x3d\x02\x01)"
            # DSA `1.2.840.10040.4.1` and DH `1.2.840.113549.1.3.1`, each across the three
            # AlgorithmIdentifier length forms. DH is `dhKeyAgreement`: `openssl genpkey
            # -paramfile` writes it, `openssl pkey -check` accepts it, and every branch above
            # passed the resulting DER as clean.
            rb"|\x02\x01\x00\x30(?:\x82..|\x81.|[\x00-\x7f])\x06"
            rb"(?:\x07\x2a\x86\x48\xce\x38\x04\x01|\x09\x2a\x86\x48\x86\xf7\x0d\x01\x03\x01)"
            # PKCS#1 RSAPrivateKey: version 0 then the modulus INTEGER, whose length may be stated
            # in any of DER's three forms. Requiring `\x02\x82` recognised only 2048-bit and larger
            # keys: `openssl rsa -outform DER -traditional` writes `02 81 81` for a 1024-bit key
            # and a short-form `02 41` for a 512-bit one, so both published intact. `0x81` and
            # `0x82` introduce a 1- and 2-byte length; a short form below 0x80 IS the length, and
            # is bounded from 0x40 up so an ordinary `02 01 00 02 xx` byte sequence does not match.
            rb"|\x30\x82..\x02\x01\x00\x02(?:\x82..|\x81.|[\x40-\x7f])\x00"
            # SEC1 ECPrivateKey: version 1, a curve-sized scalar, then the [0] curve parameters
            rb"|\x02\x01\x01\x04(?:\x20.{32}|\x30.{48}|\x42.{66})\xa0"
            # EncryptedPrivateKeyInfo: `openssl pkcs8 -topk8 -passout` in DER. The plaintext key is
            # inside an OCTET STRING, so none of the structures above appear anywhere in the file
            # and it published intact -- the ARMOURED form of the same key was caught by its
            # `-----BEGIN ENCRYPTED PRIVATE KEY-----` header, which made DER the way past.
            #
            # Anchored on the encryption-algorithm OID in the AlgorithmIdentifier: PBES2
            # (1.2.840.113549.1.5.13) or a pkcs-12 PBE (1.2.840.113549.1.12.1.x). A passphrase is
            # not much protection for a key in a public repository, and the OIDs appear only in a
            # key that is actually encrypted.
            rb"|\x30.{1,4}?\x30.\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x05\x0d"
            rb"|\x30.{1,4}?\x30.\x06\x0a\x2a\x86\x48\x86\xf7\x0d\x01\x0c\x01.",
            re.DOTALL,
        ),
    ),
    # The same key as a JSON Web Key. Node's `privateKey.export({format: "jwk"})` and every JOSE
    # library write this, `.json` and `.jwk` are ordinary publishable files, and the encoding
    # carries neither a PEM header nor DER structure -- so a complete RSA or EC private key passed
    # every check above as clean.
    #
    # Anchored on the PRIVATE members, not on JWK-ness: a public JWK is meant to be shared and
    # differs only by their absence. `d` is the private exponent or scalar in every key type; for
    # RSA the CRT parameters accompany it. Requiring a `kty` alongside keeps this off arbitrary
    # JSON that happens to carry a short `"d"` field.
    (
        "a private key",
        # Two independent markers rather than one pattern spanning both. Requiring them within a
        # window of each other was wrong in both directions: JWK members may appear in any order
        # with arbitrary extension members between them, so 5 KiB of metadata between `kty` and
        # `d` published a real RSA key; and the `[\s\S]{0,4096}?` that spanned the gap backtracked
        # over every position of a near-matching body, which took 4.2 seconds per MiB of
        # `"kty":"RSA",` repeated -- about 18 minutes for a permitted 256 MiB package.
        #
        # Order-independent and window-free, so it is exact on a real key either way round, and
        # each half is anchored on a literal that fails fast on ordinary JSON.
        _JwkPrivateKey(),
    ),
)


def _is_high_entropy(body: bytes, *, hex_is_issued: bool = False) -> bool:
    """Whether a key body looks issued rather than hand-written.

    An issued token is random over its alphabet, so each rejected shape below is one a real key
    essentially never takes. For a 16-character base62 body -- the shortest any pattern admits --
    the chance of landing in one is about 5 in 10,000,000, and for the 45-character Freesolo bodies
    actually issued it is vanishingly smaller still. The three rejected shapes are exactly the
    placeholder conventions:

      * all lowercase letters, no digit -- `fslo_retry_after_close`, `fslo_your_api_key_here`
      * all capital letters, no digit -- `fslo_YOUR_API_KEY_HERE`, `fslo_REPLACE_ME`
      * one or two distinct characters -- `fslo_XXXXXXXXXXXXXXXX`, a masked value

    Testing only for "a digit or a capital" caught the lowercase convention and missed the other
    two, so `flash env push` refused a scaffolded environment and told the author to rotate a key
    that had never existed. A false refusal is not harmless: it is the failure mode that gets a
    check switched off.

    Mixed-case and digit-bearing bodies are still treated as issued, which keeps a real key whose
    body happens to read like a word. Erring that way is deliberate -- a false refusal is visible
    and recoverable, a missed credential is permanent in a shared repository's history.

    Applied to the assignment-anchored patterns too. Exempting those was a false-positive
    regression: once the W&B body widened past 40 hex characters,
    `WANDB_API_KEY=your_wandb_api_key_here_replace_before_push` matched and a scaffolded
    environment could not be published. The exemption existed for the all-hex legacy key
    (`abcdef...` is all-lowercase-alpha, which reads as a placeholder), so that one case is
    admitted explicitly below rather than by exempting the whole group.

    `hex_is_issued` carries that admission, and only the assignment-anchored patterns set it. It
    is what tells an all-hex W&B key from `hf_deadbeefdeadbeefdeadbeefdeadbeef`: applying it to
    every pattern refused the canonical hex placeholder under an issuer prefix, because the hex
    test ran before the all-lowercase-alpha rule that would have cleared it. Withholding it costs
    no real token -- an issued `hf_`/`fslo_`/`sk-` body is base62, so one confined to `[a-f]` with
    no digit at all is not a shape they take.
    """
    text = body.decode("ascii", "ignore").replace("_", "").replace("-", "")
    if len(set(text)) <= 2:
        return False
    if hex_is_issued and _HEX_BODY.fullmatch(text):
        # a full-length hex body is a key or a hash, never a hand-written placeholder: the
        # convention is words (`your_key_here`), and those are not confined to `[a-f]`.
        return True
    return not (text.isalpha() and (text.isupper() or text.islower()))


def _match(data: bytes) -> str | None:
    """The kind of credential the literal bytes `data` contain, or None."""
    for kind, pattern in _LITERAL_PATTERNS:
        if pattern.search(data):
            return kind
    # only the assignment-anchored group admits an all-hex body: its W&B key is issued as hex,
    # while an issuer-prefixed token is base62, so an all-hex body there is the placeholder.
    for group, hex_is_issued in ((_TOKEN_PATTERNS, False), (_ASSIGNED_PATTERNS, True)):
        for kind, pattern in group:
            for match in pattern.finditer(data):
                # the alternations put the body in whichever group matched; the rest are None.
                body = next((found for found in match.groups() if found), b"")
                if _is_high_entropy(body, hex_is_issued=hex_is_issued):
                    return kind
    return None


def _match_base64(data: bytes) -> str | None:
    """The kind of credential hidden in a base64 run, or None.

    A Kubernetes Secret stores every value base64-encoded, and that is an ordinary file to keep
    beside an environment. The encoded key shares no substring with the plaintext, so none of the
    patterns can see it, and `data: <base64 of an fslo_ key>` published with exit 0.

    Only runs long enough to hold a credential are decoded, and only the base64 alphabet is
    considered, so this walks past prose. The decoded bytes go through `_match` rather than the full
    `_credential_kind`, which keeps the recursion one level deep: base64 of base64 is not a
    convention worth chasing, and unbounded re-decoding is a denial-of-service surface.

    A run is decoded in overlapping windows rather than whole, so memory stays bounded on a large
    encoded blob while a credential anywhere in it still lands whole inside some window. Slicing a
    long run into ADJACENT pieces was a bypass: a key straddling the cut decoded into neither half,
    so base64 of an ordinary config file published clean.

    Measured against the false-positive risk before adopting it: 630,011 base64-shaped runs across
    8,769 real hub files decode to zero credential matches, so this costs no legitimate publish.
    """
    for run in _BASE64_RUN.finditer(_unwrapped(data)):
        candidate = run.group(0)
        for window in _decode_windows(candidate):
            # base64 packs 3 bytes per 4 characters, so a run rarely starts on a boundary; all four
            # alignments are tried, each trimmed to a whole number of quartets.
            for start in range(min(len(window), 4)):
                chunk = window[start:]
                chunk = chunk[: len(chunk) - len(chunk) % 4]
                if len(chunk) < 24:
                    continue
                try:
                    decoded = base64.b64decode(chunk.translate(_URL_SAFE_ALPHABET), validate=True)
                except (ValueError, binascii.Error):
                    continue
                if kind := _match(decoded):
                    return kind
    return None


def _unwrapped(data: bytes) -> bytes:
    """`data` with the line breaks of a fixed-width base64 block removed, joining it into one run.

    MIME base64 breaks every 76 characters (`base64.encodebytes`, mail attachments, many
    `kubectl get -o yaml` outputs) and PEM bodies every 64. The run pattern stops at the newline,
    so a wrapped blob arrived as a series of per-line runs and a credential straddling a break
    decoded into neither -- measured at 20 of 60 possible key offsets missed, which is every key
    that happens to cross a line.

    Only FIXED-WIDTH sequences are joined, never any break between two base64 characters. That
    distinction matters: `KEY=aGVsbG8\\nOTHER=d29ybGQ` is two unrelated values, and welding those
    together would decode arbitrary line pairs and invent credentials that were never there.
    Wrapping is what the width identifies, and a real wrapped blob always has it.

    Guarded by a cheap substring test first. The joining pattern alternates over two widths and
    retries at every offset, which cost most of the scan's throughput when it ran over every chunk
    of every file; almost no file contains a wrapped block, and one that does always contains a
    full-width run of base64 characters. Measured on ordinary source: 9 MB/s without the guard,
    back to the pre-existing rate with it.
    """
    if b"\n" not in data or not _WRAPPED_HINT.search(data):
        return data
    return _WRAPPED_BLOCK.sub(lambda match: _WRAPPED_BREAK.sub(b"", match.group(0)), data)


def _decode_windows(run: bytes) -> Iterator[bytes]:
    """Overlapping slices of a base64 run, each small enough to decode eagerly."""
    if len(run) <= _BASE64_WINDOW:
        yield run
        return
    step = _BASE64_WINDOW - _BASE64_WINDOW_OVERLAP
    for start in range(0, len(run), step):
        yield run[start : start + _BASE64_WINDOW]


def _credential_kind(data: bytes) -> str | None:
    """The kind of credential `data` contains under any of its plausible text encodings.

    A wide encoding interleaves NUL bytes between ASCII characters, so a UTF-16 `env.ps1` holding
    an otherwise-detected key matches none of the byte patterns and published intact. Rather than
    decode every member (most are not text at all), strip the NUL padding of each wide form and
    re-test: that is exact for the ASCII-range characters every one of these credentials is built
    from, and costs nothing on the ordinary UTF-8 file, which has no NULs to strip.

    Narrowed text goes through the SAME checks as literal text, base64 included. Running only the
    plain patterns over it left the two supported encodings composable: a UTF-16 config file holding
    a base64 credential passed both gates individually and published.

    Only the stretches that are ACTUALLY wide-encoded are narrowed, not the whole file. Taking
    every second byte of arbitrary data invents text that was never written: machine code narrows
    into plausible-looking tokens often enough that 5 of 500 ordinary system binaries were refused
    as holding a credential -- `/usr/bin/bash` narrowed to `fslo_eietossvdrdrcsP3`. A real wide
    character keeps its padding byte NUL, so requiring an unbroken NUL run alongside the candidate
    costs nothing on genuine UTF-16/32 and leaves machine code with nothing long enough to match.
    """
    if kind := _decoded_kind(data):
        return kind
    if b"\x00" not in data:
        return None
    for width, keep in ((2, (0, 1)), (4, (0, 3))):
        for offset in keep:
            # take every `width`-th byte: for UTF-16 that is the ASCII half of each code unit, in
            # whichever of the two byte orders the file used.
            for run in _wide_runs(data, width, offset):
                if kind := _decoded_kind(run):
                    return kind
    return None


def _wide_runs(data: bytes, width: int, offset: int) -> Iterator[bytes]:
    """The stretches of `data[offset::width]` whose discarded padding byte is NUL throughout.

    Wide text pads every character with NULs, so the padding column is what separates a genuine
    UTF-16/32 region from bytes that merely narrow into something readable. Only runs long enough
    to hold a credential are yielded, which is why an ELF -- whose NULs are scattered rather than
    columnar -- produces none while a wide file produces one run covering the whole of it.
    """
    pad = offset + 1 if offset + 1 < width else offset - 1
    narrow = data[offset::width]
    columns = data[pad::width].translate(_NUL_MARKER)
    for match in _WIDE_RUN.finditer(columns, 0, min(len(narrow), len(columns))):
        yield narrow[match.start() : match.end()]


def _decoded_kind(data: bytes) -> str | None:
    """The kind of credential in `data` literally, or inside a base64 run within it."""
    return _match(data) or _match_base64(data)


class _Unscannable(Exception):
    """A member could not be scanned to the end, so the publish cannot be vouched for.

    Every limit this module imposes -- time, nesting depth, buffer size -- raises this rather than
    returning None. A limit that returns "no credential found" is indistinguishable from a clean
    scan, so the cheapest way past the whole check is to make it expensive: bury the key deeper
    than the depth cap, or behind a member too large to buffer. Refusing keeps the limits honest
    about what they mean, which is "not verified", not "verified clean".
    """


def _scan_stream(handle: IO[bytes], *, deadline: float | None = None, depth: int = 0) -> str | None:
    """The kind of credential anywhere in `handle`, or None.

    The WHOLE stream is read -- memory is bounded by the chunk size, not the total -- with an
    overlap so a credential straddling a chunk boundary is still matched. Stopping early on a byte
    count would mean a key placed after the cutoff publishes, which is the bug rather than the
    protection.

    `deadline` bounds expansion time when the bytes come from an archive. Exceeding it raises
    `_Unscannable` rather than returning None, so the caller refuses the publish.

    A stream that is ITSELF a container is expanded in turn. Nested containers are buffered to be
    reopened, so one too large to hold in memory also raises: leaving it to the scan of its literal
    bytes looked like a reasonable trade, but a deflated member's bytes hold the credential nowhere
    a pattern can see, so it was a silent bypass reachable by padding an archive past the cap.
    """
    carry = b""
    buffered = bytearray()
    tail = b""
    container_head = False
    overflowed = False
    while chunk := handle.read(_SCAN_CHUNK_BYTES):
        if deadline is not None and time.monotonic() > deadline:
            raise _Unscannable("takes too long to decompress")
        if not carry and _is_openpgp_secret_key(chunk[:24]):
            # only ever at offset 0, and `carry` is empty only on the first chunk. Every file and
            # every archive member reaches this, so the binary export is covered wherever it sits.
            return "a private key"
        if not carry:
            # Past any skippable frames first: zstd and LZ4 both allow a metadata envelope before
            # the real frame, and a head-only check saw that envelope's magic, matched neither
            # list, and passed the compressed frame behind it through as ordinary content.
            head, truncated = _after_skippable_frames(chunk[:_SKIPPABLE_SCAN_BYTES])
            if truncated:
                raise _Unscannable("begins with a frame prelude too long to read past")
            for magic, fmt in _UNEXPANDABLE_MAGIC:
                if head.startswith(magic):
                    raise _Unscannable(f"contains a {fmt} archive this check cannot expand")
        if not carry and depth:
            # tar as well as the compressed magics: a tar's own members are literal, but a
            # COMPRESSED member inside one is not, so an oversized tar that went past the buffer
            # cap is as unverifiable as an oversized gzip and must refuse rather than pass.
            container_head = _looks_compressed(chunk[:6]) or _looks_like_tar(chunk)
        if depth and not overflowed:
            buffered.extend(chunk)
            if len(buffered) > _MAX_NESTED_BUFFER_BYTES:
                if container_head:
                    raise _Unscannable("contains an archive too large to inspect")
                # Not a container by its head, so the literal scan below is complete coverage and
                # the buffer is only needed to REOPEN a container. Dropping it keeps memory bounded
                # on an ordinary large member; `tail` still decides at the end whether what went
                # past was a zip hiding behind a preamble.
                overflowed = True
                buffered = bytearray()
        if depth:
            tail = (tail + chunk)[-_ZIP_TAIL_BYTES:]
        window = carry + chunk
        if kind := _credential_kind(window):
            return kind
        carry = window[-_SCAN_OVERLAP_BYTES:]
    if overflowed and _has_zip_end_record(tail):
        raise _Unscannable("contains an archive too large to inspect")
    if buffered and _looks_like_container(bytes(buffered)):
        return _credential_in_container(bytes(buffered), deadline=deadline or 0.0, depth=depth + 1)
    return None


def _looks_like_container(data: bytes) -> bool:
    """Whether `data` is a container worth reopening, by magic OR by zip structure.

    Nested members were tested on LEADING magic alone while top-level files got `is_zipfile`, so a
    self-extracting zip one layer in -- whose first bytes are `MZ` -- was treated as final content
    and the credential in its deflated payload published. `is_zipfile` scans for the end-of-central
    -directory record, so it recognises a zip behind any preamble; applying it here makes a nested
    member as well covered as the same bytes published directly.

    Gating the recursion on this rather than recursing unconditionally keeps `_MAX_CONTAINER_DEPTH`
    honest: the depth cap raises, so calling it for an ordinary deeply-nested *file* would refuse a
    legitimate publish over nesting that never expanded anything.

    Tar counts here too. A tar's own member bytes are literal, so a top-level one needed no special
    handling for its plain members -- but nested it is a container like any other, and `tar.gz`
    holding a tar of gzipped shards left the innermost key unreached.
    """
    return (
        _looks_compressed(data[:6]) or _looks_like_tar(data) or zipfile.is_zipfile(io.BytesIO(data))
    )


def _credential_in_container(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential inside a compressed container, or None.

    A deflated zip member or a gzipped shard does not contain its credential anywhere in the file,
    so scanning the container's own bytes cannot see it -- and `flash env push` happily publishes a
    `.zip` holding an `env.sh`.

    Expansion RECURSES to `_MAX_CONTAINER_DEPTH`. Stopping at one level treated an inner member's
    bytes as final content, so a zip holding a gzipped shard -- an ordinary way to ship a dataset --
    hid a key one layer further in and published it.

    A container that will not open is not an error, and neither is one that fails partway through
    reading. An unsupported, truncated or corrupt archive falls back to the literal scan of its own
    bytes, which is the coverage it had before. Crashing on it would be a worse bug than the hole
    being closed: a half-written shard in a dataset directory is ordinary.

    Each zip member is guarded SEPARATELY. Wrapping the whole loop meant one unreadable entry
    abandoned every entry behind it, so a single encrypted member at the top of an archive hid a
    real key further down and the publish succeeded -- the same silent-pass bypass the expansion
    budget refuses, arriving through error handling instead.
    """
    if depth > _MAX_CONTAINER_DEPTH:
        raise _Unscannable("nests compressed containers too deeply to inspect")
    # EVERY applicable format is tried, not just the first one that claims a match. The detectors
    # are heuristics over untrusted bytes and one of them was decisive: `is_zipfile` searches the
    # last 64 KiB for the end-of-central-directory record, so four bytes of `PK\x05\x06` ANYWHERE
    # in a tar made it claim the file. The tar was then opened as a zip, failed, and the failure
    # was read as "nothing here" -- so adding four stray bytes to any member of a tar took a
    # gzipped credential inside it from refused to published. Falling through instead means a
    # wrong guess costs an open attempt rather than the whole scan.
    #
    # A handler that REFUSES is deferred rather than allowed to end the loop. Its limits are
    # applied to bytes that may not be its format at all -- a tar carrying a fake end-of-central
    # -directory record made the zip handler refuse for "too many members", and because
    # `_Unscannable` is deliberately not in `_UNREADABLE_ARCHIVE`, that refusal escaped before the
    # tar handler ever ran. The credential inside was real and went unreported. A handler that
    # COMPLETES has genuinely enumerated the bytes, so its answer settles the question; the
    # deferred refusal is re-raised only when no handler managed that, which keeps a real
    # oversized archive fail-closed.
    refusal: _Unscannable | None = None
    for handler in (_credential_in_zip, _credential_in_tar, _credential_in_compressed):
        try:
            if kind := handler(source, deadline=deadline, depth=depth):
                return kind
        except _Unscannable as unscannable:
            refusal = refusal or unscannable
        except _UNREADABLE_ARCHIVE:
            continue  # not this format, or corrupt in it; the remaining formats still get a turn
    if refusal is not None:
        raise refusal
    return None


def _credential_in_compressed(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential inside a gzip, bzip2 or xz stream, or None."""
    head = source[:6] if isinstance(source, bytes) else source.open("rb").read(6)
    opener = {b"BZh": bz2.open, b"\xfd7zXZ\x00": lzma.open}.get(
        next((magic for magic in (b"BZh", b"\xfd7zXZ\x00") if head.startswith(magic)), b""),
        gzip.open,
    )
    with opener(source if isinstance(source, Path) else io.BytesIO(source), "rb") as stream:
        return _scan_stream(stream, deadline=deadline, depth=depth)


def _credential_in_zip(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential in any readable member of a zip, or None."""
    # The member count is read from the end-of-central-directory record BEFORE `ZipFile` is
    # constructed. `ZipFile.__init__` parses the whole central directory and materializes every
    # `ZipInfo`, so a bound checked after it is charged the cost it exists to avoid -- measured at
    # 1.8 seconds and 239 MB of resident memory for 400,000 empty entries in a 35 MB file, all of
    # it spent before the per-member loop below ran once.
    if _zip_member_count(source, _MAX_ARCHIVE_MEMBERS) > _MAX_ARCHIVE_MEMBERS:
        raise _Unscannable("contains an archive with too many members to inspect")
    unreadable = ""
    with zipfile.ZipFile(source if isinstance(source, Path) else io.BytesIO(source)) as archive:
        for count, info in enumerate(archive.infolist(), 1):
            if count > _MAX_ARCHIVE_MEMBERS:
                raise _Unscannable("contains an archive with too many members to inspect")
            if time.monotonic() > deadline:
                raise _Unscannable("takes too long to decompress")
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
                    if kind := _scan_stream(member, deadline=deadline, depth=depth):
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
            except _UNREADABLE_ARCHIVE:
                continue  # this member is opaque; the rest of the archive still gets scanned
    if unreadable:
        raise _Unscannable(f"contains {unreadable}")
    return None


def _credential_in_tar(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential in any readable member of a tar, or None.

    A tar is not itself compressed, so its members' bytes are read by the ordinary scan already --
    but a COMPRESSED member inside one is not: `tar > shard.gz` holds the credential nowhere a
    pattern can see, exactly like `zip > shard.gz`, and only the zip form was ever expanded.
    Enumerating members hands each one to `_scan_stream`, which expands it if it is a container.

    Streamed with `r|*` rather than `r:*`: the streaming reader does not seek back over the archive,
    so a member's data is read once, in order. Members are guarded separately for the same reason
    they are in a zip -- one unreadable entry must not abandon the entries behind it.
    """
    handle = source.open("rb") if isinstance(source, Path) else io.BytesIO(source)
    try:
        with tarfile.open(fileobj=handle, mode="r|*") as archive:
            for count, info in enumerate(archive, 1):
                if count > _MAX_ARCHIVE_MEMBERS:
                    raise _Unscannable("contains an archive with too many members to inspect")
                if time.monotonic() > deadline:
                    raise _Unscannable("takes too long to decompress")
                if not info.isfile():
                    continue
                # the member NAME is checked too: a tar entry called `fslo_<key>.json` publishes
                # the key in the archive's listing whatever its contents are.
                if kind := credential_in_name(info.name):
                    return kind
                try:
                    member = archive.extractfile(info)
                    if member is None:
                        continue
                    if kind := _scan_stream(member, deadline=deadline, depth=depth):
                        return kind
                except _UNREADABLE_ARCHIVE:
                    continue
    finally:
        handle.close()
    return None


def credential_in_file(path: Path, *, deadline: float | None = None) -> str | None:
    """The kind of credential publishing `path` would leak, or None.

    Scanned as raw bytes, binary members included. Skipping binaries would be a hole rather than a
    saving: a credential sitting in a sqlite state file or a pickle is as published as one in a
    shell script, and the prefixes above cannot realistically collide with random bytes.

    `deadline` is the expansion budget, shared across a whole package when one is being scanned.
    A per-file budget multiplied by the member limit, which let an authenticated caller split
    compression bombs across thousands of files and buy hours of expansion. Left unset, one file
    gets its own budget, which is the right behaviour for a standalone call.

    Raises `_Unscannable` if an archive is too expensive to finish expanding, which the
    caller turns into a refusal: unverifiable is not the same as clean.
    """
    if deadline is None:
        deadline = time.monotonic() + _MAX_DECOMPRESS_SECONDS
    with path.open("rb") as handle:
        # The package budget covers the file's own bytes too. Leaving it off looked safe because
        # the read is bounded by the package size limit, but the limit bounds BYTES and this
        # bounds TIME: matching cost is not uniform per byte, so a large file of adversarial
        # near-matches held a worker far longer than its size suggested.
        if kind := _scan_stream(handle, deadline=deadline):
            return kind
    # `is_zipfile` is consulted inside, so a self-extracting archive is expanded despite its stub
    return _credential_in_container(path, deadline=deadline, depth=1)


def credential_in_name(name: str) -> str | None:
    """The kind of credential the path `name` itself carries, or None.

    A file whose *name* is the key leaks it through the archive's member list even when its
    contents are empty, and the published repo shows that name in its tree forever.

    `surrogatepass` rather than `surrogateescape`: the latter is a DECODE-only handler, so a name
    holding a lone surrogate raised `UnicodeEncodeError` out of a security check. Reached from the
    publish route that check turned a 400 into an uncaught 500, and reached from a tar member name
    it crashed the scan of an archive rather than reporting what was in it.
    """
    return _credential_kind(name.encode("utf-8", "surrogatepass"))


def _redacted(name: str) -> str:
    """`name` with any credential body masked, safe to print.

    The refusal names the member so the author can find it, and when the credential is IN that name
    printing it verbatim re-leaks the key into a terminal and whatever collects its output -- the
    one thing the rest of this module is careful never to do. The issuer prefix and the surrounding
    path survive, which is all that is needed to locate the file.
    """

    def _mask(match: re.Match[bytes]) -> bytes:
        body = next((index for index, group in enumerate(match.groups(), 1) if group), None)
        if body is None:
            return match.group(0)
        return match.group(0)[: match.start(body) - match.start()] + b"***"

    # `surrogatepass` for the same reason as `credential_in_name`: `surrogateescape` cannot encode
    # a lone surrogate, and this runs while REPORTING a refusal, so the crash would replace the
    # message naming the credential.
    masked = name.encode("utf-8", "surrogatepass")
    for _kind, pattern in _TOKEN_PATTERNS + _ASSIGNED_PATTERNS:
        masked = pattern.sub(_mask, masked)
    # A name detected only through base64, or as a whole key structure, has no plaintext body to
    # mask, so masking cannot help: printing any of it prints the key. A compact private JWK fits
    # in a 129-character filename, and every pattern above left it untouched, so the refusal
    # printed the complete Ed25519 scalar to the terminal and any collected logs. Withhold the
    # name and give the author the directory instead, which is enough to find a file they just
    # tried to publish.
    if any(pattern.search(masked) for _kind, pattern in _LITERAL_PATTERNS) or _match_base64(masked):
        parent = name.rsplit("/", 1)[0] if "/" in name else ""
        return (
            f"{parent}/<a file whose name encodes a credential>"
            if parent
            else ("<a file whose name encodes a credential>")
        )
    return masked.decode("utf-8", "replace")


def reject_credential_bearing_package(package_root: Path, *, display: dict[str, str]) -> None:
    """Refuse the publish if any member of the staged package carries a credential.

    Takes the STAGED package rather than the source tree, so what is scanned is exactly what is
    uploaded. Scanning the source instead left three holes: the generated README (which embeds
    `--name`, so a key passed there was published verbatim), the generated entrypoint alias, and
    the window between reading the source and copying it, in which any local process rewriting a
    file put unscanned bytes into the archive.

    Raises ValueError naming the member and the credential kind. Refusing rather than quietly
    dropping the file is deliberate: the author needs to rotate a key that has been sitting in a
    directory they just tried to publish, and a silent drop teaches them nothing.

    One expansion budget covers the whole package. Giving each file its own multiplied it by the
    member limit, so splitting compression bombs across thousands of members bought hours of work
    while every individual file looked well inside the limit.
    """
    deadline = time.monotonic() + _MAX_DECOMPRESS_SECONDS

    def unwalkable(error: OSError) -> NoReturn:
        # `os.walk` swallows descent errors by default, so a directory the scan cannot enter --
        # mode 000 in an uploaded tar, read by a non-root control plane -- had its contents skipped
        # silently and a credential inside was published intact. The same tree with the directory
        # readable is refused, so passing here was purely a function of what could be opened.
        #
        # Named relative to the package like every other refusal: the absolute path is the control
        # plane's staging directory, which is the server's business rather than the publisher's.
        failed = Path(str(error.filename or package_root))
        relative = failed.relative_to(package_root).as_posix() if failed != package_root else "."
        raise ValueError(
            f"{_redacted(display.get(relative, relative))} could not be read to check it for "
            "credentials, so publishing it would commit unscanned content to a shared environment "
            "repository. Fix the permissions on it before publishing."
        ) from None

    # sorted so the member named in the refusal is the same one on every machine.
    for root, dirs, files in os.walk(package_root, onerror=unwalkable):
        dirs.sort()
        for name in sorted(dirs) + sorted(files):
            member = Path(root) / name
            relative = member.relative_to(package_root).as_posix()
            shown = _redacted(display.get(relative, relative))
            try:
                # the NAME is checked too, directories included: a file called `fslo_<key>.json`
                # publishes the key in the repository's file tree whatever its contents are.
                kind = credential_in_name(relative) or (
                    credential_in_file(member, deadline=deadline) if member.is_file() else None
                )
            except _Unscannable as exc:
                raise ValueError(
                    f"{shown} {exc}, so it cannot be checked for credentials. Publishing it would "
                    "commit unscanned content to a shared environment repository. Unpack the "
                    "archive before publishing."
                ) from None
            if not kind:
                continue
            raise ValueError(
                f"{shown} contains what looks like {kind}. "
                "Publishing would commit it to a shared environment repository, permanently in "
                "git history. Remove the credential from the environment directory and rotate it "
                "before publishing."
            )
