"""What a credential looks like, and how to recognise one in a span of bytes.

Split out of `flash.env_secrets` to keep that module under the file-size limit. It holds the
patterns and the matching over them; the scanning, decoding and container expansion that decide
WHICH bytes to match against stay there. The dependency runs one way: this module knows nothing
about files, archives or the publish.
"""

from __future__ import annotations

import re
from typing import Protocol

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

# The shortest plaintext any pattern above admits: `xoxb-` plus a 10-character body. Stated here,
# beside the patterns that determine it, so lowering a body minimum cannot silently leave the
# base64 floor derived from it too high.
#
# Kept as a plain constant rather than computed from the compiled patterns. Deriving it means
# parsing regex source to find each alternative's prefix and repetition minimum, which is more
# machinery than the number is worth and gets the count wrong in exactly the quiet direction --
# too HIGH, which reopens the bypass this exists to close.
SHORTEST_TOKEN_BYTES = 15

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
#
# `_BLOCK_SCALAR` is what sits between the key and the body. Ordinarily nothing, but YAML may put
# the value on the FOLLOWING lines instead: `KEY: |` (literal) or `KEY: >-` (folded), then the body
# indented beneath. `\s*` alone does not cover it -- the `|` and the optional chomping indicator
# are not whitespace -- so a key written in the commonest multi-line YAML form matched nothing and
# published. A `secrets.yaml` or a Helm values file is an ordinary thing to keep beside an
# environment, and both write keys this way.
#
# The two indicators may appear in EITHER order. YAML 1.2 defines the block header as an
# indentation indicator and a chomping indicator in any order (`|2-` and `|-2` are the same
# scalar), and admitting only sign-then-digit meant `|2-` and `>2+` left the `|` unconsumed, the
# body then failed to match, and the key published. Both orders are named because a writer that
# emits an explicit indentation indicator -- which is what ruamel and several Helm chart
# generators do when the first body line is itself indented -- naturally puts the digit first.
#
# A trailing COMMENT is allowed after the indicators. YAML permits `KEY: | # generated`, which is
# what templating tools annotate injected values with, and stopping at the `#` left the body
# unmatched so the key published. Only the remainder of the header LINE is consumed -- the body
# begins on the next one -- so this cannot swallow a value that follows on the same line.
_BLOCK_SCALAR = rb"(?:[|>](?:[+-][0-9]?|[0-9][+-]?)?[ \t]*(?:\#[^\r\n]*)?\s*)?"
# YAML node properties, which may sit between the assignment and the scalar: a tag (`!!str`, or a
# named handle such as `!ruby/object`) and/or an anchor (`&name`). Either order, either alone, and
# a tag may be followed by an anchor. The parser yields the same scalar, so
# `AWS_SECRET_ACCESS_KEY: !!str <key>` and `... : &aws <key>` are live credentials that matched
# nothing -- the body was expected immediately after the block header and the optional quote.
#
# Deliberately narrow: a tag is `!` plus tag characters, an anchor `&` plus non-space. Neither can
# contain a quote or whitespace, so this cannot swallow the value it precedes.
_NODE_PROPERTIES = rb"(?:(?:![!A-Za-z0-9_/:.-]*|&[^\s\"']+)[ \t]+)*"
# The opening quote, if any. A single optional quote character consumed only ONE of the three in a
# triple-quote delimiter, leaving a quote sitting where the body had to begin, so an ordinary
# Python or TOML multiline assignment matched nothing and published. Whole delimiters are named,
# longest first, so all three forms are consumed together.
_OPEN_QUOTE = rb"(?:\"\"\"|'''|[\"'])?"
# Every private-scalar length `openssl ecparam -list_curves` produces: 25 distinct values from 14
# bytes (`secp112r1`) to 114 (`sect571r1`). Used to tie a SEC1 key's stated OCTET STRING length to
# the scalar that follows it.
#
# The floor is 14, not the 20 a first pass at this used. Twelve curve families sit below 20 --
# secp112, sect113, secp128, sect131 and their WAP/WTLS aliases -- and every one of them published
# its key intact. They are small and legacy, but a 112-bit private key is still a private key, and
# the point of reading the length rather than listing sizes is not to have a floor that guesses.
_SEC1_SCALAR_BYTES = range(0x0E, 0x73)


def _json_escapable(text: bytes, *, fold_case: bool = False) -> bytes:
    """`text` as a pattern matching itself or any character written as its `\\u00XX` escape.

    JSON says the two spellings are the same string, so a parser loads `"R\\u0053A"` as `RSA` and
    a literal-byte pattern sees neither. Applied to member NAMES and to the `kty` VALUE: escaping
    only the names left the key-type marker matchable by a one-character escape.

    `fold_case` makes each position match the escape of EITHER case of the character, for a name
    whose surrounding pattern is case-insensitive. Without it, wrapping the result in `(?i:...)`
    covers only the literal half: the escape carries the character's code point, so `SecretAccessKey`
    written `SecretAccess\\u004bey` needs the code point of `K` and a case-folded LITERAL cannot
    supply it. An escaped AWS field name published its secret intact for exactly that reason.
    """
    codes = (
        (lambda byte: (byte, byte ^ 0x20) if bytes([byte]).isalpha() else (byte,))
        if fold_case
        else (lambda byte: (byte,))
    )
    return b"".join(
        rb"(?:%s|(?i:\\[uU]00(?:%s)))"
        % (re.escape(bytes([byte])), b"|".join(b"%02x" % code for code in codes(byte)))
        for byte in text
    )


_ASSIGNED_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "a Weights & Biases API key",
        re.compile(
            # escapable per character for the same reason as the AWS name below: a `wandb` block in
            # a JSON config is an ordinary place for this key to sit, and the escaped spelling
            # loads identically.
            rb"(?i:"
            + _json_escapable(b"wandb_api_key", fold_case=True)
            + rb")[\"']?\s*[:=]\s*"
            + _BLOCK_SCALAR
            + _NODE_PROPERTIES
            + _OPEN_QUOTE
            + rb"([A-Za-z0-9_-]{40,%d})" % _MAX_BODY
        ),
    ),
    (
        "an AWS secret access key",
        re.compile(
            # `SecretAccessKey` as well as the environment-variable name: that is the field the
            # SDKs, `sts assume-role` and a credential-process document write, and it is the shape
            # a saved session lands in on disk. Anchoring on the env-var name alone meant those
            # published intact. Word-boundary-free on the left so the `aws_` prefix stays optional
            # without admitting a longer unrelated identifier ending in the same characters.
            # Both spellings admit `\u00XX` escapes per character. A credential document is JSON
            # more often than not, `"SecretAccessKey"` is the SAME field name to every parser,
            # and a literal-byte name saw neither it nor an escaped `AWS_SECRET_ACCESS_KEY` -- so
            # the identical 40-character secret published clean under a one-character escape.
            rb"(?i:"
            + _json_escapable(b"aws_secret_access_key", fold_case=True)
            + rb"|(?<![A-Za-z0-9_])"
            + _json_escapable(b"secretaccesskey", fold_case=True)
            + rb")[\"']?\s*[:=]\s*"
            + _BLOCK_SCALAR
            + _NODE_PROPERTIES
            + _OPEN_QUOTE
            # `\/` is a legal JSON escape for `/`, and an AWS secret is base64 so it carries `/`
            # about a third of the time. Encoders that escape it -- PHP's `json_encode` by default,
            # and several SDK log formatters -- broke the run of 40 into two shorter runs, so the
            # SAME key published clean purely because of how the document was serialized. Each
            # position admits the escaped spelling; the count stays 40 DECODED characters.
            + rb"((?:[A-Za-z0-9+=]|\\?/){40})(?![A-Za-z0-9/+=])"
        ),
    ),
)


class _Searchable(Protocol):
    """What this module needs of a pattern: `search(data)` returning a match or None.

    `re.Pattern[bytes]` satisfies it, and so does a detector built from more than one pattern.
    """

    def search(self, data: bytes, /) -> re.Match[bytes] | None: ...


# The `kty` half of a JWK, named at module level because `_scan_stream` needs it too: the two
# markers may sit further apart than one read chunk, and a chunked scan that only ever sees a
# window cannot pair them without remembering that the `kty` went past.
#
# Its name is escapable exactly like the private members below, and for the same reason: escaping
# only the `kty` half left the pair unmatched even when the private member was spelled plainly.


_JWK_KTY = re.compile(
    rb"\""
    + _json_escapable(b"kty")
    + rb"\"\s*:\s*\"(?:"
    + b"|".join(_json_escapable(kind) for kind in (b"RSA", b"EC", b"OKP", b"oct"))
    + rb")\""
)
# `d` is the private exponent or scalar in every key type; for RSA the CRT parameters accompany it.
# `k` is the symmetric case: an `oct` JWK holds its whole secret there and has no `d` at all, so
# naming `oct` above without it accepted the one key type where the secret IS the file -- what an
# HMAC signing key or an `A256GCM` content key exports as. A public JWK carries none of these,
# which is exactly what separates the two.
#
# Each name character is written as itself OR as its `\u00XX` escape, because JSON says the two
# are the same string and every parser agrees: `"\u0064"` IS `"d"`, so a key whose private member
# is spelled that way loads identically and exports identically, while a literal-byte pattern saw
# no `d` at all and published it. Escaping is per-character rather than whole-name so a mixed
# spelling (`"d\u0070"` for `dp`) is covered too.
#
# The hex digits are matched case-insensitively per RFC 8259, and so is the `u`: `\u0064` and
# `\U0064` name the same character. Every one of these names is ASCII, so two hex digits after
# `00` are always enough.
_JWK_ESCAPED = {
    name: b"".join(rb"(?:%c|(?i:\\[uU]00%02x))" % (byte, byte) for byte in name)
    for name in (b"d", b"dp", b"dq", b"qi", b"k")
}
#
# The VALUE admits escapes too, for the same reason the name does. A base64url scalar is all
# ASCII, so any of its characters may legally be written `\u00XX` -- a Node-exported JWK whose `d`
# begins `"\u0078..."` is the same key to `JSON.parse` and `createPrivateKey`, but a run of plain
# base64 characters matched nothing and the whole private key published. Each position is either
# one literal character or one six-character escape, and the LENGTH floor counts positions rather
# than bytes, so escaping cannot shrink a value below it either.
_JWK_VALUE_CHAR = rb"(?:[A-Za-z0-9+/\-_]|(?i:\\[uU]00[0-9a-f]{2}))"
_JWK_PRIVATE = re.compile(
    rb"\"(?:"
    + b"|".join(_JWK_ESCAPED[name] for name in (b"dp", b"dq", b"qi", b"d", b"k"))
    # longest first: `d` would otherwise win against `dp` and leave the `p` outside the quote
    + rb")\"\s*:\s*\"("
    # CAPTURED so the value goes through `_is_high_entropy` like every other pattern's body. An
    # uncaptured value made the pair fire on any long string under a private member name, so a
    # JSONL dataset with `{"d":"documentation-document"}` in one row and an ordinary public JWK in
    # another was refused as a private key -- the halves pair across the whole stream, so unrelated
    # rows combined. A real `d` is a base64url scalar and passes; an English word does not.
    + _JWK_VALUE_CHAR
    + rb"{20,})={0,2}\""
)


class _TwoMarkers:
    """A credential identified by two markers that may sit any distance apart, in either order.

    Presented as a pattern object because `_match` iterates `(kind, pattern)` pairs and calls
    `.search`. The pair cannot be one regex: a span between them (`[\\s\\S]{0,N}?`) is wrong in
    both directions -- it misses a real key whose halves sit further apart than N, and it
    backtracks over every position in between, which cost 4.2 seconds per MiB on near-matching
    input. Searching for each independently is exact at any distance and linear.

    `context` is the half that identifies the FORMAT and `payload` the half that carries the
    secret; the payload's match is returned so a caller can report where it is.
    """

    def __init__(self, context: re.Pattern[bytes], payload: re.Pattern[bytes]) -> None:
        self.context = context
        self.payload = payload

    def payload_match(self, data: bytes) -> re.Match[bytes] | None:
        """The payload half's match in `data` whose captured body looks issued, or None.

        The captured body goes through the same entropy test as every other pattern's. The comment
        on the netrc pair says placeholders are filtered by it, but nothing applied it: this class
        sits in `_LITERAL_PATTERNS`, whose loop returns on a match rather than inspecting groups,
        so ordinary prose containing `Machine` and a masked `password XXXX...` was refused as a
        credential, and a dataset row whose long `"d"` field is an English word was reported as a
        private key.

        A method rather than a filter inside `search`, because the CHUNKED scan does not call
        `search`: it pairs the halves across the whole stream by calling `context` and `payload`
        itself. Filtering in only one of the two left the streaming path -- which is every file
        over a megabyte, and every file this check actually reads -- matching placeholders exactly
        as before, so both paths ask this one question instead.
        """
        for payload in self.payload.finditer(data):
            body = next((found for found in payload.groups() if found), b"")
            if _is_high_entropy(body):
                return payload
        return None

    def search(self, data: bytes) -> re.Match[bytes] | None:
        """The payload's match when the context marker accompanies it anywhere in `data`."""
        if not (payload := self.payload_match(data)):
            return None
        return payload if self.context.search(data) else None


# An all-hex body of key length. Recognised so `_is_high_entropy` admits a legacy 40-hex W&B key,
# whose lowercase letters would otherwise read as the hand-written-placeholder convention.
_HEX_BODY = re.compile(r"[0-9a-fA-F]{32,}")


# A PEM header only means a key is present when the BODY follows it. The header alone is something
# documentation says -- "if you see -----BEGIN RSA PRIVATE KEY----- in a log, redact it" -- and
# refusing on it blocks a legitimate publish over prose about credentials. Requiring a base64 line
# after the header keeps every real key (they all carry one) and drops the mention of one.
#
# The second alternative is the encrypted form, whose body is preceded by RFC 1421 headers instead
# of starting with base64. It names those two headers exactly: a general `[A-Za-z-]+:` also accepts
# `Warning:` or `Note:`, which is prose about a key rather than a key, and reopens the very false
# positive the base64 requirement exists to close.
# The two halves of a `.netrc` entry. `machine <host>` opens an entry and is the keyword that makes
# the file a credential store; `password <token>` carries the secret.
#
# Netrc is TOKEN-separated rather than line-oriented: `curl` and several generators write a whole
# entry on one line, and anchoring either half to a line start missed exactly that form. So each
# half requires whitespace (or a boundary) before its keyword, which still keeps a sentence
# mentioning either word out while matching the entry however it is laid out. `default` opens an
# entry too, and is the fallback form. The password body must be key-length and goes through
# `_is_high_entropy` like every other body, which is what keeps `password changeme` and a
# documented placeholder out.
#
# The body stops at whitespace because netrc is token-separated. A quoted value is admitted too:
# `password "..."` is what a generator writes when the token could contain a space.
_NETRC_MACHINE = re.compile(rb"(?i)(?:^|\s)(?:machine[ \t]+\S+|default)(?:\s|$)")
_NETRC_PASSWORD = re.compile(
    rb"(?i)(?:^|\s)password[ \t]+[\"']?([A-Za-z0-9/+=_-]{32,%d})" % _MAX_BODY
)


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
    #
    # The 32 body characters are counted ACROSS line breaks, not within one line. PEM does not fix
    # a wrap width -- RFC 7468 recommends 64 but permits any -- and `openssl pkey -check` accepts a
    # key rewrapped at 16 columns as valid. Requiring 32 contiguous characters meant such a key
    # matched neither this pattern nor the 64/76-column wrapped-base64 joining, and a complete
    # Ed25519 PKCS#8 private key published. Each character may be followed by a break, so the floor
    # counts body characters however they are laid out.
    (
        "a private key block",
        re.compile(
            rb"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----[\r\n\s]*"
            rb"(?:(?:Version|Comment|MessageID|Hash|Charset):[^\r\n]*[\r\n\s]*)*"
            rb"(?:(?:[A-Za-z0-9+/=][ \t]*\r?\n?[ \t]*){32,}|Proc-Type:|DEK-Info:)"
        ),
    ),
    # PuTTY's own key format, which PuTTYgen writes and `pageant`/`plink` read. It is neither PEM
    # nor DER -- no `-----BEGIN` header, no ASN.1 -- so every structure here passed a complete
    # unencrypted private key as clean, and `.ppk` is not in the filename exclusions either.
    #
    # Anchored on the header together with the `Private-Lines` body, for the same reason the PEM
    # pattern requires base64 after its header: the header alone appears in documentation and in
    # the PUBLIC half that PuTTYgen also exports, and refusing on it would block prose about keys.
    #
    # Two independent markers rather than one pattern spanning both, for the same reason the JWK
    # detector is built that way. A `[\s\S]{0,512}?` span between them was wrong twice over: an
    # RSA-4096 public section base64-encodes to ~716 characters, so `Private-Lines` necessarily
    # fell outside the cap and a complete `.ppk` published (its payload is SSH mpints, not DER, so
    # nothing downstream caught it); and the lazy span backtracks over every position in between.
    (
        "a private key",
        _TwoMarkers(
            re.compile(rb"PuTTY-User-Key-File-\d+:[^\r\n]*[\r\n]"),
            # The body is CAPTURED so it goes through the same entropy test as every other
            # pattern's, which is what keeps a documented placeholder out of the payload half.
            re.compile(rb"Private-Lines:\s*\d+[\r\n]+([A-Za-z0-9+/=]{32,})"),
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
            #
            # Version `1` as well as `0`: RFC 8017 defines `two-prime(0)` and `multi(1)`, and a
            # real three-prime key from `openssl genrsa -primes 3` (which `openssl rsa -check`
            # accepts) begins `02 01 01` instead. Its private factors published intact.
            rb"|\x30\x82..\x02\x01[\x00\x01]\x02(?:\x82..|\x81.|[\x40-\x7f])\x00"
            # SEC1 ECPrivateKey: version 1, the private scalar as an OCTET STRING, then the [0]
            # curve parameters. Naming only 32, 48 and 66 covered P-256/384/521 and missed every
            # other supported curve, so real `prime192v1` (24) and `secp224r1` (28) keys published
            # intact. `openssl ecparam -list_curves` spans 20 to 114 bytes, so the length byte is
            # enumerated across that range with the scalar width tied to it -- a DER length cannot
            # be back-referenced as a repeat count, so the alternation is built rather than
            # written out. The `\xa0` landing exactly where the stated length ends is what keeps
            # this specific: an arbitrary `02 01 01 04` run does not satisfy it.
            rb"|\x02\x01\x01\x04(?:"
            # `re.escape` on the length byte: emitting it raw turns a length such as 0x2a into a
            # literal `*`, which is a repeat operator with nothing to repeat and fails to compile.
            + b"|".join(re.escape(bytes([size])) + rb".{%d}" % size for size in _SEC1_SCALAR_BYTES)
            + rb")\xa0"
            # EncryptedPrivateKeyInfo: `openssl pkcs8 -topk8 -passout` in DER. The plaintext key is
            # inside an OCTET STRING, so none of the structures above appear anywhere in the file
            # and it published intact -- the ARMOURED form of the same key was caught by its
            # `-----BEGIN ENCRYPTED PRIVATE KEY-----` header, which made DER the way past.
            #
            # Anchored on the encryption-algorithm OID in the AlgorithmIdentifier: PBES2
            # (1.2.840.113549.1.5.13) or a pkcs-12 PBE (1.2.840.113549.1.12.1.x). A passphrase is
            # not much protection for a key in a public repository, and the OIDs appear only in a
            # key that is actually encrypted.
            #
            # `\x05.` rather than `\x05\x0d` covers PBES1 alongside PBES2: the whole
            # `1.2.840.113549.1.5.x` arc is password-based encryption, and only `13` was named. A
            # key written by `openssl pkcs8 -topk8 -v1 PBE-SHA1-DES` (or `-v1 PBE-MD5-DES`, or
            # `-v1 PBE-SHA1-RC2-64`) carries `05 03`, `05 0a` or `05 0b` and passed as clean. The
            # arc holds nothing but PBE algorithms, so widening it admits no other structure.
            rb"|\x30.{1,4}?\x30.\x06\x09\x2a\x86\x48\x86\xf7\x0d\x01\x05."
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
        _TwoMarkers(_JWK_KTY, _JWK_PRIVATE),
    ),
    (
        "a machine password in a netrc file",
        # A `.netrc` is how `wandb login`, `huggingface-cli` and `curl` persist a credential, and
        # its password line names no service: `password <40 hex>` has neither an issuer prefix nor
        # an assignment for the patterns above to anchor on. The CLI's filename filter does not
        # cover `.netrc` either, and the server accepts whatever is uploaded -- so a standard W&B
        # netrc passed every check and committed a live key to the shared hub.
        #
        # Two markers rather than one span, for the same reason as the JWK above: the `machine`
        # line and the `password` line are usually adjacent but need not be -- the format is
        # whitespace-separated tokens, `login` may sit between them in either order, and a
        # multi-host netrc puts whole entries in between. Requiring a window would miss those while
        # a span would backtrack over every position between them.
        #
        # The `machine` half is what makes this a credential store rather than prose: the word
        # `password` alone appears in documentation, in a config schema, and in any English text.
        # Requiring the netrc keyword AND a key-length high-entropy body is what separates the file
        # from writing about one.
        _TwoMarkers(_NETRC_MACHINE, _NETRC_PASSWORD),
    ),
)


# The two-marker detectors, exposed so the chunked scan can pair their halves across chunk
# boundaries. Derived from `_LITERAL_PATTERNS` rather than listed again, so a detector added there
# is covered by the cross-chunk pairing automatically instead of silently regressing to
# within-one-window matching.
_PAIRED_PATTERNS: tuple[tuple[str, _TwoMarkers], ...] = tuple(
    (kind, pattern) for kind, pattern in _LITERAL_PATTERNS if isinstance(pattern, _TwoMarkers)
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


# The keyword each expensive pattern is anchored on, in every spelling a substring test can see.
# A buffer holding none of them cannot match that pattern, so the regex is skipped -- see
# `_keyword_absent`. Keyed by the pattern's kind so a pattern without an entry is simply never
# guarded, which is the safe default: a missing entry costs time, never a missed credential.
#
# Lowercase, because the guard tests against a lowercased copy. Only the patterns measured as
# expensive are listed; the issuer-prefixed tokens are already cheap literal prefixes.
_PATTERN_KEYWORDS: dict[str, tuple[bytes, ...]] = {
    "an AWS secret access key": (b"secretaccesskey", b"secret_access_key"),
    "a Weights & Biases API key": (b"wandb_api_key",),
    "a machine password in a netrc file": (b"password",),
}

# A JSON `\u00XX` escape, which can spell any of the keywords above in a way no substring test
# sees. Its presence disarms the guard so the full pattern runs -- the escaped spellings are
# exactly what those patterns were widened to catch, and a guard that skipped them would close the
# regex and reopen the bypass in the same change.
_ESCAPE_HINT = re.compile(rb"\\[uU]00")

# Below this size the lowercased copy costs more than the guards save, so they are not applied.
# A small buffer runs every pattern as before.
_GUARD_MIN_BYTES = 4096


def _keyword_absent(data: bytes, lowered: bytes | None, kind: str) -> bool:
    """Whether a keyword-anchored pattern cannot possibly match, tested by substring.

    A necessary condition, never a sufficient one: every pattern below is anchored on a fixed
    keyword, so a buffer containing none of that keyword's spellings cannot match it however the
    rest of the pattern is written. The regex is then skipped entirely.

    Worth doing because these patterns are the expensive ones. Admitting `\\u00XX` escapes per
    character turned each name into a chain of alternations, which took the assignment group from
    31 ms to 46 ms per MiB, and the netrc pair adds 15 ms more -- on a stream that expands to
    hundreds of MiB, that is the difference between finishing inside the time budget and refusing a
    legitimate publish over the scan's own cost. A lowercased copy plus a handful of `in` tests is
    memchr-fast: 3 ms per MiB for all of them together, against roughly 50 ms of regex.

    `lowered` is the caller's single lowercased copy, shared across the patterns so the cost is
    paid once per buffer rather than once per pattern.
    """
    if lowered is None:
        return False
    if (keywords := _PATTERN_KEYWORDS.get(kind)) is None:
        return False
    # an escape anywhere means the keyword may be spelled in a way no substring test can see
    return not any(keyword in lowered for keyword in keywords) and not _ESCAPE_HINT.search(lowered)


# A private-key armor that has opened and is STILL IN ITS HEADERS at the end of the buffer. RFC
# 4880 puts no length limit on an armor header, so `Comment:` lines can push the base64 body into
# the next scan chunk, leaving the BEGIN marker and the body in no single window.
#
# Anchored on the BEGIN line and matched to the END of the buffer, so it says "this armor is
# unfinished HERE" rather than "an armor exists somewhere". A complete key -- header, blank line,
# body -- fails it, because the body characters are not header lines. That is what keeps this from
# refusing every armored key: it fires only when the window genuinely ends mid-header.
_UNFINISHED_ARMOR = re.compile(
    rb"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----[ \t]*\r?\n"
    rb"(?:[A-Za-z][A-Za-z-]*:[^\r\n]*\r?\n)*"
    rb"(?:[A-Za-z][A-Za-z-]*:[^\r\n]*)?\Z"
)


def _unfinished_private_key_armor(window: bytes) -> bool:
    """Whether `window` ends inside the headers of a private-key armor block."""
    return _UNFINISHED_ARMOR.search(window) is not None


def _match(data: bytes) -> str | None:
    """The kind of credential the literal bytes `data` contain, or None."""
    # One lowercased copy for every keyword guard below, made only when the buffer is large enough
    # for the guards to save more than the copy costs.
    lowered = data.lower() if len(data) >= _GUARD_MIN_BYTES else None
    for kind, pattern in _LITERAL_PATTERNS:
        if _keyword_absent(data, lowered, kind):
            continue
        if pattern.search(data):
            return kind
    # only the assignment-anchored group admits an all-hex body: its W&B key is issued as hex,
    # while an issuer-prefixed token is base62, so an all-hex body there is the placeholder.
    for group, hex_is_issued in ((_TOKEN_PATTERNS, False), (_ASSIGNED_PATTERNS, True)):
        for kind, pattern in group:
            if _keyword_absent(data, lowered, kind):
                continue
            for match in pattern.finditer(data):
                # the alternations put the body in whichever group matched; the rest are None.
                body = next((found for found in match.groups() if found), b"")
                if _is_high_entropy(body, hex_is_issued=hex_is_issued):
                    return kind
    return None
