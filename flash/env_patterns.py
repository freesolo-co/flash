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
_BLOCK_SCALAR = rb"(?:[|>](?:[+-][0-9]?|[0-9][+-]?)?\s*)?"
# The opening quote, if any. A single optional quote character consumed only ONE of the three in a
# triple-quote delimiter, leaving a quote sitting where the body had to begin, so an ordinary
# Python or TOML multiline assignment matched nothing and published. Whole delimiters are named,
# longest first, so all three forms are consumed together.
_OPEN_QUOTE = rb"(?:\"\"\"|'''|[\"'])?"
_ASSIGNED_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "a Weights & Biases API key",
        re.compile(
            rb"(?i:wandb_api_key)[\"']?\s*[:=]\s*"
            + _BLOCK_SCALAR
            + _OPEN_QUOTE
            + rb"([A-Za-z0-9_-]{40,%d})" % _MAX_BODY
        ),
    ),
    (
        "an AWS secret access key",
        re.compile(
            rb"(?i:aws_secret_access_key)[\"']?\s*[:=]\s*"
            + _BLOCK_SCALAR
            + _OPEN_QUOTE
            + rb"([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"
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
    rb"\"(?:k|(?i:\\[uU]006b))(?:t|(?i:\\[uU]0074))(?:y|(?i:\\[uU]0079))\""
    rb"\s*:\s*\"(?:RSA|EC|OKP|oct)\""
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
_JWK_PRIVATE = re.compile(
    rb"\"(?:"
    + b"|".join(_JWK_ESCAPED[name] for name in (b"dp", b"dq", b"qi", b"d", b"k"))
    # longest first: `d` would otherwise win against `dp` and leave the `p` outside the quote
    + rb")\"\s*:\s*\"[A-Za-z0-9+/\-_]{20,}={0,2}\""
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

    def search(self, data: bytes) -> re.Match[bytes] | None:
        """The payload's match when the context marker accompanies it anywhere in `data`."""
        if not (payload := self.payload.search(data)):
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
            re.compile(rb"Private-Lines:\s*\d+[\r\n]+[A-Za-z0-9+/=]{32,}"),
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
