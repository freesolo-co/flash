"""Teacher-bridge transport and failure reporting for the OPD verl child.

Copied into the isolated verl child workdir as `flash_opd_bridge.py`. Everything here is about
talking to the parent's localhost bridge and about what to do when that conversation fails: post a
json request, classify the error, and drop a fallback file on disk so the parent can attribute the
child's exit even when the bridge itself is what broke.

Split out of the OPD child plugin to keep that module under the file-size limit. It deliberately
does not import the plugin back: both files land flat in the child workdir, and a cycle there would
fail at import time inside verl rather than here.
"""

from __future__ import annotations

import contextlib
import functools
import http.client
import json
import os
import re
import tempfile
import types
import urllib.error
import urllib.parse
import urllib.request

# duplicated rather than imported from the plugin: this module is copied flat into the child
# workdir alongside it, and importing back would make the pair circular there. the parent reads
# these same two numbers off the child's exit status, so they must stay in step.
_PERMANENT_TEACHER_EXIT = 86
_TRANSIENT_TEACHER_EXIT = 87
_FAILURE_FALLBACK_MAX_CHARS = 8192
# the key and the value may each be quoted: a credential that reaches a diagnostic inside a json or
# dict repr -- `{"access_token": "..."}` -- puts a closing quote between the name and the separator,
# which an unquoted-only pattern cannot cross, so the value printed verbatim. a runtime-minted
# credential is in no environment variable, so the value pass cannot catch it either and this shape
# rule is the only thing standing between it and an artifact the user can fetch.
# the auth scheme is consumed, not captured: `Authorization: Basic dXNlcjpwYXNz` otherwise matches
# `Basic` as the value and prints the credential after it verbatim -- redacting the one token in the
# line that is not secret. every scheme here is a fixed word, so consuming it cannot hide a value.
# a QUOTED value runs to its matching quote, not to the first space: `{"password":"correct horse
# battery staple"}` otherwise redacts `correct` and prints the rest. the quotes are what delimit a
# serialized field, so whitespace inside them is part of the value. an UNQUOTED value still ends at
# whitespace -- there the space is the delimiter, and running past it would eat the sentence around
# the credential. no closing quote on the line means the value runs to the end of it: fail closed.
# an ESCAPED quote inside that value is consumed as one unit (`\\.` first, so the backslash pairs
# with whatever follows): a credential carrying a delimiter survives json/repr encoding as
# `{"password":"abc\"tail"}`, and treating the escaped quote as the terminator redacts `abc` and
# prints `tail` verbatim. it is runtime-minted, so the value pass cannot remove that suffix either.
# a value ending in a literal backslash over-redacts to end of line, which is the fail-closed side.
# DIGEST is the exception to all of that: its value is a comma-separated parameter list, not one
# token, so both rules above stop inside it and print the `nonce` and `response` that are the actual
# secrets. it gets its own branch running to end of line. over-redacting the tail of a digest line
# costs a `username=` and an `algorithm=`; under-redacting it publishes a live credential.
# digest is absent by design: its own branches below run first and consume the whole line, so
# listing it here too would be dead alternation.
_SECRET_SCHEME = r"(?:(?:bearer|basic|token)\s+)?"
# AUTHORIZATION carries `scheme credential`, and the scheme space is open-ended -- Negotiate, NTLM,
# AWS4-HMAC-SHA256, any vendor word. An unrecognised scheme is the dangerous case: the bare branch
# matches the SCHEME as the value and prints the credential after it, redacting the one word on the
# line that is not secret. So an authorization value whose scheme is NOT a known single-token one
# runs to end of line, like digest. The known schemes keep the whitespace stop, because their value
# is a single token and consuming the rest of the line would eat the diagnostic AROUND an already
# fully redacted credential (`Authorization: Bearer tok while calling the teacher`).
_SECRET_AUTH_KEY = r"(?P<auth>authorization)"
# a scheme word followed by a credential; the negative lookahead is what routes the known ones back
# to the bare branch. the guard is `(?![\w-])`, not `\b`: `\b` treats a hyphen as a boundary, so
# `Digest-Custom` would read as the known `digest` and its credential would print verbatim.
# `digest` is listed because its own branches run first and already consume that line.
_SECRET_UNKNOWN_SCHEME = r"(?!(?:bearer|basic|token|digest)(?![\w-]))[A-Za-z][\w-]*\s+"
# a presigned url carries its capability in QUERY PARAMETERS whose names look nothing like the
# credential words above: `?X-Amz-Credential=...&X-Amz-Signature=...` is a complete, immediately
# usable capability and matched none of them. it is minted per-request, so it is in no environment
# variable and the value pass cannot see it either -- exactly the case this shape rule exists for.
# `sig` (azure) needs a left guard: unanchored it would fire on any word ending in those letters.
_SECRET_URL_PARAM = (
    r"x-(?:amz|goog)-(?:signature|credential|security-token)|signature|(?<![\w-])sig"
)
# a URL carries its password in USERINFO -- `scheme://user:password@host` -- where no key name
# precedes it, so the key-anchored pattern above cannot see it at all. A connection string built at
# runtime (a broker dsn, a database url) is in no environment variable, so the value pass misses it
# too. Only the password is replaced: the scheme, user and host say WHICH endpoint failed and are
# the diagnostic. The password stops at `@`, and `[^\s/?#@]` keeps a later `@` in a path or query
# from extending the match past the authority. The user may be EMPTY: `redis://:password@host` is
# the ordinary shape for a password-only dsn, and requiring a user there would leak exactly the
# runtime-built credential no environment variable holds. It is the PASSWORD being non-empty that
# keeps a bare `scheme://:@host` out, and the scheme that keeps a `//` in prose out.
_SECRET_URL_USERINFO = re.compile(r"(?i)\b([a-z][\w+.-]*://[^\s:/?#@]*:)[^\s/?#@]+(?=@)")
# a KEY-suffixed field is a credential only when a qualifier says so. `key` alone cannot join the
# list above: `cache_key`, `partition_key` and `idempotency_key` are ordinary diagnostic fields, and
# redacting them eats the message this record exists to carry. So the qualifier is required and
# enumerated -- `private_key` and `secret_key` name a credential, `primary_key` does not. The
# separator is optional so `privateKey` (camelCase, as json from a js caller arrives) matches too,
# which is also what makes azure's separator-less `AccountKey=` match the `account` qualifier.
_SECRET_KEY_FIELD = (
    r"(?:private|secret|signing|encryption|session|access|account)[-_ ]?key(?:[-_ ]?id)?"
)
# a COOKIE header is credential-bearing as a WHOLE value, not at one inner name: `Cookie: a=1;
# sessionid=X; b=2` carries the session in a semicolon-delimited list, and matching only the inner
# name leaves the rest -- including whatever else the header carries -- verbatim. So the header is
# consumed to end of line like an unknown auth scheme, rather than stopping at `;` the way a bare
# value does. A runtime-issued cookie is in no environment variable, so the value pass cannot reach
# it either. `Set-Cookie` needs no branch of its own: the key is unanchored, so it matches the
# `Cookie` inside it and the value is consumed identically -- only the literal `Set-` stays
# visible, which is a header name rather than a secret.
_SECRET_COOKIE_KEY = r"(?P<cookie>cookie)"
_SECRET_DETAIL = re.compile(
    # `pass(-|_| )?phrase` sits beside `password`/`passwd` rather than being covered by them: no
    # prefix of one is a prefix of the other, so the existing words never matched it. It is the
    # standard field name for the phrase unlocking a generated private key, and that phrase is
    # minted at runtime, so it is in no environment variable and the value pass cannot reach it.
    # The value stop is deliberately the same as `password`'s: a quoted value redacts whole, an
    # unquoted one ends at whitespace. Consuming to end of line instead would eat the diagnostic
    # around it, and a passphrase is a single field rather than a delimited list like a cookie.
    rf"(?i)(?P<key>{_SECRET_AUTH_KEY}|api[-_ ]?key|access[-_ ]?token|token|secret"
    rf"|pass(?:[-_ ]?phrase|word|wd)"
    # `credential`/`credentials` is the generic label a library reaches for when the value has no
    # more specific name, and it names the value outright rather than qualifying something else, so
    # unlike `key` it needs no qualifier. plural included: `{"credentials": ...}` is the commoner
    # json spelling. a runtime-minted value is in no environment variable, so the value pass cannot
    # reach it and this shape rule is the only thing standing between it and the record.
    rf"|credentials?|{_SECRET_COOKIE_KEY}|{_SECRET_KEY_FIELD}|{_SECRET_URL_PARAM})"
    # the quote around the key may itself be BACKSLASH-ESCAPED: a child exception that embeds
    # serialized json carries `{\"password\":\"secret\"}`, and a bare `['\"]?` stops at the
    # backslash, so the whole credential printed verbatim. a runtime-minted value is in no
    # environment variable, so the value pass cannot catch it either. distinct from an escaped
    # quote INSIDE the value, which `quoted` already handles.
    r"(?P<sep>(?:\\?['\"])?\s*[:=]\s*)"
    # the VALUE's opening quote may be escaped too, and it is tried FIRST: `\"secret\"` starts with
    # a backslash, so `quote` does not match it and `bare` stops dead at that backslash, printing
    # the credential. it ends at the matching ESCAPED quote, which keeps the tail after the field
    # (`{\"password\":\"s\"} while calling`) instead of consuming the rest of the diagnostic.
    # a doubled backslash is consumed as ONE unit BEFORE the delimiter test, so an escaped quote
    # INSIDE the encoded value (`\"abc\\\"tail\"`) is part of the value rather than its closer --
    # otherwise the match ends early and the tail after it prints verbatim.
    rf"(?:\\(?P<eq>['\"])(?P<escaped>(?:\\\\.|(?!\\(?P=eq))[^\r\n])*)"
    rf"|(?P<quote>['\"])(?:digest\s+(?P<qdigest>[^\r\n]+)"
    rf"|{_SECRET_SCHEME}(?P<quoted>(?:\\.|(?!(?P=quote))[^\r\n])*))"
    r"|digest\s+(?P<digest>[^\r\n]+)"
    # an AUTHORIZATION value with an UNRECOGNISED scheme runs to end of line: the bare branch below
    # would otherwise match that scheme and print the credential after it. This alternative is tried
    # first and its lookahead fails for the known single-token schemes, so those fall through to the
    # bare branch and keep the whitespace stop. It is guarded by `(?=...)` on the key rather than a
    # conditional, because `(?(auth)yes|no)` makes the bare branch the ELSE -- unreachable for the
    # very key that needs it, so `Bearer`/`Basic` would stop being consumed at all.
    rf"|(?(auth){_SECRET_UNKNOWN_SCHEME}[^\r\n]+|(?!))"
    # an UNQUOTED cookie header runs to end of line for the same reason, but a different stop: the
    # bare branch ends at `;`, which is the very delimiter a cookie list uses, so it would redact
    # `sessionid=X` and print the rest of the header verbatim. Same conditional shape as `auth`,
    # and it sits AFTER the quoted branches so a json-embedded cookie still ends at its closing
    # quote and keeps the object structure around it.
    r"|(?(cookie)[^\r\n]+|(?!))"
    # `&` terminates an unquoted value: without it the first signed-url parameter swallows every
    # later one, so a single <redacted> replaces the whole query string including the benign
    # `X-Amz-Expires` that says WHY a capability failed. each parameter is redacted on its own.
    rf"|{_SECRET_SCHEME}(?P<bare>[^\s,;&'\"}}]+))"
)
# component lines of a multiline credential shorter than this are punctuation such as ``}``, not
# secrets; redacting them would erase innocent text. Matches `bootstrap_secrets._MIN_SECRET_COMPONENT`.
_MIN_SECRET_COMPONENT = 8
# vocabulary ids run to ~7 digits at today's sizes; longer digit runs are not ids. see _is_token_id.
_MAX_TOKEN_ID_DIGITS = 7


def _plugin():
    """The child plugin module, imported lazily because it re-exports this one.

    Four names here are patched as attributes of `plugin` by the opd tests -- `_post_json`,
    `_mutation_distributed`, `_publish_mutation_notice`, `_coordinate_first_mutation_notice` --
    and every caller of theirs lives in this file. Resolving them through the plugin is what
    keeps those patches effective; a direct call would bind this module's own function. The flat
    name is the one that exists in the verl child workdir, where `flash` is not importable.
    """
    try:
        import flash_opd_plugin as plugin
    except ImportError:
        from flash.engine.worker.train.opd.child import plugin

    return plugin


class FlashTeacherBridgeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        classification: str,
        delivery_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.classification = classification
        self.delivery_unknown = delivery_unknown


def _post_json(url: str, token: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
            details = payload["error"]
            classification = str(details["classification"])
            message = str(details["message"])
        except (KeyError, TypeError, ValueError) as decode_error:
            raise FlashTeacherBridgeError(
                f"flash OPD bridge returned unclassified HTTP {error.code}",
                classification="permanent",
            ) from decode_error
        if classification not in {"permanent", "transient"}:
            raise FlashTeacherBridgeError(
                f"flash OPD bridge returned unknown teacher failure classification {classification!r}",
                classification="permanent",
            ) from error
        raise FlashTeacherBridgeError(message, classification=classification) from error
    except (OSError, http.client.HTTPException) as error:
        raise FlashTeacherBridgeError(
            f"flash OPD bridge transport failed: {type(error).__name__}",
            classification="transient",
            delivery_unknown=True,
        ) from error
    try:
        return json.loads(body.decode("utf-8"))
    except (TypeError, ValueError) as error:
        raise FlashTeacherBridgeError(
            "flash OPD bridge returned malformed success JSON",
            classification="permanent",
            delivery_unknown=True,
        ) from error


def _serialize_failure_fallback(classification: str, message: str) -> bytes:
    message = str(message).encode("utf-8", errors="replace").decode("utf-8")
    lower = 0
    upper = min(len(message), _FAILURE_FALLBACK_MAX_CHARS)
    serialized = json.dumps(
        {"classification": classification, "message": ""},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    while lower <= upper:
        length = (lower + upper) // 2
        candidate = json.dumps(
            {"classification": classification, "message": message[:length]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(candidate) <= _FAILURE_FALLBACK_MAX_CHARS:
            serialized = candidate
            lower = length + 1
        else:
            upper = length - 1
    return serialized


def _write_failure_fallback(
    base_path: str,
    classification: str,
    message: str,
) -> None:
    if not base_path:
        return
    process_id = os.getpid()
    record_path = f"{base_path}.{process_id}.{classification}.json"
    directory = os.path.dirname(base_path) or "."
    prefix = f".{os.path.basename(base_path)}.{process_id}.{classification}."
    payload = _serialize_failure_fallback(classification, message)
    temporary_path = ""
    try:
        descriptor, temporary_path = tempfile.mkstemp(
            dir=directory,
            prefix=prefix,
            suffix=".tmp",
        )
        try:
            remaining = payload
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("mutation failure fallback write did not progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary_path, record_path)
        temporary_path = ""
    except OSError:
        if temporary_path:
            with contextlib.suppress(OSError):
                os.unlink(temporary_path)


def _write_mutation_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_MUTATION_FAILURE_PATH", ""),
        classification,
        message,
    )


def _write_score_delivery_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH", ""),
        classification,
        message,
    )


def _write_abandonment_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_ABANDONMENT_FAILURE_PATH", ""),
        classification,
        message,
    )


def _write_resample_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_RESAMPLE_FAILURE_PATH", ""),
        classification,
        message,
    )


def _write_cycle_commit_failure_fallback(classification: str, message: str) -> None:
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH", ""),
        classification,
        message,
    )


def _secret_env_name(name: str) -> bool:
    """Whether ``name`` names a credential, by the same rule as `bootstrap_secrets`.

    Suffix and exact matches only, never a substring: the child's environment carries
    `TOKENIZERS_PARALLELISM` and `FLASH_OPD_EOS_TOKEN_IDS`, whose values are `false` and a list of
    token ids. A substring rule treats both as credentials and rewrites every occurrence of those
    values, so an ordinary word or id list in the failure message becomes `<redacted>` -- corrupting
    the diagnostic this record exists to carry.
    """
    upper = str(name).upper()
    return upper in {"AUTHORIZATION", "HF_TOKEN"} or upper.endswith(
        ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
    )


def _secret_needles(secret: str) -> set[str]:
    """Every textual form of ``secret`` worth searching for, mirroring `bootstrap_secrets`.

    A multiline credential (a PEM key, a pasted service-account blob) reaches a diagnostic one
    component line at a time, so the whole value never matches and only its registered components
    do. The 8-character floor keeps a short component such as ``}`` from erasing innocent text.

    Only the CANONICAL percent-encoding is registered. Triplet hex casing is case-insensitive and
    every casing is valid in the wild, but they cannot be enumerated: a value with n escaped
    characters has 2**n spellings, so registering a second all-lowercase form still misses every
    MIXED one (``abc%2Fdef%3aghi``). ``_needle_pattern`` makes each triplet case-insensitive at
    match time instead, which covers all of them from this one form.
    """
    forms = {secret}
    if "\n" in secret:
        forms.update(
            line for raw in secret.splitlines() if len(line := raw.strip()) >= _MIN_SECRET_COMPONENT
        )
    needles: set[str] = set()
    for form in forms:
        needles.add(form)
        # os.environ decodes non-UTF-8 bytes with surrogateescape, and quote() cannot encode a
        # surrogate. letting that raise would abort the sanitizer -- documented as never raising --
        # from inside the handler writing the failure record, replacing the real exception with a
        # UnicodeEncodeError and leaving no record at all: the opaque death this PR removes. the
        # raw form is already registered above, so only the percent-encoded spelling is lost.
        try:
            encoded = urllib.parse.quote(form, safe="")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        needles.add(encoded)
    return needles


def _needle_pattern(needle: str) -> str:
    """``needle`` as a regex matching it literally, except that percent triplets ignore hex case.

    RFC 3986 defines the triplet's hex digits as case-insensitive, so ``%2F`` and ``%2f`` are the
    same character and both are emitted in the wild. Enumerating the spellings is not an option --
    n escaped characters give 2**n of them -- so the case fold is applied here, per triplet, and the
    rest of the needle stays case-SENSITIVE: a credential is case-sensitive, and folding it whole
    would let an unrelated word that differs only in case erase itself from the diagnostic.
    """
    return re.sub(
        r"%([0-9A-Fa-f])([0-9A-Fa-f])",
        lambda m: f"%[{m[1].lower()}{m[1].upper()}][{m[2].lower()}{m[2].upper()}]",
        re.escape(needle),
    )


def _is_token_id(match: re.Match[str]) -> bool:
    """True for `token: 151643` -- a vocabulary id, not a credential.

    The exemption exists because a bad eos or token-boundary id is often the entire diagnostic this
    record carries, and shape redaction would eat it. But "any digit-only value" is far too wide a
    hole to leave in the fail-closed net: `{"access_token": "123456789012"}` is a numeric credential,
    and a runtime-minted one contributes no needle to the value pass either, so this predicate would
    be the only thing between it and an artifact the user can fetch.

    So the exemption is narrowed to the shape a vocabulary id actually has: the bare word `token`
    (never `access_token`, `api_key`, `password`, or `authorization`, none of which label an id),
    unquoted, and short enough to be one. A quoted value is a serialized field, not a number
    rendered into a sentence.
    """
    key, separator = match.group("key"), match.group("sep")
    value = match.group("bare")
    return (
        value is not None  # None means the quoted branch matched: a serialized field, not an id
        and value.isdigit()
        and len(value) <= _MAX_TOKEN_ID_DIGITS
        and "'" not in separator
        and '"' not in separator
        and key.lower() == "token"
    )


def _redact_secret_match(match: re.Match) -> str:
    """The matched credential replaced by ``<redacted>``, keeping what framed it.

    The opening quote is reprinted, never the closing one: it was not consumed, so it is still in
    the text after the match and reprinting it would double. An ESCAPED opening quote carries its
    backslash back with it, so the surrounding serialized json stays readable.
    """
    if _is_token_id(match):
        return match.group(0)
    escaped_quote = match.group("eq")
    opener = "\\" + escaped_quote if escaped_quote else match.group("quote") or ""
    return f"{match.group('key')}{match.group('sep')}{opener}<redacted>"


def _safe_child_failure_detail(error: Exception) -> str:
    """``error`` rendered with credentials removed, never raising.

    ``str(error)`` runs user-supplied ``__str__`` and ``__repr__`` code, which can raise. This
    function is only ever called while recording why the child is about to die, so letting that
    escape would replace the real exception's type, message and stage with the stringification
    error's -- destroying the diagnostic instead of writing it. The class name is always available
    without executing user code, so an unrenderable message degrades to that rather than to nothing.
    """
    try:
        message = str(error)
    except BaseException:
        # BaseException, not Exception: a __str__ raising KeyboardInterrupt or SystemExit escapes
        # the narrower clause and aborts this function, which is documented as never raising. the
        # caller then never reaches its os._exit, restoring the opaque death this path removes.
        message = "<unrenderable message>"
    secrets = {value for name, value in os.environ.items() if value and _secret_env_name(name)}
    for secret in sorted(secrets, key=len, reverse=True):
        for needle in sorted(_secret_needles(secret), key=len, reverse=True):
            pattern = _needle_pattern(needle)
            if len(needle) >= _MIN_SECRET_COMPONENT:
                message = re.sub(pattern, "<redacted>", message)
                continue
            left = r"(?<!\w)" if needle[:1].isalnum() or needle[:1] == "_" else ""
            right = r"(?!\w)" if needle[-1:].isalnum() or needle[-1:] == "_" else ""
            message = re.sub(f"{left}{pattern}{right}", "<redacted>", message)
    message = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", message)
    message = _SECRET_URL_USERINFO.sub(r"\1<redacted>", message)
    # shape redaction stays as the fail-closed net for a credential this process cannot know by
    # value -- one minted at runtime (a presigned url, a broker capability) is in neither the
    # environment nor any payload, so it contributes no needle above.
    return _SECRET_DETAIL.sub(_redact_secret_match, message)


def _write_child_failure_fallback(
    classification: str,
    stage: str,
    error: Exception,
) -> None:
    detail = _safe_child_failure_detail(error)
    _write_failure_fallback(
        os.environ.get("FLASH_OPD_CHILD_FAILURE_PATH", ""),
        classification,
        f"[stage={stage}] {type(error).__name__}: {detail}",
    )


def _fallback_classification(error: FlashTeacherBridgeError) -> str:
    return "transient" if error.delivery_unknown else error.classification


def _exit_for_score_failure(error: FlashTeacherBridgeError) -> None:
    classification = _fallback_classification(error)
    if error.delivery_unknown:
        _write_score_delivery_failure_fallback(classification, str(error))
    exit_code = (
        _PERMANENT_TEACHER_EXIT if classification == "permanent" else _TRANSIENT_TEACHER_EXIT
    )
    os._exit(exit_code)


def _unexpected_mutation_bridge_error(error: Exception) -> FlashTeacherBridgeError:
    return FlashTeacherBridgeError(
        f"unexpected mutation bridge failure: {type(error).__name__}",
        classification="permanent",
    )


def _publish_mutation_notice(url: str, token: str) -> None:
    payload = {"process_id": os.getpid()}
    try:
        _plugin()._post_json(url, token, "/mutation", payload)
        return
    except FlashTeacherBridgeError as error:
        if not error.delivery_unknown:
            _write_mutation_failure_fallback(error.classification, str(error))
            raise
    except Exception as error:
        bridge_error = _unexpected_mutation_bridge_error(error)
        _write_mutation_failure_fallback(
            bridge_error.classification,
            str(bridge_error),
        )
        raise bridge_error from error

    try:
        _plugin()._post_json(url, token, "/mutation", payload)
    except FlashTeacherBridgeError as error:
        _write_mutation_failure_fallback(error.classification, str(error))
        raise
    except Exception as error:
        bridge_error = _unexpected_mutation_bridge_error(error)
        _write_mutation_failure_fallback(
            bridge_error.classification,
            str(bridge_error),
        )
        raise bridge_error from error


def _post_no_signal_resample(url: str, token: str) -> None:
    try:
        _plugin()._post_json(url, token, "/no-signal/resample", {})
    except FlashTeacherBridgeError as error:
        _write_resample_failure_fallback(_fallback_classification(error), str(error))
        raise
    except Exception as error:
        _write_resample_failure_fallback(
            "permanent",
            f"unexpected resample bridge failure: {type(error).__name__}",
        )
        raise


def _post_no_signal_abandoned(url: str, token: str) -> dict:
    """Tell the bridge this step is abandoned; return its per-reason skip tally for the error text."""
    try:
        response = _plugin()._post_json(url, token, "/no-signal/abandoned", {})
        return dict((response or {}).get("skip_counts") or {})
    except FlashTeacherBridgeError as error:
        _write_abandonment_failure_fallback(
            _fallback_classification(error),
            str(error),
        )
        raise
    except Exception as error:
        _write_abandonment_failure_fallback(
            "permanent",
            f"unexpected abandonment bridge failure: {type(error).__name__}",
        )
        raise


def _post_teacher_cycle_committed(url: str, token: str) -> None:
    try:
        _plugin()._post_json(url, token, "/teacher-cycle/committed", {})
        return
    except FlashTeacherBridgeError as error:
        if not error.delivery_unknown:
            _write_cycle_commit_failure_fallback(error.classification, str(error))
            raise
    except Exception as error:
        _write_cycle_commit_failure_fallback(
            "permanent",
            f"unexpected cycle commitment bridge failure: {type(error).__name__}",
        )
        raise

    try:
        _plugin()._post_json(url, token, "/teacher-cycle/committed", {})
    except FlashTeacherBridgeError as error:
        _write_cycle_commit_failure_fallback(
            _fallback_classification(error),
            str(error),
        )
        raise
    except Exception as error:
        _write_cycle_commit_failure_fallback(
            "permanent",
            f"unexpected cycle commitment bridge failure: {type(error).__name__}",
        )
        raise


def _update_actor_after_teacher_cycle_commit(
    actor_rollout_wg,
    batch,
    url: str,
    token: str,
):
    _post_teacher_cycle_committed(url, token)
    return actor_rollout_wg.update_actor(batch)


def _mutation_distributed():
    import torch

    return torch.distributed


def _coordinate_first_mutation_notice(url: str, token: str) -> None:
    distributed = _plugin()._mutation_distributed()
    distributed.barrier()
    outcome: list[tuple[str, str] | None] = [None]
    if distributed.get_rank() == 0:
        try:
            _plugin()._publish_mutation_notice(url, token)
        except FlashTeacherBridgeError as error:
            outcome[0] = (error.classification, str(error))
    distributed.broadcast_object_list(outcome, src=0)
    distributed.barrier()
    if outcome[0] is not None:
        classification, message = outcome[0]
        raise FlashTeacherBridgeError(message, classification=classification)


def _wrap_optimizer_with_mutation_notice(optimizer, url: str, token: str):
    original_step = optimizer.step
    mutation_acknowledged = False

    # torch's lr scheduler patches optimizer.step to assert step ordering, and that patch reads
    # step_fn.__func__ and rebinds it with func.__get__(opt, opt.__class__). both only exist on a
    # bound method, so assigning a plain function here fails with "'function' object has no
    # attribute '__func__'" as soon as verl builds its warmup scheduler, before any gpu work
    # happens. bind the wrapper to the instance so the descriptor protocol keeps working.
    @functools.wraps(original_step)
    def step_with_notice(_optimizer, *args, **kwargs):
        nonlocal mutation_acknowledged
        if not mutation_acknowledged:
            _plugin()._coordinate_first_mutation_notice(url, token)
            mutation_acknowledged = True
        return original_step(*args, **kwargs)

    optimizer.step = types.MethodType(step_with_notice, optimizer)
    return optimizer
