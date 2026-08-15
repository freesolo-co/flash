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
# Shape-based redaction: the fail-closed net for a credential this process cannot know by VALUE.
# A runtime-minted secret (presigned url, broker capability, session cookie) is in no environment
# variable, so the value pass below contributes no needle and this pattern is the only thing between
# it and an artifact the user can fetch. Rules, each with a test in test_opd_train.py:
#   - key and value may each be quoted, and either quote may be backslash-escaped: a diagnostic
#     embedding serialized json carries `{\"password\":\"secret\"}`, which an unquoted-only
#     pattern cannot cross.
#   - a quoted value runs to its matching quote even across lines, so a PEM or serialized multiline
#     secret is one value; an unquoted one stops at whitespace so the sentence around it survives.
#   - a known auth scheme (bearer/basic/token) is consumed, not captured, or `Basic dXNlcjpwYXNz`
#     redacts the scheme and prints the credential. An UNRECOGNISED scheme runs to end of line:
#     the bare branch would otherwise treat the vendor word as the value.
#   - digest and cookie values are delimited lists, not single tokens, so both run to end of line;
#     stopping at `,` or `;` would print the `nonce` or the rest of the header verbatim.
# Over-redaction is the safe side of each rule.
_SECRET_SCHEME = r"(?:(?:bearer|basic|token)\s+)?"
_SECRET_AUTH_KEY = r"(?P<auth>authorization)"
# `(?![\w-])` not `\b`: `\b` treats a hyphen as a boundary, so `Digest-Custom` would read as the
# known `digest` and its credential would print verbatim.
_SECRET_UNKNOWN_SCHEME = r"(?!(?:bearer|basic|token|digest)(?![\w-]))[A-Za-z][\w-]*\s+"
# presigned urls carry the capability in query parameters whose names match none of the words above.
# `sig` (azure) needs a left guard or it fires on any word ending in those letters.
_SECRET_URL_PARAM = (
    r"x-(?:amz|goog)-(?:signature|credential|security-token)|signature|(?<![\w-])sig"
)
# `scheme://user:password@host`: no key name precedes the password, so the key-anchored pattern
# cannot see it. Only the password is replaced -- scheme, user and host say WHICH endpoint failed.
# The user may be empty (`redis://:password@host` is the ordinary password-only dsn shape).
_SECRET_URL_USERINFO = re.compile(r"(?i)\b([a-z][\w+.-]*://[^\s:/?#@]*:)[^\s/?#@]+(?=@)")
# `key` alone is too wide -- `cache_key`, `partition_key` are ordinary diagnostic fields -- so a
# qualifier is required and enumerated. The separator is optional so `privateKey` and azure's
# `AccountKey=` match too.
_SECRET_KEY_FIELD = (
    r"(?:private|secret|signing|encryption|session|access|account)[-_ ]?key(?:[-_ ]?id)?"
)
# unanchored, so `Set-Cookie` matches through the `Cookie` inside it.
_SECRET_COOKIE_KEY = r"(?P<cookie>cookie)"
# ODBC's PWD is a standalone field. Bound both edges so an unrelated longer key containing those
# three letters is not shape-redacted merely because this short alternative appears inside it.
_SECRET_PWD_KEY = r"(?<![\w-])pwd(?![\w-])"
# comma and semicolon delimit another field only when a key follows (`;SERVER=` / `, scope=`).
# Otherwise they are credential characters: stopping at `password=abc,runtime-secret` leaks the
# suffix. This keeps benign connection-string fields while failing closed on ambiguous punctuation.
_SECRET_NEXT_FIELD = r"[,;](?=[A-Za-z][\w.-]*(?: [A-Za-z][\w.-]*)?\s*=)"
_SECRET_DETAIL = re.compile(
    rf"(?i)(?P<key>{_SECRET_AUTH_KEY}|api[-_ ]?key|access[-_ ]?token|token|secret"
    rf"|pass(?:[-_ ]?phrase|word|wd)|{_SECRET_PWD_KEY}"
    rf"|credentials?|{_SECRET_COOKIE_KEY}|{_SECRET_KEY_FIELD}|{_SECRET_URL_PARAM})"
    r"(?P<sep>(?:\\?['\"])?\s*[:=]\s*)"
    # escaped-quote branch first: `\"secret\"` starts with a backslash, so `quote` misses it and
    # `bare` would stop dead there. `\\\\.` consumes a doubled backslash before the delimiter test.
    # `\s\S` rather than `.` keeps multiline values independent of the regex's global flags.
    rf"(?:\\(?P<eq>['\"])(?P<escaped>(?:\\\\.|(?!\\(?P=eq))[\s\S])*)"
    rf"|(?P<quote>['\"])(?:digest\s+(?P<qdigest>[^\r\n]+)"
    rf"|{_SECRET_SCHEME}(?P<quoted>(?:\\.|(?!(?P=quote))[\s\S])*))"
    r"|digest\s+(?P<digest>[^\r\n]+)"
    # `(?=...)`-style conditional, not `(?(auth)yes|no)` as the outer else: that would make the bare
    # branch unreachable for the very key that needs it.
    rf"|(?(auth){_SECRET_UNKNOWN_SCHEME}[^\r\n]+|(?!))"
    r"|(?(cookie)[^\r\n]+|(?!))"
    # `&` always terminates a signed-url parameter. Comma and semicolon terminate only before a
    # next `field=`; otherwise they are part of the issued credential and must be consumed.
    rf"|{_SECRET_SCHEME}(?P<bare>(?:(?!{_SECRET_NEXT_FIELD})[^\s&'\"}}])+))"
)
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
