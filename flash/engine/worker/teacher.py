"""Fireworks-hosted teacher client for on-policy distillation.

Reaches the GLM-5.2 teacher over the OpenAI-compatible API using only the Python standard library (no
``openai``/``requests`` dependency — the worker runs on stdlib here). One operation:

- ``score``: echo-score a student completion (``echo=true, logprobs=1, max_tokens=0``) to get the
  teacher's per-token REALIZED logprobs + CHARACTER offsets over the completion region only. This is
  the on-policy signal for the groupwise reverse-KL (gkd) loss — it pays for input tokens only.

All heavy work is inside methods; importing this module is CPU/offline-safe.
"""

from __future__ import annotations

import http.client
import io
import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from flash.engine.worker.tokenizer_align import TeacherToken

_MAX_LOGPROB_ROUNDING_ERROR = 1e-6


class TeacherError(RuntimeError):
    """Teacher API call failed. ``permanent`` marks non-retryable causes (bad key / model id /
    malformed request/response) so the worker aborts the run immediately instead of skipping every
    remaining sample and burning the whole paid GPU allocation before failing."""

    def __init__(self, *args, permanent: bool = False) -> None:
        super().__init__(*args)
        self.permanent = permanent


def _remaining_run_wall_seconds() -> float | None:
    raw_deadline = os.environ.get("FLASH_RUN_DEADLINE_AT")
    if raw_deadline is None:
        return None
    try:
        deadline = float(raw_deadline)
    except (TypeError, ValueError):
        raise TeacherError("worker run wall deadline is invalid", permanent=True) from None
    now = time.time()
    if not math.isfinite(deadline) or deadline <= 0 or not math.isfinite(now) or now <= 0:
        raise TeacherError("worker run wall deadline is invalid", permanent=True)
    return max(0.0, deadline - now)


def _validate_echo(tokens, token_logprobs, offsets, full) -> list[tuple[int, int]]:
    """Validate a teacher echo response's contract before the completion tokens are emitted.
    Raises a PERMANENT ``TeacherError`` on any malformed shape — non-list fields, a length
    disagreement, a non-numeric/non-finite/fractional/out-of-range/out-of-order text_offset, an
    echo that does not start at offset 0, a non-numeric/non-finite/positive token_logprob, or
    tokens that do not tile ``full`` char-for-char. Returns the per-token ``(start, end)`` character
    spans (computed ONCE from the validated offsets) for ``score``'s emit loop to reuse, so the span
    arithmetic is not recomputed there. A pure extraction of ``score``'s validation gauntlet (the
    SAME checks in the SAME order)."""

    def _bad(m):
        raise TeacherError(m, permanent=True)

    # The length checks and index loop below assume these are sequences. A malformed 200 with
    # token_logprobs=null or a scalar text_offset would make len()/indexing raise TypeError HERE,
    # outside teacher classification, so a consistently malformed teacher could burn every OPD step
    # before the no-signal failure. Reject non-list fields up front as a PERMANENT contract break.
    if not all(isinstance(v, list) for v in (tokens, token_logprobs, offsets)):
        _bad(
            "teacher echo response logprobs fields are not all lists "
            f"(tokens={type(tokens).__name__}, token_logprobs={type(token_logprobs).__name__}, "
            f"text_offset={type(offsets).__name__})"
        )
    n = len(tokens)
    # A well-formed echo response returns tokens / token_logprobs / text_offset of EQUAL length,
    # so require EXACT equality (not merely "not shorter than tokens"). A SHORTER logprobs/offsets
    # array makes the loop below IndexError outside teacher classification. A LONGER one is just as
    # broken: `n = len(tokens)` would then silently ignore
    # the offsets/logprobs tail, and the last token (i == n-1) takes `end = len(full)` (the i+1<n
    # fallback), reinterpreting a token that should end mid-string as spanning through the whole
    # completion and training on the wrong logprob. Both are a broken teacher contract that won't
    # fix itself on retry, so reject any length disagreement as PERMANENT and abort now (codex[bot]).
    if len(token_logprobs) != n or len(offsets) != n:
        _bad(
            f"teacher echo response arrays disagree in length: tokens={n}, "
            f"token_logprobs={len(token_logprobs)}, text_offset={len(offsets)}"
        )
    # The loop coerces each offset with int(offsets[i]). A malformed 200 can still put a
    # non-numeric value in text_offset (e.g. [0, null] or [0, "bad"]) that passes the list/length
    # guards above -- int() then raises TypeError/ValueError OUTSIDE any TeacherError, so a consistently
    # malformed teacher could burn every OPD step
    # (codex[bot]). Validate the offsets are numeric (bools excluded) and non-decreasing up front
    # and reject as a PERMANENT contract break. token_logprobs[i] may still be null (handled below).
    prev_off = None
    full_len = len(full)
    for o in offsets[:n]:
        if isinstance(o, bool) or not isinstance(o, (int, float)):
            _bad(f"teacher echo response text_offset has a non-numeric value: {o!r}")
        # int(offsets[i]) below coerces each offset to a character index into `full`. A value that
        # is merely numeric still corrupts the alignment three ways that the check above misses:
        # NaN/inf makes int() RAISE outside TeacherError; a fractional float silently TRUNCATES to a wrong index; an offset
        # outside [0, len(full)] yields a token span that starts before the completion or overshoots
        # it. Require a FINITE INTEGER within range up front and reject as PERMANENT (codex[bot]).
        # isfinite() must precede int() -- int(NaN) raises -- so the order here is load-bearing.
        if not math.isfinite(o):
            _bad(f"teacher echo response text_offset is not finite: {o!r}")
        if o != int(o):
            _bad(f"teacher echo response text_offset is not an integer: {o!r}")
        if o < 0 or o > full_len:
            _bad(f"teacher echo response text_offset {o!r} is outside [0, {full_len}]")
        if prev_off is not None and o < prev_off:
            _bad(
                f"teacher echo response text_offset is not non-decreasing: {prev_off!r} -> {o!r}"
            )
        prev_off = o
    # Tiling from offsets[0] only proves coverage of full[offsets[0]:], so the echo must START at 0.
    # A malformed 200 that DROPS a prompt prefix and echoes a clean-tiling SUFFIX (offsets[0] > 0)
    # passes every offset/tiling check, but its completion logprobs were computed by the teacher over a
    # TRUNCATED prompt -- the gkd signal would be scored against context the student never saw. Require
    # the first offset to be 0 for a non-empty echo and reject a dropped prefix as PERMANENT (codex[bot]).
    if n and int(offsets[0]) != 0:
        _bad(
            f"teacher echo does not start at offset 0 (first text_offset={int(offsets[0])}); it "
            "dropped a prompt prefix, so every completion logprob is conditioned on a truncated prompt."
        )
    # token_logprobs[i] is coerced with float(...) below (None -> 0.0 for a null realized logprob,
    # e.g. the first token). A malformed 200 can still put a non-numeric value (e.g. a "NaN" string)
    # or a non-finite float (NaN/inf) here that passes the list/length guards: a non-numeric value
    # makes float() RAISE ValueError OUTSIDE any TeacherError, and a NaN/inf feeds straight into the gkd teacher_logsum and poisons the
    # loss with a non-finite gradient. Validate up front and reject as PERMANENT (codex[bot]). None
    # is allowed HERE (prompt-context tokens legitimately carry a null realized logprob); a null on
    # a token we actually KEEP (a completion token) is rejected in the emit loop below.
    for lp in token_logprobs[:n]:
        if lp is None:
            continue
        if isinstance(lp, bool) or not isinstance(lp, (int, float)):
            _bad(f"teacher echo response token_logprobs has a non-numeric value: {lp!r}")
        if not math.isfinite(lp):
            _bad(f"teacher echo response token_logprobs has a non-finite value: {lp!r}")
        if lp > _MAX_LOGPROB_ROUNDING_ERROR:
            # A log-probability cannot exceed 0. A malformed 200 with a POSITIVE value is a
            # probability > 1: summed into teacher_logsum it poisons the reverse-KL coefficient
            # (logP_student.detach() - logP_teacher) with impossible teacher mass, so OPD would
            # train on a bogus signal instead of aborting like the other contract violations. Reject
            # as PERMANENT. The 1e-6 tolerance absorbs float rounding of a ~0 logprob on a
            # near-deterministic token (codex[bot]).
            _bad(
                f"teacher echo response token_logprobs has a positive value {lp!r} "
                "(a log-probability cannot exceed 0)."
            )
    # The echoed tokens must TILE `full` contiguously: each token's TEXT must equal the substring it
    # claims to span, full[offsets[i] : offsets[i+1]] (the last token to len(full)). The emit loop
    # below takes token i's end from offsets[i+1] (or len(full) for the last), so comparing the token
    # text to that exact substring catches EVERY way the echo can lie about a span and then have its
    # logprob trained on the wrong text: an interior gap/overlap or a truncated last token (length
    # changes), AND a same-length-but-different token (a malformed 200 echoing wrong tokens over the
    # right offsets — a length-only check would miss it). Verified against the live GLM echo that
    # composed + split multi-byte chars and JSON/regex-escaped mongo filters all echo as EXACT
    # literal substrings tiling char-for-char (a split char appears as zero-width tokens at the
    # shared offset), so this does NOT false-positive on real completions; a mismatch is a broken
    # teacher contract, rejected as PERMANENT (codex[bot]).
    # The per-token (start, end) character spans, computed ONCE from the now-validated offsets. Both
    # the tiling check below and score()'s emit loop consume this list, so the int(offsets[i]) and the
    # last-token len(full) fallback arithmetic is not recomputed (it previously ran HERE and AGAIN in
    # the emit loop).
    spans = [(int(offsets[i]), int(offsets[i + 1]) if i + 1 < n else full_len) for i in range(n)]
    for i, (start, boundary) in enumerate(spans):
        expected = full[start:boundary]
        if str(tokens[i]) != expected:
            _bad(
                f"teacher echo does not tile the input at token {i}: token text "
                f"{str(tokens[i])!r} != echoed substring full[{start}:{boundary}]={expected!r} "
                "(gap/overlap/truncation or a non-literal echo); its span would be fabricated."
            )
    return spans


class _ReusableHttpsResponse:
    def __init__(self, transport, connection, response) -> None:
        self._transport = transport
        self._connection = connection
        self._response = response
        self._read_complete = False

    def read(self) -> bytes:
        try:
            body = self._response.read()
        except http.client.IncompleteRead:
            self._transport.discard(self._connection)
            raise
        self._read_complete = True
        if self._response.will_close or self._connection.sock is None:
            self._transport.discard(self._connection)
        return body

    def close(self) -> None:
        self._response.close()
        if not self._read_complete or self._response.will_close or self._connection.sock is None:
            self._transport.discard(self._connection)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        self.close()
        return False


class _ThreadLocalHttpsTransport:
    _stale_errors = (
        http.client.RemoteDisconnected,
        http.client.CannotSendRequest,
        http.client.ResponseNotReady,
        BrokenPipeError,
        ConnectionResetError,
    )

    def __init__(self) -> None:
        self._local = threading.local()

    def discard(self, connection) -> None:
        if getattr(self._local, "connection", None) is connection:
            self._local.connection = None
            self._local.origin = None
        connection.close()

    def _connection(self, parsed, timeout: float):
        origin = (parsed.hostname, parsed.port or 443)
        connection = getattr(self._local, "connection", None)
        if connection is not None and getattr(self._local, "origin", None) != origin:
            self.discard(connection)
            connection = None
        if connection is None:
            connection = http.client.HTTPSConnection(origin[0], origin[1], timeout=timeout)
            self._local.connection = connection
            self._local.origin = origin
        else:
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
        return connection

    def urlopen(self, req: urllib.request.Request, *, timeout: float):
        parsed = urllib.parse.urlsplit(req.full_url)
        if parsed.scheme != "https":
            return urllib.request.urlopen(req, timeout=timeout)
        connection = self._connection(parsed, timeout)
        selector = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        try:
            connection.request(
                req.get_method(),
                selector,
                body=req.data,
                headers=dict(req.header_items()),
            )
            response = connection.getresponse()
        except self._stale_errors as exc:
            self.discard(connection)
            raise urllib.error.URLError(exc) from exc
        except OSError:
            self.discard(connection)
            raise
        except http.client.HTTPException as exc:
            self.discard(connection)
            raise urllib.error.URLError(exc) from exc
        if response.status < 200 or response.status >= 300:
            try:
                body = response.read()
            except http.client.IncompleteRead:
                self.discard(connection)
                raise
            if response.will_close or connection.sock is None:
                self.discard(connection)
            raise urllib.error.HTTPError(
                req.full_url,
                response.status,
                response.reason,
                response.headers,
                io.BytesIO(body),
            )
        return _ReusableHttpsResponse(self, connection, response)


class _ScoredTeacherTokens(list[TeacherToken]):
    """Completion tokens carrying the real echoed input-token count for image accounting."""

    def __init__(self, tokens: list[TeacherToken], *, input_tokens: int) -> None:
        super().__init__(tokens)
        self.input_tokens = input_tokens


def _validate_multimodal_echo(
    tokens, token_logprobs, offsets, completion_text: str
) -> tuple[int, int]:
    """Validate an image echo and locate the token suffix covering the completion exactly."""

    def _bad(message):
        raise TeacherError(message, permanent=True)

    if not all(isinstance(value, list) for value in (tokens, token_logprobs, offsets)):
        _bad(
            "teacher multimodal echo response logprobs fields are not all lists "
            f"(tokens={type(tokens).__name__}, token_logprobs={type(token_logprobs).__name__}, "
            f"text_offset={type(offsets).__name__})"
        )
    token_count = len(tokens)
    if len(token_logprobs) != token_count or len(offsets) != token_count:
        _bad(
            f"teacher multimodal echo response arrays disagree in length: tokens={token_count}, "
            f"token_logprobs={len(token_logprobs)}, text_offset={len(offsets)}"
        )
    previous_offset = None
    for offset in offsets:
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            _bad(f"teacher multimodal echo response text_offset has a non-numeric value: {offset!r}")
        if not math.isfinite(offset):
            _bad(f"teacher multimodal echo response text_offset is not finite: {offset!r}")
        if offset != int(offset):
            _bad(f"teacher multimodal echo response text_offset is not an integer: {offset!r}")
        if offset < 0:
            _bad(f"teacher multimodal echo response text_offset is negative: {offset!r}")
        if previous_offset is not None and offset < previous_offset:
            _bad(
                "teacher multimodal echo response text_offset is not non-decreasing: "
                f"{previous_offset!r} -> {offset!r}"
            )
        previous_offset = offset
    if token_count and int(offsets[0]) != 0:
        _bad(
            "teacher multimodal echo does not start at offset 0 "
            f"(first text_offset={int(offsets[0])})"
        )
    for logprob in token_logprobs:
        if logprob is None:
            continue
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            _bad(
                "teacher multimodal echo response token_logprobs has a non-numeric value: "
                f"{logprob!r}"
            )
        if not math.isfinite(logprob):
            _bad(
                "teacher multimodal echo response token_logprobs has a non-finite value: "
                f"{logprob!r}"
            )
        if logprob > _MAX_LOGPROB_ROUNDING_ERROR:
            _bad(
                "teacher multimodal echo response token_logprobs has a positive value "
                f"{logprob!r} (a log-probability cannot exceed 0)."
            )
    if not completion_text:
        return token_count, token_count
    suffix_start = token_count
    suffix_text = ""
    while suffix_start > 0 and len(suffix_text) < len(completion_text):
        suffix_start -= 1
        suffix_text = str(tokens[suffix_start]) + suffix_text
    if len(suffix_text) < len(completion_text) or not suffix_text.endswith(completion_text):
        _bad(
            "teacher multimodal echo tokens do not cover the exact completion suffix "
            f"({len(completion_text)} chars)"
        )
    return suffix_start, len(suffix_text) - len(completion_text)


class TeacherClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout: float = 90.0,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise TeacherError(
                "no teacher API key (FIREWORKS_API_KEY) available on the worker", permanent=True
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._transport = _ThreadLocalHttpsTransport()

    # -- transport -------------------------------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            remaining = _remaining_run_wall_seconds()
            if remaining is not None and remaining <= 0:
                raise TeacherError("teacher call exceeded the run wall deadline") from None
            timeout = self.timeout if remaining is None else min(self.timeout, remaining)
            req = urllib.request.Request(self.base_url + path, data=data, headers=headers)
            try:
                with self._transport.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                # HTTP response bodies may contain arbitrary private provider text. Classify only from
                # the structural status and request path, which are sufficient for retry policy.
                retryable = e.code in (408, 409, 425, 429, 500, 502, 503, 504)
                classification = "retryable" if retryable else "permanent"
                last_err = TeacherError(
                    f"teacher HTTP {e.code} on {path} ({classification})",
                    permanent=not retryable,
                )
                if not retryable:  # bad key (401/403) / model id (404) / bad request (400)
                    raise last_err from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = TeacherError(f"teacher transport error on {path}: {e}")
            except http.client.IncompleteRead as e:
                # The server sent an HTTP 200 header but the body was truncated mid-read() (dropped
                # connection / short Content-Length). IncompleteRead is an http.client.HTTPException,
                # NOT an OSError, so without this clause it escapes the retry loop. A stream of truncated
                # 200s could then burn every OPD step and fail as a permanent no-signal run instead of
                # retrying. Classify as TRANSIENT teacher infra, like the unparseable-body case below
                # (codex[bot]).
                last_err = TeacherError(f"teacher HTTP 200 body truncated mid-read on {path}: {e}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # HTTP 200 with a malformed / non-JSON body (a flaky proxy or gateway returning an
                # error page under a 200, a truncated read). The teacher contract is 200 => JSON, so
                # classify this as TRANSIENT teacher infra: retry in the loop, and if it persists the
                # exhausted last_err surfaces as a TeacherError -- NOT a raw JSONDecodeError, so a run
                # hammered by malformed 200s is retried as teacher infra instead of failing as permanent
                # no-signal.
                last_err = TeacherError(
                    f"teacher returned HTTP 200 with unparseable body on {path}: {e}"
                )
            if attempt + 1 >= self.max_retries:
                break
            delay = min(2.0 * (2**attempt), 20.0)
            remaining = _remaining_run_wall_seconds()
            if remaining is not None:
                if remaining <= 0:
                    raise TeacherError("teacher call exceeded the run wall deadline") from None
                delay = min(delay, remaining)
            if delay > 0:
                time.sleep(delay)
        raise last_err or TeacherError(f"teacher call to {path} failed")

    # -- echo scoring (gkd) ----------------------------------------------------------------------
    def score_many(self, items: list[tuple[str, str]]) -> list[list[TeacherToken]]:
        """Batch echo-score ``[(prompt_text, completion_text), ...]``.

        Fireworks' OpenAI-compatible completions endpoint accepts ``prompt`` as a list while still
        honoring ``echo=true`` and ``max_tokens=0``. OPD's hot path uses this to collapse many remote
        teacher round-trips into one request without changing the per-sample GKD signal.
        """
        if not items:
            return []
        fulls = [prompt_text + completion_text for prompt_text, completion_text in items]
        plens = [len(prompt_text) for prompt_text, _completion_text in items]
        resp = self._post(
            "/completions",
            {
                "model": self.model,
                "prompt": fulls if len(fulls) > 1 else fulls[0],
                "max_tokens": 0,
                "echo": True,
                # logprobs=1 is the minimum that returns per-token `token_logprobs` from the echo
                # endpoint; gkd uses only that realized logprob, not the top-k alternatives.
                "logprobs": 1,
                "temperature": 0,
            },
        )
        choices = resp.get("choices")
        if not isinstance(choices, list) or len(choices) != len(fulls):
            raise TeacherError(
                "teacher echo response returned the wrong number of choices "
                f"(expected {len(fulls)}, got {len(choices) if isinstance(choices, list) else type(choices).__name__})",
                permanent=True,
            )
        by_index: dict[int, dict] = {}
        for pos, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise TeacherError(
                    f"teacher echo response choice {pos} is not an object", permanent=True
                )
            idx = choice.get("index", pos)
            if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx >= len(fulls):
                raise TeacherError(
                    f"teacher echo response choice {pos} has invalid index {idx!r}",
                    permanent=True,
                )
            if idx in by_index:
                raise TeacherError(
                    f"teacher echo response duplicated choice index {idx}", permanent=True
                )
            by_index[idx] = choice
        if len(by_index) != len(fulls):
            missing = sorted(set(range(len(fulls))) - set(by_index))
            raise TeacherError(
                f"teacher echo response missing choice index(es): {missing}", permanent=True
            )
        return [
            self._tokens_from_choice(by_index[i], full=fulls[i], plen=plens[i])
            for i in range(len(fulls))
        ]

    def score_many_multimodal(
        self, items: list[tuple[str, str, list[str] | tuple[str, ...]]]
    ) -> list[list[TeacherToken]]:
        """Batch image-conditioned echo scoring with per-prompt base64 data URIs."""
        if not items:
            return []
        fulls = [prompt_text + completion_text for prompt_text, completion_text, _images in items]
        completions = [completion_text for _prompt_text, completion_text, _images in items]
        images = [list(item_images) for _prompt, _completion, item_images in items]
        resp = self._post(
            "/completions",
            {
                "model": self.model,
                "prompt": fulls if len(fulls) > 1 else fulls[0],
                "images": images if len(images) > 1 else images[0],
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": 0,
            },
        )
        choices = resp.get("choices")
        if not isinstance(choices, list) or len(choices) != len(fulls):
            raise TeacherError(
                "teacher multimodal echo response returned the wrong number of choices "
                f"(expected {len(fulls)}, got "
                f"{len(choices) if isinstance(choices, list) else type(choices).__name__})",
                permanent=True,
            )
        by_index: dict[int, dict] = {}
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise TeacherError(
                    f"teacher multimodal echo response choice {position} is not an object",
                    permanent=True,
                )
            index = choice.get("index", position)
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(fulls)
            ):
                raise TeacherError(
                    f"teacher multimodal echo response choice {position} has invalid index {index!r}",
                    permanent=True,
                )
            if index in by_index:
                raise TeacherError(
                    f"teacher multimodal echo response duplicated choice index {index}",
                    permanent=True,
                )
            by_index[index] = choice
        if len(by_index) != len(fulls):
            missing = sorted(set(range(len(fulls))) - set(by_index))
            raise TeacherError(
                f"teacher multimodal echo response missing choice index(es): {missing}",
                permanent=True,
            )
        return [
            self._multimodal_tokens_from_choice(by_index[index], completions[index])
            for index in range(len(fulls))
        ]

    def score(self, prompt_text: str, completion_text: str) -> list[TeacherToken]:
        """Echo-score one completion and return completion-region teacher tokens."""
        return self.score_many([(prompt_text, completion_text)])[0]

    def _tokens_from_choice(self, choice: dict, *, full: str, plen: int) -> list[TeacherToken]:
        completion_text = full[plen:]
        try:
            lp = choice["logprobs"]
            tokens = lp["tokens"]
            token_logprobs = lp["token_logprobs"]
            offsets = lp["text_offset"]
        except (KeyError, IndexError, TypeError) as e:
            raise TeacherError(
                f"teacher echo response missing logprobs: {e}", permanent=True
            ) from e
        spans = _validate_echo(tokens, token_logprobs, offsets, full)
        out: list[TeacherToken] = []
        for i, (start, end) in enumerate(spans):
            # Drop tokens that lie ENTIRELY in the prompt (end <= plen). KEEP a token that CROSSES
            # the prompt/completion boundary (start < plen < end): when the prompt ends in whitespace
            # (``Assistant: ``), a leading-space teacher tokenizer merges that trailing space with the
            # first completion word into one token. Dropping it (the old start<plen rule) would strip
            # the first completion token's signal — and for a ONE-token completion that leaves zero
            # teacher tokens and the whole sample is skipped. Clamp such a token's completion span to
            # [0, end-plen); its logprob carries a small fixed prompt-whitespace contribution, far
            # preferable to losing the token (and, for short completions, the entire sample).
            if end <= plen:
                continue
            # A null (None) realized logprob is legitimate ONLY for unscored prompt context; a token we
            # KEEP here overlaps the completion (end > plen), so a null means the teacher did not score
            # this completion token. Coercing it to 0.0 (log-prob 1.0 == full confidence) would train the
            # gkd loss on bogus teacher confidence, so abort like the other teacher-contract violations
            # rather than distil a fabricated signal (codex[bot]). Prompt-context nulls (end <= plen) are
            # already skipped above and never reach this check.
            if token_logprobs[i] is None:
                raise TeacherError(
                    f"teacher echo response has a null token_logprob for a completion token "
                    f"(index {i}, char offset {start}); the teacher did not score it",
                    permanent=True,
                )
            logprob = float(token_logprobs[i])
            out.append(
                TeacherToken(
                    text=tokens[i],
                    logprob=logprob,
                    start=max(0, start - plen),
                    end=end - plen,
                )
            )
        # A non-empty completion must produce at least one completion-region teacher token. An empty
        # `out` means every returned token fell ENTIRELY in the prompt (end <= plen) -- a prompt-only /
        # no-completion-token echo. The caller would otherwise see this teacher call as "ok" (it returned
        # without error) yet get no gkd signal from it, so a persistently prompt-only teacher
        # burns every OPD step before the generic no-signal failure instead of aborting on the first bad
        # echo. Reject as a PERMANENT contract break (codex[bot]). This is the downstream backstop to the
        # tiling check above: a covering echo of a non-empty completion always yields a completion token,
        # so in practice the truncation guard fires first, but this keeps the invariant explicit.
        if completion_text.strip() and not out:
            raise TeacherError(
                "teacher echo returned no completion-region tokens for a non-empty completion "
                f"({len(completion_text)} chars); the echo scored only prompt context.",
                permanent=True,
            )
        return out

    def _multimodal_tokens_from_choice(
        self, choice: dict, completion_text: str
    ) -> list[TeacherToken]:
        try:
            logprobs = choice["logprobs"]
            tokens = logprobs["tokens"]
            token_logprobs = logprobs["token_logprobs"]
            offsets = logprobs["text_offset"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TeacherError(
                f"teacher multimodal echo response missing logprobs: {exc}", permanent=True
            ) from exc
        suffix_start, prefix_chars = _validate_multimodal_echo(
            tokens, token_logprobs, offsets, completion_text
        )
        output: list[TeacherToken] = []
        cursor = -prefix_chars
        for index in range(suffix_start, len(tokens)):
            text = str(tokens[index])
            end = cursor + len(text)
            # include only tokens that overlap the completion region [0, len(completion_text)).
            # the upper bound drops any trailing (e.g. zero-width) token that starts at or after
            # the completion end, so it cannot add a spurious teacher logprob to the alignment.
            if end > 0 and cursor < len(completion_text):
                if token_logprobs[index] is None:
                    raise TeacherError(
                        "teacher multimodal echo response has a null token_logprob for a "
                        f"completion token (index {index})",
                        permanent=True,
                    )
                output.append(
                    TeacherToken(
                        text=text,
                        logprob=float(token_logprobs[index]),
                        start=max(0, cursor),
                        end=end,
                    )
                )
            cursor = end
        if completion_text and not output:
            raise TeacherError(
                "teacher multimodal echo returned no scored completion token for a non-empty "
                f"completion ({len(completion_text)} chars)",
                permanent=True,
            )
        return _ScoredTeacherTokens(output, input_tokens=len(tokens))
