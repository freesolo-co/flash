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
import json
import time
import urllib.error
import urllib.request

from flash.engine.worker.tokenizer_align import TeacherToken


class TeacherError(RuntimeError):
    """Teacher API call failed. ``permanent`` marks non-retryable causes (bad key / model id /
    malformed request/response) so the worker aborts the run immediately instead of skipping every
    remaining sample and burning the whole paid GPU allocation before failing."""

    def __init__(self, *args, permanent: bool = False) -> None:
        super().__init__(*args)
        self.permanent = permanent


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

    # -- transport -------------------------------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(self.base_url + path, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                # Retry transient server/rate-limit errors; fail fast on 4xx client errors. Guard the
                # error-body preview read: a retryable 429/5xx whose body is truncated or times out
                # makes e.read() raise IncompleteRead/OSError BEFORE last_err is set, so it would
                # escape the loop as a generic exception that _train_one swallows without classifying
                # (last_teacher_status stays None) — repeated retryable errors then end as permanent
                # no-signal instead of a retriable outage. Classify by e.code regardless (codex[bot]).
                try:
                    body_txt = e.read().decode("utf-8", "replace")[:300] if e.fp else ""
                except (http.client.IncompleteRead, OSError):
                    body_txt = "<error body unavailable>"
                retryable = e.code in (408, 409, 425, 429, 500, 502, 503, 504)
                last_err = TeacherError(
                    f"teacher HTTP {e.code} on {path}: {body_txt}", permanent=not retryable
                )
                if not retryable:  # bad key (401/403) / model id (404) / bad request (400)
                    raise last_err from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = TeacherError(f"teacher transport error on {path}: {e}")
            except http.client.IncompleteRead as e:
                # The server sent an HTTP 200 header but the body was truncated mid-read() (dropped
                # connection / short Content-Length). IncompleteRead is an http.client.HTTPException,
                # NOT an OSError, so without this clause it escapes the retry loop and _train_one
                # swallows it as an unclassified skipped sample (last_teacher_status stays None) -- a
                # stream of truncated 200s then burns every OPD step and fails as a permanent no-signal
                # run instead of retrying. Classify as TRANSIENT teacher infra, like the unparseable-
                # body case below (codex[bot]).
                last_err = TeacherError(f"teacher HTTP 200 body truncated mid-read on {path}: {e}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # HTTP 200 with a malformed / non-JSON body (a flaky proxy or gateway returning an
                # error page under a 200, a truncated read). The teacher contract is 200 => JSON, so
                # classify this as TRANSIENT teacher infra: retry in the loop, and if it persists the
                # exhausted last_err surfaces as a TeacherError -- NOT a raw JSONDecodeError, which
                # _train_one would swallow as an unclassified skip (last_teacher_status stays None) and
                # a run hammered by malformed 200s would then fail as permanent no-signal instead of
                # being retried as teacher infra.
                last_err = TeacherError(
                    f"teacher returned HTTP 200 with unparseable body on {path}: {e}"
                )
            time.sleep(min(2.0 * (2**attempt), 20.0))
        raise last_err or TeacherError(f"teacher call to {path} failed")

    # -- echo scoring (gkd) ----------------------------------------------------------------------
    def score(self, prompt_text: str, completion_text: str) -> list[TeacherToken]:
        """Echo-score ``prompt_text + completion_text`` and return the teacher tokens that fall in
        the COMPLETION region, each with its realized logprob and char offsets rebased to the
        completion (0 = first completion char). ``max_tokens=0`` so this pays for input tokens only
        (no generation); gkd needs only the realized per-token logprob (no top-k candidates)."""
        full = prompt_text + completion_text
        plen = len(prompt_text)
        resp = self._post(
            "/completions",
            {
                "model": self.model,
                "prompt": full,
                "max_tokens": 0,
                "echo": True,
                # logprobs=1 is the minimum that returns per-token `token_logprobs` from the echo
                # endpoint; gkd uses only that realized logprob, not the top-k alternatives.
                "logprobs": 1,
                "temperature": 0,
            },
        )
        try:
            lp = resp["choices"][0]["logprobs"]
            tokens = lp["tokens"]
            token_logprobs = lp["token_logprobs"]
            offsets = lp["text_offset"]
        except (KeyError, IndexError, TypeError) as e:
            raise TeacherError(
                f"teacher echo response missing logprobs: {e}", permanent=True
            ) from e
        # The length checks and index loop below assume these are sequences. A malformed 200 with
        # token_logprobs=null or a scalar text_offset would make len()/indexing raise TypeError HERE,
        # OUTSIDE the guard above — _train_one then swallows it as a generic (transient) skipped sample
        # without setting last_teacher_status, so a consistently malformed teacher burns every OPD step
        # before the no-signal failure. Reject non-list fields up front as a PERMANENT contract break.
        if not all(isinstance(v, list) for v in (tokens, token_logprobs, offsets)):
            raise TeacherError(
                "teacher echo response logprobs fields are not all lists "
                f"(tokens={type(tokens).__name__}, token_logprobs={type(token_logprobs).__name__}, "
                f"text_offset={type(offsets).__name__})",
                permanent=True,
            )
        n = len(tokens)
        # A well-formed echo response returns tokens / token_logprobs / text_offset of EQUAL length.
        # The loop below indexes token_logprobs[i] and offsets[i] for every i, so a short array (a
        # malformed 200) would raise IndexError there — and, unlike the KeyError/TypeError guarded
        # above, that IndexError escapes the method as a generic (non-TeacherError) exception. In
        # _train_one that is caught by the broad `except Exception` and treated as a TRANSIENT skipped
        # sample, so a teacher that consistently returns malformed arrays would burn every remaining
        # OPD step before the run fails with "no trained step". Treat the length mismatch as PERMANENT
        # (a broken teacher-response contract will not fix itself on retry) so the worker aborts now.
        if len(token_logprobs) < n or len(offsets) < n:
            raise TeacherError(
                f"teacher echo response arrays disagree in length: tokens={n}, "
                f"token_logprobs={len(token_logprobs)}, text_offset={len(offsets)}",
                permanent=True,
            )
        # The loop coerces each offset with int(offsets[i]). A malformed 200 can still put a
        # non-numeric value in text_offset (e.g. [0, null] or [0, "bad"]) that passes the list/length
        # guards above -- int() then raises TypeError/ValueError OUTSIDE any TeacherError, so _train_one
        # swallows it as an unclassified skip and a consistently malformed teacher burns every OPD step
        # (codex[bot]). Validate the offsets are numeric (bools excluded) and non-decreasing up front
        # and reject as a PERMANENT contract break. token_logprobs[i] may still be null (handled below).
        prev_off = None
        for o in offsets[:n]:
            if isinstance(o, bool) or not isinstance(o, (int, float)):
                raise TeacherError(
                    f"teacher echo response text_offset has a non-numeric value: {o!r}",
                    permanent=True,
                )
            if prev_off is not None and o < prev_off:
                raise TeacherError(
                    f"teacher echo response text_offset is not non-decreasing: {prev_off!r} -> {o!r}",
                    permanent=True,
                )
            prev_off = o
        out: list[TeacherToken] = []
        for i in range(n):
            start = int(offsets[i])
            end = int(offsets[i + 1]) if i + 1 < n else len(full)
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
            logprob = float(token_logprobs[i]) if token_logprobs[i] is not None else 0.0
            out.append(
                TeacherToken(
                    text=tokens[i],
                    logprob=logprob,
                    start=max(0, start - plen),
                    end=end - plen,
                )
            )
        return out
