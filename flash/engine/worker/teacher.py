"""Fireworks-hosted teacher client for on-policy distillation.

Reaches the GLM-5.2 teacher over the OpenAI-compatible API using only the Python standard library (no
``openai``/``requests`` dependency — the worker runs on stdlib here). One operation:

- ``score``: echo-score a student completion (``echo=true, logprobs=1, max_tokens=0``) to get the
  teacher's per-token REALIZED logprobs + CHARACTER offsets over the completion region only. This is
  the on-policy signal for the groupwise reverse-KL (gkd) loss — it pays for input tokens only.

All heavy work is inside methods; importing this module is CPU/offline-safe.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from flash.engine.worker.tokenizer_align import TeacherToken


class TeacherError(RuntimeError):
    """Teacher API call failed after retries."""


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
            raise TeacherError("no teacher API key (FIREWORKS_API_KEY) available on the worker")
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
                # Retry transient server/rate-limit errors; fail fast on 4xx client errors.
                body_txt = e.read().decode("utf-8", "replace")[:300] if e.fp else ""
                last_err = TeacherError(f"teacher HTTP {e.code} on {path}: {body_txt}")
                if e.code not in (408, 409, 425, 429, 500, 502, 503, 504):
                    raise last_err from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = TeacherError(f"teacher transport error on {path}: {e}")
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
            raise TeacherError(f"teacher echo response missing logprobs: {e}") from e
        n = len(tokens)
        out: list[TeacherToken] = []
        for i in range(n):
            start = int(offsets[i])
            end = int(offsets[i + 1]) if i + 1 < n else len(full)
            # Keep only tokens that lie ENTIRELY in the completion (start >= plen). This drops
            # prompt-only tokens AND boundary-crossers: when the prompt ends in whitespace
            # (``Assistant: ``), a leading-space merge token can span start<plen<end; its logprob
            # includes fixed prompt text, so it is not a clean completion-token signal. The tiny
            # sliver it would have covered is absorbed by gkd's student-only-span merge.
            if start < plen:
                continue
            logprob = float(token_logprobs[i]) if token_logprobs[i] is not None else 0.0
            out.append(
                TeacherToken(
                    text=tokens[i],
                    logprob=logprob,
                    start=start - plen,
                    end=end - plen,
                )
            )
        return out
