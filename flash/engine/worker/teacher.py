"""Fireworks-hosted teacher client for on-policy distillation.

Reaches the GLM teacher over the OpenAI-compatible API using only the Python standard library (no
``openai``/``requests`` dependency — the worker image already runs on stdlib here). Two operations:

- ``score``: echo-score a student completion (``echo=true, logprobs=k, max_tokens=0``) to get the
  teacher's per-token logprobs + top-k + CHARACTER offsets over the completion region only. This is
  the on-policy signal for the ``align``/``uld`` strategies — it pays for input tokens only.
- ``generate``: sample a target completion from the teacher (chat completions) for the ``seqkd``
  strategy; GLM's inline reasoning is stripped to the final answer.

All heavy work is inside methods; importing this module is CPU/offline-safe.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from flash.engine.worker.tokenizer_align import TeacherToken

# Marks where GLM-style inline reasoning ends; the final answer follows the last occurrence.
_REASONING_MARKERS = ("</think>", "</reasoning>", "[/think]")

# Fireworks' /completions echo endpoint caps `logprobs` (top-k alternatives per token) at 5.
_MAX_ECHO_LOGPROBS = 5


class TeacherError(RuntimeError):
    """Teacher API call failed after retries."""


class TeacherClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        top_logprobs: int = 10,
        timeout: float = 90.0,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise TeacherError("no teacher API key (FIREWORKS_API_KEY) available on the worker")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.top_logprobs = max(1, int(top_logprobs))
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

    # -- scoring (align / uld) -------------------------------------------------------------------
    def score(self, prompt_text: str, completion_text: str) -> list[TeacherToken]:
        """Echo-score ``prompt_text + completion_text`` and return the teacher tokens that fall in
        the COMPLETION region, with char offsets rebased to the completion (0 = first completion
        char). ``max_tokens=0`` so this pays for input tokens only (no generation)."""
        full = prompt_text + completion_text
        plen = len(prompt_text)
        resp = self._post(
            "/completions",
            {
                "model": self.model,
                "prompt": full,
                "max_tokens": 0,
                "echo": True,
                # Fireworks caps the echo endpoint's top-k at _MAX_ECHO_LOGPROBS; clamp so a larger
                # configured k never 400s the scoring call.
                "logprobs": min(self.top_logprobs, _MAX_ECHO_LOGPROBS),
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
        tops = lp.get("top_logprobs") or [None] * len(tokens)
        n = len(tokens)
        out: list[TeacherToken] = []
        for i in range(n):
            start = int(offsets[i])
            end = int(offsets[i + 1]) if i + 1 < n else len(full)
            if end <= plen:  # token lies entirely within the prompt
                continue
            logprob = float(token_logprobs[i]) if token_logprobs[i] is not None else 0.0
            top_map = dict(tops[i] or {})
            # Guarantee the realized token is represented (it may fall outside the top-k list).
            top_map.setdefault(tokens[i], logprob)
            out.append(
                TeacherToken(
                    text=tokens[i],
                    logprob=logprob,
                    top=tuple((s, float(v)) for s, v in top_map.items()),
                    start=max(0, start - plen),
                    end=max(0, end - plen),
                )
            )
        return out

    # -- generation (seqkd) ----------------------------------------------------------------------
    def generate(self, messages: list[dict], *, max_tokens: int, temperature: float = 0.7) -> str:
        """Sample a teacher target for a prompt (seqkd). Returns the final answer text with any
        leading GLM reasoning trace stripped."""
        resp = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            },
        )
        try:
            content = resp["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise TeacherError(f"teacher chat response missing content: {e}") from e
        return strip_reasoning(content)


def strip_reasoning(text: str) -> str:
    """Return the answer text after the last reasoning marker (GLM thinks inline by default)."""
    if not text:
        return ""
    lowered = text.lower()
    cut = -1
    for marker in _REASONING_MARKERS:
        idx = lowered.rfind(marker)
        if idx != -1:
            cut = max(cut, idx + len(marker))
    return text[cut:].strip() if cut != -1 else text.strip()
