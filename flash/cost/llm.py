"""Claude-backed cost estimator.

``LLMCostEstimator(version)`` prices a ``RunConfig`` by asking Claude with the
version-``v`` system prompt from ``prompts.py``. The Anthropic SDK is imported lazily
so the rest of the cost package (and its tests) carry no hard dependency on it; a
``transport`` callable can be injected to run the estimator fully offline (the unit
tests use this). The model's JSON reply is parsed leniently -- a missing/garbled
number degrades to ``ok=False`` rather than raising.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from .config import RunConfig
from .prompts import prompt_for_version, render_run

# (system_prompt, user_message) -> assistant text. Lets tests/offline runs bypass the API.
Transport = Callable[[str, str], str]

DEFAULT_MODEL = "claude-opus-4-8"


@dataclass(frozen=True)
class LLMEstimate:
    """One LLM pricing of a run."""

    version: int
    cost_usd: float | None
    gpu: str | None
    reasoning: str
    raw_text: str
    ok: bool
    error: str | None = None


def _extract(text: str) -> tuple[float | None, str | None, str]:
    """Lenient parse of the model reply -> (cost_usd, gpu, reasoning)."""
    # Prefer a JSON object that actually carries cost_usd (scan all candidates).
    for m in re.finditer(r"\{[^{}]*\"cost_usd\"[^{}]*\}", text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        cost = obj.get("cost_usd")
        if isinstance(cost, (int, float)):
            # gpu must be a string for the downstream _gpu_match: a model that returns a
            # numeric/structured gpu (e.g. {"gpu": 5090}) would otherwise crash the sweep
            # on est_gpu.lower(). Coerce non-strings to str; drop falsy values to None.
            # A numeric 0 (or the string "0") means "no GPU" -- treat it as absent, not a
            # falsy-but-present "0" that would then be graded against the truth class.
            raw_gpu = obj.get("gpu")
            gpu = str(raw_gpu) if raw_gpu not in (None, "", 0, "0") else None
            return float(cost), gpu, str(obj.get("reasoning", ""))
    # Fallbacks: a "cost_usd": N pair, then the first dollar figure anywhere.
    kv = re.search(r"cost_usd\"?\s*[:=]\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", text)
    if kv:
        gpu = re.search(r"gpu\"?\s*[:=]\s*\"?([A-Za-z0-9 ]+)", text)
        return float(kv.group(1)), (gpu.group(1).strip() if gpu else None), ""
    dollar = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)", text)
    if dollar:
        return float(dollar.group(1)), None, ""
    return None, None, ""


class LLMCostEstimator:
    """Estimate run cost with Claude using the version-``v`` framework prompt."""

    def __init__(
        self,
        version: int,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 4000,
        client: object | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.version = version
        self.system = prompt_for_version(version)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = client
        self._transport = transport

    def _anthropic(self) -> object:
        if self._client is None:
            import anthropic  # lazy: no hard dep for the rest of the package

            self._client = anthropic.Anthropic()
        return self._client

    def _complete(self, system: str, user: str) -> str:
        if self._transport is not None:
            return self._transport(system, user)
        client = self._anthropic()
        resp = client.messages.create(  # type: ignore[attr-defined]
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt is identical across every run priced at this version, but
            # the user message changes per run. Put the cache breakpoint on the (stable)
            # system block itself -- a top-level cache_control lands on the changing final
            # user block, so every run pays a fresh cache write instead of reusing the long
            # version prompt. Caching is a prefix match; the system block is the shared prefix.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def estimate(self, config: RunConfig) -> LLMEstimate:
        user = render_run(config)
        try:
            text = self._complete(self.system, user)
        except Exception as exc:  # network / API failure -> a failed estimate, not a crash
            return LLMEstimate(self.version, None, None, "", "", ok=False, error=str(exc))
        cost, gpu, reasoning = _extract(text)
        return LLMEstimate(
            version=self.version,
            cost_usd=cost,
            gpu=gpu,
            reasoning=reasoning,
            raw_text=text,
            ok=cost is not None and cost >= 0,
            error=None if cost is not None else "no cost_usd in reply",
        )
