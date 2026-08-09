"""sample realized rollout length through a hosted inference api, off gpu.

realized completion length is a property of (model, prompt, sampling params) and NOT of the
hardware: a faster card changes how QUICKLY tokens are emitted, never HOW MANY. so the quantity the
grpo/opd quote is missing can be measured against a hosted endpoint on cpu, instead of renting the
gpu whose price this measurement exists to compute.

what this module does NOT do is measure SECONDS. generation seconds are provider-, card- and
moment-specific, so only the token COUNT transfers to the run being quoted.

the endpoint is openai-compatible and named only in configuration, never in identifiers here: the
catalog of hosted models is data, and a second provider must not require a second module.
"""

from __future__ import annotations

import json
import os
import statistics
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

# where the samples come from. an openai-compatible chat-completions endpoint is the whole contract:
# the sampler reads `usage.completion_tokens`, `usage.prompt_tokens` and `finish_reason`, which are
# the fields such an api bills on, so a provider has every incentive to report them correctly.
ROLLOUT_SAMPLER_BASE_URL_ENV = "FLASH_ROLLOUT_SAMPLER_BASE_URL"
ROLLOUT_SAMPLER_API_KEY_ENV = "FLASH_ROLLOUT_SAMPLER_API_KEY"
DEFAULT_ROLLOUT_SAMPLER_BASE_URL = "https://openrouter.ai/api/v1"

# per-request ceiling. a hosted generation at a 2048-token cap takes tens of seconds; this bounds a
# hung request without cutting off a legitimately long one, whose length is exactly the tail this
# profile exists to capture.
REQUEST_TIMEOUT_S = 180.0

# prompt-mix dominates the variance (measured: between-prompt 304846 vs within-prompt 61082, a word
# problem running a median 1642 tokens against ~290 for arithmetic), so spread draws across DISTINCT
# prompts before repeating any one of them.
SAMPLE_POLICY_DISTINCT_PROMPTS = "distinct-prompts-round-robin"


@dataclass(frozen=True)
class RolloutSample:
    """one realized generation: the token counts, and why it stopped."""

    prompt_tokens: int
    completion_tokens: int
    truncated: bool


@dataclass(frozen=True)
class RolloutSampling:
    """what a sampling pass observed, before any trust verdict is applied."""

    samples: tuple[RolloutSample, ...]
    failures: int
    sampled_prompts: int

    @property
    def completed(self) -> int:
        return len(self.samples)

    @property
    def truncated(self) -> int:
        return sum(1 for s in self.samples if s.truncated)

    def completion_tokens(self) -> list[int]:
        return [s.completion_tokens for s in self.samples]

    def summary(self) -> dict[str, object]:
        """aggregates only. no prompts, completions, token ids or credentials leave this process.

        returns the distribution rather than a mean alone: a mean cannot tell a uniformly short
        workload from a bimodal one whose long tail sets the step time.
        """
        lengths = sorted(self.completion_tokens())
        prompts = [s.prompt_tokens for s in self.samples]
        if not lengths:
            return {
                "sampled_prompts": self.sampled_prompts,
                "completed_rollouts": 0,
                "failed_rollouts": self.failures,
            }
        return {
            "sampled_prompts": self.sampled_prompts,
            "completed_rollouts": len(lengths),
            "failed_rollouts": self.failures,
            "completion_tokens_mean": statistics.fmean(lengths),
            "completion_tokens_p50": _percentile(lengths, 0.50),
            "completion_tokens_p90": _percentile(lengths, 0.90),
            "completion_tokens_max": lengths[-1],
            "prompt_tokens_mean": statistics.fmean(prompts),
            "truncated_rollouts": self.truncated,
            "eos_rollouts": len(lengths) - self.truncated,
            "sample_policy": SAMPLE_POLICY_DISTINCT_PROMPTS,
        }


def _percentile(sorted_values: Sequence[int], q: float) -> int:
    """nearest-rank percentile of an already-sorted sequence."""
    if not sorted_values:
        return 0
    index = max(0, min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1))))
    return int(sorted_values[index])


def sampler_credentials() -> tuple[str, str]:
    """(base_url, api_key) for the hosted sampler; api_key is "" when unconfigured."""
    base = (os.environ.get(ROLLOUT_SAMPLER_BASE_URL_ENV) or "").strip()
    key = (os.environ.get(ROLLOUT_SAMPLER_API_KEY_ENV) or "").strip()
    return (base or DEFAULT_ROLLOUT_SAMPLER_BASE_URL), key


def sample_rollouts(
    *,
    model: str,
    prompts: Sequence[str],
    rollouts: int,
    max_completion_tokens: int,
    temperature: float | None,
    base_url: str,
    api_key: str,
) -> RolloutSampling:
    """draw ``rollouts`` generations, spreading across ``prompts`` before repeating one.

    never raises for a failed draw: a profile is advisory evidence, and a partial sample is reported
    as partial so the trust verdict can reject it. a caller that gets zero samples simply has no
    measurement, which returns the quote to the declared cap.
    """
    if not prompts or rollouts <= 0:
        return RolloutSampling(samples=(), failures=0, sampled_prompts=0)

    collected: list[RolloutSample] = []
    failures = 0
    used_prompts: set[int] = set()
    for index in range(rollouts):
        # round-robin: draw one from every distinct prompt before taking a second from any.
        slot = index % len(prompts)
        used_prompts.add(slot)
        sample = _one_completion(
            model=model,
            prompt=prompts[slot],
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )
        if sample is None:
            failures += 1
        else:
            collected.append(sample)
    return RolloutSampling(
        samples=tuple(collected),
        failures=failures,
        sampled_prompts=len(used_prompts),
    )


def _one_completion(
    *,
    model: str,
    prompt: str,
    max_completion_tokens: int,
    temperature: float | None,
    base_url: str,
    api_key: str,
) -> RolloutSample | None:
    """one generation's token counts, or None when the draw failed for any reason."""
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_completion_tokens),
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
            body = json.load(response)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None

    usage = body.get("usage") if isinstance(body, dict) else None
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(usage, dict) or not isinstance(choices, list) or not choices:
        return None
    try:
        completion_tokens = int(usage["completion_tokens"])
        prompt_tokens = int(usage["prompt_tokens"])
    except (KeyError, TypeError, ValueError):
        return None
    if completion_tokens <= 0 or prompt_tokens <= 0:
        return None
    first = choices[0]
    finish = first.get("finish_reason") if isinstance(first, dict) else None
    # a truncated draw contributes the CAP rather than its true length, biasing the mean downward --
    # the direction that underbills. counted, so the trust verdict can reject a censored sample
    # rather than report it as a measurement.
    return RolloutSample(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        truncated=(finish == "length"),
    )
