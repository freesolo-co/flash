"""Sampled measurement of what one grpo/opd rollout actually generates and costs to grade.

The quote currently bills ``max_completion_tokens`` -- a capacity CEILING -- as though it were the
work a step performs. Measured against a hosted copy of the same model, realized generation ran
0.323x of a 2048-token cap, so the cap overbills generation by 3.10x, and ``gen_tokens`` feeds BOTH
the rollout and update terms of every step. That single substitution is what this module exists to
enable.

Why this measures off-GPU, which is the whole design:

    How many tokens a model emits before EOS is a property of (model, prompt, sampling params).
    It is NOT a property of the card. An H100 emits the same completion an A100 does, faster.

So the token distribution -- the part the quote is missing -- is measurable through any inference
endpoint serving the same weights, for about $0.001 per profile, instead of renting a GPU. The
per-card throughput model that turns tokens into seconds is untouched; this only stops feeding it
a fabricated token count.

The two clocks are deliberately NOT symmetric:

- token counts are structural. They are keyed, cached, and do not expire.
- SECONDS are provider-, card- and moment-specific. A measured seconds-per-completion from a
  hosted endpoint says nothing about a rented card, so this module does not pretend otherwise:
  it records what it measured and stamps ``measured_at``, and the profile ages those out.

Reward latency is measured locally rather than remotely, because a grader is the environment's own
python and runs identically wherever it runs. That part genuinely is portable.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from flash.engine.reward_profile import call_bounded

# a sampled rollout is drawn per prompt in a group, the same shape grpo itself samples with.
# between-prompt variance measured 5x within-prompt variance (one word problem ran a median 1642
# tokens against ~290 for three arithmetic prompts), so DISTINCT prompts buy more precision per
# call than repeats of one. the worker spreads across prompts first and repeats second.
DEFAULT_PROFILE_GROUP_SIZE = 4

# one hosted generation of a few hundred tokens settles in well under this; the ceiling exists so a
# wedged endpoint cannot hold a quote open, not to bound normal work.
DEFAULT_ROLLOUT_TIMEOUT_S = 240.0

# total wall budget for the whole sampling pass. a profile that cannot finish inside this returns
# what it gathered and lets the trust verdict refuse it, rather than blocking a quote indefinitely.
DEFAULT_PROFILE_BUDGET_S = 900.0


class RolloutProfileUnavailable(RuntimeError):
    """No endpoint can serve this model, so no rollout sample can be taken.

    This is NOT a failure: 3 of the 6 catalog models are too small for anyone to host. The caller
    falls back to the declared cap, which is exactly today's behaviour, so an unavailable profile
    costs nothing beyond the improvement it could not deliver.
    """


@dataclass(frozen=True)
class SampledRollout:
    """One generated completion, reduced to the quantities the cost model consumes."""

    prompt_tokens: int
    completion_tokens: int
    truncated: bool
    seconds: float


def _endpoint() -> tuple[str, str]:
    """(base_url, api_key) for the sampling endpoint, or raise if none is configured."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RolloutProfileUnavailable("no rollout sampling endpoint is configured")
    base = os.environ.get("FLASH_ROLLOUT_PROFILE_BASE_URL", "https://openrouter.ai/api/v1").strip()
    return base.rstrip("/"), key


def sample_one_rollout(
    *,
    messages: Sequence[dict],
    served_model: str,
    max_completion_tokens: int,
    temperature: float | None,
    timeout_s: float = DEFAULT_ROLLOUT_TIMEOUT_S,
) -> SampledRollout:
    """Generate one completion and return only its measured shape, never its text.

    ``messages`` is the environment's own ``prompt_messages(example)`` output, unmodified: the
    profile must ask what training asks. Reformatting it here would measure the reformatting.

    Token counts come from the endpoint's ``usage`` block rather than from re-tokenizing the
    response locally. That block is what the provider BILLS on, so it is the number with an
    incentive behind its accuracy, and it counts the reasoning tokens a hybrid-thinking model emits
    but does not return in ``content`` -- 61% of measured output, invisible to any local count.
    """
    base, key = _endpoint()
    payload: dict[str, object] = {
        "model": served_model,
        "messages": list(messages),
        "max_tokens": int(max_completion_tokens),
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started

    choices = body.get("choices") or []
    if not choices:
        raise RolloutProfileUnavailable("sampling endpoint returned no choice")
    usage = body.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if not isinstance(completion_tokens, int) or not isinstance(prompt_tokens, int):
        # without an authoritative count there is nothing to measure. inferring one from the
        # returned text would silently drop reasoning tokens and under-report the workload.
        raise RolloutProfileUnavailable("sampling endpoint returned no usage token counts")
    return SampledRollout(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        # a completion stopped at the ceiling reports the ceiling, not the length the model would
        # have produced. it has to be counted separately or it drags the mean down and underbills.
        truncated=str(choices[0].get("finish_reason") or "") == "length",
        seconds=elapsed,
    )


def _percentile(values: Sequence[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(q * (len(ordered) - 1))
    return int(ordered[idx])


@dataclass(frozen=True)
class RolloutSample:
    """The aggregate a profile is built from. Aggregates only: no prompts, no completions."""

    sampled_prompts: int
    completed: int
    failed: int
    completion_tokens_mean: float
    completion_tokens_p50: int
    completion_tokens_p90: int
    completion_tokens_max: int
    prompt_tokens_mean: float
    truncated: int
    eos: int
    seconds_per_completion: float

    @classmethod
    def from_rollouts(
        cls, rollouts: Sequence[SampledRollout], *, sampled_prompts: int, failed: int
    ) -> RolloutSample:
        if not rollouts:
            return cls(
                sampled_prompts=sampled_prompts,
                completed=0,
                failed=failed,
                completion_tokens_mean=0.0,
                completion_tokens_p50=0,
                completion_tokens_p90=0,
                completion_tokens_max=0,
                prompt_tokens_mean=0.0,
                truncated=0,
                eos=0,
                seconds_per_completion=0.0,
            )
        completion = [r.completion_tokens for r in rollouts]
        truncated = sum(1 for r in rollouts if r.truncated)
        return cls(
            sampled_prompts=sampled_prompts,
            completed=len(rollouts),
            failed=failed,
            completion_tokens_mean=statistics.mean(completion),
            completion_tokens_p50=_percentile(completion, 0.5),
            completion_tokens_p90=_percentile(completion, 0.9),
            completion_tokens_max=max(completion),
            prompt_tokens_mean=statistics.mean([r.prompt_tokens for r in rollouts]),
            truncated=truncated,
            eos=len(rollouts) - truncated,
            # median, not mean: one slow response from a busy endpoint would drag a mean and this
            # number is already the least transferable thing here.
            seconds_per_completion=statistics.median([r.seconds for r in rollouts]),
        )


def sample_rollouts(
    *,
    prompts: Sequence[Sequence[dict]],
    served_model: str,
    max_completion_tokens: int,
    temperature: float | None,
    target_rollouts: int,
    group_size: int = DEFAULT_PROFILE_GROUP_SIZE,
    budget_s: float = DEFAULT_PROFILE_BUDGET_S,
    sample_one: Callable[..., SampledRollout] = sample_one_rollout,
) -> RolloutSample:
    """Draw ``target_rollouts`` completions, spread across distinct prompts first.

    Ordering matters more than it looks: between-prompt variance measured 5x within-prompt, so
    ``target_rollouts`` draws taken from one prompt describe THAT PROMPT, not the environment.
    Draws are therefore issued round-robin across prompts, and a run that exhausts its budget early
    still ends with a sample spread across the environment rather than concentrated in its first
    example.
    """
    if not prompts:
        raise RolloutProfileUnavailable("environment yielded no prompts to sample")
    rollouts: list[SampledRollout] = []
    failed = 0
    touched: set[int] = set()
    deadline = time.monotonic() + budget_s
    # round-robin: pass k draws the k-th completion of every prompt before any prompt gets k+1.
    for pass_index in range(max(1, group_size)):
        for prompt_index, messages in enumerate(prompts):
            if len(rollouts) + failed >= target_rollouts or time.monotonic() >= deadline:
                break
            remaining = deadline - time.monotonic()
            ok, _elapsed, value = call_bounded(
                lambda m=messages: sample_one(
                    messages=m,
                    served_model=served_model,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                ),
                min(DEFAULT_ROLLOUT_TIMEOUT_S, max(0.0, remaining)),
            )
            touched.add(prompt_index)
            if ok and isinstance(value, SampledRollout):
                rollouts.append(value)
            else:
                # a timed-out draw counts as failed rather than being retried: the trust verdict
                # refuses a sample dominated by failures, and silently retrying would hide that.
                failed += 1
        if len(rollouts) + failed >= target_rollouts or time.monotonic() >= deadline:
            break
        _ = pass_index
    return RolloutSample.from_rollouts(rollouts, sampled_prompts=len(touched), failed=failed)
