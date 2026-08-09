"""collect rollout evidence client-side, for the server to turn into a priced profile.

two halves are measured here because this is the only place both can run:

- realized generation length, drawn from a hosted inference api (see ``rollout_sampler``). the
  control plane could do this itself, but it has no reason to when the client is already loading the
  environment for the other half.
- the env's real per-completion grading latency, timed against the env's own scorer. the control
  plane can NEVER do this: it does not execute user environment code, by design.

what leaves this process is aggregates only -- token counts, latencies, and how many draws survived.
no prompts, completions, token ids or credentials. and none of it is a price: the server re-derives
the config digest itself and applies its own trust verdict, so this is evidence, never a quote.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from flash.engine.profiling.reward_profile import profile_reward_latency
from flash.engine.profiling.rollout_sampler import (
    sample_rollouts,
    sampler_credentials,
)
from flash.engine.profiling.workload_profile import MIN_TRUSTWORTHY_ROLLOUTS

# draw enough to clear the server's trust floor. sampling fewer guarantees a rejected profile, which
# would spend api calls to produce nothing.
DEFAULT_ROLLOUTS = MIN_TRUSTWORTHY_ROLLOUTS

# how many distinct prompts to spread those draws over. prompt difficulty dominates length variance,
# so a profile that repeats one prompt measures that prompt, not the workload.
DEFAULT_PROMPTS = 8


def collect_rollout_evidence(
    *,
    model: str,
    prompts: Sequence[str],
    max_completion_tokens: int,
    temperature: float | None,
    rollouts: int = DEFAULT_ROLLOUTS,
    score_one: Callable[[int, str], float] | None = None,
    reward_samples: Sequence[tuple[int, str]] | None = None,
) -> dict[str, Any] | None:
    """measure this config's rollouts, or return None when no measurement is possible.

    None rather than an exception on every failure path. the measured quote is an improvement when
    available and must never become a new way for a submit to fail: an unconfigured sampler, an
    unhosted model, or a dead endpoint all return the caller to the declared cap, which is exactly
    today's pricing.
    """
    base_url, api_key = sampler_credentials()
    if not api_key or not prompts:
        return None

    sampling = sample_rollouts(
        model=model,
        prompts=list(prompts)[:DEFAULT_PROMPTS],
        rollouts=max(1, int(rollouts)),
        max_completion_tokens=int(max_completion_tokens),
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    if not sampling.completed:
        return None

    evidence = dict(sampling.summary())
    evidence.update(_reward_evidence(score_one, reward_samples))
    return evidence


def _reward_evidence(
    score_one: Callable[[int, str], float] | None,
    reward_samples: Sequence[tuple[int, str]] | None,
) -> dict[str, Any]:
    """timed grading latency for this env, or an explicit no-measurement.

    a missing or untrustworthy reading reports ``reward_samples: 0`` rather than a latency of zero.
    the two are different claims -- "grading is free" versus "grading was not measured" -- and only
    the second may leave the cost model on its default.
    """
    if score_one is None or not reward_samples:
        return {"reward_seconds_per_completion": 0.0, "reward_samples": 0, "reward_failures": 0}
    profile = profile_reward_latency(score_one, list(reward_samples))
    if not profile.trustworthy:
        return {
            "reward_seconds_per_completion": 0.0,
            "reward_samples": 0,
            "reward_failures": profile.failures,
        }
    return {
        "reward_seconds_per_completion": float(profile.seconds_per_completion),
        "reward_samples": int(profile.samples),
        "reward_failures": int(profile.failures),
    }
