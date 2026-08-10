"""collect rollout evidence client-side, for the server to turn into a priced profile.

one quantity is measured here: realized generation length, drawn from a hosted inference api (see
``rollout_sampler``). length is a property of (model, prompt, sampling params), so it transfers from
the sampling host to the rented gpu. SECONDS do not, which is why no grading latency is timed on the
client -- the worker times its own scorer on the machine that will run it.

what leaves this process is aggregates only -- token counts and how many draws survived.
no prompts, completions, token ids or credentials. and none of it is a price: the server re-derives
the config digest itself and applies its own trust verdict, so this is evidence, never a quote.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flash.engine.profiling.rollout_sampler import (
    sample_rollouts,
    sampler_credentials,
)
from flash.engine.profiling.workload_profile import (
    MIN_TRUSTWORTHY_ROLLOUTS,
    ROLLOUT_SAMPLE_POLICY_VERSION,
)

# draw enough to clear the server's trust floor. sampling fewer guarantees a rejected profile, which
# would spend api calls to produce nothing.
DEFAULT_ROLLOUTS = MIN_TRUSTWORTHY_ROLLOUTS

# how many distinct prompts to spread those draws over. prompt difficulty dominates length variance,
# so a profile that repeats one prompt measures that prompt, not the workload.
DEFAULT_PROMPTS = 8


def collect_rollout_evidence(
    *,
    model: str,
    prompts: Sequence[Sequence[dict]],
    max_completion_tokens: int,
    temperature: float | None,
    top_p: float | None,
    rollouts: int = DEFAULT_ROLLOUTS,
    stop_sequences: Sequence[str] = (),
    thinking: bool | None = None,
    prompt_budget: int | None = None,
    population_rows: int = 0,
) -> dict[str, Any] | None:
    """measure this config's rollouts, or return None when no measurement is possible.

    None rather than an exception on every failure path. the measured quote is an improvement when
    available and must never become a new way for a submit to fail: an unconfigured sampler, an
    unhosted model, or a dead endpoint all return the caller to the declared cap, which is exactly
    today's pricing.

    ``population_rows`` is how many rows the run's training population holds, of which ``prompts``
    is the leading slice. it is what separates a shrunken sample from a complete one when the
    endpoint filters over-budget rows -- see ``_sample_survived_filtering``.
    """
    base_url, api_key = sampler_credentials()
    if not api_key or not prompts:
        return None

    offered = list(prompts)[:DEFAULT_PROMPTS]
    sampling = sample_rollouts(
        model=model,
        prompts=offered,
        rollouts=max(1, int(rollouts)),
        max_completion_tokens=int(max_completion_tokens),
        temperature=temperature,
        top_p=top_p,
        base_url=base_url,
        api_key=api_key,
        stop_sequences=stop_sequences,
        thinking=thinking,
        prompt_budget=prompt_budget,
    )
    if not sampling.completed:
        return None
    if not _sample_survived_filtering(sampling, intended=len(offered), population=population_rows):
        return None

    evidence = dict(sampling.summary())
    # name which sampler produced these aggregates. the server stamps the profile with ITS own
    # version, so without this a client from before a sampling-policy change could submit evidence
    # that gets recorded under the new identity and reused as if it matched.
    evidence["sample_policy_version"] = int(ROLLOUT_SAMPLE_POLICY_VERSION)
    return evidence


def _sample_survived_filtering(sampling, *, intended: int, population: int) -> bool:
    """whether the sample still spans the prompt mix it was meant to, after over-budget filtering.

    an over-budget row leaves BOTH sides of the server's coverage rule -- the workers drop it too,
    so it is correctly not counted as missing workload -- and that is exactly why the shrinkage is
    invisible downstream. with six of eight shuffled rows over budget, all draws come from the two
    survivors and the evidence reports sampled == offered == 2: complete coverage of a sample that
    quietly became a quarter of the intended prompt mix, while the workers train on the valid rows
    further down the population.

    declining rather than replenishing. replenishing means re-rendering rows mid-sampling, which
    puts ``prompt_messages()`` back on the wire in the middle of a paid pass, and the fallback costs
    nothing: an unmeasured run is priced at the declared cap, which is what this path always did.

    the ceiling matters more than the shrinkage. when the population itself is exhausted -- every
    row was offered and some are over budget -- there is nothing to replenish FROM, so the survivors
    are the whole measurable workload and the sample is as complete as it can be. only a sample that
    shrank while the population still held unoffered rows is declined.
    """
    offered = int(getattr(sampling, "offered_prompts", 0) or 0)
    if offered >= intended:
        # nothing was filtered out. the coverage and distinctness rules on the server cover the rest.
        return True
    return int(population) <= intended
