"""turn client-submitted rollout evidence into a validated, priceable profile.

the client measures; the SERVER decides. a client-supplied profile is a client-supplied price, so
nothing here trusts the submitted numbers as authoritative: the config digest is re-derived from the
spec the server already holds, and the same ``trustworthy`` verdict that gates a server-produced
profile is applied unchanged. the client can only ever supply evidence that is rejected, or evidence
that survives the same checks a first-party measurement would.

every rejection path returns None, never raises. a bad or hostile profile must leave the quote on
the declared cap -- which is exactly today's pricing -- rather than fail the submit.
"""

from __future__ import annotations

import time
from typing import Any

from flash.engine.profiling.workload_profile import (
    ROLLOUT_PROFILE_KINDS,
    RolloutWorkloadProfile,
    WorkloadProfileMismatch,
    require_matching_rollout_profile,
    rollout_profile_input_digest,
)

# bounds on the submitted aggregates. these are sanity limits on a payload from an untrusted client,
# not modelling choices: anything outside them is malformed rather than merely unusual.
_MAX_ROLLOUTS = 100_000
_MAX_TOKENS = 10_000_000
_MAX_REWARD_SECONDS = 3600.0

_REQUIRED_FIELDS = (
    "sampled_prompts",
    "completed_rollouts",
    "failed_rollouts",
    "completion_tokens_mean",
    "completion_tokens_p50",
    "completion_tokens_p90",
    "completion_tokens_max",
    "prompt_tokens_mean",
    "truncated_rollouts",
    "eos_rollouts",
    "sample_policy",
)


def rollout_profile_from_evidence(
    spec: Any,
    evidence: object,
    *,
    producer_version: str,
    now: float | None = None,
) -> dict[str, Any] | None:
    """a stored-shape rollout profile for ``spec``, or None when the evidence cannot be trusted.

    ``producer_version`` and the input digest are the server's own: re-deriving them here is what
    stops a client claiming a measurement of one config prices a different one.
    """
    # the next two are preconditions, not the enforcing layer: RolloutWorkloadProfile rejects a
    # non-rollout kind and a blank environment_revision on its own. they are kept so this function is
    # safe called directly and says plainly which specs it accepts, and so a caller that reorders the
    # construction below cannot silently lose the rule.
    if not isinstance(evidence, dict):
        return None
    if str(getattr(spec, "algorithm", "")) not in ROLLOUT_PROFILE_KINDS:
        return None
    environment = getattr(spec, "environment", None)
    environment_revision = str(getattr(environment, "resolved_sha", "") or "")
    if not environment_revision:
        # an unpinned environment cannot be measured: the profile would name contents that may
        # already have changed. the sft path requires the same pin for the same reason.
        return None

    fields = _validated_fields(evidence)
    if fields is None:
        return None

    tokenizer_revision = str(getattr(spec, "model_revision", "") or "")
    digest = rollout_profile_input_digest(
        spec,
        tokenizer_revision=tokenizer_revision,
        producer_version=producer_version,
    )
    stamp = time.time() if now is None else now
    try:
        profile = RolloutWorkloadProfile(
            input_digest=digest,
            producer_version=producer_version,
            tokenizer_revision=tokenizer_revision,
            environment_id=str(getattr(environment, "id", "") or ""),
            environment_revision=environment_revision,
            kind=str(spec.algorithm),
            reference_gpu="",
            reference_provider="",
            # generation SECONDS are provider- and card-specific, so a hosted sample cannot speak
            # for the rented gpu. only the token counts transfer; this stays 0.0 and the cost model
            # derives generation time from its own throughput terms.
            generation_seconds_per_completion=0.0,
            created_at=stamp,
            measured_at=stamp,
            **fields,
        )
    except (ValueError, TypeError):
        return None

    try:
        # the same gate a server-produced profile passes: identity, then evidence. checking it here
        # means a submitted profile can never reach the quote by a weaker path than a native one.
        require_matching_rollout_profile(
            profile.to_dict(),
            input_digest=digest,
            producer_version=producer_version,
            tokenizer_revision=tokenizer_revision,
            now=stamp,
        )
    except WorkloadProfileMismatch:
        return None
    return profile.to_dict()


def _validated_fields(evidence: dict) -> dict[str, Any] | None:
    """the submitted aggregates, coerced and bounds-checked, or None if the shape is wrong."""
    if any(name not in evidence for name in _REQUIRED_FIELDS):
        return None
    try:
        fields: dict[str, Any] = {
            "sampled_prompts": int(evidence["sampled_prompts"]),
            "completed_rollouts": int(evidence["completed_rollouts"]),
            "failed_rollouts": int(evidence["failed_rollouts"]),
            "completion_tokens_mean": float(evidence["completion_tokens_mean"]),
            "completion_tokens_p50": int(evidence["completion_tokens_p50"]),
            "completion_tokens_p90": int(evidence["completion_tokens_p90"]),
            "completion_tokens_max": int(evidence["completion_tokens_max"]),
            "prompt_tokens_mean": float(evidence["prompt_tokens_mean"]),
            "truncated_rollouts": int(evidence["truncated_rollouts"]),
            "eos_rollouts": int(evidence["eos_rollouts"]),
            "sample_policy": str(evidence["sample_policy"]),
            "reward_seconds_per_completion": float(
                evidence.get("reward_seconds_per_completion", 0.0)
            ),
            "reward_samples": int(evidence.get("reward_samples", 0)),
            "reward_failures": int(evidence.get("reward_failures", 0)),
        }
    except (TypeError, ValueError):
        return None

    # only UPPER bounds are checked here. ``RolloutWorkloadProfile.__post_init__`` already rejects
    # negatives, non-finite floats and an empty policy, and duplicating those would be a second
    # spelling of one rule -- the kind that rots when only one copy is updated. It has no ceiling,
    # though: a claimed 10^9 rollouts of 10^9 tokens each is accepted there and scores trustworthy.
    counts = (
        fields["sampled_prompts"],
        fields["completed_rollouts"],
        fields["failed_rollouts"],
        fields["truncated_rollouts"],
        fields["eos_rollouts"],
        fields["reward_samples"],
        fields["reward_failures"],
    )
    if any(value > _MAX_ROLLOUTS for value in counts):
        return None
    tokens = (
        fields["completion_tokens_mean"],
        fields["completion_tokens_p50"],
        fields["completion_tokens_p90"],
        fields["completion_tokens_max"],
        fields["prompt_tokens_mean"],
    )
    if any(value > _MAX_TOKENS for value in tokens):
        return None
    if fields["reward_seconds_per_completion"] > _MAX_REWARD_SECONDS:
        return None
    if len(fields["sample_policy"]) > 128:
        return None
    # internal consistency: the parts must add up to the whole, or the payload describes a sample
    # that never happened. the dataclass only rejects parts EXCEEDING the whole; an under-count
    # (32 draws of which 0 hit eos and 0 truncated) scores trustworthy there, so this is the only
    # check standing between that payload and a price.
    if fields["truncated_rollouts"] + fields["eos_rollouts"] != fields["completed_rollouts"]:
        return None
    # a mean above the longest draw is arithmetically impossible. the dataclass orders p50/p90
    # against max but never checks the mean, and the mean is the field that sets the price.
    if fields["completion_tokens_mean"] > fields["completion_tokens_max"]:
        return None
    return fields
