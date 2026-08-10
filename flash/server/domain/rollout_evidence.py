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
    ROLLOUT_SAMPLE_POLICY_VERSION,
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

# floor on how far an UNATTESTED measurement may discount the quote, as a fraction of the completion
# cap the run is PRICED at -- recipe default included, since the field is optional and most runs omit
# it. nothing here proves sampling happened -- re-deriving the config digest binds the
# numbers to the caller's own config, not to any real generation -- so a caller could claim a 1-token
# mean against a 2048-token cap and be billed for it: measured, that payload is accepted and scores
# `(True, '')`, and `charge_completed_run` bills the quote while `precheck_training_run` admits the
# run against it.
#
# a floor is the smallest correction that keeps the feature honest. it does not make the claim
# trustworthy; it bounds what a false one is worth, from a 2048x understatement to at most this
# factor. real samples sit far above it -- the measured means this feature exists to capture run
# several-fold under the cap, not a thousand-fold -- so a genuine measurement is unaffected and only
# implausible discounts are clamped.
#
# it is a stopgap, not the answer. the real fix is a server-side or attested measurement, which needs
# the worker's own realized lengths and is a separate change.
_MIN_UNATTESTED_MEAN_FRACTION = 0.05

_REQUIRED_FIELDS = (
    "sampled_prompts",
    "offered_prompts",
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
    # which sampler produced these aggregates. required, because this server stamps the profile with
    # its OWN version: without it, a client from before a sampling-policy change could submit
    # evidence that is recorded under the newer identity and then reused as matching.
    "sample_policy_version",
)


def _discounts_below_the_unattested_floor(fields: dict[str, Any], spec: Any) -> bool:
    """whether these aggregates claim a discount too large to accept on a client's word alone.

    the mean is the field that sets the price, and nothing on this path proves a generation ever
    happened. so the claim is compared against the run's own declared cap: a mean below
    ``_MIN_UNATTESTED_MEAN_FRACTION`` of it is rejected outright rather than priced.

    rejecting, not clamping. a clamped value would be reported as a measurement while being a
    server-invented number, and every other rejection here returns the quote to the declared cap --
    the pricing an unmeasured run always had. so an implausible claim costs its author nothing but
    the discount they could not substantiate.

    the cap is read through the SAME resolution pricing uses, not off ``spec.train``. that field is
    optional, and reading it raw made the floor fail open on exactly the runs that omit it: measured,
    a 1-token mean is rejected against an explicit cap of 512 but accepted when the field is unset,
    where ``normalized()`` still resolves the recipe cap (grpo 320, opd 512, 1536 thinking) and
    prices the claim at a 320x understatement. any bound the floor compares against has to be the
    one the quote is actually computed from, or the two drift apart again.
    """
    cap = _resolved_completion_cap(spec)
    if cap <= 0:
        # no cap even after resolution -- an algorithm that does not sample completions. such a spec
        # is rejected upstream anyway (ROLLOUT_PROFILE_KINDS), so there is no quote to underquote.
        return False
    return fields["completion_tokens_mean"] < cap * _MIN_UNATTESTED_MEAN_FRACTION


def _resolved_completion_cap(spec: Any) -> int:
    """the completion cap this spec is PRICED at, recipe defaults included, or 0 if unavailable.

    deliberately routed through ``runconfig_from_spec``/``normalized`` rather than reimplementing the
    recipe lookup: a second copy of that resolution is a second thing to keep in step, and the whole
    point of this helper is that the floor and the price read one number.
    """
    try:
        from flash.cost.spec import runconfig_from_spec

        cap = runconfig_from_spec(spec).normalized().completion_len
        return int(cap or 0)
    except Exception:
        # pricing is a heavier path than the rest of this module, so it is not allowed to turn a
        # submit into an error. failing to resolve leaves the floor inert for this payload, which is
        # the same fail-open every other rejection here has: the quote stays on the declared cap.
        return 0


def evidence_is_well_formed(evidence: object) -> bool:
    """whether ``evidence`` can pass the shape, bounds and policy checks, spec aside.

    exists so a caller can reject an unusable payload BEFORE paying for work that only exists to
    build the profile -- pinning the environment ref->sha costs a blocking github call. the answer
    depends on the evidence alone, so asking early cannot change the verdict reached later.

    a cheap precondition, never the enforcing layer: ``rollout_profile_from_evidence`` re-runs this
    and everything spec-dependent, so a caller that skips it loses latency, not safety.
    """
    return isinstance(evidence, dict) and _validated_fields(evidence) is not None


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
    if _discounts_below_the_unattested_floor(fields, spec):
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
    # reject evidence this server's sampling policy did not produce. the profile is stamped with the
    # server's version, so accepting a foreign or missing policy version would let a client on an
    # older release have its aggregates recorded under the newer identity and reused as matching.
    try:
        if int(evidence["sample_policy_version"]) != ROLLOUT_SAMPLE_POLICY_VERSION:
            return None
    except (TypeError, ValueError):
        return None
    try:
        fields: dict[str, Any] = {
            "sampled_prompts": int(evidence["sampled_prompts"]),
            "offered_prompts": int(evidence["offered_prompts"]),
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
        fields["offered_prompts"],
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
