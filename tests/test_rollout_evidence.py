"""Client-measured rollout evidence: sampling, server validation, and the fail-open contract.

The security property under test is that a client cannot price its own run. The client measures and
sends aggregates; the SERVER re-derives this config's digest and re-applies the same trust verdict a
first-party profile passes. So every hostile payload below must be rejected, and every rejection must
leave the quote on the declared cap rather than fail the submit.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from dataclasses import replace

import pytest

import flash
from flash.core.spec import EnvironmentSpec, GpuSpec, JobSpec, TrainSpec
from flash.engine.profiling import rollout_sampler
from flash.engine.profiling.rollout_sampler import (
    RolloutSample,
    RolloutSampling,
    sample_rollouts,
)
from flash.engine.profiling.workload_profile import (
    MAX_TRUSTWORTHY_TRUNCATION_RATE,
    MIN_TRUSTWORTHY_ROLLOUTS,
    ROLLOUT_LATENCY_MAX_AGE_S,
    ROLLOUT_SAMPLE_POLICY_VERSION,
)
from flash.server.domain.rollout_evidence import rollout_profile_from_evidence

VERSION = flash.__version__


def _spec(**overrides) -> JobSpec:
    base = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "train": TrainSpec(max_steps=50, batch_size=8, group_size=4, max_completion_tokens=512),
        "environment": EnvironmentSpec(id="acme/math", resolved_sha="a" * 40),
        "gpu": GpuSpec(type="H100"),
    }
    base.update(overrides)
    return JobSpec(**base)


def _with_profile(spec: JobSpec, profile: dict) -> JobSpec:
    """Attach a profile the way the runner does, stamping the version that keyed it.

    The producer version is part of the profile's identity, so a spec carrying a profile without it
    is a state the runner never produces -- and one whose quote silently drops the measurement.
    """
    return replace(spec, workload_profile=profile, workload_profile_producer_version=VERSION)


def _evidence(**overrides) -> dict:
    base = {
        "sampled_prompts": 8,
        "offered_prompts": 8,
        "completed_rollouts": 32,
        "failed_rollouts": 0,
        "completion_tokens_mean": 53.6875,
        "completion_tokens_p50": 54,
        "completion_tokens_p90": 66,
        "completion_tokens_max": 69,
        "prompt_tokens_mean": 17.0,
        "truncated_rollouts": 0,
        "eos_rollouts": 32,
        "sample_policy": "distinct-prompts-round-robin",
        "sample_policy_version": ROLLOUT_SAMPLE_POLICY_VERSION,
        "reward_seconds_per_completion": 0.0102,
        "reward_samples": 3,
        "reward_failures": 0,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------


def test_sampling_spreads_draws_across_distinct_prompts_before_repeating_one(monkeypatch):
    """Prompt mix dominates length variance, so repeating one prompt measures that prompt.

    A sampler that drew all 32 from prompts[0] would report a confident mean for a workload it never
    saw. Asserting on the round-robin ORDER is what distinguishes the two.
    """
    seen: list[str] = []

    def _fake(*, model, messages, max_completion_tokens, temperature, base_url, api_key, **_kw):
        seen.append(messages[-1]["content"])
        return RolloutSample(prompt_tokens=10, completion_tokens=40, truncated=False)

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler._one_completion", _fake, raising=True
    )
    prompts = [[{"role": "user", "content": f"p{i}"}] for i in range(4)]
    sampling = sample_rollouts(
        model="m",
        prompts=prompts,
        rollouts=8,
        max_completion_tokens=512,
        temperature=1.0,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
    )
    assert seen == ["p0", "p1", "p2", "p3", "p0", "p1", "p2", "p3"]
    assert sampling.completed == 8
    assert sampling.sampled_prompts == 4


def test_a_failed_draw_is_counted_not_raised(monkeypatch):
    """A dead endpoint mid-sample must degrade the measurement, never fail the submit."""

    calls = {"n": 0}

    def _fake(**_kwargs):
        calls["n"] += 1
        if calls["n"] % 2:
            return None
        return RolloutSample(prompt_tokens=10, completion_tokens=40, truncated=False)

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler._one_completion", _fake, raising=True
    )
    sampling = sample_rollouts(
        model="m",
        prompts=[[{"role": "user", "content": "p"}]],
        rollouts=4,
        max_completion_tokens=512,
        temperature=None,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
    )
    # the loop collects SUCCESSES, so a half-failing endpoint takes 8 attempts to return 4 samples.
    # failures are still reported: they are what the trust gate reads to judge the sample.
    assert sampling.completed == 4
    assert sampling.failures == 4


def test_a_transient_failure_does_not_leave_the_sample_one_short_of_the_floor(monkeypatch):
    """`rollouts` defaults to the server's trust floor, so an attempt-counting loop meant a single
    transient failure returned 31 of 32 samples -- rejected as too thin, and the run priced at the
    cap even though the endpoint recovered immediately."""
    calls = {"n": 0}

    def _one_bad_draw(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            return None
        return RolloutSample(prompt_tokens=10, completion_tokens=40, truncated=False)

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler._one_completion", _one_bad_draw, raising=True
    )
    sampling = sample_rollouts(
        model="m",
        prompts=[[{"role": "user", "content": "p"}]],
        rollouts=MIN_TRUSTWORTHY_ROLLOUTS,
        max_completion_tokens=512,
        temperature=None,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
    )
    assert sampling.completed == MIN_TRUSTWORTHY_ROLLOUTS, (
        "one transient failure must not put the measurement below the trust floor"
    )
    assert sampling.failures == 1


def test_a_stalling_endpoint_stops_the_sample_instead_of_delaying_the_submit(monkeypatch):
    """The draws are serial and a user is waiting on the submit behind them.

    Without a cutoff, an endpoint that accepts connections but never answers burns the per-request
    timeout on every draw -- 32 x 180s -- for a measurement that is advisory anyway.
    """
    calls = {"n": 0}

    def _always_fails(**_kwargs):
        calls["n"] += 1

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler._one_completion", _always_fails, raising=True
    )
    sampling = sample_rollouts(
        model="m",
        prompts=[[{"role": "user", "content": "p"}]],
        rollouts=32,
        max_completion_tokens=512,
        temperature=None,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
    )
    assert calls["n"] == rollout_sampler.MAX_CONSECUTIVE_FAILURES
    assert sampling.completed == 0


def test_the_failure_cutoff_counts_consecutive_draws_not_total(monkeypatch):
    """A flaky endpoint that still answers must be sampled to completion, not abandoned."""
    calls = {"n": 0}

    def _every_other(**_kwargs):
        calls["n"] += 1
        if calls["n"] % 2:
            return None
        return RolloutSample(prompt_tokens=10, completion_tokens=40, truncated=False)

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler._one_completion", _every_other, raising=True
    )
    sampling = sample_rollouts(
        model="m",
        prompts=[[{"role": "user", "content": "p"}]],
        rollouts=20,
        max_completion_tokens=512,
        temperature=None,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
    )
    # 20 successes at a 50% failure rate costs 40 attempts, and none of the failures are
    # consecutive, so the cutoff must never fire.
    assert calls["n"] == 40, "an answering endpoint must not trip the cutoff"
    assert sampling.completed == 20


@pytest.mark.parametrize("finish", ["content_filter", "tool_calls", "", None, "unknown"])
def test_a_draw_that_did_not_end_naturally_is_not_an_eos_sample(finish, monkeypatch):
    """Only "stop" is a natural end.

    A filtered or tool-call completion stopped for a reason the training rollout will not reproduce,
    and is usually short. Counting it as EOS would let such draws pass the trust gate with zero
    reported truncations and quote a length no rollout produces.
    """
    payload = {
        "usage": {"completion_tokens": 12, "prompt_tokens": 20},
        "choices": [{"finish_reason": finish}],
    }
    assert rollout_sampler._sample_from_response(payload, max_completion_tokens=512) is None


@pytest.mark.parametrize(
    ("finish", "truncated"),
    [("stop", False), ("length", True)],
)
def test_natural_and_truncated_draws_are_both_kept(finish, truncated):
    payload = {
        "usage": {"completion_tokens": 12, "prompt_tokens": 20},
        "choices": [{"finish_reason": finish}],
    }
    sample = rollout_sampler._sample_from_response(payload, max_completion_tokens=512)
    assert sample is not None
    assert sample.truncated is truncated


def test_a_system_turn_reaches_the_sampler_as_a_system_turn():
    """The worker sends what prompt_messages() returns. Flattening a system turn into the user
    message changes chat-template tokenization and how the model answers, so the sample would
    describe a different distribution than the run being quoted."""
    from flash.cli.commands.rollout_profile import _prompts_from

    class _Env:
        def prompt_messages(self, example):
            return [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": example["q"]},
            ]

    prompts = _prompts_from(_Env(), [{"q": "2+2?"}])
    assert prompts == [
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "2+2?"}]
    ]


def test_a_prompt_this_cannot_represent_faithfully_is_dropped_not_reshaped():
    """A multimodal part-list prompt has no faithful single-string form. Declining returns the
    quote to the cap; sending a reshaped one would price a distribution that is not the run's."""
    from flash.cli.commands.rollout_profile import _prompts_from

    class _Env:
        def prompt_messages(self, example):
            return [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    assert _prompts_from(_Env(), [{"q": "x"}]) is None


def test_a_mixed_dataset_declines_rather_than_measuring_only_its_plain_rows():
    """Skipping the rows this cannot represent leaves the plain ones as the WHOLE offered set.

    Coverage is then reported over the survivors, so a run whose dataset is half tool-calling
    prompts reports 100% coverage and passes the trust gate on the plain half alone -- while
    training still renders the omitted originals. Declining prices at the cap instead.
    """
    from flash.cli.commands.rollout_profile import _prompts_from

    class _Mixed:
        def prompt_messages(self, example):
            if example["kind"] == "tool":
                # carries `name`, which the worker passes to apply_chat_template and this cannot.
                return [{"role": "user", "content": "call it", "name": "tool-call"}]
            return [{"role": "user", "content": example["q"]}]

    rows = [{"kind": "tool", "q": "t"}] * 4 + [{"kind": "plain", "q": "2+2?"}] * 4
    assert _prompts_from(_Mixed(), rows) is None


def test_only_the_rows_training_consumes_are_measured():
    """grpo and opd both fence the population to a max_examples prefix. Measuring outside that
    fence samples rows the run never sees, and with a small max_examples those rows could set the
    whole measured length."""
    from flash.cli.commands.rollout_profile import _training_population

    spec = _spec(train=TrainSpec(max_steps=50, batch_size=8, group_size=4, max_examples=2))
    # the FENCE is a prefix; the order within it is the worker's shuffle, so compare as a set.
    assert sorted(_training_population(spec, [1, 2, 3, 4, 5])) == [1, 2]
    unfenced = _spec()
    assert sorted(_training_population(unfenced, [1, 2, 3, 4, 5])) == [1, 2, 3, 4, 5]


def test_the_measured_rows_are_the_ones_the_worker_shuffles_to():
    """Only a prefix of the population is ever measured, and both workers SHUFFLE before consuming
    it (`train = train[:max_examples]` then `random.Random(SEED).shuffle(train)`).

    Measuring the raw prefix instead prices whatever the dataset happens to start with. On a
    dataset ordered by length or class that is systematically unrepresentative -- and it is exactly
    the ordered datasets where the first rows differ most from the rest.
    """
    import random as _random

    from flash.cli.commands.rollout_profile import _PROMPT_ROWS, _training_population

    rows = [{"i": i} for i in range(64)]
    spec = _spec(seed=1234)

    measured = [row["i"] for row in _training_population(spec, rows)[:_PROMPT_ROWS]]

    worker_order = list(rows)
    _random.Random(1234).shuffle(worker_order)
    assert measured == [row["i"] for row in worker_order[:_PROMPT_ROWS]]
    # and it is not simply the prefix, which is what made this wrong.
    assert measured != list(range(_PROMPT_ROWS))
    # the caller's dataset must not be reordered as a side effect of asking for a quote.
    assert [row["i"] for row in rows] == list(range(64))


def test_summary_reports_the_distribution_not_a_mean_alone():
    """A mean cannot tell a uniformly short workload from a bimodal one whose tail sets step time."""
    samples = tuple(
        RolloutSample(prompt_tokens=10, completion_tokens=n, truncated=False)
        for n in [10] * 9 + [500]
    )
    summary = RolloutSampling(samples=samples, failures=0, sampled_prompts=2).summary()
    assert summary["completion_tokens_p50"] == 10
    assert summary["completion_tokens_max"] == 500
    # the mean alone (59) would hide that one draw in ten is 50x the median.
    assert summary["completion_tokens_mean"] == pytest.approx(59.0)


def test_summary_carries_no_prompt_or_completion_text():
    """Aggregates only. Prompts and completions must never leave the client process."""
    samples = (RolloutSample(prompt_tokens=10, completion_tokens=40, truncated=False),)
    summary = RolloutSampling(
        samples=samples, failures=0, sampled_prompts=1, offered_prompts=1
    ).summary()
    assert set(summary) == {
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
    }


# --------------------------------------------------------------------------------------
# server validation: a client may not price its own run
# --------------------------------------------------------------------------------------


def test_valid_evidence_becomes_a_profile_carrying_the_servers_own_digest():
    spec = _spec()
    profile = rollout_profile_from_evidence(spec, _evidence(), producer_version=VERSION)
    assert profile is not None

    from flash.engine.profiling.workload_profile import rollout_profile_input_digest

    expected = rollout_profile_input_digest(
        spec, tokenizer_revision=spec.model_revision, producer_version=VERSION
    )
    assert profile["input_digest"] == expected


def test_a_client_supplied_digest_cannot_override_the_servers():
    """The digest binds a measurement to one config. If the client could choose it, evidence from a
    32-token config would price a 4096-token one."""
    profile = rollout_profile_from_evidence(
        _spec(), _evidence(input_digest="f" * 64), producer_version=VERSION
    )
    assert profile is not None
    assert profile["input_digest"] != "f" * 64


def test_evidence_measured_at_a_different_cap_does_not_price_this_run():
    """Same evidence, two configs: the digest must let it move only the one it was taken for."""
    measured = _spec()
    profile = rollout_profile_from_evidence(measured, _evidence(), producer_version=VERSION)
    assert profile is not None

    other = replace(measured, train=replace(measured.train, max_completion_tokens=4096))
    from flash.cost.spec import runconfig_from_spec

    priced_other = replace(other, workload_profile=profile)
    assert runconfig_from_spec(priced_other).measured_completion_tokens is None


def test_a_thin_sample_is_rejected():
    """Below the trust floor the mean is not resolvable; a confident number would be fiction."""
    thin = MIN_TRUSTWORTHY_ROLLOUTS - 1
    assert (
        rollout_profile_from_evidence(
            _spec(),
            _evidence(completed_rollouts=thin, eos_rollouts=thin),
            producer_version=VERSION,
        )
        is None
    )


def test_a_heavily_truncated_sample_is_rejected():
    """A truncated draw contributes the CAP, not its true length, biasing the mean DOWN -- the
    direction that underbills. A censored sample must not read as a measurement."""
    truncated = int(32 * MAX_TRUSTWORTHY_TRUNCATION_RATE) + 2
    assert (
        rollout_profile_from_evidence(
            _spec(),
            _evidence(truncated_rollouts=truncated, eos_rollouts=32 - truncated),
            producer_version=VERSION,
        )
        is None
    )


def test_a_measurement_goes_stale_at_the_quote_and_returns_it_to_the_cap():
    """The freshness gate bites when a stored profile is re-read, not when it is created.

    The server stamps ``measured_at`` from its own clock at creation, so a profile is fresh the
    instant it is made -- the submitted evidence carries no timestamp the client could backdate. What
    the gate protects is the profile sitting on the spec afterwards: a model, provider, or
    environment drifts, and yesterday's sample stops describing today's run.
    """
    from flash.cost.spec import _rollout_profile

    spec = _spec()
    stamped = time.time() - ROLLOUT_LATENCY_MAX_AGE_S - 60
    profile = rollout_profile_from_evidence(
        spec, _evidence(), producer_version=VERSION, now=stamped
    )
    assert profile is not None, "a profile is always fresh at the moment it is stamped"

    # re-read now, a day later than the stamp: the same profile no longer speaks for this run.
    assert _rollout_profile(replace(spec, workload_profile=profile)) is None


@pytest.mark.parametrize(
    "bad",
    [
        # parts exceeding the whole. the dataclass rejects this one on its own.
        {"truncated_rollouts": 5},
        # parts summing to LESS than the whole. the dataclass reports this trustworthy, so the
        # check in _validated_fields is the only thing standing between it and a price: 32 draws of
        # which 0 hit EOS and 0 truncated describes a sample that never happened.
        {"truncated_rollouts": 0, "eos_rollouts": 0},
        {"truncated_rollouts": 4, "eos_rollouts": 4},
        # a mean above the longest draw is arithmetically impossible, and the mean is the field
        # that sets the price. the dataclass orders p50/p90 against max but never the mean.
        {"completion_tokens_mean": 5_000.0},
        {"completion_tokens_p50": 10_000},  # p50 above max
        {"completion_tokens_p90": 10_000},  # p90 above max
        # upper bounds. the dataclass has no ceiling: it scores 10^9 rollouts as trustworthy.
        {"completed_rollouts": 10**9, "eos_rollouts": 10**9},
        {"completion_tokens_mean": 10**9, "completion_tokens_max": 10**9},
        {"reward_seconds_per_completion": 10_000.0},
        {"sample_policy": "x" * 200},
        # lower bounds and non-finites, enforced downstream by the dataclass. asserted here because
        # this function is the boundary an untrusted payload crosses, wherever the rule lives.
        {"completed_rollouts": -1},
        {"completion_tokens_mean": -5.0},
        {"reward_seconds_per_completion": -1.0},
        {"sample_policy": ""},
        {"completion_tokens_mean": float("nan")},
        {"completion_tokens_mean": float("inf")},
    ],
)
def test_internally_inconsistent_or_out_of_bounds_evidence_is_rejected(bad):
    """Each of these describes a sample that could not have happened. Accepting one would let a
    crafted payload set the price."""
    assert (
        rollout_profile_from_evidence(_spec(), _evidence(**bad), producer_version=VERSION) is None
    )


@pytest.mark.parametrize(
    "missing", ["completed_rollouts", "completion_tokens_mean", "sample_policy"]
)
def test_evidence_missing_a_required_field_is_rejected(missing):
    payload = _evidence()
    payload.pop(missing)
    assert rollout_profile_from_evidence(_spec(), payload, producer_version=VERSION) is None


@pytest.mark.parametrize("evidence", [None, "not-a-dict", 42, []])
def test_non_object_evidence_is_rejected(evidence):
    assert rollout_profile_from_evidence(_spec(), evidence, producer_version=VERSION) is None


def test_sft_evidence_is_rejected():
    """SFT is priced from an exact census, not a sample. Rollout evidence must not reach it."""
    spec = _spec(algorithm="sft", train=TrainSpec(max_steps=50, batch_size=8))
    assert rollout_profile_from_evidence(spec, _evidence(), producer_version=VERSION) is None


def test_an_unpinned_environment_cannot_be_measured():
    """Without a resolved sha the profile would name contents that may already have changed."""
    spec = _spec(environment=EnvironmentSpec(id="acme/math", resolved_sha=""))
    assert rollout_profile_from_evidence(spec, _evidence(), producer_version=VERSION) is None


def test_generation_seconds_are_never_taken_from_a_hosted_sample():
    """Token COUNTS transfer off-card; SECONDS do not. A hosted provider's latency says nothing
    about the rented GPU, so the cost model must keep deriving generation time itself."""
    profile = rollout_profile_from_evidence(_spec(), _evidence(), producer_version=VERSION)
    assert profile is not None
    assert profile["generation_seconds_per_completion"] == 0.0


# --------------------------------------------------------------------------------------
# the quote
# --------------------------------------------------------------------------------------


def test_measured_length_lowers_a_grpo_quote_below_the_cap():
    from flash.cost.spec import runconfig_from_spec

    spec = _spec()
    profile = rollout_profile_from_evidence(spec, _evidence(), producer_version=VERSION)
    assert profile is not None

    capped = runconfig_from_spec(spec)
    measured = runconfig_from_spec(_with_profile(spec, profile))
    assert capped.measured_completion_tokens is None
    assert measured.measured_completion_tokens == pytest.approx(53.6875)
    # 54 realized tokens against a 512 cap: the cap-based quote prices ~10x the generation.
    assert measured.measured_prompt_tokens == pytest.approx(17.0)


def test_a_measured_reward_latency_reaches_the_cost_model():
    from flash.cost.spec import runconfig_from_spec

    spec = _spec()
    profile = rollout_profile_from_evidence(spec, _evidence(), producer_version=VERSION)
    measured = runconfig_from_spec(_with_profile(spec, profile))
    assert measured.reward_seconds_per_completion == pytest.approx(0.0102)


def test_an_unmeasured_reward_leaves_the_cost_model_on_its_default():
    """`reward_samples: 0` means "not measured", not "grading is free". Only the first may leave the
    default in place; reporting 0.0s as a measurement would assert the second."""
    from flash.cost.spec import runconfig_from_spec

    spec = _spec()
    profile = rollout_profile_from_evidence(
        spec,
        _evidence(reward_seconds_per_completion=0.0, reward_samples=0),
        producer_version=VERSION,
    )
    assert profile is not None
    measured = runconfig_from_spec(_with_profile(spec, profile))
    assert measured.reward_seconds_per_completion is None


# --------------------------------------------------------------------------------------
# fail open
# --------------------------------------------------------------------------------------


def test_the_runner_attaches_a_valid_measurement():
    from flash.runner import _attach_rollout_workload_profile

    spec = _spec()
    attached = _attach_rollout_workload_profile(spec, _evidence())
    assert attached.workload_profile


@pytest.mark.parametrize(
    "evidence",
    [None, {}, {"completed_rollouts": 1}, "garbage"],
)
def test_the_runner_fails_open_on_unusable_evidence(evidence):
    """A rollout profile is a SAMPLE, and one cannot be taken for every model. Refusing here would
    make the unhostable models unquotable to buy accuracy on the others -- so a bad measurement must
    return the quote to the declared cap, not fail the submit."""
    from flash.runner import _attach_rollout_workload_profile

    spec = _spec()
    attached = _attach_rollout_workload_profile(spec, evidence)
    assert attached.workload_profile == {}
    assert attached is spec or attached == spec


def test_the_runner_ignores_rollout_evidence_on_an_sft_spec():
    from flash.runner import _attach_rollout_workload_profile

    spec = _spec(algorithm="sft", train=TrainSpec(max_steps=50, batch_size=8))
    assert _attach_rollout_workload_profile(spec, _evidence()).workload_profile == {}


def test_a_measurement_survives_the_wire_round_trip_a_real_submit_makes(monkeypatch):
    """The spec the server validates is one the PUBLIC serializer produced, not an in-process one.

    to_dict() strips resolved_sha as a platform-managed key, so a spec that arrived over the wire
    always carries "" there -- while rollout_profile_from_evidence refuses an unpinned environment.
    Attaching before the runner pins the ref therefore rejected every real managed submit and
    silently cap-priced it. Building the spec in-process hides this, because it never loses the pin.
    """
    from flash.core.spec import JobSpec
    from flash.runner import _attach_rollout_workload_profile

    wire = JobSpec.from_dict(_spec().to_dict())
    assert wire.environment.resolved_sha == "", "to_dict must strip the platform-managed pin"

    monkeypatch.setattr(
        # _assign_resolved_env_sha imports this in-body, so the defining module is the binding
        # a patch has to reach; patching flash.runner.artifacts silently hits the real api.
        "flash.envs.loader._resolve_ref_sha",
        lambda *a, **k: "c" * 40,
    )
    attached = _attach_rollout_workload_profile(wire, _evidence())
    assert attached.workload_profile, "a measurement must reach the quote on the real submit path"


def test_an_unreachable_pin_returns_the_quote_to_the_cap_instead_of_failing(monkeypatch):
    """Pinning is best-effort: GitHub being unreachable must land on fail-open, never raise."""
    from flash.core.spec import JobSpec
    from flash.runner import _attach_rollout_workload_profile

    wire = JobSpec.from_dict(_spec().to_dict())

    def _boom(*a, **k):
        raise RuntimeError("github unreachable")

    monkeypatch.setattr("flash.envs.loader._resolve_ref_sha", _boom)
    attached = _attach_rollout_workload_profile(wire, _evidence())
    assert attached.workload_profile == {}


def test_a_measured_quote_survives_a_package_version_bump(monkeypatch):
    """The re-quote at allocation must not silently drop the measurement after a bump.

    The producer version keys the profile identity. Re-deriving it from the live
    ``flash.__version__`` makes the answer depend on which process does the arithmetic: a run that
    submitted at a measured price gets re-quoted at the cap once the package moves under it. The
    sft path stamps the version on the spec for exactly this reason; this pins the rollout path to
    the same contract.
    """
    from flash.core.spec import JobSpec
    from flash.cost.spec import _rollout_profile
    from flash.runner import _attach_rollout_workload_profile

    wire = JobSpec.from_dict(_spec().to_dict())
    monkeypatch.setattr("flash.envs.loader._resolve_ref_sha", lambda *a, **k: "c" * 40)
    attached = _attach_rollout_workload_profile(wire, _evidence())
    assert attached.workload_profile, "precondition: the measurement must have attached"
    assert attached.workload_profile_producer_version, "the keying version must travel on the spec"
    assert _rollout_profile(attached) is not None, "precondition: it re-quotes at the same version"

    monkeypatch.setattr(flash, "__version__", "9.9.99")
    assert _rollout_profile(attached) is not None, (
        "a package bump between submit and allocation must not re-quote the run at the cap"
    )


def test_configured_stop_sequences_reach_the_sampled_request(monkeypatch):
    """A run with stop strings ends generation at the first one.

    Sampling without them lets the hosted model run on toward the cap, so the measured mean is
    biased HIGH -- the direction that overbills the user this feature exists to price correctly.
    """
    seen: dict = {}

    def _capture(*, model, messages, max_completion_tokens, temperature, base_url, api_key, **kw):
        seen.update(kw)
        return RolloutSample(prompt_tokens=10, completion_tokens=40, truncated=False)

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler._one_completion", _capture, raising=True
    )
    sample_rollouts(
        model="m",
        prompts=[[{"role": "user", "content": "p"}]],
        rollouts=1,
        max_completion_tokens=512,
        temperature=None,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
        stop_sequences=("</answer>",),
    )
    assert seen.get("stop_sequences") == ("</answer>",)


def test_stop_sequences_are_sent_as_the_apis_stop_field(monkeypatch):
    """The wire contract: an OpenAI-compatible endpoint honours `stop`, so it must be in the body."""
    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_open(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return _Response()

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler.json.load",
        lambda _r: {
            "usage": {"completion_tokens": 12, "prompt_tokens": 20},
            "choices": [{"finish_reason": "stop"}],
        },
    )
    # the sampler goes through a redirect-refusing opener, not the module-level urlopen, so the
    # opener IS the seam. patching urlopen here would leave the real call path untouched.
    monkeypatch.setattr(rollout_sampler._NO_REDIRECT_OPENER, "open", _fake_open, raising=True)
    rollout_sampler._one_completion(
        model="m",
        messages=[{"role": "user", "content": "p"}],
        max_completion_tokens=64,
        temperature=None,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
        stop_sequences=("</answer>", ""),
    )
    # the empty string is dropped: an empty stop string would end every generation immediately.
    assert captured["body"]["stop"] == ["</answer>"]


def test_no_stop_field_is_sent_when_the_run_configures_none(monkeypatch):
    """An unconditional `stop: []` is not the same request; leave the key off entirely."""
    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_open(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return _Response()

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler.json.load",
        lambda _r: {
            "usage": {"completion_tokens": 12, "prompt_tokens": 20},
            "choices": [{"finish_reason": "stop"}],
        },
    )
    # the sampler goes through a redirect-refusing opener, not the module-level urlopen, so the
    # opener IS the seam. patching urlopen here would leave the real call path untouched.
    monkeypatch.setattr(rollout_sampler._NO_REDIRECT_OPENER, "open", _fake_open, raising=True)
    rollout_sampler._one_completion(
        model="m",
        messages=[{"role": "user", "content": "p"}],
        max_completion_tokens=64,
        temperature=None,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
    )
    assert "stop" not in captured["body"]


def test_the_dry_run_never_claims_a_paid_scorer_was_called():
    """reward() is no longer run on the client for any algorithm, so the notice must not imply it.

    Telling a user a paid external scorer was called when it was not is the one claim in this line
    they could act on wrongly -- and it also tells them grading cost is absent from the quote.
    """
    from flash.cli.commands import _dry_run_preview_line

    line = _dry_run_preview_line(
        algorithm="grpo",
        affordability_verified=True,
        rollout_evidence=_evidence(),
        environment_stages_run=("import", "dataset", "prompt_messages"),
    )
    assert "reward() was NOT called" in line
    assert "external scorer was really called" not in line


def test_the_dry_run_says_nothing_ran_when_profiling_never_started():
    """The genuinely-untouched path: no sampler key, or a config profiling declines before loading.

    Stated explicitly rather than relying on the parameter's default, so this keeps asserting about
    the case where the user's module really was never imported.
    """
    from flash.cli.commands import _dry_run_preview_line

    line = _dry_run_preview_line(
        algorithm="grpo",
        affordability_verified=True,
        rollout_evidence=None,
        environment_stages_run=(),
    )
    assert "did NOT import or run your environment.py" in line


def test_the_submit_path_forwards_the_runs_stop_sequences_to_the_measurement(monkeypatch, tmp_path):
    """The seam between the spec and the sampler.

    Testing the sampler alone leaves this uncovered: dropping the argument at the call site is
    silent, and the measured mean is then biased HIGH for every run that configures stop strings.
    """
    from flash.cli.commands import rollout_profile as rp

    entry = tmp_path / "package" / "environment.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("")

    class _Env:
        def dataset(self):
            return [{"q": "2+2?"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["q"]}]

    class _Client:
        def download_env_package(self, _env_id):
            return b""

    seen: dict = {}
    # collect_for_submit imports each of these in-body, so the DEFINING module is the binding a
    # patch has to reach -- patching an alias on `rp` would leave the real call untouched.
    monkeypatch.setattr(
        "flash.envs.pull.pull_environment_package_from_archive",
        lambda *_a: tmp_path / "package",
    )
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda *_a, **_k: _Env())
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_evidence.collect_rollout_evidence",
        lambda **kw: seen.update(kw),
    )

    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    spec = _spec(
        train=TrainSpec(max_steps=50, batch_size=8, group_size=4, stop_sequences=("</answer>",))
    )
    # debug=True so a broken fixture raises instead of being swallowed by the fail-open guard and
    # read as "the argument was not forwarded".
    rp.collect_for_submit(_Client(), spec, debug=True)
    assert seen.get("stop_sequences") == ("</answer>",), (
        "the run's stop strings must reach the sampler, or the measured length is biased high"
    )


def test_stop_sequences_are_part_of_the_measurements_identity():
    """The digest binds a measurement to the config it describes.

    Stop strings shorten generation, so a sample drawn WITH them is shorter than the same run
    without them would produce. If they are not keyed, those two configs share an identity and the
    stop-shortened sample can be reused to price the unstopped run -- under-quoting it. This is the
    same reason temperature is keyed, and the control below pins that reasoning.
    """
    from flash.engine.profiling.workload_profile import rollout_profile_input_digest

    def _digest(**train_kwargs):
        spec = _spec(
            train=TrainSpec(
                max_steps=50,
                batch_size=8,
                group_size=4,
                max_completion_tokens=512,
                **train_kwargs,
            )
        )
        return rollout_profile_input_digest(
            spec, tokenizer_revision="main", producer_version=VERSION
        )

    unstopped = _digest()
    assert _digest() == unstopped, "an unchanged config must key to a stable identity"
    assert _digest(stop_sequences=("</answer>",)) != unstopped, (
        "a stop-shortened sample must not be reusable for a run without stops"
    )
    # different stop strings stop at different points, so they are different distributions too.
    assert _digest(stop_sequences=("</answer>",)) != _digest(stop_sequences=("</s>",))


@pytest.mark.parametrize(
    ("field", "value"),
    [("init_from_adapter", "acme/run-1"), ("structured_outputs", '{"type":"object"}')],
)
def test_a_config_the_sampler_cannot_reproduce_is_not_measured(field, value, monkeypatch):
    """A hosted draw is one unconstrained completion from the BASE model.

    A warm-start adapter changes the weights that generate, and structured outputs impose a decoding
    grammar; both move the length distribution, and the hosted request carries neither. Measuring
    anyway would put a confident number describing different work into a persisted quote, so these
    fall back to the cap instead.
    """
    from flash.cli.commands import rollout_profile as rp

    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")

    def _must_not_run(*_a, **_k):
        raise AssertionError("an unsamplable config must not reach the sampler")

    monkeypatch.setattr(rp, "_collect", _must_not_run)
    spec = _spec(train=TrainSpec(max_steps=50, batch_size=8, group_size=4, **{field: value}))
    assert rp.collect_for_submit(object(), spec, debug=True) is None


def test_a_multi_turn_environment_is_not_priced_from_one_turn(monkeypatch, tmp_path):
    """The worker generates an assistant turn per step_episode and trains on the transcript.

    One hosted completion is the first turn only, so its mean undercounts an episode -- and would
    still clear the trust gate, replacing the cap with a number that is too low.
    """
    from flash.cli.commands import rollout_profile as rp

    entry = tmp_path / "package" / "environment.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("")

    class _MultiTurn:
        multi_turn = True

        def dataset(self):
            raise AssertionError("a multi-turn env must be declined before its dataset is read")

    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    monkeypatch.setattr(
        "flash.envs.pull.pull_environment_package_from_archive",
        lambda *_a: tmp_path / "package",
    )
    monkeypatch.setattr(
        "flash.envs.loader.load_freesolo_environment", lambda *_a, **_k: _MultiTurn()
    )

    class _Client:
        def download_env_package(self, _env_id):
            return b""

    assert rp.collect_for_submit(_Client(), _spec(), debug=True) is None


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_no_algorithm_times_the_task_reward_on_the_client_host(algorithm, monkeypatch, tmp_path):
    """Seconds measured on this machine do not describe the worker.

    reward() is either cpu-bound (a different cpu) or an external judge call (a different network
    path and geography), and the figure would then be multiplied by every completion of every step.
    This is the same rule generation already follows: token COUNTS transfer between hosts, SECONDS
    do not. The worker times its own scorer on the machine that will run it. Length sampling must
    still happen for both algorithms -- both are quoted on completion tokens.
    """
    from flash.cli.commands import rollout_profile as rp

    entry = tmp_path / "package" / "environment.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("")

    class _Env:
        def dataset(self):
            return [{"q": "2+2?"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["q"]}]

        def reward(self, *_a, **_k):
            raise AssertionError("profiling must not call the task reward on this host")

        def sft_completion(self, _example):
            return "4"

    seen: dict = {}
    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    monkeypatch.setattr(
        "flash.envs.pull.pull_environment_package_from_archive",
        lambda *_a: tmp_path / "package",
    )
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda *_a, **_k: _Env())
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_evidence.collect_rollout_evidence",
        lambda **kw: seen.update(kw),
    )

    class _Client:
        def download_env_package(self, _env_id):
            return b""

    rp.collect_for_submit(_Client(), _spec(algorithm=algorithm), debug=True)
    assert seen.get("prompts"), "completion length must still be sampled"
    assert seen.get("score_one") is None, "no scorer may be handed to a client-side timer"
    assert seen.get("reward_samples") is None


@pytest.mark.parametrize(("field", "value"), [("model_revision", "abc123")])
def test_a_run_whose_generation_the_endpoint_cannot_match_is_not_measured(
    field, value, monkeypatch
):
    """Keying a field in the digest stops a measurement being REUSED for another config.

    It cannot make this draw faithful: a pinned revision is served by the worker but not by the
    hosted model id, so the draw comes from different weights.

    Reasoning is NOT in this list. It is handled by stating the run's setting in the request and
    dropping any response that reasons anyway -- declining outright would disable measurement for
    the entire catalog, since every model in it is hybrid-thinking.
    """
    from flash.cli.commands import rollout_profile as rp

    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")

    def _must_not_run(*_a, **_k):
        raise AssertionError("an unsamplable config must not reach the sampler")

    monkeypatch.setattr(rp, "_collect", _must_not_run)
    assert rp.collect_for_submit(object(), _spec(**{field: value}), debug=True) is None


def _profiling_stubs(monkeypatch, tmp_path, env):
    """Wire collect_for_submit's in-body imports to a fake env and capture what it would measure."""
    entry = tmp_path / "package" / "environment.py"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("")

    seen: dict = {}

    class _Client:
        def download_env_package(self, _env_id):
            return b""

    monkeypatch.setattr(
        "flash.envs.pull.pull_environment_package_from_archive", lambda *_a: tmp_path / "package"
    )
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda *_a, **_k: env)
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_evidence.collect_rollout_evidence",
        lambda **kw: seen.update(kw) or {"measured": True},
    )
    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    return _Client(), seen


def test_the_evidence_carries_no_grading_measurement(monkeypatch):
    """Grading is no longer timed on the client, so no reward field may appear in the payload.

    A stray zero would be read by the cost model as "grading is free" rather than "grading was not
    measured here" -- the same conflation the reward-default work removed.
    """
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_evidence.sampler_credentials",
        lambda: ("https://example.invalid/v1", "k"),
    )
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_evidence.sample_rollouts",
        lambda **_kw: RolloutSampling(
            samples=tuple(
                RolloutSample(prompt_tokens=10, completion_tokens=40, truncated=False)
                for _ in range(MIN_TRUSTWORTHY_ROLLOUTS)
            ),
            failures=0,
            sampled_prompts=8,
        ),
    )
    from flash.engine.profiling.rollout_evidence import collect_rollout_evidence

    evidence = collect_rollout_evidence(
        model="m",
        prompts=[[{"role": "user", "content": "p"}]],
        max_completion_tokens=512,
        temperature=None,
        top_p=1.0,
    )
    assert not [key for key in evidence if key.startswith("reward")]
    # and the token half still stands on its own.
    assert evidence["completed_rollouts"] == MIN_TRUSTWORTHY_ROLLOUTS


@pytest.mark.parametrize(
    "row",
    [
        {"q": "what is in this picture?", "image": "data:image/png;base64,AAAA"},
        {"q": "what is in these?", "images": ["a.png", "b.png"]},
    ],
)
def test_a_multimodal_row_is_not_priced_from_its_text_alone(row, monkeypatch, tmp_path):
    """The workers add image blocks and the processor's expanded image tokens before generating.

    A text-only draw of the same row measures a fraction of the real prompt work. Skipping such rows
    would be worse than declining: a mixed dataset would still return a full set of text-only draws,
    pass the trust gate, and underprice every multimodal step.
    """
    from flash.cli.commands import rollout_profile as rp

    class _Env:
        multi_turn = False

        def dataset(self):
            return [row, {"q": "plain text row"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["q"]}]

    client, seen = _profiling_stubs(monkeypatch, tmp_path, _Env())
    assert rp.collect_for_submit(client, _spec(), debug=True) is None
    assert not seen, "a multimodal dataset must not reach the sampler at all"


def test_a_text_only_dataset_is_still_measured(monkeypatch, tmp_path):
    """The control: the multimodal check must not decline ordinary text runs."""
    from flash.cli.commands import rollout_profile as rp

    class _Env:
        multi_turn = False

        def dataset(self):
            return [{"q": "2+2?"}, {"q": "3+3?"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["q"]}]

    client, seen = _profiling_stubs(monkeypatch, tmp_path, _Env())
    assert rp.collect_for_submit(client, _spec(), debug=True) == {"measured": True}
    assert len(seen["prompts"]) == 2


def test_the_measurement_samples_the_rows_the_run_will_train_on(monkeypatch, tmp_path):
    """The workers seed before their first environment call; this path did not.

    Env code may consume python/numpy randomness while building its dataset, so unseeded the measured
    rows depend on this process's incidental RNG state -- while the server binds the profile to
    spec.seed. Two runs with different seeds could then share a measurement of a different sample.
    """
    from flash.cli.commands import rollout_profile as rp

    class _Env:
        multi_turn = False

        def dataset(self):
            # a dataset built with randomness, exactly what the worker's seeding exists to pin.
            return [{"q": f"row-{random.random():.12f}"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["q"]}]

    def _measured_row(spec):
        client, seen = _profiling_stubs(monkeypatch, tmp_path, _Env())
        rp.collect_for_submit(client, spec, debug=True)
        return seen["prompts"][0][0]["content"]

    # the same seed must select the same row every time, whatever this process did beforehand.
    random.seed(999)
    first = _measured_row(_spec(seed=7))
    random.seed(12345)
    second = _measured_row(_spec(seed=7))
    assert first == second, "the measured rows must follow spec.seed, not ambient RNG state"

    # and a different seed is a different sample, which is why the digest keys it.
    assert _measured_row(_spec(seed=8)) != first


def test_measuring_does_not_leave_the_process_rng_seeded(monkeypatch, tmp_path):
    """This CLI keeps running after the measurement -- it goes on to submit the run.

    Leaving the seed in place would make every later random draw in the process deterministic as a
    side effect of asking for a quote.
    """
    from flash.cli.commands import rollout_profile as rp

    class _Env:
        multi_turn = False

        def dataset(self):
            return [{"q": "2+2?"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["q"]}]

    client, _seen = _profiling_stubs(monkeypatch, tmp_path, _Env())
    random.seed(4242)
    expected = [random.random() for _ in range(3)]

    random.seed(4242)
    rp.collect_for_submit(client, _spec(seed=7), debug=True)
    assert [random.random() for _ in range(3)] == expected, (
        "the caller's RNG stream must survive the measurement"
    )


def test_the_api_key_is_not_disclosed_to_a_redirect_target():
    """urllib copies request headers onto a redirected request, Authorization included.

    So a 30x from the configured endpoint hands the user's PAID inference key to whatever host the
    Location names. The same hop also rewrites POST to GET and its response was still parsed as a
    sample, letting a foreign host dictate the measurement it was paid to leak.

    Two real local servers rather than a mocked opener: the leak lives in urllib's redirect handler,
    so a test that stubs the transport asserts on its own stub.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: dict = {}

    class _Attacker(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["authorization"] = self.headers.get("Authorization")
            body = json.dumps(
                {
                    "usage": {"completion_tokens": 10, "prompt_tokens": 5},
                    "choices": [{"finish_reason": "stop"}],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # a 302 turns the POST into a GET, so the credential arrives on the GET.
        do_GET = do_POST

        def log_message(self, *_args):
            pass

    class _Redirector(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{attacker_port}/steal")
            self.end_headers()

        def log_message(self, *_args):
            pass

    attacker = HTTPServer(("127.0.0.1", 0), _Attacker)
    attacker_port = attacker.server_address[1]
    redirector = HTTPServer(("127.0.0.1", 0), _Redirector)
    threading.Thread(target=attacker.serve_forever, daemon=True).start()
    threading.Thread(target=redirector.serve_forever, daemon=True).start()
    try:
        sample = rollout_sampler._one_completion(
            model="m",
            messages=[{"role": "user", "content": "p"}],
            max_completion_tokens=64,
            temperature=None,
            top_p=1.0,
            base_url=f"http://127.0.0.1:{redirector.server_address[1]}/v1",
            api_key="sk-USER-PAID-INFERENCE-KEY",
        )
    finally:
        attacker.shutdown()
        redirector.shutdown()

    assert seen.get("authorization") is None, "the api key must never reach another origin"
    assert sample is None, "a redirected response must not stand as a measurement"


def test_evidence_from_a_different_sampling_policy_is_rejected():
    """The server stamps the profile with ITS version, not the client's.

    So without a policy version in the payload, a client from before a sampling-policy change could
    submit aggregates that are recorded under the newer identity and later reused as matching.
    """
    spec = _spec()
    assert rollout_profile_from_evidence(spec, _evidence(), producer_version=VERSION) is not None

    stale = _evidence(sample_policy_version=ROLLOUT_SAMPLE_POLICY_VERSION + 1)
    assert rollout_profile_from_evidence(spec, stale, producer_version=VERSION) is None

    missing = _evidence()
    del missing["sample_policy_version"]
    assert rollout_profile_from_evidence(spec, missing, producer_version=VERSION) is None

    for junk in ("one", None, [1]):
        payload = _evidence(sample_policy_version=junk)
        assert rollout_profile_from_evidence(spec, payload, producer_version=VERSION) is None


def test_the_sampler_stamps_the_policy_version_it_produced(monkeypatch):
    """The seam: the server requires the field, so the client has to emit it or measure nothing."""
    # rollout_evidence imports both names at module scope, so ITS module is the binding a patch has
    # to reach -- patching the defining module would leave the real calls untouched.
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_evidence.sampler_credentials",
        lambda: ("https://example.invalid/v1", "k"),
    )
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_evidence.sample_rollouts",
        lambda **_kw: RolloutSampling(
            samples=tuple(
                RolloutSample(prompt_tokens=10, completion_tokens=40, truncated=False)
                for _ in range(MIN_TRUSTWORTHY_ROLLOUTS)
            ),
            failures=0,
            sampled_prompts=8,
        ),
    )
    from flash.engine.profiling.rollout_evidence import collect_rollout_evidence

    evidence = collect_rollout_evidence(
        model="m",
        prompts=[[{"role": "user", "content": "p"}]],
        max_completion_tokens=512,
        temperature=None,
        top_p=1.0,
    )
    assert evidence["sample_policy_version"] == ROLLOUT_SAMPLE_POLICY_VERSION
    # and it survives the server's gate, which is the point of emitting it.
    assert rollout_profile_from_evidence(_spec(), evidence, producer_version=VERSION) is not None


def test_a_message_the_sampler_cannot_reproduce_is_not_measured():
    """The workers pass the ORIGINAL message dicts to `apply_chat_template`.

    A template that reads `name` renders a different prompt than a role/content reconstruction, so
    silently dropping the field would report a confident measurement of a prompt the run never
    sends. Rejecting returns the quote to the cap, which costs nothing.
    """
    from flash.cli.commands.rollout_profile import _chat_messages

    plain = [{"role": "user", "content": "hi"}]
    assert _chat_messages(plain) == plain

    assert _chat_messages([{"role": "user", "content": "hi", "name": "alice"}]) == []
    assert _chat_messages([{"role": "assistant", "content": "hi", "tool_calls": []}]) == []


def test_a_truncated_draw_is_recorded_at_the_cap_it_will_generate_to():
    """A truncated draw is a CENSORED observation: the true length is at least the cap.

    A provider can stop below the requested cap for its own reasons -- an output ceiling, an
    exhausted context -- and keeping that smaller count would pull the mean down. Up to
    MAX_TRUSTWORTHY_TRUNCATION_RATE of such draws are accepted, so they would underquote the run.
    """
    short_truncation = {
        "usage": {"completion_tokens": 120, "prompt_tokens": 20},
        "choices": [{"finish_reason": "length"}],
    }
    sample = rollout_sampler._sample_from_response(short_truncation, max_completion_tokens=512)
    assert sample.truncated is True
    assert sample.completion_tokens == 512, (
        "a truncated draw contributes the cap, not where the provider happened to stop"
    )

    # a natural end is reported as measured -- the cap must not overwrite a real observation.
    natural = {
        "usage": {"completion_tokens": 120, "prompt_tokens": 20},
        "choices": [{"finish_reason": "stop"}],
    }
    eos = rollout_sampler._sample_from_response(natural, max_completion_tokens=512)
    assert (eos.truncated, eos.completion_tokens) == (False, 120)


def test_client_measured_grading_latency_never_reaches_the_quote():
    """The evidence must carry no reward latency at all, whatever an env's scorer does here.

    Reward seconds are multiplied by every completion of every step, so a client-host figure moves
    the quote a long way: a 5s/completion reading takes the default grpo spec from $4.14 to $10.94.
    That number describes this laptop, not the rented worker.
    """
    import inspect

    from flash.engine.profiling.rollout_evidence import collect_rollout_evidence

    # the collector must not even accept a scorer any more -- the parameter is the seam.
    parameters = inspect.signature(collect_rollout_evidence).parameters
    assert "score_one" not in parameters
    assert "reward_samples" not in parameters


def test_a_declared_secret_reaches_the_locally_run_environment(monkeypatch, tmp_path):
    """An external-judge reward() must see the same declared secret the worker gets.

    `runtime_secrets_from_local_env` returns a .env-only secret for SUBMISSION but never exports it,
    so env code executed here raised KeyError, the fail-open guard swallowed it, and the quote
    silently returned to the cap -- for exactly the envs whose grading cost matters most.
    """
    from flash.cli.commands import rollout_profile as rp

    entry = tmp_path / "package" / "environment.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("")

    seen: dict = {}

    class _Env:
        multi_turn = False

        def dataset(self):
            # env code reads its declared secret exactly where the worker's would.
            seen["visible"] = os.environ.get("JUDGE_API_KEY")
            return [{"q": "2+2?"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["q"]}]

    class _Client:
        def download_env_package(self, _env_id):
            return b""

    monkeypatch.setattr(
        "flash.envs.pull.pull_environment_package_from_archive", lambda *_a: tmp_path / "package"
    )
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda *_a, **_k: _Env())
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_evidence.collect_rollout_evidence", lambda **kw: None
    )
    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)

    spec = _spec(
        environment=EnvironmentSpec(
            id="acme/math", resolved_sha="a" * 40, secrets=("JUDGE_API_KEY",)
        )
    )
    rp.collect_for_submit(
        _Client(), spec, runtime_secrets={"JUDGE_API_KEY": "sk-local"}, debug=True
    )

    assert seen["visible"] == "sk-local", (
        "env code run for the quote must see the declared secret the worker will receive"
    )
    assert "JUDGE_API_KEY" not in os.environ, "the secret must not outlive the measurement"


def test_an_undeclared_secret_is_not_handed_to_environment_code(monkeypatch, tmp_path):
    """Scope: only what the spec DECLARES. WANDB_API_KEY is always collected, and user env code has
    no business reading it."""
    from flash.cli.commands import rollout_profile as rp

    entry = tmp_path / "package" / "environment.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("")

    seen: dict = {}

    class _Env:
        multi_turn = False

        def dataset(self):
            seen["wandb"] = os.environ.get("WANDB_API_KEY")
            return [{"q": "2+2?"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["q"]}]

    class _Client:
        def download_env_package(self, _env_id):
            return b""

    monkeypatch.setattr(
        "flash.envs.pull.pull_environment_package_from_archive", lambda *_a: tmp_path / "package"
    )
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda *_a, **_k: _Env())
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_evidence.collect_rollout_evidence", lambda **kw: None
    )
    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    # declares nothing, so nothing may be exported even though a secret was collected.
    rp.collect_for_submit(
        _Client(), _spec(), runtime_secrets={"WANDB_API_KEY": "sk-wandb"}, debug=True
    )
    assert seen["wandb"] is None, "an undeclared secret must not reach environment code"


def test_the_cost_preview_says_when_a_dry_run_would_quote_differently(monkeypatch, capsys):
    """`flash train --cost` prices rollouts at the cap; a dry-run on the SAME config measures them.

    The two deliberately disagree, and --cost is the command meant for pre-spend decisions -- so the
    difference has to be stated rather than surfacing as an unexplained change at submit.
    """
    from flash.cli.commands import _cmd_train_cost_offline

    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    assert _cmd_train_cost_offline(_spec()) == 0
    assert "declared completion cap" in capsys.readouterr().err


def test_the_cost_preview_stays_quiet_when_no_measurement_could_happen(monkeypatch, capsys):
    """No sampler key, or a config the draw cannot reproduce, means BOTH paths quote from the cap.
    Warning there would describe a disagreement that does not exist."""
    from flash.cli.commands import _cmd_train_cost_offline

    monkeypatch.delenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, raising=False)
    assert _cmd_train_cost_offline(_spec()) == 0
    assert "declared completion cap" not in capsys.readouterr().err

    # key present, but the config is declined -- still no disagreement. a warm-start adapter rather
    # than a pinned revision, since a fabricated revision fails sizing lookup before the preview.
    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    declined = _spec(
        train=TrainSpec(
            max_steps=50,
            batch_size=8,
            group_size=4,
            max_completion_tokens=512,
            init_from_adapter="acme/run-1",
        )
    )
    assert _cmd_train_cost_offline(declined) == 0
    assert "declared completion cap" not in capsys.readouterr().err


def test_generation_pricing_stays_bounded_by_the_cap_when_the_actor_drifts():
    """RL changes generation length, and the profile only ever sees the PRE-TRAINING actor.

    That exposure is real but bounded, and the bound is what makes pricing the whole horizon from an
    initial sample defensible. `_completion_tokens` clamps to the declared cap, so an actor that
    drifts LONGER converges to the cap-based quote this path used before any measurement existed --
    it never prices generation above it. An actor that drifts SHORTER is over-quoted, which is the
    safe direction.

    Asserted on the generation term itself, not on a total: a measured quote CAN exceed the
    cap-based one, because measured grading latency is real cost the cap-based quote simply omits.
    Conflating the two would let a grading regression pass as a generation bound.
    """
    from flash.cost.analytical import _completion_tokens, _sequence_tokens
    from flash.cost.spec import runconfig_from_spec

    spec = _spec()
    cap = spec.train.max_completion_tokens
    cap_cfg = runconfig_from_spec(spec).normalized()

    def _measured(mean: float):
        evidence = _evidence(
            completion_tokens_mean=mean,
            completion_tokens_p50=int(mean),
            completion_tokens_p90=min(cap, int(mean * 1.2)),
            completion_tokens_max=min(cap, int(mean * 1.3)),
        )
        profile = rollout_profile_from_evidence(spec, evidence, producer_version=VERSION)
        assert profile is not None
        return runconfig_from_spec(_with_profile(spec, profile)).normalized()

    # an actor that drifted all the way to the cap is priced at the cap, never above it.
    assert _completion_tokens(_measured(float(cap))) == pytest.approx(float(cap))
    assert _sequence_tokens(_measured(float(cap))) <= _sequence_tokens(cap_cfg)
    # the pre-training sample only ever discounts generation off the cap.
    assert _completion_tokens(_measured(53.6875)) < float(cap)

    # a mean above the sample's own max is incoherent, so the server refuses it outright rather
    # than clamping -- the ceiling is enforced before pricing, not inside it.
    incoherent = _evidence(
        completion_tokens_mean=2000.0,
        completion_tokens_p50=2000,
        completion_tokens_p90=cap,
        completion_tokens_max=cap,
    )
    assert rollout_profile_from_evidence(spec, incoherent, producer_version=VERSION) is None


def test_a_draw_starting_near_the_deadline_cannot_overrun_the_sampling_budget(monkeypatch):
    """The overall deadline must bound the WALL TIME of a pass, not merely when a draw may start.

    Checking the deadline only at the top of the loop let a draw begun just inside it stall for the
    full per-request timeout, so a "bounded" pass ran for deadline + REQUEST_TIMEOUT_S in front of a
    submit the user is waiting on. Asserting on the timeout handed to the request rather than on
    elapsed time keeps this failable without making the suite sleep.
    """
    monkeypatch.setattr(rollout_sampler, "SAMPLING_DEADLINE_S", 4.0)
    monkeypatch.setattr(rollout_sampler, "REQUEST_TIMEOUT_S", 180.0)

    timeouts: list[float] = []
    clock = {"now": 0.0}

    def _fake_completion(**kwargs) -> RolloutSample | None:
        timeouts.append(kwargs["timeout_s"])
        # each draw consumes most of the budget, so the second one starts just inside the deadline.
        clock["now"] += 3.9
        # a failed draw, which is what keeps the loop going to the next attempt.
        return None

    monkeypatch.setattr(rollout_sampler.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(rollout_sampler, "_one_completion", _fake_completion)

    rollout_sampler.sample_rollouts(
        model="m",
        prompts=[[{"role": "user", "content": "hi"}]],
        rollouts=32,
        max_completion_tokens=128,
        temperature=None,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
    )

    assert timeouts, "the pass made no request at all"
    # no single draw may be allowed to run past the moment the whole pass was due to stop.
    assert timeouts[-1] == pytest.approx(0.1)
    assert max(timeouts) <= rollout_sampler.SAMPLING_DEADLINE_S


def test_the_measurement_seeds_the_generators_the_worker_seeds(monkeypatch):
    """Env code may build its dataset through torch, which the workers seed and this did not.

    Measured before the fix: two CLI processes quoting the same spec.seed selected different rows
    ([44, 19, 93, 90, 71] vs [52, 74, 73, 81, 76]) because the profile inherited whatever torch
    state the process happened to carry, while the worker was stable. The profile then describes
    rows the run never trains on -- and still passes the trust gate.
    """
    torch = pytest.importorskip("torch")
    from flash.cli.commands.rollout_profile import _seed_environment_rngs
    from flash.engine.worker.runtime.rng import seed_training_rngs

    def rows():
        return torch.randperm(100)[:5].tolist()

    drawn = []
    for ambient in (0, 999):
        torch.manual_seed(ambient)
        _seed_environment_rngs(_spec(seed=7))
        drawn.append(rows())

    torch.manual_seed(12345)
    seed_training_rngs(7)
    from_worker = rows()

    assert drawn[0] == drawn[1], "the profile still depends on incidental process rng state"
    assert drawn[0] == from_worker, "the profile samples rows the worker would not"


def test_measuring_does_not_leave_the_process_torch_seeded(monkeypatch):
    """Seeding torch for the env's benefit must not make the rest of the CLI deterministic.

    The process goes on to submit the run. Restoring python and numpy but not torch would leave the
    same defect in a subset of the generators.
    """
    torch = pytest.importorskip("torch")
    from flash.cli.commands.rollout_profile import _restored_host_rng, _seed_environment_rngs

    def rows():
        return torch.randperm(100)[:5].tolist()

    torch.manual_seed(4242)
    undisturbed = rows()

    torch.manual_seed(4242)
    with _restored_host_rng():
        _seed_environment_rngs(_spec(seed=7))
        rows()
    after_measuring = rows()

    assert after_measuring == undisturbed


@pytest.mark.parametrize(
    ("stages", "evidence", "accepted", "must_say", "must_not_say"),
    [
        # a multi-turn env declines on the import alone: naming dataset() here would be a lie.
        (("import",), None, False, "no usable measurement came back", "dataset()"),
        # an empty dataset stops before prompt_messages().
        (("import", "dataset"), None, False, "dataset() ran", "prompt_messages()"),
        (
            ("import", "dataset", "prompt_messages"),
            None,
            False,
            "dataset() and prompt_messages() ran",
            "did NOT import",
        ),
        (
            ("import", "dataset", "prompt_messages"),
            {"completed_rollouts": 32},
            True,
            "priced from the rollouts it measured",
            "did NOT import",
        ),
        ((), None, False, "did NOT import or run your environment.py", "WAS imported locally"),
    ],
)
def test_the_dry_run_reports_the_hooks_that_ran_not_whether_it_measured(
    stages, evidence, accepted, must_say, must_not_say
):
    """Profiling runs the user's module BEFORE it can find the config unmeasurable, and the declines
    stop at DIFFERENT points.

    Inferring execution from the payload told these users their code had not been imported at all.
    Reporting a fixed pair of hooks instead would misstate it the other way: a multi-turn env is
    refused after the import, having never called dataset().
    """
    from flash.cli.commands import _dry_run_preview_line

    line = _dry_run_preview_line(
        algorithm="grpo",
        affordability_verified=True,
        rollout_evidence=evidence,
        environment_stages_run=stages,
        rollout_evidence_accepted=accepted,
    )
    assert must_say in line
    assert must_not_say not in line


def test_the_notice_reports_the_servers_verdict_not_the_clients_payload():
    """The client's only gate is "at least one draw came back".

    A deadline-truncated pass of 31 draws, or one whose truncation rate or prompt coverage fails
    the trust gate, is evidence on this side and REJECTED on the server's -- which then prices the
    run at the declared cap. Reporting the client's optimism tells the user the opposite of what
    they will be billed against, and pre-spend is exactly when that matters.
    """
    from flash.cli.commands import _dry_run_preview_line

    measured = {"completed_rollouts": 31}
    stages = ("import", "dataset", "prompt_messages")

    rejected = _dry_run_preview_line(
        algorithm="grpo",
        affordability_verified=True,
        rollout_evidence=measured,
        environment_stages_run=stages,
        rollout_evidence_accepted=False,
    )
    assert "still uses the declared completion cap" in rejected
    assert "priced from the rollouts it measured" not in rejected
    # what actually ran locally is unchanged by the server's verdict, and still reported.
    assert "dataset() and prompt_messages() ran" in rejected

    accepted = _dry_run_preview_line(
        algorithm="grpo",
        affordability_verified=True,
        rollout_evidence=measured,
        environment_stages_run=stages,
        rollout_evidence_accepted=True,
    )
    assert "priced from the rollouts it measured" in accepted


def test_the_submit_path_learns_the_environment_ran_even_with_no_evidence(monkeypatch, tmp_path):
    """The seam between profiling and the notice.

    _collect returns None for a multi-turn env, so only the load hook can tell the caller the user's
    code executed. Testing the notice alone leaves this uncovered: the hook never firing is silent,
    and the dry run goes back to claiming nothing was imported.
    """
    from flash.cli.commands import _rollout_evidence_for

    class _MultiTurn:
        multi_turn = True

        def dataset(self):  # pragma: no cover - declined before this is reached
            raise AssertionError("a multi-turn env must not be measured")

    client, _seen = _profiling_stubs(monkeypatch, tmp_path, _MultiTurn())
    monkeypatch.setattr("flash.cli.commands.client_from_config", lambda: client, raising=False)

    evidence, stages = _rollout_evidence_for(client, _spec())

    assert evidence is None
    # the import ran; dataset() did not, because a multi-turn env is declined before it.
    assert stages == ("import",)


def test_a_later_decline_reports_the_hooks_it_got_through(monkeypatch, tmp_path):
    """The stages must be reported as they RUN, not inferred from where the decline happened.

    An env whose dataset() returns rows but whose prompts are unsamplable gets further than a
    multi-turn one: dataset() really did run, and a notice that omitted it would understate what
    executed on the user's machine just as the fixed pair overstated it.
    """
    from flash.cli.commands import _rollout_evidence_for

    class _NoUsablePrompts:
        multi_turn = False

        def dataset(self):
            return [{"q": "2+2?"}]

        def prompt_messages(self, _example):
            # a shape the sampler cannot reproduce, so the measurement is declined after this ran.
            return [{"role": "user", "content": "hi", "name": "tool-call"}]

    client, _seen = _profiling_stubs(monkeypatch, tmp_path, _NoUsablePrompts())
    monkeypatch.setattr("flash.cli.commands.client_from_config", lambda: client, raising=False)

    evidence, stages = _rollout_evidence_for(client, _spec())

    assert evidence is None
    assert stages == ("import", "dataset", "prompt_messages")


@pytest.mark.parametrize(
    ("hangs_in", "expected"),
    [("dataset", ("import",)), ("prompt_messages", ("import", "dataset"))],
)
def test_a_hook_that_hangs_is_not_reported_as_having_run(monkeypatch, tmp_path, hangs_in, expected):
    """The deadline bounds the wait; it does not make the hook finish.

    The hook is still running on a daemon thread when the deadline expires, so a notice saying it
    "ran on a small sample" describes work that never completed -- the exact overclaim the
    per-stage reporting was added to remove. Both bounded hooks are covered: they are separate
    call sites, so reporting-before-checking can be reintroduced in one without the other noticing.
    """
    from flash.cli.commands import _rollout_evidence_for
    from flash.cli.commands import rollout_profile as rp

    monkeypatch.setattr(rp, "_LOCAL_HOOK_DEADLINE_S", 0.2)
    release = threading.Event()

    class _HangingHook:
        multi_turn = False

        def dataset(self):
            if hangs_in == "dataset":
                release.wait(30)
            return [{"q": "2+2?"}]

        def prompt_messages(self, _example):
            if hangs_in == "prompt_messages":
                release.wait(30)
            return [{"role": "user", "content": "2+2?"}]

    client, _seen = _profiling_stubs(monkeypatch, tmp_path, _HangingHook())
    monkeypatch.setattr("flash.cli.commands.client_from_config", lambda: client, raising=False)

    try:
        evidence, stages = _rollout_evidence_for(client, _spec())
        assert evidence is None
        assert stages == expected, f"claimed a hung hook ran: {stages}"
    finally:
        release.set()


def test_a_draw_that_reasons_against_a_non_thinking_run_is_not_a_measurement():
    """Every catalog model is hybrid-thinking, so `thinking = false` is the common case.

    Chat-completions has no portable reasoning switch, so an endpoint that reasons by default would
    answer such a run with traces it will never generate -- and those tokens are billed in
    `completion_tokens`. Declining the draw returns the quote to the cap; keeping it would quote a
    length no rollout produces.
    """
    reasoning_body = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "4", "reasoning": "the user wants 2+2..."},
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 900,
            "completion_tokens_details": {"reasoning_tokens": 880},
        },
    }

    declined = rollout_sampler._sample_from_response(
        reasoning_body, max_completion_tokens=2048, thinking=False
    )
    assert declined is None

    # the SAME response is a valid measurement for a run that asked for thinking.
    kept = rollout_sampler._sample_from_response(
        reasoning_body, max_completion_tokens=2048, thinking=True
    )
    assert kept is not None
    assert kept.completion_tokens == 900


def test_a_plain_draw_against_a_thinking_run_is_not_a_measurement():
    """The mismatch has to be rejected in BOTH directions, and this is the likelier one.

    `reasoning` and `chat_template_kwargs` are non-standard, so an endpoint may accept the request
    and ignore both, answering a `thinking = true` run with a short plain completion. Nothing about
    that response looks wrong -- it is a clean `stop` with sane token counts -- but every real
    rollout will emit traces it is missing, so it underprices the run by the whole trace length.
    """
    plain_body = {
        "choices": [{"finish_reason": "stop", "message": {"content": "4"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 12},
    }

    declined = rollout_sampler._sample_from_response(
        plain_body, max_completion_tokens=2048, thinking=True
    )
    assert declined is None

    # the same body IS the measurement for a run that asked for no reasoning.
    kept = rollout_sampler._sample_from_response(
        plain_body, max_completion_tokens=2048, thinking=False
    )
    assert kept is not None
    assert kept.completion_tokens == 12

    # a model with no reasoning mode at all (thinking=None) is not checked in either direction.
    assert (
        rollout_sampler._sample_from_response(plain_body, max_completion_tokens=2048, thinking=None)
        is not None
    )


@pytest.mark.parametrize(
    "message",
    [
        {"content": "4", "reasoning": "..."},
        {"content": "4", "reasoning_content": "..."},
        {"content": "<think>hmm</think>4"},
    ],
)
def test_reasoning_is_detected_wherever_the_provider_puts_it(message):
    """Providers disagree on where reasoning surfaces: a usage breakdown, a separate message field
    under either of two names, or the literal think tags a chat template emits. Checking only one
    would let the other two through as ordinary long completions."""
    body = {
        "choices": [{"finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 900},
    }
    assert (
        rollout_sampler._sample_from_response(body, max_completion_tokens=2048, thinking=False)
        is None
    )


def test_the_request_states_the_runs_reasoning_setting(monkeypatch):
    """Leaving reasoning to the endpoint's default is what the response check exists to catch --
    but the request should ask for the right thing first, so the common case is a usable draw
    rather than a rejected one."""
    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"finish_reason": "stop", "message": {"content": "4"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 40},
                }
            ).encode()

    def _open(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return _Response()

    monkeypatch.setattr(rollout_sampler._NO_REDIRECT_OPENER, "open", _open)

    rollout_sampler._one_completion(
        model="Qwen/Qwen3.5-4B",
        messages=[{"role": "user", "content": "2+2?"}],
        max_completion_tokens=512,
        temperature=None,
        top_p=1.0,
        base_url="https://example.invalid/v1",
        api_key="k",
        thinking=False,
    )

    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["reasoning"] == {"enabled": False}


def test_the_request_states_the_runs_nucleus_sampling(monkeypatch):
    """Both workers set `actor_rollout_ref.rollout.top_p` explicitly from `sampling_top_p`.

    Nucleus sampling decides which continuations are selectable, so it moves the stopping-length
    distribution this profile measures. Omitting it from the request inherits whatever the endpoint
    defaults to, and a draw decoded differently from training cannot price training.
    """
    captured: dict = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"finish_reason": "stop", "message": {"content": "4"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 40},
                }
            ).encode()

    def _open(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return _Response()

    monkeypatch.setattr(rollout_sampler._NO_REDIRECT_OPENER, "open", _open)

    rollout_sampler._one_completion(
        model="Qwen/Qwen3.5-4B",
        messages=[{"role": "user", "content": "2+2?"}],
        max_completion_tokens=512,
        temperature=None,
        top_p=0.9,
        base_url="https://example.invalid/v1",
        api_key="k",
    )

    assert captured["body"]["top_p"] == 0.9


def test_a_prompt_the_worker_would_drop_is_not_measured():
    """Both workers drop rows over `context - max_completion` before training ever starts.

    The hosted endpoint has no such limit, so it answers those prompts happily -- and their
    lengths would land in a profile describing steps that never run them. Long rows are also
    exactly the ones that would dominate the mean, so keeping them skews the direction that
    overbills.
    """
    over = {
        "choices": [{"finish_reason": "stop", "message": {"content": "4"}}],
        "usage": {"prompt_tokens": 1600, "completion_tokens": 40},
    }
    assert (
        rollout_sampler._sample_from_response(over, max_completion_tokens=512, prompt_budget=1536)
        is None
    )

    # a row inside the budget is measured normally, and no budget means no filter.
    within = {
        "choices": [{"finish_reason": "stop", "message": {"content": "4"}}],
        "usage": {"prompt_tokens": 1536, "completion_tokens": 40},
    }
    assert (
        rollout_sampler._sample_from_response(within, max_completion_tokens=512, prompt_budget=1536)
        is not None
    )
    assert (
        rollout_sampler._sample_from_response(over, max_completion_tokens=512, prompt_budget=None)
        is not None
    )


def test_the_submit_path_forwards_the_runs_prompt_budget_to_the_measurement(monkeypatch, tmp_path):
    """The seam: computing the budget is useless if the sampler never receives it.

    The filter lives in the response reader, so a budget that stops at the CLI leaves every
    over-budget draw in the profile exactly as before -- and nothing else would notice.
    """
    from flash.cli.commands import rollout_profile as rp

    class _Env:
        multi_turn = False

        def dataset(self):
            return [{"q": "2+2?"}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["q"]}]

    client, seen = _profiling_stubs(monkeypatch, tmp_path, _Env())
    spec = _spec(
        train=TrainSpec(
            max_steps=10,
            batch_size=8,
            group_size=4,
            max_completion_tokens=512,
            max_context_tokens=2048,
        )
    )

    rp.collect_for_submit(client, spec, debug=True)

    assert seen["prompt_budget"] == 1536
    # and the run's decoding settings travel with it.
    assert seen["top_p"] == pytest.approx(1.0)


def test_the_prompt_budget_mirrors_the_workers_arithmetic():
    """`context - max_completion`, and None when the run declares no context limit.

    Only the DECLARED context: the workers also clamp to the model's architectural limit, which
    needs a config fetch this path must not make. That clamp can only LOWER the budget, so the
    declared value stays the looser bound -- it never invents a rejection the worker would not make.
    """
    from flash.cli.commands.rollout_profile import _prompt_budget

    spec = _spec(
        train=TrainSpec(
            max_steps=10,
            batch_size=8,
            group_size=4,
            max_completion_tokens=512,
            max_context_tokens=2048,
        )
    )
    assert _prompt_budget(spec) == 1536

    # no declared context -> nothing to filter against, rather than a guessed limit.
    assert _prompt_budget(_spec()) is None

    # a context that leaves no room for a prompt filters nothing rather than everything.
    airless = _spec(
        train=TrainSpec(
            max_steps=10,
            batch_size=8,
            group_size=4,
            max_completion_tokens=2048,
            max_context_tokens=2048,
        )
    )
    assert _prompt_budget(airless) is None


def test_the_measurement_samples_at_the_top_p_the_workers_use(monkeypatch):
    """Read from the recipe the workers read, not hardcoded, so it stays correct if that changes."""
    from flash.cli.commands.rollout_profile import _sampling_top_p
    from flash.engine.plan.recipe import RECIPE

    assert _sampling_top_p(_spec(algorithm="grpo")) == float(RECIPE.rl.sampling_top_p)
    assert _sampling_top_p(_spec(algorithm="opd")) == float(RECIPE.opd.sampling_top_p)


def test_a_blocking_environment_hook_cannot_hang_the_submit(monkeypatch):
    """`dataset()` is user code. A hook that blocks forever raises nothing, so the outer `except`
    never regains control and merely CONFIGURING the optional sampler would hang `flash train`
    indefinitely. Bounding it returns the quote to cap pricing instead."""
    from flash.cli.commands import rollout_profile as rp

    monkeypatch.setattr(rp, "_LOCAL_HOOK_DEADLINE_S", 0.2)

    started = threading.Event()
    release = threading.Event()

    def _never_returns():
        started.set()
        # blocks until the test releases it, so an unbounded join would hang here rather than
        # returning None -- asserting only on the return value would pass either way.
        release.wait(30)
        return [{"q": "2+2?"}]

    began = time.monotonic()
    try:
        finished, value = rp._within(rp._LOCAL_HOOK_DEADLINE_S, _never_returns)
        elapsed = time.monotonic() - began
        assert (finished, value) == (False, None)
        assert started.is_set(), "the call never actually started"
        # the point of the fix: it came back on the DEADLINE, not when the hook finished.
        assert elapsed < 5.0, f"waited {elapsed:.1f}s, so the deadline did not bound the call"
    finally:
        release.set()


def test_a_bounded_hook_still_returns_its_value():
    """The deadline must not cost a measurement for every env whose dataset() is merely slow."""
    from flash.cli.commands import rollout_profile as rp

    assert rp._within(30.0, lambda: [{"q": "2+2?"}]) == (True, [{"q": "2+2?"}])
    # a raising hook is reported as "no measurement", like any other failure on this path -- but it
    # COMPLETED: the user's code ran, so the notice may say so. only a hang reports False.
    assert rp._within(30.0, lambda: (_ for _ in ()).throw(RuntimeError("boom"))) == (True, None)
    # a hook that legitimately returns something falsy must not be mistaken for one that hung.
    assert rp._within(30.0, list) == (True, [])


def test_a_stalled_package_download_cannot_hold_the_submit(monkeypatch, tmp_path):
    """`download_env_package` hardcodes a 1800s timeout, right for an interactive `flash env pull`
    and wrong on an advisory path: a stalled package route would hold `flash train` for half an
    hour before falling back to the cap-based quote it would have produced anyway.

    It also runs BEFORE the local-hook and sampling deadlines, so neither of those bounds it.
    """
    from flash.cli.commands import _rollout_evidence_for
    from flash.cli.commands import rollout_profile as rp

    monkeypatch.setattr(rp, "_LOCAL_HOOK_DEADLINE_S", 0.2)
    release = threading.Event()

    class _Fine:
        multi_turn = False

        def dataset(self):
            return [{"q": "2+2?"}]

        def prompt_messages(self, _example):
            return [{"role": "user", "content": "2+2?"}]

    client, _seen = _profiling_stubs(monkeypatch, tmp_path, _Fine())
    monkeypatch.setattr("flash.cli.commands.client_from_config", lambda: client, raising=False)
    monkeypatch.setattr(
        client, "download_env_package", lambda _env_id: release.wait(30) or b"", raising=False
    )

    began = time.monotonic()
    try:
        evidence, stages = _rollout_evidence_for(client, _spec())
        elapsed = time.monotonic() - began
        assert evidence is None
        # nothing of the user's code ran: the archive it lives in never arrived.
        assert stages == ()
        assert elapsed < 5.0, f"waited {elapsed:.1f}s, so the download was not bounded"
    finally:
        release.set()


def test_a_prompt_the_endpoint_always_refuses_does_not_count_as_sampled():
    """`sampled_prompts` must mean "produced a draw", not "was attempted".

    A content filter that rejects one prompt every time is invisible in the rollout count: the loop
    makes up the shortfall from the other prompts and reports full coverage. Prompt mix dominates
    length variance and the training worker does not reproduce the hosted filter, so that prompt is
    still trained on -- its (usually long-tail) length would simply be missing from the quote.
    """
    prompts = [[{"role": "user", "content": f"p{i}"}] for i in range(8)]

    def _filtered(**kwargs):
        if kwargs["messages"][0]["content"] == "p3":
            return None
        return RolloutSample(prompt_tokens=10, completion_tokens=40, truncated=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rollout_sampler, "_one_completion", _filtered)
        out = rollout_sampler.sample_rollouts(
            model="m",
            prompts=prompts,
            rollouts=32,
            max_completion_tokens=512,
            temperature=None,
            top_p=1.0,
            base_url="https://example.invalid/v1",
            api_key="k",
        )

    assert out.completed == 32
    assert out.failures > 0
    # 7 of the 8 offered prompts produced draws; the refused one must not be counted.
    assert out.summary()["sampled_prompts"] == 7
    assert out.summary()["offered_prompts"] == 8


def test_evidence_missing_a_prompt_entirely_is_not_trustworthy():
    """The honest count alone does not protect the quote -- the trust gate has to act on it.

    A profile whose numbers are all individually valid, but which silently measured only part of
    the prompt set, must not be allowed to price the run.
    """
    from flash.engine.profiling.workload_profile import (
        MIN_TRUSTWORTHY_PROMPT_COVERAGE,
        RolloutWorkloadProfile,
    )

    spec = _spec()
    full = rollout_profile_from_evidence(
        spec, _evidence(sampled_prompts=8, offered_prompts=8), producer_version=VERSION
    )
    assert full is not None
    assert RolloutWorkloadProfile.from_dict(full).trustworthy(now=time.time()) == (True, "")

    # three of eight prompts never produced a draw: below the coverage floor.
    assert MIN_TRUSTWORTHY_PROMPT_COVERAGE > 5 / 8
    partial = rollout_profile_from_evidence(
        spec, _evidence(sampled_prompts=5, offered_prompts=8), producer_version=VERSION
    )
    # rejected either at parse time or by the trust verdict; both keep it away from the quote.
    if partial is not None:
        ok, reason = RolloutWorkloadProfile.from_dict(partial).trustworthy(now=time.time())
        assert ok is False
        assert "prompts produced a draw" in reason


def test_an_install_that_cannot_load_environments_says_so_instead_of_going_quiet(
    monkeypatch, caplog
):
    """The base `freesolo-flash` package declares no dependencies and the env loader needs the
    `freesolo` SDK, so an isolated `uv tool`/`pipx` install cannot profile at all.

    Failing open is right -- the quote returns to the cap, which is what an unmeasured run always
    cost. Failing open SILENTLY is not: the user set a sampler key expecting a measured quote, and
    no output is indistinguishable from the feature working.
    """
    from flash.cli.commands import rollout_profile as rp

    def _no_sdk(*_a, **_k):
        raise ImportError(f"{rp._MISSING_ENVIRONMENT_TOOLS}: No module named ...")

    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    monkeypatch.setattr(rp, "_collect", _no_sdk)

    with caplog.at_level(logging.WARNING, logger="flash.cli"):
        assert rp.collect_for_submit(object(), _spec()) is None

    assert "rollout profiling is unavailable in this install" in caplog.text
    # and it still fails OPEN -- an unusable install must not break the submit.


def test_a_missing_dependency_in_the_users_env_does_not_blame_the_flash_install(
    monkeypatch, caplog
):
    """An import inside the user's own `environment.py` reaches the same handler.

    Both arrive as `ImportError`, but they are fixed on different machines: one means "this flash
    install cannot load environments", the other means "your env needs a dependency". Reporting the
    first for the second sends the user to reinstall a working tool while the real cause is in
    their file, and reads as if profiling worked.
    """
    from flash.cli.commands import rollout_profile as rp

    def _env_missing_dep(*_a, **_k):
        # exactly what `import pandas` at the top of a user environment.py raises.
        raise ModuleNotFoundError("No module named 'pandas'")

    monkeypatch.setenv(rollout_sampler.ROLLOUT_SAMPLER_API_KEY_ENV, "k")
    monkeypatch.setattr(rp, "_collect", _env_missing_dep)

    with caplog.at_level(logging.WARNING, logger="flash.cli"):
        assert rp.collect_for_submit(object(), _spec()) is None

    assert "your environment.py could not be imported" in caplog.text
    assert "pandas" in caplog.text, "the actual cause has to survive into the warning"
    assert "unavailable in this install" not in caplog.text


def test_the_install_failure_prefix_still_matches_the_loader():
    """The two sides are separate files, and the discriminator is a copied string literal.

    If the loader ever rewords its own diagnostic, the check above silently stops matching and
    every unusable install starts being reported as a broken user env -- with nothing failing.
    """
    import inspect

    from flash.cli.commands import rollout_profile as rp
    from flash.envs import loader

    source = inspect.getsource(loader._import_freesolo_environment_tools)
    assert rp._MISSING_ENVIRONMENT_TOOLS in source
