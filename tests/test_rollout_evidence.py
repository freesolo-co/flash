"""Client-measured rollout evidence: sampling, server validation, and the fail-open contract.

The security property under test is that a client cannot price its own run. The client measures and
sends aggregates; the SERVER re-derives this config's digest and re-applies the same trust verdict a
first-party profile passes. So every hostile payload below must be rejected, and every rejection must
leave the quote on the declared cap rather than fail the submit.
"""

from __future__ import annotations

import json
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
        base_url="https://example.invalid/v1",
        api_key="k",
    )
    assert sampling.completed == 2
    assert sampling.failures == 2


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
        base_url="https://example.invalid/v1",
        api_key="k",
    )
    assert calls["n"] == 20, "an answering endpoint must not trip the cutoff"
    assert sampling.completed == 10


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
    assert rollout_sampler._sample_from_response(payload) is None


@pytest.mark.parametrize(
    ("finish", "truncated"),
    [("stop", False), ("length", True)],
)
def test_natural_and_truncated_draws_are_both_kept(finish, truncated):
    payload = {
        "usage": {"completion_tokens": 12, "prompt_tokens": 20},
        "choices": [{"finish_reason": finish}],
    }
    sample = rollout_sampler._sample_from_response(payload)
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
    """A multimodal part-list prompt has no faithful single-string form. Dropping it returns the
    quote to the cap; sending a reshaped one would price a distribution that is not the run's."""
    from flash.cli.commands.rollout_profile import _prompts_from

    class _Env:
        def prompt_messages(self, example):
            return [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    assert _prompts_from(_Env(), [{"q": "x"}]) == []


def test_only_the_rows_training_consumes_are_measured():
    """grpo and opd both fence the population to a max_examples prefix. Measuring outside that
    fence samples rows the run never sees, and with a small max_examples those rows could set the
    whole measured length."""
    from flash.cli.commands.rollout_profile import _training_population

    spec = _spec(train=TrainSpec(max_steps=50, batch_size=8, group_size=4, max_examples=2))
    assert _training_population(spec, [1, 2, 3, 4, 5]) == [1, 2]
    unfenced = _spec()
    assert _training_population(unfenced, [1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


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
    summary = RolloutSampling(samples=samples, failures=0, sampled_prompts=1).summary()
    assert set(summary) == {
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

    def _fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return _Response()

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler.json.load",
        lambda _r: {
            "usage": {"completion_tokens": 12, "prompt_tokens": 20},
            "choices": [{"finish_reason": "stop"}],
        },
    )
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler.urllib.request.urlopen", _fake_urlopen
    )
    rollout_sampler._one_completion(
        model="m",
        messages=[{"role": "user", "content": "p"}],
        max_completion_tokens=64,
        temperature=None,
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

    def _fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data.decode())
        return _Response()

    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler.json.load",
        lambda _r: {
            "usage": {"completion_tokens": 12, "prompt_tokens": 20},
            "choices": [{"finish_reason": "stop"}],
        },
    )
    monkeypatch.setattr(
        "flash.engine.profiling.rollout_sampler.urllib.request.urlopen", _fake_urlopen
    )
    rollout_sampler._one_completion(
        model="m",
        messages=[{"role": "user", "content": "p"}],
        max_completion_tokens=64,
        temperature=None,
        base_url="https://example.invalid/v1",
        api_key="k",
    )
    assert "stop" not in captured["body"]


@pytest.mark.parametrize(
    ("reward_samples", "must_say", "must_not_say"),
    [
        (3, "reward() were run", "was NOT called"),
        (0, "was NOT called", "external scorer was really called"),
    ],
)
def test_the_dry_run_claims_reward_ran_only_when_it_did(
    reward_samples, must_say, must_not_say, capsys
):
    """Grading is skipped for a multi-turn or thread-unsafe env, and for one whose gold completions
    cannot be read -- evidence still arrives from generation sampling alone. Telling a user a paid
    external scorer was called when it was not is the one claim here they could act on wrongly."""
    from flash.cli.commands import _dry_run_preview_line

    line = _dry_run_preview_line(
        algorithm="grpo",
        affordability_verified=True,
        rollout_evidence=_evidence(reward_samples=reward_samples),
    )
    assert must_say in line
    assert must_not_say not in line


def test_the_dry_run_says_nothing_ran_when_there_was_no_measurement():
    from flash.cli.commands import _dry_run_preview_line

    line = _dry_run_preview_line(
        algorithm="grpo", affordability_verified=True, rollout_evidence=None
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
