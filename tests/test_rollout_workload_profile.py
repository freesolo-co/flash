"""Schema, digest and trust-verdict behaviour for the grpo/opd rollout workload profile."""

from __future__ import annotations

from typing import ClassVar

import pytest

from flash.workload_profile import (
    ROLLOUT_LATENCY_MAX_AGE_S,
    RolloutWorkloadProfile,
    WorkloadProfileMismatch,
    require_matching_rollout_profile,
    rollout_profile_input_digest,
    rollout_profile_input_payload,
    rollout_profile_run_id,
)

NOW = 1_700_000_000.0
DIGEST = "a" * 64


def _fields(**overrides) -> dict:
    base = {
        "input_digest": DIGEST,
        "producer_version": "1.0.87",
        "tokenizer_revision": "tok-rev",
        "environment_id": "freesolo-co/autoslm-bench",
        "environment_revision": "env-sha",
        "kind": "grpo",
        "sampled_prompts": 16,
        "completed_rollouts": 16,
        "failed_rollouts": 0,
        "completion_tokens_mean": 180.5,
        "completion_tokens_p50": 170,
        "completion_tokens_p90": 240,
        "completion_tokens_max": 256,
        "prompt_tokens_mean": 95.0,
        "truncated_rollouts": 2,
        "eos_rollouts": 14,
        "generation_seconds_per_completion": 0.42,
        "reward_seconds_per_completion": 0.0003,
        "reward_samples": 3,
        "reward_failures": 0,
        "reference_gpu": "A100 PCIe",
        "reference_provider": "runpod",
        "sample_policy": "stratified-16",
        "created_at": NOW,
        "measured_at": NOW,
    }
    base.update(overrides)
    return base


def _profile(**overrides) -> RolloutWorkloadProfile:
    return RolloutWorkloadProfile(**_fields(**overrides))


class _Train:
    max_completion_tokens = 256
    max_context_tokens = 2048
    group_size = 4
    teacher_model = ""
    temperature = None
    # horizon fields exist on the real spec and must NOT reach the key
    epochs = 2
    max_steps = 40
    max_examples = 500


class _Env:
    id = "freesolo-co/autoslm-bench"
    resolved_sha = "env-sha"
    params: ClassVar[dict] = {}


class _Spec:
    algorithm = "grpo"
    model = "Qwen/Qwen3.5-4B"
    model_revision = "main"
    seed = 0
    thinking = False
    worker_env: ClassVar[dict] = {}
    train = _Train()
    environment = _Env()


def test_roundtrip_preserves_the_measurement():
    profile = _profile()
    assert RolloutWorkloadProfile.from_dict(profile.to_dict()) == profile


def test_provenance_is_outside_the_content_digest():
    # the training worker re-derives the measurement to check the workload did not move; a
    # timestamp inside the digest would fail that check on every run for the one reason that is
    # not a workload change.
    early = _profile(created_at=NOW, measured_at=NOW)
    late = _profile(created_at=NOW + 9_000, measured_at=NOW + 9_000)
    assert early.content_digest == late.content_digest
    assert early == late


def test_a_tampered_artifact_is_rejected_by_its_own_digest():
    raw = _profile().to_dict()
    raw["completion_tokens_p50"] = 8
    with pytest.raises(ValueError, match="content digest"):
        RolloutWorkloadProfile.from_dict(raw)


def test_unknown_or_missing_fields_are_rejected():
    raw = _profile().to_dict()
    raw["surprise"] = 1
    with pytest.raises(ValueError, match="fields do not match"):
        RolloutWorkloadProfile.from_dict(raw)


@pytest.mark.parametrize("kind", ["sft", "", "rl"])
def test_only_rollout_algorithms_are_accepted(kind):
    with pytest.raises(ValueError, match="kind must be"):
        _profile(kind=kind)


def test_percentiles_must_be_ordered():
    with pytest.raises(ValueError, match="p90 completion tokens cannot be below p50"):
        _profile(completion_tokens_p50=300, completion_tokens_p90=200)
    with pytest.raises(ValueError, match="max completion tokens cannot be below p90"):
        _profile(completion_tokens_p90=300, completion_tokens_max=200)


def test_stop_reasons_cannot_exceed_the_rollouts_they_describe():
    with pytest.raises(ValueError, match="cannot exceed completed rollouts"):
        _profile(completed_rollouts=10, truncated_rollouts=6, eos_rollouts=6)


def test_negative_and_nonfinite_rates_are_rejected():
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _profile(generation_seconds_per_completion=-1.0)
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _profile(completion_tokens_mean=float("nan"))
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _profile(reward_seconds_per_completion=float("inf"))


def test_truncation_rate_reports_the_share_that_hit_the_cap():
    assert (
        _profile(completed_rollouts=16, truncated_rollouts=4, eos_rollouts=12).truncation_rate
        == 0.25
    )
    # a profile with nothing completed must not divide by zero on the way to a verdict
    assert (
        _profile(
            completed_rollouts=0, truncated_rollouts=0, eos_rollouts=0, failed_rollouts=0
        ).truncation_rate
        == 0.0
    )


# --- the trust verdict: a profile can complete and still be unquotable ------------------------


def test_a_healthy_profile_is_trustworthy():
    assert _profile().trustworthy(now=NOW) == (True, "")


def test_a_thin_sample_is_refused_even_though_nothing_failed():
    ok, reason = _profile(
        sampled_prompts=3, completed_rollouts=3, truncated_rollouts=0, eos_rollouts=3
    ).trustworthy(now=NOW)
    assert ok is False
    assert "below the 8 needed" in reason


def test_a_sample_dominated_by_failures_is_refused():
    ok, reason = _profile(
        completed_rollouts=9, failed_rollouts=9, truncated_rollouts=1, eos_rollouts=8
    ).trustworthy(now=NOW)
    assert ok is False
    assert "failure path" in reason


def test_all_empty_completions_are_refused():
    ok, reason = _profile(
        completion_tokens_mean=0.0,
        completion_tokens_p50=0,
        completion_tokens_p90=0,
        completion_tokens_max=0,
        truncated_rollouts=0,
        eos_rollouts=16,
    ).trustworthy(now=NOW)
    assert ok is False
    assert "empty" in reason


def test_measured_latency_ages_out_but_the_shape_does_not():
    profile = _profile()
    fresh_ok, _ = profile.trustworthy(now=NOW + ROLLOUT_LATENCY_MAX_AGE_S - 60)
    stale_ok, reason = profile.trustworthy(now=NOW + ROLLOUT_LATENCY_MAX_AGE_S + 60)
    assert fresh_ok is True
    assert stale_ok is False
    assert "re-measured" in reason
    # the structural measurement is unchanged by the passage of time: only the verdict moved
    assert profile.content_digest == _profile().content_digest


def test_an_unstamped_profile_is_never_trusted():
    # measured_at defaults to 0.0 on a recomputation; that must not read as "measured at the epoch
    # and therefore infinitely stale in a way someone might clamp", nor as fresh.
    ok, reason = _profile(measured_at=0.0).trustworthy(now=NOW)
    assert ok is False
    assert "re-measured" in reason


# --- keying -----------------------------------------------------------------------------------


def test_the_key_excludes_the_training_horizon():
    """A short and a long run over the same environment share one profile and one charge."""
    payload = rollout_profile_input_payload(
        _Spec(), tokenizer_revision="tok-rev", producer_version="1.0.87"
    )
    flat = repr(payload)
    assert "max_completion_tokens" in flat
    assert "group_size" in flat
    for horizon in ("epochs", "max_steps", "max_examples"):
        assert horizon not in flat


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("max_completion_tokens", 512),
        ("max_context_tokens", 4096),
        ("group_size", 8),
        ("teacher_model", "parasail-glm-52"),
        # an unset temperature is not the same distribution as an explicitly chosen one, so
        # both the None -> value transition and a value -> value change must rekey.
        ("temperature", 0.7),
    ],
)
def test_generation_settings_change_the_key(attr, value):
    before = rollout_profile_input_digest(
        _Spec(), tokenizer_revision="tok-rev", producer_version="1.0.87"
    )

    class _Changed(_Spec):
        train = type("T", (_Train,), {attr: value})()

    after = rollout_profile_input_digest(
        _Changed(), tokenizer_revision="tok-rev", producer_version="1.0.87"
    )
    assert before != after


@pytest.mark.parametrize(
    ("horizon", "value"), [("epochs", 9), ("max_steps", 999), ("max_examples", 7)]
)
def test_horizon_changes_do_not_change_the_key(horizon, value):
    before = rollout_profile_input_digest(
        _Spec(), tokenizer_revision="tok-rev", producer_version="1.0.87"
    )

    class _Changed(_Spec):
        train = type("T", (_Train,), {horizon: value})()

    after = rollout_profile_input_digest(
        _Changed(), tokenizer_revision="tok-rev", producer_version="1.0.87"
    )
    assert before == after


def test_two_explicit_temperatures_do_not_share_a_profile():
    """The None -> value case is covered above; this pins value -> value, which a naive
    truthiness check on temperature would collapse into one key."""

    def digest_at(temp):
        class _At(_Spec):
            train = type("T", (_Train,), {"temperature": temp})()

        return rollout_profile_input_digest(
            _At(), tokenizer_revision="tok-rev", producer_version="1.0.87"
        )

    assert digest_at(0.7) != digest_at(1.2)
    assert digest_at(0.7) == digest_at(0.7)


def test_run_id_requires_a_real_digest():
    assert rollout_profile_run_id(DIGEST) == f"profile-rollout-{DIGEST}"
    for bad in ("", "z" * 64, "abc", DIGEST.upper()):
        with pytest.raises(ValueError, match="sha256 hex digest"):
            rollout_profile_run_id(bad)


# --- require_matching_rollout_profile: identity AND trust, in one place ------------------------


def test_require_returns_a_matching_trustworthy_profile():
    profile = _profile()
    got = require_matching_rollout_profile(
        profile.to_dict(),
        input_digest=DIGEST,
        producer_version="1.0.87",
        tokenizer_revision="tok-rev",
        now=NOW,
    )
    assert got == profile


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"input_digest": "b" * 64}, "input digest does not match"),
        ({"producer_version": "9.9.9"}, "producer version does not match"),
        ({"tokenizer_revision": "other"}, "tokenizer revision does not match"),
    ],
)
def test_require_rejects_a_profile_for_a_different_spec(kwargs, expected):
    call = {
        "input_digest": DIGEST,
        "producer_version": "1.0.87",
        "tokenizer_revision": "tok-rev",
        "now": NOW,
    }
    call.update(kwargs)
    with pytest.raises(WorkloadProfileMismatch, match=expected):
        require_matching_rollout_profile(_profile().to_dict(), **call)


def test_require_refuses_a_matching_but_untrustworthy_profile():
    """Identity and trust are checked together: matching this spec exactly is not enough."""
    thin = _profile(sampled_prompts=2, completed_rollouts=2, truncated_rollouts=0, eos_rollouts=2)
    with pytest.raises(WorkloadProfileMismatch, match="not trustworthy"):
        require_matching_rollout_profile(
            thin.to_dict(),
            input_digest=DIGEST,
            producer_version="1.0.87",
            tokenizer_revision="tok-rev",
            now=NOW,
        )


def test_require_refuses_a_stale_profile():
    with pytest.raises(WorkloadProfileMismatch, match="not trustworthy"):
        require_matching_rollout_profile(
            _profile().to_dict(),
            input_digest=DIGEST,
            producer_version="1.0.87",
            tokenizer_revision="tok-rev",
            now=NOW + ROLLOUT_LATENCY_MAX_AGE_S + 1,
        )


def test_require_rejects_malformed_input_as_a_mismatch_not_a_valueerror():
    # callers distinguish "retry the infrastructure" from "this identity will never resolve"
    with pytest.raises(WorkloadProfileMismatch):
        require_matching_rollout_profile(
            {"nope": 1},
            input_digest=DIGEST,
            producer_version="1.0.87",
            tokenizer_revision="tok-rev",
            now=NOW,
        )


def test_the_artifact_carries_no_raw_content():
    """Aggregates and provenance only: no prompts, completions, token ids, or credentials."""
    raw = _profile().to_dict()
    for key, value in raw.items():
        assert not isinstance(value, (list, dict)), f"{key} carries a container"
    assert not any("prompt_text" in k or "completion_text" in k or "token_ids" in k for k in raw)
