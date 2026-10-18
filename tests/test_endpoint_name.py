"""Flash train endpoints are named for one run's one attempt.

Bug this file started from: every run created a Flash endpoint with the fixed name
``flash-<gpu>``. When a prior run's endpoint was still registered, runpod_flash's
``get_or_deploy_resource`` tried to *update* it and RunPod rejected it with ``GraphQL errors:
Template name must be unique``, so back-to-back runs failed to launch. The fix was a per-run
unique endpoint name.

A run owns one seed and may execute several attempts on different hosts, so the name identifies
the run plus the attempt running on it. Every attempt carries an explicit ``-a<n>`` ordinal,
attempt zero included: the retired grammar left attempt zero unsuffixed and marked later attempts
``r<n>``, which made a retry ordinal an identity and left the base name ambiguous between "the
run" and "its first attempt".
"""

from __future__ import annotations

import pytest

from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.providers.runpod.serverless.naming import (
    attempt_suffix,
    endpoint_name,
    endpoint_name_matches_run,
    run_suffix,
    run_target_of,
    select_endpoint_resources,
)

_RUN = "flash-1780837378-c220526e"


def test_endpoint_name_default():
    assert endpoint_name("RTX 5090") == "flash-5090"
    assert endpoint_name("RTX 4090") == "flash-4090"


def test_endpoint_name_unique_per_run():
    a = endpoint_name("RTX 5090", run_suffix(_RUN))
    b = endpoint_name("RTX 5090", run_suffix("flash-1780840000-deadbeef"))
    assert a.startswith("flash-5090-")
    assert a != b, "different runs must get different endpoint names (no template collision)"
    # deterministic: same run_id -> same name (an attempt reuses its own run's target)
    assert a == endpoint_name("RTX 5090", run_suffix(_RUN))
    # run_ids sharing a trailing segment (e.g. both end in the card name) must NOT collide onto
    # one endpoint -- the old split("-")[-1] bug caused stale-config reuse.
    c = endpoint_name("RTX 5090", run_suffix("val-9b-grpo-s4096-a100"))
    d = endpoint_name("RTX 5090", run_suffix("mr-4b-grpo-s8192-a100"))
    assert c != d, "run_ids sharing a tail must still get distinct endpoints"


def test_endpoint_name_sanitizes_suffix():
    name = endpoint_name("RTX 5090", "weird/sufx with spaces!!")
    assert name == "flash-5090-weirdsufxwithspaces"


def test_endpoint_name_raises_rather_than_truncating():
    """Clipping the suffix could drop the attempt ordinal, so two attempts would share a name."""
    with pytest.raises(ValueError, match="provider name budget"):
        endpoint_name("RTX 5090", "a" * 64)


def test_run_suffix_none_safe():
    assert run_suffix(None) is None
    assert endpoint_name("RTX 5090", run_suffix(None)) == "flash-5090"


def test_every_attempt_carries_an_explicit_ordinal():
    """Attempt zero is suffixed too, so no name is ambiguous between a run and its first attempt."""
    zero = attempt_suffix(_RUN, 0)
    one = attempt_suffix(_RUN, 1)
    assert zero.endswith("-a0")
    assert one.endswith("-a1")
    assert zero != one
    assert zero.removesuffix("-a0") == one.removesuffix("-a1") == run_suffix(_RUN)


def test_attempt_suffix_rejects_an_invalid_attempt():
    for bad in (-1, MAX_ATTEMPT_ID + 1, True, 1.0, "1", None):
        with pytest.raises(ValueError, match="attempt identity is invalid"):
            attempt_suffix(_RUN, bad)


def test_attempt_suffix_accepts_the_largest_supported_ordinal():
    suffix = attempt_suffix(_RUN, MAX_ATTEMPT_ID)
    assert suffix.endswith(f"-a{MAX_ATTEMPT_ID}")
    assert endpoint_name("RTX 5090", suffix).startswith("flash-5090-")


def test_run_target_recovers_the_run_across_its_attempts():
    """One target names the run, so it covers every attempt rather than only the persisted one."""
    target = endpoint_name("RTX 5090", run_suffix(_RUN))
    for attempt in (0, 1, 17, MAX_ATTEMPT_ID):
        name = endpoint_name("RTX 5090", attempt_suffix(_RUN, attempt))
        assert run_target_of(name) == target
        assert run_target_of(f"live-{name}") == target, "the SDK's live- prefix must not matter"
        assert endpoint_name_matches_run(name, target)


def test_run_target_rejects_a_name_that_carries_no_attempt():
    """A bare run target names no attempt, so it must not read as one of that run's endpoints."""
    target = endpoint_name("RTX 5090", run_suffix(_RUN))
    assert run_target_of(target) is None
    assert not endpoint_name_matches_run(target, target)


def test_run_target_rejects_a_padded_ordinal():
    """ "-a007" reads as a different attempt than "-a7" while naming the same one."""
    assert run_target_of("flash-5090-abc-a007") is None
    assert run_target_of("flash-5090-abc-a7") == "flash-5090-abc"


def test_retired_retry_grammar_does_not_match():
    """The old ``r<n>`` retry suffix is not an identity and must not be reaped as one."""
    target = endpoint_name("RTX 5090", run_suffix(_RUN))
    assert not endpoint_name_matches_run(target + "r1", target)
    assert not endpoint_name_matches_run(target + "r12", target)


def test_a_neighbouring_run_is_never_matched():
    mine = endpoint_name("RTX 5090", run_suffix(_RUN))
    theirs = endpoint_name("RTX 5090", run_suffix("flash-1780840000-deadbeef"))
    assert not endpoint_name_matches_run(theirs + "-a0", mine)


def test_select_endpoint_resources_picks_every_attempt_of_one_run():
    target = endpoint_name("RTX 5090", run_suffix(_RUN))
    other = endpoint_name("RTX 5090", run_suffix("flash-1780840000-deadbeef"))

    class _Resource:
        def __init__(self, name):
            self.name = name

    resources = {
        "mine-a0": _Resource(f"{target}-a0"),
        "mine-a1": _Resource(f"live-{target}-a1"),
        "mine-bare": _Resource(target),
        "mine-retired": _Resource(f"{target}r1"),
        "theirs": _Resource(f"{other}-a0"),
    }
    assert sorted(select_endpoint_resources(resources, target)) == ["mine-a0", "mine-a1"]
    assert select_endpoint_resources(resources, "") == []
