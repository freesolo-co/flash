"""Regression test: Flash train endpoints must be uniquely named per run.

Bug: every run created a Flash endpoint with the fixed name ``flash-<gpu>``. When a
prior run's endpoint was still registered, runpod_flash's ``get_or_deploy_resource`` tried
to *update* it and RunPod rejected it with ``GraphQL errors: Template name must be unique``,
so back-to-back runs (e.g. a sequential benchmark) failed to launch. There is no endpoint
GC/reuse, so the fix is a per-run unique endpoint name.
"""

from __future__ import annotations

import pytest

from flash.providers.runpod.serverless.endpoints import (
    _endpoint_name_matches_run,
    _run_suffix,
    attempt_suffix,
    endpoint_name,
)


def test_endpoint_name_default():
    assert endpoint_name("RTX 5090") == "flash-5090"
    assert endpoint_name("RTX 4090") == "flash-4090"


def test_endpoint_name_unique_per_run_and_attempt():
    run_a = "flash-1780837378-c220526e"
    run_b = "flash-1780840000-deadbeef"
    a0 = endpoint_name("RTX 5090", attempt_suffix(run_a, 0))
    a1 = endpoint_name("RTX 5090", attempt_suffix(run_a, 1))
    b0 = endpoint_name("RTX 5090", attempt_suffix(run_b, 0))
    assert a0.endswith("-a0")
    assert a1.endswith("-a1")
    assert len({a0, a1, b0}) == 3
    assert a0 == endpoint_name("RTX 5090", attempt_suffix(run_a, 0))
    c = endpoint_name("RTX 5090", attempt_suffix("val-9b-grpo-s4096-a100", 0))
    d = endpoint_name("RTX 5090", attempt_suffix("mr-4b-grpo-s8192-a100", 0))
    assert c != d


def test_endpoint_name_sanitizes_suffix():
    # only alnum/hyphen survive; bounded length
    name = endpoint_name("RTX 5090", "weird/sufx with spaces!!")
    assert name.startswith("flash-5090-")
    tail = name.rsplit("-", 1)[-1]
    assert tail.isalnum()
    assert len(tail) <= 24


def test_attempt_suffix_is_bounded_and_old_names_do_not_match():
    run_id = "flash-" + "x" * 200
    suffix = attempt_suffix(run_id, 987654321)
    assert suffix.endswith("-a987654321")
    assert len(suffix) <= 24
    target = endpoint_name("RTX 5090", _run_suffix(run_id))
    assert _endpoint_name_matches_run(endpoint_name("RTX 5090", suffix), target)
    assert not _endpoint_name_matches_run(target, target)
    assert not _endpoint_name_matches_run(f"{target}r1", target)
    with pytest.raises(ValueError, match="exceeds the endpoint name budget"):
        attempt_suffix(run_id, 999999999999)


def test_run_suffix_none_safe():
    assert _run_suffix(None) is None
    assert endpoint_name("RTX 5090", _run_suffix(None)) == "flash-5090"
