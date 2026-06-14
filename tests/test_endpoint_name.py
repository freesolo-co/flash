"""Regression test: Flash train endpoints must be uniquely named per run.

Bug: every run created a Flash endpoint with the fixed name ``autoslm-train-<gpu>``. When a
prior run's endpoint was still registered, runpod_flash's ``get_or_deploy_resource`` tried
to *update* it and RunPod rejected it with ``GraphQL errors: Template name must be unique``,
so back-to-back runs (e.g. a sequential benchmark) failed to launch. There is no endpoint
GC/reuse, so the fix is a per-run unique endpoint name.
"""

from autoslm.flash.train import _run_suffix, endpoint_name


def test_endpoint_name_default():
    assert endpoint_name("RTX 5090") == "autoslm-train-5090"
    assert endpoint_name("RTX 4090") == "autoslm-train-4090"


def test_endpoint_name_unique_per_run():
    a = endpoint_name("RTX 5090", _run_suffix("autoslm-1780837378-c220526e"))
    b = endpoint_name("RTX 5090", _run_suffix("autoslm-1780840000-deadbeef"))
    assert a == "autoslm-train-5090-c220526e"
    assert b == "autoslm-train-5090-deadbeef"
    assert a != b, "different runs must get different endpoint names (no template collision)"


def test_endpoint_name_sanitizes_suffix():
    # only alnum/hyphen survive; bounded length
    name = endpoint_name("RTX 5090", "weird/sufx with spaces!!")
    assert name.startswith("autoslm-train-5090-")
    tail = name.rsplit("-", 1)[-1]
    assert tail.isalnum()
    assert len(tail) <= 24


def test_run_suffix_none_safe():
    assert _run_suffix(None) is None
    assert endpoint_name("RTX 5090", _run_suffix(None)) == "autoslm-train-5090"
