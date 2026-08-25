"""The conformance suite's opt-in gate.
The suite itself only runs against a live backend, so nothing in the offline run exercises the
decision it makes about incomplete input. That decision is the whole reason a conformance run can
be trusted: a green exit has to mean the contract was checked, not that the checks were skipped.
"""

from __future__ import annotations

import types

import pytest

from tests.conftest import PLACEHOLDER_HF_REVISION
from tests.serving_conformance import conftest as gate

_ADAPTER_SOURCE = gate.adapter_source._get_wrapped_function()
_FAILED = pytest.fail.Exception
_COMPLETE = {
    "--conformance-repo": "acme/artifacts",
    "--conformance-subfolder": "sft/run-abc/adapter",
    "--conformance-base-model": "Qwen/Qwen3.5-9B",
    "--conformance-hf-revision": "8f2c1b0e5d4a39c7b6e2f014a8d35c9b7e10426f",
    "--conformance-repo-type": "dataset",
}


def _request(options: dict, monkeypatch) -> types.SimpleNamespace:
    for name in (
        "FLASH_CONFORMANCE_REPO",
        "FLASH_CONFORMANCE_SUBFOLDER",
        "FLASH_CONFORMANCE_BASE_MODEL",
        "FLASH_CONFORMANCE_HF_REVISION",
        "FLASH_CONFORMANCE_REPO_TYPE",
    ):
        monkeypatch.delenv(name, raising=False)
    return types.SimpleNamespace(
        config=types.SimpleNamespace(getoption=lambda flag: options.get(flag))
    )


@pytest.mark.parametrize("missing", sorted(set(_COMPLETE) - {"--conformance-repo-type"}))
def test_an_incomplete_conformance_run_fails_rather_than_skipping(missing, monkeypatch):
    options = {flag: value for flag, value in _COMPLETE.items() if flag != missing}
    try:
        _ADAPTER_SOURCE(_request(options, monkeypatch))
    except BaseException as exc:  # classified below, never swallowed
        raised = exc
    else:
        raised = None
    assert isinstance(raised, _FAILED), (
        f"a conformance run missing {missing} raised {type(raised).__name__}, not Failed. A skip "
        f"lets the enabled run exit green having checked only /healthz."
    )
    assert missing in str(raised), "the failure must name the option that was left out"


def test_a_placeholder_commit_is_refused(monkeypatch):
    options = {**_COMPLETE, "--conformance-hf-revision": PLACEHOLDER_HF_REVISION}
    try:
        _ADAPTER_SOURCE(_request(options, monkeypatch))
    except BaseException as exc:  # classified below, never swallowed
        raised = exc
    else:
        raised = None
    assert isinstance(raised, _FAILED), (
        f"the placeholder commit raised {type(raised).__name__}, not Failed"
    )


def test_a_complete_invocation_is_accepted(monkeypatch):
    source = _ADAPTER_SOURCE(_request(dict(_COMPLETE), monkeypatch))
    assert source["repo_id"] == "acme/artifacts"
    assert source["hf_revision"] == _COMPLETE["--conformance-hf-revision"]
    assert source["repo_type"] == "dataset"


_READY_TIMEOUT = gate.ready_timeout._get_wrapped_function()


def test_the_readiness_default_tracks_the_clients_own_budget(monkeypatch):
    from flash.serve.deployment.readiness import revision_ready_budget_seconds

    for base_model in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-35B-A3B"):
        source = {"base_model": base_model}
        request = _request({"--conformance-ready-timeout": None}, monkeypatch)
        assert _READY_TIMEOUT(request, source) == revision_ready_budget_seconds(base_model)


def test_an_explicit_readiness_override_still_wins(monkeypatch):
    request = _request({"--conformance-ready-timeout": 42.0}, monkeypatch)
    assert _READY_TIMEOUT(request, {"base_model": "Qwen/Qwen3.5-9B"}) == 42.0
