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
# `pytest.fail` raises Failed, which derives from BaseException -- `pytest.raises(Exception)` does
# NOT catch it, and the test would report the very failure it is asserting on.
_FAILED = pytest.fail.Exception

_COMPLETE = {
    "--conformance-repo": "acme/artifacts",
    "--conformance-subfolder": "sft/run-abc/adapter",
    "--conformance-base-model": "Qwen/Qwen3.5-4B",
    "--conformance-hf-revision": "8f2c1b0e5d4a39c7b6e2f014a8d35c9b7e10426f",
    "--conformance-repo-type": "dataset",
}


def _request(options: dict, monkeypatch) -> types.SimpleNamespace:
    """A pytest request whose options are `options` and whose environment is empty.

    The environment is cleared because the fixture falls back to `FLASH_CONFORMANCE_*` variables,
    and a developer who has them exported would otherwise see a different answer than CI.
    """
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
    """`--serving-url` was passed, so a missing adapter is an error, not a reason to skip.

    Skipped, the run still exits 0 having checked only /healthz and the 404s -- nothing of
    registration, readiness, activation, chat, or teardown. That green exit is worse than no run at
    all, because it is reported as the contract passing.
    """
    options = {flag: value for flag, value in _COMPLETE.items() if flag != missing}
    # Caught as BaseException and classified by hand, NOT with `pytest.raises(_FAILED)`. Both
    # `pytest.fail` and `pytest.skip` raise BaseException subclasses, so a `raises(Failed)` here
    # would let the Skipped propagate and SKIP this test -- exit 0, regression invisible, which is
    # the same false-green this test exists to prevent.
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
    """The placeholder satisfies the revision-id grammar, so only an explicit check catches it.

    Left in, registration fails minutes later on the GPU as an unresolvable commit rather than here
    as the missing argument it is.
    """
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
    """The guard above must not reject a run that supplied everything."""
    source = _ADAPTER_SOURCE(_request(dict(_COMPLETE), monkeypatch))
    assert source["repo_id"] == "acme/artifacts"
    assert source["hf_revision"] == _COMPLETE["--conformance-hf-revision"]
    assert source["repo_type"] == "dataset"
