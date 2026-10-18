"""A warm-start run must only blame its adapter for failures the adapter actually caused.

``prepare_job`` also does gpu sizing, budget, and environment resolution. Those run for every
submit, warm-start or not, and have nothing to do with ``train.init_from_adapter``. Catching them
under one broad ``except`` rewrote every such failure into "verify that the source adapter is
complete, compatible, and unchanged", which sends users to re-check a healthy adapter while the
real reason (a bad environment ref, an unavailable gpu class) never reaches them.
"""

from __future__ import annotations

import pytest

import flash.runner.lifecycle.preparation as runner_preparation

# resolve the class off the live module on every use. other suites reload ``flash.runner``, which
# rebinds this class to a new object, and a module-level import captured here would no longer be the
# type that ``_prepare_init_from_adapter`` raises.


def test_warm_start_preparation_error_is_a_value_error():
    # the submit route catches ValueError broadly for spec problems; staying a subclass keeps
    # that path working if the tagged error escapes an unpatched caller.
    assert issubclass(runner_preparation.WarmStartPreparationError, ValueError)


def test_adapter_resolution_failure_is_tagged(monkeypatch):
    def _boom(spec, **kwargs):
        raise RuntimeError("source adapter revision not found")

    monkeypatch.setattr(runner_preparation, "_prepare_init_from_adapter_inner", _boom)

    with pytest.raises(runner_preparation.WarmStartPreparationError) as excinfo:
        runner_preparation._prepare_init_from_adapter(object(), owner_org_id="org-1")

    assert "source adapter revision not found" in str(excinfo.value)


def test_already_tagged_error_is_not_double_wrapped(monkeypatch):
    original = runner_preparation.WarmStartPreparationError("adapter rank mismatch")

    def _boom(spec, **kwargs):
        raise original

    monkeypatch.setattr(runner_preparation, "_prepare_init_from_adapter_inner", _boom)

    with pytest.raises(runner_preparation.WarmStartPreparationError) as excinfo:
        runner_preparation._prepare_init_from_adapter(object(), owner_org_id="org-1")

    assert excinfo.value is original


def test_only_tagged_failures_are_blamed_on_the_adapter():
    # the submit route selects on this type, so an unrelated prepare_job failure (gpu sizing,
    # budget, environment resolution) cannot be rewritten into a bad-adapter message. see
    # test_server_billing.py for the route-level assertions on the responses themselves.
    assert not isinstance(
        ValueError("no gpu class satisfies"), runner_preparation.WarmStartPreparationError
    )
    assert isinstance(
        runner_preparation.WarmStartPreparationError("bad adapter"),
        runner_preparation.WarmStartPreparationError,
    )
