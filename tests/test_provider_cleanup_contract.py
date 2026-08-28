from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

from flash.providers.core.capabilities import (
    CleanupOutcome,
    CleanupResult,
    ProviderCapabilities,
    confirm_run_absent,
    is_cleanup_confirmed,
    sweep_orphans,
)


class _StringSpoof(str):
    __slots__ = ()


def test_cleanup_result_confirmation_is_explicit() -> None:
    deleted = CleanupResult(CleanupOutcome.DELETED, confirmed_deleted_ids=("i-1",))
    absent = CleanupResult(CleanupOutcome.ABSENT)

    assert deleted.confirmed is True
    assert deleted.deleted_count == 1
    assert absent.confirmed is True
    for result in (
        CleanupResult(CleanupOutcome.PRESENT, surviving_ids=("i-2",)),
        CleanupResult(CleanupOutcome.UNCONFIRMED, unresolved_ids=("i-3",)),
        CleanupResult(CleanupOutcome.RETRYABLE),
        CleanupResult(CleanupOutcome.UNSUPPORTED),
    ):
        assert result.confirmed is False


def test_cleanup_result_defines_no_boolean_protocol() -> None:
    assert "__bool__" not in CleanupResult.__dict__
    assert CleanupResult(CleanupOutcome.DELETED).confirmed is True
    assert CleanupResult(CleanupOutcome.RETRYABLE).confirmed is False


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param("deleted", id="strenum-equal-string"),
        pytest.param("unknown", id="unknown-string"),
        pytest.param(None, id="none"),
        pytest.param(True, id="bool"),
        pytest.param(1, id="integer"),
        pytest.param(object(), id="object"),
    ],
)
def test_cleanup_result_rejects_non_enum_outcomes(outcome) -> None:
    with pytest.raises(TypeError, match="actual CleanupOutcome member"):
        CleanupResult(outcome)


def test_cleanup_result_rejects_magicmock_enum_spoof() -> None:
    spoofed = MagicMock(spec=CleanupOutcome.DELETED)

    assert isinstance(spoofed, CleanupOutcome)
    with pytest.raises(TypeError, match="actual CleanupOutcome member"):
        CleanupResult(spoofed)


@pytest.mark.parametrize(
    "field",
    ["confirmed_deleted_ids", "surviving_ids", "unresolved_ids"],
)
@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param(" padded ", id="surrounding-whitespace"),
        pytest.param(_StringSpoof("spoofed"), id="string-spoof"),
        pytest.param(True, id="bool"),
        pytest.param(1, id="integer"),
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param({}, id="dict"),
        pytest.param(set(), id="set"),
        pytest.param(("nested",), id="tuple"),
    ],
)
def test_cleanup_result_rejects_invalid_evidence_entries(field, entry) -> None:
    kwargs = {field: (entry,)}
    with pytest.raises((TypeError, ValueError), match=field):
        CleanupResult(CleanupOutcome.RETRYABLE, **kwargs)


@pytest.mark.parametrize(
    "field",
    ["confirmed_deleted_ids", "surviving_ids", "unresolved_ids"],
)
def test_cleanup_result_rejects_duplicate_evidence(field) -> None:
    with pytest.raises(ValueError, match="duplicate resource ids"):
        CleanupResult(CleanupOutcome.RETRYABLE, **{field: ("same", "same")})


@pytest.mark.parametrize(
    "collection",
    [
        pytest.param("resource", id="plain-string"),
        pytest.param(True, id="bool"),
        pytest.param(1, id="integer"),
        pytest.param({"resource"}, id="set"),
        pytest.param({"resource": True}, id="dict"),
        pytest.param(object(), id="object"),
    ],
)
def test_cleanup_result_rejects_invalid_evidence_collections(collection) -> None:
    with pytest.raises(TypeError, match="list or tuple"):
        CleanupResult(CleanupOutcome.RETRYABLE, unresolved_ids=collection)


def test_confirmed_revalidates_a_forged_cleanup_result() -> None:
    forged = object.__new__(CleanupResult)
    object.__setattr__(forged, "outcome", "deleted")
    object.__setattr__(forged, "confirmed_deleted_ids", ())
    object.__setattr__(forged, "surviving_ids", None)
    object.__setattr__(forged, "unresolved_ids", None)

    with pytest.raises(TypeError, match="actual CleanupOutcome member"):
        _ = forged.confirmed
    assert is_cleanup_confirmed(forged) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"confirmed_deleted_ids": ("same",), "surviving_ids": ("same",)},
            id="deleted-surviving",
        ),
        pytest.param(
            {"confirmed_deleted_ids": ("same",), "unresolved_ids": ("same",)},
            id="deleted-unresolved",
        ),
        pytest.param(
            {"surviving_ids": ("same",), "unresolved_ids": ("same",)},
            id="surviving-unresolved",
        ),
    ],
)
def test_cleanup_result_rejects_cross_category_contradictions(kwargs) -> None:
    with pytest.raises(ValueError, match="cannot mark one resource"):
        CleanupResult(CleanupOutcome.UNCONFIRMED, **kwargs)


@pytest.mark.parametrize(
    ("outcome", "kwargs"),
    [
        pytest.param(CleanupOutcome.DELETED, {}, id="deleted-without-id"),
        pytest.param(
            CleanupOutcome.DELETED,
            {"confirmed_deleted_ids": ["i-1"]},
            id="deleted-with-id",
        ),
        pytest.param(CleanupOutcome.ABSENT, {}, id="absent"),
        pytest.param(
            CleanupOutcome.PRESENT,
            {"surviving_ids": ["i-2"]},
            id="present",
        ),
        pytest.param(
            CleanupOutcome.UNCONFIRMED,
            {"unresolved_ids": ["i-3"]},
            id="unconfirmed",
        ),
        pytest.param(
            CleanupOutcome.UNCONFIRMED,
            {"confirmed_deleted_ids": ["i-1"], "unresolved_ids": ["i-3"]},
            id="partial-deletion",
        ),
        pytest.param(CleanupOutcome.RETRYABLE, {}, id="retryable"),
        pytest.param(
            CleanupOutcome.RETRYABLE,
            {"unresolved_ids": ["i-3"]},
            id="retryable-with-target",
        ),
        pytest.param(CleanupOutcome.UNSUPPORTED, {}, id="unsupported"),
        pytest.param(
            CleanupOutcome.UNSUPPORTED,
            {"unresolved_ids": ["run-1"]},
            id="unsupported-with-run",
        ),
    ],
)
def test_cleanup_result_accepts_consistent_evidence(outcome, kwargs) -> None:
    result = CleanupResult(outcome, **kwargs)

    assert result.outcome is outcome
    assert isinstance(result.confirmed_deleted_ids, tuple)
    assert result.surviving_ids is None or isinstance(result.surviving_ids, tuple)
    assert result.unresolved_ids is None or isinstance(result.unresolved_ids, tuple)


@pytest.mark.parametrize(
    ("outcome", "kwargs"),
    [
        pytest.param(
            CleanupOutcome.DELETED,
            {"surviving_ids": ("i-1",)},
            id="deleted-with-survivor",
        ),
        pytest.param(
            CleanupOutcome.DELETED,
            {"unresolved_ids": ("i-1",)},
            id="deleted-with-unresolved",
        ),
        pytest.param(
            CleanupOutcome.ABSENT,
            {"confirmed_deleted_ids": ("i-1",)},
            id="absent-with-deleted",
        ),
        pytest.param(
            CleanupOutcome.ABSENT,
            {"surviving_ids": ("i-1",)},
            id="absent-with-survivor",
        ),
        pytest.param(
            CleanupOutcome.ABSENT,
            {"unresolved_ids": ("i-1",)},
            id="absent-with-unresolved",
        ),
        pytest.param(CleanupOutcome.PRESENT, {}, id="present-without-survivor"),
        pytest.param(
            CleanupOutcome.PRESENT,
            {"surviving_ids": ("i-1",), "confirmed_deleted_ids": ("i-2",)},
            id="present-with-deleted",
        ),
        pytest.param(
            CleanupOutcome.PRESENT,
            {"surviving_ids": ("i-1",), "unresolved_ids": ("i-2",)},
            id="present-with-unresolved",
        ),
        pytest.param(CleanupOutcome.UNCONFIRMED, {}, id="unconfirmed-without-target"),
        pytest.param(
            CleanupOutcome.UNCONFIRMED,
            {"surviving_ids": ("i-1",), "unresolved_ids": ("i-2",)},
            id="unconfirmed-with-survivor",
        ),
        pytest.param(
            CleanupOutcome.UNCONFIRMED,
            {"confirmed_deleted_ids": ("i-1",), "unresolved_ids": ("i-1",)},
            id="deleted-and-unresolved-same-id",
        ),
        pytest.param(
            CleanupOutcome.RETRYABLE,
            {"confirmed_deleted_ids": ("i-1",)},
            id="retryable-with-deleted",
        ),
        pytest.param(
            CleanupOutcome.RETRYABLE,
            {"surviving_ids": ("i-1",)},
            id="retryable-with-survivor",
        ),
        pytest.param(
            CleanupOutcome.UNSUPPORTED,
            {"confirmed_deleted_ids": ("i-1",)},
            id="unsupported-with-deleted",
        ),
        pytest.param(
            CleanupOutcome.UNSUPPORTED,
            {"surviving_ids": ("i-1",)},
            id="unsupported-with-survivor",
        ),
    ],
)
def test_cleanup_result_rejects_contradictory_evidence(outcome, kwargs) -> None:
    with pytest.raises(ValueError, match=r"cleanup (outcomes|evidence)"):
        CleanupResult(outcome, **kwargs)


def test_cleanup_result_normalizes_ids_and_is_frozen() -> None:
    result = CleanupResult(
        CleanupOutcome.UNCONFIRMED,
        confirmed_deleted_ids=["i-1"],
        unresolved_ids=["i-3"],
    )
    present = CleanupResult(CleanupOutcome.PRESENT, surviving_ids=["i-2"])

    assert result.confirmed_deleted_ids == ("i-1",)
    assert present.surviving_ids == ("i-2",)
    assert result.unresolved_ids == ("i-3",)
    with pytest.raises(FrozenInstanceError):
        result.outcome = CleanupOutcome.DELETED


def test_capability_helpers_invoke_callbacks_or_report_unsupported() -> None:
    seen: list[object] = []

    def confirm(run_id: str) -> CleanupResult:
        seen.append(run_id)
        return CleanupResult(CleanupOutcome.ABSENT)

    def sweep(active, known, should_stop=None) -> CleanupResult:
        seen.extend((active, known))
        return CleanupResult(CleanupOutcome.DELETED, confirmed_deleted_ids=("7",))

    capabilities = ProviderCapabilities(False, True, confirm, sweep)
    active = {"live"}
    known = {"live", "done"}

    assert confirm_run_absent(capabilities, "run-1").outcome is CleanupOutcome.ABSENT
    result = sweep_orphans(capabilities, active, known)
    assert result.outcome is CleanupOutcome.DELETED
    assert result.deleted_count == 1
    assert seen == ["run-1", active, known]

    unsupported = ProviderCapabilities(False, False, None, None)
    assert confirm_run_absent(unsupported, "run-2").outcome is CleanupOutcome.UNSUPPORTED
    assert sweep_orphans(unsupported).outcome is CleanupOutcome.UNSUPPORTED


def test_capability_helpers_reject_invalid_callback_objects() -> None:
    invalid = ProviderCapabilities(False, True, lambda _run_id: "absent", lambda *_args: object())

    confirmation = confirm_run_absent(invalid, "run-invalid")
    sweep = sweep_orphans(invalid)

    assert confirmation.outcome is CleanupOutcome.RETRYABLE
    assert confirmation.unresolved_ids == ("run-invalid",)
    assert sweep.outcome is CleanupOutcome.RETRYABLE


def test_capability_helpers_contain_noncallable_callbacks() -> None:
    capabilities = ProviderCapabilities(False, True, object(), object())

    confirmation = confirm_run_absent(capabilities, "run-noncallable")
    sweep = sweep_orphans(capabilities)

    assert confirmation.outcome is CleanupOutcome.RETRYABLE
    assert confirmation.unresolved_ids == ("run-noncallable",)
    assert sweep.outcome is CleanupOutcome.RETRYABLE


def test_capability_helpers_contain_raising_callbacks() -> None:
    def raise_confirmation(_run_id: str) -> CleanupResult:
        raise RuntimeError("confirmation failed")

    reached: list[bool] = []

    def raise_sweep(_active, _known, _should_stop=None) -> CleanupResult:
        # the dispatcher's broad handler turns a STALE signature's TypeError into RETRYABLE too,
        # so record that the body actually ran: otherwise this asserts nothing about containment.
        reached.append(True)
        raise RuntimeError("sweep failed")

    capabilities = ProviderCapabilities(False, True, raise_confirmation, raise_sweep)

    confirmation = confirm_run_absent(capabilities, "run-raising")
    sweep = sweep_orphans(capabilities)

    assert confirmation.outcome is CleanupOutcome.RETRYABLE
    assert confirmation.unresolved_ids == ("run-raising",)
    assert sweep.outcome is CleanupOutcome.RETRYABLE
    assert reached == [True]


def test_capability_helpers_revalidate_mutated_callback_results() -> None:
    forged = CleanupResult(CleanupOutcome.ABSENT)
    object.__setattr__(forged, "unresolved_ids", ("forged-resource",))
    capabilities = ProviderCapabilities(False, True, lambda _run_id: forged, lambda *_args: forged)

    confirmation = confirm_run_absent(capabilities, "run-forged")
    sweep = sweep_orphans(capabilities)

    assert confirmation.outcome is CleanupOutcome.RETRYABLE
    assert confirmation.unresolved_ids == ("run-forged",)
    assert sweep.outcome is CleanupOutcome.RETRYABLE


def test_callback_produced_unsupported_is_rejected() -> None:
    capabilities = ProviderCapabilities(
        False,
        True,
        lambda run_id: CleanupResult(CleanupOutcome.UNSUPPORTED, unresolved_ids=(run_id,)),
        lambda *_args: CleanupResult(CleanupOutcome.UNSUPPORTED),
    )

    confirmation = confirm_run_absent(capabilities, "run-supported")
    sweep = sweep_orphans(capabilities)

    assert confirmation.outcome is CleanupOutcome.RETRYABLE
    assert confirmation.unresolved_ids == ("run-supported",)
    assert sweep.outcome is CleanupOutcome.RETRYABLE


def test_unsupported_is_emitted_only_when_capability_is_absent() -> None:
    unsupported = ProviderCapabilities(False, False, None, None)

    confirmation = confirm_run_absent(unsupported, "run-unsupported")
    sweep = sweep_orphans(unsupported)

    assert confirmation.outcome is CleanupOutcome.UNSUPPORTED
    assert confirmation.unresolved_ids == ("run-unsupported",)
    assert sweep.outcome is CleanupOutcome.UNSUPPORTED


@pytest.mark.parametrize(
    ("provider_factory", "jobs_module"),
    [
        (
            "flash.providers.lambda_.execution.provider:LambdaProvider",
            "flash.providers.lambda_.jobs",
        ),
        ("flash.providers.vast.execution.provider:VastProvider", "flash.providers.vast.jobs"),
    ],
)
def test_the_stop_callback_reaches_each_real_provider_sweep(
    provider_factory, jobs_module, monkeypatch
) -> None:
    """Identity, not truthiness: the exact callback the lifespan owns must arrive at the jobs-level
    sweep through the real facade. A test that only asserts the sweep ran stays green when a
    wrapper silently drops ``should_stop=should_stop``, which is precisely how a shutdown signal
    goes missing and destructive teardowns keep running after the server was told to stop."""
    import importlib

    module_path, _, attr = provider_factory.partition(":")
    provider = getattr(importlib.import_module(module_path), attr)()
    jobs = importlib.import_module(jobs_module)
    seen: list[object] = []
    monkeypatch.setattr(
        jobs,
        "sweep_orphans",
        lambda **kwargs: (
            seen.append(kwargs.get("should_stop"))
            or CleanupResult(CleanupOutcome.DELETED, confirmed_deleted_ids=("i-9",))
        ),
    )

    def stop() -> bool:
        return False

    result = sweep_orphans(
        provider.capabilities, active_labels={"live"}, known_labels={"live"}, should_stop=stop
    )

    assert result.confirmed_deleted_ids == ("i-9",)
    assert len(seen) == 1
    assert seen[0] is stop  # the same object, not a re-wrapped or defaulted stand-in


def test_a_provider_that_ignores_the_stop_callback_is_detected() -> None:
    """Sabotage guard for the test above: a facade that accepts ``should_stop`` and forwards
    ``None`` must be distinguishable from one that forwards it, or the identity assertion above
    would be satisfied by any provider that merely runs."""
    seen: list[object] = []
    dropping = ProviderCapabilities(
        False,
        True,
        None,
        lambda active, known, should_stop=None: (
            seen.append(None)
            or CleanupResult(CleanupOutcome.DELETED, confirmed_deleted_ids=("i-9",))
        ),
    )

    def stop() -> bool:
        return False

    sweep_orphans(dropping, should_stop=stop)

    assert seen == [None]
    assert seen[0] is not stop
