import math

import pytest

from flash.benchmarks.evidence import digest
from flash.benchmarks.report import request_summary, summarize_case, timing_summary
from flash.benchmarks.schema import EVENT, Complete, Money, Quality, Request, Span


def quality(at=10.0, value=0.8, samples=10, **kwargs):
    return Quality(
        at_s=at,
        value=value,
        samples=samples,
        metric="accuracy",
        checkpoint=f"run/step-{int(at)}",
        **kwargs,
    )


def money(ledger="infrastructure_actual", amount=20.0, start=0.0, end=20.0, line="gpu"):
    return Money(
        start_s=start,
        end_s=end,
        ledger=ledger,
        amount_usd=amount,
        receipt="invoice-1",
        resource="gpu-1",
        line_item=line,
    )


def test_overlap_nested_spans_and_unmeasured_phases():
    spans = [
        Span(phase="rollout", start_s=2.0, end_s=10.0, source="worker"),
        Span(phase="rollout", start_s=3.0, end_s=5.0, source="nested"),
        Span(phase="environment_reward", start_s=6.0, end_s=12.0, source="reward worker"),
    ]
    result = timing_summary(spans, 15.0)
    assert result["phase_busy_s"]["rollout"] == 8.0
    assert result["phase_busy_s"]["training"] is None
    assert result["phase_exclusive_s"]["rollout"] == 4.0
    assert result["phase_exclusive_s"]["environment_reward"] == 2.0
    assert result["overlap_s"] == 4.0
    assert result["unattributed_s"] == 5.0
    assert (
        sum(value or 0 for value in result["phase_exclusive_s"].values())
        + result["overlap_s"]
        + result["unattributed_s"]
        == 15.0
    )


def test_first_held_out_quality_crossing_not_final_or_insufficient_samples(
    case, request_data, evidence
):
    events = [
        *evidence[:-1],
        quality(at=15.0, value=0.9),
        quality(at=5.0, value=0.99, samples=1),
        quality(at=10.0),
        quality(at=18.0, value=0.2),
        evidence[-1],
    ]
    result = summarize_case(case, request_data, events, process_ok=True)
    assert result["time_to_target_s"] == 10.0
    assert result["target_checkpoint"] == "run/step-10"
    assert [point["at_s"] for point in result["quality_curve"]] == [5.0, 10.0, 15.0, 18.0]


def test_missed_target_is_null_and_not_a_free_or_fast_result(case, request_data, evidence):
    result = summarize_case(
        case, request_data, [*evidence[:-1], quality(value=0.1), evidence[-1]], process_ok=True
    )
    assert result["valid"]
    assert not result["target_reached"]
    assert result["time_to_target_s"] is None
    assert result["money"]["infrastructure_actual"]["total_usd"] is None
    assert result["money"]["infrastructure_actual"]["to_target"] is None


def test_lower_is_better_and_zero_baseline_target(case, request_data, evidence):
    case.target.direction = "at_most"
    result = summarize_case(
        case,
        request_data,
        [*evidence[:-1], quality(at=0.0, value=0.0), evidence[-1]],
        process_ok=True,
    )
    assert result["time_to_target_s"] == 0.0


def test_actual_cost_estimate_charge_and_target_bounds_remain_separate(
    case, request_data, evidence
):
    complete = Complete(
        at_s=20.0,
        outcome="succeeded",
        complete_ledgers=["infrastructure_actual", "customer_charge"],
    )
    events = [
        *evidence[:-1],
        quality(),
        money(),
        money("customer_charge", 50.0),
        money("infrastructure_estimate", 5.0),
        complete,
    ]
    result = summarize_case(case, request_data, events, process_ok=True)["money"]
    assert result["infrastructure_actual"]["total_usd"] == 20.0
    assert result["customer_charge"]["total_usd"] == 50.0
    assert result["infrastructure_estimate"]["total_usd"] is None
    assert result["infrastructure_estimate"]["known_usd"] == 5.0
    assert result["infrastructure_actual"]["to_target"] == {
        "lower_usd": 0.0,
        "upper_usd": 20.0,
        "uniform_interval_allocation_usd": 10.0,
    }


def test_phase_cost_allocations_reconcile_overlapping_resources_and_point_charges(
    case, request_data, evidence
):
    events = [
        *evidence[:-1],
        money(),
        money(amount=10.0, line="cpu"),
        money(amount=3.0, start=7.0, end=7.0, line="fixed"),
        Span(start_s=0.0, end_s=20.0, phase="rollout", source="worker"),
        Span(start_s=10.0, end_s=20.0, phase="training", source="worker"),
        evidence[-1],
    ]
    allocation = summarize_case(case, request_data, events, process_ok=True)["money"][
        "infrastructure_actual"
    ]["phase_allocation_usd"]
    assert allocation["rollout"] == 22.5
    assert allocation["training"] == 7.5
    assert allocation["unattributed"] == 3.0
    assert sum(allocation.values()) == 33.0


def test_duplicate_receipts_rejected(case, request_data, evidence):
    with pytest.raises(ValueError, match="duplicate receipt"):
        summarize_case(
            case, request_data, [*evidence[:-1], money(), money(), evidence[-1]], process_ok=True
        )


@pytest.mark.parametrize("change", ["provenance", "config", "cache", "process"])
def test_runtime_mismatch_does_not_qualify_a_result(case, request_data, evidence, change):
    if change == "provenance":
        evidence[0].revisions = evidence[0].revisions.model_copy(update={"code": "9" * 40})
    elif change == "config":
        request_data["configuration"]["seed"] = 43
    elif change == "cache":
        evidence[1].observed = "unknown"
    result = summarize_case(case, request_data, evidence, process_ok=change != "process")
    assert not result["valid"]


def test_failures_retained_but_not_mixed_into_success_percentiles():
    requests = [
        Request(start_s=0.0, end_s=2.0, customer="alice", adapter="a", success=True, ttft_s=0.5),
        Request(start_s=0.0, end_s=0.01, customer="alice", adapter="a", success=False),
        Request(start_s=1.0, end_s=5.0, customer="bob", adapter="b", success=True),
    ]
    result = request_summary(requests)
    assert result["concurrent_customers_and_adapters_observed"]
    alice, bob = result["by_customer_adapter"]
    assert alice["latency_s"]["p50"] == 2.0
    assert alice["failures"] == 1
    assert alice["failed_latency_s"] == [0.01]
    assert bob["ttft_s"]["p95"] is None


def test_sequential_or_failed_requests_cannot_prove_concurrency():
    requests = [
        Request(start_s=0.0, end_s=2.0, customer="alice", adapter="a", success=True),
        Request(start_s=2.0, end_s=4.0, customer="bob", adapter="b", success=True),
        Request(start_s=0.0, end_s=4.0, customer="bob", adapter="b", success=False),
    ]
    assert not request_summary(requests)["concurrent_customers_and_adapters_observed"]


@pytest.mark.parametrize(
    "events",
    [
        [Complete(at_s=1.0, outcome="succeeded"), quality()],
        [quality(), Complete(at_s=1.0, outcome="succeeded")],
        [Complete(at_s=1.0, outcome="succeeded"), Complete(at_s=1.0, outcome="succeeded")],
    ],
)
def test_bad_completion_or_clock_origins_rejected(case, request_data, events):
    with pytest.raises(ValueError, match=r"complete|completion"):
        summarize_case(case, request_data, events, process_ok=True)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_quality_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        quality(value=value)


@pytest.mark.parametrize(
    "changes", [{"start_s": -1.0}, {"end_s": 0.0}, {"ttft_s": 4.0}, {"unknown": True}]
)
def test_invalid_request_evidence_rejected(changes):
    raw = {
        "kind": "request",
        "start_s": 1.0,
        "end_s": 2.0,
        "customer": "a",
        "adapter": "x",
        "success": True,
        **changes,
    }
    with pytest.raises(ValueError, match="validation error"):
        EVENT.validate_python(raw)


def test_partial_trial_has_no_success_verdict(case, request_data, evidence):
    result = summarize_case(
        case, request_data, [*evidence[:-1], quality()], process_ok=True, observed_end_s=25.0
    )
    assert not result["valid"]
    assert result["outcome"] == "incomplete"
    assert result["timing"]["wall_s"] == 25.0


def test_reconciled_zero_is_distinct_from_missing_cost(case, request_data, evidence):
    events = [
        *evidence[:-1],
        quality(),
        Complete(at_s=20.0, outcome="succeeded", complete_ledgers=["customer_charge"]),
    ]
    result = summarize_case(case, request_data, events, process_ok=True)
    assert result["money"]["customer_charge"]["total_usd"] == 0.0
    assert result["money"]["infrastructure_actual"]["total_usd"] is None
    assert not result["accounting_complete"]
    assert "allocation" in result["measurement_gaps"]
    assert "infrastructure_actual" in result["measurement_gaps"]


def test_resolved_runtime_config_retained_and_fingerprinted_separately(
    case, request_data, evidence
):
    evidence[0].resolved_configuration.parameters = {"allocated_gpu": "B200", "gpu_count": 2}
    result = summarize_case(
        case, request_data, [*evidence[:-1], quality(), evidence[-1]], process_ok=True
    )
    assert result["valid"]
    assert result["resolved_configuration_digest"] == digest(
        evidence[0].resolved_configuration.model_dump()
    )
    assert result["resolved_configuration_digest"] != digest(request_data["configuration"])
    evidence[0].resolved_configuration.seed = 43
    assert not summarize_case(
        case, request_data, [*evidence[:-1], quality(), evidence[-1]], process_ok=True
    )["valid"]
