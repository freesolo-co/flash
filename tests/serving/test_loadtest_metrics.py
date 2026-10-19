from __future__ import annotations

from flash.serving.loadtest.metrics import summarize_events


def _event(index: int, **overrides):
    event = {
        "type": "request_terminal",
        "request_id": f"request-{index}",
        "phase_name": "sustained",
        "phase_kind": "sustained",
        "target_name": "model-a" if index < 2 else "model-b",
        "profile_name": "short",
        "scheduled_ns": index * 1_000_000_000,
        "dispatch_ns": index * 1_000_000_000 + 10_000_000,
        "headers_ns": index * 1_000_000_000 + 20_000_000,
        "first_generated_ns": index * 1_000_000_000 + 30_000_000,
        "first_visible_ns": index * 1_000_000_000 + 40_000_000,
        "completed_ns": index * 1_000_000_000 + 100_000_000,
        "authored_window_seconds": 3.0,
        "outcome": "success",
        "http_status": 200,
        "error_class": None,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cached_tokens": 2,
        "retry_count": 0,
        "client_in_flight_at_dispatch": index + 1,
    }
    event.update(overrides)
    return event


def test_summary_is_deterministic_and_dimensioned() -> None:
    events = [_event(0), _event(1), _event(2)]
    first = summarize_events(events, fake=True)
    second = summarize_events(list(reversed(events)), fake=True)
    assert first == second
    assert set(first["per_phase"]) == {"sustained"}
    assert set(first["per_target"]) == {"model-a", "model-b"}
    assert set(first["per_profile"]) == {"short"}
    assert first["overall"]["counts"]["success"] == 3
    assert first["overall"]["latency"]["ttft_ms"]["p99"] == 20.0
    assert first["overall"]["usage"]["completion_tokens"] == 15
    assert first["overall"]["peak_client_in_flight"] == 3
    assert first["production_claims_allowed"] is False
    assert any("fake or test" in value for value in first["claim_limitations"])


def test_missing_usage_never_estimates_token_throughput() -> None:
    events = [
        _event(0, prompt_tokens=None, completion_tokens=None, cached_tokens=None),
        _event(1),
    ]
    usage = summarize_events(events, fake=False)["overall"]["usage"]
    assert usage["covered_requests"] == 1
    assert usage["successful_requests"] == 2
    assert usage["coverage_ratio"] == 0.5
    assert usage["completion_tokens"] == 5


def test_capacity_429_admission_and_retry_counts_are_separate() -> None:
    events = [
        _event(0, outcome="error", http_status=503, error_class="exact_capacity_503"),
        _event(1, outcome="error", http_status=503, error_class="other_503"),
        _event(2, outcome="error", http_status=429, error_class="http_429"),
        _event(
            3,
            outcome="client_admission_missed",
            http_status=None,
            error_class="client_admission_missed",
            dispatch_ns=None,
            headers_ns=None,
            first_generated_ns=None,
            first_visible_ns=None,
        ),
    ]
    overall = summarize_events(events, fake=True)["overall"]
    assert overall["error_counts"] == {
        "client_admission_missed": 1,
        "exact_capacity_503": 1,
        "http_429": 1,
        "other_503": 1,
    }
    assert overall["exact_capacity_ratio"] == 1 / 3
    assert overall["http_429_count"] == 1
    assert overall["retry_count"] == 0


def _overload(index: int, **overrides):
    return _event(index, phase_name="overload", phase_kind="overload", **overrides)


def test_overload_without_capacity_rejection_is_never_reported_as_success() -> None:
    """a deployment with no capacity contract cannot demonstrate overload over public http.

    dev exposes no retryable capacity rejection, so a quiet overload phase must not be read as
    headroom. the verdict says not-demonstrated and the summary carries the limitation.
    """
    events = [_overload(0), _overload(1)]
    summary = summarize_events(events, fake=False, capacity_expectations={"overload": False})
    verdict = summary["overload"]["overload"]
    assert verdict["verdict"] == "overload_not_demonstrated"
    assert verdict["exact_capacity_503"] == 0
    assert verdict["expects_capacity_contract"] is False
    assert any("not evidence of headroom" in value for value in summary["claim_limitations"])


def test_declared_capacity_contract_that_never_rejects_is_a_failed_expectation() -> None:
    events = [_overload(0), _overload(1)]
    summary = summarize_events(events, fake=False, capacity_expectations={"overload": True})
    assert (
        summary["overload"]["overload"]["verdict"]
        == "overload_not_demonstrated_despite_declared_contract"
    )
    assert not any("not evidence of headroom" in value for value in summary["claim_limitations"])


def test_authored_capacity_caveat_survives_an_overload_phase_that_produced_no_rows() -> None:
    """the no-capacity caveat is authored, so evidence that never reached overload keeps it.

    an interrupted run's journal can end before the overload phase contributes a single row.
    deriving the caveat from observed rows would drop it exactly there, so the less complete
    evidence would advertise itself as less limited than a full run over the same deployment.
    """
    events = [_event(0)]
    summary = summarize_events(events, fake=False, capacity_expectations={"overload": False})
    assert summary["overload"] == {}
    assert any("not evidence of headroom" in value for value in summary["claim_limitations"])


def test_observed_capacity_rejection_is_reported_separately_from_other_503_and_429() -> None:
    events = [
        _overload(0, outcome="error", http_status=503, error_class="exact_capacity_503"),
        _overload(1, outcome="error", http_status=503, error_class="other_503"),
        _overload(2, outcome="error", http_status=429, error_class="http_429"),
    ]
    verdict = summarize_events(events, fake=False, capacity_expectations={"overload": True})[
        "overload"
    ]["overload"]
    assert verdict["verdict"] == "capacity_rejection_observed"
    assert verdict["exact_capacity_503"] == 1
    assert verdict["other_503"] == 1
    assert verdict["http_429"] == 1


def test_offered_rate_across_phases_sums_windows_instead_of_taking_the_longest() -> None:
    """a group spanning several phases must divide by all their windows, not one.

    ``overall``, ``per_target`` and ``per_profile`` each combine every phase. dividing the combined
    count by a single phase's window inflates offered rps by the factor of the phases it ignored,
    which would overstate the load the harness claims to have offered.
    """
    events = [
        _event(0, phase_name="sustained", authored_window_seconds=3.0),
        _event(1, phase_name="sustained", authored_window_seconds=3.0),
        _event(2, phase_name="mixed", phase_kind="mixed", authored_window_seconds=1.0),
    ]
    overall = summarize_events(events, fake=True)["overall"]["rates"]["offered"]
    assert overall["denominator_basis"] == "authored_windows"
    assert overall["denominator_seconds"] == 4.0
    assert overall["requests_per_second"] == 0.75
    per_phase = summarize_events(events, fake=True)["per_phase"]
    assert per_phase["sustained"]["rates"]["offered"]["denominator_seconds"] == 3.0
    assert per_phase["mixed"]["rates"]["offered"]["denominator_seconds"] == 1.0


def test_offered_rate_is_undefined_when_only_some_phases_declare_a_window() -> None:
    """a warm phase has no authored window, so a mixed group has no defensible denominator."""
    events = [
        _event(0, phase_name="warm", phase_kind="warm", authored_window_seconds=None),
        _event(1, phase_name="sustained", authored_window_seconds=3.0),
    ]
    offered = summarize_events(events, fake=True)["overall"]["rates"]["offered"]
    assert offered["denominator_basis"] == "undefined_mixed_phase_windows"
    assert offered["denominator_seconds"] is None
    assert offered["requests_per_second"] is None


def test_offered_rate_falls_back_to_scheduled_span_when_no_phase_declares_a_window() -> None:
    events = [
        _event(0, phase_name="warm", phase_kind="warm", authored_window_seconds=None),
        _event(1, phase_name="warm", phase_kind="warm", authored_window_seconds=None),
    ]
    offered = summarize_events(events, fake=True)["overall"]["rates"]["offered"]
    assert offered["denominator_basis"] == "scheduled_span"
    assert offered["denominator_seconds"] == 1.0


def test_authored_vs_achieved_mix_and_duration_windows_are_explicit() -> None:
    events = [
        _event(0),
        _event(1, dispatch_ns=None, error_class="client_admission_missed"),
        _event(2),
    ]
    summary = summarize_events(events, fake=True)
    mix = summary["overall"]["mix"]["target"]
    assert mix["model-a"]["authored_count"] == 2
    assert mix["model-a"]["achieved_count"] == 1
    window = summary["sustained_mixed_windows"]["sustained"]
    assert window["authored_seconds"] == 3.0
    assert window["scheduled_start_ns"] == 0
    assert window["completion_end_ns"] == 2_100_000_000
