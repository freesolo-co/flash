# offline parity tests for the opd teacher-failure classification + skip accounting state machine
# (flash/engine/worker/opd_teacher_semantics.py). no torch / no openrlhf required.
from flash.engine.worker.opd_teacher_semantics import (
    ABORT_PERMANENT,
    COMPLETE,
    DETERMINISTIC_SHORTFALL,
    RETRY,
    SKIP_TRANSIENT,
    TRANSIENT_SHORTFALL,
    SkipAccounting,
    classify_teacher_failure,
    classify_under_run,
)


def test_permanent_failure_aborts_regardless_of_attempts():
    assert classify_teacher_failure("permanent", attempt=1, max_attempts=3) == ABORT_PERMANENT
    assert classify_teacher_failure("permanent", attempt=3, max_attempts=3) == ABORT_PERMANENT


def test_unclassified_failure_is_fail_closed_permanent():
    assert classify_teacher_failure("weird", attempt=1, max_attempts=3) == ABORT_PERMANENT


def test_transient_retries_until_bound_then_skips_not_aborts():
    assert classify_teacher_failure("transient", attempt=1, max_attempts=3) == RETRY
    assert classify_teacher_failure("transient", attempt=2, max_attempts=3) == RETRY
    # bound reached -> skip the sample and continue the run (NOT abort)
    assert classify_teacher_failure("transient", attempt=3, max_attempts=3) == SKIP_TRANSIENT


def test_transient_single_attempt_skips_immediately():
    assert classify_teacher_failure("transient", attempt=1, max_attempts=1) == SKIP_TRANSIENT


def test_classify_teacher_failure_validates_inputs():
    for bad in ({"attempt": 0}, {"attempt": 1, "max_attempts": 0}):
        try:
            classify_teacher_failure("transient", attempt=bad.get("attempt", 1), max_attempts=bad.get("max_attempts", 3))
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_skip_accounting_counts_by_reason():
    acc = SkipAccounting()
    acc.record_ok()
    acc.record_ok()
    acc.record_transient()
    acc.record_no_signal()
    acc.record_transient()
    assert acc.ok == 2
    assert acc.transient == 2
    assert acc.no_signal == 1
    assert acc.skip_counts["transient"] == 2
    assert acc.skip_counts["no_signal"] == 1


def test_new_transient_excludes_restored_baseline():
    # a resumed run restores cumulative transient failures; only NEW ones make a shortfall retriable
    acc = SkipAccounting.restored(transient=5, skip_counts={"transient": 5})
    assert acc.transient == 5
    assert acc.new_transient == 0
    acc.record_transient()
    assert acc.transient == 6
    assert acc.new_transient == 1


def test_step_signal_accounting():
    acc = SkipAccounting()
    acc.record_step(had_signal=True)
    acc.record_step(had_signal=False)
    acc.record_step(had_signal=True)
    assert acc.steps_total == 3
    assert acc.steps_with_signal == 2


def test_under_run_complete_when_all_steps_have_signal():
    assert classify_under_run(steps_with_signal=10, steps_expected=10, new_transient_failures=0) == COMPLETE
    # more signal steps than expected (defensive) still counts as complete
    assert classify_under_run(steps_with_signal=11, steps_expected=10, new_transient_failures=3) == COMPLETE


def test_under_run_transient_shortfall_is_retriable():
    # a shortfall caused by NEW transient failures -> retriable infra
    assert (
        classify_under_run(steps_with_signal=7, steps_expected=10, new_transient_failures=2)
        == TRANSIENT_SHORTFALL
    )


def test_under_run_deterministic_shortfall_when_no_new_transient():
    # a shortfall with no transient failures -> the run genuinely could not align (not retriable)
    assert (
        classify_under_run(steps_with_signal=7, steps_expected=10, new_transient_failures=0)
        == DETERMINISTIC_SHORTFALL
    )


def test_end_to_end_transient_flap_does_not_abort_but_terminal_retries():
    # simulate: 10 steps expected; a transient teacher flap skips samples on 3 steps such that those
    # steps produce no signal, but the run keeps going (no permanent abort). the terminal gate then
    # classifies the shortfall as transient -> retry.
    acc = SkipAccounting()
    for step in range(10):
        flapped = step in (2, 5, 8)
        if flapped:
            # transient failure on the only sample this step -> skip, no signal this step
            assert classify_teacher_failure("transient", attempt=2, max_attempts=2) == SKIP_TRANSIENT
            acc.record_transient()
            acc.record_step(had_signal=False)
        else:
            acc.record_ok()
            acc.record_step(had_signal=True)
    assert acc.steps_with_signal == 7
    assert acc.new_transient == 3
    assert (
        classify_under_run(acc.steps_with_signal, acc.steps_total, acc.new_transient)
        == TRANSIENT_SHORTFALL
    )
