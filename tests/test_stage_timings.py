import threading

import pytest

from flash.engine.worker.runtime import timings


@pytest.fixture(autouse=True)
def isolated_timings():
    timings.reset()
    yield
    timings.reset()


def test_concurrent_records_are_lossless_and_snapshots_detached():
    threads = [
        threading.Thread(target=lambda: [timings.record("upload", 2.0) for _ in range(50)])
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    result = timings.snapshot()
    assert result["upload"] == {"count": 200, "seconds": 400.0, "max_seconds": 2.0, "failures": 0}
    result["upload"]["seconds"] = 0
    assert timings.snapshot()["upload"]["seconds"] == 400.0


def test_bad_durations_do_not_poison_report():
    for value in (float("nan"), float("inf"), -1):
        timings.record("bad", value)
    assert timings.snapshot() == {}
    timings.record("startup", 3, failed=True)
    assert timings.snapshot()["startup"]["failures"] == 1


def test_regression_check_rejects_mismatched_or_missing_evidence():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "regression", Path(__file__).parents[1] / "scripts/check_startup_regression.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = {
        "configuration": {"gpu": "H100", "workload": "test"},
        "stage_timings": {"init": {"seconds": 10, "count": 1}},
    }
    candidate = {**baseline, "stage_timings": {"init": {"seconds": 13, "count": 1}}}
    assert module.regressions(baseline, candidate, max_ratio=1.2) == ["init: 10.00s -> 13.00s"]
    with pytest.raises(ValueError, match="same model"):
        module.regressions(baseline, {**candidate, "configuration": {}}, max_ratio=1.2)
    with pytest.raises(ValueError, match="missing or invalid"):
        module.regressions(baseline, {**candidate, "stage_timings": {}}, max_ratio=1.2)
