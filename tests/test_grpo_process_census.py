from __future__ import annotations

import subprocess
import sys
import time

from flash.engine.worker.verl.process_census import GrpoProcessCensus


def test_process_census_reports_numeric_bounded_summary_and_clean_terminal():
    census = GrpoProcessCensus(__import__("os").getpid(), interval_s=0.02).start()
    script = """
import subprocess
import sys
import threading
import time

threads = [threading.Thread(target=time.sleep, args=(0.35,)) for _ in range(3)]
for thread in threads:
    thread.start()
children = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.3)']) for _ in range(2)]
for child in children:
    child.wait()
for thread in threads:
    thread.join()
"""
    proc = subprocess.Popen([sys.executable, "-c", script])
    deadline = time.monotonic() + 2.0
    while proc.poll() is None and time.monotonic() < deadline:
        census.sample_step()
        time.sleep(0.02)
    assert proc.wait(timeout=1) == 0
    summary = census.stop()

    assert all(isinstance(value, int) for value in summary.values())
    assert summary["peak_processes"] >= 3
    assert summary["peak_threads"] >= 5
    assert summary["peak_single_process_threads"] >= 4
    assert summary["terminal_processes"] == summary["baseline_processes"]
    assert summary["terminal_threads"] == summary["baseline_threads"]
    assert len(summary) <= 16


def test_process_census_source_avoids_sensitive_proc_metadata():
    import inspect

    import flash.engine.worker.verl.process_census as census

    source = inspect.getsource(census)
    for forbidden in ("cmdline", "environ", "/fd", "status", "username"):
        assert forbidden not in source
