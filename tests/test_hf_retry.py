from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from flash.providers._hf_retry import hf_call


def test_hf_call_times_out_while_callback_is_in_flight() -> None:
    release = threading.Event()
    started = threading.Event()

    def hang() -> None:
        started.set()
        release.wait()

    before = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="upload exceeded the run wall deadline"):
            hf_call(
                hang,
                "upload",
                logger=logging.getLogger(__name__),
                deadline_at=time.time() + 0.05,
            )
        assert time.monotonic() - before < 0.5
        assert started.wait(timeout=0.5)
    finally:
        release.set()


def test_hf_call_timeout_does_not_block_interpreter_shutdown() -> None:
    script = """
import logging
import threading
import time

from flash.providers._hf_retry import hf_call


def hang():
    threading.Event().wait()


try:
    hf_call(
        hang,
        "download",
        logger=logging.getLogger(__name__),
        deadline_at=time.time() + 0.05,
    )
except TimeoutError:
    print("timed-out", flush=True)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "timed-out"
