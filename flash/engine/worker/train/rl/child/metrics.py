"""Forward Verl's durable metric journal through the driver, outside Ray log forwarding."""

from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from pathlib import Path


def _metric_line(record: dict) -> str:
    step, data = record["step"], record["data"]
    if isinstance(step, bool) or not isinstance(step, int) or not isinstance(data, dict):
        raise ValueError("invalid GRPO metric journal record")
    fields = [f"step:{step}"]
    fields.extend(
        f"{key}:{value!r}"
        for key, value in data.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return " - ".join(fields)


def _forward_metrics(handle, stopped: threading.Event, emit) -> None:
    pending = b""
    while True:
        # Only EOF observed after producer shutdown proves the journal is drained.
        # A write and shutdown can otherwise race with an earlier empty read.
        producer_finished = stopped.is_set()
        chunk = handle.read(65536)
        if chunk:
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                emit(_metric_line(json.loads(line)))
        elif producer_finished:
            if pending:
                raise ValueError("truncated GRPO metric journal record")
            return
        else:
            stopped.wait(0.05)


@contextmanager
def stream_metrics(path: str, *, emit=None):
    """Stream during training and drain every committed record before the driver exits.

    Ray's asynchronous actor-log forwarding may discard its final backlog when the driver
    exits. Verl's file logger writes each metric record synchronously with buffering disabled;
    the driver owns this reader and joins it after the trainer RPC completes.
    """
    if emit is None:

        def emit(line):
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    journal = Path(path)
    journal.write_bytes(b"")  # A retry must never replay a previous attempt's evidence.
    stopped = threading.Event()
    errors: list[BaseException] = []

    def forward():
        try:
            with journal.open("rb") as handle:
                _forward_metrics(handle, stopped, emit)
        except BaseException as exc:
            errors.append(exc)

    reader = threading.Thread(target=forward, name="flash-grpo-metrics", daemon=True)
    reader.start()
    try:
        yield
    finally:
        stopped.set()
        reader.join()
        if errors:
            raise RuntimeError("GRPO metric journal forwarding failed") from errors[0]
