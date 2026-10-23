"""Forward Verl's durable metric journal through the driver, outside Ray log forwarding."""

from __future__ import annotations

import json
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

# Every scalar consumed by the parent's step accounting. The complete record stays in
# step_metrics.jsonl; only this bounded record crosses stdout alongside Ray's other writers.
_METRIC_KEYS = (
    "actor/pg_loss",
    "actor/grad_norm",
    "actor/kl_loss",
    "actor/entropy",
    "critic/rewards/mean",
    "critic/advantages/min",
    "critic/advantages/max",
    "response_length/mean",
    "response_length/clip_ratio",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "rollout_pool/gpus",
    "rollout_pool/replicas",
    "rollout_pool/training_gpus",
    "rollout_pool/published_step",
)


def _metric_line(record: dict) -> str:
    step, data = record["step"], record["data"]
    if isinstance(step, bool) or not isinstance(step, int) or not isinstance(data, dict):
        raise ValueError("invalid GRPO metric journal record")
    fields = [f"step:{step}"]
    fields.extend(
        f"{key}:{value!r}"
        for key in _METRIC_KEYS
        if (value := data.get(key)) is not None
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    return " - ".join(fields)


def _forward_metrics(handle, stopped: threading.Event, emit) -> None:
    pending = b""
    while True:
        # Observe completion before reading. A writer can append and stop between
        # an EOF read and an event check; that EOF cannot prove the final drain.
        draining = stopped.is_set()
        chunk = handle.read(65536)
        if chunk:
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                emit(_metric_line(json.loads(line)))
        elif draining:
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
            # Ray/tqdm may leave a colored warning or progress bar without a newline.
            # A single write within PIPE_BUF cannot interleave with another pipe writer.
            # TextIOWrapper writes of the full multi-kilobyte diagnostic row could split.
            payload = ("\n" + line + "\n").encode()
            fd = sys.stdout.fileno()
            if len(payload) > os.fpathconf(fd, "PC_PIPE_BUF"):
                raise ValueError("GRPO metric record exceeds the atomic pipe-write limit")
            if os.write(fd, payload) != len(payload):
                raise OSError("incomplete GRPO metric pipe write")

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
