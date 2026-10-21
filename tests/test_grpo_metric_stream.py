"""Training-step evidence must survive delayed or absent Ray actor-log forwarding."""

from __future__ import annotations

import io
import json
import threading

import pytest

from flash.engine.worker.train.rl.child.metrics import _forward_metrics, stream_metrics
from flash.engine.worker.verl.child_io import parse_verl_step_metrics, verl_step_number


def _record(step):
    return (
        json.dumps(
            {"step": step, "data": {"actor/grad_norm": 0.25, "critic/rewards/mean": 0.5}}
        ).encode()
        + b"\n"
    )


def test_live_steps_and_final_backlog_survive_without_any_ray_stdout(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(_record(999))
    received = []
    first_step = threading.Event()

    def emit(line):
        received.append(line)
        first_step.set()

    with stream_metrics(str(path), emit=emit), path.open("wb", buffering=0) as writer:
        writer.write(_record(1))
        assert first_step.wait(5), "progress must arrive before training exits"
        # A final burst larger than the reader buffer must drain fully on exit.
        writer.write(b"".join(_record(step) for step in range(2, 2049)))
    assert [verl_step_number(line) for line in received] == list(range(1, 2049))
    assert parse_verl_step_metrics(received[-1])["step"] == 2048


def test_partial_write_is_reassembled_before_parsing(tmp_path):
    path = tmp_path / "metrics.jsonl"
    received = []
    with stream_metrics(str(path), emit=received.append), path.open("wb", buffering=0) as writer:
        record = _record(1)
        writer.write(record[:10])
        writer.write(record[10:])
    assert [verl_step_number(line) for line in received] == [1]


@pytest.mark.parametrize("payload", [b'{"step":1', b"not-json\n", b'{"step":true,"data":{}}\n'])
def test_corrupt_or_truncated_evidence_fails_closed(tmp_path, payload):
    path = tmp_path / "metrics.jsonl"
    with (
        pytest.raises(RuntimeError, match="metric journal forwarding failed"),
        stream_metrics(str(path), emit=lambda line: None),
        path.open("wb", buffering=0) as writer,
    ):
        writer.write(payload)


def test_failed_trainer_still_drains_completed_steps_and_keeps_its_error(tmp_path):
    path = tmp_path / "metrics.jsonl"
    received = []

    def train():
        with stream_metrics(str(path), emit=received.append):
            path.write_bytes(_record(1))
            raise ValueError("trainer failed")

    with pytest.raises(ValueError, match="trainer failed"):
        train()
    assert [verl_step_number(line) for line in received] == [1]


@pytest.mark.parametrize("payload", [_record(1), b"not-json\n", b'{"step":1'])
def test_shutdown_between_empty_read_and_stop_check_still_drains(payload):
    stopped = threading.Event()
    received = []

    class FinalWriteAfterEmptyRead(io.BytesIO):
        def read(self, size=-1):
            if not stopped.is_set():
                # The first read saw EOF, but the writer appended its final record
                # and signalled shutdown before the reader checked the stop event.
                stopped.set()
                return b""
            return super().read(size)

    with FinalWriteAfterEmptyRead(payload) as handle:
        if payload == _record(1):
            _forward_metrics(handle, stopped, received.append)
            assert [verl_step_number(line) for line in received] == [1]
        else:
            with pytest.raises(ValueError, match=r"Expecting value|truncated GRPO metric journal"):
                _forward_metrics(handle, stopped, received.append)
