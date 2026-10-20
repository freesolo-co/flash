"""Training-step evidence must survive delayed or absent Ray actor-log forwarding."""

from __future__ import annotations

import json
import subprocess
import sys
import threading

import pytest

from flash.engine.worker.train.rl.child.metrics import stream_metrics
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


def test_completion_after_eof_read_still_drains_the_final_write():
    from flash.engine.worker.train.rl.child.metrics import _forward_metrics

    stopped = threading.Event()
    received = []
    record = _record(1)

    class ConcurrentAppend:
        reads = 0

        def read(self, size):
            self.reads += 1
            if self.reads == 1:
                return record[:10]
            if self.reads == 2:
                # The reader saw EOF just before the final append and stop.
                stopped.set()
                return b""
            return record[10:] if self.reads == 3 else b""

    _forward_metrics(ConcurrentAppend(), stopped, received.append)
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


@pytest.mark.parametrize("step", [23, 26, 30])
def test_colored_ray_warning_cannot_hide_an_adjacent_optimizer_step(step):
    line = (
        "\x1b[36m(AgentLoopWorker pid=6688)\x1b[0m warning"
        "\x1b[32m [repeated 6x across cluster]\x1b[0m"
        f"step:{step} - actor/grad_norm:\x1b[32m0.25\x1b[0m"
        " - critic/rewards/mean:0.5 - critic/advantages/min:-0.5"
        " - critic/advantages/max:0.5"
    )
    assert verl_step_number(line) == step
    assert parse_verl_step_metrics(line) == {
        "step": step,
        "grad_norm": 0.25,
        "reward": 0.5,
        "advantage_min": -0.5,
        "advantage_max": 0.5,
    }


@pytest.mark.parametrize("prefix", ["global_", "path/", "not_a_"])
def test_stripping_color_does_not_turn_a_path_or_metric_name_into_a_step(prefix):
    line = prefix + "\x1b[0mstep:23 - actor/grad_norm:0.25"
    assert verl_step_number(line) is None
    assert parse_verl_step_metrics(line) is None


def test_default_stream_emitter_starts_a_fresh_line_after_unterminated_warning(tmp_path, capfd):
    path = tmp_path / "metrics.jsonl"
    sys.stdout.write("warning\x1b[0m")
    sys.stdout.flush()
    with stream_metrics(str(path)):
        path.write_bytes(_record(1))
    lines = capfd.readouterr().out.splitlines()
    assert any(line.startswith("step:1 - ") for line in lines)


def test_large_diagnostics_cannot_expand_the_atomic_validation_record(tmp_path):
    from flash.engine.worker.train.rl.child.metrics import _METRIC_KEYS

    path = tmp_path / "metrics.jsonl"
    data = dict.fromkeys(_METRIC_KEYS, 0.25)
    data.update({f"extra/diagnostic_{i}": 123.456 for i in range(1000)})
    received = []
    with stream_metrics(str(path), emit=received.append):
        path.write_text(json.dumps({"step": 32, "data": data}) + "\n")
    assert len(received[0].encode()) + 2 < 512  # POSIX's minimum PIPE_BUF.
    assert parse_verl_step_metrics(received[0])["grad_norm"] == 0.25
    assert len(json.loads(path.read_text())["data"]) > 1000


def test_atomic_record_carries_every_metric_consumed_by_parent_accounting():
    import ast
    import inspect

    from flash.engine.worker.train.entry import rl_train_runner
    from flash.engine.worker.train.rl.child.metrics import _METRIC_KEYS
    from flash.engine.worker.verl import child_io

    needed = {key for key, _ in child_io._VERL_METRIC_FIELDS}
    needed.update([child_io._ADVANTAGE_MIN_KEY, child_io._ADVANTAGE_MAX_KEY])
    tree = ast.parse(inspect.getsource(rl_train_runner._ingest_step_metrics))
    needed.update(
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "parse_verl_metric"
        and isinstance(node.args[1], ast.Constant)
    )
    assert needed <= set(_METRIC_KEYS)


def test_metric_records_survive_concurrent_large_colored_stdout_writes():
    script = r"""
import json, os, tempfile, threading, time
from pathlib import Path
from flash.engine.worker.train.rl.child.metrics import stream_metrics

def warnings():
    for _ in range(100):
        os.write(1, b"\x1b[36mwarning " + b"x" * 8000 + b"\x1b[0m")
        time.sleep(0.001)

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "metrics.jsonl"
    noise = threading.Thread(target=warnings)
    data = {f"diagnostic/{i}": 123.456 for i in range(1000)}
    data.update({"actor/grad_norm": 0.25, "critic/rewards/mean": 0.5})
    with stream_metrics(str(path)), path.open("wb", buffering=0) as writer:
        noise.start()
        for step in range(1, 129):
            writer.write(json.dumps({"step": step, "data": data}).encode() + b"\n")
            time.sleep(0.001)
        noise.join()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30, check=True
    )
    rows = [
        parse_verl_step_metrics(line) for line in completed.stdout.splitlines() if "step:" in line
    ]
    assert [row["step"] for row in rows] == list(range(1, 129))
    assert all(row["grad_norm"] == 0.25 for row in rows)
