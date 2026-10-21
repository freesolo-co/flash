from __future__ import annotations

import ast
import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.engine.worker.train.opd.child import bridge, multiturn
from flash.engine.worker.train.opd.orchestration import failures


class _Loop:
    def __init__(self, loop):
        self.loop = loop

    # the prompt now goes through the shared media path before any turn runs, so the stub has to
    # model the three calls the real vllm loop exposes. a text-only prompt carries no media, which
    # is what these failure-path tests use -- the empty dict is the honest answer, not a shortcut.
    async def process_multi_modal_info(self, _messages):
        return {}

    def _get_mm_processor_kwargs(self, _audio_data=None):
        return {}

    async def apply_chat_template(self, _prompt, **_kwargs):
        return [1, 2]

    async def _run_turns(self, *_args, **_kwargs):
        return None


@pytest.fixture
def failure_path(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASH_OPD_BRIDGE_URL", "http://bridge")
    monkeypatch.setenv("FLASH_OPD_BRIDGE_TOKEN", "token")
    monkeypatch.setenv("FLASH_OPD_SEED", "3")
    monkeypatch.setenv("FLASH_OPD_MAX_TURNS", "2")
    monkeypatch.setenv("FLASH_OPD_MAX_MODEL_LEN", "128")
    monkeypatch.setenv(
        "FLASH_OPD_ENV_CAPABILITIES",
        json.dumps(["new_rollout_state", "record_model_turn", "env_reply", "rollout_done"]),
    )
    fallback = tmp_path / "score-failure"
    monkeypatch.setenv("FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH", str(fallback))
    monkeypatch.setenv("FLASH_OPD_ROLLOUT_FAILURE_PATH", str(tmp_path / "rollout-failure"))
    return fallback


def test_delivery_unknown_failure_is_durable_before_marker_and_exit_is_deferred(
    failure_path: Path,
):
    order = []

    def post_json(_url, _token, path, _payload):
        if path == "/multiturn/start":
            return {"max_turns": 1}
        if path == "/multiturn/score":
            raise bridge.FlashTeacherBridgeError(
                "authorization=secret bridge response vanished",
                classification="permanent",
                delivery_unknown=True,
            )
        assert path == "/multiturn/close"
        order.append("close")
        return {"ok": True}

    def mark_prompt_failure(uid, partition_id, global_steps, status):
        records = list(Path(failure_path.parent).glob(f"{failure_path.name}.*.json"))
        assert records
        payload = json.loads(records[0].read_text())
        assert payload["classification"] == "transient"
        assert "secret" not in payload["message"]
        order.append(("marker", uid, partition_id, global_steps, status))

    def exit_process(code):
        order.append(("exit", code))

    async def run():
        actor = _Loop(asyncio.get_running_loop())
        with pytest.raises(AssertionError, match="exit returned unexpectedly"):
            await multiturn._opd_run(
                actor,
                {},
                post_json=post_json,
                score_failure_handler=bridge._defer_score_failure,
                fatal_rollout_exit_code=bridge._fatal_rollout_exit_code,
                mark_prompt_failure=mark_prompt_failure,
                permanent_teacher_exit=86,
                transient_teacher_exit=87,
                exit_process=exit_process,
                raw_prompt=[{"role": "user", "content": "hi"}],
                global_steps=4,
                index=0,
                uid="prompt-uid",
            )

    asyncio.run(run())
    assert order == [
        ("marker", "prompt-uid", "train", 4, "failure"),
        "close",
        ("exit", 87),
    ]


class GenerateSentinelError(RuntimeError):
    pass


class _GenerateFailureServer:
    async def generate(self, **_kwargs):
        raise GenerateSentinelError(
            "generate-sentinel before-super-secret-value-after " + "界" * 5000
        )


class _GenerateFailureLoop(_Loop):
    def __init__(self, loop):
        super().__init__(loop)
        self.server_manager = _GenerateFailureServer()
        self.rollout_config = SimpleNamespace(response_length=16)
        self.tokenizer = object()


def test_generate_failure_is_durable_sanitized_and_first_before_marker_and_exit(
    failure_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("FLASH_TEST_API_KEY", "super-secret-value")
    rollout_path = failure_path.parent / "rollout-failure"
    record_file = rollout_path.with_suffix(".json")
    order = []
    unfitted_payload_sizes = []
    real_fsync = os.fsync
    real_link = os.link
    real_payload = bridge._rollout_failure_payload

    def fsync(descriptor):
        kind = "directory-fsync" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file-fsync"
        order.append(kind)
        return real_fsync(descriptor)

    def link(source, target):
        assert target == str(record_file)
        order.append("atomic-link")
        return real_link(source, target)

    def rollout_failure_payload(classification, exception_type, message, traceback_text):
        unfitted_payload_sizes.append(
            len(
                json.dumps(
                    {
                        "version": 1,
                        "kind": "multiturn_rollout_exception",
                        "classification": classification,
                        "exception_type": exception_type,
                        "message": message,
                        "traceback": traceback_text,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
        )
        return real_payload(classification, exception_type, message, traceback_text)

    monkeypatch.setattr(bridge.os, "fsync", fsync)
    monkeypatch.setattr(bridge.os, "link", link)
    monkeypatch.setattr(bridge, "_rollout_failure_payload", rollout_failure_payload)

    def register(_name):
        return lambda cls: cls

    loop_type = multiturn.build_flash_multi_turn_agent_loop(
        register=register,
        agent_loop_base=_GenerateFailureLoop,
        agent_loop_output=SimpleNamespace,
        post_json=lambda _url, _token, path, _payload: (
            {"max_turns": 1} if path == "/multiturn/start" else {"ok": True}
        ),
        score_failure_handler=bridge._defer_score_failure,
        fatal_rollout_exit_code=bridge._fatal_rollout_exit_code,
        mark_prompt_failure=lambda *_args: None,
        deterministic_seed=lambda *_args: 3,
    )

    def mark_prompt_failure(uid, partition_id, global_steps, status):
        assert record_file.is_file()
        assert record_file.stat().st_size <= 8192
        record = json.loads(record_file.read_text())
        parsed = bridge._read_rollout_failure_fallback(str(rollout_path))
        assert parsed is not None
        assert record["version"] == 1
        assert record["kind"] == "multiturn_rollout_exception"
        assert all(record[key].strip() == value for key, value in parsed.items())
        assert record["classification"] == "permanent"
        assert record["exception_type"] == "GenerateSentinelError"
        assert "generate-sentinel" in record["message"]
        assert "界" in record["message"]
        assert "GenerateSentinelError" in record["traceback"]
        assert "super-secret-value" not in json.dumps(record)
        assert len(record["message"]) <= 2000
        assert len(record["traceback"]) <= 5000
        order.append(("marker", uid, partition_id, global_steps, status))

    async def run():
        actor = loop_type(asyncio.get_running_loop())
        with pytest.raises(AssertionError, match="exit returned unexpectedly"):
            await multiturn._opd_run(
                actor,
                {},
                post_json=lambda _url, _token, path, _payload: (
                    {"max_turns": 1} if path == "/multiturn/start" else {"ok": True}
                ),
                score_failure_handler=bridge._defer_score_failure,
                fatal_rollout_exit_code=bridge._fatal_rollout_exit_code,
                mark_prompt_failure=mark_prompt_failure,
                permanent_teacher_exit=86,
                transient_teacher_exit=87,
                exit_process=lambda code: order.append(("exit", code)),
                raw_prompt=[{"role": "user", "content": "hi"}],
                global_steps=4,
                index=0,
                uid="prompt-uid",
            )

    asyncio.run(run())
    assert len(unfitted_payload_sizes) == 1
    assert unfitted_payload_sizes[0] > 8192
    assert order == [
        "file-fsync",
        "atomic-link",
        "directory-fsync",
        ("marker", "prompt-uid", "train", 4, "failure"),
        ("exit", 86),
    ]
    bridge._write_rollout_failure_fallback(RuntimeError("later failure"), "permanent")
    assert "generate-sentinel" in json.loads(record_file.read_text())["message"]


class _UnrenderableError(RuntimeError):
    def __str__(self):
        raise RuntimeError("string rendering failed")


@pytest.mark.parametrize("render_failure", ["str", "traceback"])
def test_rollout_failure_rendering_failure_still_writes_type_only_record(
    failure_path: Path,
    monkeypatch,
    render_failure,
):
    error = _UnrenderableError() if render_failure == "str" else RuntimeError("hidden detail")
    if render_failure == "traceback":

        def fail_traceback(*_args, **_kwargs):
            raise RuntimeError("traceback failed")

        monkeypatch.setattr(bridge.traceback, "format_exception", fail_traceback)

    bridge._write_rollout_failure_fallback(error, "transient")

    record = json.loads((failure_path.parent / "rollout-failure.json").read_text())
    assert record["classification"] == "transient"
    assert record["exception_type"] == type(error).__name__
    assert record["message"] == "diagnostic rendering failed"
    assert record["traceback"] == ""
    assert len(json.dumps(record, ensure_ascii=False).encode("utf-8")) <= 8192


@pytest.mark.usefixtures("failure_path")
def test_marker_and_cleanup_baseexceptions_cannot_suppress_original_exit():
    calls = []

    def post_json(_url, _token, path, _payload):
        if path == "/multiturn/start":
            return {"max_turns": 1}
        if path == "/multiturn/score":
            raise RuntimeError("fatal rollout")
        calls.append("close")
        raise KeyboardInterrupt("cleanup")

    def marker(*_args):
        calls.append("marker")
        raise SystemExit("marker")

    def exit_process(code):
        calls.append(("exit", code))

    async def run():
        actor = _Loop(asyncio.get_running_loop())
        with pytest.raises(AssertionError, match="exit returned unexpectedly"):
            await multiturn._opd_run(
                actor,
                {},
                post_json=post_json,
                score_failure_handler=bridge._defer_score_failure,
                fatal_rollout_exit_code=bridge._fatal_rollout_exit_code,
                mark_prompt_failure=marker,
                permanent_teacher_exit=86,
                transient_teacher_exit=87,
                exit_process=exit_process,
                raw_prompt=[{"role": "user", "content": "hi"}],
                global_steps=4,
                index=0,
                uid="prompt-uid",
            )

    asyncio.run(run())
    assert calls == ["marker", "close", ("exit", 86)]


def test_a_marked_prompt_failure_still_reports_the_recorded_child_evidence():
    # the wedge fix routes a delivery-unknown score failure through the existing delivery record,
    # so the parent must keep naming that record rather than falling back to the bare exit code.
    with pytest.raises(
        RuntimeError, match="permanent teacher score delivery failure: child detail"
    ):
        failures._raise_verl_failure(
            86,
            None,
            score_delivery_failure=("permanent", "child detail"),
        )
    # with no record at all, the bare exit code remains the classification of last resort.
    with pytest.raises(RuntimeError, match="permanent teacher bridge failure"):
        failures._raise_verl_failure(86, None)


@pytest.mark.parametrize(
    "failure_name",
    [
        "mutation_failure",
        "cycle_commit_failure",
        "no_signal_failure",
        "score_delivery_failure",
        "teacher_failure",
    ],
)
@pytest.mark.parametrize("classification", ["permanent", "transient"])
def test_specific_child_failure_precedes_simultaneous_rollout_failure(
    failure_name,
    classification,
):
    from flash.engine.worker.perf import RetriableInfraError

    detail = f"{failure_name}-wins"
    specific_failure = {failure_name: (classification, detail)}
    specific_failure.setdefault("teacher_failure", None)
    expected_type = RetriableInfraError if classification == "transient" else RuntimeError
    rollout_failure = {
        "classification": "permanent",
        "exception_type": "RolloutError",
        "message": "rollout-loses",
        "traceback": "",
    }

    with pytest.raises(expected_type) as excinfo:
        failures._raise_verl_failure(
            1,
            rollout_failure=rollout_failure,
            **specific_failure,
        )

    assert type(excinfo.value) is expected_type
    assert detail in str(excinfo.value)
    assert "rollout-loses" not in str(excinfo.value)


def test_parent_reconciliation_reports_rollout_record_before_subprocess_status(
    monkeypatch,
    tmp_path,
):
    import flash.engine.worker.train.entry.opd_train_runner as opd_runner

    rollout_path = tmp_path / "rollout-failure"
    monkeypatch.setenv("FLASH_OPD_ROLLOUT_FAILURE_PATH", str(rollout_path))
    try:
        raise GenerateSentinelError("parent-reconciliation-sentinel")
    except GenerateSentinelError as error:
        bridge._write_rollout_failure_fallback(error, "permanent")

    workload = SimpleNamespace(
        score_delivery_failure_path="",
        rollout_failure_path=str(rollout_path),
        resample_failure_path="",
        abandonment_failure_path="",
        mutation_failure_path="",
        cycle_commit_failure_path="",
    )
    parent_bridge = SimpleNamespace(
        teacher_failure=None,
        mutation_failure=None,
        _promote_pending_teacher_failure=lambda: False,
    )
    with pytest.raises(RuntimeError) as excinfo:
        opd_runner._reconcile_child_failures(
            workload,
            parent_bridge,
            1,
            truncation_window=None,
        )
    detail = str(excinfo.value)
    assert "permanent multi-turn OPD rollout failure" in detail
    assert "GenerateSentinelError" in detail
    assert "parent-reconciliation-sentinel" in detail
    assert "subprocess exited with status" not in detail


def test_copied_child_shims_import_flat_without_flash_package(tmp_path):
    import flash.engine.worker.train.entry.opd_train_runner as runner

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    runner._write_child_shims(
        SimpleNamespace(),
        SimpleNamespace(shim_dir=str(shim_dir)),
        None,
        [],
    )
    script = """
import importlib
import importlib.util
import os
import sys

sys.path.insert(0, os.getcwd())
assert importlib.util.find_spec("flash") is None
bridge = importlib.import_module("flash_opd_bridge")
multiturn = importlib.import_module("flash_opd_multiturn")
replay_guard = importlib.import_module("flash_opd_replay_guard")
assert multiturn._write_rollout_failure_fallback is bridge._write_rollout_failure_fallback
assert replay_guard._read_rollout_failure_fallback is bridge._read_rollout_failure_fallback
assert replay_guard._render_rollout_failure is bridge._render_rollout_failure
"""

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        cwd=shim_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_every_flat_module_a_copied_shim_imports_is_itself_copied(tmp_path):
    """A `flash_*` import with no matching copy is a crash the child reaches only on a paid GPU.

    The test above imports three modules by hand, so a bundle entry nothing in it names can go
    missing unnoticed. This reads the imports out of the copied sources instead, which keeps the
    guard honest as the bundle grows. Parsing beats importing here: several of these modules pull
    in torch or verl at module scope and would skip rather than fail on a CPU runner.
    """
    import flash.engine.worker.train.entry.opd_train_runner as runner

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    runner._write_child_shims(
        SimpleNamespace(),
        SimpleNamespace(shim_dir=str(shim_dir)),
        None,
        [],
    )

    copied = {path.stem for path in shim_dir.glob("*.py")}
    assert "flash_opd_plugin" in copied, sorted(copied)

    missing = {}
    for name in sorted(copied):
        imported = set()
        for node in ast.walk(ast.parse((shim_dir / f"{name}.py").read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names if alias.name.startswith("flash_")}
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("flash_"):
                imported.add(node.module)
        absent = imported - copied
        if absent:
            missing[name] = sorted(absent)

    assert missing == {}, f"copied shims import modules the bundle never copies: {missing}"
