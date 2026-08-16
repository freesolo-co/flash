from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from pathlib import Path

import pytest

from flash.engine.worker.train.opd import failures
from flash.engine.worker.train.opd.child import bridge, multiturn


class _Loop:
    def __init__(self, loop):
        self.loop = loop

    async def apply_chat_template(self, _prompt):
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


def test_child_shim_set_includes_flat_diagnostics_and_replay_modules():
    importlib.import_module("flash.engine.worker.opd_train")
    runner = importlib.import_module("flash.engine.worker.opd_train_runner")
    source = inspect.getsource(runner._write_child_shims)
    assert '"flash_child_diagnostics.py"' in source
    assert '"flash_opd_replay_guard.py"' in source
