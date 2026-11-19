"""An environment's external-service outage is an infrastructure fault, not the run's own failure.

Shape of miniwob-glm flash-1795098896-e2a4d966 (2026-11-19): after ~1.9 GPU-hours, the MiniWoB
browser service answered 503 through all six of the environment client's own retries, the
environment raised ``RuntimeError("MiniWoB browser service failed (503) after 6 attempts")`` from
inside the HTTPError handler, and Flash failed the run as ``job_failed`` with no retry. Its
``run.error`` named only "train phase 'rl' produced no /tmp/metrics.json"; the cause sat in
``last_heartbeat.error``.
"""

from __future__ import annotations

import time
import urllib.error
from types import SimpleNamespace

import pytest

import flash.engine.worker.entry.worker as worker_entry
import flash.engine.worker.train.entry.backend_common as backend_common
import flash.engine.worker.train.entry.rl_train_runner as rl_runner
import flash.runner.lifecycle.attempts as runner_attempts
import flash.runner.lifecycle.state as runner_state
from flash.engine.worker.perf.lifecycle import RetriableInfraError
from flash.engine.worker.train.rl.child.multiturn import post_json
from flash.engine.worker.train.rl.rollout.multi_turn import start_reward_server
from flash.engine.worker.train.rl.rollout.service_outage import is_environment_service_outage


def _miniwob_outage() -> RuntimeError:
    """Raise exactly as the MiniWoB client does once its retries are spent."""
    try:
        try:
            raise urllib.error.HTTPError(
                "https://browser.example/replay", 503, "Service Unavailable", None, None
            )
        except urllib.error.HTTPError:
            raise RuntimeError("MiniWoB browser service failed (503) after 6 attempts") from None
    except RuntimeError as exc:
        return exc


def _raised_inside(handled: BaseException, raised: BaseException) -> BaseException:
    try:
        try:
            raise handled
        except BaseException:
            raise raised  # noqa: B904 - the implicit context is what is under test
    except BaseException as exc:
        return exc


def _third_party(module: str, name: str, **attrs) -> Exception:
    cls = type(name, (Exception,), {"__module__": module})
    exc = cls("boom")
    for key, value in attrs.items():
        setattr(exc, key, value)
    return exc


@pytest.mark.parametrize(
    "exc",
    [
        _miniwob_outage(),
        urllib.error.URLError("connection refused"),
        ConnectionResetError("reset by peer"),
        TimeoutError("read timed out"),
        _raised_inside(ConnectionRefusedError(), ValueError("service unreachable")),
        _third_party("requests.exceptions", "ConnectionError"),
        _third_party("httpx", "ReadTimeout"),
        _third_party("requests.exceptions", "HTTPError", response=SimpleNamespace(status_code=502)),
    ],
    ids=["miniwob", "urlerror", "reset", "timeout", "context", "requests", "httpx", "requests-5xx"],
)
def test_transport_faults_anywhere_in_the_chain_are_service_outages(exc):
    assert is_environment_service_outage(exc)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("MiniWoB browser service failed (503) after 6 attempts"),
        KeyError("reward"),
        ValueError("observation must be a string"),
        _raised_inside(
            urllib.error.HTTPError("https://x/replay", 400, "Bad Request", None, None),
            RuntimeError("MiniWoB browser service failed (400)"),
        ),
        _raised_inside(KeyError("task"), RuntimeError("unknown task")),
        _third_party("requests.exceptions", "HTTPError", response=SimpleNamespace(status_code=404)),
    ],
    ids=["bare-runtime", "keyerror", "valueerror", "http-400", "keyerror-context", "requests-404"],
)
def test_environment_code_errors_stay_job_failures(exc):
    assert not is_environment_service_outage(exc)


def _bridge_reply(score):
    server, url = start_reward_server(score, example_count=1)
    try:
        with pytest.raises(RuntimeError) as raised:
            post_json(url, "/score", {"index": 0, "solution_str": "x"}, error_style="reward")
        return str(raised.value)
    finally:
        server.shutdown()


def test_bridge_marks_an_env_service_outage_and_the_trainer_retries_it():
    def score(index, solution):
        raise _miniwob_outage()

    message = _bridge_reply(score)
    assert "(HTTP 502): environment service outage: RuntimeError: MiniWoB browser" in message

    tail = backend_common.ChildOutputTail()
    tail.record("[verl] Traceback (most recent call last):\n")
    tail.record(f"[verl] RuntimeError: {message}\n")
    tail.record("[verl] RuntimeError: GRPO replay observed failed prompt 'p' at global_steps=15\n")
    with pytest.raises(RetriableInfraError) as raised:
        backend_common.raise_for_classified_verl_exit(1, tail)
    assert "MiniWoB browser service failed (503) after 6 attempts" in str(raised.value)
    assert worker_entry._worker_failure_flags(raised.value) == {"retriable": True, "oom": False}


def test_bridge_keeps_an_environment_bug_a_job_failure():
    def score(index, solution):
        raise KeyError("reward")

    message = _bridge_reply(score)
    assert "(HTTP 503): KeyError" in message
    assert "environment service outage" not in message

    tail = backend_common.ChildOutputTail()
    tail.record(f"[verl] RuntimeError: {message}\n")
    assert backend_common.raise_for_classified_verl_exit(1, tail) is None


def test_env_raising_the_marker_words_itself_is_not_an_outage():
    def score(index, solution):
        raise RuntimeError("environment service outage: but really a bug")

    message = _bridge_reply(score)
    tail = backend_common.ChildOutputTail()
    tail.record(f"[verl] RuntimeError: {message}\n")
    assert backend_common.raise_for_classified_verl_exit(1, tail) is None


class _FakeProc:
    pid = 4242

    def __init__(self, lines):
        self.stdout = iter(lines)
        self.returncode = None

    def wait(self, timeout=None):
        self.returncode = 1
        return 1

    def poll(self):
        return None  # keeps the silence watchdog unarmed; this test is about the exit only


def test_grpo_exit_after_a_multi_turn_env_outage_is_retriable(monkeypatch):
    monkeypatch.setattr(rl_runner, "kill_process_group", lambda proc, *, process_group_id: None)
    worker = "[verl] (AgentLoopWorkerTQ pid=5151) "
    lines = [
        f"{worker}Traceback (most recent call last):\n",
        (
            f"{worker}RuntimeError: flash multi-turn bridge could not serve /multiturn/step "
            "(HTTP 502): environment service outage: RuntimeError: MiniWoB browser service "
            "failed (503) after 6 attempts\n"
        ),
        "[verl] RuntimeError: GRPO replay observed failed prompt '4296e409' at global_steps=15\n",
    ]
    stream = rl_runner._GrpoSubprocessStream(_FakeProc(lines))
    list(stream)
    with pytest.raises(RetriableInfraError, match="external service failed during rollouts"):
        stream.wait_and_classify()


def _terminal_artifacts(heartbeat, *, attempt=0):
    from flash.providers._lifecycle.instances.poll_instance import _TerminalArtifacts

    adapter = SimpleNamespace(
        signal_interruption_is_preemption=True,
        current_attempt=attempt,
        failure_detail=lambda marker: (marker or {}).get("error") or "worker stopped",
    )
    return _TerminalArtifacts(
        adapter,
        say=print,
        heartbeat_reader=lambda force=False: heartbeat,
        launch_ts=100.0,
        absolute_deadline=None,
    )


_NO_METRICS = (
    "RuntimeError: train phase 'rl' produced no /tmp/metrics.json (it crashed before finishing)"
)
_CAUSE = (
    "RuntimeError: verl.trainer.main_ppo exited 1: RuntimeError: GRPO replay observed failed "
    "prompt 'p' at global_steps=15"
)


def test_run_error_leads_with_the_worker_heartbeat_cause():
    heartbeat = {"attempt": 0, "ts": time.time(), "error": _CAUSE, "retriable": False}
    result = _terminal_artifacts(heartbeat)._fail_from_marker(
        {"retriable": False, "error": _NO_METRICS}
    )
    assert result.failure == "job_failed"
    assert result.detail.startswith(_CAUSE)
    assert _NO_METRICS in result.detail


def test_retriable_heartbeat_cause_is_carried_and_retried():
    heartbeat = {
        "attempt": 0,
        "ts": time.time(),
        "error": "RetriableInfraError: x",
        "retriable": True,
    }
    result = _terminal_artifacts(heartbeat)._fail_from_marker(
        {"retriable": False, "error": _NO_METRICS}
    )
    assert result.failure == "worker_retriable"
    assert result.detail.startswith("RetriableInfraError: x")


def test_stale_attempt_heartbeat_never_names_the_cause():
    heartbeat = {"attempt": 0, "ts": time.time(), "error": _CAUSE, "retriable": True}
    result = _terminal_artifacts(heartbeat, attempt=1)._fail_from_marker(
        {"retriable": False, "error": _NO_METRICS}
    )
    assert result.failure == "job_failed"
    assert result.detail == _NO_METRICS


def _decide(monkeypatch, tmp_path, failure, *, max_retries=0):
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import Candidate
    from flash.runner.lifecycle import status as runner_status
    from flash.runner.supervise.retry_decision import FailureObservation

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id=f"failure-class-{failure}",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=max_retries),
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    claim = runner_attempts.reserve_verified_attempt_launch(spec.run_id)
    remote = {"provider": "modal", "job_id": "job", "attempt": claim.attempt}
    assert runner_attempts.persist_claimed_remote(spec.run_id, claim, remote)
    chosen = Candidate("modal", "H100", 1, 80, 1)
    plan = runner_attempts.decide_attempt_failure(
        spec.run_id,
        claim_token=claim.token,
        expected_remote=runner_status._load_status_json(spec.run_id)["remote"],
        observation=FailureObservation(failure, chosen=chosen, candidates=(chosen,)),
        attempt=claim.attempt,
    )
    return plan, runner_status.get_status(spec.run_id)


def test_terminal_job_failure_is_classified_job(monkeypatch, tmp_path):
    plan, status = _decide(monkeypatch, tmp_path, "job_failed")
    assert not plan.retry
    assert status.failure_class == "job"
    status.state = "failed"
    assert status.to_dict()["failure_class"] == "job"


def test_retried_infra_failure_leaves_the_class_unset(monkeypatch, tmp_path):
    plan, status = _decide(monkeypatch, tmp_path, "worker_retriable", max_retries=2)
    assert plan.retry
    assert "resume from last checkpoint" in plan.action
    assert status.failure_class is None


def test_failure_class_mapping():
    from flash.runner.supervise.retry_decision import failure_class

    for failure in ("worker_retriable", "job_preempted", "stalled", "poll_error", "boot_timeout"):
        assert failure_class(failure) == "infra", failure
    for failure in ("job_failed", "oom", None):
        assert failure_class(failure) == "job", failure


def test_failure_class_is_hidden_until_the_run_failed():
    status = runner_state.RunStatus(run_id="r", state="running", spec={}, failure_class="infra")
    assert status.to_dict()["failure_class"] is None
    status.state = "failed"
    assert status.to_dict()["failure_class"] == "infra"


def test_reattached_run_keeps_writing_its_own_log(monkeypatch, tmp_path):
    import flash.runner.supervise.attach as runner_attach
    import flash.server.platform.runtime as runtime

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(
        runtime, "runs_file_path", lambda run_id, suffix: str(tmp_path / f"{run_id}{suffix}")
    )
    later = []

    def attach(run_id, log_stream=None):
        print(f"attach: {run_id} ended (job_failed); evaluating recovery", file=log_stream)
        later.append(log_stream)
        return "status"

    monkeypatch.setattr(runner_attach, "attach_run", attach)
    assert runtime._attach_with_run_log("run-x") == "status"
    # a reconciliation task scheduled by attach writes after attach itself returned.
    print("attach: run-x not retrying", file=later[0], flush=True)
    log = (tmp_path / "run-x.log").read_text()
    assert "evaluating recovery" in log
    assert "not retrying" in log
