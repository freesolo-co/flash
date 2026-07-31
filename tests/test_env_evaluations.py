"""Tests for Flash-native environment evaluation sidecars and CLI execution."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

import flash.cli as cli
from flash.envs.evaluations import (
    BaseEvalSuite,
    EvalCase,
    EvalResult,
    EvalSuiteReport,
    has_evaluations,
    load_evaluation_suites,
    normalize_eval_result,
)

# --upload validates the project id before spending anything, so tests must pass a real UUID.
_PROJECT_ID = "11111111-1111-1111-1111-111111111111"
_EXPLICIT_TARGET = "flash-1/step-3"


def _environment_dir(tmp_path: Path) -> Path:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment():\n    return None\n")
    return env_dir


def test_load_evaluation_suites_calls_factory_with_environment(tmp_path) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "eval_sidecar_helper.py").write_text("SUITE_NAME = 'held-out'\n")
    (env_dir / "evaluations.py").write_text(
        "from eval_sidecar_helper import SUITE_NAME\n"
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = SUITE_NAME\n"
        "    def __init__(self, environment): self.environment = environment\n"
        "    def cases(self): return [EvalCase(input='2+2', expected='4')]\n"
        "def load_evaluations(environment=None): return [Suite(environment)]\n"
    )
    marker = object()

    suites = load_evaluation_suites(env_dir, environment=marker)

    assert [suite.name for suite in suites] == ["held-out"]
    assert suites[0].environment is marker
    assert suites[0].cases() == [EvalCase(input="2+2", expected="4")]


def test_load_evaluation_suites_supports_zero_arg_factory(tmp_path) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'zero-arg'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(): return [Suite()]\n"
    )

    suites = load_evaluation_suites(env_dir, environment=object())

    assert [suite.name for suite in suites] == ["zero-arg"]


def test_load_evaluation_suites_supports_module_fallback(tmp_path) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'fallback'\n"
        "    def cases(self): return []\n"
        "EVALUATIONS = (Suite(),)\n"
    )

    assert [suite.name for suite in load_evaluation_suites(env_dir)] == ["fallback"]


def test_has_evaluations_checks_existence_without_importing(tmp_path) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text("this is not valid python\n")

    assert has_evaluations(env_dir) is True


def test_load_evaluation_suites_reports_malformed_sidecar_path(tmp_path) -> None:
    env_dir = _environment_dir(tmp_path)
    sidecar = env_dir / "evaluations.py"
    sidecar.write_text("EVALUATIONS = [object()]\n")

    try:
        load_evaluation_suites(env_dir)
    except TypeError as exc:
        message = str(exc)
    else:
        raise AssertionError("malformed evaluation suite unexpectedly loaded")

    assert str(sidecar) in message
    assert "non-empty string name" in message


def test_base_eval_suite_does_not_pass_empty_expected() -> None:
    suite = BaseEvalSuite()
    result = suite.score(EvalCase(input="say anything", expected=""), "anything")

    assert result.passed is False
    assert result.score == 0.0


def test_normalize_eval_result_accepts_bool_and_float() -> None:
    case = EvalCase(id="one", input="prompt")

    passed = normalize_eval_result(case, "answer", True)
    scored = normalize_eval_result(case, "answer", 0.25)

    assert passed == EvalResult(case_id="one", passed=True, score=1.0, response="answer")
    assert scored == EvalResult(case_id="one", passed=True, score=0.25, response="answer")


def test_eval_suite_report_math_and_zero_case_edge() -> None:
    report = EvalSuiteReport(
        name="math",
        results=(
            EvalResult(case_id="a", passed=True, score=1.0, response="a"),
            EvalResult(case_id="b", passed=False, score=0.25, response="b"),
        ),
    )
    empty = EvalSuiteReport(name="empty", results=())

    assert report.passed == 1
    assert report.total == 2
    assert report.pass_rate == 0.5
    assert report.mean_score == 0.625
    assert empty.passed == 0
    assert empty.total == 0
    assert empty.pass_rate == 0.0
    assert empty.mean_score == 0.0


def test_eval_suite_report_excludes_transport_errors_from_metrics() -> None:
    # a case that never reached the model is missing data. averaging its 0.0 in would report
    # 50% for a suite whose one real measurement passed, hiding a broken deployment as low quality.
    report = EvalSuiteReport(
        name="partial",
        results=(
            EvalResult(case_id="ok", passed=True, score=1.0, response="a"),
            EvalResult(
                case_id="dead", passed=False, score=0.0, response="", error="generation failed"
            ),
        ),
    )

    assert report.errors == 1
    assert report.total == 2
    assert report.passed == 1
    assert report.pass_rate == 1.0
    assert report.mean_score == 1.0


def test_env_eval_reports_error_count_and_fails_overall(monkeypatch, tmp_path, capsys) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'mixed'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='good', input='good', expected='good'),\n"
        "        EvalCase(id='dead', input='dead', expected='dead'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class PartialClient:
        def chat_stream(self, target, messages, **kwargs):
            if messages[0]["content"] == "dead":
                raise ConnectionError("connection reset")
            yield "good"

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", PartialClient)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir)]) == 1

    captured = capsys.readouterr()
    # the one case that actually ran passed, so the rate must be 100% over real measurements...
    assert "suite mixed: 1/2 passed pass_rate=100.00% mean_score=1.000000" in captured.out
    # ...and the unreached case must still be visible and still fail the run.
    assert "errors=1 (excluded from pass_rate and mean_score)" in captured.out
    assert "overall: FAIL" in captured.err


@pytest.mark.parametrize(
    "target",
    ["flash-1/step-3", "flash-1@step-3." + "a" * 40],
)
def test_env_eval_scores_deployed_target_offline(monkeypatch, tmp_path, capsys, target) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'math'\n"
        "    def cases(self): return [EvalCase(id='sum', input='2+2', expected='4')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client:
        def __init__(self):
            self.calls = []

        def chat_stream(self, target, messages, **kwargs):
            self.calls.append((target, messages, kwargs))
            yield "4"

    client = Client()
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", lambda: client)

    result = cli.main(
        [
            "env",
            "eval",
            target,
            str(env_dir),
            "--temperature",
            "0.2",
            "--max-tokens",
            "17",
        ]
    )

    assert result == 0
    assert client.calls == [
        (
            target,
            [{"role": "user", "content": "2+2"}],
            {"temperature": 0.2, "max_tokens": 17},
        )
    ]
    output = capsys.readouterr().out
    assert "case sum: PASS score=1.000000" in output
    assert "suite math: 1/1 passed pass_rate=100.00% mean_score=1.000000" in output
    assert "overall: PASS" in output


def test_env_eval_pins_bare_run_alias_before_generating_and_uploading(
    monkeypatch, tmp_path, capsys
) -> None:
    # a bare run id is a mutable deployment alias. resolving it once prevents later cases from
    # reaching a replacement adapter while one report claims every score came from one model.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'pinned'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='first', input='first', expected='first'),\n"
        "        EvalCase(id='second', input='second', expected='second'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    revision = "flash-1@final." + "a" * 40

    class Client:
        def __init__(self):
            self.deployment_calls = []
            self.targets = []

        def deployment_for(self, run_id):
            self.deployment_calls.append(run_id)
            return {"run_id": run_id, "state": "ready", "adapter_revision": revision}

        def chat_stream(self, target, messages, **kwargs):
            self.targets.append(target)
            yield messages[0]["content"]

    client = Client()
    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", lambda: client)
    _patch_upload(monkeypatch, uploader)

    assert (
        cli.main(["env", "eval", "flash-1", str(env_dir), "--upload", "--project", _PROJECT_ID])
        == 0
    )

    assert client.deployment_calls == ["flash-1"]
    assert client.targets == [revision, revision]
    assert uploader.calls[0]["model"] == revision
    assert f"resolved evaluation target flash-1 to {revision}" in capsys.readouterr().out


def test_env_eval_concurrency_preserves_case_order(monkeypatch, tmp_path, capsys) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'ordered'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='first', input='first', expected='first'),\n"
        "        EvalCase(id='second', input='second', expected='second'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    second_started = threading.Event()

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            prompt = messages[0]["content"]
            if prompt == "first":
                assert second_started.wait(timeout=2)
            else:
                second_started.set()
            yield prompt

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--concurrency", "2"]) == 0

    output = capsys.readouterr().out
    assert output.index("case first: PASS") < output.index("case second: PASS")


def test_env_eval_rejects_invalid_target_before_loading(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )

    assert cli.main(["env", "eval", "bad target"]) == 1
    assert "invalid evaluation target" in capsys.readouterr().err


def _upload_env_dir(tmp_path: Path, *, dead_case: bool = False) -> Path:
    """An env whose suite has one passing case, optionally plus one that never generates."""
    env_dir = _environment_dir(tmp_path)
    extra = "        EvalCase(id='dead', input='dead', expected='dead'),\n" if dead_case else ""
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'math'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='sum', input='2+2', expected='4'),\n"
        f"{extra}"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    return env_dir


class _RecordingUpload:
    """Stands in for flash.client.upload_eval_run, capturing the kwargs it was called with."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True, "eval_run_id": "run-1"}


def _patch_upload(monkeypatch, uploader) -> None:
    monkeypatch.setattr("flash.client.upload_eval_run", uploader, raising=False)
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: ("url", "key-1"))


def test_env_eval_upload_requires_a_project_id(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )
    env_dir = _upload_env_dir(tmp_path)

    # the guard must fire before any generation happens, so no paid work is wasted.
    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload"]) == 1
    assert "--upload requires a valid --project" in capsys.readouterr().err


def test_env_eval_project_without_upload_is_rejected(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )
    env_dir = _upload_env_dir(tmp_path)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--project", _PROJECT_ID]) == 1
    assert "--project only applies with --upload" in capsys.readouterr().err


def test_env_eval_without_upload_never_calls_the_api(monkeypatch, tmp_path) -> None:
    env_dir = _upload_env_dir(tmp_path)

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir)]) == 0
    # the default must stay local-only: evaluating without --upload writes nothing.
    assert uploader.calls == []


def test_env_eval_upload_sends_every_case_with_the_project_id(monkeypatch, tmp_path) -> None:
    env_dir = _upload_env_dir(tmp_path)

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert (
        cli.main(
            ["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload", "--project", _PROJECT_ID]
        )
        == 0
    )

    assert len(uploader.calls) == 1
    call = uploader.calls[0]
    assert call["project_id"] == _PROJECT_ID
    assert call["suite_name"] == "math"
    assert call["model"] == _EXPLICIT_TARGET
    assert [case["case_id"] for case in call["cases"]] == ["sum"]
    assert call["cases"][0]["success"] is True
    assert call["cases"][0]["actual"] == "4"
    assert call["cases"][0]["expected"] == "4"


def test_env_eval_upload_reports_an_errored_case_verbatim(monkeypatch, tmp_path) -> None:
    env_dir = _upload_env_dir(tmp_path, dead_case=True)

    class PartialClient:
        def chat_stream(self, target, messages, **kwargs):
            if messages[0]["content"] == "dead":
                raise ConnectionError("connection reset")
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", PartialClient)
    _patch_upload(monkeypatch, uploader)

    # the suite fails overall because one case never generated...
    assert (
        cli.main(
            ["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload", "--project", _PROJECT_ID]
        )
        == 1
    )

    # ...but both cases are still uploaded, and the failure carries its error rather than
    # arriving as a silent zero the server would average in as real model behaviour.
    cases = {case["case_id"]: case for case in uploader.calls[0]["cases"]}
    assert set(cases) == {"sum", "dead"}
    assert cases["dead"]["error"] is not None
    assert "generation failed" in cases["dead"]["error"]
    assert cases["sum"]["error"] is None


def test_env_eval_upload_records_suites_that_cannot_load_cases(monkeypatch, tmp_path) -> None:
    # a failed or empty cases() result is the suite verdict, so omitting it from --upload leaves
    # the dashboard with a partial run that hides the exact failure responsible for the cli exit.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite\n"
        "class BrokenSuite(BaseEvalSuite):\n"
        "    name = 'broken'\n"
        "    def cases(self): raise RuntimeError('dataset missing')\n"
        "class EmptySuite(BaseEvalSuite):\n"
        "    name = 'empty'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [BrokenSuite(), EmptySuite()]\n"
    )

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError("case loading failures must not generate")

    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert (
        cli.main(
            ["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload", "--project", _PROJECT_ID]
        )
        == 1
    )

    assert [(call["suite_name"], call["status"]) for call in uploader.calls] == [
        ("broken", "failed"),
        ("empty", "failed"),
    ]
    assert "case loading failed" in uploader.calls[0]["error"]
    assert "dataset missing" in uploader.calls[0]["error"]
    assert uploader.calls[1]["error"] == "suite produced no cases"
    assert [call["cases"][0]["case_id"] for call in uploader.calls] == ["load", "load"]
    assert all(call["cases"][0]["input"] is None for call in uploader.calls)


def test_env_eval_rejects_a_malformed_project_before_generating(
    monkeypatch, tmp_path, capsys
) -> None:
    # upload_eval_run requires a canonical UUID. validating only that --project is nonblank
    # would buy every model request and then fail at the upload -- and because upload failure
    # deliberately does not change the verdict, the run would still print `overall: PASS`
    # with nothing recorded anywhere.
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )
    env_dir = _upload_env_dir(tmp_path)

    assert (
        cli.main(
            ["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload", "--project", "not-a-uuid"]
        )
        == 1
    )
    assert "must be a valid UUID" in capsys.readouterr().err


def test_env_eval_upload_keeps_duplicate_case_ids_with_their_own_input(
    monkeypatch, tmp_path
) -> None:
    # results carry the disambiguated id (`same#2`), so a payload keyed on the raw `case.id`
    # both drops the second case and hands the first result the *second* case's input and
    # expected value. that uploads a graded case describing a question it never answered.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'dupes'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='same', input='first', expected='first'),\n"
        "        EvalCase(id='same', input='second', expected='second'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield messages[0]["content"]

    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert (
        cli.main(
            ["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload", "--project", _PROJECT_ID]
        )
        == 0
    )

    cases = {case["case_id"]: case for case in uploader.calls[0]["cases"]}
    assert set(cases) == {"same", "same#2"}
    assert cases["same"]["input"] == "first"
    assert cases["same"]["expected"] == "first"
    assert cases["same#2"]["input"] == "second"
    assert cases["same#2"]["expected"] == "second"


def test_env_eval_upload_failure_does_not_relabel_a_passing_suite(
    monkeypatch, tmp_path, capsys
) -> None:
    env_dir = _upload_env_dir(tmp_path)

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    from flash.client import ClientError

    def failing_upload(**kwargs):
        raise ClientError("freesolo is unreachable")

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, failing_upload)

    # the suite genuinely passed; a failed upload is reported but must not turn it into a
    # FAIL, which would read as the model having gotten the answer wrong.
    assert (
        cli.main(
            ["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload", "--project", _PROJECT_ID]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "upload failed" in captured.err
    assert "overall: PASS" in captured.out


def test_env_eval_upload_without_login_reports_the_missing_key(
    monkeypatch, tmp_path, capsys
) -> None:
    env_dir = _upload_env_dir(tmp_path)

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)
    monkeypatch.setattr("flash.client.upload_eval_run", uploader, raising=False)
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: ("url", None))

    assert (
        cli.main(
            ["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload", "--project", _PROJECT_ID]
        )
        == 0
    )
    assert "not logged in" in capsys.readouterr().err
    # no key means no request was attempted at all.
    assert uploader.calls == []


def test_env_eval_blank_stream_errors_without_scoring(monkeypatch, tmp_path, capsys) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'blank'\n"
        "    def cases(self): return [EvalCase(id='one', input='hello', expected='hi')]\n"
        "    def score(self, case, response): raise AssertionError('score must not run')\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class BlankClient:
        def chat_stream(self, target, messages, **kwargs):
            yield "   "

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", BlankClient)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir)]) == 1

    captured = capsys.readouterr()
    assert "case one: FAIL score=0.000000" in captured.out
    assert f"generation failed: no response text from {_EXPLICIT_TARGET}" in captured.out
    assert "score must not run" not in captured.out + captured.err
    assert "overall: FAIL" in captured.err


def test_base_eval_suite_can_match_a_falsy_expected_answer() -> None:
    # `expected or ""` would erase these: 0 and False are real gold answers, and treating
    # them as "no gold" makes every such case permanently unpassable.
    suite = BaseEvalSuite()

    assert suite.score(EvalCase(input="how many?", expected=0), "the answer is 0").passed is True
    assert suite.score(EvalCase(input="true?", expected=False), "False").passed is True
    assert suite.score(EvalCase(input="how many?", expected=0), "the answer is 5").passed is False
    # an absent expected still has nothing to compare against.
    assert suite.score(EvalCase(input="anything", expected=None), "whatever").passed is False


def test_env_eval_scoring_that_exits_fails_only_its_own_case(monkeypatch, tmp_path, capsys) -> None:
    # sys.exit() inside score() raises SystemExit, which is not an Exception. Catching only
    # Exception would abort the run and discard every case already graded.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "import sys\n"
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'exiting'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='ok', input='ok', expected='ok'),\n"
        "        EvalCase(id='exits', input='exits', expected='exits'),\n"
        "    ]\n"
        "    def score(self, case, response):\n"
        "        if case.id == 'exits': sys.exit('scorer bailed')\n"
        "        return super().score(case, response)\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield messages[0]["content"]

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir)]) == 1

    captured = capsys.readouterr()
    # the case that exited is recorded as a scoring error...
    assert "case exits: FAIL" in captured.out
    assert "scoring failed: scorer bailed" in captured.out
    # ...and the case graded before it survives rather than being lost with the process.
    assert "case ok: PASS score=1.000000" in captured.out
    assert "overall: FAIL" in captured.err


def test_env_eval_serializes_scoring_across_worker_threads(monkeypatch, tmp_path, capsys) -> None:
    # a suite is one user object shared by every worker. a scorer holding mutable state that
    # is raced produces silently wrong scores, which is worse than scoring serially.
    env_dir = _environment_dir(tmp_path)
    witness = tmp_path / "max_inside.txt"
    # the sidecar records the peak overlap to a file: the suite object that actually ran
    # lives inside the command, and reloading the module would build a fresh one.
    (env_dir / "evaluations.py").write_text(
        "import time\n"
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        f"WITNESS = {str(witness)!r}\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'racy'\n"
        "    def __init__(self):\n"
        "        self.inside = 0\n"
        "        self.max_inside = 0\n"
        "    def cases(self): return [\n"
        "        EvalCase(id=f'c{i}', input=f'c{i}', expected=f'c{i}') for i in range(6)\n"
        "    ]\n"
        "    def score(self, case, response):\n"
        "        self.inside += 1\n"
        "        self.max_inside = max(self.max_inside, self.inside)\n"
        "        time.sleep(0.02)\n"
        "        self.inside -= 1\n"
        "        open(WITNESS, 'w').write(str(self.max_inside))\n"
        "        return super().score(case, response)\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    generating = threading.Barrier(6, timeout=5)

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            # force all six generations to be genuinely in flight together, so any
            # unserialized scoring really would overlap.
            generating.wait()
            yield messages[0]["content"]

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--concurrency", "6"]) == 0

    # never more than one scorer in flight, despite six concurrent generations.
    assert witness.read_text() == "1"
    assert "overall: PASS" in capsys.readouterr().out


def test_env_eval_empty_suite_is_not_a_pass(monkeypatch, tmp_path, capsys) -> None:
    # 0/0 graded is not a green suite. reporting PASS here hides a sidecar whose cases()
    # silently returned nothing.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'empty'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError("no cases means no generation")

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir)]) == 1

    captured = capsys.readouterr()
    assert "suite empty has no cases to run" in captured.err
    assert "overall: FAIL" in captured.err
    assert "overall: PASS" not in captured.out


def test_env_eval_disambiguates_duplicate_case_ids(monkeypatch, tmp_path, capsys) -> None:
    # two cases sharing an id would collide in the report and in the uploaded payload,
    # showing one graded case where two ran.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'dupes'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='same', input='a', expected='a'),\n"
        "        EvalCase(id='same', input='b', expected='b'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield messages[0]["content"]

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir)]) == 0

    output = capsys.readouterr().out
    assert "case same: PASS" in output
    assert "case same#2: PASS" in output
    assert "suite dupes: 2/2 passed" in output


def test_env_eval_debug_surfaces_the_load_traceback(monkeypatch, tmp_path) -> None:
    # a broken sidecar is a bug, and --debug asked for its traceback. swallowing it into a
    # one-line message leaves nothing to debug with.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text("raise RuntimeError('sidecar exploded')\n")
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())

    with pytest.raises(RuntimeError, match="sidecar exploded"):
        cli.main(["--debug", "env", "eval", _EXPLICIT_TARGET, str(env_dir)])


def test_evaluation_sidecar_can_import_helpers_lazily(tmp_path) -> None:
    # the package dir must stay importable after load: sidecars import local helpers inside
    # cases()/score(), which run long after the module is executed.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "lazy_helper.py").write_text("GOLD = 'deferred'\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'lazy'\n"
        "    def cases(self):\n"
        "        from lazy_helper import GOLD\n"
        "        return [EvalCase(id='one', input=GOLD, expected=GOLD)]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    suites = load_evaluation_suites(env_dir)

    assert suites[0].cases() == [EvalCase(id="one", input="deferred", expected="deferred")]


def test_evaluation_sidecar_load_does_not_grow_sys_path(tmp_path) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'repeat'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    load_evaluation_suites(env_dir)
    after_first = list(sys.path)
    load_evaluation_suites(env_dir)

    assert sys.path == after_first


def test_case_ids_stay_unique_when_a_raw_id_looks_like_a_suffix(tmp_path) -> None:
    # ids "x", "x", "x#2" disambiguate the second "x" to "x#2", which is already a raw id. a
    # counter that does not re-check produces two "x#2" entries, so the uploaded payload and
    # the printed report collapse two graded cases into one and silently drop a result.
    from flash.cli.env_eval import _case_ids

    cases = [
        EvalCase(id="x", input="a"),
        EvalCase(id="x", input="b"),
        EvalCase(id="x#2", input="c"),
    ]

    ids = _case_ids(cases)

    assert len(set(ids)) == len(ids)
    assert ids[0] == "x"


def test_env_eval_scores_on_the_calling_thread(monkeypatch, tmp_path) -> None:
    # a scorer that installs a signal-based timeout raises "signal only works in main thread"
    # off-thread. serializing with a lock still runs it in a worker, so the failure would be
    # recorded as the model failing every case rather than as a broken harness.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "import threading\n"
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'threads'\n"
        "    def cases(self): return [EvalCase(id=str(i), input='q') for i in range(4)]\n"
        "    def score(self, case, response):\n"
        "        assert threading.current_thread() is threading.main_thread(), 'scored off-thread'\n"
        "        return 1.0\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield "ok"

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--concurrency", "4"]) == 0


def test_env_eval_concurrent_results_stay_in_case_order(monkeypatch, tmp_path) -> None:
    # generation completes out of order under --concurrency. pairing results with case ids by
    # completion order rather than submission index would hand each result another case's id,
    # mislabelling every graded case in the report and the upload.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase, EvalResult\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'ordered'\n"
        "    def cases(self):\n"
        "        return [EvalCase(id=str(i), input=str(i)) for i in range(6)]\n"
        "    def score(self, case, response):\n"
        "        return EvalResult(case_id=case.id, passed=case.input == response,\n"
        "                          score=1.0, response=response)\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class SlowFirstClient:
        def chat_stream(self, target, messages, **kwargs):
            content = messages[0]["content"]
            # case 0 finishes last, so completion order is the reverse of submission order.
            time.sleep(0.05 if content == "0" else 0.0)
            yield content

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", SlowFirstClient)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--concurrency", "6"]) == 0


def test_load_evaluations_accepts_a_positional_only_environment(tmp_path) -> None:
    # `load_evaluations(environment, /)` declares the parameter but cannot take it by name.
    # filtering on membership alone raises TypeError; dropping it instead would hand the suite
    # environment=None, silently downgrading a real scorer to substring matching.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'positional'\n"
        "    def __init__(self, environment): self.environment = environment\n"
        "    def cases(self): return [EvalCase(input='2+2', expected='4')]\n"
        "def load_evaluations(environment, /): return [Suite(environment)]\n"
    )
    marker = object()

    suites = load_evaluation_suites(env_dir, environment=marker)

    assert suites[0].environment is marker


def test_env_eval_forwards_environment_params_to_the_loader(monkeypatch, tmp_path) -> None:
    # an environment whose load_environment() requires a param cannot load without it, and one
    # that branches on --split would be graded against a different dataset than it trains on.
    env_dir = _upload_env_dir(tmp_path)
    seen: dict = {}

    def _loader(path, **kwargs):
        seen.update(kwargs)
        return object()

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", _loader)
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert (
        cli.main(
            [
                "env",
                "eval",
                _EXPLICIT_TARGET,
                str(env_dir),
                "--split",
                "held_out",
                "--param",
                "difficulty=3",
            ]
        )
        == 0
    )

    assert seen == {"split": "held_out", "difficulty": 3}


def test_env_eval_rejects_non_finite_temperature(monkeypatch, capsys) -> None:
    # the chat route refuses non-finite temperatures per request, so without an argparse-level
    # guard every case submits the bad value and comes back a generation failure -- one
    # malformed flag reading as the model failing the whole suite.
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )

    for bad in ("nan", "inf", "-inf"):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(["env", "eval", _EXPLICIT_TARGET, "--temperature", bad])
        assert excinfo.value.code == 2
    assert "must be a finite number" in capsys.readouterr().err


def test_env_eval_uploads_each_suite_with_its_own_start_time(
    monkeypatch, tmp_path
) -> None:
    # every suite uploads as its own run. sharing one timestamp backdates each later run to
    # before the earlier suites ran and inflates its dashboard duration with their work.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class First(BaseEvalSuite):\n"
        "    name = 'first'\n"
        "    def cases(self): return [EvalCase(id='a', input='a', expected='a')]\n"
        "class Second(BaseEvalSuite):\n"
        "    name = 'second'\n"
        "    def cases(self): return [EvalCase(id='b', input='b', expected='b')]\n"
        "def load_evaluations(environment=None): return [First(), Second()]\n"
    )

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            time.sleep(0.01)
            yield messages[0]["content"]

    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert (
        cli.main(
            ["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload", "--project", _PROJECT_ID]
        )
        == 0
    )

    starts = [call["started_at"] for call in uploader.calls]
    assert len(starts) == 2
    # the second suite cannot have started before the first one finished its work.
    assert starts[1] > starts[0]


def test_env_eval_reports_upload_timeout_without_a_traceback(
    monkeypatch, tmp_path, capsys
) -> None:
    # a socket timeout surfaces as a bare TimeoutError, not a URLError. untranslated it escapes
    # the upload handler after the whole paid evaluation ran, replacing the promised nonfatal
    # message and final verdict with a traceback. the real upload path runs here, so the socket
    # is what fails -- patching upload_eval_run itself would skip the translation being tested.
    env_dir = _upload_env_dir(tmp_path)

    class Client:
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    def _timeout(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("flash.envs.loader.load_freesolo_environment", lambda _path: object())
    monkeypatch.setattr("flash.client.client_from_config", Client)
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: ("url", "key-1"))
    monkeypatch.setattr("urllib.request.urlopen", _timeout)

    assert (
        cli.main(
            ["env", "eval", _EXPLICIT_TARGET, str(env_dir), "--upload", "--project", _PROJECT_ID]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "upload failed" in captured.err
    # the suite itself passed, so the verdict must survive the upload failure.
    assert "overall: PASS" in captured.out


def test_freesolo_request_translates_socket_timeout(monkeypatch) -> None:
    from flash.client.http import RequestTimeoutError, _freesolo_request

    def _timeout(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", _timeout)

    with pytest.raises(RequestTimeoutError) as excinfo:
        _freesolo_request("POST", "/v1/eval-runs", "key-1", base_url="https://example.test")
    assert "timed out" in str(excinfo.value)


def test_load_evaluations_receives_the_environment_after_other_positional_parameters(
    tmp_path,
) -> None:
    # a factory whose `environment` follows another positional-only parameter used to stop the
    # match at the earlier name and pass `environment` by keyword, which positional-only
    # syntax forbids -- so the sidecar failed to load rather than being graded by its scorer.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'positional'\n"
        "    def __init__(self, environment): self.environment = environment\n"
        "    def cases(self): return [EvalCase(id='a', input='a', expected='a')]\n"
        "def load_evaluations(options=None, environment=None, /):\n"
        "    return [Suite(environment)]\n"
    )
    sentinel = object()

    suites = load_evaluation_suites(env_dir / "environment.py", environment=sentinel)

    # the real environment must arrive; None would silently downgrade a scorer that needs it.
    assert suites[0].environment is sentinel
