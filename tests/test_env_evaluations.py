"""Tests for Flash-native environment evaluation sidecars and CLI execution."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import flash.cli.parsing.main as cli
from flash.envs.meta.evaluations import (
    BaseEvalSuite,
    EvalCase,
    EvalResult,
    EvalSuiteReport,
    has_evaluations,
    load_evaluation_suites,
    normalize_eval_result,
)

# an explicit --project is validated before spending anything, so tests must pass a real UUID.
_PROJECT_ID = "11111111-1111-1111-1111-111111111111"
_PUBLISHED_SLUG = "acme/example-project/starter"
# a full immutable revision: the one target shape that needs no resolution, so a test about
# anything else does not have to stub `deployments()`. `RUN/step-N` is a shorthand the CLI now
# pins first, which is its own contract (`test_env_eval_pins_a_step_shorthand_...`).
_EXPLICIT_TARGET = "flash-1/step-3"


def _run_record(spec: dict) -> dict:
    return {"verified_checkpoints": [_EXPLICIT_TARGET], "spec": spec}


class _EvalClient:
    """Base for a double used by an environment evaluation.

    Defaults cover spec lookup and package download so individual doubles focus on their behavior.
    Missing package registration fails explicitly.
    """

    _env_dir: Path | None = None

    def get_run(self, run_id):
        return _run_record(
            {
                "thinking": False,
                "project": _PROJECT_ID,
                "environment": {"id": _PUBLISHED_SLUG},
            }
        )

    def deployments(self):
        return [
            {
                "run_id": "flash-1",
                "deployment": {"state": "ready", "checkpoint_id": _EXPLICIT_TARGET},
            }
        ]

    def warm_chat_step_selector(self, target):
        return None

    def download_env_package(self, env_id):
        # packed on download, not on registration: the fixture dir is created first and its
        # evaluations.py written after, so packing eagerly would publish an environment whose
        # sidecar the test had not written yet.
        if self._env_dir is None:
            raise AssertionError(
                f"env eval downloaded {env_id} but this test registered no environment; "
                "call _patch_published_env(monkeypatch, env_dir)"
            )
        return _environment_package_bytes(self._env_dir)


def _environment_package_bytes(env_dir: Path) -> bytes:
    """`env_dir` as the tar.gz the control plane's package route returns.

    A real archive rather than a patched resolver: `env eval` downloads the package and extracts it,
    so packing here leaves both halves live and lets the tests catch a break in either.
    """
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(env_dir.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(env_dir)))
    return buffer.getvalue()


def _patch_published_env(monkeypatch, env_dir: Path) -> None:
    """Serve `env_dir` as the published slug's package, with no network.

    The control plane is the single seam the whole published-environment path now goes through:
    `env eval` downloads the package, extracts it to a temp dir, and grades the environment, the
    suites, and the sidecar from that one extracted copy. Packing the fixture into the archive the
    client returns therefore redirects all of it, and -- unlike patching the resolver -- leaves the
    download, the extraction, and the slug-vs-local-path resolution running for real.
    """
    monkeypatch.setattr(_EvalClient, "_env_dir", env_dir)


def _environment_dir(tmp_path: Path, *, monkeypatch: pytest.MonkeyPatch | None = None) -> Path:
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment():\n    return None\n")
    if monkeypatch is not None:
        _patch_published_env(monkeypatch, env_dir)
    return env_dir


def test_load_evaluation_suites_calls_factory_with_environment(tmp_path) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "eval_sidecar_helper.py").write_text("SUITE_NAME = 'held-out'\n")
    (env_dir / "evaluations.py").write_text(
        "from eval_sidecar_helper import SUITE_NAME\n"
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
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
        "from flash.envs.meta.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'zero-arg'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(): return [Suite()]\n"
    )

    suites = load_evaluation_suites(env_dir, environment=object())

    assert [suite.name for suite in suites] == ["zero-arg"]


def test_load_evaluation_suites_supports_positional_only_factory(tmp_path) -> None:
    # a positional-only parameter is recognized by name but cannot be passed by one, so the
    # signature filter accepted `load_evaluations(environment, /)` and python itself then rejected
    # the call -- a sidecar the documented contract says is supported.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'positional-only'\n"
        "    def __init__(self, environment): self.environment = environment\n"
        "    def cases(self): return [EvalCase(input='2+2', expected='4')]\n"
        "def load_evaluations(environment, /): return [Suite(environment)]\n"
    )
    marker = object()

    suites = load_evaluation_suites(env_dir, environment=marker)

    assert [suite.name for suite in suites] == ["positional-only"]
    # and it receives the environment, rather than merely not crashing
    assert suites[0].environment is marker


def _lazy_helper_env(root: Path, label: str) -> Path:
    """An env whose sidecar imports its sibling helper lazily, inside cases() and score()."""
    env_dir = root / label
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    (env_dir / "helper.py").write_text(f"LABEL = {label!r}\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        f"    name = {label!r}\n"
        "    def cases(self):\n"
        "        from helper import LABEL\n"
        "        return [EvalCase(id=LABEL, input='x', expected='x')]\n"
        "    def score(self, case, response):\n"
        "        from helper import LABEL\n"
        "        return EvalCase and float(response == LABEL)\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    return env_dir


def test_lazily_imported_siblings_stay_bound_to_their_own_environment(tmp_path) -> None:
    """A helper imported inside cases()/score() must be the caller's own, not the last one loaded.

    Plain names cached in ``sys.modules`` can silently make a later environment use earlier grading
    code, so lazy imports must remain scoped to their package.
    """
    first = _lazy_helper_env(tmp_path, "alpha")
    second = _lazy_helper_env(tmp_path, "beta")

    alpha = load_evaluation_suites(first, environment=None)[0]
    beta = load_evaluation_suites(second, environment=None)[0]

    # interleaved on purpose: the binding must survive another package being loaded in between.
    assert [case.id for case in alpha.cases()] == ["alpha"]
    assert [case.id for case in beta.cases()] == ["beta"]
    # scoring resolves the sibling independently of cases(), so it is checked independently.
    assert alpha.score(EvalCase(id="a", input="x", expected="x"), "alpha") == 1.0
    assert beta.score(EvalCase(id="b", input="x", expected="x"), "beta") == 1.0
    # and the process is left as it was found, so a long CLI run does not accumulate state.
    assert [path for path in sys.path if str(tmp_path) in path] == []
    assert "helper" not in sys.modules


def _package_helper_env(root: Path, label: str, *, namespace: bool) -> Path:
    """An env whose sidecar eagerly imports a sibling PACKAGE, `from graders.rules import LABEL`."""
    env_dir = root / label
    (env_dir / "graders").mkdir(parents=True)
    (env_dir / "environment.py").write_text("def load_environment(**k):\n    return None\n")
    if not namespace:
        (env_dir / "graders" / "__init__.py").write_text("")
    (env_dir / "graders" / "rules.py").write_text(f"LABEL = {label!r}\n")
    (env_dir / "evaluations.py").write_text(
        "from graders.rules import LABEL\n"
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        f"    name = {label!r}\n"
        "    def cases(self): return [EvalCase(id=LABEL, input='x', expected='x')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    return env_dir


@pytest.mark.parametrize("namespace", [False, True], ids=["regular-package", "namespace-package"])
def test_sibling_packages_do_not_leak_between_environments(tmp_path, namespace: bool) -> None:
    """A sidecar's sibling package belongs to its own environment, not the first one loaded.

    Evict nested submodules too; namespace packages may have no ``__init__.py`` or spec origin.
    """
    first = _package_helper_env(tmp_path, "alpha", namespace=namespace)
    second = _package_helper_env(tmp_path, "beta", namespace=namespace)

    alpha = load_evaluation_suites(first, environment=None)[0]
    beta = load_evaluation_suites(second, environment=None)[0]

    assert [case.id for case in alpha.cases()] == ["alpha"]
    assert [case.id for case in beta.cases()] == ["beta"]
    assert [name for name in sys.modules if name.split(".")[0] == "graders"] == []


def test_load_evaluation_suites_supports_module_fallback(tmp_path) -> None:
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite\n"
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


def test_eval_result_rejects_a_pass_that_also_reports_an_error() -> None:
    # an error means the case was never graded, so three readings of it disagreed: the report
    # excluded it and failed the command, the console printed PASS, and the upload recorded
    # `success: true` beside the error. one contract, checked where it is built.
    with pytest.raises(ValueError, match="passed must be False when an error is reported"):
        EvalResult(case_id="a", passed=True, score=1.0, response="4", error="judge unavailable")

    # a failed result carrying an error is the normal shape and still builds.
    errored = EvalResult(
        case_id="a", passed=False, score=0.0, response="4", error="judge unavailable"
    )
    assert errored.error == "judge unavailable"


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
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'mixed'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='good', input='good', expected='good'),\n"
        "        EvalCase(id='dead', input='dead', expected='dead'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class PartialClient(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            if messages[0]["content"] == "dead":
                raise ConnectionError("connection reset")
            yield "good"

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", PartialClient)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1

    captured = capsys.readouterr()
    # the one case that actually ran passed, so the rate must be 100% over real measurements...
    assert "suite mixed: 1/2 passed pass_rate=100.00% mean_score=1.000000" in captured.out
    # ...and the unreached case must still be visible and still fail the run.
    assert "errors=1 (excluded from pass_rate and mean_score)" in captured.out
    assert "overall: FAIL" in captured.err


def test_env_eval_scores_deployed_target_offline(monkeypatch, tmp_path, capsys) -> None:
    """A full immutable revision is the one target that needs no resolution.

    The shorthands -- a bare run id and `RUN/step-N` -- each name something the CLI must pin to a
    revision first, which is their own contract (`test_env_eval_pins_...`).
    """
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'math'\n"
        "    def cases(self): return [EvalCase(id='sum', input='2+2', expected='4')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client(_EvalClient):
        def __init__(self):
            self.calls = []
            self.run_calls = 0

        def get_run(self, run_id):
            self.run_calls += 1
            if self.run_calls > 1:
                raise RuntimeError("a second run lookup is unavailable")
            return super().get_run(run_id)

        def chat_stream(self, target, messages, **kwargs):
            self.calls.append((target, messages, kwargs))
            yield "4"

    client = Client()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", lambda: client)

    result = cli.main(
        [
            "env",
            "eval",
            _EXPLICIT_TARGET,
            "--temperature",
            "0.2",
            "--max-tokens",
            "17",
            "--no-upload",
        ]
    )

    assert result == 0
    assert client.run_calls == 1
    assert client.calls == [
        (
            _EXPLICIT_TARGET,
            [{"role": "user", "content": "2+2"}],
            {"temperature": 0.2, "max_tokens": 17},
        )
    ]
    output = capsys.readouterr().out
    assert "case sum: PASS score=1.000000" in output
    assert "suite math: 1/1 passed pass_rate=100.00% mean_score=1.000000" in output
    assert "overall: PASS" in output


def test_env_eval_refuses_a_pinned_step_whose_run_lost_its_verified_ledger(
    monkeypatch, tmp_path, capsys
) -> None:
    """a pinned checkpoint without verified membership must fail before evaluation starts."""
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def __init__(self):
            self.targets = []

        def get_run(self, run_id):
            return {
                "verified_checkpoints": [],
                "spec": {"thinking": False, "environment": {"id": _PUBLISHED_SLUG}},
            }

        def warm_chat_step_selector(self, target):
            return None

        def chat_stream(self, target, messages, **kwargs):
            self.targets.append(target)
            yield "4"

    client = Client()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", lambda: client)

    assert cli.main(["env", "eval", "flash-1/step-3"]) == 1
    captured = capsys.readouterr()
    assert "has not passed deployment verification" in captured.err
    # one failure up front, not one per case: no generation was attempted.
    assert client.targets == []


def test_env_eval_concurrency_preserves_case_order(monkeypatch, tmp_path, capsys) -> None:
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'ordered'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='first', input='first', expected='first'),\n"
        "        EvalCase(id='second', input='second', expected='second'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    second_started = threading.Event()

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            prompt = messages[0]["content"]
            if prompt == "first":
                assert second_started.wait(timeout=2)
            else:
                second_started.set()
            yield prompt

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--concurrency", "2", "--no-upload"]) == 0

    output = capsys.readouterr().out
    assert output.index("case first: PASS") < output.index("case second: PASS")


def test_env_eval_rejects_invalid_target_before_loading(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )

    assert cli.main(["env", "eval", "bad target"]) == 1
    assert "invalid evaluation target" in capsys.readouterr().err


def test_env_eval_rejects_a_local_directory_positional(tmp_path, capsys) -> None:
    """The evaluated run now supplies the environment, so a second positional is a usage error."""
    env_dir = _environment_dir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["env", "eval", _EXPLICIT_TARGET, str(env_dir)])

    assert excinfo.value.code == 2
    captured = capsys.readouterr().err
    assert "unrecognized arguments" in captured
    assert str(env_dir) in captured


def test_env_eval_no_upload_still_requires_a_published_environment(monkeypatch, capsys) -> None:
    """Opting out of recording does not invent the environment whose suites should be graded."""

    class Client(_EvalClient):
        def __init__(self):
            self.generated = 0

        def get_run(self, run_id):
            return {
                "verified_checkpoints": [_EXPLICIT_TARGET],
                "spec": {"thinking": False, "project": _PROJECT_ID},
            }

        def chat_stream(self, target, messages, **kwargs):
            self.generated += 1
            yield "4"

    client = Client()
    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.client.client_from_config", lambda: client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1
    assert client.generated == 0
    assert uploader.calls == []
    assert "trains on no published environment" in capsys.readouterr().err


def test_env_eval_refuses_a_published_environment_without_a_sidecar(
    monkeypatch, tmp_path, capsys
) -> None:
    """A published environment must carry the held-out suites that define its evaluation."""
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError("a missing sidecar must fail before generation")

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1
    assert (
        "no evaluations.py sidecar found beside the environment entrypoint"
        in capsys.readouterr().err
    )


def _upload_env_dir(
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch | None = None,
    dead_case: bool = False,
) -> Path:
    """An env whose suite has one passing case, optionally plus one that never generates."""
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    extra = "        EvalCase(id='dead', input='dead', expected='dead'),\n" if dead_case else ""
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
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
    # the accessibility preflight is exercised by its own tests; here it just has to say yes.
    monkeypatch.setattr(
        "flash.client.get_project", lambda project_id, _api_key: {"id": project_id}, raising=False
    )


def test_env_eval_records_the_published_environment_it_graded(monkeypatch, tmp_path) -> None:
    """Two published environments must not upload the same provenance.

    The report identifies the published environment that supplied the scorer. Distinct slugs must
    stay distinct even when both resolve onto local test fixtures, or the dashboard would merge two
    different measurements under one environment.
    """

    class Client(_EvalClient):
        def __init__(self, slug):
            self.slug = slug

        def get_run(self, run_id):
            return _run_record(
                {
                    "thinking": False,
                    "project": _PROJECT_ID,
                    "environment": {"id": self.slug},
                }
            )

        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    _patch_upload(monkeypatch, uploader)

    slugs = ["acme/example-project/easy", "acme/example-project/hard"]
    for slug in slugs:
        root = tmp_path / slug.rsplit("/", 1)[-1]
        root.mkdir()
        env_dir = _upload_env_dir(root)
        _patch_published_env(monkeypatch, env_dir)
        monkeypatch.setattr("flash.client.client_from_config", lambda slug=slug: Client(slug))

        assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 0

    assert [call["environment_reference"] for call in uploader.calls] == slugs


def test_env_eval_refuses_to_record_a_run_whose_project_is_unknown(
    monkeypatch, tmp_path, capsys
) -> None:
    """Recording is the default, but never under a project nobody named.

    The project comes from the evaluated run itself. When that run names none and the user did not
    either, there is no correct place to file a permanent result -- and picking a first, sole, or
    example project would file it somewhere the user never chose. Refuse before any paid work, and
    name both ways out."""
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def get_run(self, run_id):
            return {
                "verified_checkpoints": [_EXPLICIT_TARGET],
                "spec": {"thinking": False, "environment": {"id": _PUBLISHED_SLUG}},
            }

        def warm_chat_step_selector(self, target):
            return None

        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError("no case may generate with nowhere to record it")
            yield ""

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET]) == 1
    captured = capsys.readouterr().err
    assert "its project is unknown" in captured
    assert "--project" in captured
    assert "--no-upload" in captured
    assert uploader.calls == []


def test_env_eval_upload_rejects_a_non_uuid_project_before_paying(
    monkeypatch, tmp_path, capsys
) -> None:
    # the preflight only tested for non-blank while `upload_eval_run` requires a canonical UUID, so
    # `proj-9` bought every model request and then failed at upload with nowhere to put the results
    # -- the waste this guard exists to prevent.
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", "proj-9"]) == 1
    captured = capsys.readouterr().err
    assert "--project must be a valid PROJECT_ID" in captured
    assert "valid UUID" in captured


@pytest.mark.parametrize("status", [403, 404])
def test_env_eval_upload_rejects_an_inaccessible_project_before_paying(
    monkeypatch, tmp_path, capsys, status: int
) -> None:
    """A well-formed UUID is not a project this caller can upload to.

    Validating only the shape meant a deleted project, or one belonging to another organization,
    bought every model request and was rejected at upload -- and because upload failure
    deliberately does not change the verdict, the run still printed `overall: PASS` with nothing
    recorded anywhere. `env setup` already resolves the project before scaffolding.
    """
    from flash.client import ApiError

    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )
    # a HOSTED url: ownership is resolved against the backend only when the plane is Freesolo's,
    # so a placeholder that reads as self-hosted would take the shape-only branch and never reach
    # `get_project` -- disarming the very refusal this test exists to prove.
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("https://flash.freesolo.co", "key-1")
    )

    def _denied(project_id, api_key, api_url=None):
        raise ApiError(status, "denied")

    monkeypatch.setattr("flash.client.get_project", _denied, raising=False)
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 1
    captured = capsys.readouterr().err
    assert "--project must be a valid PROJECT_ID" in captured
    assert "is not accessible" in captured


def test_env_eval_preflight_never_shows_the_plane_key_to_the_hosted_api(monkeypatch) -> None:
    """A self-hosted preflight shape-checks locally instead of asking api.freesolo.co.

    The stored key on such a plane IS FREESOLO_INTERNAL_KEY, which controls the plane, so an
    ownership lookup against the hosted backend would leak it (SELF_HOSTING.md) -- and be rejected,
    breaking `flash env eval` for every self-hosted operator.
    """
    from flash.cli.commands.env.testing.eval import _require_accessible_project

    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("http://127.0.0.1:8080", "operator-key")
    )

    def _leaked(*_args, **_kwargs):
        raise AssertionError("the self-hosted plane key must never reach the hosted freesolo api")

    monkeypatch.setattr("flash.client.get_project", _leaked, raising=False)

    assert _require_accessible_project(_PROJECT_ID) == _PROJECT_ID


def test_env_eval_preflight_still_resolves_ownership_when_hosted(monkeypatch) -> None:
    """The hosted path keeps its real accessibility check, so an inaccessible project still refuses."""
    from flash.cli.commands.env.testing.eval import _require_accessible_project
    from flash.client import ApiError, ClientError

    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("https://flash.freesolo.co", "key-1")
    )

    def _denied(*_args, **_kwargs):
        raise ApiError(403, "denied")

    monkeypatch.setattr("flash.client.get_project", _denied, raising=False)

    with pytest.raises(ClientError, match="is not accessible"):
        _require_accessible_project(_PROJECT_ID)


def test_env_eval_upload_requires_credentials_before_paying(monkeypatch, tmp_path, capsys) -> None:
    """Uploading without a key cannot be discovered after the suite has already been bought."""
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: ("url", ""))
    monkeypatch.setattr(
        "flash.client.get_project",
        lambda project_id, api_key: (_ for _ in ()).throw(
            AssertionError("no project lookup without a key")
        ),
        raising=False,
    )
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 1
    assert "not logged in" in capsys.readouterr().err


def test_env_eval_project_with_no_upload_is_rejected(monkeypatch, capsys) -> None:
    """Naming a project while opting out of recording asks for two contradictory things."""
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )

    args = ["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID, "--no-upload"]
    assert cli.main(args) == 1
    assert "--project cannot be combined with --no-upload" in capsys.readouterr().err


def test_env_eval_no_upload_never_records_anything(monkeypatch, tmp_path) -> None:
    """`--no-upload` is the only way to score without leaving a record."""
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def get_run(self, run_id):
            return _run_record(
                {
                    "thinking": False,
                    "project": _PROJECT_ID,
                    "environment": {"id": _PUBLISHED_SLUG},
                }
            )

        def warm_chat_step_selector(self, target):
            return None

        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 0
    # opting out is explicit and total: not even a run the spec could file goes anywhere.
    assert uploader.calls == []


def test_env_eval_records_under_the_evaluated_runs_own_project(monkeypatch, tmp_path) -> None:
    """The default records, and it files under the project that owns the graded weights.

    Not a chosen project and not a fallback: an evaluation of `flash-1` belongs to whatever project
    `flash-1` belongs to, which its own spec already names."""
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def get_run(self, run_id):
            return _run_record(
                {
                    "thinking": False,
                    "project": _PROJECT_ID,
                    "environment": {"id": _PUBLISHED_SLUG},
                }
            )

        def warm_chat_step_selector(self, target):
            return None

        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET]) == 0
    assert [call["project_id"] for call in uploader.calls] == [_PROJECT_ID]


def test_env_eval_project_flag_overrides_the_runs_own_project(monkeypatch, tmp_path) -> None:
    """An explicit `--project` files the results somewhere else on purpose."""
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)
    other = "22222222-2222-2222-2222-222222222222"

    class Client(_EvalClient):
        def get_run(self, run_id):
            return _run_record(
                {
                    "thinking": False,
                    "project": _PROJECT_ID,
                    "environment": {"id": _PUBLISHED_SLUG},
                }
            )

        def warm_chat_step_selector(self, target):
            return None

        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", other]) == 0
    assert [call["project_id"] for call in uploader.calls] == [other]


@pytest.mark.parametrize("as_github_ref", [False, True], ids=["managed-slug", "hub-github-ref"])
def test_env_eval_records_the_hub_environment_the_run_trains_on(
    monkeypatch, tmp_path, as_github_ref
) -> None:
    """The dashboard has to be able to open what it shows in the environment column.

    Recording the resolved entrypoint put a path from one developer's machine in a permanent shared
    record: it names no environment anyone else can open, and two people evaluating the same run
    filed two different provenances for one measurement. The run's own environment is the identity
    the rest of the dashboard is keyed by, and a hub `github:` ref is canonicalized to the slug it
    denotes -- the same normalization the submit route applies when it records the run.
    """
    from flash.envs.loading.loader import managed_slug_to_github_ref

    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)
    expected = "acme/example-project/starter"
    # built by the library rather than hand-written, so the fixture cannot drift from the ref format
    # the loader actually parses.
    spec_environment = managed_slug_to_github_ref(expected) if as_github_ref else expected

    class Client(_EvalClient):
        def get_run(self, run_id):
            return _run_record(
                {
                    "thinking": False,
                    "project": _PROJECT_ID,
                    "environment": {"id": spec_environment},
                }
            )

        def warm_chat_step_selector(self, target):
            return None

        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET]) == 0
    assert [call["environment_reference"] for call in uploader.calls] == [expected]


def test_env_eval_refuses_a_run_that_names_no_published_environment(monkeypatch, capsys) -> None:
    """An evaluation with nothing published to name is refused rather than recorded against a path."""

    class Client(_EvalClient):
        def get_run(self, run_id):
            return {
                "verified_checkpoints": [_EXPLICIT_TARGET],
                "spec": {"thinking": False, "project": _PROJECT_ID},
            }

        def warm_chat_step_selector(self, target):
            return None

        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError("a run without a published environment must not be graded")

    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET]) == 1
    assert uploader.calls == []
    assert "trains on no published environment" in capsys.readouterr().err


def test_env_eval_refuses_a_generic_github_reference(monkeypatch, capsys) -> None:
    """A reference that is not a hub slug names no environment page, so it cannot be recorded.

    `github:owner/repo@main:path/environment.py` is a supported way to train, and
    `_spec_environment_id` returns it verbatim because it denotes no hub page. Nonempty was the only
    guard, so such a run graded and then uploaded a reference the dashboard cannot resolve to
    anything -- the unlinked provenance this command exists to eliminate, reintroduced through the
    one reference shape nobody thought to exclude.
    """

    class Client(_EvalClient):
        def get_run(self, run_id):
            return _run_record(
                {
                    "thinking": False,
                    "project": _PROJECT_ID,
                    "environment": {"id": "github:acme/envs@main:starter/environment.py"},
                }
            )

        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError("an unpublished reference must not be graded")

    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET]) == 1
    assert uploader.calls == []
    assert "is not a published environment" in capsys.readouterr().err


def test_env_eval_grades_the_hub_package_over_a_same_named_local_directory(
    monkeypatch, tmp_path, capsys
) -> None:
    """A working copy named like the slug must not supply the suites for the published environment.

    Download once and grade the extracted package so cwd resolution cannot substitute uncommitted
    local sidecars under a valid published provenance.
    """
    published = tmp_path / "published"
    published.mkdir()
    (published / "environment.py").write_text("def load_environment():\n    return None\n")
    (published / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'published'\n"
        "    def cases(self): return [EvalCase(id='sum', input='2+2', expected='4')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    _patch_published_env(monkeypatch, published)

    # the decoy: a directory whose path is exactly the slug, relative to the cwd.
    local = tmp_path / "cwd" / _PUBLISHED_SLUG
    local.mkdir(parents=True)
    (local / "environment.py").write_text("def load_environment():\n    return None\n")
    (local / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'local-working-copy'\n"
        "    def cases(self): return [EvalCase(id='sum', input='2+2', expected='4')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    monkeypatch.chdir(tmp_path / "cwd")

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path, **_k: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 0

    # the hub's suite ran, not the working copy's.
    assert [call["suite_name"] for call in uploader.calls] == ["published"]
    assert "local-working-copy" not in capsys.readouterr().out
    # and the provenance is still the slug, never the path it was graded from.
    assert uploader.calls[0]["environment_reference"] == _PUBLISHED_SLUG


def test_env_eval_upload_sends_every_case_with_the_project_id(monkeypatch, tmp_path) -> None:
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 0

    assert len(uploader.calls) == 1
    call = uploader.calls[0]
    assert call["project_id"] == _PROJECT_ID
    assert call["suite_name"] == "math"
    assert call["model"] == _EXPLICIT_TARGET
    assert [case["case_id"] for case in call["cases"]] == ["sum"]
    assert call["cases"][0]["success"] is True
    assert call["cases"][0]["actual"] == "4"
    assert call["cases"][0]["expected"] == "4"


def test_env_eval_upload_keeps_duplicate_id_cases_distinct(monkeypatch, tmp_path) -> None:
    # `_case_ids` disambiguates a reused id to `same#2`, but the upload looked its cases back up by
    # the RAW `case.id`. every duplicate after the first missed, uploading a null input and expected
    # while the local report read correctly -- losing exactly the cases the disambiguation exists to
    # keep apart.
    env_dir = tmp_path / "local-env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment():\n    return object()\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'math'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='same', input='2+2', expected='4'),\n"
        "        EvalCase(id='same', input='3+3', expected='6'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    _patch_published_env(monkeypatch, env_dir)

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            # answer each prompt correctly, so a dropped case cannot hide behind a failure
            yield "6" if "3+3" in str(messages) else "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert (
        cli.main(
            [
                "env",
                "eval",
                _EXPLICIT_TARGET,
                "--project",
                _PROJECT_ID,
            ]
        )
        == 0
    )

    cases = uploader.calls[0]["cases"]
    assert [case["case_id"] for case in cases] == ["same", "same#2"]
    # the second case kept its own input and expected rather than uploading nulls
    assert [case["input"] for case in cases] == ["2+2", "3+3"]
    assert [case["expected"] for case in cases] == ["4", "6"]


def test_env_eval_upload_reports_an_errored_case_verbatim(monkeypatch, tmp_path) -> None:
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch, dead_case=True)

    class PartialClient(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            if messages[0]["content"] == "dead":
                raise ConnectionError("connection reset")
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", PartialClient)
    _patch_upload(monkeypatch, uploader)

    # the suite fails overall because one case never generated...
    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 1

    # ...but both cases are still uploaded, and the failure carries its error rather than
    # arriving as a silent zero the server would average in as real model behaviour.
    cases = {case["case_id"]: case for case in uploader.calls[0]["cases"]}
    assert set(cases) == {"sum", "dead"}
    assert cases["dead"]["error"] is not None
    assert "generation failed" in cases["dead"]["error"]
    assert cases["sum"]["error"] is None
    # and the recorded run agrees with the exit code above. only failures BEFORE case execution used
    # to be marked failed, so a suite whose cases died mid-run uploaded as a completed run with no
    # error while the cli printed `overall: FAIL`.
    assert uploader.calls[0]["status"] == "failed"
    assert uploader.calls[0]["error"] == "1/2 case(s) failed to generate or score"


def test_env_eval_upload_records_a_clean_suite_as_completed(monkeypatch, tmp_path) -> None:
    # the mirror of the case above: deriving status from report.errors must not relabel a suite
    # that graded every case, which is the state the dashboard reads as a real measurement.
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 0

    assert uploader.calls[0]["status"] == "completed"
    assert uploader.calls[0]["error"] is None


def test_env_eval_upload_records_suites_that_cannot_load_cases(monkeypatch, tmp_path) -> None:
    # a failed or empty cases() result is the suite verdict, so omitting it from the record leaves
    # the dashboard with a partial run that hides the exact failure responsible for the cli exit.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite\n"
        "class BrokenSuite(BaseEvalSuite):\n"
        "    name = 'broken'\n"
        "    def cases(self): raise RuntimeError('dataset missing')\n"
        "class EmptySuite(BaseEvalSuite):\n"
        "    name = 'empty'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [BrokenSuite(), EmptySuite()]\n"
    )

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError("case loading failures must not generate")

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 1

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
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", "not-a-uuid"]) == 1
    assert "must be a valid UUID" in capsys.readouterr().err


def test_env_eval_upload_keeps_duplicate_case_ids_with_their_own_input(
    monkeypatch, tmp_path
) -> None:
    # results carry the disambiguated id (`same#2`), so a payload keyed on the raw `case.id`
    # both drops the second case and hands the first result the *second* case's input and
    # expected value. that uploads a graded case describing a question it never answered.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'dupes'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='same', input='first', expected='first'),\n"
        "        EvalCase(id='same', input='second', expected='second'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield messages[0]["content"]

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 0

    cases = {case["case_id"]: case for case in uploader.calls[0]["cases"]}
    assert set(cases) == {"same", "same#2"}
    assert cases["same"]["input"] == "first"
    assert cases["same"]["expected"] == "first"
    assert cases["same#2"]["input"] == "second"
    assert cases["same#2"]["expected"] == "second"


def test_env_eval_upload_failure_does_not_relabel_a_passing_suite(
    monkeypatch, tmp_path, capsys
) -> None:
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    from flash.client import ClientError

    def failing_upload(**kwargs):
        raise ClientError("freesolo is unreachable")

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, failing_upload)

    # the suite genuinely passed; a failed upload is reported but must not turn it into a
    # FAIL, which would read as the model having gotten the answer wrong.
    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 0
    captured = capsys.readouterr()
    assert "upload failed" in captured.err
    assert "overall: PASS" in captured.out


def test_env_eval_upload_reports_a_key_lost_after_the_suite_ran(
    monkeypatch, tmp_path, capsys
) -> None:
    """Losing the key mid-run reports the failure without discarding the verdict.

    A missing key at startup is now refused before any generation is bought, so this guard is
    reachable only when the credential disappears while a long evaluation is running -- a logout,
    a rewritten config, a cleared env var. That is a genuine after-the-fact failure like any
    network fault, so it must not relabel a suite that actually passed as failing.
    """
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    keys = iter([("url", "key-1")])

    def _load_credentials():
        # the preflight gets a key; by upload time the user has logged out.
        return next(keys, ("url", None))

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    monkeypatch.setattr("flash.client.upload_eval_run", uploader, raising=False)
    monkeypatch.setattr(
        "flash.client.get_project", lambda project_id, _api_key: {"id": project_id}, raising=False
    )
    monkeypatch.setattr("flash.client.config.load_credentials", _load_credentials)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 0
    assert "not logged in" in capsys.readouterr().err
    # no key means no request was attempted at all.
    assert uploader.calls == []


def test_env_eval_blank_stream_errors_without_scoring(monkeypatch, tmp_path, capsys) -> None:
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'blank'\n"
        "    def cases(self): return [EvalCase(id='one', input='hello', expected='hi')]\n"
        "    def score(self, case, response): raise AssertionError('score must not run')\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class BlankClient(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "   "

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", BlankClient)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1

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
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "import sys\n"
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
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

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield messages[0]["content"]

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1

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
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    witness = tmp_path / "max_inside.txt"
    # the sidecar records the peak overlap to a file: the suite object that actually ran
    # lives inside the command, and reloading the module would build a fresh one.
    (env_dir / "evaluations.py").write_text(
        "import time\n"
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
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

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            # force all six generations to be genuinely in flight together, so any
            # unserialized scoring really would overlap.
            generating.wait()
            yield messages[0]["content"]

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--concurrency", "6", "--no-upload"]) == 0

    # never more than one scorer in flight, despite six concurrent generations.
    assert witness.read_text() == "1"
    assert "overall: PASS" in capsys.readouterr().out


def test_env_eval_empty_suite_is_not_a_pass(monkeypatch, tmp_path, capsys) -> None:
    # 0/0 graded is not a green suite. reporting PASS here hides a sidecar whose cases()
    # silently returned nothing.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'empty'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError("no cases means no generation")

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1

    captured = capsys.readouterr()
    assert "suite empty has no cases to run" in captured.err
    assert "overall: FAIL" in captured.err
    assert "overall: PASS" not in captured.out


def test_env_eval_disambiguates_duplicate_case_ids(monkeypatch, tmp_path, capsys) -> None:
    # two cases sharing an id would collide in the report and in the uploaded payload,
    # showing one graded case where two ran.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'dupes'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='same', input='a', expected='a'),\n"
        "        EvalCase(id='same', input='b', expected='b'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield messages[0]["content"]

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 0

    output = capsys.readouterr().out
    assert "case same: PASS" in output
    assert "case same#2: PASS" in output
    assert "suite dupes: 2/2 passed" in output


def test_env_eval_debug_surfaces_the_load_traceback(monkeypatch, tmp_path) -> None:
    # a broken sidecar is a bug, and --debug asked for its traceback. swallowing it into a
    # one-line message leaves nothing to debug with.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text("raise RuntimeError('sidecar exploded')\n")
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", _EvalClient)

    with pytest.raises(RuntimeError, match="sidecar exploded"):
        cli.main(["--debug", "env", "eval", _EXPLICIT_TARGET, "--no-upload"])


def test_evaluation_sidecar_can_import_helpers_lazily(tmp_path) -> None:
    # the package dir must stay importable after load: sidecars import local helpers inside
    # cases()/score(), which run long after the module is executed.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "lazy_helper.py").write_text("GOLD = 'deferred'\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
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
        "from flash.envs.meta.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'repeat'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    load_evaluation_suites(env_dir)
    after_first = list(sys.path)
    load_evaluation_suites(env_dir)

    assert sys.path == after_first


def test_evaluation_sidecars_do_not_share_a_sibling_module_name(tmp_path) -> None:
    # two packages, each with its own `helper`. the first load left `helper` in sys.modules, so the
    # second sidecar's `from helper import GOLD` found the FIRST package's module already cached and
    # reused it -- silently running the wrong cases, with no import error to notice.
    for package, gold in (("alpha", "ALPHA"), ("beta", "BETA")):
        env_dir = tmp_path / package
        env_dir.mkdir()
        (env_dir / "environment.py").write_text("def load_environment():\n    return None\n")
        (env_dir / "helper.py").write_text(f"GOLD = {gold!r}\n")
        (env_dir / "evaluations.py").write_text(
            "from helper import GOLD\n"
            "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
            "class Suite(BaseEvalSuite):\n"
            f"    name = {package!r}\n"
            "    def cases(self): return [EvalCase(id='c', input=GOLD, expected=GOLD)]\n"
            "def load_evaluations(environment=None): return [Suite()]\n"
        )

    alpha = load_evaluation_suites(tmp_path / "alpha")[0]
    beta = load_evaluation_suites(tmp_path / "beta")[0]

    assert alpha.cases()[0].input == "ALPHA"
    # the second package graded its own cases rather than the first package's.
    assert beta.cases()[0].input == "BETA"


def test_a_sidecar_sibling_wins_over_an_unrelated_module_already_cached(tmp_path) -> None:
    """sys.modules is consulted before sys.path, so a cached `helper` must be displaced.

    A process holding an unrelated top-level `helper` handed it to a sidecar importing its own
    sibling of that name: the suite graded with another module's constants, and no import error
    ever surfaced. `_forget_sidecar_siblings` could not correct it -- the name was present before
    the scope, so the loop never examined it.
    """
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "helper.py").write_text("GOLD = 'UNRELATED'\n")
    sys.path.insert(0, str(unrelated))
    try:
        import helper as preexisting

        assert preexisting.GOLD == "UNRELATED"
    finally:
        sys.path.remove(str(unrelated))

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment():\n    return None\n")
    (env_dir / "helper.py").write_text("GOLD = 'OWN-SIBLING'\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'shadowed'\n"
        "    def cases(self):\n"
        "        from helper import GOLD\n"
        "        return [EvalCase(id='c', input=GOLD, expected=GOLD)]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    try:
        suite = load_evaluation_suites(env_dir)[0]
        # the sidecar's own sibling, not the module the process already had cached
        assert suite.cases()[0].input == "OWN-SIBLING"
        # and the displaced module is handed back to the rest of the process untouched
        assert sys.modules["helper"] is preexisting
        assert sys.modules["helper"].GOLD == "UNRELATED"
    finally:
        sys.modules.pop("helper", None)


@pytest.mark.parametrize("namespace", [False, True], ids=["regular-package", "namespace-package"])
def test_a_sidecar_package_submodule_wins_over_one_already_cached(
    tmp_path, namespace: bool
) -> None:
    """Displacing a package must displace its cached submodules too.

    ``from graders.rules`` reads the submodule cache directly. Claim PEP 420 namespace directories
    too, even without ``__init__.py``.
    """
    unrelated = tmp_path / "unrelated"
    (unrelated / "graders").mkdir(parents=True)
    if not namespace:
        (unrelated / "graders" / "__init__.py").write_text("")
    (unrelated / "graders" / "rules.py").write_text("GOLD = 'UNRELATED'\n")
    sys.path.insert(0, str(unrelated))
    try:
        import graders.rules as preexisting

        assert preexisting.GOLD == "UNRELATED"
    finally:
        sys.path.remove(str(unrelated))

    env_dir = tmp_path / "env"
    (env_dir / "graders").mkdir(parents=True)
    (env_dir / "environment.py").write_text("def load_environment():\n    return None\n")
    if not namespace:
        (env_dir / "graders" / "__init__.py").write_text("")
    (env_dir / "graders" / "rules.py").write_text("GOLD = 'OWN-SIBLING'\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'submodule'\n"
        "    def cases(self):\n"
        "        from graders.rules import GOLD\n"
        "        return [EvalCase(id='c', input=GOLD, expected=GOLD)]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    try:
        suite = load_evaluation_suites(env_dir)[0]
        # this environment's own rules, not the ones another package left cached
        assert suite.cases()[0].input == "OWN-SIBLING"
        # and both the package and its submodule are handed back to the process untouched
        assert sys.modules["graders.rules"] is preexisting
        assert sys.modules["graders.rules"].GOLD == "UNRELATED"
    finally:
        for name in [n for n in list(sys.modules) if n == "graders" or n.startswith("graders.")]:
            del sys.modules[name]


def test_a_sidecar_namespace_package_wins_over_one_already_cached(tmp_path) -> None:
    """A sibling directory without __init__.py owns its plain name just as a package does.

    PEP 420 makes a bare `graders/` importable, but the owned-name scan required the marker file,
    so a namespace sibling was never counted -- another environment's cached `graders.rules`
    survived the sweep and the suite graded with that environment's scoring code.
    """
    unrelated = tmp_path / "unrelated"
    (unrelated / "graders").mkdir(parents=True)
    (unrelated / "graders" / "__init__.py").write_text("")
    (unrelated / "graders" / "rules.py").write_text("GOLD = 'UNRELATED'\n")
    sys.path.insert(0, str(unrelated))
    try:
        import graders.rules as preexisting

        assert preexisting.GOLD == "UNRELATED"
    finally:
        sys.path.remove(str(unrelated))

    env_dir = tmp_path / "env"
    # no __init__.py: this environment's `graders` is a namespace package
    (env_dir / "graders").mkdir(parents=True)
    (env_dir / "environment.py").write_text("def load_environment():\n    return None\n")
    (env_dir / "graders" / "rules.py").write_text("GOLD = 'OWN-SIBLING'\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'namespace'\n"
        "    def cases(self):\n"
        "        from graders.rules import GOLD\n"
        "        return [EvalCase(id='c', input=GOLD, expected=GOLD)]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    try:
        suite = load_evaluation_suites(env_dir)[0]
        # this environment's own rules, not the ones another package left cached
        assert suite.cases()[0].input == "OWN-SIBLING"
        # and the process keeps the module it had before the sidecar ran
        assert sys.modules["graders.rules"] is preexisting
        assert sys.modules["graders.rules"].GOLD == "UNRELATED"
    finally:
        for name in [n for n in list(sys.modules) if n == "graders" or n.startswith("graders.")]:
            del sys.modules[name]


def test_a_lazily_imported_helper_keeps_its_state_across_cases(tmp_path) -> None:
    """A helper's module-level state survives from one callback to the next.

    Discarding this sidecar's modules on scope exit re-executed the helper for every callback, so
    a counter restarted at 1 for every case and an expensive judge model or connection was rebuilt
    each time. Parking keeps the name free for other sidecars without resetting the module.
    """
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "environment.py").write_text("def load_environment():\n    return None\n")
    (env_dir / "counter.py").write_text("LOADS = []\nLOADS.append(1)\nCALLS = 0\n")
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase, EvalResult\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'stateful'\n"
        "    def cases(self):\n"
        "        return [EvalCase(id='a', input='x', expected='x'),\n"
        "                EvalCase(id='b', input='y', expected='y')]\n"
        "    def score(self, case, response):\n"
        "        import counter\n"
        "        counter.CALLS += 1\n"
        "        return EvalResult(case_id=case.id, passed=True, score=float(counter.CALLS),\n"
        "                          response=response, reason=str(len(counter.LOADS)))\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    suite = load_evaluation_suites(env_dir)[0]
    cases = suite.cases()
    first = suite.score(cases[0], "x")
    second = suite.score(cases[1], "y")

    # the counter advances rather than restarting, and the module executed exactly once
    assert (first.score, second.score) == (1.0, 2.0)
    assert (first.reason, second.reason) == ("1", "1")
    # parking, not leaking: the name is still free for the next environment's own `counter`
    assert "counter" not in sys.modules


def test_a_sidecar_load_leaves_unrelated_cached_modules_alone(tmp_path) -> None:
    # the control for the scope of the isolation. only modules resolving to the package directory
    # are dropped: evicting a stdlib or third-party module the sidecar happened to import first
    # would re-execute unrelated code on every later load.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "import base64\n"
        "from flash.envs.meta.evaluations import BaseEvalSuite\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'imports-stdlib'\n"
        "    def cases(self): return []\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    sys.modules.pop("base64", None)
    load_evaluation_suites(env_dir)

    assert "base64" in sys.modules


def test_env_eval_forwards_split_and_params_to_the_environment(monkeypatch, tmp_path) -> None:
    # an env whose load_environment() REQUIRES a setting could not be evaluated at all, and one that
    # merely defaults built a differently-configured scorer than the run trains on, with no
    # --param/--split path to correct it.
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)
    loaded: list[dict] = []

    def _load(path, *args, **kwargs):
        # mirrors an env that requires its difficulty: parameterless is a hard failure.
        if "difficulty" not in kwargs:
            raise TypeError("load_environment() missing required argument: 'difficulty'")
        loaded.append(kwargs)
        return object()

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    monkeypatch.setattr("flash.envs.loading.loader.load_freesolo_environment", _load)
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert (
        cli.main(
            [
                "env",
                "eval",
                _EXPLICIT_TARGET,
                "--param",
                "difficulty=3",
                "--split",
                "holdout",
                "--no-upload",
            ]
        )
        == 0
    )
    # TOML scalar typing survives, exactly as [environment.params] would carry it.
    assert loaded == [{"difficulty": 3, "split": "holdout"}]


def test_env_eval_is_an_org_binding_command(monkeypatch, tmp_path, capsys) -> None:
    # results are recorded under a project resolved from the ambient key, so a shadowing
    # FREESOLO_API_KEY belonging to another org has to be reported BEFORE the paid requests run, not
    # discovered at upload time with the whole evaluation already spent.
    from flash.cli.commands.env.testing.eval import cmd_env_eval
    from flash.cli.parsing.main import _ORG_BINDING_COMMANDS

    assert cmd_env_eval in _ORG_BINDING_COMMANDS

    monkeypatch.setattr(
        "flash.cli.parsing.main.shadowed_login_warning", lambda: "key belongs to other-org"
    )
    monkeypatch.setattr(
        "flash.client.client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    # the warning lands even on the run that fails its preflight: it precedes every request.
    cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", "proj-9"])
    assert "key belongs to other-org" in capsys.readouterr().err


def test_env_eval_abort_does_not_join_in_flight_generations(monkeypatch, tmp_path) -> None:
    # Ctrl-C during a concurrent eval must reach the CLI's handler immediately. Under `with
    # ThreadPoolExecutor(...)` the implicit shutdown(wait=True) joined every in-flight request
    # first, and a chat_stream call may block for up to 30 minutes -- so an aborted eval looked hung
    # for that long.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'aborted'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='interrupts', input='interrupts'),\n"
        "        EvalCase(id='blocks', input='blocks'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    release_slow_case = threading.Event()
    slow_case_started = threading.Event()

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            if messages[0]["content"] == "blocks":
                # stands in for a generation still streaming when the user aborts.
                slow_case_started.set()
                assert release_slow_case.wait(timeout=60), "slow case was never released"
                yield "late"
                return
            assert slow_case_started.wait(timeout=10)
            raise KeyboardInterrupt

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    returned = threading.Event()

    def _abort() -> None:
        # 130 is the CLI's own KeyboardInterrupt exit code.
        assert (
            cli.main(["env", "eval", _EXPLICIT_TARGET, "--concurrency", "2", "--no-upload"]) == 130
        )
        returned.set()

    aborting = threading.Thread(target=_abort, daemon=True)
    aborting.start()
    try:
        # the whole point: this returns while the slow generation is still blocked. joining it
        # would hang here until the 60s release below, which is what the bug did for 30 minutes.
        assert returned.wait(timeout=10), "abort waited on the in-flight generation"
    finally:
        release_slow_case.set()
        aborting.join(timeout=10)


def test_env_eval_abort_does_not_hold_the_process_open(tmp_path) -> None:
    """The aborted eval must let the INTERPRETER exit, not just return from main().

    `shutdown(wait=False)` returns immediately but concurrent.futures registers an
    interpreter-exit hook that joins every worker anyway, so the process stayed alive until the
    in-flight chat_stream hit its 30-minute timeout even though the CLI had already reported
    `aborted`. The test above measures main()'s return, which the executor already
    satisfied; only process exit can distinguish the two, so this one runs a real subprocess.
    """
    driver = tmp_path / "abort_driver.py"
    driver.write_text(
        "import argparse, threading, sys\n"
        "from flash.cli.commands.env.testing.eval import _generate_concurrently\n"
        "release = threading.Event()\n"
        "def chat(*a, **k):\n"
        # stands in for a generation still streaming when the user aborts. a real one blocks on
        # the client's 30-minute read timeout; 300s is far past any plausible clean exit.
        "    release.wait(timeout=300)\n"
        "    return 'late'\n"
        "import flash.cli.commands.env.testing.eval as env_eval\n"
        "env_eval._generate_case = chat\n"
        # the fan-out settles the step-selector capability on this thread first, so the client
        # has to answer that call; 't' carries no step, so the real one does nothing here.
        "class Client:\n"
        "    def warm_chat_step_selector(self, target): return None\n"
        "args = argparse.Namespace(concurrency=2)\n"
        "worker = threading.Thread(\n"
        "    target=_generate_concurrently, args=(Client(), 't', [[{'role': 'user'}]], args),\n"
        "    daemon=True,\n"
        ")\n"
        "worker.start()\n"
        "import time; time.sleep(0.5)\n"  # let the generation actually start
        "print('aborted', flush=True)\n"
    )
    completed = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True,
        text=True,
        # generous vs a clean exit, far under the 300s the blocked generation would impose
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "aborted" in completed.stdout


def test_case_ids_stay_unique_when_an_id_looks_like_a_disambiguated_one() -> None:
    # the disambiguated form is itself a legal explicit id, so the suffix has to be retried
    # until it is free. cases `a`, `a`, `a#2` resolved to `a`, `a#2`, `a#2` and reintroduced
    # the very collision `_case_ids` exists to remove: the upload pairs positionally, but the
    # printed report and the recorded case ids still showed one label for two graded cases.
    from flash.cli.commands.env.testing.eval import _case_ids

    def ids(*case_ids):
        return _case_ids([EvalCase(id=case_id, input="x") for case_id in case_ids])

    assert ids("a", "a", "a#2") == ["a", "a#2", "a#2#2"]
    # an explicit id colliding with a positional one, in both orders
    assert ids(None, "1") == ["1", "1#2"]
    assert ids("a#2", "a", "a") == ["a#2", "a", "a#3"]
    # the ordinary duplicate case keeps its plain sequential suffixes
    assert ids("same", "same", "same") == ["same", "same#2", "same#3"]


def test_env_eval_scores_on_the_calling_thread(monkeypatch, tmp_path, capsys) -> None:
    # a lock prevents overlap but not thread affinity. a scorer holding a resource created while the
    # suite was loaded -- a sqlite connection, a tokenizer bound to its creating thread -- failed
    # every case from a worker even though nothing was concurrent. only generation is parallel;
    # score() always runs on the thread that ran the command.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    witness = tmp_path / "scoring_threads.txt"
    (env_dir / "evaluations.py").write_text(
        "import threading\n"
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        f"WITNESS = {str(witness)!r}\n"
        "OWNER = threading.current_thread().name\n"
        "class Suite(BaseEvalSuite):\n"
        "    def cases(self): return [\n"
        "        EvalCase(id=f'c{i}', input=f'c{i}', expected=f'c{i}') for i in range(6)\n"
        "    ]\n"
        "    name = 'affine'\n"
        "    def score(self, case, response):\n"
        "        with open(WITNESS, 'a') as fh:\n"
        "            fh.write(threading.current_thread().name + '\\n')\n"
        "        return super().score(case, response)\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    generating = threading.Barrier(6, timeout=5)
    generation_threads: set[str] = set()

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            # all six really are in flight together, so generation is genuinely concurrent and
            # the assertion below is about where scoring runs, not about a serial fallback.
            generating.wait()
            generation_threads.add(threading.current_thread().name)
            yield messages[0]["content"]

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--concurrency", "6", "--no-upload"]) == 0

    scored_on = witness.read_text().split()
    assert len(scored_on) == 6
    assert set(scored_on) == {threading.current_thread().name}
    # generation did run off this thread, so the suite above is not passing by accident
    assert generation_threads
    assert threading.current_thread().name not in generation_threads
    assert "overall: PASS" in capsys.readouterr().out


def test_scored_response_preserves_existing_structure_and_still_parses_plain_text() -> None:
    """Only the first parser has the raw generation needed to retain its structured views.

    Episode turn recording already stores an answer-compatible string with raw and thinking attached.
    Parsing that value again is answer-idempotent but metadata-destructive, while single-turn eval still
    arrives as plain text and must continue to build the same structure here.
    """
    from flash.cli.commands.env.testing.eval import _scored_response
    from flash.envs.loading.adapter import _ScoredResponseText

    raw = "<think>2+2 is 4</think>4"
    stored = _ScoredResponseText("4", raw=raw, thinking="2+2 is 4")

    preserved = _scored_response(stored, thinking=True)
    parsed = _scored_response(raw, thinking=True)

    assert preserved is stored
    assert preserved.raw == raw
    assert preserved.thinking == "2+2 is 4"
    assert parsed == "4"
    assert parsed.raw == raw
    assert parsed.thinking == "2+2 is 4"


def test_env_eval_reports_the_whole_completion_it_graded() -> None:
    # stripping for the scorer must not shorten what is recorded: the reasoning is what makes a
    # failed case diagnosable, so the result keeps the full emission and the upload carries it.
    # the strip itself is covered by `test_env_eval_strips_reasoning_only_for_a_thinking_run`;
    # what is pinned here is that the two views stay distinct rather than the stripped answer
    # overwriting the record.
    from flash.cli.commands.env.testing.eval import _score_case

    seen: list[str] = []

    class Suite:
        def score(self, case, response):
            seen.append(response)
            # a thinking-aware scorer can still reach the reasoning through the structured views
            assert response.thinking == "2+2 is 4"
            assert response.raw == "<think>2+2 is 4</think>4"
            return response == "4"

    case = EvalCase(id="c1", input="2+2", expected="4")
    result = _score_case(Suite(), case, "c1", "<think>2+2 is 4</think>4", thinking=True)

    assert seen == ["4"]
    assert result.passed
    assert result.response == "<think>2+2 is 4</think>4"


@pytest.mark.parametrize("scorer_raises", [False, True])
def test_env_eval_records_raw_episode_generation_on_success_and_failure(scorer_raises) -> None:
    """episode state carries answer-only text, but its raw generation is the diagnostic record."""
    from flash.cli.commands.env.testing.eval import _case_payload, _score_case
    from flash.envs.loading.adapter import _ScoredResponseText

    raw = "<think>2+2 is 4</think>4"
    stored = _ScoredResponseText("4", raw=raw, thinking="2+2 is 4")

    class Suite:
        def score(self, case, response, state=None):
            assert response is stored
            if scorer_raises:
                raise RuntimeError("scorer bailed")
            return 1.0

    case = EvalCase(id="c1", input="2+2", expected="4")
    result = _score_case(Suite(), case, "c1", stored, state={"response_text": stored})

    assert result.response == raw
    assert _case_payload(case, result)["actual"] == raw
    assert result.error == ("scoring failed: scorer bailed" if scorer_raises else None)


def test_env_eval_uploads_a_suite_that_failed_to_load(monkeypatch, tmp_path, capsys) -> None:
    # skipping the upload for a suite that never graded a case left the dashboard showing the
    # earlier suites as a completed run with the failing suite simply absent -- a green-looking
    # evaluation whose CLI exit code was 1.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Good(BaseEvalSuite):\n"
        "    name = 'good'\n"
        "    def cases(self): return [EvalCase(id='sum', input='2+2', expected='4')]\n"
        "class Broken(BaseEvalSuite):\n"
        "    name = 'broken'\n"
        "    def cases(self): raise RuntimeError('cases blew up')\n"
        "def load_evaluations(environment=None): return [Good(), Broken()]\n"
    )

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert (
        cli.main(
            [
                "env",
                "eval",
                _EXPLICIT_TARGET,
                "--project",
                _PROJECT_ID,
            ]
        )
        == 1
    )

    # both suites reach the dashboard, and the broken one carries why it graded nothing
    assert [call["suite_name"] for call in uploader.calls] == ["good", "broken"]
    broken = uploader.calls[1]["cases"]
    assert [case["case_id"] for case in broken] == ["load"]
    assert "cases blew up" in broken[0]["error"]
    # the case has no input behind it, which is exactly why the pairing must not be strict
    assert broken[0]["input"] is None
    assert "suite broken failed to load cases" in capsys.readouterr().err


def test_env_eval_reports_invalid_episode_flag_as_a_suite_failure(
    monkeypatch, tmp_path, capsys
) -> None:
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import EvalCase\n"
        "class Suite:\n"
        "    name = 'invalid-episode-flag'\n"
        "    grades_episodes = 'false'\n"
        "    def cases(self): return [EvalCase(id='c', input='go')]\n"
        "    def score(self, case, response): return True\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", _EvalClient)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1
    captured = capsys.readouterr()
    assert "suite invalid-episode-flag failed to load cases" in captured.err
    assert "grades_episodes must be a bool, got str" in captured.err
    assert "Traceback" not in captured.err
    assert "overall: FAIL" in captured.err


def test_env_eval_concurrent_results_stay_in_case_order(monkeypatch, tmp_path) -> None:
    # generation completes out of order under --concurrency. pairing results with case ids by
    # completion order rather than submission index would hand each result another case's id,
    # mislabelling every graded case in the report and the upload.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase, EvalResult\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'ordered'\n"
        "    def cases(self):\n"
        "        return [EvalCase(id=str(i), input=str(i)) for i in range(6)]\n"
        "    def score(self, case, response):\n"
        "        return EvalResult(case_id=case.id, passed=case.input == response,\n"
        "                          score=1.0, response=response)\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class SlowFirstClient(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            content = messages[0]["content"]
            # case 0 finishes last, so completion order is the reverse of submission order.
            time.sleep(0.05 if content == "0" else 0.0)
            yield content

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", SlowFirstClient)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--concurrency", "6", "--no-upload"]) == 0


@pytest.mark.parametrize("bad", ["nan", "inf", "Infinity", "1e999"])
def test_env_eval_rejects_a_non_finite_temperature_before_paying(monkeypatch, tmp_path, bad):
    # `float("nan")` and `float("1e999")` both parse, so argparse took them and every case then
    # spent a request the serving route rejects for being non-finite -- one bad flag turned into one
    # doomed paid request per case, the whole evaluation billed and nothing graded. the environment
    # loads fine here: reaching generation at all is the defect.
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError(f"paid request issued with temperature={kwargs['temperature']}")
            yield ""  # pragma: no cover - generator protocol only

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["env", "eval", _EXPLICIT_TARGET, "--temperature", bad])

    # argparse's own usage error, which is exit 2 rather than the CLI's 1
    assert excinfo.value.code == 2


@pytest.mark.parametrize("bad", ["-1", "-0.5", "-1e-9"])
def test_env_eval_rejects_a_negative_temperature_before_paying(monkeypatch, tmp_path, bad):
    # a negative value is finite, so it passed the non-finite guard above and every case then spent
    # a request the OpenAI sampling contract rejects -- one bad flag recorded as one generation
    # failure per case rather than one usage error. training already enforces this floor on its own
    # temperature (`flash/schema/__init__.py`, `minimum=0.0`).
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            raise AssertionError(f"paid request issued with temperature={kwargs['temperature']}")
            yield ""  # pragma: no cover - generator protocol only

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["env", "eval", _EXPLICIT_TARGET, "--temperature", bad])

    # argparse's own usage error, which is exit 2 rather than the CLI's 1
    assert excinfo.value.code == 2


def test_env_eval_still_accepts_an_ordinary_temperature(monkeypatch, tmp_path) -> None:
    # the control: rejecting non-finite values must not narrow the flag's real range.
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)
    seen: list[float] = []

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            seen.append(kwargs["temperature"])
            yield "4"

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--temperature", "0.7", "--no-upload"]) == 0
    assert seen == [0.7]


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


def test_env_eval_uploads_each_suite_with_its_own_start_time(monkeypatch, tmp_path) -> None:
    # every suite uploads as its own run. sharing one timestamp backdates each later run to
    # before the earlier suites ran and inflates its dashboard duration with their work.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class First(BaseEvalSuite):\n"
        "    name = 'first'\n"
        "    def cases(self): return [EvalCase(id='a', input='a', expected='a')]\n"
        "class Second(BaseEvalSuite):\n"
        "    name = 'second'\n"
        "    def cases(self): return [EvalCase(id='b', input='b', expected='b')]\n"
        "def load_evaluations(environment=None): return [First(), Second()]\n"
    )

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            time.sleep(0.01)
            yield messages[0]["content"]

    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 0

    starts = [call["started_at"] for call in uploader.calls]
    assert len(starts) == 2
    # the second suite cannot have started before the first one finished its work.
    assert starts[1] > starts[0]


def test_env_eval_reports_upload_timeout_without_a_traceback(monkeypatch, tmp_path, capsys) -> None:
    # a socket timeout surfaces as a bare TimeoutError, not a URLError. untranslated it escapes
    # the upload handler after the whole paid evaluation ran, replacing the promised nonfatal
    # message and final verdict with a traceback. the real upload path runs here, so the socket
    # is what fails -- patching upload_eval_run itself would skip the translation being tested.
    env_dir = _upload_env_dir(tmp_path, monkeypatch=monkeypatch)

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "4"

    def _timeout(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: ("url", "key-1"))
    # only the upload may time out. the project preflight uses the same socket, so leaving it
    # unpatched would fail the run before it reached the translation under test.
    monkeypatch.setattr(
        "flash.client.get_project", lambda project_id, _api_key: {"id": project_id}, raising=False
    )
    monkeypatch.setattr("urllib.request.urlopen", _timeout)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 0

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


def test_timeout_does_not_invent_a_route_the_request_never_used(monkeypatch) -> None:
    """A reverse-proxy prefix must not be silently dropped from the reported endpoint.

    ``displayable_url`` reduces a base to scheme and host so credentials in the authority cannot
    leak. Concatenating the path back onto that result rebuilds a URL the client never requested:
    with ``FREESOLO_BASE_URL=https://host/proxy`` the real request goes to ``/proxy/v1/...`` while
    the message would read ``https://host/v1/...``, sending an operator to inspect the wrong route.
    """
    from flash.client.http import RequestTimeoutError, _freesolo_request

    def _timeout(req, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", _timeout)

    with pytest.raises(RequestTimeoutError) as excinfo:
        _freesolo_request(
            "POST", "/v1/eval-runs", "key-1", base_url="https://user:pw@example.test/proxy"
        )
    message = str(excinfo.value)
    # the credential is still gone -- this must not be fixed by printing the raw base back.
    assert "pw" not in message
    assert "user:" not in message
    # and the message must not assert a concatenated route that was never requested.
    assert "https://example.test/v1/eval-runs" not in message
    assert "/v1/eval-runs" in message  # the path is still reported, just not glued to the host


def test_load_evaluations_receives_the_environment_after_other_positional_parameters(
    tmp_path,
) -> None:
    # a factory whose `environment` follows another positional-only parameter used to stop the
    # match at the earlier name and pass `environment` by keyword, which positional-only
    # syntax forbids -- so the sidecar failed to load rather than being graded by its scorer.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
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


def test_load_evaluations_receives_a_positional_only_environment_beside_var_kwargs(
    tmp_path,
) -> None:
    # `**kwargs` accepts any NAME, but it cannot make a positional-only parameter keyword-passable.
    # taking the kwargs shortcut put `environment` into the **options bag and left the required
    # positional unfilled, so this valid factory raised TypeError and the sidecar never loaded.
    env_dir = _environment_dir(tmp_path)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'positional-kwargs'\n"
        "    def __init__(self, environment, options): \n"
        "        self.environment = environment\n"
        "        self.options = options\n"
        "    def cases(self): return [EvalCase(id='a', input='a', expected='a')]\n"
        "def load_evaluations(environment, /, **options):\n"
        "    return [Suite(environment, options)]\n"
    )
    sentinel = object()

    suites = load_evaluation_suites(env_dir / "environment.py", environment=sentinel)

    assert suites[0].environment is sentinel
    # and it arrived positionally, not duplicated into the kwargs bag.
    assert suites[0].options == {}


def test_env_eval_sends_the_environments_own_prompt(monkeypatch, tmp_path) -> None:
    """Evaluation must grade the prompt training builds, not the bare case input.

    Every training path renders its prompt through `env.prompt_messages(example)`, which runs the
    environment's `start_episode` and injects the training contract as the system message. Sending
    only `case.input` grades the model on a prompt no run ever trains on, so the suite can fail for
    a missing system contract rather than for anything the model did.
    """
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'prompted'\n"
        "    def cases(self): return [EvalCase(id='a', input='2+2', expected='ok')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Environment:
        def prompt_messages(self, example):
            return [
                {"role": "system", "content": "answer as a calculator"},
                {"role": "user", "content": example["input"]},
            ]

    sent: list[list[dict]] = []

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            sent.append(messages)
            yield "ok"

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment",
        lambda _path, **_kwargs: Environment(),
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 0

    assert sent == [
        [
            {"role": "system", "content": "answer as a calculator"},
            {"role": "user", "content": "2+2"},
        ]
    ]


def test_env_eval_prompt_failure_fails_only_its_own_case(monkeypatch, tmp_path, capsys) -> None:
    # prompt_messages is user code. a raise on one example must be that case's error, not a
    # crash that loses every case already graded.
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'partial'\n"
        "    def cases(self): return [\n"
        "        EvalCase(id='good', input='good', expected='good'),\n"
        "        EvalCase(id='bad', input='bad', expected='bad'),\n"
        "    ]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Environment:
        def prompt_messages(self, example):
            if example["input"] == "bad":
                raise RuntimeError("no template for this row")
            return [{"role": "user", "content": example["input"]}]

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield messages[-1]["content"]

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment",
        lambda _path, **_kwargs: Environment(),
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1

    output = capsys.readouterr().out
    assert "case good: PASS" in output
    assert "case bad: FAIL" in output
    assert "prompt construction failed: no template for this row" in output
    # the broken case is an error, not a zero the model earned.
    assert "errors=1 (excluded from pass_rate and mean_score)" in output


def test_env_eval_sends_the_prompt_images_training_builds(monkeypatch, tmp_path) -> None:
    """A multimodal case must reach the backend as the prompt training built, images included.

    Apply ``normalize_prompt_images`` after ``prompt_messages`` so top-level images and local paths
    become the same remote-safe blocks used by training.
    """
    image_module = pytest.importorskip("PIL.Image")

    env_package = tmp_path / "pkg"
    dataset = env_package / "dataset"
    dataset.mkdir(parents=True)
    image_module.new("RGB", (2, 2), (255, 0, 0)).save(dataset / "cat.png", format="PNG")

    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'vision'\n"
        "    def cases(self):\n"
        # one case carries the image as top-level metadata, the other as an env-emitted block
        "        return [EvalCase(id='top', input='what is this?', expected='cat',\n"
        "                         metadata={'image': 'dataset/cat.png'}),\n"
        "                EvalCase(id='block', input='what is this?', expected='cat')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Environment:
        package_root = str(env_package)

        def prompt_messages(self, example):
            if example.get("id") != "block":
                return [{"role": "user", "content": example["input"]}]
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example["input"]},
                        {"type": "image_url", "image_url": {"url": "dataset/cat.png"}},
                    ],
                }
            ]

    sent: list[list[dict]] = []

    class Client(_EvalClient):
        def get_run(self, run_id):
            return {
                "verified_checkpoints": [_EXPLICIT_TARGET],
                "spec": {"environment": {"id": _PUBLISHED_SLUG}},
            }

        def chat_stream(self, target, messages, **kwargs):
            sent.append(messages)
            yield "cat"

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path, **_k: Environment()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 0

    assert len(sent) == 2
    for messages in sent:
        blocks = messages[0]["content"]
        # the top-level image used to vanish here, leaving a bare string and a text-only grade
        assert isinstance(blocks, list), f"expected content blocks, got {blocks!r}"
        images = [b for b in blocks if b.get("type") == "image_url"]
        assert len(images) == 1
        url = images[0]["image_url"]["url"]
        # a data URI, not "dataset/cat.png": the backend has no access to this filesystem
        assert url.startswith("data:image/png;base64,"), url


def _stub_pil(monkeypatch, mime: str = "image/png") -> None:
    """Stand in for Pillow at the one boundary multimodal.py uses it.

    Pillow is not installed by the test extras (`uv sync --extra server --dev`), so the real-image
    test above skips in CI and pins nothing there. Only image *inspection* needs PIL -- the
    normalization, descriptor, and data-URI code under test is ours -- so faking that boundary keeps
    the assertion about our own path rather than about Pillow being present.
    """
    from types import ModuleType

    class _Image:
        def __init__(self, data: bytes) -> None:
            self.width, self.height, self.format = 2, 2, "PNG"
            self.size = (self.width, self.height)
            self.mode = "RGB"
            self.n_frames = 1
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def getexif(self):
            return {}

        def verify(self):
            return None

        def load(self):
            return None

        def convert(self, _mode):
            return self

        def close(self):
            return None

    module = ModuleType("PIL")
    image_module = ModuleType("PIL.Image")
    image_ops_module = ModuleType("PIL.ImageOps")
    image_module.Image = _Image
    image_module.DecompressionBombWarning = UserWarning
    image_module.MIME = {"PNG": mime}
    image_module.open = lambda stream: _Image(stream.getvalue())
    image_module.new = lambda *_a, **_k: _Image(b"")
    image_ops_module.exif_transpose = lambda image, **_kwargs: image
    module.Image = image_module
    module.ImageOps = image_ops_module
    monkeypatch.setitem(sys.modules, "PIL", module)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_module)
    monkeypatch.setitem(sys.modules, "PIL.ImageOps", image_ops_module)
    monkeypatch.setattr("flash.content.multimodal._is_pil_image", lambda _value: False)


def test_env_eval_sends_a_top_level_image_the_prompt_never_mentions(monkeypatch, tmp_path) -> None:
    """The same defect as above, pinned where CI can actually see it.

    The real-image test skips without Pillow, which the test extras do not install -- so on CI
    nothing catches a regression here. Stubbing only the PIL boundary runs the normalization and
    data-URI code for real: an `image` field with no placeholder block is appended to the first user
    message, which is how training sees it, and sending prompt_messages() unchanged would lose it
    with no error at all.
    """
    from flash.cli.commands.env.testing.eval import _case_messages

    _stub_pil(monkeypatch)

    class Environment:
        package_root = None

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["input"]}]

    case = EvalCase(
        id="row",
        input="describe it",
        expected="a cat",
        metadata={"image": "data:image/png;base64,ZmFrZQ=="},
    )
    messages = _case_messages(Environment(), case)

    assert len(messages) == 1
    text, image = messages[0]["content"]
    assert text == {"type": "text", "text": "describe it"}
    assert image == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,ZmFrZQ=="},
    }


def test_env_eval_leaves_a_text_only_prompt_exactly_as_the_environment_built_it() -> None:
    # the common path must stay byte-identical: no content-block rewrapping for a text suite, or
    # every existing non-multimodal environment would be graded on a prompt shape it never trained
    # on. this is the half that must NOT change, so it needs no PIL at all.
    from flash.cli.commands.env.testing.eval import _case_messages

    class Environment:
        def prompt_messages(self, example):
            return [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": example["input"]},
            ]

    messages = _case_messages(Environment(), EvalCase(id="t", input="2+2", expected="4"))

    assert messages == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "2+2"},
    ]


def test_env_eval_strips_reasoning_only_for_a_thinking_run(monkeypatch, tmp_path, capsys) -> None:
    """Graders must see what training graded, and only when the run actually reasons.

    Strip thinking exactly when the run spec says ``thinking``; raw reasoning breaks scorers, while
    unconditional stripping corrupts ordinary answers that mention ``<think>``.
    """

    def _suite(name: str, expected: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        env_dir = _environment_dir(root, monkeypatch=monkeypatch)
        (env_dir / "evaluations.py").write_text(
            "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase, EvalResult\n"
            f"EXPECTED = {expected!r}\n"
            "class Suite(BaseEvalSuite):\n"
            "    name = 'reasoning'\n"
            "    def cases(self): return [EvalCase(id='sum', input='2+2', expected=EXPECTED)]\n"
            "    def score(self, case, response):\n"
            # exact match, not the base class substring test: the whole question is which
            # string reaches the scorer, and a substring check passes on either one.
            "        ok = str(response) == EXPECTED\n"
            "        return EvalResult(case_id='sum', passed=ok, score=float(ok),\n"
            "                          response=str(response))\n"
            "def load_evaluations(environment=None): return [Suite()]\n"
        )
        return env_dir

    class Client(_EvalClient):
        def __init__(self, thinking, response):
            self._thinking = thinking
            self._response = response

        def get_run(self, run_id):
            return _run_record(
                {
                    "thinking": self._thinking,
                    "environment": {"id": _PUBLISHED_SLUG},
                }
            )

        def chat_stream(self, target, messages, **kwargs):
            yield self._response

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )

    # a thinking run: the reasoning is cut, so the grader sees the answer it was trained against.
    # before the fix the scorer saw the whole `<think>...` string and the case failed.
    monkeypatch.setattr(
        "flash.client.client_from_config", lambda: Client(True, "<think>2+2 is 4</think>4")
    )
    _suite("thinking", "4")
    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 0

    # a non-thinking run answering *about* the tag: `strip_think` would cut at the bare mention and
    # leave "answer: ", so stripping here would fail a correct response.
    mention = "answer: <think> is a reasoning tag"
    monkeypatch.setattr("flash.client.client_from_config", lambda: Client(False, mention))
    _suite("plain", mention)
    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 0

    assert capsys.readouterr().out.count("case sum: PASS") == 2


def test_env_eval_refuses_to_grade_when_the_plane_never_answered(
    monkeypatch, tmp_path, capsys
) -> None:
    """An unreachable plane is retryable, so it must not be graded through.

    The chat requests can succeed while this one lookup fails, and then `thinking` stays false and
    every `<think>...</think>answer` reaches the scorer raw -- a whole paid suite uploaded as false
    failures. Distinct from an ApiError, where the plane answered and there is nothing to retry.
    """
    from flash.client import ClientError

    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'reasoning'\n"
        "    def cases(self): return [EvalCase(id='sum', input='2+2', expected='4')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client(_EvalClient):
        def __init__(self):
            self.generated = 0

        def get_run(self, run_id):
            raise ClientError("cannot reach the Flash service at https://api.example")

        def chat_stream(self, target, messages, **kwargs):
            self.generated += 1
            yield "<think>2 plus 2</think>4"

    client = Client()
    uploader = _RecordingUpload()
    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", lambda: client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 1
    # it stops BEFORE generation: nothing is bought, and no false failure is filed.
    assert client.generated == 0
    assert uploader.calls == []
    captured = capsys.readouterr()
    assert "could not resolve verified checkpoints for flash-1" in captured.err
    assert "overall: FAIL" in captured.err


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (503, "could not resolve verified checkpoints for flash-1"),
        (429, "could not resolve verified checkpoints for flash-1"),
        (500, "could not resolve verified checkpoints for flash-1"),
        (404, "could not resolve verified checkpoints for flash-1"),
        (403, "could not resolve verified checkpoints for flash-1"),
    ],
)
def test_env_eval_refuses_on_run_spec_api_error(monkeypatch, capsys, status, message) -> None:
    """Both API error classes stop grading, while their messages preserve retry guidance.

    `ApiError` subclasses `ClientError`, so the ordering still matters: 5xx and 429 mean the plane
    may recover, while a definitive 4xx answered about this run. Neither can proceed because the
    unreadable spec is what names the published environment and its suites.
    """
    from flash.client import ApiError

    class Client(_EvalClient):
        def __init__(self):
            self.generated = 0

        def get_run(self, run_id):
            raise ApiError(status, f"plane returned {status}")

        def chat_stream(self, target, messages, **kwargs):
            self.generated += 1
            yield "<think>2 plus 2</think>4"

    client = Client()
    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.client.client_from_config", lambda: client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--project", _PROJECT_ID]) == 1
    assert client.generated == 0
    assert uploader.calls == []
    captured = capsys.readouterr().err
    assert message in captured


def test_env_eval_refuses_when_the_run_spec_is_unreadable(monkeypatch, capsys) -> None:
    """An unreadable spec cannot supply the published environment, so grading is refused."""

    class Client(_EvalClient):
        def __init__(self):
            self.generated = 0

        def get_run(self, run_id):
            raise RuntimeError("run metadata is unavailable")

        def chat_stream(self, target, messages, **kwargs):
            self.generated += 1
            yield "4"

    client = Client()
    uploader = _RecordingUpload()
    monkeypatch.setattr("flash.client.client_from_config", lambda: client)
    _patch_upload(monkeypatch, uploader)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1
    assert client.generated == 0
    assert uploader.calls == []
    captured = capsys.readouterr().err
    assert "could not resolve verified checkpoints for flash-1" in captured
    assert "overall: FAIL" in captured


class _MultiTurnEnvironment:
    """Minimal stand-in for the multi-turn half of `FreesoloEnvironment`.

    Only the surface `env eval` drives: prompt state, one recorded model turn at a time, an env
    reply between turns, and a done signal. The transcript it accumulates is what a real
    multi-turn scorer grades.
    """

    multi_turn = True
    max_turns = 3

    def __init__(self) -> None:
        self.generated: list[str] = []

    def new_rollout_state(self, example):
        return {"messages": [{"role": "user", "content": "turn-1"}], "turns": []}

    def record_model_turn(self, state, content):
        self.generated.append(content)
        state["turns"].append(content)
        state["messages"] = [*state["messages"], {"role": "assistant", "content": content}]
        # what a real env exposes as the scored text: the whole transcript, not the last reply.
        state["response_text"] = "|".join(state["turns"])

    def rollout_done(self, state, max_turns=None):
        return len(state["turns"]) >= self.max_turns

    def env_reply(self, messages, state):
        reply = [{"role": "user", "content": f"turn-{len(state['turns']) + 1}"}]
        state["messages"] = [*messages, *reply]
        return reply


def _multi_turn_env_dir(tmp_path, monkeypatch):
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'episodes'\n"
        # the suite opts in: a multi-turn env alone must not promote a suite to episode grading.
        "    grades_episodes = True\n"
        "    def cases(self): return [EvalCase(id='only', input='only', expected='a|b|c')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    return env_dir


def test_env_eval_drives_every_turn_of_a_multi_turn_environment(
    monkeypatch, tmp_path, capsys
) -> None:
    """A multi-turn case must generate once per turn, not once per case.

    Scoring one reply of a task whose reward reads the whole transcript reports a number for a
    different task, which is indistinguishable from a weak model.
    """
    _multi_turn_env_dir(tmp_path, monkeypatch)
    environment = _MultiTurnEnvironment()
    replies = iter(["a", "b", "c"])

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield next(replies)

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: environment
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 0
    # one generation per turn, and the graded text is the accumulated transcript.
    assert environment.generated == ["a", "b", "c"]
    captured = capsys.readouterr()
    assert "case only: PASS" in captured.out
    assert "each episode will still be played out with one generation per turn" in captured.err
    assert "only the episode's final response text, not the transcript" in captured.err
    assert "score(case, response, state)" in captured.err


def test_env_eval_reports_a_failed_turn_without_losing_the_case(
    monkeypatch, tmp_path, capsys
) -> None:
    """A generation failure mid-episode fails that case, not the whole suite.

    The failure is raised on the second turn, which only ever happens once the episode is driven,
    so this also pins that a partially-played episode reports the reason instead of a bare 0.0.
    """
    _multi_turn_env_dir(tmp_path, monkeypatch)
    environment = _MultiTurnEnvironment()

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            if environment.generated:
                raise RuntimeError("upstream refused")
            yield "a"

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: environment
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1
    captured = capsys.readouterr()
    # the upstream reason has to survive to the case line; a bare 0.0 would read as a weak model.
    assert "case only: FAIL" in captured.out
    assert "upstream refused" in captured.out
    assert "errors=1" in captured.out


def test_env_eval_keeps_one_generation_per_case_for_single_turn(
    monkeypatch, tmp_path, capsys
) -> None:
    """The single-turn path must not start driving episodes."""
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'single'\n"
        "    def cases(self): return [EvalCase(id='only', input='only', expected='only')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    calls: list[str] = []

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            calls.append(messages[0]["content"])
            yield messages[0]["content"]

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 0
    assert calls == ["only"]
    assert "case only: PASS" in capsys.readouterr().out


def test_env_eval_does_not_promote_a_first_action_suite_to_an_episode(
    monkeypatch, tmp_path, capsys
) -> None:
    """A multi-turn ENVIRONMENT must not turn a single-shot SUITE into an episode.

    `flash env setup --multi` scaffolds exactly this pair: a multi-turn env plus a suite that
    grades only the opening action. Playing its cases as episodes scored the wrong turn, and the
    scaffolded case carries no `output` for `step_episode` to advance from, so a well-formed
    opening action came back as an error instead of 1.0.
    """
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'first-action'\n"
        # no grades_episodes flag: this suite scores one reply, like every pre-flag suite.
        "    def cases(self): return [EvalCase(id='only', input='only', expected='50')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )
    environment = _MultiTurnEnvironment()
    calls: list[str] = []

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            calls.append("gen")
            yield "50"

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: environment
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 0
    # one generation, graded as the opening action -- the episode driver never ran.
    assert calls == ["gen"]
    assert environment.generated == []
    captured = capsys.readouterr()
    assert "case only: PASS" in captured.out
    assert "one generation per turn" not in captured.err


def test_env_eval_refuses_an_episode_suite_on_a_single_turn_environment(
    monkeypatch, tmp_path, capsys
) -> None:
    """An episode suite on a single-turn env has no transcript to grade, so say so."""
    env_dir = _environment_dir(tmp_path, monkeypatch=monkeypatch)
    (env_dir / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase\n"
        "class Suite(BaseEvalSuite):\n"
        "    name = 'episodes'\n"
        "    grades_episodes = True\n"
        "    def cases(self): return [EvalCase(id='only', input='only', expected='x')]\n"
        "def load_evaluations(environment=None): return [Suite()]\n"
    )

    class Client(_EvalClient):
        def chat_stream(self, target, messages, **kwargs):
            yield "x"

    monkeypatch.setattr(
        "flash.envs.loading.loader.load_freesolo_environment", lambda _path: object()
    )
    monkeypatch.setattr("flash.client.client_from_config", Client)

    assert cli.main(["env", "eval", _EXPLICIT_TARGET, "--no-upload"]) == 1
    captured = capsys.readouterr()
    # an error, not a 0.0: a mismatched pairing is unmeasurable, not a bad model.
    assert "single-turn" in captured.out + captured.err
    assert "errors=1" in captured.out


def test_env_eval_normalizes_images_on_every_episode_turn() -> None:
    """Each turn's prompt must go through the same image normalization as a single-turn case.

    `_drive_episode` builds its prompt from rollout state rather than `_case_messages`, so it does
    not inherit that normalization for free. Sending the state messages raw ships a data-URI-less
    `image` block the chat API cannot read, and the case would still be graded -- scoring a model
    that never saw the image.
    """
    import argparse

    from flash.cli.commands.env.testing import episode as episode_module
    from flash.cli.commands.env.testing import eval as env_eval

    class Environment:
        multi_turn = True
        max_turns = 1
        package_root = None

        def new_rollout_state(self, example):
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
                            },
                            {"type": "text", "text": "describe it"},
                        ],
                    }
                ],
                "turns": [],
            }

        def record_model_turn(self, state, content):
            state["turns"].append(content)
            state["response_text"] = content

        def rollout_done(self, state, max_turns=None):
            return True

        def env_reply(self, messages, state):
            return []

    sent: list[list[dict]] = []

    def capture(client, target, messages, args):
        sent.append(messages)
        return "a cat"

    original = env_eval._generate_case
    env_eval._generate_case = capture
    try:
        episode_module._drive_episode(
            object(),
            "t",
            Environment(),
            EvalCase(id="row", input="describe it"),
            argparse.Namespace(),
        )
    finally:
        env_eval._generate_case = original

    assert len(sent) == 1
    content = sent[0][0]["content"]
    # the raw `image` block must be gone, rewritten into the image_url shape the backend reads.
    assert not any(block.get("type") == "image" for block in content)
    assert {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        },
    } in content


def test_env_eval_hands_the_finished_episode_to_a_suite_that_accepts_it(capsys) -> None:
    """A transcript-grading suite must receive the episode, not just the last turn.

    `record_model_turn` overwrites `state["response_text"]` on every turn, so the scalar is the
    LAST reply. Scoring a running-total task on that grades one turn of an episode the run just
    paid to play out -- the same defect as generating once, moved one step later.
    """
    import argparse

    from flash.cli.commands.env.testing import episode as episode_module
    from flash.cli.commands.env.testing import eval as env_eval

    class Environment:
        multi_turn = True
        max_turns = 3
        package_root = None

        def new_rollout_state(self, example):
            return {"messages": [{"role": "user", "content": "go"}], "turns": []}

        def record_model_turn(self, state, content):
            state["turns"].append(content)
            state["messages"] = [*state["messages"], {"role": "assistant", "content": content}]
            state["response_text"] = content

        def rollout_done(self, state, max_turns=None):
            return len(state["turns"]) >= 3

        def env_reply(self, messages, state):
            reply = [{"role": "user", "content": "next"}]
            state["messages"] = [*messages, *reply]
            return reply

    class TranscriptSuite:
        name = "running-total"
        grades_episodes = True

        def cases(self):
            return [EvalCase(id="c", input="x", expected="1,3,6")]

        def score(self, case, response, state):
            turns = state.get("turns") or []
            seen = ",".join(turns)
            return EvalResult(
                case_id=case.id,
                passed=seen == case.expected,
                score=1.0 if seen == case.expected else 0.0,
                response=response,
                reason=f"saw {seen}",
            )

    replies = iter(["1", "3", "6"])
    original = env_eval._generate_case
    env_eval._generate_case = lambda client, target, messages, args: next(replies)
    try:
        suite = TranscriptSuite()
        results = episode_module._run_episode_cases(
            object(), "t", suite, suite.cases(), argparse.Namespace(concurrency=1), Environment()
        )
    finally:
        env_eval._generate_case = original

    assert results[0].error is None
    assert results[0].score == 1.0, results[0].reason
    assert results[0].reason == "saw 1,3,6"
    assert "one generation per turn" not in capsys.readouterr().err


def test_env_eval_keeps_the_two_argument_scorer_contract_for_episodes(capsys, monkeypatch) -> None:
    """A suite that takes only (case, response) must not be handed a third argument."""
    import argparse

    from flash.cli.commands.env.testing import episode as episode_module
    from flash.cli.commands.env.testing import eval as env_eval

    style_checks = 0
    real_state_argument = episode_module._state_argument

    def counted_state_argument(scorer):
        nonlocal style_checks
        style_checks += 1
        return real_state_argument(scorer)

    monkeypatch.setattr(episode_module, "_state_argument", counted_state_argument)

    class Environment:
        multi_turn = True
        max_turns = 2
        package_root = None

        def new_rollout_state(self, example):
            return {"messages": [{"role": "user", "content": "go"}], "turns": []}

        def record_model_turn(self, state, content):
            state["turns"].append(content)
            state["messages"] = [*state["messages"], {"role": "assistant", "content": content}]
            state["response_text"] = content

        def rollout_done(self, state, max_turns=None):
            return len(state["turns"]) >= 2

        def env_reply(self, messages, state):
            return [{"role": "user", "content": "next"}]

    class TextOnlySuite:
        name = "last-turn"
        grades_episodes = True

        def cases(self):
            return [
                EvalCase(id="c1", input="x", expected="b"),
                EvalCase(id="c2", input="x", expected="d"),
            ]

        def score(self, case, response):
            return EvalResult(
                case_id=case.id,
                passed=response == case.expected,
                score=1.0 if response == case.expected else 0.0,
                response=response,
            )

    replies = iter(["a", "b", "c", "d"])
    original = env_eval._generate_case
    env_eval._generate_case = lambda client, target, messages, args: next(replies)
    try:
        suite = TextOnlySuite()
        results = episode_module._run_episode_cases(
            object(), "t", suite, suite.cases(), argparse.Namespace(concurrency=1), Environment()
        )
    finally:
        env_eval._generate_case = original

    # both cases are scored on their last turn, with no unexpected third argument.
    assert [result.error for result in results] == [None, None]
    assert [result.score for result in results] == [1.0, 1.0]
    assert style_checks == 1
    warning = capsys.readouterr().err
    assert warning.count("each episode will still be played out") == 1


def test_grades_episodes_requires_a_boolean() -> None:
    from flash.cli.commands.env.testing.episode import _grades_episodes

    class Suite:
        name = "episode"
        grades_episodes = "false"

    with pytest.raises(TypeError, match="grades_episodes must be a bool, got str"):
        _grades_episodes(Suite())


def test_state_argument_detects_how_the_scorer_takes_state() -> None:
    # detecting only WHETHER state is accepted is not enough: **kwargs and keyword-only scorers
    # reject a third positional, and *args scorers reject the keyword. the two groups are disjoint,
    # so the call style has to come from the signature.
    from flash.cli.commands.env.testing.episode import _state_argument

    assert _state_argument(lambda case, response: None) is None
    assert _state_argument(None) is None
    assert _state_argument(lambda case, response, state=None: None) == "keyword"
    assert _state_argument(lambda case, response, *, state=None: None) == "keyword"
    assert _state_argument(lambda case, response, **kwargs: None) == "keyword"
    assert _state_argument(lambda case, response, *args, **kwargs: None) == "keyword"
    assert _state_argument(lambda *args: None) == "positional"
    assert _state_argument(lambda case, *args: None) == "positional"
    assert _state_argument(lambda case, response, *args: None) == "positional"
    assert _state_argument(lambda case, response, state, /: None) == "positional"
    assert _state_argument(lambda case, response, state, /, **kwargs: None) == "positional"
    assert _state_argument(lambda case, response, threshold=0.5: None) is None
    assert _state_argument(lambda case, response, threshold=0.5, *args: None) is None
    assert _state_argument(lambda case, response, threshold, state, /: None) is None


@pytest.mark.parametrize(
    "scorer_source",
    [
        "def score(self, case, response, state=None): return _graded(case, state)",
        "def score(self, case, response, state, /): return _graded(case, state)",
        "def score(self, case, response, state, /, **kwargs): return _graded(case, state)",
        "def score(self, case, response, *, state=None): return _graded(case, state)",
        "def score(self, case, response, **kwargs): return _graded(case, kwargs.get('state'))",
        "def score(self, *args): return _graded(args[0], args[2])",
        "def score(self, case, *args): return _graded(case, args[1])",
        "def score(self, case, response, *args): return _graded(case, args[0])",
    ],
    ids=[
        "state",
        "state_positional_only",
        "state_positional_only_kwargs",
        "keyword_only",
        "kwargs",
        "varargs_only",
        "case_then_varargs",
        "case_response_then_varargs",
    ],
)
def test_every_state_accepting_scorer_shape_actually_receives_the_episode(
    scorer_source: str,
) -> None:
    """Each signature that opts in must be CALLED in the way it can accept.

    Detection alone shipped a scorer that was marked supported and then called with a third
    positional argument it could not take, so every case came back `scoring failed` -- a hard zero
    that looks like the model failed. Asserting the score, not the detection, is what catches that.
    """
    from flash.cli.commands.env.testing.episode import _state_argument
    from flash.cli.commands.env.testing.eval import _score_case

    namespace: dict = {
        "_graded": lambda case, state: (
            1.0 if ",".join((state or {}).get("turns", [])) == case.expected else 0.0
        )
    }
    exec(
        f"class Suite:\n    name = 'suite'\n    grades_episodes = True\n    {scorer_source}",
        namespace,
    )
    suite = namespace["Suite"]()

    case = EvalCase(input="count up", expected="1,3,6", id="c1")
    state = {"turns": ["1", "3", "6"], "response_text": "6"}

    style = _state_argument(suite.score)
    assert style is not None
    result = _score_case(suite, case, "c1", "6", state=state, state_keyword=style == "keyword")

    # the whole transcript reached the scorer, and no TypeError was reported as a scoring failure
    assert result.error is None
    assert result.score == 1.0


def test_env_eval_applies_the_last_model_turn_before_scoring_at_the_turn_cap() -> None:
    """The final model action must reach the scored state, including at the turn cap.

    `env_reply` is what runs the env's `step_episode`, so a state that never sees the last action
    is scored with board state, metadata and `final_response_text` one move stale -- the model is
    graded as though its last turn never happened.

    Training does apply it. `rl/multi_turn.py` gates its reply solely on `rollout_done`, which
    counts `state["turn"]` -- incremented only by `env_reply`, never by `record_model_turn` -- so
    at the cap the counter is still one short, the check passes, and the env is stepped.

    What is skipped is only the inter-turn glue: no SECOND reply that would append a user turn no
    model turn will ever answer, which is what `opd/bridge.py` guards against.
    """
    import argparse

    from flash.cli.commands.env.testing import episode as episode_module
    from flash.cli.commands.env.testing import eval as env_eval

    class Environment:
        multi_turn = True
        max_turns = 2
        package_root = None

        def __init__(self):
            self.env_replies = 0

        def new_rollout_state(self, example):
            return {"messages": [{"role": "user", "content": "go"}], "turns": [], "turn": 0}

        def record_model_turn(self, state, content):
            state["turns"].append(content)
            state["messages"].append({"role": "assistant", "content": content})
            state["response_text"] = content

        def rollout_done(self, state, max_turns=None):
            # mirrors the adapter: the counter only advances on env_reply
            if state.get("done"):
                return True
            return max_turns is not None and int(state.get("turn", 0)) >= int(max_turns)

        def env_reply(self, messages, state):
            self.env_replies += 1
            state["turn"] = int(state.get("turn", 0)) + 1
            # stand-in for step_episode's side effects: only a reply makes the last action count
            state["applied"] = list(state["turns"])
            state["messages"] = [*messages, {"role": "user", "content": "next"}]
            return [{"role": "user", "content": "next"}]

    replies = iter(["a", "b"])
    environment = Environment()
    original = env_eval._generate_case
    env_eval._generate_case = lambda client, target, messages, args: next(replies)
    try:
        state = episode_module._drive_episode(
            object(), "t", environment, EvalCase(id="c", input="x"), argparse.Namespace()
        )
    finally:
        env_eval._generate_case = original

    assert state["turns"] == ["a", "b"]
    # the capped turn "b" was applied, so the scored state is not one action stale
    assert state["applied"] == ["a", "b"]
    # one inter-turn reply plus the final apply; no third that would glue on an unanswerable prompt
    assert environment.env_replies == 2


def test_env_eval_does_not_step_an_environment_that_already_reported_done() -> None:
    """An env that ended the episode itself must not be stepped again before scoring.

    `env_reply` runs `step_episode`. Once the env reports done there is nothing left to apply, and
    both `env test` and the RL worker stop there. Stepping anyway scores a state past the end of
    the episode -- and for an env that validates each action, can raise on a turn that never was.
    """
    import argparse

    from flash.cli.commands.env.testing import episode as episode_module
    from flash.cli.commands.env.testing import eval as env_eval

    class Environment:
        multi_turn = True
        max_turns = 5
        package_root = None

        def __init__(self):
            self.env_replies = 0

        def new_rollout_state(self, example):
            return {"messages": [{"role": "user", "content": "go"}], "turns": [], "turn": 0}

        def record_model_turn(self, state, content):
            state["turns"].append(content)
            state["messages"].append({"role": "assistant", "content": content})
            state["response_text"] = content
            # the env decides the episode is over the moment it sees this action
            if content == "stop":
                state["done"] = True

        def rollout_done(self, state, max_turns=None):
            if state.get("done"):
                return True
            return max_turns is not None and int(state.get("turn", 0)) >= int(max_turns)

        def env_reply(self, messages, state):
            self.env_replies += 1
            state["turn"] = int(state.get("turn", 0)) + 1
            state["messages"] = [*messages, {"role": "user", "content": "next"}]
            return [{"role": "user", "content": "next"}]

    replies = iter(["stop"])
    environment = Environment()
    original = env_eval._generate_case
    env_eval._generate_case = lambda client, target, messages, args: next(replies)
    try:
        state = episode_module._drive_episode(
            object(), "t", environment, EvalCase(id="c", input="x"), argparse.Namespace()
        )
    finally:
        env_eval._generate_case = original

    assert state["turns"] == ["stop"]
    # the env ended it; no step may follow
    assert environment.env_replies == 0


def test_env_eval_honours_a_per_example_turn_budget_below_the_env_ceiling() -> None:
    """A case whose `max_episode_turns` is lower than `max_turns` must stop at its own budget.

    `rollout_done` gives the per-example budget precedence (`flash/envs/loading/adapter.py`), and both
    training bridges derive the same effective limit (`opd/bridge.py:512`, `rl/multi_turn.py:174`).
    Counting only the dataset-wide ceiling generates turns training would never have run.
    """
    import argparse

    from flash.cli.commands.env.testing import episode as episode_module
    from flash.cli.commands.env.testing import eval as env_eval

    class Environment:
        multi_turn = True
        max_turns = 6
        package_root = None

        def __init__(self):
            self.env_replies = 0

        def new_rollout_state(self, example):
            return {
                "messages": [{"role": "user", "content": "go"}],
                "turns": [],
                "turn": 0,
                # this case gets a shorter budget than the dataset-wide ceiling
                "max_episode_turns": 2,
            }

        def record_model_turn(self, state, content):
            state["turns"].append(content)
            state["messages"].append({"role": "assistant", "content": content})
            state["response_text"] = content

        def rollout_done(self, state, max_turns=None):
            # deliberately honours ONLY the cap it is handed, so this test measures the cap the
            # driver computes rather than the env re-deriving the budget for it. the real adapter
            # applies the same precedence internally (flash/envs/loading/adapter.py:670), which would mask
            # a driver that passed the wrong ceiling.
            if state.get("done"):
                return True
            return max_turns is not None and int(state.get("turn", 0)) >= int(max_turns)

        def env_reply(self, messages, state):
            self.env_replies += 1
            state["turn"] = int(state.get("turn", 0)) + 1
            state["messages"] = [*messages, {"role": "user", "content": "next"}]
            return [{"role": "user", "content": "next"}]

    generated = []

    def _fake_generate(client, target, messages, args):
        generated.append(messages)
        return f"turn{len(generated)}"

    environment = Environment()
    original = env_eval._generate_case
    env_eval._generate_case = _fake_generate
    try:
        state = episode_module._drive_episode(
            object(), "t", environment, EvalCase(id="c", input="x"), argparse.Namespace()
        )
    finally:
        env_eval._generate_case = original

    # two turns, not the six the dataset-wide ceiling would have allowed
    assert state["turns"] == ["turn1", "turn2"]
    assert len(generated) == 2


def test_env_eval_strips_reasoning_from_the_state_a_thinking_episode_scores() -> None:
    """A thinking run must not hand the suite raw `<think>` text through the episode state.

    `_score_case` strips the separate `response` argument, so scoring that alone hides this. But a
    `grades_episodes` suite reads `state["response_text"]`, which `record_model_turn` fills via
    `_scored_turn_text` -- and that returns the turn UNSTRIPPED while the adapter's `thinking` flag
    is at its default False. The suite then sees `<think>2+2 is 4</think>4` and marks a correct
    answer wrong. Training sets the flag on the env, so eval has to as well.
    """
    import argparse

    from flash.cli.commands.env.testing import episode as episode_module
    from flash.cli.commands.env.testing import eval as env_eval
    from flash.envs.loading.adapter import FreesoloEnvironment

    class Environment:
        """Uses the real adapter's turn-recording, which is where the stripping lives."""

        multi_turn = True
        max_turns = 1
        package_root = None

        def __init__(self):
            self.thinking = False
            self.prompt_opens_thinking = False

        def new_rollout_state(self, example):
            return {"messages": [{"role": "user", "content": "2+2?"}], "turns": [], "turn": 0}

        def record_model_turn(self, state, content):
            scored = FreesoloEnvironment._scored_turn_text(self, content)
            state["turns"].append(scored)
            state["messages"].append({"role": "assistant", "content": content})
            state["response_text"] = scored

        def rollout_done(self, state, max_turns=None):
            if state.get("done"):
                return True
            return max_turns is not None and int(state.get("turn", 0)) >= int(max_turns)

        def env_reply(self, messages, state):
            state["turn"] = int(state.get("turn", 0)) + 1
            return []

    seen = {}

    class Suite:
        name = "episode"
        grades_episodes = True

        def score(self, case, response, state=None):
            seen["state_response"] = str((state or {}).get("response_text"))
            return 1.0 if seen["state_response"] == "4" else 0.0

    environment = Environment()
    original = env_eval._generate_case
    env_eval._generate_case = lambda client, target, messages, args: "<think>2+2 is 4</think>4"
    try:
        results = episode_module._run_episode_cases(
            object(),
            "t",
            Suite(),
            [EvalCase(id="c", input="2+2?")],
            argparse.Namespace(),
            environment,
            thinking=True,
        )
    finally:
        env_eval._generate_case = original

    # the scorer read the answer, not the reasoning wrapper
    assert seen["state_response"] == "4"
    assert results[0].score == 1.0
    # and the caller's environment is left as it was found
    assert environment.thinking is False


def test_loaded_suite_still_receives_episode_state_through_the_scope_wrapper(tmp_path) -> None:
    """The loader's scope wrapper must not hide a sidecar's episode-state parameter.

    `load_evaluation_suites` wraps every suite to bind it to its package directory. That wrapper
    is what callers inspect, so a fixed `(case, response)` signature on it made EVERY real
    sidecar look like a two-argument scorer: `grades_episodes` still forwarded through
    `__getattr__` and promised transcript grading, while the state was silently dropped and the
    suite raised on the state it was told to expect. Only a suite constructed directly in a test
    escaped it, which is why unit coverage passed while the shipped path did not.
    """
    from flash.cli.commands.env.testing.episode import _grades_episodes, _state_argument
    from flash.envs.meta.evaluations import load_evaluation_suites

    package = tmp_path / "env_pkg"
    package.mkdir()
    (package / "environment.py").write_text(
        "def load_environment(**kwargs):\n    return object()\n"
    )
    (package / "evaluations.py").write_text(
        "from flash.envs.meta.evaluations import EvalCase\n"
        "\n"
        "class EpisodeSuite:\n"
        "    name = 'episode'\n"
        "    grades_episodes = True\n"
        "    def cases(self):\n"
        "        return [EvalCase(input='go', expected='1,3,6', id='c1')]\n"
        "    def score(self, case, response, state=None):\n"
        "        return 1.0 if ','.join((state or {}).get('turns', [])) == case.expected else 0.0\n"
        "\n"
        "class PlainSuite:\n"
        "    name = 'plain'\n"
        "    def cases(self):\n"
        "        return [EvalCase(input='go', expected='x', id='c1')]\n"
        "    def score(self, case, response):\n"
        "        return 1.0 if response == case.expected else 0.0\n"
        "\n"
        "def load_evaluations(environment=None, **kwargs):\n"
        "    return [EpisodeSuite(), PlainSuite()]\n"
    )

    episode_suite, plain_suite = load_evaluation_suites(str(package / "environment.py"))

    # the opt-in survives the wrapper, and so does the parameter that makes it meaningful
    assert _grades_episodes(episode_suite) is True
    assert _state_argument(episode_suite.score) == "keyword"
    # and the state actually arrives, rather than being dropped by a fixed-arity delegate
    case = episode_suite.cases()[0]
    assert episode_suite.score(case, "6", state={"turns": ["1", "3", "6"]}) == 1.0

    # a two-argument suite must NOT be reported as taking state, or it gets an argument it
    # cannot accept -- forwarding *args/**kwargs alone would report every suite as accepting it
    assert _grades_episodes(plain_suite) is False
    assert _state_argument(plain_suite.score) is None
    assert plain_suite.score(plain_suite.cases()[0], "x") == 1.0
