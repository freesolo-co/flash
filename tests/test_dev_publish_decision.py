"""A just-published dev version must not be mistaken for a new release."""

import importlib.util
import os
import subprocess
import urllib.error
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/publish-dev.yml").read_text())
STEPS = WORKFLOW["jobs"]["publish-pypi-dev"]["steps"]


@pytest.fixture
def probe(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "check_dev_release", ROOT / "scripts/check_dev_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("VERSION", "1.4.7")
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


@pytest.mark.parametrize(
    ("statuses", "decision"),
    [([200], "false"), ([404, 200], "false"), ([404, 404, 200], "false"), ([404] * 3, "true")],
)
def test_publish_decision_rechecks_fresh_metadata(probe, monkeypatch, capsys, statuses, decision):
    requests = []
    remaining = iter(statuses)

    def open_url(request, timeout):
        requests.append(request)
        status = next(remaining)
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "missing", {}, None)
        return nullcontext(SimpleNamespace(status=status))

    monkeypatch.setattr(probe.urllib.request, "urlopen", open_url)
    probe.main()
    assert capsys.readouterr().out == f"publish={decision}\n"
    assert len(requests) == len(statuses)
    assert len({request.full_url for request in requests}) == len(requests)
    for request in requests:
        assert request.full_url.startswith("https://pypi.org/pypi/freesolo-flash-dev/1.4.7/json?")
        assert request.get_header("Cache-control") == "no-cache"


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("offline"),
        TimeoutError(),
        *[
            urllib.error.HTTPError("https://pypi.org", code, "failed", {}, None)
            for code in (403, 429, 500)
        ],
    ],
)
def test_lookup_failure_never_emits_permission_to_publish(probe, monkeypatch, capsys, error):
    def fail(request, timeout):
        raise error

    monkeypatch.setattr(probe.urllib.request, "urlopen", fail)
    with pytest.raises(type(error)):
        probe.main()
    assert capsys.readouterr().out == ""


def test_workflow_uses_probe_and_retains_conflicting_tag_guard(tmp_path):
    trigger = WORKFLOW.get("on", WORKFLOW.get(True))
    assert "scripts/check_dev_release.py" in trigger["push"]["paths"]
    decide = next(step for step in STEPS if step.get("id") == "decide")
    assert decide["run"] == 'python3 scripts/check_dev_release.py >> "$GITHUB_OUTPUT"'
    guard = next(
        step for step in STEPS if step.get("name") == "Fail if the dev tag exists on another commit"
    )
    assert guard["if"] == "steps.decide.outputs.publish == 'true'"
    git = tmp_path / "git"
    git.write_text("#!/bin/sh\nprintf '%s\\t%s\\n' old-sha refs/tags/dev-v1.4.7\n")
    git.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", guard["run"]],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "VERSION": "1.4.7",
            "GITHUB_SHA": "new-sha",
        },
    )
    assert result.returncode != 0
    assert "Refusing to publish" in result.stdout
