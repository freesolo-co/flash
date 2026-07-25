"""`flash traces export` and the trace import offered by `flash env setup`.

Both read the freesolo backend through flash.cli.traces, so these tests stub that module's
client calls and drive main() in-process: no server, no stored credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import flash.cli as cli
from flash.cli import env_setup, traces
from flash.client import ApiError, ClientError

_PROJECTS = [
    {"id": "proj-1", "name": "support triage", "updated_at": "2026-07-02T00:00:00Z"},
    {"id": "proj-2", "name": "docs qa", "updated_at": "2026-07-01T00:00:00Z"},
]
_RECORDS = [
    {"input": "What is 2 + 2?", "output": "4"},
    {"input": "What is 3 + 5?", "output": "8"},
]


@pytest.fixture
def fake_traces(monkeypatch):
    """Stub the freesolo traces client; records what each command asked for."""
    calls: dict[str, list] = {"projects": [], "records": []}

    def list_projects(api_key, base_url=None):
        calls["projects"].append(api_key)
        return _PROJECTS

    def export_records(project_id, api_key, base_url=None, limit=None):
        calls["records"].append(project_id)
        return {"records": _RECORDS, "traces": 3, "skipped": 1}

    monkeypatch.setattr(traces, "list_trace_projects", list_projects)
    monkeypatch.setattr(traces, "export_trace_records", export_records)
    monkeypatch.setattr(traces, "load_credentials", lambda: ("https://flash", "fs-key"))
    return calls


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _reject_traces_prompt(why: str):
    """A render.select stand-in that fails if asked about traces, taking the default otherwise.

    The scaffold asks several questions through the same helper, so a guard has to single out
    the traces one by title."""

    def select(title, options, default=0):
        if title.startswith("Start from"):
            pytest.fail(f"must not offer traces: {why}")
        return options[default][0]

    return select


def test_traces_export_writes_env_records(fake_traces, monkeypatch, tmp_path, capsys) -> None:
    """The exported file is a drop-in dataset: one {"input","output"} object per line."""
    monkeypatch.chdir(tmp_path)

    assert cli.main(["traces", "export", "--project", "proj-1"]) == 0

    out = tmp_path / "dataset/train.jsonl"
    assert _rows(out) == _RECORDS
    assert fake_traces["records"] == ["proj-1"]
    # no project listing needed when --project names one
    assert fake_traces["projects"] == []
    printed = capsys.readouterr().out
    assert "exported 2 training rows to dataset/train.jsonl" in printed
    assert "1 traces skipped" in printed  # the unusable trace is reported, not hidden


def test_traces_export_honours_output_flag(fake_traces, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    assert cli.main(["traces", "export", "--project", "proj-1", "-o", "rows/mine.jsonl"]) == 0

    assert _rows(tmp_path / "rows/mine.jsonl") == _RECORDS  # parent dir created


def test_traces_export_refuses_to_clobber_without_force(fake_traces, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "dataset/train.jsonl"
    existing.parent.mkdir()
    existing.write_text("keep me\n")

    assert cli.main(["traces", "export", "--project", "proj-1"]) == 1
    assert existing.read_text() == "keep me\n"

    assert cli.main(["traces", "export", "--project", "proj-1", "--force"]) == 0
    assert _rows(existing) == _RECORDS


def test_traces_export_without_project_is_actionable_when_scripted(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    """Non-interactive and ambiguous: name the projects instead of hanging on a prompt."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(traces, "_interactive", lambda: False)

    assert cli.main(["traces", "export"]) == 1
    err = capsys.readouterr().err
    assert "--project" in err
    assert "proj-1" in err
    assert "proj-2" in err


def test_traces_export_picks_the_only_project_without_prompting(
    fake_traces, monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(traces, "list_trace_projects", lambda *a, **k: [_PROJECTS[0]])
    monkeypatch.setattr(traces, "_interactive", lambda: False)

    assert cli.main(["traces", "export"]) == 0

    assert fake_traces["records"] == ["proj-1"]


def test_traces_export_requires_login(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(traces, "load_credentials", lambda: ("https://flash", None))

    assert cli.main(["traces", "export", "--project", "proj-1"]) == 1
    assert "flash login" in capsys.readouterr().err


def test_traces_export_reports_a_project_with_no_usable_traces(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        traces, "export_trace_records", lambda *a, **k: {"records": [], "traces": 0, "skipped": 0}
    )

    assert cli.main(["traces", "export", "--project", "proj-1"]) == 1
    assert "no exportable traces" in capsys.readouterr().err
    assert not (tmp_path / "dataset/train.jsonl").exists()  # nothing written on failure


def test_env_setup_can_start_from_a_projects_traces(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    """Picking a project in `env setup` lands its traces straight in dataset/train.jsonl."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda args: True)
    asked: list[str] = []

    def fake_select(title, options, default=0):
        asked.append(title)
        return options[0][0] if title.startswith("Start from") else options[default][0]

    monkeypatch.setattr(env_setup.render, "select", fake_select)

    assert cli.main(["env", "setup"]) == 0

    assert asked[0] == "Start from a project's traces?"  # the data question comes first
    assert _rows(tmp_path / "dataset/train.jsonl") == _RECORDS
    assert fake_traces["records"] == ["proj-1"]
    # max_examples tracks the rows actually written, so a run trains on all of them
    for config in ("configs/sft.toml", "configs/rl.toml", "configs/opd.toml"):
        text = (tmp_path / config).read_text()
        assert "max_examples = 2" in text
        assert "exported from your traces" in text
    assert "exported 2 rows from your traces" in capsys.readouterr().out


def test_env_setup_scaffolds_starter_rows_when_declined(fake_traces, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda args: True)
    # the picker's last option ("start from scratch") is the default
    monkeypatch.setattr(
        env_setup.render, "select", lambda title, options, default=0: options[default][0]
    )

    assert cli.main(["env", "setup"]) == 0

    assert '"input":"What is 2 + 2?"' in (tmp_path / "dataset/train.jsonl").read_text()
    assert fake_traces["records"] == []
    assert "the starter dataset has 2" in (tmp_path / "configs/sft.toml").read_text()


@pytest.mark.parametrize(
    "failure",
    [
        ClientError("offline"),
        ApiError(503, "traces unavailable"),
        OSError("connection reset"),
    ],
)
def test_env_setup_falls_back_silently_when_traces_are_unreachable(
    monkeypatch, tmp_path, capsys, failure
) -> None:
    """Being offline or unauthenticated must never block scaffolding a working environment."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda args: True)
    monkeypatch.setattr(traces, "load_credentials", lambda: ("https://flash", "fs-key"))

    def boom(*args, **kwargs):
        raise failure

    monkeypatch.setattr(traces, "list_trace_projects", boom)
    monkeypatch.setattr(env_setup.render, "select", _reject_traces_prompt("no projects to offer"))

    assert cli.main(["env", "setup"]) == 0

    assert '"input":"What is 2 + 2?"' in (tmp_path / "dataset/train.jsonl").read_text()
    assert capsys.readouterr().err == ""  # a silent fallback, not a scary warning


def test_env_setup_does_not_offer_traces_when_logged_out(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda args: True)
    monkeypatch.setattr(traces, "load_credentials", lambda: ("https://flash", None))
    monkeypatch.setattr(
        traces,
        "list_trace_projects",
        lambda *a, **k: pytest.fail("must not call the backend without a key"),
    )
    monkeypatch.setattr(
        env_setup.render, "select", lambda title, options, default=0: options[default][0]
    )

    assert cli.main(["env", "setup"]) == 0

    assert '"input":"What is 2 + 2?"' in (tmp_path / "dataset/train.jsonl").read_text()


def test_env_setup_keeps_an_existing_dataset(fake_traces, monkeypatch, tmp_path) -> None:
    """An existing dataset is never overwritten, so we don't download rows we'd discard."""
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset/train.jsonl"
    dataset.parent.mkdir()
    dataset.write_text('{"input":"mine","output":"kept"}\n')
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda args: True)
    monkeypatch.setattr(env_setup.render, "select", _reject_traces_prompt("existing dataset"))

    assert cli.main(["env", "setup"]) == 0

    assert dataset.read_text() == '{"input":"mine","output":"kept"}\n'
    assert fake_traces["records"] == []


def test_env_setup_never_prompts_for_traces_under_yes(fake_traces, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        traces,
        "list_trace_projects",
        lambda *a, **k: pytest.fail("--yes must not reach the backend"),
    )

    assert cli.main(["env", "setup", "--yes"]) == 0

    assert '"input":"What is 2 + 2?"' in (tmp_path / "dataset/train.jsonl").read_text()


def test_records_to_jsonl_round_trips() -> None:
    parsed = [json.loads(line) for line in traces.records_to_jsonl(_RECORDS).splitlines()]
    assert parsed == _RECORDS
    assert traces.records_to_jsonl([]) == ""


def test_project_options_label_by_name_and_skip_id_less_rows() -> None:
    options = traces.project_options([*_PROJECTS, {"name": "no id"}])
    assert [(value, label) for value, label, _hint in options] == [
        ("proj-1", "support triage"),
        ("proj-2", "docs qa"),
    ]
