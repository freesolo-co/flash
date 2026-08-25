"""`flash traces export` and the trace import offered by `flash env setup`.

Both read the freesolo backend through flash.cli.commands.ops.traces, so these tests stub that module's
client calls and drive main() in-process: no server, no stored credentials.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

import flash.cli.parsing.main as cli
from flash.cli.commands.env.ops import setup as env_setup
from flash.cli.commands.ops import traces
from flash.client import ApiError, ClientError

# Trace export and the hosted project directory exist only on Freesolo's own deployment, so these
# tests must present a Freesolo-hosted control-plane url. Any other hostname is how the CLI infers
# a self-hosted plane (`client.http.has_freesolo_backend`), where these commands refuse by design.
_HOSTED_URL = "https://flash.freesolo.co"

_PROJECTS = [
    {
        "id": "11111111-1111-4111-8111-111111111111",
        "name": "support triage",
        "updated_at": "2026-07-02T00:00:00Z",
    },
    {
        "id": "22222222-2222-4222-8222-222222222222",
        "name": "docs qa",
        "updated_at": "2026-07-01T00:00:00Z",
    },
]
_RECORDS = [
    {"input": "What is 2 + 2?", "output": "4"},
    {"input": "What is 3 + 5?", "output": "8"},
]
# the same traces as prompts: the reply-less third call survives here.
_PROMPTS = [
    {"input": "What is 2 + 2?"},
    {"input": "What is 3 + 5?"},
    {"input": "What is 9 + 1?"},
]
_RAW = [{"id": "trace-1", "model": "gpt-4o-mini", "spans": [{"id": "span-1"}]}]


@pytest.fixture
def fake_traces(monkeypatch):
    """Stub the freesolo traces client; records what each command asked for."""
    calls: dict[str, list] = {"projects": [], "records": [], "formats": []}

    def list_projects(api_key, base_url=None):
        calls["projects"].append(api_key)
        return _PROJECTS

    def export_records(project_id, api_key, base_url=None, limit=None, export_format=None):
        calls["records"].append(project_id)
        calls["formats"].append(export_format)
        if export_format == traces.PROMPTS_FORMAT:
            # prompts keep the reply-less call that records drops, so this shape
            # yields more rows and skips nothing.
            return {"records": _PROMPTS, "traces": 3, "skipped": 0, "format": export_format}
        if export_format == traces.RAW_FORMAT:
            return {"records": _RAW, "traces": 3, "skipped": 0, "format": export_format}
        return {"records": _RECORDS, "traces": 3, "skipped": 1, "format": "records"}

    monkeypatch.setattr(traces, "list_trace_projects", list_projects)
    monkeypatch.setattr(traces, "export_trace_records", export_records)
    # A Freesolo-HOSTED url: trace export only exists on that deployment (traces are recorded by
    # the freesolo SDK into its backend), so every behaviour below is the hosted path. A
    # placeholder hostname reads as self-hosted and would short-circuit all of it at the guard.
    monkeypatch.setattr(traces, "load_credentials", lambda: (_HOSTED_URL, "fs-key"))
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: (_HOSTED_URL, "fs-key"))
    monkeypatch.setattr("flash.client.list_projects", lambda api_key: _PROJECTS)
    monkeypatch.setattr(
        "flash.client.get_project",
        lambda project_id, api_key: {"id": project_id, "name": "selected"},
    )
    return calls


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _reject_traces_prompt(why: str):
    """A render.select stand-in that fails if asked about traces, taking the default otherwise.

    The scaffold asks several questions through the same helper, so a guard has to single out
    the traces one by title."""

    def select(title, options, default=0):
        if title.startswith("Use this project"):
            pytest.fail(f"must not offer traces: {why}")
        return options[default][0]

    return select


def test_traces_export_writes_env_records(fake_traces, monkeypatch, tmp_path, capsys) -> None:
    """The exported file is a drop-in dataset: one {"input","output"} object per line."""
    monkeypatch.chdir(tmp_path)

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    out = tmp_path / "dataset/train.jsonl"
    assert _rows(out) == _RECORDS
    assert fake_traces["records"] == ["11111111-1111-4111-8111-111111111111"]
    # no project listing needed when --project names one
    assert fake_traces["projects"] == []
    printed = capsys.readouterr().out
    assert "exported 2 training rows to dataset/train.jsonl" in printed
    assert "1 traces skipped" in printed  # the unusable trace is reported, not hidden


def test_traces_export_honours_output_flag(fake_traces, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    assert (
        cli.main(
            [
                "traces",
                "export",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "-o",
                "rows/mine.jsonl",
            ]
        )
        == 0
    )

    assert _rows(tmp_path / "rows/mine.jsonl") == _RECORDS  # parent dir created


def test_traces_export_refuses_to_clobber_without_force(fake_traces, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "dataset/train.jsonl"
    existing.parent.mkdir()
    existing.write_text("keep me\n")

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 1
    assert existing.read_text() == "keep me\n"

    assert (
        cli.main(
            ["traces", "export", "--project", "11111111-1111-4111-8111-111111111111", "--force"]
        )
        == 0
    )
    assert _rows(existing) == _RECORDS


def _no_terminal(monkeypatch) -> None:
    """A redirected stdin, the ordinary scripted case."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: False))


def test_traces_export_without_project_is_actionable_when_scripted(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    """Non-interactive and ambiguous: name the projects instead of hanging on a prompt."""
    monkeypatch.chdir(tmp_path)
    _no_terminal(monkeypatch)

    assert cli.main(["traces", "export"]) == 1
    err = capsys.readouterr().err
    assert "--project" in err
    assert "11111111-1111-4111-8111-111111111111" in err
    assert "22222222-2222-4222-8222-222222222222" in err


def test_traces_export_requires_the_only_project_explicitly(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(traces, "list_trace_projects", lambda *a, **k: [_PROJECTS[0]])
    _no_terminal(monkeypatch)

    assert cli.main(["traces", "export"]) == 1

    assert "--project" in capsys.readouterr().err
    assert fake_traces["records"] == []


def test_traces_export_under_ci_never_prompts(fake_traces, monkeypatch, tmp_path, capsys) -> None:
    # CI runners hand out a pseudo-tty, so isatty() is True in exactly the case that must not
    # prompt -- and `traces export` has no --yes to escape a prompt with, so a picker here hangs
    # the job until the runner's timeout kills it. input() raises so prompting fails loudly
    # instead of blocking this test forever.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))

    def _boom(*a, **k):
        raise AssertionError("prompted under CI")

    monkeypatch.setattr("builtins.input", _boom)

    assert cli.main(["traces", "export"]) == 1
    assert "--project" in capsys.readouterr().err
    assert fake_traces["records"] == []


def test_traces_export_requires_login(monkeypatch, tmp_path, capsys) -> None:
    """Logged out against the HOSTED plane, where a login is what's missing. On a self-hosted
    plane the command is unavailable outright, so pointing this at one would assert the wrong
    refusal (see test_traces_export_refuses_on_a_self_hosted_plane)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(traces, "load_credentials", lambda: (_HOSTED_URL, None))

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 1
    assert f"{traces.CLI_NAME} login" in capsys.readouterr().err


def test_traces_export_sends_the_key_from_the_url_it_decided_on(monkeypatch, tmp_path) -> None:
    """The self-hosted guard reads the url, and the key travels to the backend. Both must come
    from ONE credential read: if the key is re-read afterwards, a `flash login` landing in that
    window pairs a hosted-url decision with a newly stored self-hosted plane credential, sending
    that credential to the hosted backend -- the exact leak the guard exists to prevent.
    """
    monkeypatch.chdir(tmp_path)
    reads = iter(
        [
            (_HOSTED_URL, "hosted-key"),  # the read the guard decides on
            ("http://my-plane:8080", "operator-key"),  # a concurrent login lands here
        ]
    )
    monkeypatch.setattr(traces, "load_credentials", lambda: next(reads))

    sent: list[str] = []

    def _record(project_id, api_key, export_format=None):
        sent.append(api_key)
        return {"format": "records", "records": [{"input": "a", "output": "b"}]}

    monkeypatch.setattr(traces, "export_trace_records", _record)

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 0
    # the plane credential from the second read must never reach the hosted backend
    assert sent == ["hosted-key"]


def test_a_logged_out_snapshot_does_not_reread_the_config(monkeypatch, tmp_path, capsys) -> None:
    """A snapshot that legitimately held no key must refuse, not fall back to a fresh read. The
    logged-out caller is exactly the one whose re-read could pick up a credential stored by a
    concurrent `flash login` and pair it with the url this command already decided on.
    """
    monkeypatch.chdir(tmp_path)
    reads = iter(
        [
            (_HOSTED_URL, None),  # the snapshot the guard decides on: logged out
            (_HOSTED_URL, "key-stored-by-a-concurrent-login"),  # must never be reached
        ]
    )
    monkeypatch.setattr(traces, "load_credentials", lambda: next(reads))
    monkeypatch.setattr(
        traces, "export_trace_records", lambda *a, **k: pytest.fail("re-read the config")
    )

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 1
    assert f"{traces.CLI_NAME} login" in capsys.readouterr().err


def test_traces_export_reports_a_project_with_no_usable_traces(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        traces, "export_trace_records", lambda *a, **k: {"records": [], "traces": 0, "skipped": 0}
    )

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 1
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
        return options[0][0] if title.startswith("Use this project") else options[default][0]

    monkeypatch.setattr(env_setup.render, "select", fake_select)

    assert cli.main(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    assert asked[0] == "Use this project's recorded traces as starter rows?"
    assert _rows(tmp_path / "dataset/train.jsonl") == _RECORDS
    assert fake_traces["records"] == ["11111111-1111-4111-8111-111111111111"]
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

    assert cli.main(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

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
def test_env_setup_falls_back_to_starter_rows_when_selected_project_has_no_traces(
    fake_traces, monkeypatch, tmp_path, capsys, failure
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda args: True)
    monkeypatch.setattr(
        traces, "export_trace_records", lambda *a, **k: (_ for _ in ()).throw(failure)
    )
    monkeypatch.setattr(
        env_setup.render,
        "select",
        lambda title, options, default=0: (
            options[0][0] if title.startswith("Use this project") else options[default][0]
        ),
    )

    assert cli.main(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0
    assert '"input":"What is 2 + 2?"' in (tmp_path / "dataset/train.jsonl").read_text()
    assert "using the starter dataset" in capsys.readouterr().err


def test_env_setup_does_not_fallback_when_project_listing_fails(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda args: True)
    # a HOSTED url, because listing is the hosted path: against a self-hosted plane there is no
    # project directory to enumerate, so setup refuses before it would ever call list_projects.
    monkeypatch.setattr(
        "flash.client.config.load_credentials", lambda: ("https://flash.freesolo.co", "fs-key")
    )
    monkeypatch.setattr(
        "flash.client.list_projects",
        lambda api_key: (_ for _ in ()).throw(ClientError("project service offline")),
    )

    assert cli.main(["env", "setup"]) == 1
    assert "project service offline" in capsys.readouterr().err
    assert not (tmp_path / "environment.py").exists()


def test_env_setup_requires_login_for_project_validation(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("flash.client.config.load_credentials", lambda: ("https://flash", None))

    assert cli.main(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 1
    assert "not logged in" in capsys.readouterr().err
    assert not (tmp_path / "environment.py").exists()


def test_env_setup_keeps_an_existing_dataset(fake_traces, monkeypatch, tmp_path) -> None:
    """An existing dataset is never overwritten, so we don't download rows we'd discard."""
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset/train.jsonl"
    dataset.parent.mkdir()
    dataset.write_text('{"input":"mine","output":"kept"}\n')
    monkeypatch.setattr(env_setup, "_setup_interactive", lambda args: True)
    monkeypatch.setattr(env_setup.render, "select", _reject_traces_prompt("existing dataset"))

    assert cli.main(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    assert dataset.read_text() == '{"input":"mine","output":"kept"}\n'
    assert fake_traces["records"] == []


def test_env_setup_never_prompts_for_traces_under_yes(fake_traces, monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        traces,
        "list_trace_projects",
        lambda *a, **k: pytest.fail("--yes must not reach the backend"),
    )

    assert (
        cli.main(["env", "setup", "--project", "11111111-1111-4111-8111-111111111111", "--yes"])
        == 0
    )

    assert '"input":"What is 2 + 2?"' in (tmp_path / "dataset/train.jsonl").read_text()


def test_records_to_jsonl_round_trips() -> None:
    parsed = [json.loads(line) for line in traces.records_to_jsonl(_RECORDS).splitlines()]
    assert parsed == _RECORDS
    assert traces.records_to_jsonl([]) == ""


def test_project_options_label_by_name_and_skip_id_less_rows() -> None:
    options = traces.project_options([*_PROJECTS, {"name": "no id"}])
    assert [(value, label, hint) for value, label, hint in options] == [
        (
            "11111111-1111-4111-8111-111111111111",
            "support triage",
            "11111111-1111-4111-8111-111111111111",
        ),
        ("22222222-2222-4222-8222-222222222222", "docs qa", "22222222-2222-4222-8222-222222222222"),
    ]


def test_traces_export_defaults_to_env_records(fake_traces, monkeypatch, tmp_path) -> None:
    """No --format keeps the shape every existing caller already gets."""
    monkeypatch.chdir(tmp_path)

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    assert fake_traces["formats"] == [traces.RECORDS_FORMAT]


def test_records_export_hint_includes_required_project(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    assert (
        f"{traces.CLI_NAME} env push --project <project-uuid> --name my-env ."
        in capsys.readouterr().out
    )


def test_traces_export_prompts_shape_keeps_reply_less_calls(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    """GRPO samples its own completions, so a call with no reply is still a prompt."""
    monkeypatch.chdir(tmp_path)

    assert (
        cli.main(
            [
                "traces",
                "export",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "--format",
                "prompts",
            ]
        )
        == 0
    )

    rows = _rows(tmp_path / "dataset/train.jsonl")
    assert rows == _PROMPTS
    # every row is prompt-only: a gold completion here would be the wrong shape.
    assert all(set(row) == {"input"} for row in rows)
    assert fake_traces["formats"] == [traces.PROMPTS_FORMAT]
    printed = capsys.readouterr().out
    assert "exported 3 prompts to dataset/train.jsonl" in printed
    # nothing was skipped, so no skip note should appear.
    assert "skipped" not in printed


def test_traces_export_raw_shape_is_not_called_training_rows(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    """Raw rows are for taking away; calling them training rows would mislead."""
    monkeypatch.chdir(tmp_path)

    assert (
        cli.main(
            [
                "traces",
                "export",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "--format",
                "raw",
            ]
        )
        == 0
    )

    # raw defaults off the training dataset path; see the dedicated test below.
    assert _rows(tmp_path / "traces.raw.jsonl") == _RAW
    assert fake_traces["formats"] == [traces.RAW_FORMAT]
    printed = capsys.readouterr().out
    assert "exported 1 traces to traces.raw.jsonl" in printed
    assert "training rows" not in printed


def test_traces_export_rejects_an_algorithm_as_a_format(fake_traces, monkeypatch, tmp_path) -> None:
    """SFT and GRPO read the same records file, so they are not formats."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "traces",
                "export",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "--format",
                "sft",
            ]
        )

    # argparse rejects an out-of-choices value before any request is made.
    assert exit_info.value.code == 2
    assert fake_traces["formats"] == []


def test_traces_export_empty_records_points_at_the_prompts_shape(
    fake_traces, monkeypatch, tmp_path, capsys
) -> None:
    """A project whose calls have no replies is still exportable as prompts."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        traces,
        "export_trace_records",
        lambda *a, **k: {"records": [], "traces": 2, "skipped": 2, "format": "records"},
    )

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 1

    assert "--format prompts" in capsys.readouterr().err


def test_traces_export_skip_note_names_the_right_missing_half(
    monkeypatch, tmp_path, capsys
) -> None:
    """A prompts skip means no usable request; blaming a missing response would
    contradict the reason the prompts shape exists."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(traces, "load_credentials", lambda: (_HOSTED_URL, "fs-key"))
    monkeypatch.setattr(
        traces,
        "export_trace_records",
        lambda *a, **k: {
            "records": [{"input": "hi"}],
            "traces": 2,
            "skipped": 1,
            "format": "prompts",
        },
    )

    assert (
        cli.main(
            [
                "traces",
                "export",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "--format",
                "prompts",
            ]
        )
        == 0
    )

    printed = capsys.readouterr().out
    assert "1 traces skipped: no usable request)" in printed
    # the records-shaped reason must not leak into a prompts export.
    assert "request/response pair" not in printed


def test_traces_export_refuses_a_format_the_backend_ignored(monkeypatch, tmp_path, capsys) -> None:
    """A backend predating format support returns records for every request.

    Writing those rows out as `raw` would label {input, output} pairs as a
    complete trace dump -- an incomplete backup reported as success.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(traces, "load_credentials", lambda: (_HOSTED_URL, "fs-key"))
    monkeypatch.setattr(
        traces,
        "export_trace_records",
        # no "format" key at all, as an older backend would answer.
        lambda *a, **k: {"records": _RECORDS, "traces": 2, "skipped": 0},
    )

    assert (
        cli.main(
            [
                "traces",
                "export",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "--format",
                "raw",
            ]
        )
        == 1
    )

    assert "did not honour --format raw" in capsys.readouterr().err
    # nothing may be written when the shape could not be trusted.
    assert not (tmp_path / "traces.raw.jsonl").exists()


def test_traces_export_default_records_works_against_a_format_blind_backend(
    monkeypatch, tmp_path
) -> None:
    """The default request cannot be mislabelled: records is what such a backend returns."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(traces, "load_credentials", lambda: (_HOSTED_URL, "fs-key"))
    monkeypatch.setattr(
        traces,
        "export_trace_records",
        lambda *a, **k: {"records": _RECORDS, "traces": 2, "skipped": 0},
    )

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 0

    assert _rows(tmp_path / "dataset/train.jsonl") == _RECORDS


def test_traces_export_refuses_an_explicit_mismatch_on_the_default_format(
    monkeypatch, tmp_path, capsys
) -> None:
    """Regression: the default request skipped the check entirely, so a backend that explicitly
    answered `raw` had its trace rows written to dataset/train.jsonl as if they were environment
    records. That path is auto-selected as a run's dataset_path, so a later `env push` + `train`
    would feed spans to a run that cannot read them.

    A MISSING format still means records -- that is what a format-blind backend returns, and the
    test above pins it -- but an explicit label is the backend stating what it actually converted,
    and any value other than the one asked for must be refused whichever format was requested.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(traces, "load_credentials", lambda: (_HOSTED_URL, "fs-key"))
    monkeypatch.setattr(
        traces,
        "export_trace_records",
        lambda *a, **k: {"records": _RAW, "traces": 2, "skipped": 0, "format": "raw"},
    )

    assert cli.main(["traces", "export", "--project", "11111111-1111-4111-8111-111111111111"]) == 1

    assert "did not honour --format records" in capsys.readouterr().err
    # nothing may be written when the shape could not be trusted
    assert not (tmp_path / "dataset/train.jsonl").exists()


def test_raw_export_stays_off_the_training_dataset_path(fake_traces, monkeypatch, tmp_path) -> None:
    """dataset/train.jsonl is auto-selected as a run's dataset_path, and raw rows
    are not trainable, so raw must default somewhere else."""
    monkeypatch.chdir(tmp_path)

    assert (
        cli.main(
            [
                "traces",
                "export",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "--format",
                "raw",
            ]
        )
        == 0
    )

    assert _rows(tmp_path / "traces.raw.jsonl") == _RAW
    assert not (tmp_path / "dataset/train.jsonl").exists()


def test_prompts_export_keeps_the_training_dataset_path(fake_traces, monkeypatch, tmp_path) -> None:
    """Prompts are training input for GRPO, so they belong on the default path."""
    monkeypatch.chdir(tmp_path)

    assert (
        cli.main(
            [
                "traces",
                "export",
                "--project",
                "11111111-1111-4111-8111-111111111111",
                "--format",
                "prompts",
            ]
        )
        == 0
    )

    assert _rows(tmp_path / "dataset/train.jsonl") == _PROMPTS
