"""Hermetic branch coverage for CLI rendering and terminal-state helpers."""

from __future__ import annotations

from types import SimpleNamespace

import flash.cli.commands as commands


class _Client:
    def __init__(self, *, checkpoints=None, deployments=None, run=None, worker_output=None) -> None:
        self._checkpoints = checkpoints or []
        self._deployments = deployments or []
        self._run = run or {"run_id": "flash-1", "state": "done"}
        self._worker_output = worker_output or {}

    def checkpoints(self, run_id: str):
        assert run_id == "flash-1"
        return self._checkpoints

    def deployments(self):
        return self._deployments

    def get_run(self, run_id: str):
        assert run_id == "flash-1"
        return self._run

    def get_worker_output(self, run_id: str):
        assert run_id == "flash-1"
        return self._worker_output


def test_env_list_filters_non_environments_and_uses_styled_renderer(
    tmp_path, monkeypatch, capsys
) -> None:
    """Environment listing must include only loadable sources before handing them to the renderer."""
    (tmp_path / "environment.py").write_text("# root\n")
    environments = tmp_path / "environments"
    environments.mkdir()
    (environments / "standalone.py").write_text("# standalone\n")
    (environments / "ignore.txt").write_text("not python\n")
    (environments / "__private.py").write_text("# private\n")
    canonical = environments / "canonical-name"
    canonical.mkdir()
    (canonical / "environment.py").write_text("# canonical\n")
    module = environments / "module-name"
    module.mkdir()
    (module / "module_name.py").write_text("# module\n")
    (environments / "empty").mkdir()
    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "env_list", lambda paths: seen.extend(paths) or "styled-envs")

    assert commands.cmd_env_list(SimpleNamespace()) == 0

    assert capsys.readouterr().out == "styled-envs\n"
    assert seen == [
        ".",
        "environments/canonical-name",
        "environments/module-name",
        "environments/standalone.py",
    ]


def test_follow_run_returns_success_and_failure_codes(monkeypatch, capsys) -> None:
    """Following a run must print its final state and preserve terminal success semantics."""
    client = _Client(run={"run_id": "flash-1", "state": "done"})
    monkeypatch.setattr(commands, "_render_status", lambda status: f"final:{status['state']}")
    monkeypatch.setattr(commands, "_poll_logs", lambda *args, **kwargs: ("done", False))
    assert commands._follow_run(client, "flash-1") == 0
    assert capsys.readouterr().out == "final:done\n"

    monkeypatch.setattr(commands, "_poll_logs", lambda *args, **kwargs: ("failed", False))
    assert commands._follow_run(client, "flash-1") == 1
    assert capsys.readouterr().out == "final:done\n"


def test_print_worker_output_skips_empty_sections_and_styles_nonempty_output(
    monkeypatch, capsys
) -> None:
    """Worker output must omit empty artifacts while retaining a styled section heading."""
    client = _Client(worker_output={"empty.txt": "", "console.txt": "worker line"})
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "log_section", lambda name: f"section:{name}")

    assert commands._print_worker_output(client, "flash-1") is True

    assert capsys.readouterr().out == "section:console.txt\nworker line\n"


def test_log_follow_progress_preserves_nonnumeric_realized_cost() -> None:
    """Malformed realized cost values must remain visible instead of crashing log following."""
    state, progress = commands._log_follow_progress(
        {"state": "running", "realized_cost_usd": "pending"}, "unknown"
    )

    assert state == "running"
    assert progress == "running realized_cost=pending"


def test_checkpoints_empty_and_styled_paths(monkeypatch, capsys) -> None:
    """Checkpoint listing must render both empty and populated styled states without plain leakage."""
    client = _Client()
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "empty", lambda *args: "empty-checkpoints")
    assert commands.cmd_checkpoints(SimpleNamespace(run_id="flash-1")) == 0
    assert capsys.readouterr().out == "empty-checkpoints\n"

    client._checkpoints = [{"step": 12}]
    monkeypatch.setattr(commands.render, "checkpoints_table", lambda run_id, rows: "checkpoint-table")
    assert commands.cmd_checkpoints(SimpleNamespace(run_id="flash-1")) == 0
    assert capsys.readouterr().out == "checkpoint-table\n"


def test_checkpoints_plain_path_prints_canonical_refs(monkeypatch, capsys) -> None:
    """Plain checkpoint output must stay grep-friendly and include the deployment hint on stderr."""
    client = _Client(checkpoints=[{"step": 12}, {"step": 24}])
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    assert commands.cmd_checkpoints(SimpleNamespace(run_id="flash-1")) == 0

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "step 12 flash-1/step-12",
        "step 24 flash-1/step-24",
    ]
    assert "deploy one with" in captured.err


def test_deployments_empty_and_styled_paths(monkeypatch, capsys) -> None:
    """Deployment listing must distinguish empty and populated styled states."""
    client = _Client()
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "empty", lambda *args: "empty-deployments")
    assert commands.cmd_deployments(SimpleNamespace(json=False)) == 0
    assert capsys.readouterr().out == "empty-deployments\n"

    client._deployments = [{"run_id": "flash-1", "deployment": {"state": "ready"}}]
    monkeypatch.setattr(commands.render, "deployments_table", lambda rows: "deployment-table")
    assert commands.cmd_deployments(SimpleNamespace(json=False)) == 0
    assert capsys.readouterr().out == "deployment-table\n"


def test_deployments_plain_path_handles_final_and_detailed_rows(monkeypatch, capsys) -> None:
    """Plain deployment output must expose immutable revision, final step, model, and failure detail."""
    rows = [
        {
            "run_id": "fallback-run",
            "deployment": {
                "run_id": "flash-1",
                "checkpoint_step": None,
                "adapter_revision": "a" * 40,
                "state": "failed",
                "verified_at": 123,
                "openai_model": "model-1",
                "error": "broken",
            },
        }
    ]
    monkeypatch.setattr(commands, "client_from_config", lambda: _Client(deployments=rows))
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    assert commands.cmd_deployments(SimpleNamespace(json=False)) == 0

    output = capsys.readouterr().out
    assert "flash-1" in output
    assert "final" in output
    assert "model-1" in output
    assert "broken" in output


def test_chat_rejects_invalid_target_without_constructing_a_client(monkeypatch, capsys) -> None:
    """Invalid chat targets must fail locally before any API client or network boundary is touched."""
    monkeypatch.setattr(
        commands,
        "client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )

    result = commands.cmd_chat(SimpleNamespace(run_id="bad target", message="hello", system=None))

    assert result == 1
    assert "invalid chat target" in capsys.readouterr().err


def test_chat_prints_styled_label_before_streaming(monkeypatch, capsys) -> None:
    """Styled chat must label assistant output while preserving streamed content."""
    class ChatClient:
        def chat_stream(self, run_id, messages, **kwargs):
            assert run_id == "flash-1"
            assert messages == [{"role": "user", "content": "hello"}]
            yield "hi"

    monkeypatch.setattr(commands, "client_from_config", ChatClient)
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "chat_label", lambda: "assistant-label")

    result = commands.cmd_chat(
        SimpleNamespace(
            run_id="flash-1",
            message="hello",
            system=None,
            temperature=0.0,
            max_tokens=32,
        )
    )

    assert result == 0
    assert capsys.readouterr().out.startswith("assistant-label\nhi")
