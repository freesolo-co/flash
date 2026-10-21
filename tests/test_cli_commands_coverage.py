"""Hermetic branch coverage for CLI rendering and terminal-state helpers."""

from __future__ import annotations

from types import SimpleNamespace

import flash.cli.commands.env.ops.list as env_list_commands
from flash.cli.commands.ops import deploy as deploy_commands
from flash.cli.commands.ops import runs as commands


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
    monkeypatch.setattr(env_list_commands.render, "styled", lambda: True)
    monkeypatch.setattr(
        env_list_commands.env_panels,
        "env_list",
        lambda paths, *, published, unavailable: seen.extend(paths) or "styled-envs",
    )
    monkeypatch.setattr(env_list_commands, "_published_envs", lambda: ([], None))

    assert env_list_commands.cmd_env_list(SimpleNamespace()) == 0

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
    monkeypatch.setattr(
        commands,
        "_poll_logs",
        lambda *args, **kwargs: commands._LogPollResult("done", False, None),
    )
    assert commands._follow_run(client, "flash-1") == 0
    assert capsys.readouterr().out == "final:done\n"

    monkeypatch.setattr(
        commands,
        "_poll_logs",
        lambda *args, **kwargs: commands._LogPollResult("failed", False, None),
    )
    assert commands._follow_run(client, "flash-1") == 1
    assert capsys.readouterr().out == "final:done\n"


def test_print_worker_output_skips_empty_sections_and_styles_nonempty_output(
    monkeypatch, capsys
) -> None:
    """Worker output must omit empty artifacts while retaining a styled section heading."""
    client = _Client(worker_output={"empty.txt": "", "console.txt": "worker line"})
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "log_section", lambda name: f"section:{name}")

    assert commands._print_worker_output(commands._worker_sections(client, "flash-1")) is True

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
    monkeypatch.setattr(deploy_commands, "client_from_config", lambda: client)
    monkeypatch.setattr(deploy_commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "empty", lambda *args: "empty-checkpoints")
    assert commands.cmd_checkpoints(SimpleNamespace(run_id="flash-1")) == 0
    assert capsys.readouterr().out == "empty-checkpoints\n"

    client._checkpoints = [{"step": 12}]
    monkeypatch.setattr(
        commands.tables, "checkpoints_table", lambda run_id, rows: "checkpoint-table"
    )
    assert commands.cmd_checkpoints(SimpleNamespace(run_id="flash-1")) == 0
    assert capsys.readouterr().out == "checkpoint-table\n"


def test_checkpoints_plain_path_prints_canonical_refs(monkeypatch, capsys) -> None:
    """Plain checkpoint output must stay grep-friendly and include the deployment hint on stderr."""
    client = _Client(checkpoints=[{"step": 12}, {"step": 24}])
    monkeypatch.setattr(commands, "client_from_config", lambda: client)
    monkeypatch.setattr(deploy_commands, "client_from_config", lambda: client)
    monkeypatch.setattr(deploy_commands, "client_from_config", lambda: client)
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
    monkeypatch.setattr(deploy_commands, "client_from_config", lambda: client)
    monkeypatch.setattr(deploy_commands, "client_from_config", lambda: client)
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "empty", lambda *args: "empty-deployments")
    assert deploy_commands.cmd_deployments(SimpleNamespace(json=False)) == 0
    assert capsys.readouterr().out == "empty-deployments\n"

    client._deployments = [{"run_id": "flash-1", "deployment": {"state": "ready"}}]
    monkeypatch.setattr(commands.tables, "deployments_table", lambda rows: "deployment-table")
    assert deploy_commands.cmd_deployments(SimpleNamespace(json=False)) == 0
    assert capsys.readouterr().out == "deployment-table\n"


def test_deployments_plain_path_handles_final_and_detailed_rows(monkeypatch, capsys) -> None:
    """Plain deployment output must expose checkpoint identity, final step, and failure detail."""
    rows = [
        {
            "run_id": "fallback-run",
            "deployment": {
                "run_id": "flash-1",
                "checkpoint_step": None,
                "checkpoint_id": "flash-1/final",
                "state": "failed",
                "verified_at": 123,
                "openai_model": "model-1",
                "error": "broken",
            },
        }
    ]
    monkeypatch.setattr(commands, "client_from_config", lambda: _Client(deployments=rows))
    monkeypatch.setattr(deploy_commands, "client_from_config", lambda: _Client(deployments=rows))
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    assert deploy_commands.cmd_deployments(SimpleNamespace(json=False)) == 0

    output = capsys.readouterr().out
    assert "flash-1" in output
    assert "final" in output
    assert "model-1" in output
    assert "broken" in output
    # the verified_at epoch is rendered as an explicit UTC timestamp, not a raw number
    assert "1970-01-01 00:02 UTC" in output
    assert "123" not in output


def test_chat_rejects_invalid_target_without_constructing_a_client(monkeypatch, capsys) -> None:
    """Invalid chat targets must fail locally before any API client or network boundary is touched."""
    monkeypatch.setattr(
        commands,
        "client_from_config",
        lambda: (_ for _ in ()).throw(AssertionError("client must not be constructed")),
    )

    result = deploy_commands.cmd_chat(
        SimpleNamespace(run_id="bad target", message="hello", system=None)
    )

    assert result == 1
    assert "invalid chat target" in capsys.readouterr().err


def test_chat_prints_styled_label_before_streaming(monkeypatch, capsys) -> None:
    """Styled chat must label assistant output while preserving streamed content."""

    class ChatClient:
        def chat_stream(self, run_id, messages, **kwargs):
            assert run_id == "flash-1/final"
            assert messages == [{"role": "user", "content": "hello"}]
            yield "hi"

    monkeypatch.setattr(commands, "client_from_config", ChatClient)
    monkeypatch.setattr(deploy_commands, "client_from_config", ChatClient)
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "chat_label", lambda: "assistant-label")

    result = deploy_commands.cmd_chat(
        SimpleNamespace(
            run_id="flash-1/final",
            message="hello",
            system=None,
            temperature=0.0,
            max_tokens=32,
        )
    )

    assert result == 0
    assert capsys.readouterr().out.startswith("assistant-label\nhi")


def test_chat_fails_when_the_stream_carries_no_text(monkeypatch, capsys) -> None:
    """An empty but successful chat stream must fail loudly instead of exiting 0 with no output.

    A serving path that stopped applying the run's chat template returns a well-formed response
    carrying no assistant text. Exiting 0 there makes that indistinguishable from a model that
    answered nothing, so the surface cannot be trusted as a health check.
    """

    class EmptyChatClient:
        def chat_stream(self, run_id, messages, **kwargs):
            return iter(())

    monkeypatch.setattr(commands, "client_from_config", EmptyChatClient)
    monkeypatch.setattr(deploy_commands, "client_from_config", EmptyChatClient)
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    result = deploy_commands.cmd_chat(
        SimpleNamespace(
            run_id="flash-1/final",
            message="hello",
            system=None,
            temperature=0.0,
            max_tokens=32,
        )
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "no response text from flash-1/final" in captured.err

    # `models list` enumerates supported base models and carries no deployment state, so it cannot
    # investigate either condition this message names. `models deployments` is the one that can.
    assert f"{deploy_commands.CLI_NAME} models deployments" in captured.err
    assert "models list" not in captured.err


def test_chat_treats_a_whitespace_only_stream_as_no_response(monkeypatch, capsys) -> None:
    """Blank chunks are the same empty answer as no chunks, and must fail the same way.

    Counting any chunk as text exited 0 here, so a serving path that stopped applying the run's
    chat template and emitted only whitespace read as a healthy response. Spaces are as
    non-empty as the `assistant` label the test below keeps off stdout, so they must not be
    printed either -- the contract is exit 1 with empty stdout.
    """

    class BlankChatClient:
        def chat_stream(self, run_id, messages, **kwargs):
            yield "  "
            yield "\n"
            yield "\t"

    monkeypatch.setattr(commands, "client_from_config", BlankChatClient)
    monkeypatch.setattr(deploy_commands, "client_from_config", BlankChatClient)
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "chat_label", lambda: "assistant-label")

    result = deploy_commands.cmd_chat(
        SimpleNamespace(
            run_id="flash-1/final",
            message="hello",
            system=None,
            temperature=0.0,
            max_tokens=32,
        )
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "no response text from flash-1/final" in captured.err


def test_chat_preserves_leading_whitespace_once_the_stream_has_text(monkeypatch, capsys) -> None:
    """Holding blank chunks back must not edit a response that turns out to have text.

    A model whose first chunks are whitespace still answered, so the held chunks are released
    verbatim and the output stays byte-identical to what the stream produced.
    """

    class LeadingBlankChatClient:
        def chat_stream(self, run_id, messages, **kwargs):
            yield "  "
            yield "hi"

    monkeypatch.setattr(commands, "client_from_config", LeadingBlankChatClient)
    monkeypatch.setattr(deploy_commands, "client_from_config", LeadingBlankChatClient)
    monkeypatch.setattr(commands.render, "styled", lambda: False)

    result = deploy_commands.cmd_chat(
        SimpleNamespace(
            run_id="flash-1/final",
            message="hello",
            system=None,
            temperature=0.0,
            max_tokens=32,
        )
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "  hi\n"


def test_chat_keeps_stdout_empty_when_a_styled_stream_carries_no_text(monkeypatch, capsys) -> None:
    """The empty-stream contract is "exit 1 with empty stdout", and styling must not break it.

    Styling turns on automatically on a tty and explicitly via FLASH_STYLE=1, so this is the
    default interactive path, not an edge case. Printing the `assistant` label before the first
    chunk leaves it on stdout when the stream turns out to be empty, which is precisely what a
    caller capturing output through a PTY would then have to treat as a response.
    """

    class EmptyChatClient:
        def chat_stream(self, run_id, messages, **kwargs):
            return iter(())

    monkeypatch.setattr(commands, "client_from_config", EmptyChatClient)
    monkeypatch.setattr(deploy_commands, "client_from_config", EmptyChatClient)
    monkeypatch.setattr(commands.render, "styled", lambda: True)
    monkeypatch.setattr(commands.render, "chat_label", lambda: "assistant-label")

    result = deploy_commands.cmd_chat(
        SimpleNamespace(
            run_id="flash-1/final",
            message="hello",
            system=None,
            temperature=0.0,
            max_tokens=32,
        )
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert "no response text from flash-1/final" in captured.err


def test_superseded_attempt_heartbeats_are_tagged_in_the_log_stream(capsys) -> None:
    """`runs log | grep HEARTBEAT | tail -1` must not silently return a dead attempt's heartbeat.

    The plane appends the highest UPLOADED attempt after the chronological log, so during a retry
    the last heartbeat on screen belongs to the attempt that just died. The section heading says
    so, but a heading does not survive a pipe: the reporter's monitor read
    ``step 0, 0 completions, device H200`` for twenty minutes while the run was live on B200,
    because H200 was the card that had already OOMed and been torn down.
    """
    commands._print_worker_output(
        {
            "console_rl_attempt0.txt": (
                'HEARTBEAT {"stage":"rl_step","step":76,"attempt":0,"device_name":"H200"}\n'
                "Traceback (most recent call last):\n"
            )
        },
        printed_any=True,
        current_attempt=1,
    )

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "HEARTBEAT" in line]
    assert lines, "the dead attempt's console dump must still be printed"
    # the stale line carries its own provenance, so it cannot be mistaken for the live attempt
    assert "[superseded attempt=0; current attempt=1]" in lines[-1]
    # and the failure dump itself is preserved -- it is why the retry exists
    assert "Traceback (most recent call last):" in out


def test_filtering_superseded_heartbeats_leaves_the_live_attempt_last(capsys) -> None:
    """A consumer that drops tagged lines gets the LIVE attempt's heartbeat, not the dead one."""
    print('HEARTBEAT {"stage":"rl_step","step":1,"attempt":1,"device_name":"B200"}')
    commands._print_worker_output(
        {
            "console_rl_attempt0.txt": (
                'HEARTBEAT {"stage":"rl_step","step":76,"attempt":0,"device_name":"H200"}\n'
            )
        },
        printed_any=True,
        current_attempt=1,
    )

    heartbeats = [line for line in capsys.readouterr().out.splitlines() if "HEARTBEAT" in line]
    live = [line for line in heartbeats if "[superseded" not in line]
    assert live[-1].endswith('{"stage":"rl_step","step":1,"attempt":1,"device_name":"B200"}')


def test_the_canonical_console_cannot_reach_the_pipe_untagged(capsys) -> None:
    """The unscoped ``console_<phase>.txt`` encodes no attempt, so it must not escape tagging.

    The plane fetches BOTH console names and appends the canonical one last, so it is the final
    section on screen. Its filename carries no ``_attemptN``, which is exactly what the tagging
    predicate keys on -- so a name-derived rule leaves it unmarked and
    ``runs log | grep HEARTBEAT | tail -1`` trusts it. On a retry whose terminal upload never ran
    (``os._exit(124)`` runs no ``finally`` blocks) that file belongs to an OLDER attempt, which is
    the precise failure ``_mark_superseded_heartbeats`` exists to prevent.

    Unknown provenance is not proof of liveness: when the live attempt is known and the artifact
    cannot name its own, it must be treated as superseded rather than silently trusted.
    """
    commands._print_worker_output(
        {
            "console_rl.txt": (
                'HEARTBEAT {"stage":"rl_step","step":76,"attempt":0,"device_name":"H200"}\n'
            )
        },
        printed_any=True,
        current_attempt=1,
    )

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "HEARTBEAT" in line]
    assert lines, "the canonical console must still be printed"
    assert "[superseded" in lines[-1], (
        f"the canonical console's heartbeat reached the pipe unmarked: {lines[-1]}"
    )


def test_current_attempt_heartbeats_are_left_untouched(capsys) -> None:
    """Only a SUPERSEDED attempt is tagged; the live attempt's own artifact must read normally."""
    commands._print_worker_output(
        {"console_rl_attempt1.txt": 'HEARTBEAT {"stage":"rl_step","step":4,"attempt":1}\n'},
        printed_any=True,
        current_attempt=1,
    )

    out = capsys.readouterr().out
    assert "[superseded" not in out


def test_teardown_window_still_tags_the_dead_workers_heartbeat(capsys) -> None:
    """An explicitly null ``remote`` is PROOF no worker is live, so every heartbeat is a dead one.

    ``live_attempt`` answers ``None`` both for "the worker is torn down" and for "no attempt is
    known", and collapsing the two disabled tagging exactly during the retry window the tagging
    exists for -- the dead attempt's heartbeat reached ``grep HEARTBEAT | tail -1`` unmarked while
    replacement capacity was still being acquired.
    """
    from flash.cli.commands.ops.worker_output import _NO_LIVE_WORKER

    commands._print_worker_output(
        {
            "console_rl_attempt0.txt": (
                'HEARTBEAT {"stage":"rl_step","step":76,"attempt":0,"device_name":"H200"}\n'
            )
        },
        printed_any=True,
        current_attempt=_NO_LIVE_WORKER,
    )

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "HEARTBEAT" in line]
    assert lines, "the dead attempt's console dump must still be printed"
    assert "[superseded attempt=0; worker torn down]" in lines[-1], (
        f"a torn-down worker's heartbeat reached the pipe unmarked: {lines[-1]}"
    )
    assert "worker torn down; no live attempt" in out, "the heading must say so too"


def test_follow_also_tags_a_teardown_it_ends_inside(capsys) -> None:
    """``--follow`` must derive the live attempt by the same rule the one-shot path uses.

    The two log paths reach ``_print_worker_output`` differently: the one-shot path snapshots the
    status through ``_snapshot_live_attempt``, while ``--follow`` carries the attempt it saw while
    streaming. When that stream ends mid-teardown, the raw ``live_attempt`` is ``None`` -- so the
    follow printed the dead worker's heartbeats unmarked, which is the exact failure the tagging
    exists to prevent, reachable through the more common of the two commands.
    """
    terminal = {
        "state": "failed",
        "remote": None,
        "last_heartbeat": {"attempt": 0},
    }

    class _Client:
        def get_logs(self, _run_id, offset=0):
            return {"logs": "", "offset": 0, "state": "failed"}

        def get_run(self, _run_id):
            return terminal

        def get_worker_output(self, _run_id):
            return {
                "console_rl_attempt0.txt": (
                    'HEARTBEAT {"stage":"rl_step","step":76,"attempt":0,"device_name":"H200"}\n'
                )
            }

    result = commands._poll_logs(_Client(), "flash-1", interval=0.0)
    assert result.live_attempt == commands.live_attempt_of(terminal)

    commands._print_worker_output(
        _Client().get_worker_output("flash-1"),
        printed_any=result.printed_any,
        current_attempt=result.live_attempt,
    )
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "HEARTBEAT" in line]
    assert lines, "the dead attempt's console dump must still be printed"
    assert "[superseded attempt=0; worker torn down]" in lines[-1], (
        f"a follow that ended mid-teardown leaked an unmarked dead heartbeat: {lines[-1]}"
    )


def test_snapshot_distinguishes_teardown_from_an_unknown_attempt() -> None:
    """``remote: null`` means torn down; a failed lookup means unknown. They are not the same."""
    from flash.cli.commands.ops.worker_output import _NO_LIVE_WORKER, _snapshot_live_attempt
    from flash.client import ClientError

    class _Run:
        def __init__(self, payload):
            self._payload = payload

        def get_run(self, _run_id):
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    torn_down = {"state": "running", "remote": None, "last_heartbeat": {"attempt": 0}}
    assert _snapshot_live_attempt(_Run(torn_down), "flash-1") == _NO_LIVE_WORKER
    live = {"state": "running", "remote": {"attempt": 1}, "last_heartbeat": {"attempt": 0}}
    assert _snapshot_live_attempt(_Run(live), "flash-1") == 1
    # a lookup failure must NOT claim there is no live worker
    assert _snapshot_live_attempt(_Run(ClientError("boom")), "flash-1") is None
    # nor must a payload that simply carries no remote key
    no_remote = {"state": "running", "last_heartbeat": {"attempt": 0}}
    assert _snapshot_live_attempt(_Run(no_remote), "flash-1") == 0


def test_heartbeat_tagging_is_idempotent() -> None:
    """Re-printing an already-tagged dump must not stack a second marker onto the line."""
    from flash.cli.commands.ops.worker_output import _mark_superseded_heartbeats

    once = _mark_superseded_heartbeats('HEARTBEAT {"step":1}\n', 0, 1)
    twice = _mark_superseded_heartbeats(once, 0, 1)
    assert once == twice
    assert once.count("[superseded") == 1
