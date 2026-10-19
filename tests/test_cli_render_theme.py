"""The themed CLI output is TTY-gated: piped/scripted output stays byte-for-byte machine
readable, while ``FLASH_STYLE=1`` forces the human theme. These tests pin both sides of that
gate so a future tweak to the theme can't silently change the machine contract that `jq`,
scripts, and the trainer agent depend on.
"""

from __future__ import annotations

import argparse
import json

import pytest

import flash.cli.commands.ops.deploy as cli_deploy
import flash.cli.commands.ops.runs as cli_runs
import flash.cli.parsing.main as cli
from flash import __version__
from flash.cli.ui import env_panels, render, tables
from flash.cli.ui import heartbeat as heartbeat_ui


class _Client:
    def get_run(self, run_id: str) -> dict:
        return {
            "run_id": run_id,
            "state": "done",
            "cost_usd": 0.25,
            "error": None,
            "spec": {
                "project": "11111111-1111-4111-8111-111111111111",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
            },
        }


@pytest.fixture
def fake_client(monkeypatch) -> None:
    monkeypatch.setattr(cli_runs, "client_from_config", lambda *a, **k: _Client())


def test_styled_flag_overrides_tty(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    assert render.styled() is True
    monkeypatch.setenv("FLASH_STYLE", "0")
    assert render.styled() is False


def test_plain_path_stays_machine_readable(monkeypatch, fake_client, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "0")

    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"flash {__version__}"

    assert cli.main(["models", "list"]) == 0
    out = capsys.readouterr().out
    # plain output is bare model ids, one per line, with no themed header or detail columns
    assert "Qwen/Qwen3.5-9B" in out
    assert "supported base models" not in out
    assert "Qwen/Qwen3.5-9B" in out
    assert "\t" not in out

    assert cli.main(["runs", "status", "flash-1"]) == 0
    payload = json.loads(capsys.readouterr().out)  # must stay parseable JSON
    assert payload["state"] == "done"
    assert payload["cost_usd"] == 0.25


def test_styled_path_is_themed_but_lossless(monkeypatch, fake_client, capsys) -> None:
    # Themed layout on, color off (NO_COLOR), so highlighted-JSON tokens stay contiguous and we
    # can assert the full payload survives the themed view.
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    assert cli.main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "supported base models" in out  # themed header
    assert "Qwen/Qwen3.5-9B" in out
    assert "PARAMS" not in out  # ids only — no reintroduced detail columns

    assert cli.main(["runs", "status", "flash-1"]) == 0
    out = capsys.readouterr().out
    # curated panel up top, but the full JSON is still emitted below (no data dropped)
    assert "details" in out
    assert "cost_usd" in out
    assert '"state": "done"' in out

    assert cli.main(["runs", "status", "flash-1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "done"


def test_runs_and_status_hide_provider_names(monkeypatch) -> None:
    # Provider metadata may exist in API payloads for lifecycle/accounting, but human CLI summaries
    # show only the managed GPU class and redact backend fields from styled details.
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    run = {
        "run_id": "flash-1",
        "state": "done",
        "spec": {
            "project": "11111111-1111-4111-8111-111111111111",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
        },
        "remote": {"allocated_gpu": "RTX 4090", "provider": "runpod", "flash_arm": "runpod"},
    }
    runs = tables.runs_table([run])
    status = render.run_status(run)
    assert "RTX 4090" in runs
    assert "RTX 4090" in status
    assert "runpod" not in runs.lower()
    assert "runpod" not in status.lower()
    assert "provider" not in status.lower()
    assert "flash_arm" not in status.lower()


def test_runs_and_status_prefer_allocated_gpu(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    allocated_run = {
        "run_id": "flash-allocated",
        "state": "done",
        "spec": {
            "project": "11111111-1111-4111-8111-111111111111",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "gpu": {"type": "RTX Pro 6000"},
        },
        "remote": {"allocated_gpu": "B200"},
    }
    runs = tables.runs_table([allocated_run])
    status = render.run_status(allocated_run)
    status_panel = status.split("details", 1)[0]
    assert "B200" in runs
    assert "B200" in status_panel
    assert "RTX Pro 6000" not in runs
    assert "RTX Pro 6000" not in status_panel

    spec_only_run = {
        "run_id": "flash-spec-only",
        "state": "done",
        "spec": {"gpu": {"type": "RTX Pro 6000"}},
    }
    assert "RTX Pro 6000" in tables.runs_table([spec_only_run])
    assert "RTX Pro 6000" in render.run_status(spec_only_run)

    # the removed legacy remote["gpu"] key is no longer honored: the label resolves from the
    # authoritative allocated_gpu or the provisional spec type, never the dead key.
    legacy_key_run = {
        "run_id": "flash-legacy-key",
        "state": "done",
        "spec": {"gpu": {"type": "RTX Pro 6000"}},
        "remote": {"gpu": "B200"},
    }
    assert "B200" not in tables.runs_table([legacy_key_run])
    assert "RTX Pro 6000" in tables.runs_table([legacy_key_run])


def test_color_respects_no_color(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert render._color() is True
    assert "\x1b[" in render.badge("done")

    monkeypatch.setenv("NO_COLOR", "1")
    assert render._color() is False
    assert "\x1b[" not in render.badge("done")  # layout kept, color dropped


def test_render_json_plain_matches_json_dumps(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "0")
    obj = {"run_id": "flash-1", "state": "done", "cost_usd": 0.25, "nested": {"a": [1, 2]}}
    assert render._json(obj) == json.dumps(obj, indent=2)


def test_train_submitted_note(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = render.submitted("flash-xyz")
    assert "flash-xyz" in out
    assert "submitted" in out
    assert "--follow" in out  # tells the user how to re-attach after Ctrl-C


def test_train_submitted_note_names_this_channel_executable(monkeypatch) -> None:
    """The hand-off must name the executable the user actually has installed.

    On the dev channel the console script is `flash-dev` and CLI_NAME is rewritten to match, so a
    hardcoded `flash runs cancel ...` hands out a command that does not exist on that install --
    and the command it hands out is the one that stops the billing.
    """
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    # stand in for the dev channel. asserting against the real CLI_NAME would pass on the release
    # channel even if the name were hardcoded, since there it is literally "flash".
    monkeypatch.setattr(render, "CLI_NAME", "flash-dev")

    out = render.submitted("flash-xyz")

    assert "`flash-dev runs log flash-xyz --follow`" in out
    assert "`flash-dev runs cancel flash-xyz`" in out


def test_styled_renderers_are_ascii_locale_safe(monkeypatch) -> None:
    # On an ASCII / non-UTF-8 stdout, the themed renderers (which use em dashes, bullets, etc.)
    # must degrade rather than raise UnicodeEncodeError once styling is forced on.
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    class _AsciiStdout:
        encoding = "ascii"

        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(render.sys, "stdout", _AsciiStdout())
    outputs = [
        tables.models_table([{"id": "acme/x"}]),
        tables.gpus_table([("RTX 5090", 32, 0.99)], "Tip: selection is automatic — no pinning"),
        tables.runs_table([{"run_id": "r", "state": "done", "spec": {}}]),
        tables.deployments_table([{"run_id": "r", "deployment": {"state": "ready"}}]),
        render.env_setup(
            ["environment.py", "dataset/train.jsonl", "configs/rl.toml"],
            "11111111-1111-4111-8111-111111111111",
        ),
        env_panels.env_list([]),
        render.empty("runs", "0 runs", "no runs yet — submit one with `flash train`"),
        render.submitted("flash-xyz"),
        render.run_status({"run_id": "r", "state": "failed", "spec": {}, "error": "boom — bad"}),
        tables.checkpoints_table(
            "r", [{"step": 8, "repo_id": "acme/x", "subfolder": "grpo/step-8"}]
        ),
        render.cancelled({"run_id": "r", "state": "cancelled"}),
        render.deployed(
            {"run_id": "r", "state": "deployed", "endpoint_name": "ep", "openai_base_url": "u"}
        ),
        render.undeployed({"run_id": "r", "deleted_endpoints": ["ep"]}),
        render.exported(
            {"checkpoint_id": "r/final", "repository": "acme/x", "url": "u", "private": True}
        ),
        render.error("config invalid — bad [environment] id"),
        render.warn("FREESOLO_API_KEY is set — it will override the saved login"),
        render.note("exporting adapter — downloading then re-uploading…"),
        render.log_section("console_sft.txt"),
    ]
    for out in outputs:
        out.encode("ascii")  # raises if any non-ASCII slipped through


def test_checkpoints_and_mutations_are_curated_not_raw(monkeypatch) -> None:
    """checkpoints and the run-lifecycle mutations (cancel/deploy/undeploy/export) used to print a
    plain list / raw JSON on the styled path; they now render themed tables and curated cards."""
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    ck = tables.checkpoints_table(
        "flash-1", [{"step": 8, "repo_id": "acme/x", "subfolder": "grpo/step-8"}]
    )
    # themed header + table, not a bare `step N` list
    assert "checkpoints" in ck
    assert "STEP" in ck
    assert "flash-1/step-8" in ck  # the canonical short checkpoint ref

    dep = render.deployed(
        {
            "run_id": "flash-1",
            "checkpoint_id": "flash-1/step-20",
            "state": "deployed",
            "endpoint_name": "ep",
            "openai_base_url": "https://x",
        }
    )
    # curated card, not a raw json dump
    assert "deployed" in dep
    assert "flash-1/step-20" in dep
    assert "endpoint" in dep
    assert "{" not in dep

    exp = render.exported(
        {
            "checkpoint_id": "flash-1/final",
            "repository": "acme/x",
            "url": "https://x",
            "private": True,
        }
    )
    assert "exported" in exp
    assert "flash-1/final" in exp
    assert "acme/x" in exp
    assert "{" not in exp


def test_cancel_noop_on_terminal_run_is_not_a_false_confirmation(monkeypatch) -> None:
    """`flash runs cancel` against an already-terminal run is a server-side no-op that returns the
    unchanged state. The themed card must not flash a green "cancel requested" for that case —
    only a real transition to `cancelled` earns the confirmation; otherwise it stays honest."""
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    # a genuine cancel transition keeps the green confirmation
    live = render.cancelled({"run_id": "flash-1", "state": "cancelled"})
    assert "cancel requested for flash-1" in live
    # a no-op against a finished run: no false confirmation, just an honest "already done" + badge
    noop = render.cancelled({"run_id": "flash-1", "state": "done"})
    assert "cancel requested" not in noop
    assert "already done" in noop
    assert "done" in noop  # the real terminal state is still surfaced via the badge


def test_deploy_dry_run_is_not_a_false_success(monkeypatch) -> None:
    """`flash deploy --dry-run` validates without creating a deployment (server state `dry_run`);
    the themed card must not flash a green `✓ deploy` for it — only a real deploy confirms."""
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    # a real deploy still confirms
    live = render.deployed({"run_id": "flash-1", "state": "ready", "endpoint_name": "ep"})
    assert "deploy" in live
    # a dry run is a neutral validation line, not a success confirmation
    dry = render.deployed({"run_id": "flash-1", "state": "dry_run", "endpoint_name": "ep"})
    assert "dry run" in dry
    assert "nothing deployed" in dry
    assert "dry_run" not in dry


def test_undeploy_confirms_disabled_immutable_records(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    named = render.undeployed(
        {
            "run_id": "flash-1",
            "checkpoint_id": "flash-1/final",
            "disabled_checkpoints": ["flash-1/final"],
        }
    )
    assert "torn down flash-1/final" in named
    assert "disabled flash-1/final" in named
    idempotent = render.undeployed(
        {
            "run_id": "flash-1",
            "checkpoint_id": "flash-1/final",
            "disabled_checkpoints": [],
        }
    )
    assert "torn down flash-1/final" in idempotent


def test_export_card_reflects_requested_privacy(monkeypatch, capsys) -> None:
    """The control-plane export result carries no `private` field, so the styled card must read the
    privacy from the request (exports are private unless `--public`) instead of always saying
    `public` — otherwise a private adapter is misreported as world-readable."""
    import argparse

    import flash.client.runtime_secrets as runtime_secrets

    class _ExportClient:
        def export(self, adapter_id, *, repository, hf_token, private):
            # mirror the real server response shape, which has NO `private` key
            return {
                "run_id": adapter_id,
                "adapter_id": adapter_id,
                "repository": repository,
                "url": f"https://huggingface.co/{repository}",
                "source": "src:adapter",
            }

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(runtime_secrets, "resolve_hf_token", lambda *a, **k: "hf_x")
    monkeypatch.setattr(
        "flash.cli.commands.ops.deploy._hf_identity_and_write_access", lambda *_: "acme"
    )
    monkeypatch.setattr(cli_deploy, "client_from_config", lambda *a, **k: _ExportClient())

    # default export (no --public) is private; the card must say so, not "public"
    args = argparse.Namespace(adapter_id="flash-1", repository="acme/x", public=False, api_key=None)
    assert cli_deploy.cmd_export(args) == 0
    out = capsys.readouterr().out
    assert "private" in out
    assert "public" not in out

    # an explicit --public export is reported as public
    args.public = True
    assert cli_deploy.cmd_export(args) == 0
    assert "public" in capsys.readouterr().out


def test_error_path_themed_on_tty_plain_on_machine(monkeypatch, capsys) -> None:
    """main()'s catch-all error is the red ✗ idiom on a styled terminal, but stays the plain
    `error: {exc}` prefix on the machine path (what scripts and test_cli_errors.py match on)."""

    def _boom(*a, **k):
        raise ValueError("bad [environment] id")

    monkeypatch.setattr(cli_runs, "client_from_config", _boom)

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli.main(["runs", "list"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("✗")  # ✗ leads the themed line
    assert "error:" in err
    assert "bad [environment] id" in err

    monkeypatch.setenv("FLASH_STYLE", "0")
    assert cli.main(["runs", "list"]) == 1
    assert capsys.readouterr().err.startswith("error:")  # machine path unchanged


def test_argparse_usage_error_themed_on_tty_plain_on_machine(monkeypatch, capsys) -> None:
    """argparse handles usage errors (a missing arg, a bad subcommand) itself inside parse_args,
    before main()'s catch-all runs. On a styled terminal those now get the same red ✗ idiom + a
    dimmed `--help` pointer; the machine path keeps argparse's raw `usage: ... / prog: error:`
    block. Either way the exit code stays 2 (the agent contract and scripts match on it)."""

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])  # no subcommand -> argparse's "the following arguments are required" error
    assert excinfo.value.code == 2  # argparse's usage-error exit code is preserved
    err = capsys.readouterr().err
    assert err.startswith("✗")  # ✗ leads the themed line, like main()'s catch-all
    assert "error:" in err
    assert "--help" in err  # the dimmed next-step pointer replaces the raw usage block
    assert "usage:" not in err  # argparse's unstyled usage block is gone on a TTY

    monkeypatch.setenv("FLASH_STYLE", "0")
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("usage:")  # machine path keeps argparse's raw usage block
    assert "error:" in err
    assert "✗" not in err


def test_unexpected_error_themed_on_tty_traceback_on_machine(monkeypatch, capsys) -> None:
    """An error outside _USER_ERRORS (e.g. a read-only ~/.flash, a full disk) used to dump a raw
    Python traceback. On a styled terminal it now gets the red ✗ idiom + a `--debug` pointer; the
    machine path (and --debug) keep the full traceback as the bug signal CI and bug reports rely
    on, so the byte-for-byte machine contract for unexpected errors is unchanged."""

    def _boom(*a, **k):
        raise OSError("disk full")  # not in _USER_ERRORS (a plain OSError, not FileNotFoundError)

    monkeypatch.setattr(cli_runs, "client_from_config", _boom)

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli.main(["runs", "list"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("✗")  # red ✗ idiom, not a raw traceback
    assert "disk full" in err
    assert "--debug runs" in err  # names the exact command, with --debug BEFORE the subcommand
    assert "Traceback (most recent call last)" not in err

    # machine path: the raw exception propagates (a traceback), exactly as before this handler
    monkeypatch.setenv("FLASH_STYLE", "0")
    with pytest.raises(OSError, match="disk full"):
        cli.main(["runs", "list"])


def test_invalid_command_suggests_closest_match(monkeypatch, capsys) -> None:
    """An unknown command gets a short themed `did you mean '<closest>'?` suggestion instead of
    argparse's full `(choose from ...)` dump; the machine path keeps argparse's exact text."""

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["models", "deploly"])  # a typo of canonical `models deploy`
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unknown command 'deploly' (did you mean 'deploy'?)" in err
    assert "choose from" not in err  # the full choice list is gone on a TTY
    assert "--help" in err

    monkeypatch.setenv("FLASH_STYLE", "0")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["deploly"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err  # machine path unchanged
    assert "choose from" in err  # the full argparse list stays on the machine path


@pytest.mark.parametrize("bad_args", [["--key", "X"], ["--key=X"]])
def test_invalid_login_flag_suggests_api_key(monkeypatch, capsys, bad_args) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["login", *bad_args])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized argument '--key' (did you mean '--api-key'?)" in err
    assert "--help" in err


def test_invalid_flag_machine_path_keeps_argparse_message(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "0")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["login", "--key", "X"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "error: unrecognized arguments: --key X" in err
    assert "did you mean" not in err


def test_invalid_flag_without_close_match_keeps_original_message(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["login", "--xyzzy", "X"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments: --xyzzy X" in err
    assert "did you mean" not in err


def test_flag_valid_on_another_command_is_not_suggested_to_itself(monkeypatch, capsys) -> None:
    # --repository is real on `models export`, so the cross-subcommand candidate pool contains it
    # and the closest match to '--repository' is itself. echoing the rejected token back as its own
    # correction reads as a bug in the cli, so no suggestion is better than that one.
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["login", "--repository", "X"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments: --repository X" in err
    assert "did you mean '--repository'" not in err


def test_a_root_flag_suggestion_says_where_the_flag_goes(monkeypatch, capsys) -> None:
    """A correction the user cannot follow is not a correction.

    Root flags parse only before the subcommand, so `flash login --verbos` -> "did you mean
    '--verbose'?" sends the user to `flash login --verbose`, which fails identically. The suggestion
    has to carry the position, since that is the actual difference between working and not.
    """
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["login", "--verbos"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "did you mean '--verbose'" in err
    assert "it goes before the command" in err
    assert "flash --verbose ..." in err


def test_a_subcommand_flag_suggestion_is_not_given_a_reposition_hint(monkeypatch, capsys) -> None:
    """Only root-only flags need repositioning; a subcommand's own flag is already in place."""
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["runs", "log", "flash-1", "--folow"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "did you mean '--follow'" in err
    assert "it goes before the command" not in err


@pytest.mark.parametrize(
    ("argv", "absent"),
    [
        # --follow is real on `runs log`, not on `login`; --api-key is real on `login`, not here.
        (["login", "--folow"], "--follow"),
        (["runs", "log", "flash-1", "--api-ke"], "--api-key"),
    ],
)
def test_a_flag_from_an_unrelated_command_is_not_suggested(
    monkeypatch, capsys, argv, absent
) -> None:
    """A suggestion the user cannot follow is worse than none.

    The candidate pool has to span subcommands for a suggestion to be reachable at all, since
    argparse reports a subcommand's unknown tokens from the root. Drawing from the WHOLE tree
    though offers flags that belong to some other command, so the corrected line is rejected
    exactly like the first one and the hint has cost the user a round trip.
    """
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert absent not in err
    assert "did you mean" not in err


def test_a_correctly_spelled_root_flag_is_repositioned_not_respelled(monkeypatch, capsys) -> None:
    """`--verbose` after a command is a placement error, so there is nothing to correct it to.

    Dropping the exact token from the pool is what keeps a typo from being echoed back as its own
    fix, but it also removes `--verbose` here and lets the nearest sibling stand in: the answer was
    "did you mean '--version'?", a different flag with a different meaning.
    """
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["train", "x.toml", "--verbose"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "it goes before the command" in err
    assert "flash --verbose ..." in err
    assert "--version" not in err


def test_nested_flag_typo_is_scoped_to_parser_at_its_position(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["runs", "--folow", "log", "flash-1"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments: --folow" in err
    assert "--follow" not in err
    assert "did you mean" not in err


def test_repeated_short_root_flag_is_repositioned_without_downgrading(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["login", "-vv"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized argument '-vv'" in err
    assert "it goes before the command" in err
    assert "flash -vv ..." in err
    assert "flash -v ..." not in err


def test_flag_typo_after_option_terminator_has_no_unusable_suggestion(monkeypatch, capsys) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["runs", "log", "flash-1", "--", "--folow"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments: --folow" in err
    assert "--follow" not in err
    assert "did you mean" not in err


def test_a_lone_dash_reaches_argparses_own_message(monkeypatch, capsys) -> None:
    """A bare `-` names no option, so the typo machinery must not answer it with a flag.

    `_get_option_tuples` indexes the token's second character, so asking argparse about a lone dash
    raised IndexError from inside the handler whose whole job is to improve an error message.
    """
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["runs", "log", "flash-1", "-"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments: -" in err
    assert "did you mean" not in err
    assert "-v" not in err


@pytest.mark.parametrize("shape", ["tuple3", "tuple4", "list"])
def test_a_repeated_short_root_flag_is_recognized_under_every_parse_optional_shape(
    monkeypatch, shape
) -> None:
    """`_parse_optional` has three shapes across the interpreters `requires-python` admits.

    A 3-tuple before 3.11.9/3.12.3, a 4-tuple after the separator was inserted, and from 3.12.11
    a LIST of those tuples. Each one has broken this predicate in a different way: the 4-tuple
    unpack raised ValueError on the older interpreters, and the arity guard written for both
    tuples silently rejected the list, which put `-vv` back on the path that downgrades it to `-v`.

    The predicate is exercised directly rather than through `cli.main`, because argparse's own
    parse loop reads this same value: reshaping it parser-wide fails inside argparse and would
    prove nothing about our handling.
    """
    from flash.cli.parsing.errors import _is_repeated_root_short_option

    root = argparse.ArgumentParser()
    root.add_argument("-v", action="count")
    original = root._parse_optional

    def reshaped(arg_string):
        parsed = original(arg_string)
        if isinstance(parsed, list):  # normalize the interpreter's own shape away first
            parsed = parsed[0] if len(parsed) == 1 else None
        if parsed is None:
            return None
        if shape == "tuple3":  # drop the separator argparse inserted at index 2
            return (parsed[0], parsed[1], parsed[3]) if len(parsed) == 4 else parsed
        if shape == "list":
            return [parsed]
        return parsed

    monkeypatch.setattr(root, "_parse_optional", reshaped)
    assert _is_repeated_root_short_option(root, "-vv", frozenset({"-v"})) is True
    # a short flag that is not a repeat of itself must not be repositioned as one.
    assert _is_repeated_root_short_option(root, "-vx", frozenset({"-v"})) is False


def test_theme_light_and_dark_use_different_brand_colors(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setenv("TERM", "xterm-256color")  # color stays on under TERM=dumb CI
    monkeypatch.delenv("NO_COLOR", raising=False)

    monkeypatch.setenv("COLORFGBG", "15;0")  # light text on dark background -> dark theme
    dark = render.badge("done")
    assert "87;255;143" in dark  # bright brand green (#57ff8f) reads on a dark terminal

    monkeypatch.setenv("COLORFGBG", "0;15")  # dark text on light background -> light theme
    light = render.badge("done")
    assert "0;105;92" in light  # deep teal (#00695c) — the website's light-surface green
    assert dark != light


def test_theme_follows_terminal_background(monkeypatch) -> None:
    monkeypatch.setenv("COLORFGBG", "0;15")  # dark text on light background
    assert render._theme() == "light"
    monkeypatch.setenv("COLORFGBG", "15;0")  # light text on dark background
    assert render._theme() == "dark"
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert render._theme() == "dark"  # default when nothing indicates a light terminal


def test_run_status_surfaces_heartbeat_stage_and_age(monkeypatch) -> None:
    """The curated panel must show the worker stage and heartbeat age so a throttled (quiet)
    heartbeat reads as "alive N minutes ago", not as a dead worker."""
    import time as _time

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    fresh = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {
            "project": "11111111-1111-4111-8111-111111111111",
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
        },
        "last_heartbeat": {"stage": "sft_step", "step": 42, "ts": _time.time() - 125},
    }
    out = render.run_status(fresh)
    assert "worker" in out
    assert "sft_step" in out
    assert "step 42" in out
    assert "2m ago" in out
    assert "quiet is not dead" not in out  # fresh heartbeat needs no hint

    stale = dict(fresh, last_heartbeat={"stage": "sft_initializing", "ts": _time.time() - 840})
    out = render.run_status(stale)
    assert "sft_initializing" in out
    assert "14m ago" in out
    assert "quiet is not dead" in out, "a stale heartbeat on a running run must show the hint"

    ping = dict(
        fresh, last_heartbeat={"stage": "sft_initializing", "ts": _time.time(), "liveness": True}
    )
    out = render.run_status(ping)
    assert "alive ping" in out

    done = dict(stale, state="done")
    assert "quiet is not dead" not in render.run_status(done), "no hint on terminal runs"

    no_hb = {"run_id": "flash-1", "state": "running", "spec": {}}
    out = render.run_status(no_hb)
    assert "worker" not in out.split("details")[0]  # no empty heartbeat rows

    # malformed heartbeat fields must render defensively, never raise
    weird = dict(fresh, last_heartbeat={"stage": 123, "step": "seven", "ts": "not-a-number"})
    out = render.run_status(weird)
    assert "123" in out  # non-string stage still shown
    assert "ago" not in out.split("details")[0]  # unusable ts -> no age row


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_run_status_explains_rl_warmup(monkeypatch, stage: str) -> None:
    import time as _time

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    status = {
        "run_id": "flash-warmup",
        "state": "running",
        "spec": {
            "project": "11111111-1111-4111-8111-111111111111",
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
        },
        "last_heartbeat": {"stage": stage, "ts": _time.time()},
    }

    out = render.run_status(status)

    assert f"warming up (stage={stage})" in out
    assert "typically several minutes, sometimes 15-20 min" in out
    assert "setup is not billed" in out
    assert "do not cancel" in out

    status["state"] = "done"
    assert "warming up" not in render.run_status(status).split("details")[0]


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_run_status_omits_warmup_claim_for_stale_heartbeat(monkeypatch, stage: str) -> None:
    import time as _time

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    status = {
        "run_id": "flash-warmup",
        "state": "running",
        "spec": {
            "project": "11111111-1111-4111-8111-111111111111",
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
        },
        "last_heartbeat": {"stage": stage, "ts": _time.time() - 1201},
    }

    out = render.run_status(status)

    assert stage in out
    assert "20m ago" in out
    # the silence is still explained, but WHICH explanation depends on the stage: a liveness-backed
    # setup stage pings every ~4 min, so at 20m the panel says so and names the vanished-instance
    # possibility, while a one-shot ping like rl_train_start has only the generic throttle hint.
    # what must hold for both is that the age is accounted for and nothing reassures.
    assert ("quiet is not dead" in out) or ("longer than throttling explains" in out)
    assert "warming up" not in out
    assert "do not cancel" not in out


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_run_status_omits_warmup_claim_for_prior_attempt_heartbeat(monkeypatch, stage: str) -> None:
    import time as _time

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    # a recovered run flips back to running for attempt 1, but last_heartbeat is the fresh-looking
    # setup ping the dead attempt 0 left behind (retries reuse the seed heartbeat path). reassuring
    # "do not cancel" against it would mislabel a retry stall as healthy warmup.
    status = {
        "run_id": "flash-warmup",
        "state": "running",
        "spec": {
            "project": "11111111-1111-4111-8111-111111111111",
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
        },
        "remote": {"attempt": 1},
        "last_heartbeat": {"stage": stage, "ts": _time.time(), "attempt": 0},
    }

    out = render.run_status(status).split("details")[0]

    assert stage in out  # the stage is still reported factually
    assert "warming up" not in out
    assert "do not cancel" not in out


@pytest.mark.parametrize("stage", ["rl_train_start", "rl_initializing"])
def test_run_status_explains_warmup_when_heartbeat_matches_attempt(monkeypatch, stage: str) -> None:
    import time as _time

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    # a fresh heartbeat from the live attempt keeps the reassurance: the gate must not over-suppress.
    status = {
        "run_id": "flash-warmup",
        "state": "running",
        "spec": {
            "project": "11111111-1111-4111-8111-111111111111",
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "grpo",
        },
        "remote": {"attempt": 2},
        "last_heartbeat": {"stage": stage, "ts": _time.time(), "attempt": 2},
    }

    out = render.run_status(status)

    assert f"warming up (stage={stage})" in out
    assert "do not cancel" in out


def test_heartbeat_is_current_attempt_rejects_malformed_identities() -> None:
    is_current = heartbeat_ui.heartbeat_is_current_attempt

    # a malformed heartbeat attempt cannot prove it is the live attempt, so it earns no reassurance
    # and must never crash the display path (e.g. on inf, which the old int() coercion raised on).
    for bad in ["1", " 1 ", True, 2.7, float("inf"), -1, object()]:
        assert is_current({"remote": {"attempt": 1}}, {"attempt": bad}) is False

    # a malformed live attempt reads as unknown -> fall back to age gating (keep showing the note)
    for bad in ["1", True, 2.7, float("inf"), -1]:
        assert is_current({"remote": {"attempt": bad}}, {"attempt": 1}) is True

    # canonical int identities: exact match shows, mismatch suppresses
    assert is_current({"remote": {"attempt": 2}}, {"attempt": 2}) is True
    assert is_current({"remote": {"attempt": 2}}, {"attempt": 1}) is False


def test_live_attempt_is_one_rule_for_every_surface_that_names_an_attempt() -> None:
    """The spinner, the status line, and the worker-artifact labels must not disagree.

    They are read within one screen of each other, so two provenance rules read as a run that is on
    two attempts at once -- and the artifact labels would call the live attempt's console "previous".
    """
    live = heartbeat_ui.live_attempt

    # the plane's live attempt wins over the ping, which may be the superseded worker's.
    assert live({"remote": {"attempt": 2}, "last_heartbeat": {"attempt": 1}}) == 2
    # `remote` absent entirely (a managed payload that omits it): the ping is all there is.
    assert live({"last_heartbeat": {"attempt": 3}}) == 3
    # a `remote` that carries no usable attempt is not an answer either, so the ping still applies.
    assert live({"remote": {}, "last_heartbeat": {"attempt": 1}}) == 1
    for bad in ["1", True, 2.7, float("inf"), -1]:
        assert live({"remote": {"attempt": bad}, "last_heartbeat": {"attempt": 4}}) == 4
    # explicitly null `remote` is the teardown window: the attached ping belongs to a dead worker,
    # so there is no live attempt to report rather than that worker's.
    assert live({"remote": None, "last_heartbeat": {"attempt": 1}}) is None
    # nothing usable anywhere reads as unknown, never as attempt 0.
    assert live({}) is None
    assert live({"last_heartbeat": {"attempt": "1"}}) is None


def test_progress_age_always_adds_to_heartbeat_age(monkeypatch):
    from flash.cli.ui import heartbeat as heartbeat_ui
    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(heartbeat_ui.time, "time", lambda: 5000.0)

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "grpo"},
    }
    heartbeat = {
        "stage": "rl_step",
        "step": 14,
        "ts": 1990.0,
        "progress_age_s": 20.0,
    }

    real_out = render.run_status(dict(base, last_heartbeat=heartbeat))
    liveness_out = render.run_status(dict(base, last_heartbeat={**heartbeat, "liveness": True}))
    stale_progress_out = render.run_status(
        dict(base, last_heartbeat={**heartbeat, "progress_age_s": 1200.0})
    )

    for out in (real_out, liveness_out):
        assert "last known progress can be as old as 3030.0s" in out
    assert "last known progress can be as old as 4210.0s" in stale_progress_out

    for out in (real_out, liveness_out, stale_progress_out):
        assert "upload throttling no longer explains the gap" in out
        assert "this signal does not show recent progress" in out
        assert "proves recent worker-side progress" not in out
        assert "advances normally" not in out


def test_fresh_liveness_upload_surfaces_worker_measured_progress_gap(monkeypatch):

    monkeypatch.setattr(heartbeat_ui.time, "time", lambda: 2000.0)
    obj = {
        "state": "running",
        "last_heartbeat": {
            "stage": "rl_step",
            "step": 14,
            "ts": 1990.0,
            "liveness": True,
            "progress_age_s": 1800.0,
        },
    }

    pairs = heartbeat_ui._heartbeat_pairs(obj)
    assert ("worker", "rl_step · step 14 · alive ping") in pairs
    assert ("heartbeat", "10s ago") in pairs
    progress = dict(pairs)["progress"]
    assert "last known progress can be as old as 1810.0s" in progress
    assert "upload throttling no longer explains the gap" in progress
    assert heartbeat_ui._QUIET_HEARTBEAT_HINT not in progress
    assert len([label for label, _ in pairs if label == "progress"]) == 1


def test_fresh_opd_liveness_upload_surfaces_worker_measured_progress_gap(monkeypatch):

    monkeypatch.setattr(heartbeat_ui.time, "time", lambda: 2000.0)
    obj = {
        "state": "running",
        "last_heartbeat": {
            "stage": "opd_step",
            "step": 1,
            "ts": 1990.0,
            "liveness": True,
            "progress_age_s": 1800.0,
        },
    }

    pairs = heartbeat_ui._heartbeat_pairs(obj)
    assert ("worker", "opd_step · step 1 · alive ping") in pairs
    assert ("heartbeat", "10s ago") in pairs
    progress = dict(pairs)["progress"]
    assert "last known progress can be as old as 1810.0s" in progress
    assert "upload throttling no longer explains the gap" in progress


def test_sub_throttle_progress_age_preserves_legacy_stale_step_hint(monkeypatch):

    monkeypatch.setattr(heartbeat_ui.time, "time", lambda: 2000.0)
    base = {
        "state": "running",
        "remote": {"attempt": 2},
    }
    old_worker = {
        "stage": "rl_step",
        "step": 14,
        "attempt": 2,
        "ts": 1400.0,
    }

    def progress(hb: dict) -> str:
        return dict(heartbeat_ui._heartbeat_pairs(dict(base, last_heartbeat=hb)))["progress"]

    old_hint = progress(old_worker)
    assert progress({**old_worker, "progress_age_s": 220.0}) == old_hint
    assert progress({**old_worker, "progress_age_s": 20.0, "liveness": True}) == old_hint


@pytest.mark.parametrize(
    "bad_progress_age", [10**400, float("inf"), float("nan"), -1.0, True, "bad"]
)
def test_invalid_progress_age_falls_back_to_legacy_hint(monkeypatch, bad_progress_age):

    monkeypatch.setattr(heartbeat_ui.time, "time", lambda: 2000.0)
    base = {
        "state": "running",
        "remote": {"attempt": 2},
    }
    heartbeat_base = {
        "stage": "rl_step",
        "step": 14,
        "attempt": 2,
        "ts": 800.0,
    }

    legacy = dict(heartbeat_ui._heartbeat_pairs({**base, "last_heartbeat": heartbeat_base}))[
        "progress"
    ]
    invalid = dict(
        heartbeat_ui._heartbeat_pairs(
            {
                **base,
                "last_heartbeat": {**heartbeat_base, "progress_age_s": bad_progress_age},
            }
        )
    )["progress"]
    assert invalid == legacy


def test_missing_progress_age_preserves_legacy_stale_step_hint(monkeypatch):

    monkeypatch.setattr(heartbeat_ui.time, "time", lambda: 2000.0)
    obj = {
        "state": "running",
        "remote": {"attempt": 2},
        "last_heartbeat": {
            "stage": "rl_step",
            "step": 14,
            "attempt": 2,
            "ts": 800.0,
        },
    }

    assert heartbeat_ui._heartbeat_pairs(obj) == [
        ("worker", "rl_step · step 14"),
        ("heartbeat", "20m ago"),
        (
            "progress",
            (
                "the step above is the last one UPLOADED, not necessarily the one training is on; "
                "a throttled worker can hold it for many minutes while the trainer advances "
                "normally. uploads are held up to 15 min, so compare the age above against that "
                "(and your [wandb] run, if configured) before treating this as a stall"
            ),
        ),
    ]


def test_stale_training_step_is_labelled_as_reporting_lag(monkeypatch):
    """A frozen step on a throttled worker must not read as a stalled trainer (AS-018/AS-019).

    The pre-existing quiet hint points at `flash runs log`, which reads the same uploaded
    heartbeats -- so when the step counter is what went stale, that advice is a dead end.
    """
    import time as _time

    from flash.cli.ui import heartbeat as heartbeat_ui
    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "grpo"},
    }

    stale = dict(base, last_heartbeat={"stage": "rl_step", "step": 1, "ts": _time.time() - 1200})
    out = render.run_status(stale)
    assert "last one UPLOADED" in out
    assert "before treating this as a stall" in out

    # the hint must not send the user to `runs log` for worker output: cmd_log prints the worker
    # console only after the run reaches a terminal state, and the console artifact itself uploads
    # hourly, so neither can answer "is it still training?" at the 300s the hint fires.
    assert "runs log" not in out
    assert "worker output" not in out
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    # the reported incident ages sit inside the 900s throttle window; a gate at 900s would stay
    # silent on exactly the runs that prompted the fix.
    for incident_age in (559, 687):
        seen = dict(
            base,
            last_heartbeat={"stage": "rl_step", "step": 1, "ts": _time.time() - incident_age},
        )
        assert "last one UPLOADED" in render.run_status(seen), incident_age

    # a recently uploaded step is live enough to trust as-is.
    fresh = dict(base, last_heartbeat={"stage": "rl_step", "step": 73, "ts": _time.time() - 100})
    assert "last one UPLOADED" not in render.run_status(fresh)

    # one explanation for one silence: the quiet hint points at `runs log`, which reads the same
    # frozen heartbeats, so it must not ride along with the progress row.
    assert heartbeat_ui._QUIET_HEARTBEAT_HINT not in out

    # a SETUP stage has no step to be stale about -- it gets the warmup/quiet hints instead.
    setup = dict(base, last_heartbeat={"stage": "sft_initializing", "ts": _time.time() - 1200})
    assert "last one UPLOADED" not in render.run_status(setup)

    # a training stage with no step reported yet has nothing to qualify.
    stepless = dict(base, last_heartbeat={"stage": "rl_step", "ts": _time.time() - 1200})
    assert "last one UPLOADED" not in render.run_status(stepless)

    # terminal runs are not progressing, so the reassurance would be wrong.
    done = dict(stale, state="done")
    assert "last one UPLOADED" not in render.run_status(done)

    # step 0 is the cold first step: no optimizer update landed, so there is no hidden later step.
    step_zero = dict(
        base, last_heartbeat={"stage": "rl_step", "step": 0, "ts": _time.time() - 1200}
    )
    assert "last one UPLOADED" not in render.run_status(step_zero)

    # opd_step uses the same upload throttle, so its uploaded step can lag behind training too.
    opd = dict(base, last_heartbeat={"stage": "opd_step", "step": 4, "ts": _time.time() - 1200})
    assert "last one UPLOADED" in render.run_status(opd)

    # a heartbeat from a superseded attempt describes a dead worker, not throttled progress.
    superseded = dict(
        base,
        remote={"attempt": 2},
        last_heartbeat={"stage": "rl_step", "step": 1, "attempt": 1, "ts": _time.time() - 1200},
    )
    assert "last one UPLOADED" not in render.run_status(superseded)

    # ...but the live attempt still gets the reassurance.
    live = dict(
        base,
        remote={"attempt": 2},
        last_heartbeat={"stage": "rl_step", "step": 1, "attempt": 2, "ts": _time.time() - 1200},
    )
    assert "last one UPLOADED" in render.run_status(live)

    # w&b is optional, so the advice must name a signal that always exists. `runs log` was that
    # signal until it turned out it cannot answer the question while the run is live (see the
    # `runs log` assertion above); the heartbeat age row can, is rendered from the same payload,
    # and so is never absent.
    assert "the age above" in out
    assert "if configured" in out


def test_long_silence_at_a_liveness_setup_stage_names_both_causes(monkeypatch):
    """A frozen setup stage must not read as ordinary throttling (issue 26).

    These stages hold a liveness thread on a ~240s cadence, so a much older heartbeat is NOT the
    upload throttle. It is either one long blocking call (a cold per-datacenter weight cache) or a
    vanished instance -- and the panel cannot tell which. The generic quiet hint says "quiet is not
    dead", which asserts the healthy reading of exactly the ambiguity it cannot resolve.
    """
    import time as _time

    from flash.cli.ui import heartbeat as heartbeat_ui
    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "sft"},
    }

    # model_prefetching is the stage that actually pulls base weights, so it is the one whose
    # silence the cold-cache reading describes.
    frozen = dict(base, last_heartbeat={"stage": "model_prefetching", "ts": _time.time() - 1200})
    out = render.run_status(frozen)
    assert "longer than throttling explains" in out
    # every reading, because the panel genuinely cannot distinguish them -- naming only one is how
    # a user either cancels a healthy download or waits on a box that is already gone.
    assert "cold weight cache" in out
    assert "instance is gone" in out
    # heartbeat uploads are best-effort and roll the throttle slot back on failure, so a healthy
    # worker can keep running while its age grows without bound. presenting only "blocked or dead"
    # sends that user to wait for an attempt change that is never coming.
    assert "uploads may be failing" in out
    # one explanation per silence: the generic hint must not ride along with the specific one.
    assert heartbeat_ui._QUIET_HEARTBEAT_HINT not in out

    # inside the cadence, silence is ordinary and must stay unremarked by this hint.
    fresh = dict(base, last_heartbeat={"stage": "sft_model_load", "ts": _time.time() - 300})
    assert "longer than throttling explains" not in render.run_status(fresh)

    # a one-shot ping (no liveness thread) has no 240s cadence to be measured against, so the
    # generic throttle hint remains the honest reading there.
    one_shot = dict(base, last_heartbeat={"stage": "rl_train_start", "ts": _time.time() - 1200})
    assert "longer than throttling explains" not in render.run_status(one_shot)

    # a terminal run is not waiting on anything.
    done = dict(frozen, state="done")
    assert "longer than throttling explains" not in render.run_status(done)

    # a superseded attempt's ping describes a worker that is already gone; calling that a possibly
    # healthy download would hide that the replacement has published nothing at all.
    superseded = dict(
        base,
        remote={"attempt": 2},
        last_heartbeat={"stage": "sft_model_load", "attempt": 1, "ts": _time.time() - 1200},
    )
    assert "longer than throttling explains" not in render.run_status(superseded)

    # a training step is the other hint's job; the two must never both fire.
    stepping = dict(
        base, last_heartbeat={"stage": "sft_step", "step": 3, "ts": _time.time() - 1200}
    )
    stepping_out = render.run_status(stepping)
    assert "longer than throttling explains" not in stepping_out
    assert "last one UPLOADED" in stepping_out


def test_setup_hint_does_not_blame_a_download_on_a_stage_that_never_downloads(monkeypatch):
    """The hint fires for every liveness setup stage, but only some of them fetch weights.

    Telling a user that `sft_configuring` is "downloading tens of GB" sends them hunting for a
    transfer that stage never performs. A wrong explanation is its own misdiagnosis, which is the
    failure class this hint was added to remove -- so the download clause has to be conditional.
    """
    import time as _time

    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "sft"},
    }

    frozen = dict(base, last_heartbeat={"stage": "sft_configuring", "ts": _time.time() - 1200})
    out = render.run_status(frozen)
    # still flagged: the silence is just as unexplained by throttling here.
    assert "longer than throttling explains" in out
    assert "instance is gone" in out
    # but not attributed to a download this stage does not perform.
    assert "downloads tens of GB" not in out


def test_warmup_reassurance_yields_once_the_silence_is_unexplained(monkeypatch):
    """The warmup and stale-setup windows overlap, and they say opposite things.

    `rl_initializing` counts as fresh for 1200s but goes silent-unexplained at 900s, so between
    those the panel printed "setup is not billed; do not cancel" directly above "the instance may
    be gone ... check before cancelling". A panel that contradicts itself is worse than either row
    alone: the user cannot act on it at all.
    """
    import time as _time

    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "rl"},
    }

    # inside the overlap: the specific hint wins, the reassurance stands down.
    overlap = dict(base, last_heartbeat={"stage": "rl_initializing", "ts": _time.time() - 1080})
    out = render.run_status(overlap)
    assert "longer than throttling explains" in out
    assert "do not cancel" not in out, "reassurance printed alongside a possible-dead-instance hint"

    # and warmup still reassures where it is the honest reading.
    fresh = dict(base, last_heartbeat={"stage": "rl_initializing", "ts": _time.time() - 300})
    fresh_out = render.run_status(fresh)
    assert "do not cancel" in fresh_out
    assert "longer than throttling explains" not in fresh_out


def test_opd_model_load_is_not_described_as_a_weight_download(monkeypatch):
    """`opd_model_load` reads the cached config with local_files_only; it downloads nothing.

    Both model-load stages are emitted only after `prefetch_model` returns, so neither is the
    tens-of-GB base-weight transfer (that is `model_prefetching`). SFT still fetches a warm-start
    adapter inside its span, so it keeps a download explanation -- a smaller one.
    """
    import time as _time

    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "opd"},
    }

    opd = dict(base, last_heartbeat={"stage": "opd_model_load", "ts": _time.time() - 1200})
    opd_out = render.run_status(opd)
    assert "longer than throttling explains" in opd_out
    assert "tens of GB" not in opd_out
    assert "can block for minutes" in opd_out

    # sft's span MAY fetch an adapter, but only when the run warm-starts from one -- most do not,
    # so the wording has to hedge rather than assert a transfer that usually is not happening.
    sft = dict(base, last_heartbeat={"stage": "sft_model_load", "ts": _time.time() - 1200})
    sft_out = render.run_status(sft)
    assert "if the run warm-starts from one" in sft_out
    assert "tens of GB" not in sft_out
    # the hub pull is not the per-datacenter weight volume, so the region is not the explanation.
    sft_dc = dict(
        base,
        last_heartbeat={"stage": "sft_model_load", "ts": _time.time() - 1200, "dc": "EU-RO-1"},
    )
    assert "datacenter above" not in render.run_status(sft_dc)


def test_stale_datacenter_is_labelled_as_the_previous_attempt(monkeypatch):
    """A retry reuses the heartbeat path, so `dc` can belong to the worker that already died.

    This row exists so two runs of the same config are comparable by region. Presenting the dead
    attempt's region as the live one's corrupts exactly that comparison: the replacement may still
    be provisioning, or may have landed somewhere with a completely different cache state.
    """
    import time as _time

    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "sft"},
    }

    superseded = dict(
        base,
        remote={"attempt": 2},
        last_heartbeat={
            "stage": "sft_model_load",
            "attempt": 1,
            "ts": _time.time() - 1200,
            "dc": "EU-RO-1",
        },
    )
    out = render.run_status(superseded).split("details", 1)[0]
    assert "EU-RO-1" in out, "the previous region is still worth showing"
    assert "previous attempt" in out, "it is presented as the live attempt's region"

    # when the heartbeat IS the live attempt, the row stays unqualified.
    current = dict(
        base,
        remote={"attempt": 2},
        last_heartbeat={
            "stage": "sft_model_load",
            "attempt": 2,
            "ts": _time.time() - 1200,
            "dc": "EU-RO-1",
        },
    )
    current_out = render.run_status(current).split("details", 1)[0]
    assert "EU-RO-1" in current_out
    assert "previous attempt" not in current_out


def test_cadence_is_asserted_as_fact_only_when_a_liveness_ping_proves_it():
    """Worker code is content-addressed per submission, so a pre-upgrade run still emits one-shot
    pings while a newer CLI reads them. Claiming "pings every ~4 min" as fact is then false, and
    every reading built on that premise inherits the error.

    The hint must still FIRE in both cases: a run frozen inside its very first 240s window has no
    liveness ping either, and that is the primary freeze this hint was written to diagnose. So the
    distinction belongs in the wording, not in whether the diagnosis appears at all.
    """
    from flash.cli.ui.heartbeat import _stale_setup_hint

    proven = _stale_setup_hint({"stage": "sft_model_load", "liveness": True}, 1200.0, running=True)
    assert "pings every ~4 min" in proven
    assert "expected to ping" not in proven

    unproven = _stale_setup_hint({"stage": "sft_model_load"}, 1200.0, running=True)
    assert unproven, "the hint must still fire; a first-window freeze has no liveness ping either"
    assert "expected to ping" in unproven
    assert "unless this run predates that" in unproven


def test_no_stage_clause_contradicts_the_sentence_it_completes():
    """The clause is a parenthetical inside "it may be inside one long blocking call (...)".

    A clause saying no long call is expected there negates the sentence mid-parenthesis and steers
    the user away from the healthy reading. Every one of these stages holds a liveness wrap
    BECAUSE it can block for minutes, so no stage may deny that. Checked across the whole set
    rather than per-stage: this bug arrived via an else branch that no single-stage test covered.
    """
    from flash.cli.ui.heartbeat import _LIVENESS_SETUP_STAGES, _stale_setup_hint

    denials = ("no long call", "does no download", "is unusual here")
    for stage in sorted(_LIVENESS_SETUP_STAGES):
        hint = _stale_setup_hint({"stage": stage}, 1200.0, running=True)
        assert hint, f"{stage} is in the set but produced no hint"
        clause = hint.split("one long blocking call (", 1)[1].split(")", 1)[0]
        assert clause, f"{stage} produced an empty clause"
        for denial in denials:
            assert denial not in clause, (
                f"{stage} clause denies the call it is describing: {clause}"
            )


@pytest.mark.parametrize("stage", ["sft_finalizing", "rl_finalizing", "opd_finalizing"])
def test_finalizing_silence_is_not_called_unusual(stage, monkeypatch):
    """These export and upload the adapter, so a long blocking stretch is expected, not anomalous.

    They still earn the stale hint (they hold a keepalive wrap on the 240s cadence), but telling a
    user their silence is unusual is worst here: training is done and only the upload stands between
    them and their result, so a nudge toward cancelling destroys finished work.
    """
    import time as _time

    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    obj = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "sft"},
        "last_heartbeat": {"stage": stage, "ts": _time.time() - 1200},
    }
    out = render.run_status(obj)
    assert "longer than throttling explains" in out
    assert "cold mount or a venv install" not in out, "the generic fallback misnames this wait"
    assert "exports and uploads the adapter" in out


def test_setup_hint_cites_the_datacenter_only_when_the_row_is_rendered(monkeypatch):
    """`dc` is optional, and the datacenter row is omitted when it is absent.

    The hint says "the datacenter above", so without this the text points at a row that was never
    rendered. Both read the same heartbeat dict, so the reference must follow the row.
    """
    import time as _time

    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "sft"},
    }

    without_dc = dict(
        base, last_heartbeat={"stage": "model_prefetching", "ts": _time.time() - 1200}
    )
    out = render.run_status(without_dc)
    assert "datacenter above" not in out, "cites a row that is not on the panel"
    # the download itself is still the right explanation for this stage.
    assert "downloads tens of GB" in out

    with_dc = dict(
        base,
        last_heartbeat={"stage": "model_prefetching", "ts": _time.time() - 1200, "dc": "EU-RO-1"},
    )
    out_dc = render.run_status(with_dc)
    assert "datacenter above" in out_dc
    assert "EU-RO-1" in out_dc


def test_status_shows_the_datacenter_the_worker_landed_in(monkeypatch):
    """Base weights come from a per-datacenter cache volume and the allocator does not pin a region.

    The worker already stamps `dc` on every heartbeat, but nothing rendered it -- so an identical
    config that relaunched into a cold region looked like an unexplainable freeze.
    """
    import time as _time

    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-0.8B", "algorithm": "sft"},
    }

    located = dict(
        base, last_heartbeat={"stage": "sft_model_load", "ts": _time.time() - 60, "dc": "EU-RO-1"}
    )
    assert "EU-RO-1" in render.run_status(located).split("details")[0]

    # absent on providers that do not report one: render nothing rather than an empty row.
    unlocated = dict(base, last_heartbeat={"stage": "sft_model_load", "ts": _time.time() - 60})
    assert "datacenter" not in render.run_status(unlocated).split("details")[0]


def test_quiet_hint_does_not_send_users_to_an_hourly_log(monkeypatch):
    """Worker stdout uploads hourly, so `runs log` cannot answer "is it alive?" mid-run.

    The hint used to point there, which is what made a healthy short run look hung.
    """
    import time as _time

    from flash.cli.ui import heartbeat as heartbeat_ui
    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    quiet = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "grpo"},
        "last_heartbeat": {"stage": "rl_train_start", "ts": _time.time() - 400},
    }
    out = render.run_status(quiet).split("details")[0]

    assert heartbeat_ui._QUIET_HEARTBEAT_HINT in out
    assert "runs log" not in out
    # it has to name the surfaces that do update while the run is live.
    assert "the age above" in out
    assert "hourly" in out


def test_a_cleared_remote_does_not_present_a_dead_attempts_region_as_live(monkeypatch):
    """`remote: null` is the relaunch window, so its attached `dc` is the torn-down worker's.

    `heartbeat_is_current_attempt` answers True here because it cannot prove otherwise from the
    identity alone: there is no live attempt number to compare against. Reading that True as
    "current" printed the dead attempt's region unqualified, which corrupts the exact comparison the
    row exists for -- the replacement may land in a region with a completely different cache state.

    The log-follow spinner already draws this line the same way (`_log_follow_progress`), so this
    asserts the two surfaces agree rather than inventing a second contract.
    """
    import time as _time

    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "sft"},
    }
    heartbeat = {
        "stage": "sft_model_load",
        "attempt": 1,
        "ts": _time.time() - 1200,
        "dc": "EU-RO-1",
    }

    cleared = dict(base, remote=None, last_heartbeat=dict(heartbeat))
    out = render.run_status(cleared).split("details", 1)[0]
    assert "EU-RO-1" in out, "the region is still worth showing"
    assert "previous attempt" in out, "a cleared remote means that worker is already gone"

    # an ABSENT `remote` is a plane that never surfaces the field, not a teardown. it must keep
    # falling back, or every such plane labels a perfectly live worker as dead.
    absent = dict(base, last_heartbeat=dict(heartbeat))
    assert "previous attempt" not in render.run_status(absent).split("details", 1)[0]


def test_a_superseded_ping_is_not_called_alive_by_the_quiet_hint(monkeypatch):
    """The panel must not label a row `previous attempt` and then reassure about that same ping.

    `quiet is not dead` is written about a live worker holding uploads on the throttle. Printed
    beside a heartbeat whose worker is provably torn down it is simply false, and it is the longer,
    more reassuring of the two readings, so it wins: a run between attempts reads as healthy.
    """
    import time as _time

    from flash.cli.ui import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-9B", "algorithm": "grpo"},
    }
    # rl_train_start is deliberate: it is in neither hint's stage set, so before this fix nothing
    # suppressed the quiet hint and the contradiction was reachable.
    superseded = dict(
        base,
        remote={"attempt": 2},
        last_heartbeat={"stage": "rl_train_start", "attempt": 1, "ts": _time.time() - 400},
    )
    out = render.run_status(superseded).split("details", 1)[0]
    assert "quiet is not dead" not in out, "that ping's worker is gone; quiet IS dead there"
    # it is replaced, not merely dropped: the silence still needs an explanation, and this one is
    # true. it must also not send the user to cancel, since the replacement may be provisioning.
    assert "already gone" in out
    assert "has not published one yet" in out

    # the control: same age, same stage, live attempt. the ordinary reassurance must survive.
    live = dict(
        base,
        remote={"attempt": 2},
        last_heartbeat={"stage": "rl_train_start", "attempt": 2, "ts": _time.time() - 400},
    )
    live_out = render.run_status(live).split("details", 1)[0]
    assert "quiet is not dead" in live_out
    assert "already gone" not in live_out


def test_at_most_one_progress_hint_can_ever_fire():
    """The `or` chain in `_heartbeat_pairs` must not be what keeps the panel to a single reading.

    Only one `progress` row is rendered, so if two hints could both fire, the chain's hand-written
    order would silently decide which diagnosis a user sees -- and reordering it would change the
    displayed reading with nothing failing. The three predicates are meant to be mutually exclusive
    by their own gates instead: `current_attempt` splits `_superseded_hint` from the other two, and
    disjoint stage sets split those two from each other. That is the property worth pinning, because
    it is what makes the order an implementation detail rather than load-bearing behavior.

    Asserted by exhaustive enumeration rather than a few samples: the gates interact across four
    inputs, and a hand-picked case cannot show that no combination overlaps.
    """
    import itertools
    import time as _time

    from flash.cli.ui import heartbeat as hb_mod

    stages = sorted(
        hb_mod._LIVENESS_SETUP_STAGES
        | hb_mod._TRAINING_STEP_STAGES
        | hb_mod._WARMUP_STAGES
        | {"unknown_stage"}
    )
    now = _time.time()
    overlaps = []
    for stage, age, current, running, step in itertools.product(
        stages,
        # straddle both gates (300s quiet, 900s setup-silent) from either side.
        [10.0, 400.0, 1000.0, 5000.0],
        [True, False],
        [True, False],
        [None, 0, 455],
    ):
        heartbeat = {"stage": stage, "ts": now - age, "step": step}
        fired = [
            name
            for name, hint in (
                (
                    "stale_step",
                    hb_mod._stale_step_hint(
                        heartbeat, age, running=running, current_attempt=current
                    ),
                ),
                (
                    "stale_setup",
                    hb_mod._stale_setup_hint(
                        heartbeat, age, running=running, current_attempt=current
                    ),
                ),
                (
                    "superseded",
                    hb_mod._superseded_hint(age, running=running, current_attempt=current),
                ),
            )
            if hint
        ]
        if len(fired) > 1:
            overlaps.append((stage, age, current, running, step, fired))
    assert not overlaps, (
        f"two hints fire for the same heartbeat, so the chain order decides: {overlaps[:3]}"
    )


def test_each_hint_owns_the_heartbeats_it_is_written_for():
    """Mutual exclusivity is worthless if the wrong hint owns a case, or none does.

    Pairs with the test above: that one proves at most one fires, this one proves the right one
    does. A superseded ping must get the superseded reading on BOTH a setup stage and a training
    stage -- and in particular must never be told its dead worker's step is merely lagging.
    """
    import time as _time

    from flash.cli.ui.heartbeat import _heartbeat_pairs

    def progress(stage: str, *, attempt: int, **hb) -> str:
        pairs = _heartbeat_pairs(
            {
                "state": "running",
                "remote": {"attempt": 2},
                "last_heartbeat": {
                    "stage": stage,
                    "attempt": attempt,
                    "ts": _time.time() - 1200,
                    **hb,
                },
            }
        )
        return dict(pairs).get("progress", "")

    # superseded: the stage-aware hints decline it, and the superseded reading is what is left.
    assert "already gone" in progress("sft_model_load", attempt=1)
    assert "already gone" in progress("sft_step", attempt=1, step=455)
    assert "last one UPLOADED" not in progress("sft_step", attempt=1, step=455)

    # live: the stage-aware hints own it, and each says the more specific thing.
    assert "longer than throttling explains" in progress("sft_model_load", attempt=2)
    assert "last one UPLOADED" in progress("sft_step", attempt=2, step=455)
    assert "already gone" not in progress("sft_step", attempt=2, step=455)


def test_superseded_hint_stays_quiet_when_there_is_no_silence_to_explain():
    """Below the quiet threshold the label alone is accurate and nothing contradicts it.

    A hint about a stale heartbeat printed next to a 30-second-old one is noise, and noise on the
    panel is what trained users to ignore it.
    """
    from flash.cli.ui.heartbeat import _superseded_hint

    assert _superseded_hint(30.0, running=True, current_attempt=False) is None
    assert _superseded_hint(1200.0, running=True, current_attempt=False)
    # terminal runs are not waiting on anything, and a live attempt is the other hints' business.
    assert _superseded_hint(1200.0, running=False, current_attempt=False) is None
    assert _superseded_hint(1200.0, running=True, current_attempt=True) is None
    assert _superseded_hint(None, running=True, current_attempt=False) is None
