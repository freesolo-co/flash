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
        render.exported({"adapter_id": "r", "repository": "acme/x", "url": "u", "private": True}),
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
            "state": "deployed",
            "endpoint_name": "ep",
            "openai_base_url": "https://x",
        }
    )
    # curated card, not a raw json dump
    assert "deployed" in dep
    assert "endpoint" in dep
    assert "{" not in dep

    exp = render.exported(
        {"adapter_id": "flash-1", "repository": "acme/x", "url": "https://x", "private": True}
    )
    assert "exported" in exp
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
            "disabled_aliases": ["flash-1"],
            "disabled_revisions": ["flash-1@final." + "a" * 40],
        }
    )
    assert "torn down flash-1" in named
    assert "flash-1@final" in named
    idempotent = render.undeployed(
        {"run_id": "flash-1", "disabled_aliases": [], "disabled_revisions": []}
    )
    assert "torn down flash-1" in idempotent


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
