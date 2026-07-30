"""The themed CLI output is TTY-gated: piped/scripted output stays byte-for-byte machine
readable, while ``FLASH_STYLE=1`` forces the human theme. These tests pin both sides of that
gate so a future tweak to the theme can't silently change the machine contract that `jq`,
scripts, and the trainer agent depend on.
"""

from __future__ import annotations

import json

import pytest

import flash.cli as cli
from flash import __version__
from flash.cli import render


class _Client:
    def get_run(self, run_id: str) -> dict:
        return {
            "run_id": run_id,
            "state": "done",
            "cost_usd": 0.25,
            "error": None,
            "spec": {
                "project": "11111111-1111-4111-8111-111111111111",
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": "grpo",
            },
        }


@pytest.fixture
def fake_client(monkeypatch) -> None:
    monkeypatch.setattr(cli.commands, "client_from_config", lambda *a, **k: _Client())


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
    assert "Qwen/Qwen3.5-0.8B" in out
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
    assert "Qwen/Qwen3.5-0.8B" in out
    assert "PARAMS" not in out  # ids only — no reintroduced detail columns

    assert cli.main(["runs", "status", "flash-1"]) == 0
    out = capsys.readouterr().out
    # curated panel up top, but the full JSON is still emitted below (no data dropped)
    assert "details" in out
    assert "cost_usd" in out
    assert '"state": "done"' in out


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
    runs = render.runs_table([run])
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
    runs = render.runs_table([allocated_run])
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
    assert "RTX Pro 6000" in render.runs_table([spec_only_run])
    assert "RTX Pro 6000" in render.run_status(spec_only_run)

    # the removed legacy remote["gpu"] key is no longer honored: the label resolves from the
    # authoritative allocated_gpu or the provisional spec type, never the dead key.
    legacy_key_run = {
        "run_id": "flash-legacy-key",
        "state": "done",
        "spec": {"gpu": {"type": "RTX Pro 6000"}},
        "remote": {"gpu": "B200"},
    }
    assert "B200" not in render.runs_table([legacy_key_run])
    assert "RTX Pro 6000" in render.runs_table([legacy_key_run])


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
        render.models_table([{"id": "acme/x"}]),
        render.gpus_table([("RTX 5090", 32, 0.99)], "Tip: selection is automatic — no pinning"),
        render.runs_table([{"run_id": "r", "state": "done", "spec": {}}]),
        render.deployments_table([{"run_id": "r", "deployment": {"state": "ready"}}]),
        render.env_setup(
            ["environment.py", "dataset/train.jsonl", "configs/rl.toml"],
            "11111111-1111-4111-8111-111111111111",
        ),
        render.env_list([]),
        render.empty("runs", "0 runs", "no runs yet — submit one with `flash train`"),
        render.submitted("flash-xyz"),
        render.run_status({"run_id": "r", "state": "failed", "spec": {}, "error": "boom — bad"}),
        render.checkpoints_table(
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

    ck = render.checkpoints_table(
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
    """`flash cancel` against an already-terminal run is a server-side no-op that returns the
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
    monkeypatch.setattr(cli.commands, "client_from_config", lambda *a, **k: _ExportClient())

    # default export (no --public) is private; the card must say so, not "public"
    args = argparse.Namespace(adapter_id="flash-1", repository="acme/x", public=False, api_key=None)
    assert cli.commands.cmd_export(args) == 0
    out = capsys.readouterr().out
    assert "private" in out
    assert "public" not in out

    # an explicit --public export is reported as public
    args.public = True
    assert cli.commands.cmd_export(args) == 0
    assert "public" in capsys.readouterr().out


def test_error_path_themed_on_tty_plain_on_machine(monkeypatch, capsys) -> None:
    """main()'s catch-all error is the red ✗ idiom on a styled terminal, but stays the plain
    `error: {exc}` prefix on the machine path (what scripts and test_cli_errors.py match on)."""

    def _boom(*a, **k):
        raise ValueError("bad [environment] id")

    monkeypatch.setattr(cli.commands, "client_from_config", _boom)

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

    monkeypatch.setattr(cli.commands, "client_from_config", _boom)

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
    assert "did you mean 'deploy'?" in err
    assert "choose from" not in err  # the full choice list is gone on a TTY
    assert "--help" in err

    monkeypatch.setenv("FLASH_STYLE", "0")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["deploly"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err  # machine path unchanged
    assert "choose from" in err  # the full argparse list stays on the machine path


def test_theme_light_and_dark_use_different_brand_colors(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setenv("TERM", "xterm-256color")  # color stays on under TERM=dumb CI
    monkeypatch.delenv("NO_COLOR", raising=False)

    monkeypatch.setenv("FLASH_THEME", "dark")
    dark = render.badge("done")
    assert "87;255;143" in dark  # bright brand green (#57ff8f) reads on a dark terminal

    monkeypatch.setenv("FLASH_THEME", "light")
    light = render.badge("done")
    assert "0;105;92" in light  # deep teal (#00695c) — the website's light-surface green
    assert dark != light


def test_theme_follows_terminal_background(monkeypatch) -> None:
    monkeypatch.delenv("FLASH_THEME", raising=False)
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
    assert "quiet is not dead" in out
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
    is_current = render.heartbeat_is_current_attempt

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


def test_stale_training_step_is_labelled_as_reporting_lag(monkeypatch):
    """A frozen step on a throttled worker must not read as a stalled trainer (AS-018/AS-019).

    The pre-existing quiet hint points at `flash runs log`, which reads the same uploaded
    heartbeats -- so when the step counter is what went stale, that advice is a dead end.
    """
    import time as _time

    from flash.cli import render

    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    base = {
        "run_id": "flash-1",
        "state": "running",
        "spec": {"model": "Qwen/Qwen3.5-0.8B", "algorithm": "grpo"},
    }

    stale = dict(
        base, last_heartbeat={"stage": "rl_step", "step": 1, "ts": _time.time() - 1200}
    )
    out = render.run_status(stale)
    assert "last one UPLOADED" in out
    assert "before treating this as a stall" in out

    # a step that is merely throttled-quiet, not stale, must stay silent.
    fresh = dict(base, last_heartbeat={"stage": "rl_step", "step": 73, "ts": _time.time() - 400})
    assert "last one UPLOADED" not in render.run_status(fresh)

    # a SETUP stage has no step to be stale about -- it gets the warmup/quiet hints instead.
    setup = dict(
        base, last_heartbeat={"stage": "sft_initializing", "ts": _time.time() - 1200}
    )
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

    # opd_step force-commits at the 60s floor, so an old one is a real stall, not reporting lag.
    opd = dict(base, last_heartbeat={"stage": "opd_step", "step": 4, "ts": _time.time() - 1200})
    assert "last one UPLOADED" not in render.run_status(opd)

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

    # w&b is optional, so the advice must name a signal that always exists.
    assert "flash runs log" in out
    assert "if configured" in out
