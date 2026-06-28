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
            "spec": {"model": "Qwen/Qwen3.5-4B", "algorithm": "grpo"},
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

    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    # plain output is bare model ids, one per line, with no themed header or detail columns
    assert "Qwen/Qwen3.5-0.8B" in out
    assert "supported base models" not in out
    assert all(line in out for line in ("Qwen/Qwen3.5-9B", "openbmb/MiniCPM5-1B"))
    assert "\t" not in out

    assert cli.main(["status", "flash-1"]) == 0
    payload = json.loads(capsys.readouterr().out)  # must stay parseable JSON
    assert payload["state"] == "done"
    assert payload["cost_usd"] == 0.25


def test_styled_path_is_themed_but_lossless(monkeypatch, fake_client, capsys) -> None:
    # Themed layout on, color off (NO_COLOR), so highlighted-JSON tokens stay contiguous and we
    # can assert the full payload survives the themed view.
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")

    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "supported base models" in out  # themed header
    assert "Qwen/Qwen3.5-0.8B" in out
    assert "PARAMS" not in out  # ids only — no reintroduced detail columns

    assert cli.main(["status", "flash-1"]) == 0
    out = capsys.readouterr().out
    # curated panel up top, but the full JSON is still emitted below (no data dropped)
    assert "details" in out
    assert "cost_usd" in out
    assert '"state": "done"' in out


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
        render.deployments_table([{"run_id": "r", "deployment": {"gpu": "RTX 4090"}}]),
        render.env_setup(["environment.py", "datasets/train.jsonl", "configs/rl.toml"]),
        render.env_list([]),
        render.empty("runs", "0 runs", "no runs yet — submit one with `flash train`"),
        render.submitted("flash-xyz"),
        render.run_status({"run_id": "r", "state": "failed", "spec": {}, "error": "boom — bad"}),
        render.checkpoints_table(
            "r", [{"step": 8, "repo_id": "acme/x", "subfolder": "grpo/step-8"}]
        ),
        render.cancelled({"run_id": "r", "state": "cancelled"}),
        render.deployed({"run_id": "r", "state": "deployed", "endpoint_name": "ep", "url": "u"}),
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
    assert "acme/x:grpo/step-8" in ck

    dep = render.deployed(
        {"run_id": "flash-1", "state": "deployed", "endpoint_name": "ep", "url": "https://x"}
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
    live = render.deployed(
        {"run_id": "flash-1", "state": "ready", "endpoint_name": "ep", "gpu": "A100"}
    )
    assert "deploy" in live
    # a dry run is a neutral validation line, not a success confirmation
    dry = render.deployed(
        {"run_id": "flash-1", "state": "dry_run", "endpoint_name": "ep", "gpu": "A100"}
    )
    assert "dry run" in dry
    assert "nothing deployed" in dry
    assert "dry_run" not in dry  # the old green `✓ deploy ● dry_run` badge form is gone


def test_undeploy_confirms_teardown_and_names_endpoints(monkeypatch) -> None:
    """`flash undeploy` always confirms the teardown: the server clears the deployment record
    idempotently, so an empty `deleted_endpoints` (the serving adapter was already gone / 404) is
    still a real teardown, not a no-op the response can prove. A real deregistration names what
    was removed; the empty case must never claim nothing was torn down."""
    monkeypatch.setenv("FLASH_STYLE", "1")
    monkeypatch.setenv("NO_COLOR", "1")
    # a real teardown confirms and names what was deregistered
    named = render.undeployed({"run_id": "flash-1", "deleted_endpoints": ["live-x"]})
    assert "torn down flash-1" in named
    assert "live-x" in named
    # an idempotent teardown (serving adapter already gone) still confirms — no false no-op
    idempotent = render.undeployed({"run_id": "flash-1", "deleted_endpoints": []})
    assert "torn down flash-1" in idempotent


def test_export_card_reflects_requested_privacy(monkeypatch, capsys) -> None:
    """The control-plane export result carries no `private` field, so the styled card must read the
    privacy from the request (exports are private unless `--public`) instead of always saying
    `public` — otherwise a private adapter is misreported as world-readable."""
    import argparse

    import flash.client.runtime_secrets as runtime_secrets

    class _ExportClient:
        def export(self, adapter_id, *, repository, hf_token, step, private):
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
    args = argparse.Namespace(
        adapter_id="flash-1", repository="acme/x", step=None, public=False, api_key=None
    )
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
    assert cli.main(["runs"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("✗")  # ✗ leads the themed line
    assert "error:" in err
    assert "bad [environment] id" in err

    monkeypatch.setenv("FLASH_STYLE", "0")
    assert cli.main(["runs"]) == 1
    assert capsys.readouterr().err.startswith("error:")  # machine path unchanged


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
