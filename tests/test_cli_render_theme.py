"""The themed CLI output is TTY-gated: piped/scripted output stays byte-for-byte machine
readable, while ``FLASH_STYLE=1`` forces the human theme. These tests pin both sides of that
gate so a future tweak to the theme can't silently change the machine contract that `jq`,
scripts, and the trainer agent depend on.
"""

from __future__ import annotations

import json

import pytest

from flash import __version__
from flash.cli import main as cli
from flash.cli.main import render


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
        render.env_list([], []),
        render.empty("runs", "0 runs", "no runs yet — submit one with `flash train`"),
        render.submitted("flash-xyz"),
        render.run_status({"run_id": "r", "state": "failed", "spec": {}, "error": "boom — bad"}),
    ]
    for out in outputs:
        out.encode("ascii")  # raises if any non-ASCII slipped through


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
