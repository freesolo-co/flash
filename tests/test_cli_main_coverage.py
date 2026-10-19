"""Hermetic execution coverage for the module CLI entrypoint."""

from __future__ import annotations

import runpy

import pytest

import flash.cli.parsing.main as cli


def test_cli_module_exits_with_main_return_code(monkeypatch) -> None:
    """Executing `python -m flash.cli` must pass the CLI result directly to SystemExit."""
    monkeypatch.setattr(cli, "main", lambda: 7)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("flash.cli.__main__", run_name="__main__")

    assert exc_info.value.code == 7
