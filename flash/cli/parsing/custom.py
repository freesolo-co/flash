"""Scaffold a local custom algorithm without importing GPU or provider libraries."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path


def cmd_custom_init(args):
    destination = Path(args.path)
    source = files("flash.custom").joinpath("starter.py").read_text()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        handle.write(source)
    print(f"Created {destination}. Edit its data and loss, then run: python {destination}")


def add_custom_commands(sub: argparse._SubParsersAction):
    custom = sub.add_parser("custom", help="build a local algorithm controlling a dedicated GPU")
    commands = custom.add_subparsers(dest="custom_cmd", required=True)
    init = commands.add_parser("init", help="write an editable DPO training script")
    init.add_argument("path", nargs="?", default="algorithm.py", help="new script path")
    init.set_defaults(func=cmd_custom_init)
