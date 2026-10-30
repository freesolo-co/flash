"""Scaffold a local custom algorithm without importing GPU or provider libraries."""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from flash._internal.channel import CLI_NAME
from flash.core.spec import require_project_id


def cmd_custom_init(args):
    destination = Path(args.path)
    source = files("flash.custom").joinpath("starter.py").read_text()
    outputs = {destination: source}
    project = getattr(args, "project", None)
    if project is not None:
        project = require_project_id(project)
        config = destination.with_suffix(".toml")
        if config == destination:
            raise ValueError("use a Python file path, such as algorithm.py, for the algorithm")
        outputs[config] = (
            f'project = {json.dumps(project)}\nmodel = "Qwen/Qwen3.5-9B"\n'
            'algorithm = "custom"\n\n[custom]\n'
            f"file = {json.dumps(destination.name)}\n\n"
            "[train]\nmax_steps = 10\nsave_every = 2\n"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, content in outputs.items():
            with path.open("x") as handle:
                created.append(path)
                handle.write(content)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    if project is not None:
        print(f"Created {destination} and {config}. Run: {CLI_NAME} train {config}")
    else:
        print(
            f'Created {destination}. Set algorithm = "custom" and [custom] file = "{destination.name}" '
            f"in your TOML, then run: {CLI_NAME} train run.toml"
        )
    print("Async by default. For blocking calls, use def main(client) and remove await.")


def add_custom_commands(sub: argparse._SubParsersAction):
    custom = sub.add_parser("custom", help="write a local algorithm for managed Flash training")
    commands = custom.add_subparsers(dest="custom_cmd", required=True)
    init = commands.add_parser("init", help="write an editable async DPO training script")
    init.add_argument("path", nargs="?", default="algorithm.py", help="new script path")
    init.add_argument(
        "--project", help="project UUID; also create a training TOML beside the script"
    )
    init.set_defaults(func=cmd_custom_init)
