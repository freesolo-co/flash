"""Training and run-management CLI parser registration."""

from __future__ import annotations

import argparse

from flash.cli.commands.ops.runs import cmd_cancel, cmd_checkpoints, cmd_log, cmd_runs, cmd_status
from flash.cli.commands.ops.train import cmd_train


def _gpu_count_override(value: str) -> str:
    """Convert a GPU count to the canonical config override."""
    try:
        return f"gpu.count={int(value)}"
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of gpus, got {value!r}") from None


def _add_train_commands(sub: argparse._SubParsersAction) -> None:
    """`train`: config composition, overrides, cost preflight."""
    train = sub.add_parser("train", help="submit a managed training run from a TOML config")
    train.add_argument("config")
    train.add_argument(
        "--config",
        dest="extra_configs",
        action="append",
        default=[],
        help="additional TOML to deep-merge (config composition); repeatable",
    )
    train.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="override a config value; repeatable",
    )
    train.add_argument(
        "--gpus",
        dest="overrides",
        action="append",
        type=_gpu_count_override,
        metavar="N",
        help=(
            "optional card ceiling; sets [gpu] count (1-8). omit it with an unpinned gpu type to "
            "auto-size the smallest fitting geometry; an authored value pins the ceiling, and only "
            "(1, 2, 4, 8) are ever provisioned"
        ),
    )
    train.add_argument("--dry-run", action="store_true")
    train.add_argument(
        "--cost",
        action="store_true",
        help="print the pre-flight USD cost for the config and exit (no submit)",
    )
    train.add_argument(
        "--background",
        action="store_true",
        help="submit and return immediately instead of following logs",
    )
    train.set_defaults(func=cmd_train)


def _add_runs_commands(sub: argparse._SubParsersAction) -> None:
    """`runs list/status/log/cancel/checkpoint`."""
    runs = sub.add_parser("runs", help="manage training runs")
    runs.set_defaults(func=cmd_runs)  # hidden bare `flash runs` shim for deployed agents
    runs_sub = runs.add_subparsers(dest="runs_cmd", required=False)

    runs_list = runs_sub.add_parser("list", help="list runs and their state/cost")
    runs_list.set_defaults(func=cmd_runs)

    runs_status = runs_sub.add_parser("status", help="show a run's current status")
    runs_status.add_argument("run_id")
    runs_status.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="poll status until the run ends without replaying logs",
    )
    runs_status.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable run status",
    )
    runs_status.set_defaults(func=cmd_status)

    runs_log = runs_sub.add_parser(
        "log",
        help="print a run's full logs, including worker console/error artifacts",
    )
    runs_log.add_argument("run_id")
    runs_log.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="stream logs until the run reaches a terminal state",
    )
    runs_log.set_defaults(func=cmd_log)

    runs_cancel = runs_sub.add_parser("cancel", help="cancel a run")
    runs_cancel.add_argument("run_id")
    runs_cancel.set_defaults(func=cmd_cancel)

    runs_checkpoint = runs_sub.add_parser(
        "checkpoint", help="list a run's deployable per-step RL checkpoints"
    )
    runs_checkpoint.add_argument("run_id")
    runs_checkpoint.set_defaults(func=cmd_checkpoints)
