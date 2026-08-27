"""Environment CLI parser registration."""

from __future__ import annotations

import argparse

from flash.cli.commands.env.ops.list import cmd_env_list
from flash.cli.commands.env.ops.push import cmd_env_delete, cmd_env_pull, cmd_env_push
from flash.cli.commands.env.ops.setup import cmd_env_setup
from flash.cli.commands.env.testing.eval import (
    _MAX_CONCURRENCY,
    bounded_concurrency,
    cmd_env_eval,
    finite_float,
    positive_int,
)
from flash.cli.commands.env.testing.test import cmd_env_test
from flash.core.catalog import ALGORITHMS


def _add_env_commands(sub: argparse._SubParsersAction) -> None:
    """`env setup/list/test/eval/push/pull/delete`.

    Split one level further than the other families because `env` carries seven subcommands; the
    call order here is the order they appear in `flash env --help`.
    """
    env = sub.add_parser("env", help="manage Freesolo environments")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    _add_env_setup_command(env_sub)
    _add_env_test_commands(env_sub)
    _add_env_eval_command(env_sub)
    _add_env_publish_commands(env_sub)


def _add_env_setup_command(env_sub: argparse._SubParsersAction) -> None:
    """`env setup`: the scaffold generator and its turn-mode / reasoning switches."""
    setup = env_sub.add_parser("setup", help="create a starter Freesolo environment scaffold")
    setup_mode = setup.add_mutually_exclusive_group()
    setup_mode.add_argument(
        "--single-turn",
        dest="turn_mode",
        action="store_const",
        const="single",
        help="scaffold a single-turn environment (prompt -> one response). This is the default.",
    )
    setup_mode.add_argument(
        "--multi-turn",
        dest="turn_mode",
        action="store_const",
        const="multi",
        help="scaffold a multi-turn environment (bounded episode with step_episode / score_episode).",
    )
    setup_reason = setup.add_mutually_exclusive_group()
    setup_reason.add_argument(
        "--reasoning",
        dest="reasoning",
        action="store_const",
        const=True,
        help="scaffold configs with reasoning enabled (thinking = true, with a raised token budget).",
    )
    setup_reason.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_const",
        const=False,
        help="scaffold configs without reasoning. This is the default.",
    )
    setup.add_argument(
        "--project",
        metavar="PROJECT_UUID",
        help=(
            "Freesolo project UUID for all generated configs and environment publication; "
            "required with --yes, a redirected stdin, or any other noninteractive run, where "
            "there is no prompt to choose one"
        ),
    )
    setup.add_argument(
        "-y",
        "--yes",
        dest="yes",
        action="store_true",
        help="accept defaults without prompting (single-turn, no reasoning).",
    )
    # turn_mode / reasoning default to None (unset) so the scaffold can tell an explicit flag
    # apart from "not chosen yet" and, on an interactive terminal, ask instead of assuming.
    setup.set_defaults(func=cmd_env_setup, turn_mode=None, reasoning=None, yes=False)


def _add_env_test_commands(env_sub: argparse._SubParsersAction) -> None:
    """`env list` and `env test`: inspect published environments and local sources."""
    env_list = env_sub.add_parser(
        "list", help="list published environments and local environment sources"
    )
    env_list.set_defaults(func=cmd_env_list)

    env_test = env_sub.add_parser(
        "test",
        help="locally validate an environment by driving a few offline episodes before pushing",
        description=(
            "locally validate an environment by driving a few offline episodes before pushing"
        ),
    )
    env_test.add_argument(
        "path",
        nargs="?",
        default=".",
        help="local environment directory or environment.py path",
    )
    env_test.add_argument(
        "--split",
        default=None,
        help=(
            "dataset split to validate, matching [environment.params] split (default: train). "
            "use the split the run actually trains on, e.g. --split train_sft"
        ),
    )
    env_test.add_argument(
        "--algorithm",
        default="grpo",
        choices=ALGORITHMS,
        help=(
            "algorithm the environment is for, matching [train] algorithm (default: grpo). "
            "only grpo trains from the environment reward, so sft and opd skip the reward gate"
        ),
    )
    env_test.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "extra load_environment() kwarg, as in [environment.params] (repeatable). "
            'values parse as TOML scalars, so 1/true/"x" keep their types'
        ),
    )
    env_test.set_defaults(func=cmd_env_test)


def _add_env_eval_command(env_sub: argparse._SubParsersAction) -> None:
    """`env eval`: score a deployed model against its environment's held-out suites."""
    env_eval = env_sub.add_parser(
        "eval",
        help="score a deployed model against its own environment's held-out suites",
        description=(
            "score a deployed model against the held-out suites of the published environment "
            "its run trains on"
        ),
    )
    env_eval.add_argument(
        "target",
        metavar="TARGET",
        help="permanent deployed checkpoint id: RUN_ID/final or RUN_ID/step-N",
    )
    # the same two knobs `env test` exposes, and for the same reason: an env whose
    # `load_environment()` requires a difficulty or reads a non-default split cannot be evaluated at
    # all without them, and a held-out suite scored against a differently-configured environment
    # than the run trains on is not measuring the run.
    env_eval.add_argument(
        "--split",
        default=None,
        help=(
            "dataset split to evaluate against, matching [environment.params] split "
            "(default: train)"
        ),
    )
    env_eval.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "extra load_environment() kwarg, as in [environment.params] (repeatable). "
            'values parse as TOML scalars, so 1/true/"x" keep their types'
        ),
    )
    env_eval.add_argument("--suite", help="run only the named evaluation suite")
    env_eval.add_argument(
        "--max-cases",
        type=positive_int,
        metavar="N",
        help="run at most N cases from each selected suite",
    )
    env_eval.add_argument("--temperature", type=finite_float, default=0.0)
    env_eval.add_argument("--max-tokens", type=positive_int, default=512, metavar="N")
    env_eval.add_argument(
        "--concurrency",
        type=bounded_concurrency,
        default=1,
        metavar="N",
        help=f"parallel model requests (1-{_MAX_CONCURRENCY})",
    )
    env_eval.add_argument(
        "--no-upload",
        dest="upload",
        action="store_false",
        help="score locally without recording the results in the dashboard",
    )
    env_eval.add_argument(
        "--project",
        metavar="PROJECT_ID",
        help=(
            "project id to record results under (default: the project the evaluated run belongs to)"
        ),
    )
    env_eval.set_defaults(func=cmd_env_eval)


def _add_env_publish_commands(env_sub: argparse._SubParsersAction) -> None:
    """`env push/pull/delete`: the hub round trip."""
    env_push = env_sub.add_parser("push", help="upload a local Freesolo environment")
    env_push.add_argument(
        "--name",
        required=True,
        help="Freesolo environment name to publish or update",
    )
    env_push.add_argument(
        "--project",
        required=True,
        metavar="PROJECT_UUID",
        help="Freesolo project UUID that owns the environment",
    )
    env_push.add_argument("path", nargs="?", default=".")
    env_push.set_defaults(func=cmd_env_push)

    env_pull = env_sub.add_parser(
        "pull", help="download a published Freesolo environment (or one file from it)"
    )
    env_pull.add_argument(
        "env_id",
        help='the managed Freesolo environment slug "your-org/your-project/your-env"',
    )
    env_pull.add_argument(
        "path",
        nargs="?",
        help="optional single file within the env to fetch, e.g. dataset/train.jsonl",
    )
    env_pull.add_argument(
        "-o",
        "--output",
        help="output file (with PATH) or directory (whole env); defaults to the env/file name",
    )
    env_pull.add_argument("-f", "--force", action="store_true", help="overwrite existing output")
    env_pull.set_defaults(func=cmd_env_pull)

    env_delete = env_sub.add_parser("delete", help="delete a published Freesolo environment")
    env_delete.add_argument(
        "env_id", help="the Freesolo environment id to delete, e.g. your-org/your-project/your-env"
    )
    env_delete.add_argument(
        "--project",
        required=True,
        metavar="PROJECT_UUID",
        help="Freesolo project UUID that owns the environment",
    )
    env_delete.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the confirmation prompt",
    )
    env_delete.set_defaults(func=cmd_env_delete)
