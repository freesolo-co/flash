"""CLI for the managed Flash service.

Every run-lifecycle command is a thin HTTP call to the Flash control plane —
users authenticate with their freesolo API key (`flash login` verifies it against
the freesolo backend), never with provider credentials. Config parsing/validation
and `--dry-run` stay fully local.
"""

from __future__ import annotations

import argparse
import sys

from flash import __version__
from flash._logging import configure_logging, get_logger
from flash._update_check import emit_update_notice, maybe_start_update_check

# Command handlers + the patched client surface live in submodules; re-export them so
# `flash.cli.main` stays the single public import surface (and so monkeypatching
# `flash.cli.main.commands` reaches the bare globals the handlers read).
from flash.cli.main.commands import (  # noqa: F401
    _CLI_DONE_STATES,
    _OK_STATES,
    _STARTER_ENV_PY,
    _USER_ERRORS,
    _follow_run,
    _poll_logs,
    client_from_config,
    cmd_cancel,
    cmd_chat,
    cmd_checkpoints,
    cmd_deploy,
    cmd_deployments,
    cmd_env_list,
    cmd_env_setup,
    cmd_gpus,
    cmd_login,
    cmd_models,
    cmd_runs,
    cmd_status,
    cmd_train,
    cmd_undeploy,
    cmd_version,
    cmd_whoami,
    verify_freesolo_key,
)
from flash.cli.main.envpush import cmd_env_install, cmd_env_push

logger = get_logger("flash.cli.main")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flash", description="Managed LoRA post-training")
    parser.add_argument("-V", "--version", action="version", version=f"flash {__version__}")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show full tracebacks on error",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (-v for info, -vv for debug)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    version = sub.add_parser("version", help="print the Flash version")
    version.set_defaults(func=cmd_version)

    login = sub.add_parser(
        "login",
        help="log in with your freesolo API key (create one at https://freesolo.co/sign-in)",
    )
    login.add_argument(
        "--api-key",
        help="your freesolo API key (default: FREESOLO_API_KEY); create it at "
        "https://freesolo.co/sign-in",
    )
    login.add_argument(
        "--freesolo-url",
        dest="freesolo_url",
        help="freesolo backend base URL (default: FREESOLO_BASE_URL or https://api.freesolo.co)",
    )
    login.add_argument(
        "--api-url", help="flash control-plane URL for training calls (default: FLASH_API_URL)"
    )
    login.set_defaults(func=cmd_login)

    whoami = sub.add_parser("whoami", help="show the identity behind your stored key")
    whoami.set_defaults(func=cmd_whoami)

    models = sub.add_parser("models", help="list supported base models")
    models.set_defaults(func=cmd_models)

    gpus = sub.add_parser("gpus", help="list managed GPU classes with live $/hr")
    gpus.set_defaults(func=cmd_gpus)

    env = sub.add_parser("env", help="manage Freesolo environments")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    setup = env_sub.add_parser("setup", help="create a starter Freesolo environment scaffold")
    setup.set_defaults(func=cmd_env_setup)

    env_list = env_sub.add_parser("list", help="list installed + local environments")
    env_list.set_defaults(func=cmd_env_list)

    env_install = env_sub.add_parser("install", help="record a Freesolo environment")
    env_install.add_argument("env_id", help="the Freesolo environment id to record")
    env_install.set_defaults(func=cmd_env_install)

    env_push = env_sub.add_parser("push", help="upload a local Freesolo environment")
    env_push.add_argument(
        "--name",
        required=True,
        help="Freesolo environment name to publish or update",
    )
    env_push.add_argument("path", nargs="?", default=".")
    env_push.set_defaults(func=cmd_env_push)

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

    status = sub.add_parser("status", help="show a run's status, logs, or follow logs")
    status.add_argument("run_id")
    status.add_argument(
        "--logs",
        action="store_true",
        help="print current logs before status — the orchestrator log plus the train-subprocess "
        "stdout + traceback (console_/error_<phase>.txt) fetched from the run's HF artifact repo",
    )
    status.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="stream logs until the run ends, then print final status",
    )
    status.set_defaults(func=cmd_status)

    runs = sub.add_parser("runs", help="list runs and their state/cost")
    runs.set_defaults(func=cmd_runs)

    cancel = sub.add_parser("cancel", help="cancel a run")
    cancel.add_argument("run_id")
    cancel.set_defaults(func=cmd_cancel)

    checkpoints = sub.add_parser(
        "checkpoints", help="list a run's deployable per-step RL checkpoints"
    )
    checkpoints.add_argument("run_id")
    checkpoints.set_defaults(func=cmd_checkpoints)

    deploy = sub.add_parser("deploy")
    deploy.add_argument("run_id")
    deploy.add_argument("--dry-run", action="store_true")
    deploy.add_argument(
        "--step",
        type=int,
        default=None,
        help="deploy a specific intermediate checkpoint (see `flash checkpoints <run_id>`) "
        "instead of the run's final adapter; works even for a run cancelled mid-RL",
    )
    deploy.set_defaults(func=cmd_deploy)

    undeploy = sub.add_parser("undeploy", help="tear down a run's serving endpoint")
    undeploy.add_argument("run_id")
    undeploy.set_defaults(func=cmd_undeploy)

    deployments = sub.add_parser("deployments", help="list active serving deployments")
    deployments.set_defaults(func=cmd_deployments)

    chat = sub.add_parser("chat", help="chat with a deployed adapter")
    chat.add_argument("run_id")
    chat.add_argument("-m", "--message", required=True)
    chat.add_argument("--max-tokens", type=int, default=2048)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.set_defaults(func=cmd_chat)

    # The control plane is operator-only and run as a separate one-off service via the
    # `flash-server` console script (flash.server.__main__:main), not a `flash` subcommand.

    args = parser.parse_args(argv)
    configure_logging(verbosity=getattr(args, "verbose", 0))
    debug = getattr(args, "debug", False)
    # Kick off a once-a-day PyPI version check in the background; the "new release available"
    # notice (if any) prints to stderr after the command output (see emit_update_notice).
    update_check = maybe_start_update_check()
    try:
        return args.func(args)
    except _USER_ERRORS as exc:
        if debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        return 130
    finally:
        emit_update_notice(update_check)
