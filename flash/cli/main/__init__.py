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
    cmd_attach,
    cmd_cancel,
    cmd_chat,
    cmd_cost,
    cmd_deploy,
    cmd_deployments,
    cmd_env_init,
    cmd_env_list,
    cmd_gpus,
    cmd_lab_setup,
    cmd_login,
    cmd_logs,
    cmd_models,
    cmd_ps,
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

    login = sub.add_parser("login", help="log in with your freesolo API key (verified by freesolo)")
    login.add_argument(
        "--api-key",
        help="your freesolo API key (default: FREESOLO_API_KEY); created in the dashboard",
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

    lab = sub.add_parser("lab", help="local authoring scaffolds")
    lab_sub = lab.add_subparsers(dest="lab_cmd", required=True)
    setup = lab_sub.add_parser("setup", help="scaffold environments/ + configs/ in the cwd")
    setup.set_defaults(func=cmd_lab_setup)

    models = sub.add_parser("models", help="list supported base models")
    models.set_defaults(func=cmd_models)

    gpus = sub.add_parser("gpus", help="list managed GPU classes with live $/hr")
    gpus.set_defaults(func=cmd_gpus)

    env = sub.add_parser("env", help="manage Freesolo environments")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    init = env_sub.add_parser("init", help="scaffold a new local Freesolo environment")
    init.add_argument("name")
    init.set_defaults(func=cmd_env_init)

    env_list = env_sub.add_parser("list", help="list installed + local environments")
    env_list.set_defaults(func=cmd_env_list)

    env_install = env_sub.add_parser("install", help="record a GitHub Freesolo environment")
    env_install.add_argument("env_id", help="the GitHub environment id to record")
    env_install.set_defaults(func=cmd_env_install)

    env_push = env_sub.add_parser(
        "push", help="upload a local Freesolo env to GitHub; prints its env id"
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

    status = sub.add_parser("status", help="show a run's full status JSON")
    status.add_argument("run_id")
    status.set_defaults(func=cmd_status)

    attach = sub.add_parser(
        "attach", help="follow a running job's logs to completion (resumable any time)"
    )
    attach.add_argument("run_id")
    attach.set_defaults(func=cmd_attach)

    ps = sub.add_parser("ps", help="list runs and their state/cost")
    ps.set_defaults(func=cmd_ps)

    cost = sub.add_parser("cost", help="show a run's accrued cost (USD)")
    cost.add_argument("run_id")
    cost.set_defaults(func=cmd_cost)

    cancel = sub.add_parser("cancel", help="cancel a run (best-effort)")
    cancel.add_argument("run_id")
    cancel.set_defaults(func=cmd_cancel)

    logs = sub.add_parser("logs")
    logs.add_argument("run_id")
    logs.add_argument("-f", "--follow", action="store_true", help="stream new log lines")
    logs.set_defaults(func=cmd_logs)

    deploy = sub.add_parser("deploy")
    deploy.add_argument("run_id")
    deploy.add_argument(
        "--mode",
        choices=["dev", "always-on"],
        default="dev",
        help="dev: scale-to-zero, cold start after idle, $0 when unused (default). "
        "always-on: one warm worker 24/7, no cold starts, continuous billing.",
    )
    deploy.add_argument(
        "--idle-timeout",
        type=int,
        default=300,
        help="dev mode: seconds of inactivity before the worker scales to zero (default 300)",
    )
    deploy.add_argument("--dry-run", action="store_true")
    deploy.set_defaults(func=cmd_deploy)

    undeploy = sub.add_parser("undeploy", help="tear down a run's serving endpoint")
    undeploy.add_argument("run_id")
    undeploy.set_defaults(func=cmd_undeploy)

    deployments = sub.add_parser("deployments", help="list active serving deployments")
    deployments.set_defaults(func=cmd_deployments)

    chat = sub.add_parser("chat", help="chat with a deployed adapter")
    chat.add_argument("run_id")
    chat.add_argument("-m", "--message", required=True)
    chat.add_argument("--max-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.set_defaults(func=cmd_chat)

    # The control plane is operator-only and run as a separate one-off service via the
    # `flash-server` console script (flash.server.__main__:main), not a `flash` subcommand.

    args = parser.parse_args(argv)
    configure_logging(verbosity=getattr(args, "verbose", 0))
    debug = getattr(args, "debug", False)
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
