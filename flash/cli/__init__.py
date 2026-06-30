"""CLI for the managed Flash service."""

from __future__ import annotations

import argparse
import difflib
import re
import shlex
import sys
from typing import NoReturn

from flash import __version__
from flash._channel import CLI_NAME
from flash._logging import configure_logging, get_logger
from flash._update_check import emit_update_notice, maybe_start_update_check
from flash.cli import render

# Command handlers + the patched client surface live in submodules; re-export them so
# `flash.cli` stays the single public import surface (and so monkeypatching
# `flash.cli.commands` reaches the bare globals the handlers read).
from flash.cli.commands import (  # noqa: F401
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
    cmd_export,
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
from flash.cli.envpush import cmd_env_delete, cmd_env_pull, cmd_env_push

logger = get_logger("flash.cli")

# Themed `flash --help` catalog. Groups are ordered along the training workflow; each row's
# summary is the short one-liner the themed grid shows (the verbose per-command text stays on
# every subparser's own `help=` / `<cmd> --help`). test_cli_help.py asserts these rows stay in
# lockstep with the registered subcommands, so a newly added command can't go silently unlisted.
_HELP_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "getting started",
        [
            ("login", "log in with your freesolo API key"),
            ("whoami", "show the identity behind your stored key"),
            ("version", "print the flash version"),
        ],
    ),
    (
        "catalog",
        [
            ("models", "list supported base models"),
            ("gpus", "list managed GPU classes with estimated $/hr"),
        ],
    ),
    (
        "environments",
        [
            ("env setup", "scaffold a starter Freesolo environment"),
            ("env list", "list local environment sources"),
            ("env push", "upload a local environment"),
            ("env pull", "download a published environment or file"),
            ("env delete", "delete a published environment"),
        ],
    ),
    (
        "training",
        [
            ("train", "submit a managed run from a TOML config"),
            ("status", "show a run's status, logs, or live follow"),
            ("runs", "list runs with their state and cost"),
            ("checkpoints", "list a run's deployable RL checkpoints"),
            ("cancel", "cancel a running job"),
        ],
    ),
    (
        "serving & export",
        [
            ("deploy", "deploy a run's adapter to an endpoint"),
            ("chat", "chat with a deployed adapter"),
            ("deployments", "list active serving deployments"),
            ("undeploy", "tear down a run's endpoint"),
            ("export", "export an adapter to your HuggingFace repo"),
        ],
    ),
]

_HELP_OPTIONS: list[tuple[str, str]] = [
    ("-h, --help", "show this help and exit"),
    ("-V, --version", "print the flash version"),
    ("--debug", "show full tracebacks on error"),
    ("-v, --verbose", "increase log verbosity (-v info, -vv debug)"),
]


def _friendly_message(message: str) -> str:
    """Shorten argparse's verbose ``invalid choice: 'x' (choose from a, b, c, ...)`` into a concise
    ``unknown command 'x' (did you mean 'y'?)`` — the single closest match instead of dumping the
    whole list. Other messages pass through untouched. Styled path only; the machine path keeps
    argparse's exact text (scripts and the error tests match on the literal `invalid choice`)."""
    m = re.search(r"invalid choice: '([^']*)'(?: \(choose from (.*)\))?", message)
    if not m:
        return message
    bad, raw_choices = m.group(1), m.group(2) or ""
    choices = [c.strip().strip("'\"") for c in raw_choices.split(",") if c.strip()]
    near = difflib.get_close_matches(bad, choices, n=1)
    return f"unknown command '{bad}'" + (f" (did you mean '{near[0]}'?)" if near else "")


class _ThemedParser(argparse.ArgumentParser):
    """Base parser whose usage errors match the rest of the themed CLI.

    argparse handles usage errors itself inside `parse_args()` — a missing required argument, an
    unknown flag, a bad subcommand choice, a bad `type=` conversion — by calling `error()`, which
    prints a raw `usage: ...` block plus `prog: error: msg` and exits 2, long before main()'s
    catch-all handler ever runs. So those errors never picked up the red ✗ idiom the rest of the
    CLI uses (this is the unstyled blob you get from a bare `flash` or a typo'd flag). We override
    `error()` to emit `render.error()` + a dimmed `--help` pointer on a styled terminal, while the
    machine path keeps argparse's exact text and exit code 2 that scripts, the agent contract, and
    the error tests match on.
    """

    def error(self, message: str) -> NoReturn:
        if not render.styled():
            super().error(message)  # argparse's raw usage + `prog: error: msg`, then exit 2
        # themed twin: the red ✗ error line (same idiom as main()'s catch-all and `flash login`),
        # then a dimmed pointer at this parser's own --help instead of the raw usage block. An
        # "invalid choice" becomes a short "did you mean" suggestion (see _friendly_message).
        print(render.error(_friendly_message(message)), file=sys.stderr)
        # dimmed pointer at THIS parser's own --help (argparse sets prog per parser: `flash --help`
        # for the root, `flash <cmd> --help` for a subcommand) instead of the raw usage block.
        print(render.arrow(f"run `{self.prog} --help` for usage"), file=sys.stderr)
        self.exit(2)  # keep argparse's usage-error exit code


class _FlashParser(_ThemedParser):
    """Root parser that renders the themed help page on a styled stdout.

    Every parser (root + subcommands) inherits `_ThemedParser`'s themed `error()`, so a usage
    error on any command gets the red ✗ idiom on a TTY. Only the root parser overrides
    `format_help`, so `flash <cmd> --help` keeps argparse's standard layout. Piped or scripted
    `flash --help` also falls back to argparse, so existing greps stay byte-for-byte. Overriding
    `format_help` (not the help action) preserves argparse's `--help` exit-0 flow.
    """

    def format_help(self) -> str:
        if not render.styled():
            return super().format_help()
        usage = f"{CLI_NAME} [--debug] [-v] <command> [args]"
        footers = [
            f"new here? run `{CLI_NAME} login`, then `{CLI_NAME} env setup`",
            f"train after publishing: `{CLI_NAME} env push --name my-env .`, "
            f"then `{CLI_NAME} train configs/rl.toml`",
            f"any command in depth: `{CLI_NAME} <command> --help`",
            "docs: https://freesolo.co/docs",
        ]
        return render.help_page(
            "managed LoRA post-training", usage, _HELP_GROUPS, _HELP_OPTIONS, footers
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the fully-configured root parser. Extracted from main() so tests can introspect the
    registered subcommands and keep the themed help catalog (_HELP_GROUPS) in lockstep."""
    parser = _FlashParser(prog=CLI_NAME, description="Managed LoRA post-training")
    parser.add_argument("-V", "--version", action="version", version=f"{CLI_NAME} {__version__}")
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
    # subparsers theme their usage errors (parser_class=_ThemedParser) but not their help, so
    # `flash <cmd> --help` keeps the standard layout; only the root parser themes its help (see
    # _FlashParser). Nested `env` subcommands inherit _ThemedParser automatically (the env parser
    # is itself a _ThemedParser, so its add_subparsers defaults to the same class).
    sub = parser.add_subparsers(dest="cmd", required=True, parser_class=_ThemedParser)

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

    gpus = sub.add_parser("gpus", help="list managed GPU classes with estimated $/hr")
    gpus.set_defaults(func=cmd_gpus)

    env = sub.add_parser("env", help="manage Freesolo environments")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    setup = env_sub.add_parser("setup", help="create a starter Freesolo environment scaffold")
    setup.set_defaults(func=cmd_env_setup)

    env_list = env_sub.add_parser("list", help="list local environment sources")
    env_list.set_defaults(func=cmd_env_list)

    env_push = env_sub.add_parser("push", help="upload a local Freesolo environment")
    env_push.add_argument(
        "--name",
        required=True,
        help="Freesolo environment name to publish or update",
    )
    env_push.add_argument("path", nargs="?", default=".")
    env_push.set_defaults(func=cmd_env_push)

    env_pull = env_sub.add_parser(
        "pull", help="download a published Freesolo environment (or one file from it)"
    )
    env_pull.add_argument(
        "env_id",
        help='the Freesolo environment id: a managed slug "your-name/your-env", a '
        '"github:owner/repo@ref:path" ref, or a github.com URL',
    )
    env_pull.add_argument(
        "path",
        nargs="?",
        help="optional single file within the env to fetch, e.g. datasets/train.jsonl",
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
        "env_id", help="the Freesolo environment id to delete, e.g. you/your-env"
    )
    env_delete.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the confirmation prompt",
    )
    env_delete.set_defaults(func=cmd_env_delete)

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

    deploy = sub.add_parser("deploy", help="deploy a run's adapter to a serving endpoint")
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

    export = sub.add_parser("export", help="export a trained adapter to your own HuggingFace repo")
    export.add_argument(
        "--adapter-id",
        dest="adapter_id",
        required=True,
        help="the Freesolo adapter id (the run id) to export",
    )
    export.add_argument(
        "--repository",
        required=True,
        help="destination HuggingFace repo 'owner/name' (created if it doesn't exist)",
    )
    export.add_argument(
        "--api-key",
        help="HuggingFace token with write access to --repository "
        "(default: HF_TOKEN from your shell or a local .env / .env.local)",
    )
    export.add_argument(
        "--step",
        type=int,
        default=None,
        help="export a specific intermediate checkpoint (see `flash checkpoints <adapter-id>`) "
        "instead of the run's final adapter; works even for a run cancelled mid-RL",
    )
    export.add_argument(
        "--public",
        action="store_true",
        help="create the destination repo as public (default: private)",
    )
    export.set_defaults(func=cmd_export)

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

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbosity=getattr(args, "verbose", 0))
    debug = getattr(args, "debug", False)
    update_check = maybe_start_update_check()
    try:
        return args.func(args)
    except _USER_ERRORS as exc:
        if debug:
            raise
        # themed red ✗ on a styled terminal (same idiom as `flash login` failures); the machine
        # path keeps the plain `error: {exc}` prefix that scripts and tests match on.
        if render.styled():
            print(render.error(str(exc)), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(render.note("aborted") if render.styled() else "aborted", file=sys.stderr)
        return 130
    except Exception as exc:
        # anything outside _USER_ERRORS — e.g. a read-only ~/.flash when `flash login` saves the
        # key, or a non-writable cwd on `flash env setup` — would otherwise dump a raw Python
        # traceback, the least themed output there is. On a styled terminal show the red ✗ idiom
        # + a `--debug` pointer instead; the machine path (and --debug) keep the full traceback,
        # which is the bug signal CI and `--debug` bug reports rely on.
        if debug or not render.styled():
            raise
        # point at the exact command to re-run, copy-pasteable. --debug is a root-level flag, so it
        # must come BEFORE the subcommand (argparse rejects `flash runs --debug`); place it right
        # after the program name. raw_args never contains --debug here — that path re-raises above.
        cmd = " ".join([CLI_NAME, "--debug", *(shlex.quote(a) for a in raw_args)])
        print(render.error(str(exc) or exc.__class__.__name__), file=sys.stderr)
        print(render.arrow(f"run `{cmd}` for the full traceback"), file=sys.stderr)
        return 1
    finally:
        emit_update_notice(update_check)
