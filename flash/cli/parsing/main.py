"""CLI for the managed Flash service."""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import NoReturn

from flash import __version__
from flash._internal.channel import BRAND_NAME, CLI_NAME
from flash._internal.logging import configure_logging
from flash.cli.commands.env.ops.push import cmd_env_delete, cmd_env_push
from flash.cli.commands.env.ops.setup import cmd_env_setup
from flash.cli.commands.env.testing.eval import cmd_env_eval
from flash.cli.commands.ops.account import cmd_projects_create
from flash.cli.commands.ops.deploy import cmd_deploy, cmd_export, cmd_undeploy
from flash.cli.commands.ops.runs import cmd_cancel
from flash.cli.commands.ops.traces import cmd_traces_export
from flash.cli.commands.ops.train import cmd_train
from flash.cli.parsing import account, env, models, traces, training
from flash.cli.parsing.errors import friendly_error
from flash.cli.parsing.serve_parser import _add_serve_commands
from flash.cli.ui import render
from flash.client import ClientError
from flash.client.config import shadowed_login_warning
from flash.schema import ConfigError

# argv flags whose VALUE is a credential, so it must never be echoed back to the user. both the
# freesolo key (`flash login`) and the HuggingFace token (`flash models export`) use this name.
_CREDENTIAL_FLAGS = frozenset({"--api-key"})
_USER_ERRORS = (ConfigError, ClientError, FileNotFoundError, ValueError)

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
            ("projects create", "create a Freesolo project"),
            ("projects list", "list Freesolo projects and UUIDs"),
            ("version", "print the flash version"),
        ],
    ),
    (
        "catalog",
        [
            ("models list", "list supported base models"),
            ("gpus", "list managed GPU classes with estimated $/hr"),
        ],
    ),
    (
        "environments",
        [
            ("env setup", "scaffold a starter Freesolo environment"),
            ("env list", "list local environment sources"),
            ("env test", "validate a local environment offline"),
            ("env eval", "score a deployed model against its published environment's suites"),
            ("env push", "upload a local environment"),
            ("env pull", "download a published environment or file"),
            ("env delete", "delete a published environment"),
            ("traces export", "turn a project's traces into a dataset"),
        ],
    ),
    (
        "training",
        [
            ("train", "submit a managed run from a TOML config"),
            ("runs list", "list runs with their state and cost"),
            ("runs status", "show a run's current status"),
            ("runs log", "print or follow a run's logs"),
            ("runs checkpoint", "list a run's deployable RL checkpoints"),
            ("runs cancel", "cancel a running job"),
        ],
    ),
    (
        "serving & export",
        [
            ("models deploy", "deploy a run's adapter to an endpoint"),
            ("models chat", "chat with a deployed adapter"),
            ("models deployments", "list active serving deployments"),
            ("models undeploy", "tear down a run's endpoint"),
            ("models export", "export an adapter to your HuggingFace repo"),
        ],
    ),
    (
        "self-hosted serving",
        [
            ("serve deploy", "provision a serving deployment in your own provider account"),
            ("serve status", "show a deployment's state in your own provider account"),
            ("serve undeploy", "tear down a deployment in your own provider account"),
        ],
    ),
]

_HELP_OPTIONS: list[tuple[str, str]] = [
    ("-h, --help", "show this help and exit"),
    ("-V, --version", "print the flash version"),
    ("--debug", "show full tracebacks on error"),
    ("-v, --verbose", "increase log verbosity (-v info, -vv debug)"),
]


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
        # then a dimmed pointer at this parser's own --help instead of the raw usage block. unknown
        # commands and flags get a short "did you mean" suggestion (see `friendly_error`). the argv
        # stash is what tells it which subcommand was selected; only the root parser carries one.
        print(
            render.error(friendly_error(message, self, getattr(self, "_flash_argv", None))),
            file=sys.stderr,
        )
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
            (
                f"train after publishing: `{CLI_NAME} env push --project PROJECT_UUID --name my-env .`, "
                f"then `{CLI_NAME} train configs/sft.toml`"
            ),
            f"any command in depth: `{CLI_NAME} <command> --help`",
            "docs: https://docs.freesolo.co",
        ]
        return render.help_page(
            "managed LoRA post-training", usage, _HELP_GROUPS, _HELP_OPTIONS, footers
        )


def _build_parser() -> argparse.ArgumentParser:
    """Build the fully-configured root parser. Extracted from main() so tests can introspect the
    registered subcommands and keep the themed help catalog (_HELP_GROUPS) in lockstep.

    Registration is split into one ``_add_*_commands`` helper per command family. The call order
    below is the order subcommands appear in help, so it is part of the CLI's surface, not an
    implementation detail. ``models_sub`` is threaded from the models family into the deployment
    family because `deploy`/`chat`/`export` are `flash models` subcommands registered after the
    top-level `runs` group.
    """
    parser = _FlashParser(prog=CLI_NAME, description="Managed LoRA post-training")
    _add_root_flags(parser)
    # subparsers theme their usage errors (parser_class=_ThemedParser) but not their help, so
    # `flash <cmd> --help` keeps the standard layout; only the root parser themes its help (see
    # _FlashParser). Nested `env` subcommands inherit _ThemedParser automatically (the env parser
    # is itself a _ThemedParser, so its add_subparsers defaults to the same class).
    sub = parser.add_subparsers(
        dest="cmd", required=True, parser_class=_ThemedParser, metavar="<command>"
    )

    account._add_auth_commands(sub)
    account._add_project_commands(sub)
    models_sub = models._add_model_commands(sub)
    env._add_env_commands(sub)
    traces._add_traces_commands(sub)
    training._add_train_commands(sub)
    training._add_runs_commands(sub)
    models._add_deployment_commands(models_sub)
    _add_serve_commands(sub)
    # the control plane is operator-only and run as a separate one-off service via the
    # `flash-server` console script (flash.server.asgi.cli:main), not a `flash` subcommand.

    return parser


def _add_root_flags(parser: argparse.ArgumentParser) -> None:
    """Flags on the root parser itself, ahead of any subcommand."""
    parser.add_argument("-V", "--version", action="version", version=f"{BRAND_NAME} {__version__}")
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


# commands that bind work to an organization: either they write to it, or they resolve a project
# with the ambient key and bake it into local artifacts that later writes inherit. a shadowed login
# silently binds these to the wrong org, so they warn. commands whose only effect is output the user
# reads stay quiet, and `flash whoami` already shows the key source in its own output.
_ORG_BINDING_COMMANDS = frozenset(
    {
        cmd_train,
        cmd_deploy,
        cmd_undeploy,
        cmd_export,
        cmd_cancel,
        cmd_projects_create,
        cmd_env_push,
        cmd_env_delete,
        # remotely read-only, but both pick a project from the ambient key and write it into the
        # working tree: `traces export` fills dataset/train.jsonl, `env setup` embeds the project
        # uuid in the generated configs. warning only at the later `train` is too late, because the
        # wrong org is already scaffolded in by then.
        cmd_traces_export,
        cmd_env_setup,
        # every case makes an authenticated `chat_stream` request against the ambient key, whether
        # or not results are uploaded, and access to the target is resolved from that key. so the
        # warning has to fire before the run either way: by the time a wrong-org target is reported
        # inaccessible, every paid request has already been spent against the other organization.
        cmd_env_eval,
    }
)


def _warn_if_login_shadowed(args) -> None:
    """Surface an ambient FREESOLO_API_KEY that binds a command to another org."""
    if getattr(args, "func", None) not in _ORG_BINDING_COMMANDS:
        return
    # `train --cost` parses its config inside the command and every algorithm authenticates for the
    # server-prepared quote. that path emits this warning itself after parsing, so suppress it here.
    if getattr(args, "cost", False):
        return
    message = shadowed_login_warning()
    if not message:
        return
    print(render.warn(message) if render.styled() else f"warning: {message}", file=sys.stderr)


def _is_credential_flag(arg: str) -> bool:
    """Whether `arg` is a credential flag, including the abbreviations argparse accepts.

    argparse allows any unambiguous prefix of a long option, so `--api-k` and `--api-ke` both reach
    `api_key`. Matching the full spelling exactly would leave those spellings unredacted and print
    the key -- the redaction would look correct while missing every abbreviated invocation.

    Ambiguity is deliberately NOT consulted. Whether `--api` is ambiguous depends on the SUBPARSER:
    `flash login` also defines `--api-url` so argparse refuses it there, but `flash models export`
    defines no such option, so `--ap`, `--api` and `--api-` are all unambiguous there and bind the
    HuggingFace token. Judging ambiguity against a fixed rival list gets that case exactly backwards
    and leaves the token printable. Erring the other way is free: if the parser does reject the
    spelling, nothing was bound, and redacting an argument in an already-failing command loses
    nothing.
    """
    if not arg.startswith("--") or len(arg) <= 2:
        return False
    return any(flag.startswith(arg) for flag in _CREDENTIAL_FLAGS)


def _redacted_args(raw_args: list[str]) -> list[str]:
    """`raw_args` with any credential value replaced, for echoing a command back to the user.

    The unexpected-error handler suggests re-running the exact command with --debug, which would
    otherwise reproduce a `--api-key <key>` the user typed -- printing the key into a terminal, a
    CI log or a pasted bug report. That is the same exposure `--api-key` already carries in process
    listings, and the fix costs nothing: the value is never what makes the traceback useful.
    """
    redacted: list[str] = []
    drop_value = False
    for arg in raw_args:
        if drop_value:
            redacted.append("<redacted>")
            drop_value = False
            continue
        if _is_credential_flag(arg):
            redacted.append(arg)
            drop_value = True
            continue
        flag, sep, _ = arg.partition("=")
        # `--api-key=secret` is one argv entry, so the split form has to be handled separately or
        # the value rides along inside it.
        redacted.append(f"{flag}=<redacted>" if sep and _is_credential_flag(flag) else arg)
    return redacted


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    parser = _build_parser()
    # the tokens this invocation was actually given, for the error hook: argparse reports a
    # subcommand's unknown flags from the ROOT parser, so scoping the suggestion to the selected
    # command needs the argv `main` was called with. reading sys.argv there instead would be right
    # only for the console-script path and wrong for every embedded caller -- including the tests,
    # where it silently reads pytest's own argv and restores the whole-tree pool.
    parser._flash_argv = raw_args
    args = parser.parse_args(argv)
    configure_logging(verbosity=getattr(args, "verbose", 0))
    debug = getattr(args, "debug", False)
    _warn_if_login_shadowed(args)
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
        cmd = " ".join([CLI_NAME, "--debug", *(shlex.quote(a) for a in _redacted_args(raw_args))])
        print(render.error(str(exc) or exc.__class__.__name__), file=sys.stderr)
        print(render.arrow(f"run `{cmd}` for the full traceback"), file=sys.stderr)
        return 1
