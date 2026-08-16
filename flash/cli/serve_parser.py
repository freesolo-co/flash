"""Parser wiring for self-hosted serving commands."""

from __future__ import annotations

import argparse


def _add_serve_commands(sub: argparse._SubParsersAction) -> None:
    """`serve setup|status|teardown`: run a serving backend on your own Modal account.

    Kept separate from `models deploy` on purpose: those commands use a serving backend, while
    these commands stand one up. A hosted-plane user never needs these.
    """
    from flash.cli.commands.serve import cmd_serve_setup, cmd_serve_status, cmd_serve_teardown

    serve = sub.add_parser("serve", help="host a serving backend on your own Modal account")
    serve_sub = serve.add_subparsers(dest="serve_cmd", required=True)

    setup = serve_sub.add_parser("setup", help="generate and deploy a serving app for a model")
    setup.add_argument("--model", required=True, help="base model id to serve")
    setup.add_argument(
        "--output", default=None, help="where to write the app (default: flash_serving_app.py)"
    )
    setup.add_argument(
        "--scaledown-window",
        type=int,
        default=None,
        help="seconds of idle before the GPU container stops (2-1200, default: 300)",
    )
    setup.add_argument("--dry-run", action="store_true", help="write the app but do not deploy it")
    setup.add_argument("--force", action="store_true", help="overwrite an existing app file")
    setup.add_argument("-y", "--yes", action="store_true", help="deploy without confirming")
    setup.set_defaults(func=cmd_serve_setup)

    status = serve_sub.add_parser("status", help="check the configured serving backend")
    status.set_defaults(func=cmd_serve_status)

    teardown = serve_sub.add_parser("teardown", help="stop the Modal serving app")
    teardown.add_argument("--model", required=True, help="base model id whose app to stop")
    teardown.add_argument("-y", "--yes", action="store_true", help="stop without confirming")
    teardown.set_defaults(func=cmd_serve_teardown)
