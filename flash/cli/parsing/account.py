"""Account and project CLI parser registration."""

from __future__ import annotations

import argparse

from flash.cli.commands.ops.account import (
    cmd_login,
    cmd_projects_create,
    cmd_projects_list,
    cmd_version,
    cmd_whoami,
)


def _add_auth_commands(sub: argparse._SubParsersAction) -> None:
    """`version`, `login`, `whoami`: identity and build introspection."""
    version = sub.add_parser("version", help="print the Flash version")
    version.set_defaults(func=cmd_version)

    login = sub.add_parser(
        "login",
        help="log in with your freesolo API key (create one at https://freesolo.co/sign-in)",
    )
    login.add_argument(
        "--api-key",
        help=(
            "your freesolo API key (default: FREESOLO_API_KEY); prefer the environment variable "
            "because argument values are visible in process listings; create a key at "
            "https://freesolo.co/sign-in"
        ),
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


def _add_project_commands(sub: argparse._SubParsersAction) -> None:
    """`projects create` / `projects list`."""
    projects = sub.add_parser("projects", help="manage Freesolo projects")
    projects_sub = projects.add_subparsers(dest="projects_cmd", required=True)
    projects_create = projects_sub.add_parser("create", help="create a Freesolo project")
    projects_create.add_argument("name", metavar="NAME")
    projects_create.add_argument("--description", help="optional project description")
    projects_create.set_defaults(func=cmd_projects_create)
    projects_list = projects_sub.add_parser("list", help="list Freesolo projects and UUIDs")
    projects_list.set_defaults(func=cmd_projects_list)
