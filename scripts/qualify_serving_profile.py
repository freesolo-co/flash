"""build one explicitly named nonshipping serving qualification plan.

this tool resolves the same bundle and builds the same modal plan as `flash serve deploy`, but it
never calls a provider and never changes a profile's shipping qualification flag. provider
credentials are not accepted as arguments or read from the environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from flash.cli.commands.serving.deploy import _build_provider_plan, _deployment_bundle
from flash.cli.parsing.serve_parser import _add_serve_commands
from flash.server.domain.ops.serving_resources import dry_run_deployment


def build_qualification_plan(args: argparse.Namespace, qualification_cell: str):
    """build exactly the named model/provider cell without enabling it for shipping."""

    expected = f"{args.model}:{args.provider}"
    if qualification_cell != expected:
        raise ValueError(
            f"qualification cell {qualification_cell!r} does not match CLI cell {expected!r}"
        )
    bundle = _deployment_bundle(args)
    return bundle, _build_provider_plan(bundle)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="build one provider-free modal customer-serving qualification plan"
    )
    parser.add_argument(
        "--qualification-cell",
        required=True,
        help="exact MODEL:PROVIDER cell, for example Qwen/Qwen3.8-27B:modal",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_serve_commands(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.serve_cmd != "deploy":
        raise ValueError("qualification requires the serve deploy command")
    bundle, plan = build_qualification_plan(args, args.qualification_cell)
    for key, value in dry_run_deployment(bundle).items():
        print(f"{key:26} {value}")
    print(f"plan_type                  {type(plan).__name__}")
    print("qualification only: no provider was contacted and no profile was enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
