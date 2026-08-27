"""Parser wiring for customer-owned serving deployment lifecycle commands.

``serve deploy`` provisions one deployment in the user's Modal account. ``serve undeploy`` tears
down that same exact deployment generation. Kept separate from ``models deploy``
and ``models undeploy``, which use the hosted serving backend rather than customer-owned resources.
"""

from __future__ import annotations

import argparse
import math


def _positive_finite_seconds(raw: str) -> float:
    """Reject a timeout that cannot describe a future deadline.

    Without this, `--timeout 0`, a negative, `nan`, or `inf` parses fine and only fails inside the
    provider's `validate_deadline`, which raises `ValueError` outside the lifecycle error handling
    -- so the CLI prints a traceback *after* resolving and downloading the Hub inputs, instead of
    the ordinary argument error the user should get before any work happens.
    """
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(
            f"must be a positive, finite number of seconds, not {raw!r}"
        )
    return value


def _positive_int(raw: str) -> int:
    """reject a nonpositive integer before deployment input resolution.

    a bare `type=int` accepts `0` and negatives, while downstream validation runs only after
    `resolve_adapter` has already resolved and downloaded the hub inputs. argparse can reject these
    values before they cost a hub round trip.
    """
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {raw!r}")
    return value


def _add_serve_commands(sub: argparse._SubParsersAction) -> None:
    serve = sub.add_parser("serve", help="manage serving in your own provider account")
    serve_sub = serve.add_subparsers(dest="serve_cmd", required=True)
    _add_serve_deploy(serve_sub)
    _add_serve_status(serve_sub)
    _add_serve_undeploy(serve_sub)


def _add_deployment_arguments(command: argparse.ArgumentParser) -> None:
    """add the exact deployment identity shared by deploy and undeploy."""

    command.add_argument(
        "--provider",
        required=True,
        choices=("modal",),
        help="required provider for customer-owned serving",
    )
    command.add_argument("--model", required=True, help="base model id to serve")
    command.add_argument("--run", required=True, help="run id whose adapter to serve")
    command.add_argument(
        "--deployment-id", required=True, help="stable id for this deployment across generations"
    )
    command.add_argument(
        "--generation",
        type=_positive_int,
        default=1,
        help="generation number for this deployment (default: 1)",
    )
    command.add_argument(
        "--image",
        required=True,
        help="digest-qualified serving image reference (name@sha256:...)",
    )
    command.add_argument("--artifact-repo", required=True, help="hub repo holding the adapter")
    command.add_argument(
        "--artifact-subfolder", required=True, help="path within the repo holding the adapter"
    )
    # flash runs publish adapters into dataset repos, so that stays the default. the override
    # exists for an adapter published as a model repo, where resolution would otherwise fail
    # against the wrong repo type rather than reporting that the repo is a different kind.
    command.add_argument(
        "--artifact-repo-type",
        choices=("dataset", "model"),
        default="dataset",
        help="hub repo type holding the adapter (default: dataset, as flash runs publish)",
    )
    command.add_argument(
        "--lora-rank", type=_positive_int, required=True, help="the adapter's lora rank"
    )
    command.add_argument(
        "--checkpoint-step",
        type=_positive_int,
        default=None,
        help="serve one saved step instead of the final adapter",
    )
    command.add_argument(
        "--thinking", action="store_true", help="default this adapter to thinking mode"
    )
    # modal placement
    command.add_argument("--modal-workspace", default="", help="modal workspace name")
    command.add_argument("--modal-environment", default="", help="modal environment name")
    # modal builds web urls as `<workspace>-<web-suffix>--<label>.modal.run`. the suffix is a
    # per-environment setting the operator chooses, not the environment name, and one environment
    # per workspace may have none, so it cannot be derived and is omitted for that environment.
    command.add_argument(
        "--modal-web-suffix",
        default="",
        help="modal environment web suffix, if the environment has one (see `modal environment`)",
    )
    # modal prices a pinned region above an unpinned one and a narrower region draws on a smaller
    # capacity pool, so prefer a broad value ("us-east") over a specific one ("us-east-1").
    command.add_argument(
        "--modal-region", default="", help="modal region, e.g. us-east (broad is cheaper)"
    )
    command.add_argument(
        "--timeout",
        type=_positive_finite_seconds,
        default=3600.0,
        help="seconds to wait for the provider operation (default: 3600)",
    )


def _add_serve_deploy(serve_sub: argparse._SubParsersAction) -> None:
    """`serve deploy`: provision one exact deployment in your own Modal account.

    Runs the packaged serving image, so a deployment is pinned to an image digest and an engine
    identity rather than to generated source that could drift from what was validated.

    Credentials are deliberately NOT arguments. They are read from the environment for this one
    request and never persisted, so they cannot leak into shell history, a process list, or a
    durable record. The command reads ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` for the provider,
    ``FLASH_SERVING_KEY`` for the endpoint's own auth, and
    ``HF_TOKEN`` so the container can hydrate its adapter -- required for a private artifact repo,
    which is the default.
    """

    deploy = serve_sub.add_parser(
        "deploy",
        help="provision one deployment on your own modal account",
    )
    _add_deployment_arguments(deploy)
    deploy.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate every input without contacting the provider",
    )
    from flash.cli.commands.serving.deploy import cmd_serve_deploy

    deploy.set_defaults(func=cmd_serve_deploy)


def _add_existing_deployment_identity(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--deployment-identity",
        default="",
        help=(
            "immutable identity printed by deploy; avoids Hub resolution when checking or removing "
            "an existing deployment"
        ),
    )


def _add_serve_status(serve_sub: argparse._SubParsersAction) -> None:
    """`serve status`: read and prove one exact deployment without provider mutation."""

    status = serve_sub.add_parser(
        "status",
        help="show the proved state of one deployment in your provider account",
    )
    _add_deployment_arguments(status)
    _add_existing_deployment_identity(status)
    from flash.cli.commands.serving.status import cmd_serve_status

    status.set_defaults(func=cmd_serve_status)


def _add_serve_undeploy(serve_sub: argparse._SubParsersAction) -> None:
    """`serve undeploy`: remove one exact deployment generation and prove absence.

    Provider credentials remain request-only environment values. Provider-assigned resource ids are
    not secrets; the deploy command prints them so ordinary teardown can bind every deletion to the
    exact generation. Modal ids may all be omitted to reclaim an ambiguous create from the immutable
    deployment identity printed by deploy.
    """

    undeploy = serve_sub.add_parser(
        "undeploy",
        help="tear down one deployment and prove its resources are absent",
    )
    _add_deployment_arguments(undeploy)
    _add_existing_deployment_identity(undeploy)
    undeploy.add_argument("--modal-app-id", default="", help="modal app id printed by deploy")
    undeploy.add_argument("--modal-volume-id", default="", help="modal volume id printed by deploy")
    undeploy.add_argument(
        "--modal-inference-secret-id",
        default="",
        help="modal inference secret id printed by deploy",
    )
    from flash.cli.commands.serving.undeploy import cmd_serve_undeploy

    undeploy.set_defaults(func=cmd_serve_undeploy)
