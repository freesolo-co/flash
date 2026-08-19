"""Parser wiring for self-hosted serving commands."""

from __future__ import annotations

import argparse


def _add_serve_commands(sub: argparse._SubParsersAction) -> None:
    """`serve setup|status|teardown`: run a serving backend on your own Modal account.

    Kept separate from `models deploy` on purpose: those commands use a serving backend, while
    these commands stand one up. A hosted-plane user never needs these.
    """
    from flash.cli.commands.serve import cmd_serve_setup, cmd_serve_status, cmd_serve_teardown
    from flash.cli.commands.serve_deploy import cmd_serve_deploy

    serve = sub.add_parser("serve", help="host a serving backend on your own Modal account")
    serve_sub = serve.add_subparsers(dest="serve_cmd", required=True)

    _add_serve_deploy(serve_sub, cmd_serve_deploy)

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


def _add_serve_deploy(serve_sub: argparse._SubParsersAction, handler) -> None:
    """`serve deploy`: provision one exact deployment in your own Modal or RunPod account.

    Distinct from `serve setup`, which writes and deploys a generated Modal app. This one runs the
    packaged serving image, so the deployment is pinned to an image digest and an engine identity
    rather than to whatever the generated source happened to say.

    Credentials are deliberately NOT arguments. They are read from the environment for this one
    request and never persisted, so they cannot leak into shell history, a process list, or a
    durable record. The command reads ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` or
    ``RUNPOD_API_KEY`` for the provider, ``FLASH_SERVING_KEY`` for the endpoint's own auth, and
    ``HF_TOKEN`` so the container can hydrate its adapter -- required for a private artifact repo,
    which is the default.
    """

    deploy = serve_sub.add_parser(
        "deploy",
        help="provision one deployment on your own modal or runpod account",
    )
    deploy.add_argument(
        "--provider",
        required=True,
        choices=("modal", "runpod"),
        help="which of your accounts to provision in",
    )
    deploy.add_argument("--model", required=True, help="base model id to serve")
    deploy.add_argument("--run", required=True, help="run id whose adapter to serve")
    deploy.add_argument(
        "--deployment-id", required=True, help="stable id for this deployment across generations"
    )
    deploy.add_argument(
        "--generation",
        type=int,
        default=1,
        help="generation number for this deployment (default: 1)",
    )
    deploy.add_argument(
        "--image",
        required=True,
        help="digest-qualified serving image reference (name@sha256:...)",
    )
    deploy.add_argument("--artifact-repo", required=True, help="hub repo holding the adapter")
    deploy.add_argument(
        "--artifact-subfolder", required=True, help="path within the repo holding the adapter"
    )
    # flash runs publish adapters into dataset repos, so that stays the default. the override
    # exists for an adapter published as a model repo, where resolution would otherwise fail
    # against the wrong repo type rather than reporting that the repo is a different kind.
    deploy.add_argument(
        "--artifact-repo-type",
        choices=("dataset", "model"),
        default="dataset",
        help="hub repo type holding the adapter (default: dataset, as flash runs publish)",
    )
    deploy.add_argument("--lora-rank", type=int, required=True, help="the adapter's lora rank")
    deploy.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="serve one saved step instead of the final adapter",
    )
    deploy.add_argument(
        "--thinking", action="store_true", help="default this adapter to thinking mode"
    )
    # modal placement
    deploy.add_argument("--modal-workspace", default="", help="modal workspace name")
    deploy.add_argument("--modal-environment", default="", help="modal environment name")
    # modal prices a pinned region above an unpinned one and a narrower region draws on a smaller
    # capacity pool, so prefer a broad value ("us-east") over a specific one ("us-east-1").
    deploy.add_argument(
        "--modal-region", default="", help="modal region, e.g. us-east (broad is cheaper)"
    )
    # runpod placement
    deploy.add_argument("--runpod-account", default="", help="runpod account id")
    deploy.add_argument("--runpod-data-center", default="", help="runpod data center id")
    deploy.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="seconds to wait for the deployment to become ready (default: 1800)",
    )
    deploy.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and validate every input without contacting the provider",
    )
    deploy.set_defaults(func=handler)
