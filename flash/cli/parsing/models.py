"""Model catalog and deployment CLI parser registration."""

from __future__ import annotations

import argparse
import math

from flash.cli.commands.ops.catalog import cmd_gpus, cmd_models
from flash.cli.commands.ops.deploy import (
    cmd_chat,
    cmd_deploy,
    cmd_deployments,
    cmd_export,
    cmd_undeploy,
)


def _wait_seconds(value: str) -> float:
    """Parse a bounded deployment wait timeout."""
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a number of seconds, got {value!r}. if {value!r} is the run id, put it "
            f"before the flag (`deploy {value} --wait`) or give --wait an explicit timeout"
        ) from None
    if not math.isfinite(seconds):
        raise argparse.ArgumentTypeError(f"--wait needs a finite number of seconds, got {value!r}")
    if seconds < 0:
        raise argparse.ArgumentTypeError(f"--wait cannot be negative, got {value!r}")
    return seconds


def _add_model_commands(sub: argparse._SubParsersAction) -> argparse._SubParsersAction:
    """`models list` and `gpus`, returning the `models` subparser action.

    The returned action is what `_add_deployment_commands` hangs `deploy`/`undeploy`/`export`/
    `deployments`/`chat` off, further down the registration order.
    """
    models = sub.add_parser("models", help="work with models and deployments")
    models.set_defaults(func=cmd_models)  # hidden bare `flash models` shim, mirrors `flash runs`
    models_sub = models.add_subparsers(dest="models_cmd", required=False)
    models_list = models_sub.add_parser("list", help="list supported base models")
    models_list.set_defaults(func=cmd_models)

    gpus = sub.add_parser("gpus", help="list managed GPU classes with estimated $/hr")
    gpus.set_defaults(func=cmd_gpus)
    return models_sub


def _add_deployment_commands(models_sub: argparse._SubParsersAction) -> None:
    """`models deploy/undeploy/export/deployments/chat`, hung off the `models` subparser."""
    deploy = models_sub.add_parser(
        "deploy", help="deploy an exact checkpoint to a serving endpoint"
    )
    deploy.add_argument(
        "run_id",
        metavar="CHECKPOINT_ID",
        help="permanent checkpoint id: RUN_ID/final or RUN_ID/step-N",
    )
    deploy.add_argument("--dry-run", action="store_true")
    deploy.add_argument(
        "--wait",
        nargs="?",
        type=_wait_seconds,
        const=2400.0,
        default=None,
        metavar="SECONDS",
        help=(
            "block until the requested revision is ready or failed (default timeout 2400s). "
            "without it, deploy returns while the revision is still queued"
        ),
    )
    deploy.set_defaults(func=cmd_deploy)

    undeploy = models_sub.add_parser("undeploy", help="tear down an exact checkpoint deployment")
    undeploy.add_argument(
        "run_id",
        metavar="CHECKPOINT_ID",
        help="permanent checkpoint id: RUN_ID/final or RUN_ID/step-N",
    )
    undeploy.set_defaults(func=cmd_undeploy)

    export = models_sub.add_parser(
        "export", help="export a trained adapter to your own HuggingFace repo"
    )
    export.add_argument(
        "--adapter-id",
        dest="adapter_id",
        required=True,
        metavar="CHECKPOINT_ID",
        help="permanent checkpoint id: RUN_ID/final or RUN_ID/step-N",
    )
    export.add_argument(
        "--repository",
        required=True,
        help="destination HuggingFace repo 'owner/name' (created if it doesn't exist)",
    )
    export.add_argument(
        "--api-key",
        help=(
            "HuggingFace token with write access to --repository; prefer HF_TOKEN or a local "
            ".env / .env.local because argument values are visible in process listings"
        ),
    )
    export.add_argument(
        "--public",
        action="store_true",
        help="create the destination repo as public (default: private)",
    )
    export.set_defaults(func=cmd_export)

    deployments = models_sub.add_parser("deployments", help="list active serving deployments")
    deployments.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable deployment records",
    )
    deployments.set_defaults(func=cmd_deployments)

    chat = models_sub.add_parser("chat", help="chat with a deployed adapter")
    chat.add_argument(
        "run_id",
        metavar="TARGET",
        help="permanent deployed checkpoint id: RUN_ID/final or RUN_ID/step-N",
    )
    chat.add_argument("-m", "--message", required=True)
    chat.add_argument(
        "--system",
        default=None,
        help="optional system prompt sent ahead of the user message "
        "(training-prompt parity for evals)",
    )
    chat.add_argument("--max-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.set_defaults(func=cmd_chat)
