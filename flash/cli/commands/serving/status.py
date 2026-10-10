"""report the proved state of one exact customer-owned modal deployment.

This is the user-facing read-only path into ``reconcile_modal_deployment``. It never mutates provider
resources and never turns an unproved state into ready or absent.
"""

from __future__ import annotations

import sys
import time

from flash.cli.commands.serving.deploy import (
    INFERENCE_KEY_ENV,
    _build_provider_plan,
    _credentials,
    _err,
    _optional_env,
    _report_handle,
    _require_modal_provider,
)
from flash.cli.commands.serving.identity import existing_deployment_bundle


def _report_status(result) -> int:
    print(f"deployment  {result.deployment_id} generation {result.generation}")
    print(f"provider    {result.provider}")
    print(f"engine      {result.engine_id}")
    print(f"image       {result.image_digest}")
    print(f"status      {result.status}")
    if result.handle is not None:
        _report_handle(result.handle, endpoint=result.status == "ready")

    if result.status == "ready":
        print("health      endpoint readiness proved")
        return 0
    if result.status == "provisioning":
        print("health      not ready yet")
        return 0
    if result.status == "absent":
        print("resources   confirmed absent")
        return 0
    if result.error_code:
        print(f"error       {result.error_code}", file=sys.stderr)
    if result.status == "outcome_unknown":
        print(
            "error: the deployment could not be proved healthy or absent. resources may still "
            "exist and bill.",
            file=sys.stderr,
        )
    else:
        print(
            "error: the provider status check failed; no ready or absent state was proved.",
            file=sys.stderr,
        )
    return 1


def cmd_serve_status(args) -> int:
    try:
        _require_modal_provider(args.provider)
        bundle = existing_deployment_bundle(args)
        # the provider plan applies the exact hostname contract before provider access.
        _build_provider_plan(bundle)
    except (ValueError, TypeError) as exc:
        return _err(str(exc))

    try:
        credentials = _credentials()
        inference_token = _optional_env(INFERENCE_KEY_ENV)
        from flash.serve.provisioning import ServingRuntimeSecrets

        runtime_secrets = (
            ServingRuntimeSecrets(inference_token) if inference_token is not None else None
        )
    except (ValueError, TypeError) as exc:
        return _err(
            f"{exc}. credentials are read from the environment for this one request and are never "
            "stored"
        )
    deadline_at = time.monotonic() + float(args.timeout)
    from flash.serve.provisioning.modal.execution.operations import reconcile_modal_deployment

    result = reconcile_modal_deployment(
        bundle,
        credentials,
        runtime_secrets,
        deadline_at=deadline_at,
    )
    return _report_status(result)
