"""report the proved state of one exact customer-owned serving deployment.

This is the user-facing read-only path into ``reconcile_modal_deployment`` and
``reconcile_runpod_deployment``. It never mutates provider resources and never turns an unproved
state into ready or absent.
"""

from __future__ import annotations

import sys
import time

from flash.cli.commands.serve_deploy import (
    _build_provider_plan,
    _credentials,
    _err,
    _report_handle,
    _runtime_secrets,
)
from flash.cli.commands.serve_identity import existing_deployment_bundle


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
    provider = args.provider
    try:
        bundle = existing_deployment_bundle(args)
        # bundle construction only validates the generic placement type. the provider plan applies
        # the exact hostname contract before reconciliation can contact the provider.
        _build_provider_plan(provider, bundle)
    except (ValueError, TypeError) as exc:
        return _err(str(exc))

    try:
        credentials = _credentials(provider)
        runtime_secrets = _runtime_secrets()
    except (ValueError, TypeError) as exc:
        return _err(
            f"{exc}. credentials are read from the environment for this one request and are never "
            f"stored"
        )

    deadline_at = time.monotonic() + float(args.timeout)
    if provider == "modal":
        from flash.serve.provisioning.modal import reconcile_modal_deployment

        result = reconcile_modal_deployment(
            bundle,
            credentials,
            runtime_secrets,
            deadline_at=deadline_at,
        )
    else:
        from flash.serve.provisioning.runpod import reconcile_runpod_deployment

        result = reconcile_runpod_deployment(
            bundle,
            credentials,
            runtime_secrets,
            deadline_at=deadline_at,
        )
    return _report_status(result)
