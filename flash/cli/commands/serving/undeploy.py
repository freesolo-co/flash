"""tear down one exact customer-owned modal deployment and prove absence.

The deployment bundle is reconstructed from the immutable identity printed by ``serve deploy``.
Provider-assigned ids bind ordinary deletion to one exact generation. When an ambiguous create
returned no ids, explicit undeploy reclaims exact deterministic identities instead. Provider
credentials remain request-scoped environment values.
"""

from __future__ import annotations

import sys
import time

from flash.cli.commands.serving.deploy import _credentials, _err, _require_modal_provider
from flash.cli.commands.serving.identity import existing_deployment_bundle


def _required_arg(args, name: str) -> str:
    value = getattr(args, name, "") or ""
    if not value:
        raise ValueError(f"--{name.replace('_', '-')} is required for modal")
    return value


def _provider_handle(args, bundle):
    """bind user-supplied modal ids to deterministic names and exact provenance."""

    from flash.serve.control import ModalProviderHandle
    from flash.serve.provisioning.modal.planning.plan import build_modal_create_plan

    plan = build_modal_create_plan(bundle, phase="finalized")
    id_names = (
        "modal_app_id",
        "modal_volume_id",
        "modal_inference_secret_id",
    )
    supplied_ids = [name for name in id_names if getattr(args, name, "")]
    if not supplied_ids:
        return None
    if len(supplied_ids) != len(id_names):
        raise ValueError("modal provider ids must be supplied together or omitted for reclaim")
    return ModalProviderHandle(
        deployment_id=bundle.spec.deployment_id,
        generation=bundle.spec.generation,
        engine_id=bundle.spec.engine.engine_id,
        workspace_name=plan.placement.workspace_name,
        app_id=_required_arg(args, "modal_app_id"),
        app_name=plan.names.app_or_pod,
        volume_id=_required_arg(args, "modal_volume_id"),
        volume_name=plan.names.volume,
        inference_secret_id=_required_arg(args, "modal_inference_secret_id"),
        inference_secret_name=plan.names.inference_secret,
        environment=plan.placement.environment,
        region=plan.placement.region,
        image_digest=bundle.image.digest,
        public_url=plan.expected_public_url,
    )


def _report_undeploy(result) -> int:
    print(f"deployment  {result.deployment_id} generation {result.generation}")
    print(f"provider    {result.provider}")
    print(f"status      {result.status}")
    if result.status == "absent":
        print("resources   confirmed absent")
        return 0
    if result.error_code:
        print(f"error       {result.error_code}", file=sys.stderr)
    print(
        "error: resource absence could not be proved. resources may still exist and bill; do not "
        "treat this deployment as stopped.",
        file=sys.stderr,
    )
    return 1


def cmd_serve_undeploy(args) -> int:
    try:
        _require_modal_provider(args.provider)
        bundle = existing_deployment_bundle(args)
        handle = _provider_handle(args, bundle)
        from flash.serve.provisioning.modal.execution.operations import _validate_handle
        from flash.serve.provisioning.modal.planning.plan import build_modal_create_plan

        plan = build_modal_create_plan(bundle, phase="finalized")
        # validate user-authored provider ids before teardown. ids may all be absent only when an
        # ambiguous create returned no handle; the immutable identity then authorizes exact reclaim.
        if handle is not None:
            _validate_handle(plan, handle)
        credentials = _credentials()
    except (ValueError, TypeError) as exc:
        return _err(str(exc))

    deadline_at = time.monotonic() + float(args.timeout)
    from flash.serve.provisioning.modal.execution.operations import teardown_modal_deployment

    result = teardown_modal_deployment(
        bundle,
        handle,
        credentials,
        deadline_at=deadline_at,
    )
    return _report_undeploy(result)
