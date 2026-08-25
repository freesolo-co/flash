"""tear down one exact customer-owned serving deployment and prove absence.

The deployment bundle is reconstructed from the immutable identity printed by ``serve deploy``.
Provider-assigned ids bind ordinary deletion to one exact generation. When an ambiguous create
returned no ids, explicit undeploy reclaims exact deterministic identities instead. Provider
credentials remain request-scoped environment values.
"""

from __future__ import annotations

import sys
import time

from flash.cli.commands.serving.deploy import _credentials, _err
from flash.cli.commands.serving.identity import existing_deployment_bundle


def _required_arg(args, name: str) -> str:
    value = getattr(args, name, "") or ""
    if not value:
        raise ValueError(f"--{name.replace('_', '-')} is required for {args.provider}")
    return value


def _reject_foreign_ids(args, names: tuple[str, ...]) -> None:
    supplied = [name for name in names if getattr(args, name, "")]
    if supplied:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in supplied)
        raise ValueError(f"{flags} cannot be used with --provider {args.provider}")


def _provider_handle(args, bundle):
    """bind user-supplied provider ids to deterministic names and exact provenance."""

    if args.provider == "modal":
        from flash.serve.control import ModalProviderHandle
        from flash.serve.provisioning.modal.planning.plan import build_modal_create_plan

        _reject_foreign_ids(
            args,
            (
                "runpod_pod_id",
                "runpod_network_volume_id",
                "runpod_template_id",
                "runpod_inference_secret_id",
            ),
        )
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

    from flash.serve.control import RunPodProviderHandle
    from flash.serve.provisioning.runpod.plan import build_runpod_create_plan

    _reject_foreign_ids(
        args,
        ("modal_app_id", "modal_volume_id", "modal_inference_secret_id"),
    )
    plan = build_runpod_create_plan(bundle)
    id_names = (
        "runpod_pod_id",
        "runpod_network_volume_id",
        "runpod_template_id",
        "runpod_inference_secret_id",
    )
    supplied_ids = [name for name in id_names if getattr(args, name, "")]
    if not supplied_ids:
        return None
    if len(supplied_ids) != len(id_names):
        raise ValueError("runpod provider ids must be supplied together or omitted for reclaim")
    pod_id = _required_arg(args, "runpod_pod_id")
    return RunPodProviderHandle(
        deployment_id=bundle.spec.deployment_id,
        generation=bundle.spec.generation,
        engine_id=bundle.spec.engine.engine_id,
        account_id=plan.placement.account_id,
        pod_id=pod_id,
        pod_name=plan.names.app_or_pod,
        network_volume_id=_required_arg(args, "runpod_network_volume_id"),
        network_volume_name=plan.names.volume,
        template_id=_required_arg(args, "runpod_template_id"),
        template_name=plan.names.template,
        inference_secret_id=_required_arg(args, "runpod_inference_secret_id"),
        inference_secret_name=plan.names.inference_secret,
        data_center_id=plan.placement.data_center_id,
        image_digest=bundle.image.digest,
        public_url=f"https://{pod_id}-8000.proxy.runpod.net",
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
    provider = args.provider
    try:
        bundle = existing_deployment_bundle(args)
        handle = _provider_handle(args, bundle)
        if provider == "modal":
            from flash.serve.provisioning.modal.execution.operations import _validate_handle
            from flash.serve.provisioning.modal.planning.plan import build_modal_create_plan

            plan = build_modal_create_plan(bundle, phase="finalized")
        else:
            from flash.serve.provisioning.runpod.operations import _validate_handle
            from flash.serve.provisioning.runpod.plan import build_runpod_create_plan

            plan = build_runpod_create_plan(bundle)
        # validate user-authored provider ids before teardown. either provider may have none only
        # when an ambiguous create returned no handle; the immutable deployment identity then
        # authorizes the bounded exact-name reclaim instead.
        if handle is not None:
            _validate_handle(plan, handle)
        credentials = _credentials(provider)
    except (ValueError, TypeError) as exc:
        return _err(str(exc))

    deadline_at = time.monotonic() + float(args.timeout)
    if provider == "modal":
        from flash.serve.provisioning.modal.execution.operations import (
            teardown_modal_deployment,
        )

        result = teardown_modal_deployment(
            bundle,
            handle,
            credentials,
            deadline_at=deadline_at,
        )
    else:
        from flash.serve.provisioning.runpod.operations import teardown_runpod_deployment

        result = teardown_runpod_deployment(
            bundle,
            handle,
            credentials,
            deadline_at=deadline_at,
        )
    return _report_undeploy(result)
