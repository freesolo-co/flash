"""tear down one exact customer-owned serving deployment and prove absence.

The deployment bundle is reconstructed from the same immutable arguments as ``serve deploy``.
Provider-assigned resource ids come from deploy or status output and bind deletion to one exact
resource generation. Provider credentials remain request-scoped environment values.
"""

from __future__ import annotations

import sys
import time

from flash.cli.commands.serve_deploy import _credentials, _deployment_bundle, _err


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
        from flash.serve.provisioning._modal_plan import build_modal_create_plan

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
    from flash.serve.provisioning._runpod_plan import build_runpod_create_plan

    _reject_foreign_ids(
        args,
        ("modal_app_id", "modal_volume_id", "modal_inference_secret_id"),
    )
    plan = build_runpod_create_plan(bundle)
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
        bundle = _deployment_bundle(args)
        handle = _provider_handle(args, bundle)
        if provider == "modal":
            from flash.serve.provisioning._modal_plan import build_modal_create_plan
            from flash.serve.provisioning.modal import _validate_handle

            plan = build_modal_create_plan(bundle, phase="finalized")
        else:
            from flash.serve.provisioning._runpod_plan import build_runpod_create_plan
            from flash.serve.provisioning.runpod import _validate_handle

            plan = build_runpod_create_plan(bundle)
        # validate the user-authored identity before provider teardown. only this local validation is
        # caught; the provider call remains outside so lifecycle defects still surface in full.
        _validate_handle(plan, handle)
        credentials = _credentials(provider)
    except (ValueError, TypeError) as exc:
        return _err(str(exc))

    deadline_at = time.monotonic() + float(args.timeout)
    if provider == "modal":
        from flash.serve.provisioning.modal import teardown_modal_deployment

        result = teardown_modal_deployment(
            bundle,
            handle,
            credentials,
            deadline_at=deadline_at,
        )
    else:
        from flash.serve.provisioning.runpod import teardown_runpod_deployment

        result = teardown_runpod_deployment(
            bundle,
            handle,
            credentials,
            deadline_at=deadline_at,
        )
    return _report_undeploy(result)
