"""provision one exact deployment on the user's own modal account.

Composes and provisions the complete Modal deployment bundle.

Provider credentials are read from the process environment for the duration of one call and
wrapped in the request-scoped credential boundary. They are never accepted as arguments (a
process list and shell history are both readable), never written anywhere, and never logged --
the credential types refuse to serialize, and provider failures surface as sanitized codes rather
than raw responses.
"""

from __future__ import annotations

import os
import sys
import time

from flash.serve.control import ModalCredentials
from flash.serve.provisioning import (
    FreshDeploymentArtifactTokenRequired,
    InterruptedProvisioning,
)

MODAL_TOKEN_ID_ENV = "MODAL_TOKEN_ID"
MODAL_TOKEN_SECRET_ENV = "MODAL_TOKEN_SECRET"
# the serving app authenticates callers with this. a provider endpoint url is public, so without
# it anyone who finds the url can load adapters and spend the customer's gpu budget.
INFERENCE_KEY_ENV = "FLASH_SERVING_KEY"
# the container hydrates its adapter from the hub itself, so it needs the same read token the
# control plane used to resolve that adapter. an artifact repo is private by default, and without
# this the launcher reaches `_prepare_cache` with nothing to authenticate as and dies with
# "artifact token is required when serving cache hydration is missing" -- after the provider has
# already created and started billing for every resource.
ARTIFACT_TOKEN_ENV = "HF_TOKEN"


def _err(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def _optional_env(name: str) -> str | None:
    """read a token that is genuinely optional, mapping unset and blank alike to absence."""

    return os.environ.get(name, "").strip() or None


def _credentials(provider: str):
    if provider != "modal":
        raise ValueError("Flash serving requires Modal")
    return ModalCredentials(
        token_id=_required_env(MODAL_TOKEN_ID_ENV),
        token_secret=_required_env(MODAL_TOKEN_SECRET_ENV),
    )


def _image(reference: str):
    from flash.serve.provisioning import ServingImage

    if "@" not in reference:
        raise ValueError(
            "the serving image must be digest-qualified (name@sha256:...). a tag can be moved "
            "to different content after the deployment is recorded"
        )
    _, digest = reference.rsplit("@", 1)
    return ServingImage(reference=reference, digest=digest)


def _report_handle(handle, *, endpoint: bool = True) -> None:
    """Print the Modal resource identity needed for status and teardown."""
    if endpoint:
        print(f"endpoint    {handle.public_url}")
    print(f"app id      {handle.app_id}")
    print(f"volume id   {handle.volume_id}")
    print(f"secret id   {handle.inference_secret_id}")


def _report(result, bundle) -> int:
    print(f"deployment  {result.deployment_id} generation {result.generation}")
    print(f"provider    {result.provider}")
    print(f"engine      {result.engine_id}")
    print(f"image       {result.image_digest}")
    print(f"status      {result.status}")
    handle = result.handle
    if handle is not None:
        _report_handle(handle)
    _report_identity(bundle)
    if result.status == "ready":
        return 0
    if result.error_code:
        print(f"error       {result.error_code}", file=sys.stderr)
    if result.error_reason:
        print(f"reason      {result.error_reason}", file=sys.stderr)
    if str(result.error_reason or "").startswith("artifact_cleanup_") and handle is not None:
        resource_kind = "app"
        resource_id = handle.app_id
        print(
            f"\nthe service reached readiness on {resource_kind} {resource_id}, but artifact cleanup did not settle. the {resource_kind} is live and billing. run `flash serve status` to inspect the deployment, then `flash serve undeploy` to stop it billing if you do not want it.",
            file=sys.stderr,
        )
    elif result.status == "provisioning" and handle is not None:
        print(
            f"\nthe app {handle.app_id} did not prove ready before the deadline, but may be live and billing. run `flash serve status` to inspect the deployment, then `flash serve undeploy` to stop it billing if you do not want it.",
            file=sys.stderr,
        )
    elif result.status == "outcome_unknown":
        print(
            "\nthe provider outcome could not be confirmed. resources may exist; run `flash serve status` to inspect them, then `flash serve undeploy` to stop them billing before retrying rather than provisioning again.",
            file=sys.stderr,
        )
    return 1


def _report_identity(bundle) -> None:
    """print recovery identity without replacing the reported outcome if encoding fails."""

    try:
        from flash.cli.commands.serve_identity import encode_deployment_identity

        print(f"identity    {encode_deployment_identity(bundle)}")
    except Exception:
        return


def _build_provider_plan(provider: str, bundle) -> None:
    """run the provider's own plan validation without contacting it."""
    from flash.serve.provisioning._modal_plan import build_modal_create_plan

    build_modal_create_plan(bundle, phase="finalized")
    return


def _runtime_secrets():
    """build request-scoped endpoint secrets from the environment."""

    from flash.serve.provisioning import ServingRuntimeSecrets

    return ServingRuntimeSecrets(
        inference_token=_required_env(INFERENCE_KEY_ENV),
        artifact_token=_optional_env(ARTIFACT_TOKEN_ENV),
    )


def _deployment_bundle(args):
    """rebuild the exact immutable deployment input shared by deploy and undeploy."""
    from flash.serve.control import DeploymentRequest
    from flash.serve.profiles import get_profile, placement_for
    from flash.serve.resolve import execution_inputs, resolve_adapter, resolve_base_revision
    from flash.server.domain.serving_resources import resolve_deployment_bundle

    provider = args.provider
    profile = get_profile(args.model)
    image = _image(args.image)
    placement = placement_for(
        profile,
        provider,
        workspace_name=getattr(args, "modal_workspace", "") or "",
        environment=getattr(args, "modal_environment", "") or "",
        region=getattr(args, "modal_region", "") or "",
        web_suffix=(getattr(args, "modal_web_suffix", "") or "") or None,
    )
    base_revision = resolve_base_revision(profile.served_model)
    resolved = resolve_adapter(
        run_id=args.run,
        artifact_repo_id=args.artifact_repo,
        artifact_subfolder=args.artifact_subfolder,
        artifact_repo_type=getattr(args, "artifact_repo_type", "dataset"),
        base_model=args.model,
        base_model_revision=resolve_base_revision(args.model),
        lora_rank=args.lora_rank,
        checkpoint_step=getattr(args, "checkpoint_step", None),
        thinking_default=bool(getattr(args, "thinking", False)),
    )
    engine = profile.engine(
        model_revision=base_revision, tokenizer_revision=base_revision, image=image
    )
    request = DeploymentRequest(
        deployment_id=args.deployment_id,
        generation=args.generation,
        provider=provider,
        placement=placement,
        engine=engine,
        adapters=(resolved.adapter,),
    )
    return resolve_deployment_bundle(request, execution_inputs(profile, image, (resolved,)), image)


def cmd_serve_deploy(args) -> int:
    provider = args.provider
    try:
        bundle = _deployment_bundle(args)
    except (ValueError, TypeError) as exc:
        return _err(str(exc))
    from flash.server.domain.serving_resources import dry_run_deployment

    try:
        _build_provider_plan(provider, bundle)
    except (ValueError, TypeError) as exc:
        return _err(str(exc))
    if getattr(args, "dry_run", False):
        for key, value in dry_run_deployment(bundle).items():
            print(f"{key:18} {value}")
        print("\ndry run: no provider was contacted and nothing was provisioned.")
        return 0
    try:
        credentials = _credentials(provider)
        runtime_secrets = _runtime_secrets()
    except ValueError as exc:
        return _err(
            f"{exc}. provider credentials are read from the environment for this one request and are never stored"
        )
    deadline_at = time.monotonic() + float(args.timeout)
    try:
        from flash.serve.provisioning.modal import provision_modal_deployment

        result = provision_modal_deployment(
            bundle, credentials, runtime_secrets, deadline_at=deadline_at
        )
    except FreshDeploymentArtifactTokenRequired as exc:
        return _err(f"{ARTIFACT_TOKEN_ENV} is not set. {exc}")
    except InterruptedProvisioning as interrupted:
        _report_identity(bundle)
        _warn_unconfirmed_cleanup(interrupted.provider, args.deployment_id)
        raise
    return _report(result, bundle)


def _warn_unconfirmed_cleanup(provider: str, deployment_id: str) -> None:
    """say what ctrl-c actually left behind, before the generic handler says "aborted"."""

    print(
        f"\nwarning: interrupted before ready, and {provider} cleanup could not be confirmed for "
        f"{deployment_id}. resources may still exist and bill; run `flash serve status` to inspect "
        f"them, then `flash serve undeploy` before retrying rather than provisioning again.",
        file=sys.stderr,
    )
