"""provision one exact deployment on the user's own modal or runpod account.

This is the command path into ``provision_modal_deployment`` / ``provision_runpod_deployment``.
Before this existed, both functions had only test callers: nothing composed the complete
``DeploymentBundle`` they require, so no command could reach a provider.

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

from flash.serve.control import ModalCredentials, RunPodCredentials
from flash.serve.provisioning import InterruptedProvisioning

MODAL_TOKEN_ID_ENV = "MODAL_TOKEN_ID"
MODAL_TOKEN_SECRET_ENV = "MODAL_TOKEN_SECRET"
RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"
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
    """build request-scoped provider credentials from the environment."""

    if provider == "modal":
        return ModalCredentials(
            token_id=_required_env(MODAL_TOKEN_ID_ENV),
            token_secret=_required_env(MODAL_TOKEN_SECRET_ENV),
        )
    return RunPodCredentials(api_key=_required_env(RUNPOD_API_KEY_ENV))


def _image(reference: str):
    from flash.serve.provisioning import ServingImage

    if "@" not in reference:
        raise ValueError(
            "the serving image must be digest-qualified (name@sha256:...). a tag can be moved "
            "to different content after the deployment is recorded"
        )
    _, digest = reference.rsplit("@", 1)
    return ServingImage(reference=reference, digest=digest)


def _report(result) -> int:
    print(f"deployment  {result.deployment_id} generation {result.generation}")
    print(f"provider    {result.provider}")
    print(f"engine      {result.engine_id}")
    print(f"image       {result.image_digest}")
    print(f"status      {result.status}")
    handle = result.handle
    if handle is not None:
        print(f"endpoint    {handle.public_url}")
    if result.status == "ready":
        return 0
    if result.error_code:
        print(f"error       {result.error_code}", file=sys.stderr)
    if result.status == "outcome_unknown":
        # the provider may or may not hold live resources. saying "failed" here would invite a
        # retry that double-provisions and bills twice.
        print(
            "\nthe provider outcome could not be confirmed. resources may exist; reconcile "
            "before retrying rather than provisioning again.",
            file=sys.stderr,
        )
    return 1


def _build_provider_plan(provider: str, bundle) -> None:
    """run the provider's own plan validation without contacting it."""

    if provider == "modal":
        from flash.serve.provisioning._modal_plan import build_modal_create_plan

        build_modal_create_plan(bundle, phase="finalized")
        return
    from flash.serve.provisioning._runpod_plan import build_runpod_create_plan

    build_runpod_create_plan(bundle)


def cmd_serve_deploy(args) -> int:
    from flash.serve.control import DeploymentRequest
    from flash.serve.profiles import ProfileError, get_profile, placement_for
    from flash.serve.provisioning import DeploymentBundle, ServingRuntimeSecrets
    from flash.serve.resolve import execution_inputs, resolve_adapter

    provider = args.provider
    try:
        profile = get_profile(args.model)
        image = _image(args.image)
        placement = placement_for(
            profile,
            provider,
            workspace_name=getattr(args, "modal_workspace", "") or "",
            environment=getattr(args, "modal_environment", "") or "",
            region=getattr(args, "modal_region", "") or "",
            # "" means the no-suffix environment, which is a real modal configuration, so it maps
            # to None rather than being rejected as missing.
            web_suffix=(getattr(args, "modal_web_suffix", "") or "") or None,
            account_id=getattr(args, "runpod_account", "") or "",
            data_center_id=getattr(args, "runpod_data_center", "") or "",
        )
    except (ProfileError, ValueError) as exc:
        return _err(str(exc))

    try:
        from flash.serve.resolve import resolve_base_revision

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
    # `ResolveError` subclasses `ValueError`, so this still reports resolution failures the same
    # way. the wider catch also covers validation raised beneath the resolver -- a negative
    # `--checkpoint-step` reaches `format_adapter_revision`, and a nonimmutable revision reaches
    # `ResolvedAdapter`, both of which raise plain `ValueError`. those are bad user input, so they
    # belong on the normal cli error path rather than the unexpected-error traceback.
    except ValueError as exc:
        return _err(str(exc))

    try:
        engine = profile.engine(
            model_revision=base_revision,
            tokenizer_revision=base_revision,
            image=image,
        )
        request = DeploymentRequest(
            deployment_id=args.deployment_id,
            generation=args.generation,
            provider=provider,
            placement=placement,
            engine=engine,
            adapters=(resolved.adapter,),
        )
        from flash.server.domain.serving_resources import (
            dry_run_deployment,
            resolve_deployment_bundle,
        )

        bundle: DeploymentBundle = resolve_deployment_bundle(
            request,
            execution_inputs(profile, image, (resolved,)),
            image,
        )
    except (ValueError, TypeError) as exc:
        return _err(str(exc))

    try:
        # build the provider plan too, not just the bundle. the generic placement types accept
        # any nonempty string, while the provider-specific `_validate_placement` is what
        # enforces the hostname charset -- so an uppercase workspace or an underscored region
        # passed the dry run and then raised an uncaught ValueError during a real deployment,
        # after hub resolution. "validated every input" has to mean the ones the provider
        # applies. this runs on both paths: gating it on `--dry-run` meant the real deployment
        # -- the one that allocates and bills -- was the only path that surfaced a raw traceback.
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
        # the artifact token is passed rather than omitted whenever the operator has one. it is
        # optional because a public artifact repo needs none, and the provisioning layer treats
        # absence as "hydration cannot run": with a token it uses the two-phase bootstrap that
        # deploys with the secret, hydrates the volume, then redeploys without it and deletes the
        # secret, so the token never outlives hydration.
        runtime_secrets = ServingRuntimeSecrets(
            inference_token=_required_env(INFERENCE_KEY_ENV),
            artifact_token=_optional_env(ARTIFACT_TOKEN_ENV),
        )
    except ValueError as exc:
        return _err(
            f"{exc}. provider credentials are read from the environment for this one request "
            f"and are never stored"
        )

    if _optional_env(ARTIFACT_TOKEN_ENV) is None:
        # a fresh volume holds neither adapters nor base weights, so the container hydrates both
        # from the hub on its first start -- and that path is token-only end to end:
        # `launch._prepare_cache` rejects every cache miss outright when the artifact token is
        # None, and both hydration functions read the token from a descriptor that
        # `read_artifact_token_fd` refuses to treat as empty. a public repository does not change
        # that, so permitting the deploy would create and bill provider resources for a container
        # that cannot reach readiness. reject it here instead, where nothing has been created.
        return _err(
            f"{ARTIFACT_TOKEN_ENV} is not set. a new deployment hydrates its serving cache from "
            f"the hub before the engine starts, and that hydration requires a token even when "
            f"the repositories are public"
        )

    deadline_at = time.monotonic() + float(args.timeout)
    try:
        if provider == "modal":
            from flash.serve.provisioning.modal import provision_modal_deployment

            result = provision_modal_deployment(
                bundle, credentials, runtime_secrets, deadline_at=deadline_at
            )
        else:
            from flash.serve.provisioning.runpod import provision_runpod_deployment

            result = provision_runpod_deployment(
                bundle, credentials, runtime_secrets, deadline_at=deadline_at
            )
    except InterruptedProvisioning as interrupted:
        # the interrupt still propagates -- the user pressed Ctrl-C and the exit code must say so.
        # but the generic handler prints only "aborted", which reads as "nothing was created",
        # and here something was: cleanup ran and could not confirm the resources are gone.
        _warn_unconfirmed_cleanup(interrupted.provider, args.deployment_id)
        raise
    return _report(result)


def _warn_unconfirmed_cleanup(provider: str, deployment_id: str) -> None:
    """say what ctrl-c actually left behind, before the generic handler says "aborted"."""

    print(
        f"\nwarning: interrupted before ready, and {provider} cleanup could not be confirmed for "
        f"{deployment_id}. resources may still exist and bill; reconcile before retrying rather "
        f"than provisioning again.",
        file=sys.stderr,
    )
