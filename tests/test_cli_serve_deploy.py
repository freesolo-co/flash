"""`serve deploy`: provider routing, credential handling, and fail-closed validation."""

from __future__ import annotations

import argparse

import pytest

from flash.cli.commands.serving import deploy as serve_deploy
from flash.cli.commands.serving.deploy import cmd_serve_deploy
from flash.cli.commands.serving.identity import encode_deployment_identity
from flash.cli.parsing.serve_parser import _add_serve_commands
from flash.serve.deployment.profiles import get_profile
from flash.serve.provisioning import InterruptedProvisioning

DIGEST = "sha256:" + "a" * 64
IMAGE = f"ghcr.io/freesolo-co/freesolo-flash-serve@{DIGEST}"
CERTIFIED_DIGEST = "sha256:2bf27b51f6e4b7f0b2d805d96202579d94868e2c594b7c496777d350ad6936f6"
CERTIFIED_IMAGE = f"ghcr.io/freesolo-co/freesolo-flash-serve@{CERTIFIED_DIGEST}"
MODEL = "Qwen/Qwen3.5-9B"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_serve_commands(sub)
    return parser.parse_args(argv)


def _required_deploy_argv() -> list[str]:
    return [
        "serve",
        "deploy",
        "--provider",
        "modal",
        "--model",
        MODEL,
        "--run",
        "run1",
        "--deployment-id",
        "deployment1",
        "--image",
        IMAGE,
        "--artifact-repo",
        "Freesolo-Co/artifacts",
        "--artifact-subfolder",
        "adapter",
    ]


def _args(**overrides) -> argparse.Namespace:
    base = [
        "serve",
        "deploy",
        "--provider",
        "modal",
        "--model",
        MODEL,
        "--run",
        "run1",
        "--deployment-id",
        "deployment1",
        "--image",
        IMAGE,
        "--artifact-repo",
        "Freesolo-Co/artifacts",
        "--artifact-subfolder",
        "rl/run1/seed0/adapter",
        "--lora-rank",
        "32",
        "--modal-workspace",
        "workspace",
        "--modal-environment",
        "dev",
        "--modal-region",
        "us-east",
    ]
    parsed = _parse(base)
    for key, value in overrides.items():
        setattr(parsed, key, value)
    return parsed


def test_dry_run_contacts_no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*_args, **_kwargs):
        raise AssertionError("a dry run must not reach the provider")

    _stub_resolution(monkeypatch)
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _explode,
    )
    # deliberately no credentials in the environment: a dry run must not require them either.
    for name in (
        serve_deploy.MODAL_TOKEN_ID_ENV,
        serve_deploy.MODAL_TOKEN_SECRET_ENV,
        serve_deploy.INFERENCE_KEY_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    assert cmd_serve_deploy(_args(dry_run=True)) == 0


def test_missing_credentials_fail_before_any_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*_args, **_kwargs):
        raise AssertionError("provisioning ran without credentials")

    _stub_resolution(monkeypatch)
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _explode,
    )
    monkeypatch.delenv(serve_deploy.MODAL_TOKEN_ID_ENV, raising=False)
    monkeypatch.delenv(serve_deploy.MODAL_TOKEN_SECRET_ENV, raising=False)
    monkeypatch.setenv(serve_deploy.INFERENCE_KEY_ENV, "inference-key")

    assert cmd_serve_deploy(_args()) == 1


def test_hub_token_reaches_provisioning_so_the_container_can_hydrate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the container hydrates its adapter from the hub itself, and artifact repos are private by
    # default. the command resolved the adapter with HF_TOKEN and then built the runtime secrets
    # without it, so every real deployment reached the launcher with no way to authenticate and
    # died in `_prepare_cache` after the provider had created and started billing for the app,
    # volume, and secrets. nothing offline noticed because no test read the secrets the
    # command actually passed.
    seen: list[tuple[str, str | None]] = []

    def _capture(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        seen.append(secrets._reveal_for_launch())
        return _result(bundle)

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, "hf-token")
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _capture,
    )

    assert cmd_serve_deploy(_args()) == 0
    assert seen == [("inference-key", "hf-token")]


def test_the_deploy_proceeds_once_a_token_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the guard must reject only the unhydratable case: with a token the deploy runs and that
    # token reaches provisioning so the container can hydrate.
    seen: list[tuple[str, str | None]] = []

    def _capture(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        seen.append(secrets._reveal_for_launch())
        return _result(bundle)

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, "hf-token")
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _capture,
    )

    assert cmd_serve_deploy(_args()) == 0
    assert seen == [("inference-key", "hf-token")]


def test_resolver_validation_failures_are_cli_errors_not_tracebacks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # bad user input reaches validation below the resolver: a negative --checkpoint-step and an
    # invalid immutable checkpoint both raise from resolver-owned validation
    # as plain ValueError. catching only ResolveError let those escape as an unexpected-error
    # traceback after the artifact files had already been downloaded.
    _stub_environment(monkeypatch)

    def _raise_plain(**_kwargs):
        raise ValueError("invalid immutable adapter revision components")

    monkeypatch.setattr("flash.serve.deployment.resolve.resolve_adapter", _raise_plain)
    monkeypatch.setattr(
        "flash.serve.deployment.resolve.resolve_base_revision", lambda *_a, **_k: "d" * 40
    )

    assert cmd_serve_deploy(_args()) == 1
    assert "invalid immutable adapter revision components" in capsys.readouterr().err


def test_credentials_are_never_command_arguments() -> None:
    # a process list and shell history are both readable, so the token must not be expressible as
    # a flag even by mistake.
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_serve_commands(sub)
    text = parser.format_help() + _deploy_help(parser)

    for forbidden in ("--token", "--api-key", "--secret", "--password", "--credential"):
        assert forbidden not in text


def test_tag_only_image_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # a tag can be repointed at different content after the deployment record is written.
    _stub_resolution(monkeypatch)

    assert cmd_serve_deploy(_args(image="ghcr.io/freesolo-co/freesolo-flash-serve:dev")) == 1


@pytest.mark.parametrize(
    "retired_model",
    ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B"],
)
def test_removed_model_is_refused_before_resolution(
    monkeypatch: pytest.MonkeyPatch, retired_model: str
) -> None:
    def _explode(**_kwargs):
        raise AssertionError("resolution ran for an unsupported model")

    monkeypatch.setattr("flash.serve.deployment.resolve.resolve_adapter", _explode)

    assert cmd_serve_deploy(_args(model=retired_model)) == 1


@pytest.mark.parametrize("mutation", ["missing", "extra", "drifted", "structurally_invalid"])
def test_registry_inconsistency_fails_before_artifact_resolution(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from dataclasses import replace

    from flash.serve.deployment import profiles

    def _explode(**_kwargs):
        raise AssertionError("artifact resolution ran with an inconsistent profile registry")

    monkeypatch.setattr("flash.serve.deployment.resolve.resolve_adapter", _explode)
    if mutation == "missing":
        monkeypatch.delitem(profiles._PROFILES, "Qwen/Qwen3.8-27B")
    elif mutation == "extra":
        monkeypatch.setitem(
            profiles._PROFILES,
            "extra/model",
            replace(get_profile(MODEL), model_id="extra/model"),
        )
    elif mutation == "drifted":
        monkeypatch.setitem(
            profiles._PROFILES,
            MODEL,
            replace(get_profile(MODEL), max_model_len=65536),
        )
    else:
        monkeypatch.setitem(
            profiles._PROFILES,
            MODEL,
            replace(get_profile(MODEL), modal_live_qualified=1),
        )

    assert cmd_serve_deploy(_args(dry_run=True)) == 1


@pytest.mark.parametrize(
    ("model", "expected_calls", "expected_model_revision", "expected_tokenizer_revision"),
    [
        (
            "Qwen/Qwen3.5-9B",
            ["Qwen/Qwen3.5-9B", "Freesolo-Co/Qwen3.5-9B-FP8"],
            "b" * 40,
            "b" * 40,
        ),
        (
            "Qwen/Qwen3.8-27B",
            ["Qwen/Qwen3.8-27B"],
            "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
            "a" * 40,
        ),
        (
            "Qwen/Qwen3.6-35B-A3B",
            ["Qwen/Qwen3.6-35B-A3B"],
            "a" * 40,
            "a" * 40,
        ),
    ],
)
def test_revision_resolution_reuses_only_matching_repositories(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected_calls: list[str],
    expected_model_revision: str,
    expected_tokenizer_revision: str,
) -> None:
    _stub_resolution(monkeypatch)
    calls: list[str] = []
    revisions = {
        "Qwen/Qwen3.5-9B": "a" * 40,
        "Freesolo-Co/Qwen3.5-9B-FP8": "b" * 40,
        "Qwen/Qwen3.8-27B": "a" * 40,
        "Qwen/Qwen3.6-35B-A3B": "a" * 40,
    }

    def _resolve(model_id: str) -> str:
        calls.append(model_id)
        return revisions[model_id]

    monkeypatch.setattr("flash.serve.deployment.resolve.resolve_base_revision", _resolve)
    bundle = serve_deploy._deployment_bundle(_args(model=model, dry_run=True))

    assert calls == expected_calls
    assert bundle.spec.engine.model_revision == expected_model_revision
    assert bundle.spec.engine.tokenizer_revision == expected_tokenizer_revision


def test_tokenizer_uses_the_adapter_resolved_logical_base_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    _stub_resolution(monkeypatch)
    from flash.serve.deployment import resolve as resolver

    original = resolver.resolve_adapter

    def _resolve(**kwargs):
        resolved = original(**kwargs)
        return replace(
            resolved,
            adapter=replace(resolved.adapter, base_model_revision="e" * 40),
        )

    monkeypatch.setattr(resolver, "resolve_adapter", _resolve)
    bundle = serve_deploy._deployment_bundle(_args(model="Qwen/Qwen3.8-27B", dry_run=True))

    assert bundle.manifest.logical_base_revision == "e" * 40
    assert bundle.spec.engine.model_revision == "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
    assert bundle.spec.engine.tokenizer_revision == "e" * 40


@pytest.mark.parametrize("model", ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B"])
def test_certified_modal_image_reaches_resolution_before_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model: str,
) -> None:
    _stub_resolution(monkeypatch)
    monkeypatch.delenv(serve_deploy.MODAL_TOKEN_ID_ENV, raising=False)
    monkeypatch.delenv(serve_deploy.MODAL_TOKEN_SECRET_ENV, raising=False)

    assert cmd_serve_deploy(_args(model=model, provider="modal", image=CERTIFIED_IMAGE)) == 1
    error = capsys.readouterr().err
    assert "MODAL_TOKEN_ID is not set" in error
    assert "pending exact live qualification" not in error
    assert "qualified only for certified image digest" not in error


@pytest.mark.parametrize("model", ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B"])
def test_uncertified_modal_image_fails_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model: str,
) -> None:
    def _explode(_args):
        raise AssertionError("bundle resolution ran before image qualification")

    monkeypatch.setattr(serve_deploy, "_deployment_bundle", _explode)

    assert cmd_serve_deploy(_args(model=model, provider="modal", image=IMAGE)) == 1
    error = capsys.readouterr().err
    assert f"{model} modal serving profile is qualified only for certified image digest" in error
    assert f"requested {DIGEST}" in error


@pytest.mark.parametrize("model", ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B"])
def test_synthetic_unqualified_modal_profile_still_fails_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
) -> None:
    from dataclasses import replace

    from flash.serve.deployment import profiles

    def _explode(**_kwargs):
        raise AssertionError("resolution ran before live qualification")

    profile = get_profile(model)
    monkeypatch.setitem(
        profiles._PROFILES,
        model,
        replace(profile, modal_live_qualified=False),
    )
    monkeypatch.setattr("flash.serve.deployment.resolve.resolve_adapter", _explode)

    assert cmd_serve_deploy(_args(model=model, provider="modal", image=CERTIFIED_IMAGE)) == 1


def test_outcome_unknown_is_not_reported_as_a_plain_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # a retry after an unconfirmed outcome can double-provision and bill twice.
    identities: list[str] = []

    def _unknown(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        from flash.serve.control import DeploymentResult

        identities.append(encode_deployment_identity(bundle))
        return DeploymentResult.from_spec(
            bundle.spec,
            status="outcome_unknown",
            error_code="transport_failed",
            error_reason="readiness_deadline_unproven",
        )

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _unknown,
    )

    assert cmd_serve_deploy(_args()) == 1
    captured = capsys.readouterr()
    assert "outcome_unknown" in captured.out
    assert f"identity    {identities[0]}\n" in captured.out
    assert "reason      readiness_deadline_unproven" in captured.err
    assert "flash serve status" in captured.err
    assert "flash serve undeploy" in captured.err

    def _encoding_fails(_bundle):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(
        "flash.cli.commands.serving.identity.encode_deployment_identity", _encoding_fails
    )
    assert cmd_serve_deploy(_args()) == 1
    assert "outcome_unknown" in capsys.readouterr().out


def test_identity_reporting_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _interrupted(_bundle):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "flash.cli.commands.serving.identity.encode_deployment_identity", _interrupted
    )

    with pytest.raises(KeyboardInterrupt):
        serve_deploy._report_identity(object())


@pytest.mark.parametrize(
    ("error_code", "error_reason"),
    [
        ("provider_rejected", "artifact_cleanup_delete_rejected"),
        ("resource_ambiguous", "artifact_cleanup_conflict"),
        ("resource_ambiguous", "artifact_cleanup_observation_failed"),
        ("resource_ambiguous", "artifact_cleanup_delete_unknown"),
    ],
)
def test_modal_artifact_cleanup_result_warns_that_the_app_is_live_and_billing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_code: str,
    error_reason: str,
) -> None:
    from flash.serve.control import DeploymentResult

    def _cleanup_failure(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        ready = _result(bundle)
        return DeploymentResult.from_spec(
            bundle.spec,
            status="failed" if error_code == "provider_rejected" else "outcome_unknown",
            handle=ready.handle,
            error_code=error_code,
            error_reason=error_reason,
        )

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _cleanup_failure,
    )

    assert cmd_serve_deploy(_args(provider="modal")) == 1
    captured = capsys.readouterr()
    assert "app ap-1111111111111111111111" in captured.err
    assert "app is live and billing" in captured.err
    assert "flash serve status" in captured.err
    assert "flash serve undeploy" in captured.err


def test_interrupted_deploy_prints_recovery_identity_without_masking_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    identities: list[str] = []

    def _interrupted(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        identities.append(encode_deployment_identity(bundle))
        raise InterruptedProvisioning("modal")

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _interrupted,
    )

    with pytest.raises(InterruptedProvisioning):
        cmd_serve_deploy(_args())

    captured = capsys.readouterr()
    assert captured.out.startswith("identity    ")
    assert captured.out == f"identity    {identities[0]}\n"
    assert "flash serve status" in captured.err
    assert "flash serve undeploy" in captured.err

    def _encoding_fails(_bundle):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(
        "flash.cli.commands.serving.identity.encode_deployment_identity", _encoding_fails
    )
    with pytest.raises(InterruptedProvisioning):
        cmd_serve_deploy(_args())


def _deploy_help(parser: argparse.ArgumentParser) -> str:
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        serve = action.choices.get("serve")
        if serve is None:
            continue
        for serve_action in serve._actions:
            if isinstance(serve_action, argparse._SubParsersAction):
                deploy = serve_action.choices.get("deploy")
                if deploy is not None:
                    return deploy.format_help()
    return ""


def _result(bundle):
    """a ready result must carry the provider handle its own validator requires."""

    from flash.serve.control import DeploymentResult, ModalProviderHandle

    spec = bundle.spec
    common = {
        "deployment_id": spec.deployment_id,
        "generation": spec.generation,
        "engine_id": spec.engine.engine_id,
        "image_digest": spec.engine.image_digest,
    }
    handle = ModalProviderHandle(
        workspace_name=spec.placement.workspace_name,
        app_id="ap-" + "1" * 22,
        app_name="flash-app-test",
        volume_id="vo-" + "1" * 22,
        volume_name="flash-volume-test",
        inference_secret_id="st-" + "1" * 22,
        inference_secret_name="flash-inference-secret-test",
        environment=spec.placement.environment,
        # from the spec, not a literal: the handle is validated against the planned placement,
        # so a hardcoded region silently becomes a mismatch the moment the plan carries one.
        region=spec.placement.region,
        public_url="https://workspace--flash-app-test.modal.run",
        **common,
    )
    return DeploymentResult.from_spec(spec, status="ready", handle=handle)


def _stub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_ID_ENV, "token-id")
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_SECRET_ENV, "token-secret")
    monkeypatch.setenv(serve_deploy.INFERENCE_KEY_ENV, "inference-key")
    # a deployable environment includes the artifact token: hydration on a fresh volume is
    # token-only, so the command rejects a missing one before contacting any provider. tests
    # about that rejection unset it explicitly.
    monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, "hf-token")


def _stub_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve against fixed hub facts so the command path is tested without the network."""

    from flash.schema import format_checkpoint_ref
    from flash.serve.app import AdapterExecutionInput, ArtifactFile, aggregate_file_digest
    from flash.serve.control import ResolvedAdapter
    from flash.serve.deployment.resolve import ResolvedDeploymentInputs

    artifact_revision = "c" * 40
    files = (
        ArtifactFile("adapter_config.json", 1308, "1" * 64),
        ArtifactFile("adapter_model.safetensors", 43346432, "2" * 64),
    )
    checkpoint_id = format_checkpoint_ref("run1", None)

    def _fake_base_revision(model_id: str) -> str:
        return "d" * 40

    def _fake_resolve(**kwargs) -> ResolvedDeploymentInputs:
        adapter = ResolvedAdapter(
            run_id=kwargs["run_id"],
            checkpoint_id=checkpoint_id,
            artifact_repo_id=kwargs["artifact_repo_id"],
            artifact_repo_type="dataset",
            artifact_revision=artifact_revision,
            artifact_digest=aggregate_file_digest(files),
            artifact_subfolder=kwargs["artifact_subfolder"],
            base_model=kwargs["base_model"],
            base_model_revision=kwargs["base_model_revision"],
            lora_rank=kwargs["lora_rank"],
            thinking_default=bool(kwargs.get("thinking_default", False)),
            structured_outputs_default_json=None,
        )
        return ResolvedDeploymentInputs(
            adapter=adapter,
            execution=AdapterExecutionInput(checkpoint_id=checkpoint_id, files=files),
        )

    monkeypatch.setattr("flash.serve.deployment.resolve.resolve_adapter", _fake_resolve)
    monkeypatch.setattr("flash.serve.deployment.resolve.resolve_base_revision", _fake_base_revision)


def _historical_identity(
    monkeypatch: pytest.MonkeyPatch, args: argparse.Namespace, retired_model: str
) -> str:
    from dataclasses import replace

    from flash.cli.commands.serving.identity import encode_deployment_identity
    from flash.serve.app import AdapterExecutionInput, ExecutionInputs, build_serving_manifest
    from flash.serve.control import DeploymentSpec
    from flash.serve.provisioning import DeploymentBundle

    _stub_resolution(monkeypatch)
    current = serve_deploy._deployment_bundle(args)
    engine = replace(
        current.spec.engine,
        served_model=retired_model,
        tokenizer_model=retired_model,
    )
    adapters = tuple(
        replace(adapter, base_model=retired_model) for adapter in current.spec.adapters
    )
    spec = DeploymentSpec(
        deployment_id=current.spec.deployment_id,
        generation=current.spec.generation,
        provider=current.spec.provider,
        placement=current.spec.placement,
        engine=engine,
        adapters=adapters,
    )
    inputs = ExecutionInputs(
        expected_oci_digest=current.manifest.expected_oci_digest,
        engine_args=current.manifest.engine_args,
        tokenizer_kwargs=current.manifest.tokenizer_kwargs,
        processor_kwargs=current.manifest.processor_kwargs,
        adapters=tuple(
            AdapterExecutionInput(checkpoint_id=entry.checkpoint_id, files=entry.files)
            for entry in current.manifest.adapters
        ),
    )
    historical = DeploymentBundle(
        spec=spec,
        manifest=build_serving_manifest(spec, inputs),
        image=current.image,
    )
    return encode_deployment_identity(historical)


@pytest.mark.parametrize("bad", ["0", "-1", "-0.5", "nan", "inf", "-inf"])
def test_timeout_that_cannot_describe_a_future_deadline_is_rejected_at_parse(bad: str) -> None:
    """A bad `--timeout` must fail as an argument error, before any resolution or download.

    `time.monotonic() + float(args.timeout)` turns each of these into an already-expired or
    non-finite deadline, which the provider's `_validate_deadline` raises `ValueError` for -- outside
    the lifecycle error handling, so the user saw a traceback after the Hub inputs had already been
    resolved and downloaded.
    """
    with pytest.raises(SystemExit):
        _parse(
            [
                "serve",
                "deploy",
                "--provider",
                "modal",
                "--model",
                MODEL,
                "--run",
                "run1",
                "--deployment-id",
                "deployment1",
                "--image",
                IMAGE,
                "--artifact-repo",
                "Freesolo-Co/artifacts",
                "--artifact-subfolder",
                "adapter",
                "--lora-rank",
                "32",
                "--timeout",
                bad,
            ]
        )


def test_a_positive_finite_timeout_is_accepted() -> None:
    # parsed, not set post-hoc: _args assigns attributes after parsing, which would skip the
    # argparse type entirely and pass no matter what the type does.
    args = _parse(
        [
            "serve",
            "deploy",
            "--provider",
            "modal",
            "--model",
            MODEL,
            "--run",
            "run1",
            "--deployment-id",
            "deployment1",
            "--image",
            IMAGE,
            "--artifact-repo",
            "Freesolo-Co/artifacts",
            "--artifact-subfolder",
            "adapter",
            "--lora-rank",
            "32",
            "--timeout",
            "0.5",
        ]
    )
    assert args.timeout == 0.5


@pytest.mark.parametrize(
    ("field", "bad"),
    [("modal_workspace", "UPPER"), ("modal_region", "us_east"), ("modal_web_suffix", "dev_test")],
)
def test_dry_run_rejects_placement_a_real_deployment_would_reject(
    monkeypatch: pytest.MonkeyPatch, field: str, bad: str
) -> None:
    """A dry run that prints success must not be followed by a deploy that dies on the same input.

    `ModalPlacement` and `placement_for` accept any nonempty string; the hostname charset is
    enforced by `_validate_placement` inside `build_modal_create_plan`, which a dry run never
    reached. So `--dry-run` reported "validated every input" for an uppercase workspace or an
    underscored region, and the real deployment then raised an uncaught `ValueError` -- as a
    traceback, after Hub resolution had already run.
    """
    _stub_resolution(monkeypatch)

    def _explode(*_args, **_kwargs):
        raise AssertionError("a dry run must not reach the provider")

    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _explode,
    )

    assert cmd_serve_deploy(_args(dry_run=True, **{field: bad})) == 1


@pytest.mark.parametrize("bad", ["0", "-1", "-999"])
def test_a_generation_that_cannot_order_deployments_is_rejected_at_parse(bad: str) -> None:
    """A bad `--generation` must fail as an argument error, before any resolution or download.

    Same shape as the `--timeout` case above, one step earlier in the command. A bare `type=int`
    accepts `0` and negatives, and `_require_positive_int` does not see the value until
    `DeploymentRequest` is constructed -- which is after `resolve_adapter` has already resolved and
    downloaded the Hub inputs. The eventual error is a clean one rather than a traceback, so the
    cost is a wasted Hub round trip, not a crash; argparse can refuse it outright for nothing.
    """
    with pytest.raises(SystemExit):
        _parse(
            [
                "serve",
                "deploy",
                "--provider",
                "modal",
                "--model",
                MODEL,
                "--run",
                "run1",
                "--deployment-id",
                "deployment1",
                "--image",
                IMAGE,
                "--artifact-repo",
                "Freesolo-Co/artifacts",
                "--artifact-subfolder",
                "adapter",
                "--lora-rank",
                "32",
                "--generation",
                bad,
            ]
        )


def test_a_positive_generation_is_accepted() -> None:
    # parsed, not set post-hoc: _args assigns attributes after parsing, which would skip the
    # argparse type entirely and pass no matter what the type does.
    args = _parse([*_required_deploy_argv(), "--lora-rank", "32", "--generation", "3"])
    assert args.generation == 3


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_a_nonpositive_lora_rank_is_rejected_at_parse(bad: str) -> None:
    with pytest.raises(SystemExit):
        _parse([*_required_deploy_argv(), "--lora-rank", bad])


@pytest.mark.parametrize("bad", ["0", "-1", "-999"])
def test_a_nonpositive_checkpoint_step_is_rejected_at_parse(bad: str) -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                *_required_deploy_argv(),
                "--lora-rank",
                "32",
                "--checkpoint-step",
                bad,
            ]
        )


def test_provider_is_required_and_restricted_to_supported_providers() -> None:
    # vast, lambda, kubernetes, and serverless runpod are explicitly out of scope, so an
    # unsupported provider must be refused by the parser rather than reaching a provider call.
    with pytest.raises(SystemExit):
        _parse(
            [
                "serve",
                "deploy",
                "--provider",
                "vast",
                "--model",
                MODEL,
                "--run",
                "run1",
                "--deployment-id",
                "deployment1",
                "--image",
                IMAGE,
                "--artifact-repo",
                "Freesolo-Co/artifacts",
                "--artifact-subfolder",
                "adapter",
                "--lora-rank",
                "32",
            ]
        )


def test_tokenless_adoption_reaches_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str | None]] = []

    def _adopt(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        seen.append(secrets._reveal_for_launch())
        return _result(bundle)

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, "   ")
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _adopt,
    )

    assert serve_deploy._optional_env(serve_deploy.ARTIFACT_TOKEN_ENV) is None
    assert cmd_serve_deploy(_args()) == 0
    assert seen == [("inference-key", None)]


def test_tokenless_fresh_create_rejection_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _reject(*_args, **_kwargs):
        raise serve_deploy.FreshDeploymentArtifactTokenRequired

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.delenv(serve_deploy.ARTIFACT_TOKEN_ENV, raising=False)
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _reject,
    )

    assert cmd_serve_deploy(_args()) == 1
    assert capsys.readouterr().err == (
        f"error: {serve_deploy.ARTIFACT_TOKEN_ENV} is not set. a new deployment hydrates its "
        "serving cache from the hub before the engine starts, and that hydration requires a token "
        "even when the repositories are public\n"
    )


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B"],
)
def test_every_catalog_profile_builds_a_provider_free_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model: str,
) -> None:
    _stub_resolution(monkeypatch)

    assert cmd_serve_deploy(_args(model=model, dry_run=True)) == 0
    output = capsys.readouterr().out
    assert "dry run: no provider was contacted" in output
    assert "engine_id" in output


def test_9b_qualification_remains_image_digest_agnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)

    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        lambda bundle, credentials, secrets, *, deadline_at, **_kwargs: _result(bundle),
    )

    assert cmd_serve_deploy(_args(model=MODEL, image=IMAGE)) == 0
