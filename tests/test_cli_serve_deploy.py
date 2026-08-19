"""`serve deploy`: provider routing, credential handling, and fail-closed validation."""

from __future__ import annotations

import argparse

import pytest

from flash.cli.commands import serve_deploy
from flash.cli.commands.serve_deploy import cmd_serve_deploy
from flash.cli.serve_parser import _add_serve_commands

DIGEST = "sha256:" + "a" * 64
IMAGE = f"ghcr.io/freesolo-co/freesolo-flash-serve@{DIGEST}"
MODEL = "Qwen/Qwen3.5-9B"


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_serve_commands(sub)
    return parser.parse_args(argv)


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
    ]
    parsed = _parse(base)
    for key, value in overrides.items():
        setattr(parsed, key, value)
    return parsed


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


def test_deploy_routes_to_the_named_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_modal(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        calls.append("modal")
        return _result(bundle)

    def _fake_runpod(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        calls.append("runpod")
        return _result(bundle)

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _fake_modal)
    monkeypatch.setattr("flash.serve.provisioning.runpod.provision_runpod_deployment", _fake_runpod)

    assert cmd_serve_deploy(_args()) == 0
    assert (
        cmd_serve_deploy(
            _args(
                provider="runpod",
                modal_workspace="",
                modal_environment="",
                runpod_account="account",
                runpod_data_center="US-KS-2",
            )
        )
        == 0
    )

    assert calls == ["modal", "runpod"]


def test_dry_run_contacts_no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(*_args, **_kwargs):
        raise AssertionError("a dry run must not reach the provider")

    _stub_resolution(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _explode)
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
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _explode)
    monkeypatch.delenv(serve_deploy.MODAL_TOKEN_ID_ENV, raising=False)
    monkeypatch.delenv(serve_deploy.MODAL_TOKEN_SECRET_ENV, raising=False)
    monkeypatch.setenv(serve_deploy.INFERENCE_KEY_ENV, "inference-key")

    assert cmd_serve_deploy(_args()) == 1


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


def test_unknown_model_is_refused_before_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(**_kwargs):
        raise AssertionError("resolution ran for an unsupported model")

    monkeypatch.setattr("flash.serve.resolve.resolve_adapter", _explode)

    assert cmd_serve_deploy(_args(model="Qwen/Qwen3.5-0.8B")) == 1


def test_outcome_unknown_is_not_reported_as_a_plain_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # a retry after an unconfirmed outcome can double-provision and bill twice.
    def _unknown(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        from flash.serve.control import DeploymentResult

        return DeploymentResult.from_spec(
            bundle.spec, status="outcome_unknown", error_code="transport_failed"
        )

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _unknown)

    assert cmd_serve_deploy(_args()) == 1
    captured = capsys.readouterr()
    assert "outcome_unknown" in captured.out
    assert "reconcile" in captured.err


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

    from flash.serve.control import (
        DeploymentResult,
        ModalProviderHandle,
        RunPodProviderHandle,
    )

    spec = bundle.spec
    common = {
        "deployment_id": spec.deployment_id,
        "generation": spec.generation,
        "engine_id": spec.engine.engine_id,
        "image_digest": spec.engine.image_digest,
    }
    if spec.provider == "modal":
        handle = ModalProviderHandle(
            workspace_name=spec.placement.workspace_name,
            app_id="ap-" + "1" * 22,
            app_name="flash-app-test",
            volume_id="vo-" + "1" * 22,
            volume_name="flash-volume-test",
            inference_secret_id="st-" + "1" * 22,
            inference_secret_name="flash-inference-secret-test",
            environment=spec.placement.environment,
            region=None,
            public_url="https://workspace--flash-app-test.modal.run",
            **common,
        )
    else:
        handle = RunPodProviderHandle(
            account_id=spec.placement.account_id,
            pod_id="pod1234567890",
            pod_name="flash-pod-test",
            network_volume_id="vol1234567890",
            network_volume_name="flash-volume-test",
            template_id="tpl1234567890",
            template_name="flash-template-test",
            inference_secret_id="sec1234567890",
            inference_secret_name="flash-inference-secret-test",
            data_center_id=spec.placement.data_center_id,
            public_url="https://pod1234567890-8000.proxy.runpod.net",
            **common,
        )
    return DeploymentResult.from_spec(spec, status="ready", handle=handle)


def _stub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_ID_ENV, "token-id")
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_SECRET_ENV, "token-secret")
    monkeypatch.setenv(serve_deploy.RUNPOD_API_KEY_ENV, "api-key")
    monkeypatch.setenv(serve_deploy.INFERENCE_KEY_ENV, "inference-key")


def _stub_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve against fixed hub facts so the command path is tested without the network."""

    from flash.serve.app import AdapterExecutionInput, ArtifactFile, aggregate_file_digest
    from flash.serve.control import AdapterAliasIntent, ResolvedAdapter
    from flash.serve.resolve import ResolvedDeploymentInputs

    artifact_revision = "c" * 40
    files = (
        ArtifactFile("adapter_config.json", 1308, "1" * 64),
        ArtifactFile("adapter_model.safetensors", 43346432, "2" * 64),
    )
    revision = f"run1@final.{artifact_revision}"

    def _fake_base_revision(model_id: str) -> str:
        return "d" * 40

    def _fake_resolve(**kwargs) -> ResolvedDeploymentInputs:
        adapter = ResolvedAdapter(
            run_id=kwargs["run_id"],
            checkpoint="final",
            adapter_revision=revision,
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
            alias_intent=AdapterAliasIntent(activate=True, expected_adapter_revision=None),
        )
        return ResolvedDeploymentInputs(
            adapter=adapter,
            execution=AdapterExecutionInput(adapter_revision=revision, files=files),
        )

    monkeypatch.setattr("flash.serve.resolve.resolve_adapter", _fake_resolve)
    monkeypatch.setattr("flash.serve.resolve.resolve_base_revision", _fake_base_revision)
