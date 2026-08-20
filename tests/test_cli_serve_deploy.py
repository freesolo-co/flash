"""`serve deploy`: provider routing, credential handling, and fail-closed validation."""

from __future__ import annotations

import argparse
import pathlib
import re
import tomllib

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
        "--modal-region",
        "us-east",
    ]
    parsed = _parse(base)
    for key, value in overrides.items():
        setattr(parsed, key, value)
    # the base args are modal's, so overriding provider="runpod" would otherwise leave modal
    # placement fields set and placement_for would reject them as foreign inputs -- a fixture
    # artifact, not the behavior under test. each provider keeps only its own fields.
    if parsed.provider == "runpod":
        parsed.modal_workspace = ""
        parsed.modal_environment = ""
        parsed.modal_region = ""
        parsed.runpod_account = parsed.runpod_account or "account1"
        parsed.runpod_data_center = parsed.runpod_data_center or "US-KS-2"
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


def test_hub_token_reaches_provisioning_so_the_container_can_hydrate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the container hydrates its adapter from the hub itself, and artifact repos are private by
    # default. the command resolved the adapter with HF_TOKEN and then built the runtime secrets
    # without it, so every real deployment reached the launcher with no way to authenticate and
    # died in `_prepare_cache` -- after the provider had created and started billing for the app,
    # volume, secret, and pod. nothing offline noticed because no test read the secrets the
    # command actually passed.
    seen: list[tuple[str, str | None]] = []

    def _capture(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        seen.append(secrets._reveal_for_launch())
        return _result(bundle)

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, "hf-token")
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _capture)

    assert cmd_serve_deploy(_args()) == 0
    assert seen == [("inference-key", "hf-token")]


def test_absent_hub_token_stays_absent_rather_than_becoming_an_empty_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a public artifact repo needs no token, and the provisioning layer reads absence as "skip the
    # bootstrap phase entirely". a blank string is not absence: `ServingRuntimeSecrets` rejects an
    # empty artifact token outright, so forwarding one would turn an unset variable into a hard
    # failure for exactly the deployments that need no token at all.
    seen: list[tuple[str, str | None]] = []

    def _capture(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        seen.append(secrets._reveal_for_launch())
        return _result(bundle)

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, "   ")
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _capture)

    assert cmd_serve_deploy(_args()) == 0
    assert seen == [("inference-key", None)]


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
            # from the spec, not a literal: the handle is validated against the planned placement,
            # so a hardcoded region silently becomes a mismatch the moment the plan carries one.
            region=spec.placement.region,
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


def test_self_hosting_docs_document_a_command_that_exists() -> None:
    """The documented serving procedure must be runnable as written.

    SELF_HOSTING.md previously walked operators through `flash serve setup` and a `serve-modal`
    extra. Both are deleted, so anyone following that section installed a nonexistent extra and then
    hit "invalid choice" on the first command -- a self-hosting doc that cannot be followed at all.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    doc = (root / "SELF_HOSTING.md").read_text(encoding="utf-8")

    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    for gone in ("flash serve setup", "serve-modal"):
        assert gone not in doc, gone
    assert "serve-modal" not in extras

    # every flag the documented example passes must be one the parser actually accepts, and every
    # required flag must appear -- otherwise the example fails at parse time.
    example = doc.split("flash serve deploy \\")[1].split("```")[0]
    documented = set(re.findall(r"--[a-z-]+", example))

    parser = argparse.ArgumentParser()
    _add_serve_commands(parser.add_subparsers(dest="cmd", required=True))
    deploy = (
        parser._subparsers._group_actions[0]
        .choices["serve"]
        ._subparsers._group_actions[0]
        .choices["deploy"]
    )
    accepted = {option for action in deploy._actions for option in action.option_strings}
    required = {
        action.option_strings[0]
        for action in deploy._actions
        if action.required and action.option_strings
    }

    assert documented <= accepted, documented - accepted
    assert required <= documented, required - documented


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
