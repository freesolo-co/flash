"""`serve deploy`: provider routing, credential handling, and fail-closed validation."""

from __future__ import annotations

import argparse
import pathlib
import re
import shlex
import tomllib

import pytest

from flash.cli.commands import serve_deploy
from flash.cli.commands.serve_deploy import cmd_serve_deploy
from flash.cli.commands.serve_identity import encode_deployment_identity
from flash.cli.serve_parser import _add_serve_commands
from flash.serve.profiles import get_profile, placement_for
from flash.serve.provisioning import InterruptedProvisioning

DIGEST = "sha256:" + "a" * 64
IMAGE = f"ghcr.io/freesolo-co/freesolo-flash-serve@{DIGEST}"
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


def test_a_blank_hub_token_is_treated_as_absent_not_as_a_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # a blank string is not a credential: `_optional_env` normalizes it to absence, and
    # `ServingRuntimeSecrets` rejects an empty artifact token outright. a whitespace-only
    # variable must therefore take the same rejection path as an unset one rather than being
    # forwarded as a secret or crashing inside the secret boundary.
    def _explode(*_args, **_kwargs):
        raise AssertionError("provisioning ran with a blank artifact token")

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, "   ")
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _explode)

    assert serve_deploy._optional_env(serve_deploy.ARTIFACT_TOKEN_ENV) is None
    assert cmd_serve_deploy(_args()) == 1
    assert serve_deploy.ARTIFACT_TOKEN_ENV in capsys.readouterr().err


def test_missing_artifact_token_is_rejected_before_any_provider_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # a fresh volume must hydrate before the engine starts, and that path is token-only end to
    # end: `launch._prepare_cache` rejects every cache miss when the artifact token is None, and
    # `read_artifact_token_fd` refuses an empty descriptor. the command used to pass
    # artifact_token=None straight through, so the launcher raised "artifact token is required
    # when serving cache hydration is missing" only after the provider had created and started
    # billing for the app, volume, and pod. a public repository does not change that, so the
    # rejection is unconditional rather than conditioned on repository visibility.
    def _explode(*_args, **_kwargs):
        raise AssertionError("provisioning ran without a way to hydrate the cache")

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _explode)

    # unset and blank alike mean "no token": `_optional_env` maps both to absence, and a blank
    # one would be rejected inside `ServingRuntimeSecrets` anyway.
    for value in (None, "", "   "):
        if value is None:
            monkeypatch.delenv(serve_deploy.ARTIFACT_TOKEN_ENV, raising=False)
        else:
            monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, value)

        assert cmd_serve_deploy(_args()) == 1
        assert serve_deploy.ARTIFACT_TOKEN_ENV in capsys.readouterr().err


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
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _capture)

    assert cmd_serve_deploy(_args()) == 0
    assert seen == [("inference-key", "hf-token")]


def test_resolver_validation_failures_are_cli_errors_not_tracebacks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # bad user input reaches validation below the resolver: a negative --checkpoint-step raises
    # from format_adapter_revision and a nonimmutable revision raises from ResolvedAdapter, both
    # as plain ValueError. catching only ResolveError let those escape as an unexpected-error
    # traceback after the artifact files had already been downloaded.
    _stub_environment(monkeypatch)

    def _raise_plain(**_kwargs):
        raise ValueError("invalid immutable adapter revision components")

    monkeypatch.setattr("flash.serve.resolve.resolve_adapter", _raise_plain)
    monkeypatch.setattr("flash.serve.resolve.resolve_base_revision", lambda *_a, **_k: "d" * 40)

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


def test_unknown_model_is_refused_before_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(**_kwargs):
        raise AssertionError("resolution ran for an unsupported model")

    monkeypatch.setattr("flash.serve.resolve.resolve_adapter", _explode)

    assert cmd_serve_deploy(_args(model="Qwen/Qwen3.5-0.8B")) == 1


def test_outcome_unknown_is_not_reported_as_a_plain_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # a retry after an unconfirmed outcome can double-provision and bill twice.
    identities: list[str] = []

    def _unknown(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        from flash.serve.control import DeploymentResult

        identities.append(encode_deployment_identity(bundle))
        return DeploymentResult.from_spec(
            bundle.spec, status="outcome_unknown", error_code="transport_failed"
        )

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _unknown)

    assert cmd_serve_deploy(_args()) == 1
    captured = capsys.readouterr()
    assert "outcome_unknown" in captured.out
    assert f"identity    {identities[0]}\n" in captured.out
    assert "flash serve status" in captured.err
    assert "flash serve undeploy" in captured.err

    def _encoding_fails(_bundle):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(
        "flash.cli.commands.serve_identity.encode_deployment_identity", _encoding_fails
    )
    assert cmd_serve_deploy(_args()) == 1
    assert "outcome_unknown" in capsys.readouterr().out


def test_identity_reporting_does_not_swallow_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _interrupted(_bundle):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "flash.cli.commands.serve_identity.encode_deployment_identity", _interrupted
    )

    with pytest.raises(KeyboardInterrupt):
        serve_deploy._report_identity(object())


def test_readiness_timeout_names_the_pod_and_points_to_teardown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from flash.serve.control import DeploymentResult

    def _timed_out(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        ready = _result(bundle)
        return DeploymentResult.from_spec(
            bundle.spec,
            status="failed",
            handle=ready.handle,
            error_code="readiness_timeout",
        )

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.runpod.provision_runpod_deployment", _timed_out)

    assert cmd_serve_deploy(_args(provider="runpod")) == 1
    captured = capsys.readouterr()
    assert "readiness did not prove within the deadline" in captured.err
    assert "pod pod1234567890" in captured.err
    assert "flash serve undeploy" in captured.err
    assert "outcome could not be confirmed" not in captured.err


def test_interrupted_deploy_prints_recovery_identity_without_masking_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    identities: list[str] = []

    def _interrupted(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        identities.append(encode_deployment_identity(bundle))
        raise InterruptedProvisioning("modal")

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _interrupted)

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
        "flash.cli.commands.serve_identity.encode_deployment_identity", _encoding_fails
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
    # a deployable environment includes the artifact token: hydration on a fresh volume is
    # token-only, so the command rejects a missing one before contacting any provider. tests
    # about that rejection unset it explicitly.
    monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, "hf-token")


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

    # argparse-required is not the same as runnable. The `--modal-*` placement flags are optional
    # to argparse (RunPod takes its own pair instead), but `placement_for` requires exactly one
    # provider's full set, so a modal example without them exits with "modal placement requires
    # environment, region, workspace_name". Drive the real resolver rather than re-listing flags.
    parsed = deploy.parse_args(shlex.split(example.replace("\\\n", " ")))
    placement_for(
        get_profile(parsed.model),
        parsed.provider,
        workspace_name=parsed.modal_workspace,
        environment=parsed.modal_environment,
        region=parsed.modal_region,
        web_suffix=(parsed.modal_web_suffix or None),
        account_id=parsed.runpod_account,
        data_center_id=parsed.runpod_data_center,
    )

    # and the documented install must supply what the command imports: the base install has no
    # dependencies at all, so `pip install freesolo-flash` alone dies in `_hub_api`.
    assert "pip install 'freesolo-flash[server]'" in doc


def test_self_hosting_docs_do_not_route_the_plane_key_to_a_customer_endpoint() -> None:
    """The documented way to call a `serve deploy` endpoint must be the one it authenticates.

    The packaged app's `_authorize` reads `Authorization: Bearer` and nothing else -- there is no
    internal-key acceptance anywhere in `flash/serve/app/`. `FREESOLO_SERVING_URL` drives the
    separate multi-LoRA backend, and every request it sends carries `FREESOLO_INTERNAL_KEY` via
    `X-Freesolo-Internal-Key`. So documenting the deployment URL under `FREESOLO_SERVING_URL` does
    not merely fail `401`: it first sends the key that controls the whole plane to a provider
    endpoint, which `docs/serving-contract.md` forbids ("must never be sent to a customer-owned
    endpoint"). Asserting on the app's real check rather than on prose keeps the two in step.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    doc = (root / "SELF_HOSTING.md").read_text(encoding="utf-8")

    app = (root / "flash" / "serve" / "app" / "http.py").read_text(encoding="utf-8")
    assert "internal-key" not in app.casefold(), "the packaged app must not read the plane key"
    assert '"bearer"' in app.casefold(), "the packaged app must authenticate with bearer"

    # the deployment section must show the bearer call, and must not hand its url to the command
    # family that authenticates with the plane key.
    section = doc.split("## Serving")[1]
    assert "Authorization: Bearer $FLASH_SERVING_KEY" in section
    exported = re.findall(r"export FREESOLO_SERVING_URL=(\S+)", section)
    assert not any("modal.run" in value or "proxy.runpod" in value for value in exported), exported


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

    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _explode)

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


@pytest.mark.parametrize("bad", ["-1", "-999"])
def test_checkpoint_step_zero_is_accepted_but_negatives_are_rejected_at_parse(bad: str) -> None:
    accepted = _parse(
        [
            *_required_deploy_argv(),
            "--lora-rank",
            "32",
            "--checkpoint-step",
            "0",
        ]
    )
    assert accepted.checkpoint_step == 0

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
