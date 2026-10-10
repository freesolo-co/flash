"""`serve undeploy`: exact provider routing, proof reporting, and credential isolation."""

from __future__ import annotations

import argparse
import pickle
import sys

import pytest

from flash.cli.commands.serving import deploy as serve_deploy
from flash.cli.commands.serving import undeploy as serve_undeploy
from flash.cli.commands.serving.deploy import cmd_serve_deploy
from flash.cli.commands.serving.undeploy import cmd_serve_undeploy
from flash.cli.parsing.serve_parser import _add_serve_commands
from flash.serve.control import DeploymentResult
from flash.serve.deployment.resolve import ResolveError
from tests.test_cli_serve_deploy import IMAGE, MODEL, _historical_identity, _stub_resolution


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_serve_commands(sub)
    return parser.parse_args(argv)


def _args() -> argparse.Namespace:
    base = [
        "serve",
        "undeploy",
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
    ]
    base.extend(
        [
            "--modal-workspace",
            "workspace",
            "--modal-environment",
            "dev",
            "--modal-region",
            "us-east",
            "--modal-app-id",
            "ap-" + "1" * 22,
            "--modal-volume-id",
            "vo-" + "1" * 22,
            "--modal-inference-secret-id",
            "st-" + "1" * 22,
        ]
    )
    return _parse(base)


def _stub_credentials(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    token_id = "modal-token-id-do-not-print"
    token_secret = "modal-token-secret-do-not-print"
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_ID_ENV, token_id)
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_SECRET_ENV, token_secret)
    return token_id, token_secret


def _result(bundle, status: str, error_code: str | None = None) -> DeploymentResult:
    return DeploymentResult.from_spec(bundle.spec, status=status, error_code=error_code)


def _deployment_identity(monkeypatch: pytest.MonkeyPatch, args: argparse.Namespace) -> str:
    from flash.cli.commands.serving.identity import encode_deployment_identity

    _stub_resolution(monkeypatch)
    bundle = serve_deploy._deployment_bundle(args)
    return encode_deployment_identity(bundle)


def _args_with_identity(monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    args = _args()
    args.deployment_identity = _deployment_identity(monkeypatch, args)
    return args


def _fail_hub_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    def _base_failure(*_args, **_kwargs):
        raise ResolveError("could not resolve the commit for unavailable/model")

    def _adapter_failure(*_args, **_kwargs):
        raise ResolveError("could not resolve the revision for unavailable/artifact")

    monkeypatch.setattr("flash.serve.deployment.resolve.resolve_base_revision", _base_failure)
    monkeypatch.setattr("flash.serve.deployment.resolve.resolve_adapter", _adapter_failure)


@pytest.mark.parametrize(
    "retired_model",
    ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B"],
)
def test_undeploy_uses_immutable_identity_for_removed_model(
    monkeypatch: pytest.MonkeyPatch, retired_model: str
) -> None:
    args = _args()
    args.deployment_identity = _historical_identity(monkeypatch, args, retired_model)
    args.model = retired_model
    _stub_credentials(monkeypatch)
    seen = []

    def _teardown(bundle, handle, credentials, *, deadline_at, **_kwargs):
        seen.append(bundle.spec.adapters[0].base_model)
        return _result(bundle, "absent")

    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment",
        _teardown,
    )

    assert cmd_serve_undeploy(args) == 0
    assert seen == [retired_model]


def test_undeploy_uses_deploy_time_names_after_the_model_tip_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flash.serve.provisioning.modal.planning.plan import build_modal_create_plan

    args = _args()
    args.deployment_identity = _deployment_identity(monkeypatch, args)
    deployed = serve_deploy._deployment_bundle(args)
    deployed_name = build_modal_create_plan(deployed).names.app_or_pod

    monkeypatch.setattr(
        "flash.serve.deployment.resolve.resolve_base_revision", lambda *_a, **_k: "e" * 40
    )
    current_tip = serve_deploy._deployment_bundle(args)
    current_name = build_modal_create_plan(current_tip).names.app_or_pod
    assert current_name != deployed_name
    _stub_credentials(monkeypatch)
    seen: list[tuple[str, str]] = []

    def _teardown(bundle, handle, credentials, *, deadline_at, **_kwargs):
        seen.append((build_modal_create_plan(bundle).names.app_or_pod, handle.app_name))
        return _result(bundle, "absent")

    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment",
        _teardown,
    )

    assert cmd_serve_undeploy(args) == 0
    assert seen == [(deployed_name, deployed_name)]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("generation", "does not match --generation"),
        ("provider_id", "does not match the pinned provider contract"),
    ],
)
def test_undeploy_identity_rejects_mismatched_destructive_input(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    message: str,
) -> None:
    args = _args()
    args.deployment_identity = _deployment_identity(monkeypatch, args)
    _fail_hub_resolution(monkeypatch)
    _stub_credentials(monkeypatch)
    if mutation == "generation":
        args.generation = 2
    else:
        args.modal_app_id = "wrong-provider-id"

    def _explode(*_args, **_kwargs):
        raise AssertionError("teardown ran with mismatched provider identity")

    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment",
        _explode,
    )

    assert cmd_serve_undeploy(args) == 1
    assert message in capsys.readouterr().err


def test_undeploy_hub_failure_requires_the_printed_identity_before_teardown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _args()
    _fail_hub_resolution(monkeypatch)
    _stub_credentials(monkeypatch)

    def _explode(*_args, **_kwargs):
        raise AssertionError("teardown ran without a validated deployment identity")

    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment",
        _explode,
    )

    assert cmd_serve_undeploy(args) == 1
    captured = capsys.readouterr()
    assert "--deployment-identity is required" in captured.err
    assert "pass the value printed by `flash serve deploy`" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation", 1, "does not match the exact deployment generation"),
        ("app_id", "wrong-provider-id", "does not match the pinned provider contract"),
    ],
)
def test_undeploy_rejects_mismatched_handle_input_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
    message: str,
) -> None:
    _stub_credentials(monkeypatch)
    args = _args()
    if field == "generation":
        args.generation = 2
    args.deployment_identity = _deployment_identity(monkeypatch, args)
    original_provider_handle = serve_undeploy._provider_handle

    def _mismatched_handle(parsed_args, bundle):
        handle = original_provider_handle(parsed_args, bundle)
        # simulate ids copied from a different generation or mistyped after provider output. the
        # frozen handle normally prevents mutation, but the cli boundary must still validate what it
        # is about to hand to teardown rather than relying on provider code to report user input.
        object.__setattr__(handle, field, value)
        return handle

    monkeypatch.setattr(serve_undeploy, "_provider_handle", _mismatched_handle)

    assert cmd_serve_undeploy(args) == 1
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_modal_undeploy_without_provider_ids_routes_to_identity_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args_with_identity(monkeypatch)
    args.modal_app_id = ""
    args.modal_volume_id = ""
    args.modal_inference_secret_id = ""
    _stub_credentials(monkeypatch)
    handles = []

    def _reclaim(bundle, handle, credentials, *, deadline_at, **_kwargs):
        handles.append(handle)
        return _result(bundle, "absent")

    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment",
        _reclaim,
    )

    assert cmd_serve_undeploy(args) == 0
    assert handles == [None]


def test_modal_undeploy_rejects_partial_provider_ids(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args_with_identity(monkeypatch)
    args.modal_app_id = ""
    _stub_credentials(monkeypatch)

    def _explode(*_args, **_kwargs):
        raise AssertionError("teardown ran with a partial provider handle")

    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment",
        _explode,
    )

    assert cmd_serve_undeploy(args) == 1
    assert "modal provider ids must be supplied together or omitted" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("status", "error_code"),
    [("failed", "conflict"), ("outcome_unknown", "transport_failed")],
)
def test_unproved_undeploy_is_a_clear_nonzero_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    error_code: str,
) -> None:
    def _unproved(bundle, handle, credentials, *, deadline_at, **_kwargs):
        return _result(bundle, status, error_code)

    _stub_credentials(monkeypatch)
    args = _args_with_identity(monkeypatch)
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment",
        _unproved,
    )

    assert cmd_serve_undeploy(args) == 1
    captured = capsys.readouterr()
    assert f"status      {status}" in captured.out
    assert "resource absence could not be proved" in captured.err
    assert "confirmed absent" not in captured.out


def test_credentials_never_enter_argv_output_or_persisted_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    seen_repr: list[str] = []

    def _capture(bundle, handle, credentials, *, deadline_at, **_kwargs):
        seen_repr.append(repr(credentials))
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(credentials)
        return _result(bundle, "absent")

    secrets = _stub_credentials(monkeypatch)
    args = _args_with_identity(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["flash", "serve", "undeploy"])
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment",
        _capture,
    )

    assert cmd_serve_undeploy(args) == 0
    captured = capsys.readouterr()
    exposed = "\n".join((*sys.argv, captured.out, captured.err, *seen_repr))
    assert all(secret not in exposed for secret in secrets)
    assert list(tmp_path.iterdir()) == []
    assert seen_repr == ["ModalCredentials(<redacted>)"]


def test_deploy_output_exposes_exact_ids_needed_by_undeploy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.test_cli_serve_deploy import _args as deploy_args
    from tests.test_cli_serve_deploy import _result as ready_result
    from tests.test_cli_serve_deploy import _stub_environment

    def _ready(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        return ready_result(bundle)

    _stub_resolution(monkeypatch)
    _stub_environment(monkeypatch)
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.provision_modal_deployment",
        _ready,
    )

    assert cmd_serve_deploy(deploy_args()) == 0
    output = capsys.readouterr().out
    assert "app id      ap-" in output
    assert "volume id   vo-" in output
    assert "secret id   st-" in output
    assert "identity    " in output


def test_parser_wires_serve_undeploy_without_credential_flags() -> None:
    args = _args()
    assert args.serve_cmd == "undeploy"
    assert args.func is cmd_serve_undeploy

    parser = argparse.ArgumentParser()
    _add_serve_commands(parser.add_subparsers(dest="cmd", required=True))
    help_text = parser.format_help()
    serve = parser._subparsers._group_actions[0].choices["serve"]
    undeploy = serve._subparsers._group_actions[0].choices["undeploy"]
    help_text += undeploy.format_help()
    for forbidden in ("--token", "--api-key", "--password", "--credential"):
        assert forbidden not in help_text


def test_undeploy_uses_supplied_identity_when_hub_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    args = _args()
    args.deployment_identity = _deployment_identity(monkeypatch, args)
    _fail_hub_resolution(monkeypatch)
    _stub_credentials(monkeypatch)

    def _teardown(bundle, handle, credentials, *, deadline_at, **_kwargs):
        calls.append(handle.app_id)
        return _result(bundle, "absent")

    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment",
        _teardown,
    )

    assert cmd_serve_undeploy(args) == 0
    assert calls == ["ap-" + "1" * 22]


def test_undeploy_routes_to_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _modal(bundle, handle, credentials, *, deadline_at, **_kwargs):
        calls.append(handle.app_id)
        return _result(bundle, "absent")

    _stub_credentials(monkeypatch)
    args = _args_with_identity(monkeypatch)
    monkeypatch.setattr(
        "flash.serve.provisioning.modal.execution.operations.teardown_modal_deployment", _modal
    )

    assert cmd_serve_undeploy(args) == 0
    assert calls == ["ap-" + "1" * 22]
