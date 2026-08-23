"""`serve status`: read-only provider routing and honest deployment state reporting."""

from __future__ import annotations

import argparse
import pickle
import sys

import pytest

from flash.cli.commands import serve_deploy
from flash.cli.commands.serve_status import cmd_serve_status
from flash.cli.serve_parser import _add_serve_commands
from flash.serve.control import DeploymentResult
from tests.test_cli_serve_deploy import IMAGE, MODEL, _stub_resolution
from tests.test_cli_serve_deploy import _result as ready_result


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _add_serve_commands(parser.add_subparsers(dest="cmd", required=True))
    return parser.parse_args(argv)


def _args(provider: str = "modal") -> argparse.Namespace:
    base = [
        "serve",
        "status",
        "--provider",
        provider,
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
    if provider == "modal":
        base.extend(
            [
                "--modal-workspace",
                "workspace",
                "--modal-environment",
                "dev",
                "--modal-region",
                "us-east",
            ]
        )
    else:
        base.extend(
            [
                "--runpod-account",
                "account1",
                "--runpod-data-center",
                "US-KS-2",
            ]
        )
    return _parse(base)


def _args_with_identity(
    monkeypatch: pytest.MonkeyPatch, provider: str = "modal"
) -> argparse.Namespace:
    from flash.cli.commands.serve_identity import encode_deployment_identity

    args = _args(provider)
    _stub_resolution(monkeypatch)
    args.deployment_identity = encode_deployment_identity(serve_deploy._deployment_bundle(args))
    return args


def _stub_environment(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, str, str, str]:
    values = (
        "modal-token-id-do-not-print",
        "modal-token-secret-do-not-print",
        "runpod-api-key-do-not-print",
        "inference-key-do-not-print",
        "artifact-token-do-not-print",
    )
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_ID_ENV, values[0])
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_SECRET_ENV, values[1])
    monkeypatch.setenv(serve_deploy.RUNPOD_API_KEY_ENV, values[2])
    monkeypatch.setenv(serve_deploy.INFERENCE_KEY_ENV, values[3])
    monkeypatch.setenv(serve_deploy.ARTIFACT_TOKEN_ENV, values[4])
    return values


def _result(bundle, status: str) -> DeploymentResult:
    ready = ready_result(bundle)
    if status == "ready":
        return ready
    if status == "absent":
        return DeploymentResult.from_spec(bundle.spec, status="absent")
    if status == "provisioning":
        return DeploymentResult.from_spec(bundle.spec, status=status, handle=ready.handle)
    error_code = "transport_failed" if status == "outcome_unknown" else "conflict"
    return DeploymentResult.from_spec(
        bundle.spec,
        status=status,
        handle=ready.handle,
        error_code=error_code,
    )


def test_status_rejects_provider_invalid_placement_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_environment(monkeypatch)
    args = _args_with_identity(monkeypatch)
    args.modal_workspace = "UPPER"

    assert cmd_serve_status(args) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err


def test_status_bundle_failure_is_not_mislabeled_as_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _args_with_identity(monkeypatch)
    args.model = "not-a-catalog-model"

    assert cmd_serve_status(args) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "credentials are read from the environment" not in captured.err


def test_status_credential_failure_keeps_request_scoped_guidance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _args_with_identity(monkeypatch)
    monkeypatch.delenv(serve_deploy.MODAL_TOKEN_ID_ENV, raising=False)
    monkeypatch.delenv(serve_deploy.MODAL_TOKEN_SECRET_ENV, raising=False)

    assert cmd_serve_status(args) == 1
    captured = capsys.readouterr()
    assert (
        "credentials are read from the environment for this one request and are never stored"
        in (captured.err)
    )


def test_status_uses_deploy_time_identity_after_the_model_tip_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flash.cli.commands.serve_identity import encode_deployment_identity
    from flash.serve.provisioning._modal_plan import build_modal_create_plan

    args = _args()
    _stub_resolution(monkeypatch)
    deployed = serve_deploy._deployment_bundle(args)
    args.deployment_identity = encode_deployment_identity(deployed)
    deployed_name = build_modal_create_plan(deployed).names.app_or_pod

    monkeypatch.setattr("flash.serve.resolve.resolve_base_revision", lambda *_a, **_k: "e" * 40)
    current_tip = serve_deploy._deployment_bundle(args)
    current_name = build_modal_create_plan(current_tip).names.app_or_pod
    assert current_name != deployed_name

    seen: list[str] = []

    def _reconcile(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        seen.append(build_modal_create_plan(bundle).names.app_or_pod)
        return _result(bundle, "absent")

    _stub_environment(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.reconcile_modal_deployment", _reconcile)

    assert cmd_serve_status(args) == 0
    assert seen == [deployed_name]


def test_status_without_inference_key_reports_provider_observable_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, object]] = []

    def _modal(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        observed.append(("modal", secrets))
        return _result(bundle, "absent")

    def _runpod(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        observed.append(("runpod", secrets))
        return _result(bundle, "provisioning")

    _stub_environment(monkeypatch)
    monkeypatch.delenv(serve_deploy.INFERENCE_KEY_ENV)
    modal_args = _args_with_identity(monkeypatch, "modal")
    runpod_args = _args_with_identity(monkeypatch, "runpod")
    monkeypatch.setattr("flash.serve.provisioning.modal.reconcile_modal_deployment", _modal)
    monkeypatch.setattr("flash.serve.provisioning.runpod.reconcile_runpod_deployment", _runpod)

    assert cmd_serve_status(modal_args) == 0
    assert cmd_serve_status(runpod_args) == 0
    assert observed == [("modal", None), ("runpod", None)]


def test_status_routes_to_the_named_read_only_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _modal(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        calls.append("modal")
        return _result(bundle, "ready")

    def _runpod(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        calls.append("runpod")
        return _result(bundle, "ready")

    _stub_environment(monkeypatch)
    modal_args = _args_with_identity(monkeypatch, "modal")
    runpod_args = _args_with_identity(monkeypatch, "runpod")
    monkeypatch.setattr("flash.serve.provisioning.modal.reconcile_modal_deployment", _modal)
    monkeypatch.setattr("flash.serve.provisioning.runpod.reconcile_runpod_deployment", _runpod)

    assert cmd_serve_status(modal_args) == 0
    assert cmd_serve_status(runpod_args) == 0
    assert calls == ["modal", "runpod"]


@pytest.mark.parametrize(
    ("status", "expected_code", "stream", "message"),
    [
        ("ready", 0, "out", "health      endpoint readiness proved"),
        ("provisioning", 0, "out", "health      not ready yet"),
        ("absent", 0, "out", "resources   confirmed absent"),
        ("failed", 1, "err", "provider status check failed"),
        ("outcome_unknown", 1, "err", "could not be proved healthy or absent"),
    ],
)
def test_status_surfaces_every_outcome_distinctly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    expected_code: int,
    stream: str,
    message: str,
) -> None:
    def _reconcile(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        return _result(bundle, status)

    _stub_environment(monkeypatch)
    args = _args_with_identity(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.reconcile_modal_deployment", _reconcile)

    assert cmd_serve_status(args) == expected_code
    captured = capsys.readouterr()
    assert f"status      {status}" in captured.out
    assert message in getattr(captured, stream)
    assert ("endpoint    " in captured.out) is (status == "ready")


@pytest.mark.parametrize("status", ["provisioning", "absent", "failed", "outcome_unknown"])
def test_non_ready_status_never_prints_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
) -> None:
    def _reconcile(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        return _result(bundle, status)

    _stub_environment(monkeypatch)
    args = _args_with_identity(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.reconcile_modal_deployment", _reconcile)

    cmd_serve_status(args)
    output = capsys.readouterr().out
    assert "status      ready" not in output
    assert "endpoint readiness proved" not in output
    assert "endpoint    " not in output


def test_status_credentials_never_enter_argv_output_or_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    seen_repr: list[str] = []

    def _capture(bundle, credentials, secrets, *, deadline_at, **_kwargs):
        seen_repr.extend((repr(credentials), repr(secrets)))
        for value in (credentials, secrets):
            with pytest.raises(TypeError, match="cannot be serialized"):
                pickle.dumps(value)
        return _result(bundle, "ready")

    secret_values = _stub_environment(monkeypatch)
    args = _args_with_identity(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["flash", "serve", "status"])
    monkeypatch.setattr("flash.serve.provisioning.modal.reconcile_modal_deployment", _capture)

    assert cmd_serve_status(args) == 0
    captured = capsys.readouterr()
    exposed = "\n".join((*sys.argv, captured.out, captured.err, *seen_repr))
    assert all(secret not in exposed for secret in secret_values)
    assert list(tmp_path.iterdir()) == []
    assert seen_repr == ["ModalCredentials(<redacted>)", "ServingRuntimeSecrets(<redacted>)"]


def test_parser_wires_serve_status_without_credential_flags() -> None:
    args = _args()
    assert args.serve_cmd == "status"
    assert args.func is cmd_serve_status

    parser = argparse.ArgumentParser()
    _add_serve_commands(parser.add_subparsers(dest="cmd", required=True))
    serve = parser._subparsers._group_actions[0].choices["serve"]
    status = serve._subparsers._group_actions[0].choices["status"]
    help_text = parser.format_help() + status.format_help()
    for forbidden in ("--token", "--api-key", "--password", "--credential"):
        assert forbidden not in help_text
