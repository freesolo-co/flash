"""`serve undeploy`: exact provider routing, proof reporting, and credential isolation."""

from __future__ import annotations

import argparse
import pickle
import sys

import pytest

from flash.cli.commands import serve_deploy
from flash.cli.commands.serve_deploy import cmd_serve_deploy
from flash.cli.commands.serve_undeploy import cmd_serve_undeploy
from flash.cli.serve_parser import _add_serve_commands
from flash.serve.control import DeploymentResult
from tests.test_cli_serve_deploy import IMAGE, MODEL, _stub_resolution


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_serve_commands(sub)
    return parser.parse_args(argv)


def _args(provider: str = "modal") -> argparse.Namespace:
    base = [
        "serve",
        "undeploy",
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
                "--modal-app-id",
                "ap-" + "1" * 22,
                "--modal-volume-id",
                "vo-" + "1" * 22,
                "--modal-inference-secret-id",
                "st-" + "1" * 22,
            ]
        )
    else:
        base.extend(
            [
                "--runpod-account",
                "account1",
                "--runpod-data-center",
                "US-KS-2",
                "--runpod-pod-id",
                "pod1234567890",
                "--runpod-network-volume-id",
                "vol1234567890",
                "--runpod-template-id",
                "tpl1234567890",
                "--runpod-inference-secret-id",
                "sec1234567890",
            ]
        )
    return _parse(base)


def _stub_credentials(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, str]:
    token_id = "modal-token-id-do-not-print"
    token_secret = "modal-token-secret-do-not-print"
    api_key = "runpod-api-key-do-not-print"
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_ID_ENV, token_id)
    monkeypatch.setenv(serve_deploy.MODAL_TOKEN_SECRET_ENV, token_secret)
    monkeypatch.setenv(serve_deploy.RUNPOD_API_KEY_ENV, api_key)
    return token_id, token_secret, api_key


def _result(bundle, status: str, error_code: str | None = None) -> DeploymentResult:
    return DeploymentResult.from_spec(bundle.spec, status=status, error_code=error_code)


def test_undeploy_routes_to_the_named_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def _modal(bundle, handle, credentials, *, deadline_at, **_kwargs):
        calls.append(("modal", handle.app_id))
        return _result(bundle, "absent")

    def _runpod(bundle, handle, credentials, *, deadline_at, **_kwargs):
        calls.append(("runpod", handle.pod_id))
        return _result(bundle, "absent")

    _stub_resolution(monkeypatch)
    _stub_credentials(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.teardown_modal_deployment", _modal)
    monkeypatch.setattr("flash.serve.provisioning.runpod.teardown_runpod_deployment", _runpod)

    assert cmd_serve_undeploy(_args("modal")) == 0
    assert cmd_serve_undeploy(_args("runpod")) == 0
    assert calls == [("modal", "ap-" + "1" * 22), ("runpod", "pod1234567890")]


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

    _stub_resolution(monkeypatch)
    _stub_credentials(monkeypatch)
    monkeypatch.setattr("flash.serve.provisioning.modal.teardown_modal_deployment", _unproved)

    assert cmd_serve_undeploy(_args()) == 1
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

    _stub_resolution(monkeypatch)
    secrets = _stub_credentials(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["flash", "serve", "undeploy"])
    monkeypatch.setattr("flash.serve.provisioning.modal.teardown_modal_deployment", _capture)

    assert cmd_serve_undeploy(_args()) == 0
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
    monkeypatch.setattr("flash.serve.provisioning.modal.provision_modal_deployment", _ready)

    assert cmd_serve_deploy(deploy_args()) == 0
    output = capsys.readouterr().out
    assert "app id      ap-" in output
    assert "volume id   vo-" in output
    assert "secret id   st-" in output


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
