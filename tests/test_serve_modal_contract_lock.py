"""golden locks for the surviving customer-owned modal serving contract."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json

import pytest

from flash.cli.commands.serving import deploy as serve_deploy
from flash.cli.commands.serving.identity import (
    decode_deployment_identity,
    encode_deployment_identity,
)
from flash.cli.parsing.serve_parser import _add_serve_commands
from flash.serve.provisioning.modal.planning.plan import build_modal_create_plan
from tests.test_cli_serve_deploy import _args, _stub_resolution

_MODAL_SPEC_ID = "1370367ffe6619d5b2f8b8c8029cccd54e0ed8c378219a0d6d595b68d4a71ff6"
_MODAL_IDENTITY_SHA256 = "2d173ff773412d5d734e25d0ae62dc3ce9c713c4dc4ee2150ed00c2c003172f3"
_MODAL_NAMES = {
    "app_or_pod": "flash-app-qgk7cxtxa2eibnj66v52qn47q3nagm7m",
    "volume": "flash-volume-pzy7tcniofzzh2euyndg2uryxjbqeyae",
    "template": "flash-template-sl3c2eim5vse5kyqirq7gxufnjhsrm5y",
    "inference_secret": "flash-inference-secret-neu6fultjtcp4gqrpn3h7htshrev3agd",
    "artifact_secret": "flash-artifact-secret-rfujgq3mvcncllc5m4bvoeklxls7jprl",
}
_MODAL_TAGS = (
    ("flash-deployment", "xxOvS4LyqBaN6-V5n2-PzBCz9R_VYmltmj_nFrKLEQ4"),
    ("flash-engine", "IeNs3_D5VQyTvGo74ixCLxN7SNyLGQNxyCBNtsdv_6s"),
    ("flash-generation", "1"),
    ("flash-image", "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqo"),
    ("flash-manifest", "xQ0eWiLXZ4sf9yl16IjoTXTBtQlbIl6F2MzsXSEam9k"),
    ("flash-phase", "finalized"),
    ("flash-spec", "E3A2f_5mGdWy-LjIApzM1U4O2MN4IZoNbVlbaNSnH_Y"),
    ("flash-topology", "BV35IUTw5mOtCatrPcCvxOeo7U9kDPRXaGcLyNGT3ak"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_serve_commands(parser.add_subparsers(dest="cmd", required=True))
    return parser


def _required_argv(command: str, provider: str) -> list[str]:
    argv = [
        "serve",
        command,
        "--provider",
        provider,
        "--model",
        "Qwen/Qwen3.5-9B",
        "--run",
        "run1",
        "--deployment-id",
        "deployment1",
        "--image",
        "ghcr.io/freesolo-co/freesolo-flash-serve@sha256:" + "a" * 64,
        "--artifact-repo",
        "Freesolo-Co/artifacts",
        "--artifact-subfolder",
        "rl/run1/seed0/adapter",
        "--lora-rank",
        "32",
    ]
    if provider == "modal":
        argv.extend(
            [
                "--modal-workspace",
                "workspace",
                "--modal-environment",
                "dev",
                "--modal-region",
                "us-east",
            ]
        )
    return argv


def _decode_raw(identity: str) -> dict[str, object]:
    raw = base64.urlsafe_b64decode(identity + "=" * (-len(identity) % 4))
    return json.loads(raw)


def test_modal_schema_v2_identity_spec_and_plan_are_byte_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_resolution(monkeypatch)
    bundle = serve_deploy._deployment_bundle(_args())
    identity = encode_deployment_identity(bundle)
    decoded = decode_deployment_identity(identity)
    payload = _decode_raw(identity)
    plan = build_modal_create_plan(bundle, phase="finalized")

    assert bundle.spec.spec_id == _MODAL_SPEC_ID
    assert hashlib.sha256(identity.encode("ascii")).hexdigest() == _MODAL_IDENTITY_SHA256
    assert encode_deployment_identity(decoded) == identity
    assert payload["schema"] == "flash.cli.serving.deployment-identity"
    assert payload["version"] == 2
    assert payload["provider"] == "modal"
    assert list(payload["placement"]) == [
        "environment",
        "gpu",
        "gpu_count",
        "region",
        "web_suffix",
        "workspace_name",
    ]
    assert payload["placement"] == {
        "environment": "dev",
        "gpu": "L40S",
        "gpu_count": 1,
        "region": "us-east",
        "web_suffix": None,
        "workspace_name": "workspace",
    }
    assert {name: getattr(plan.names, name) for name in plan.names.__slots__} == _MODAL_NAMES
    assert plan.tags == _MODAL_TAGS
    assert (
        plan.expected_public_url
        == "https://workspace--fsw-pwwgucwduwmnlbbpprw63vuqbeiv32wkiq7p72tqhuzivae7.modal.run"
    )


@pytest.mark.parametrize("command", ["deploy", "status", "undeploy"])
def test_provider_remains_required_without_a_modal_default(command: str) -> None:
    parser = _parser()
    argv = _required_argv(command, "modal")
    provider_index = argv.index("--provider")
    del argv[provider_index : provider_index + 2]

    with pytest.raises(SystemExit):
        parser.parse_args(argv)


@pytest.mark.parametrize("command", ["deploy", "status", "undeploy"])
def test_runpod_provider_is_rejected_by_parser_before_command_resolution(command: str) -> None:
    parser = _parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_required_argv(command, "runpod"))


def test_historical_runpod_identity_is_rejected_without_a_compatibility_decoder() -> None:
    payload = {
        "image_reference": "ghcr.io/example/serve@sha256:" + "a" * 64,
        "manifest": "historical-runpod-manifest",
        "placement": {
            "account_id": "account1",
            "container_disk_gb": 100,
            "data_center_id": "US-KS-2",
            "gpu_count": 1,
            "gpu_type_id": "NVIDIA L40S",
            "volume_size_gb": 120,
        },
        "provider": "runpod",
        "schema": "flash.cli.serving.deployment-identity",
        "version": 1,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    identity = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    with pytest.raises(ValueError, match="schema is not supported"):
        decode_deployment_identity(identity)
