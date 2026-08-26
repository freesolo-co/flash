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

_MODAL_SPEC_ID = "5decff67910d3e7e0dc3603df06e0ebc1a3d5775168d8d407096d12837e4d915"
_MODAL_IDENTITY_SHA256 = "e27c305fece66e861e202d8d7679bdb8d4a1086d8472cab5489b7cac51d8e9a1"
_MODAL_NAMES = {
    "app_or_pod": "flash-app-hfkh2247mvjpggvtzvd4knofscpo7flr",
    "volume": "flash-volume-g4z2wyskb2alosewun6jlvmmo6butmnf",
    "template": "flash-template-w5vimpv5zq7mlpwcouhea44ciglvumz4",
    "inference_secret": "flash-inference-secret-3zo7xk3ogb3qgs3wzsxanpxnzpbus3uy",
    "artifact_secret": "flash-artifact-secret-w6ssvtgv6owrqhe53ivympahpfstsdan",
}
_MODAL_TAGS = (
    ("flash-deployment", "xxOvS4LyqBaN6-V5n2-PzBCz9R_VYmltmj_nFrKLEQ4"),
    ("flash-engine", "VDkehhtfTiKXjz3ujU0fUpdhxgxrKdkiejK-ZMTEW_I"),
    ("flash-generation", "1"),
    ("flash-image", "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqo"),
    ("flash-manifest", "fpj0Dlfj_Ckm2PkBk6VGysP2zwcGGCGr_qV4ovoEras"),
    ("flash-phase", "finalized"),
    ("flash-spec", "Xez_Z5ENPn4Nw2A98G4OvBo9V3UWjY1AcJbRKDfk2RU"),
    ("flash-topology", "cU3m3ClQlkL5ww-Vwx0dbwkcTu_G-uU8fzTeg5-wBDE"),
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


def test_modal_schema_v1_identity_spec_and_plan_are_byte_stable(
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
    assert payload["version"] == 1
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
        == "https://workspace--fsw-ep334fd5gaczpvb2n63scnojwcqvegeaag5ww5a7oeniyach.modal.run"
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

    with pytest.raises(ValueError, match="provider must be modal"):
        decode_deployment_identity(identity)
