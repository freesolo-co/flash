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

_MODAL_SPEC_ID = "1446ab28c4bdc4087e9af72fb08950636213136529a791d89a7b68fe38a0c2c9"
_MODAL_IDENTITY_SHA256 = "1daf422fc339c61877060c446843c7b97a72fa90fb4c39c49be4db25612f0e44"
_MODAL_NAMES = {
    "app_or_pod": "flash-app-nzsf6nktjdgbgjx63v2yqgdxh3cwc6ss",
    "volume": "flash-volume-i2fv6yrr5vebd6voygsoi4hddayf2ggp",
    "template": "flash-template-qpexjixbfpnjzm37v2hu46xnbdhbzx4b",
    "inference_secret": "flash-inference-secret-mjlqns73nknt2u3gajw2nja4lz424nim",
    "artifact_secret": "flash-artifact-secret-w3vk3gxoueyjpjjqpodub33gdjr3idhk",
}
_MODAL_TAGS = (
    ("flash-deployment", "xxOvS4LyqBaN6-V5n2-PzBCz9R_VYmltmj_nFrKLEQ4"),
    ("flash-engine", "VwqLYZPunnh3WiJ6BO0r3xXcjurn6etO1tFuIq3cG9g"),
    ("flash-generation", "1"),
    ("flash-image", "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqo"),
    ("flash-manifest", "HBb_x6sZoADLP62F7J7YzdcGugRkD-7GoaLbKWr4HhE"),
    ("flash-phase", "finalized"),
    ("flash-spec", "FEarKMS9xAh-mvcvsIlQY2ITE2Upp5HYmnto_jigwsk"),
    ("flash-topology", "Anai4glmcG-PYt0wQbSGV7rwqthAMVRMvJEl00ls6Mo"),
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
        == "https://workspace--fsw-dylgttjqzd7m5xx7kzp5hqzg3rlv6omhrqgphwvaaiacihdx.modal.run"
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
