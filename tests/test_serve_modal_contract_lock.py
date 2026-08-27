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

_MODAL_SPEC_ID = "6ae4d74ee15c1dfc05f352afbcd99eb100265e54b93e606faa014173644c6805"
_MODAL_IDENTITY_SHA256 = "98a7731a6eae01963a4c3b7e9f413c80021ff1d911aa38c7a633dea0733cda8d"
_MODAL_NAMES = {
    "app_or_pod": "flash-app-zptooedz4yfjaqhxks5etacu23bkqi2b",
    "volume": "flash-volume-u7ve3ykgp7eboeqytu4rljdbnqrzkezm",
    "template": "flash-template-5xm3lmkqqyxs2d7hinslcc4varygvtyp",
    "inference_secret": "flash-inference-secret-rmnhna6qfrvhr7lf4l2u5kys2acbetnc",
    "artifact_secret": "flash-artifact-secret-72ogacmlmvwgswa26iuaakch25hi3tdh",
}
_MODAL_TAGS = (
    ("flash-deployment", "xxOvS4LyqBaN6-V5n2-PzBCz9R_VYmltmj_nFrKLEQ4"),
    ("flash-engine", "G_VoYu0hpBNT3-DbeaCzKwHKn6MbEgvDGiTvPhoF7Jw"),
    ("flash-generation", "1"),
    ("flash-image", "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqo"),
    ("flash-manifest", "WJXGrlHLrQ3Ju0x4Ufa3bor-FwsbvDvcYJHUOYFZWRQ"),
    ("flash-phase", "finalized"),
    ("flash-spec", "auTXTuFcHfwF81KvvNmesQAmXlS5PmBvqgFBc2RMaAU"),
    ("flash-topology", "Ypl_b2aBI81W-dGw6BbFxchnVretqCLS7MsxDUJThL8"),
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
        == "https://workspace--fsw-i7ihnvcwpdxglv55t5kul6kgdwiyzmc7wdxib23lbsqeocqo.modal.run"
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
