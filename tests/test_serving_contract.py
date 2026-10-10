"""Hosted serving client contract tests."""

from __future__ import annotations

import types

import httpx
import pytest

import flash.serve.contract.errors as serving_errors
import flash.serve.contract.urls as serving_urls
import flash.serve.deployment.adapter_check as adapter_check
import flash.serve.request.transport as serving_transport
from flash.schema import format_checkpoint_ref, parse_checkpoint_ref
from flash.serve.contract.protocol import (
    PERMANENT_CHECKPOINT_IDENTITY_CAPABILITY,
    PREFERRED_SERVING_CAPABILITIES,
    REQUIRED_SERVING_CAPABILITIES,
    ServingHealthError,
    parse_serving_health,
)
from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serve.contract.urls import serving_base_url
from flash.serve.deployment.deploy import Deployment, deploy_adapter, undeploy_adapter


@pytest.fixture(autouse=True)
def _stub_shared_http_client(monkeypatch):
    class _Client:
        def request(self, method, url, **kwargs):
            return getattr(httpx, method.lower())(url, **kwargs)

    monkeypatch.setattr(serving_transport, "_http_client", lambda: _Client())


def test_dependency_light_health_parser_normalizes_the_serving_contract() -> None:
    health = parse_serving_health(
        {
            "ok": True,
            "requires_key": False,
            "base_models": ["Qwen/Qwen3.5-9B"],
            "capabilities": sorted(REQUIRED_SERVING_CAPABILITIES | PREFERRED_SERVING_CAPABILITIES),
        }
    )
    assert health.ok is True
    assert health.requires_key is False
    assert health.base_models == ("Qwen/Qwen3.5-9B",)
    assert set(health.capabilities) >= REQUIRED_SERVING_CAPABILITIES


@pytest.mark.parametrize(
    ("payload", "expected_ok"),
    [
        ({"capabilities": [], "ok": False}, False),
        ({"capabilities": []}, None),
    ],
)
def test_dependency_light_health_parser_preserves_optional_ok(payload, expected_ok):
    assert parse_serving_health(payload).ok is expected_ok


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ([], "non_object"),
        ({}, "capabilities_not_list"),
        ({"capabilities": [1]}, "capabilities_not_strings"),
    ],
)
def test_dependency_light_health_parser_rejects_malformed_payloads(payload, code) -> None:
    with pytest.raises(ServingHealthError) as exc_info:
        parse_serving_health(payload)
    assert exc_info.value.code == code


def test_schema_uses_one_permanent_checkpoint_identity() -> None:
    assert format_checkpoint_ref("flash-1", None) == "flash-1/final"
    assert format_checkpoint_ref("flash-1", 2) == "flash-1/step-2"
    assert parse_checkpoint_ref("flash-1/final") == ("flash-1", None)
    assert parse_checkpoint_ref("flash-1/step-2") == ("flash-1", 2)
    for invalid in ("flash-1", "flash-1/step-02", "flash-1@final." + "a" * 40):
        assert parse_checkpoint_ref(invalid) is None


def test_serving_base_url_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("FREESOLO_SERVING_URL", raising=False)
    assert serving_base_url() == serving_urls.default_serving_url()
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/")
    assert serving_base_url() == "https://serve.example"
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    assert serving_base_url() == "https://serve.example"


def test_deploy_dry_run_uses_explicit_final_checkpoint() -> None:
    dep = deploy_adapter("r1", "Qwen/Qwen3.5-9B", "repo", "rl/r1/seed0", dry_run=True)
    data = dep.to_dict()
    assert data["state"] == "dry_run"
    assert data["checkpoint_id"] == "r1/final"
    assert data["openai_model"] == "r1/final"
    assert "gpu" not in data
    assert "mode" not in data


@pytest.mark.parametrize(
    ("marker", "targets_images"),
    [(None, True), (r"^(?!model)(?:\.|$).*$", False), (pytest.param(..., False, id="missing"))],
)
def test_adapter_artifact_metadata_reads_the_exported_modality_marker(
    monkeypatch, marker, targets_images
) -> None:
    config = {"r": 32}
    if marker is not ...:
        config["exclude_modules"] = marker
    monkeypatch.setattr(
        adapter_check,
        "_load_adapter_config",
        lambda *_args, **_kwargs: (config, "run/adapter/adapter_config.json"),
    )
    monkeypatch.setattr(
        adapter_check,
        "_verify_adapter_artifact_tensors",
        lambda *a, **k: ({"path": "adapter.safetensors", "size": 1},),
    )

    metadata = adapter_check.adapter_artifact_metadata(
        "org/repo", "run/adapter", artifact_revision="a" * 40
    )

    assert metadata.lora_rank == 32
    assert metadata.targets_images is targets_images
    assert len(metadata.artifact_digest) == 64


def _stub_deploy_preconditions(monkeypatch, deploy_mod) -> None:
    monkeypatch.setattr(deploy_mod, "_registered_adapter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(deploy_mod, "resolve_artifact_revision", lambda repo: "a" * 40)
    monkeypatch.setattr(
        adapter_check,
        "adapter_artifact_metadata",
        lambda *a, **k: types.SimpleNamespace(
            lora_rank=32,
            targets_images=False,
            artifact_digest="b" * 64,
        ),
    )
    monkeypatch.setattr(
        deploy_mod,
        "_require_serving_capabilities",
        lambda **_kwargs: {PERMANENT_CHECKPOINT_IDENTITY_CAPABILITY},
    )


def test_real_deploy_translates_serving_5xx_to_serving_error(monkeypatch) -> None:
    import flash.serve.deployment.deploy as deploy_mod

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    req = httpx.Request("POST", "https://serve.example/adapters")
    resp = httpx.Response(500, text="no base-model engines loaded", request=req)
    _stub_deploy_preconditions(monkeypatch, deploy_mod)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: resp)

    with pytest.raises(serving_errors.ServingError) as exc_info:
        deploy_adapter(
            "flash-1-abc",
            "Qwen/Qwen3.5-9B",
            "repo",
            "rl/r1/seed0",
            org_id="org-1",
        )
    assert exc_info.value.status_code == 500
    assert "no base-model engines loaded" in str(exc_info.value)


def test_real_deploy_4xx_hint_points_at_client(monkeypatch) -> None:
    import flash.serve.deployment.deploy as deploy_mod

    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    req = httpx.Request("POST", "https://serve.example/adapters")
    resp = httpx.Response(401, text="invalid internal key", request=req)
    _stub_deploy_preconditions(monkeypatch, deploy_mod)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: resp)

    with pytest.raises(serving_errors.ServingError) as exc_info:
        deploy_adapter(
            "flash-1-abc",
            "Qwen/Qwen3.5-9B",
            "repo",
            "rl/r1/seed0",
            org_id="org-1",
        )
    assert exc_info.value.status_code == 401
    assert "FREESOLO_INTERNAL_KEY" in str(exc_info.value)


def _stub_healthz(
    monkeypatch, deploy_mod, capabilities: list[str], *, ok: bool | None = None
) -> None:
    """stub the serving /healthz response used by the capability preflight."""

    class _Resp:
        def json(self):
            payload = {"capabilities": list(capabilities)}
            if ok is not None:
                payload["ok"] = ok
            return payload

    monkeypatch.setattr(serving_transport, "serving_request", lambda method, url, **k: _Resp())


@pytest.mark.parametrize("ok", [None, True], ids=["omitted", "true"])
def test_require_capabilities_accepts_healthy_or_compatible_health_payload(monkeypatch, ok):
    import flash.serve.deployment.deploy as deploy_mod

    capabilities = sorted(REQUIRED_SERVING_CAPABILITIES | PREFERRED_SERVING_CAPABILITIES)
    _stub_healthz(monkeypatch, deploy_mod, capabilities, ok=ok)

    assert deploy_mod._require_serving_capabilities() == set(capabilities)


def test_false_health_rejects_before_adapter_registration(monkeypatch):
    import flash.serve.deployment.deploy as deploy_mod
    from flash.serve.contract.errors import ServingError

    monkeypatch.setattr(deploy_mod, "resolve_artifact_revision", lambda repo: "a" * 40)
    monkeypatch.setattr(
        adapter_check,
        "adapter_artifact_metadata",
        lambda *args, **kwargs: types.SimpleNamespace(lora_rank=32, targets_images=False),
    )
    calls = []

    class _Resp:
        def json(self):
            return {
                "ok": False,
                "capabilities": sorted(
                    REQUIRED_SERVING_CAPABILITIES | PREFERRED_SERVING_CAPABILITIES
                ),
            }

    def request(method, url, **kwargs):
        calls.append((method, url))
        if method != "GET":
            pytest.fail("adapter deployment continued after unhealthy serving response")
        return _Resp()

    monkeypatch.setattr(serving_transport, "serving_request", request)

    with pytest.raises(ServingError, match="reported ok=false"):
        deploy_adapter("flash-1-abc", "Qwen/Qwen3.5-9B", "repo", "rl/r1/seed0")

    assert calls == [("GET", f"{serving_urls.serving_base_url()}/healthz")]


def test_capability_preflight_preserves_non_200_health_failure(monkeypatch):
    import flash.serve.deployment.deploy as deploy_mod
    from flash.serve.contract.errors import ServingError

    url = f"{serving_urls.serving_base_url()}/healthz"
    response = httpx.Response(
        503,
        json={"ok": False, "capabilities": sorted(REQUIRED_SERVING_CAPABILITIES)},
        request=httpx.Request("GET", url),
    )
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: response)

    with pytest.raises(ServingError, match="HTTP 503") as exc_info:
        deploy_mod._require_serving_capabilities()

    assert exc_info.value.status_code == 503


def test_deploy_registers_one_exact_checkpoint_without_activation(monkeypatch) -> None:
    import flash.serve.deployment.deploy as deploy_mod

    _stub_deploy_preconditions(monkeypatch, deploy_mod)
    requests: list[tuple[str, str, object]] = []

    class _Response:
        status_code = 202

        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def json(self):
            return {}

    def request(method, url, *, json=None, **kwargs):
        requests.append((method, url, json))
        return _Response()

    monkeypatch.setattr(serving_transport, "serving_request", request)
    monkeypatch.setattr(deploy_mod, "_wait_checkpoint_ready", lambda *a, **k: {})

    result = deploy_mod.deploy_adapter(
        "flash-1",
        "Qwen/Qwen3.5-9B",
        "org/repo",
        "sft/flash-1/checkpoints/step_20",
        checkpoint_step=20,
        org_id="org-1",
    )

    assert result.checkpoint_id == "flash-1/step-20"
    assert result.openai_model == "flash-1/step-20"
    assert len(requests) == 1
    method, url, body = requests[0]
    assert method == "POST"
    assert url.endswith("/adapters")
    expected = {
        "adapter_id": "flash-1/step-20",
        "repo_id": "org/repo",
        "base_model": "Qwen/Qwen3.5-9B",
        "subfolder": "sft/flash-1/checkpoints/step_20/adapter",
        "repo_type": "dataset",
        "checkpoint": "flash-1/step-20",
        "run_id": "flash-1",
        "checkpoint_step": 20,
        "artifact_revision": "a" * 40,
        "artifact_digest": "b" * 64,
        "lora_rank": 32,
        "thinking": False,
        "org_id": "org-1",
    }
    assert body == {
        **expected,
        "artifact_fingerprint": immutable_binding_fingerprint(expected),
    }


def test_registration_conflict_is_not_masked(monkeypatch) -> None:
    import flash.serve.deployment.deploy as deploy_mod

    _stub_deploy_preconditions(monkeypatch, deploy_mod)
    monkeypatch.setattr(
        serving_transport,
        "serving_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            serving_errors.ServingError("checkpoint conflict", status_code=409)
        ),
    )

    with pytest.raises(serving_errors.ServingError, match="checkpoint conflict"):
        deploy_mod.deploy_adapter(
            "flash-1",
            "Qwen/Qwen3.5-9B",
            "org/repo",
            "sft/flash-1/seed0",
            org_id="org-1",
        )


def test_undeploy_calls_exact_checkpoint_delete(monkeypatch) -> None:
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example")
    deleted_urls: list[str] = []

    class _Response:
        status_code = 200

        def json(self):
            return {
                "run_id": "flash-1",
                "checkpoint_id": "flash-1/final",
                "disabled_checkpoints": ["flash-1/final"],
            }

    def request(method, url, **kwargs):
        assert method == "DELETE"
        deleted_urls.append(url)
        return _Response()

    monkeypatch.setattr(serving_transport, "serving_request", request)
    result = undeploy_adapter("flash-1/final", org_id="org-1")

    assert result["checkpoint_id"] == "flash-1/final"
    assert result["disabled_checkpoints"] == ["flash-1/final"]
    assert result["serving_deregistered"] is True
    assert deleted_urls == ["https://serve.example/adapters/flash-1%2Ffinal"]


def test_undeploy_404_is_clean(monkeypatch) -> None:
    class _Response:
        status_code = 404

    monkeypatch.setattr(serving_transport, "serving_request", lambda *a, **k: _Response())
    assert undeploy_adapter("flash-1/final", org_id="org-1") == {
        "checkpoint_id": "flash-1/final",
        "disabled_checkpoints": [],
        "serving_deregistered": False,
    }


def test_deployment_roundtrip_dict() -> None:
    deployment = Deployment(
        run_id="r",
        model="m",
        adapter_hf_prefix="p",
        openai_model="r/final",
        endpoint_name="https://serve.example",
        openai_base_url="https://serve.example/v1",
        checkpoint_id="r/final",
    )
    data = deployment.to_dict()
    assert data["checkpoint_id"] == "r/final"
    assert data["openai_base_url"] == "https://serve.example/v1"
    assert "url" not in data


def test_serving_capability_preflight_requires_permanent_identity(monkeypatch) -> None:
    import flash.serve.deployment.deploy as deploy_mod

    class _Response:
        def json(self):
            return {"capabilities": []}

    monkeypatch.setattr(serving_transport, "serving_request", lambda *a, **k: _Response())
    with pytest.raises(serving_errors.ServingError, match=PERMANENT_CHECKPOINT_IDENTITY_CAPABILITY):
        deploy_mod._require_serving_capabilities()


def test_capability_preflight_reads_healthz(monkeypatch) -> None:
    import flash.serve.deployment.deploy as deploy_mod

    seen: dict[str, str] = {}

    class _Response:
        def json(self):
            return {"capabilities": [PERMANENT_CHECKPOINT_IDENTITY_CAPABILITY]}

    def request(method, url, **kwargs):
        seen["method"] = method
        seen["url"] = url
        return _Response()

    monkeypatch.setattr(serving_transport, "serving_request", request)
    deploy_mod._require_serving_capabilities()
    assert seen == {"method": "GET", "url": f"{serving_urls.serving_base_url()}/healthz"}


def test_new_deployment_does_not_duplicate_existing_v1_suffix(monkeypatch) -> None:
    monkeypatch.setenv("FREESOLO_SERVING_URL", "https://serve.example/v1/")
    dep = deploy_adapter("r1", "Qwen/Qwen3.5-9B", "repo", "rl/r1/seed0", dry_run=True)
    assert dep.openai_base_url == "https://serve.example/v1"


def test_the_serving_extra_declares_every_module_scope_import_of_the_serving_app() -> None:
    import ast
    import re
    import sys
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.split(r"[<>=!\[; ]", name, maxsplit=1)[0].strip().lower().replace("-", "_")
        for name in pyproject["project"]["optional-dependencies"]["serving"]
    }
    declared |= {"pil"} if "pillow" in declared else set()
    declared |= {"dotenv"} if "python_dotenv" in declared else set()

    stdlib = set(sys.stdlib_module_names)
    missing: dict[str, set[str]] = {}
    for path in sorted((root / "flash" / "serving").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.col_offset == 0:
                top = (node.module or "").split(".")[0]
            elif isinstance(node, ast.Import) and node.col_offset == 0:
                top = node.names[0].name.split(".")[0]
            else:
                continue
            key = top.lower()
            if not top or top == "flash" or key in stdlib or key in declared:
                continue
            missing.setdefault(top, set()).add(str(path.relative_to(root)))

    assert not missing
