"""import purity, immutable planning, handles, and credential boundaries."""

from __future__ import annotations

import copy
import json
import pickle
import subprocess
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

from flash.serve.control import (
    DeploymentRequest,
    DeploymentResult,
    DeploymentSpec,
    EngineIdentity,
    ModalCredentials,
    ModalPlacement,
    ModalProviderHandle,
    PlanningError,
    ResolvedAdapter,
    canonical_mapping_fingerprint,
    plan_deployment,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "1" * 40
TOKENIZER_REVISION = "2" * 40
BASE_MODEL_REVISION = "4" * 40
IMAGE_DIGEST = "sha256:" + "3" * 64
ENGINE_ARGS_FINGERPRINT = canonical_mapping_fingerprint({"dtype": "bfloat16"})
TOKENIZER_KWARGS_FINGERPRINT = canonical_mapping_fingerprint({"use_fast": True})
PROCESSOR_KWARGS_FINGERPRINT = canonical_mapping_fingerprint({"max_pixels": 1024})
SECRET_SENTINEL = "provider-secret-sentinel"
_ADAPTER_ENGINES: dict[int, EngineIdentity] = {}
_ADAPTER_REFERENCES: list[ResolvedAdapter] = []


def _engine(**overrides: object) -> EngineIdentity:
    values: dict[str, object] = {
        "served_model": "org/serving-mirror",
        "model_revision": MODEL_REVISION,
        "tokenizer_model": "org/tokenizer-mirror",
        "tokenizer_revision": TOKENIZER_REVISION,
        "image_digest": IMAGE_DIGEST,
        "modality": "text",
        "runtime_family": "vllm-0.23.0-pr42120",
        "dtype": "bfloat16",
        "quantization": None,
        "kv_cache_dtype": None,
        "tensor_parallel_size": 1,
        "max_model_len": 8192,
        "max_num_seqs": 8,
        "max_num_batched_tokens": None,
        "max_loras": 2,
        "max_cpu_loras": 3,
        "max_lora_rank": 64,
        "gpu_memory_utilization": 0.9,
        "cpu_offload_gb": 0.0,
        "image_limit": None,
        "mm_processor_cache_gb": 0.0,
        "enable_tower_connector_lora": False,
        "reasoning_parser": None,
        "trust_remote_code": False,
        "engine_args_fingerprint": ENGINE_ARGS_FINGERPRINT,
        "tokenizer_kwargs_fingerprint": TOKENIZER_KWARGS_FINGERPRINT,
        "processor_kwargs_fingerprint": PROCESSOR_KWARGS_FINGERPRINT,
    }
    values.update(overrides)
    return EngineIdentity(**values)


def _adapter(
    index: int,
    engine: EngineIdentity | None = None,
    **overrides: object,
) -> ResolvedAdapter:
    identity = engine or _engine()
    run_id = f"run-{index}"
    artifact_revision = f"{index + 10:040x}"
    values: dict[str, object] = {
        "run_id": run_id,
        "checkpoint_id": f"{run_id}/final",
        "artifact_repo_id": "flash-owned/runs",
        "artifact_repo_type": "model",
        "artifact_revision": artifact_revision,
        "artifact_digest": f"{index + 100:064x}",
        "artifact_subfolder": f"sft/{run_id}",
        "base_model": "org/logical-base",
        "base_model_revision": BASE_MODEL_REVISION,
        "lora_rank": 16,
        "thinking_default": False,
        "structured_outputs_default_json": '{"json_object":true}',
    }
    values.update(overrides)
    adapter = ResolvedAdapter(**values)
    _ADAPTER_REFERENCES.append(adapter)
    _ADAPTER_ENGINES[id(adapter)] = identity
    return adapter


def _modal_request(*adapters: ResolvedAdapter, provider: str = "modal") -> DeploymentRequest:
    selected = tuple(adapters) or (_adapter(1),)
    return DeploymentRequest(
        deployment_id="deployment-1",
        generation=2,
        provider=provider,
        placement=ModalPlacement(
            workspace_name="workspace-1",
            environment="main",
            gpu="B200",
            region="us-east",
        ),
        engine=_ADAPTER_ENGINES.get(id(selected[0]), _engine()),
        adapters=selected,
    )


def test_engine_id_is_canonical_stable_and_sensitive_to_every_field() -> None:
    identity = _engine()
    assert identity.engine_id == _engine().engine_id
    assert list(json.loads(identity.canonical_json)) == sorted(
        entry.name for entry in fields(identity)
    )

    alternatives: dict[str, object] = {
        "served_model": "org/other-serving-checkpoint",
        "model_revision": "5" * 40,
        "tokenizer_model": "org/other-tokenizer",
        "tokenizer_revision": "6" * 40,
        "image_digest": "sha256:" + "7" * 64,
        "modality": "multimodal",
        "runtime_family": "vllm-other",
        "dtype": "float16",
        "quantization": "awq",
        "kv_cache_dtype": "fp8",
        "tensor_parallel_size": 2,
        "max_model_len": 4096,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 4096,
        "max_loras": 1,
        "max_cpu_loras": 5,
        "max_lora_rank": 128,
        "gpu_memory_utilization": 0.8,
        "cpu_offload_gb": 1.0,
        "image_limit": 4,
        "mm_processor_cache_gb": 1.0,
        "enable_tower_connector_lora": True,
        "reasoning_parser": "qwen3",
        "trust_remote_code": True,
        "engine_args_fingerprint": "8" * 64,
        "tokenizer_kwargs_fingerprint": "9" * 64,
        "processor_kwargs_fingerprint": "a" * 64,
    }
    assert set(alternatives) == {entry.name for entry in fields(identity)}
    for name, value in alternatives.items():
        overrides = {name: value}
        if name == "modality":
            overrides["image_limit"] = 4
        elif name == "image_limit":
            overrides["modality"] = "multimodal"
        elif name == "enable_tower_connector_lora":
            overrides.update(modality="multimodal", image_limit=4)
        assert replace(identity, **overrides).engine_id != identity.engine_id, name


def test_engine_identity_enforces_modality_and_image_limit_consistency() -> None:
    text = _engine()
    multimodal = _engine(
        modality="multimodal",
        image_limit=4,
        enable_tower_connector_lora=True,
    )
    assert text.image_limit is None
    assert multimodal.image_limit == 4

    with pytest.raises(ValueError, match="text engines cannot declare"):
        _engine(image_limit=1)
    with pytest.raises(ValueError, match="multimodal engines require"):
        _engine(modality="multimodal")
    with pytest.raises(ValueError, match="text engines cannot enable tower connector"):
        _engine(enable_tower_connector_lora=True)


def test_engine_identity_normalizes_numeric_equivalence_and_requires_integer_types() -> None:
    identities = [
        _engine(cpu_offload_gb=value, mm_processor_cache_gb=value) for value in (0, 0.0, -0.0)
    ]
    assert identities[0] == identities[1] == identities[2]
    assert len({identity.engine_id for identity in identities}) == 1
    assert all(identity.cpu_offload_gb == 0.0 for identity in identities)

    with pytest.raises(ValueError, match="tensor_parallel_size must be an integer"):
        _engine(tensor_parallel_size=1.0)


def test_canonical_mapping_fingerprints_order_and_construction_inputs() -> None:
    first = {"nested": {"b": 2.0, "a": [0.0, -0.0]}, "enabled": True}
    second = {"enabled": True, "nested": {"a": [0, 0], "b": 2}}
    assert canonical_mapping_fingerprint(first) == canonical_mapping_fingerprint(second)
    assert canonical_mapping_fingerprint({"use_fast": True}) != canonical_mapping_fingerprint(
        {"use_fast": False}
    )
    assert _engine(tokenizer_kwargs_fingerprint="b" * 64).engine_id != _engine().engine_id
    assert _engine(processor_kwargs_fingerprint="c" * 64).engine_id != _engine().engine_id
    for field_name in (
        "engine_args_fingerprint",
        "tokenizer_kwargs_fingerprint",
        "processor_kwargs_fingerprint",
    ):
        with pytest.raises(ValueError, match=field_name):
            _engine(**{field_name: "not-a-digest"})


def test_multiple_compatible_adapters_return_one_order_stable_spec() -> None:
    engine = _engine()
    first = _adapter(1, engine, lora_rank=16)
    second = _adapter(2, engine, lora_rank=32)

    forward = plan_deployment(_modal_request(first, second))
    reverse = plan_deployment(_modal_request(second, first))

    assert type(forward) is DeploymentSpec
    assert forward == reverse
    assert forward.engine is engine
    assert forward.adapters == (first, second)
    assert "engine" not in {entry.name for entry in fields(ResolvedAdapter)}
    assert "engine" in {entry.name for entry in fields(DeploymentRequest)}
    assert "groups" not in {entry.name for entry in fields(forward)}
    assert "group_id" not in {entry.name for entry in fields(forward)}
    assert "group_name" not in {entry.name for entry in fields(forward)}


def test_deployment_spec_requires_canonical_adapter_order_at_every_boundary() -> None:
    engine = _engine()
    planned = plan_deployment(_modal_request(_adapter(2, engine), _adapter(1, engine)))
    reversed_adapters = tuple(reversed(planned.adapters))

    with pytest.raises(ValueError, match="canonical ordering"):
        DeploymentSpec(
            deployment_id=planned.deployment_id,
            generation=planned.generation,
            provider=planned.provider,
            placement=planned.placement,
            engine=planned.engine,
            adapters=reversed_adapters,
        )

    object.__setattr__(planned, "adapters", reversed_adapters)
    with pytest.raises(ValueError, match="canonical ordering"):
        _ = planned.spec_id


def test_request_is_the_single_engine_authority() -> None:
    engine = _engine(dtype="float16")
    adapter = _adapter(1, _engine())
    request = replace(_modal_request(adapter), engine=engine)

    spec = plan_deployment(request)

    assert spec.engine is engine
    assert spec.engine.dtype == "float16"
    assert "engine" not in {entry.name for entry in fields(spec.adapters[0])}


def test_capacity_at_limit_succeeds_and_capacity_plus_one_fails_without_chunking() -> None:
    engine = _engine(max_loras=2, max_cpu_loras=3)
    exact = (_adapter(1, engine), _adapter(2, engine), _adapter(3, engine))
    spec = plan_deployment(_modal_request(*exact))
    assert engine.adapter_capacity == 3
    assert spec.adapters == exact

    with pytest.raises(PlanningError, match=r"validated max_cpu_loras capacity"):
        plan_deployment(_modal_request(*exact, _adapter(4, engine)))


def test_adapter_rank_is_per_adapter_under_one_engine_ceiling() -> None:
    engine = _engine(max_lora_rank=64)
    spec = plan_deployment(
        _modal_request(
            _adapter(1, engine, lora_rank=8),
            _adapter(2, engine, lora_rank=64),
        )
    )
    assert [adapter.lora_rank for adapter in spec.adapters] == [8, 64]
    assert "lora_rank" not in {entry.name for entry in fields(engine)}

    with pytest.raises(PlanningError, match="exceeds engine max_lora_rank"):
        plan_deployment(_modal_request(_adapter(1, engine, lora_rank=65)))


def test_logical_adapter_base_model_may_differ_from_exact_served_checkpoint() -> None:
    adapter = _adapter(
        1,
        _engine(served_model="mirror/prequantized", model_revision="5" * 40),
        base_model="vendor/logical-base",
        base_model_revision="6" * 40,
    )
    spec = plan_deployment(_modal_request(adapter))
    assert spec.adapters[0].base_model == "vendor/logical-base"
    assert spec.engine.served_model == "mirror/prequantized"
    assert spec.adapters[0].base_model_revision != spec.engine.model_revision


@pytest.mark.parametrize(
    "changes",
    [
        {"base_model": "vendor/other-base"},
        {"base_model_revision": "7" * 40},
        {"base_model": "vendor/other-base", "base_model_revision": "7" * 40},
    ],
)
def test_planning_rejects_mixed_logical_base_provenance(changes: dict[str, object]) -> None:
    engine = _engine()
    first = _adapter(1, engine)
    second = _adapter(2, engine, **changes)
    with pytest.raises(PlanningError, match="same logical base model and revision"):
        plan_deployment(_modal_request(first, second))


def test_numeric_equivalent_engines_produce_permutation_stable_specs() -> None:
    first = _adapter(1, _engine(cpu_offload_gb=0))
    second = _adapter(2, _engine(cpu_offload_gb=-0.0))
    forward = plan_deployment(_modal_request(first, second))
    reverse = plan_deployment(_modal_request(second, first))
    assert forward == reverse


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"checkpoint_id": "run-1"}, "checkpoint_id"),
        ({"checkpoint_id": "run-1/step-01"}, "checkpoint_id"),
        ({"checkpoint_id": "other/final"}, "does not belong"),
        ({"artifact_repo_id": "missing-owner"}, "owner/name"),
        ({"artifact_repo_type": "space"}, "model or dataset"),
        ({"artifact_revision": "A" * 40}, "lowercase"),
        ({"artifact_revision": "g" * 40}, "lowercase"),
        ({"artifact_digest": "A" * 64}, "lowercase"),
        ({"artifact_digest": "a" * 63}, "lowercase"),
        ({"artifact_subfolder": "/sft/run-1"}, "safe relative"),
        ({"artifact_subfolder": "sft/../run-1"}, "unsafe"),
        ({"artifact_subfolder": "sft\\run-1"}, "safe relative"),
        ({"base_model": ""}, "base_model"),
        ({"base_model_revision": "A" * 40}, "lowercase"),
        ({"lora_rank": 0}, "positive"),
    ],
)
def test_resolved_adapter_rejects_invalid_intrinsic_fields(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _adapter(1, **changes)


@pytest.mark.parametrize(
    "revision",
    ["main", "release", "A" * 40, "a" * 39, "a" * 41, " " + "a" * 40, "a" * 40 + " "],
)
def test_engine_identity_requires_exact_immutable_revisions(revision: str) -> None:
    for field_name in ("model_revision", "tokenizer_revision"):
        with pytest.raises(ValueError, match=field_name):
            _engine(**{field_name: revision})
    assert _engine(model_revision="a" * 40, tokenizer_revision="b" * 40)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"max_loras": 0}, "max_loras"),
        ({"max_cpu_loras": 0}, "max_cpu_loras"),
        ({"max_lora_rank": 0}, "max_lora_rank"),
    ],
)
def test_engine_identity_rejects_invalid_capacity_on_direct_construction(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _engine(**changes)


def test_duplicate_checkpoint_identities_are_rejected() -> None:
    adapter = _adapter(1)
    with pytest.raises(PlanningError, match="duplicate checkpoint identity"):
        plan_deployment(_modal_request(adapter, adapter))


def test_sibling_checkpoints_remain_independent_in_one_deployment() -> None:
    first = _adapter(1)
    second = replace(
        first,
        checkpoint_id="run-1/step-2",
        artifact_revision="d" * 40,
        artifact_digest="e" * 64,
        artifact_subfolder="sft/run-1-step-2",
    )

    planned = plan_deployment(_modal_request(first, second))
    assert [adapter.checkpoint_id for adapter in planned.adapters] == [
        "run-1/final",
        "run-1/step-2",
    ]


def _modal_handle(spec: DeploymentSpec) -> ModalProviderHandle:
    return ModalProviderHandle(
        deployment_id=spec.deployment_id,
        generation=spec.generation,
        engine_id=spec.engine.engine_id,
        workspace_name="workspace-1",
        app_id="ap-" + "A" * 22,
        app_name="flash-app",
        volume_id="vo-" + "V" * 22,
        volume_name="flash-volume",
        inference_secret_id="st-" + "S" * 22,
        inference_secret_name="flash-inference-secret",
        environment="main",
        region="us-east",
        image_digest=spec.engine.image_digest,
        public_url="https://flash-app.modal.run",
    )


@pytest.mark.parametrize(
    ("status", "with_handle", "error_code"),
    [
        ("ready", True, None),
        ("provisioning", False, None),
        ("provisioning", True, None),
        ("failed", False, "provider_rejected"),
        ("failed", True, "provider_rejected"),
        ("outcome_unknown", False, "resource_ambiguous"),
        ("outcome_unknown", True, "resource_ambiguous"),
        ("absent", False, None),
    ],
)
def test_deployment_result_accepts_complete_lifecycle_matrix(
    status: str,
    with_handle: bool,
    error_code: str | None,
) -> None:
    spec = plan_deployment(_modal_request(_adapter(1)))
    handle = _modal_handle(spec) if with_handle else None
    result = DeploymentResult.from_spec(
        spec,
        status=status,
        handle=handle,
        error_code=error_code,
    )
    assert result.status == status


@pytest.mark.parametrize(
    ("status", "with_handle", "error_code"),
    [
        ("ready", False, None),
        ("ready", True, "conflict"),
        ("provisioning", False, "conflict"),
        ("failed", False, None),
        ("outcome_unknown", True, None),
        ("absent", True, None),
        ("absent", False, "conflict"),
    ],
)
def test_deployment_result_rejects_invalid_lifecycle_matrix(
    status: str,
    with_handle: bool,
    error_code: str | None,
) -> None:
    spec = plan_deployment(_modal_request(_adapter(1)))
    handle = _modal_handle(spec) if with_handle else None
    with pytest.raises(ValueError, match=r"require|cannot carry"):
        DeploymentResult.from_spec(
            spec,
            status=status,
            handle=handle,
            error_code=error_code,
        )


def test_deployment_result_equality_and_fields_include_public_provenance() -> None:
    first_spec = plan_deployment(_modal_request(_adapter(1)))
    second_spec = replace(first_spec, generation=first_spec.generation + 1)
    first = DeploymentResult.from_spec(first_spec, status="ready", handle=_modal_handle(first_spec))
    second = DeploymentResult.from_spec(
        second_spec,
        status="ready",
        handle=_modal_handle(second_spec),
    )
    other_placement_spec = replace(
        first_spec,
        placement=replace(first_spec.placement, environment="staging"),
    )
    other_placement = DeploymentResult.from_spec(
        other_placement_spec,
        status="ready",
        handle=replace(_modal_handle(other_placement_spec), environment="staging"),
    )
    different_adapters_spec = plan_deployment(
        _modal_request(first_spec.adapters[0], _adapter(2, first_spec.engine))
    )
    different_adapters = DeploymentResult.from_spec(
        different_adapters_spec,
        status="ready",
        handle=_modal_handle(different_adapters_spec),
    )

    assert first != second
    assert first != other_placement
    assert first != different_adapters
    assert first.spec_id != different_adapters.spec_id
    assert first_spec.spec_id != different_adapters_spec.spec_id
    assert first.generation == first_spec.generation
    assert second.generation == second_spec.generation
    assert first.spec_id == first_spec.spec_id
    assert different_adapters.spec_id == different_adapters_spec.spec_id
    assert first.placement == first_spec.placement


def test_two_placements_differing_only_in_web_suffix_are_two_specs() -> None:
    """The web suffix is part of the spec identity, because it is part of the public URL.

    Modal builds a web url as `<workspace>-<web_suffix>--<label>.modal.run`, so a placement that
    differs only in its suffix names a different endpoint. If `_placement_payload` omits it, the
    two hash to one `spec_id`: the identity no longer distinguishes the origin it describes,
    and the probe's `provenance["spec_id"]` check would accept a deployment reachable at a
    different url than the one that was planned.
    """
    without = plan_deployment(_modal_request())
    with_suffix = plan_deployment(
        DeploymentRequest(
            deployment_id="deployment-1",
            generation=2,
            provider="modal",
            placement=ModalPlacement(
                workspace_name="workspace-1",
                environment="main",
                gpu="B200",
                region="us-east",
                web_suffix="team",
            ),
            engine=without.engine,
            adapters=without.adapters,
        )
    )

    assert without.spec_id != with_suffix.spec_id
    assert without.placement != with_suffix.placement


@pytest.mark.parametrize(
    "url",
    [
        "http://flash-app.modal.run",
        "https://user@flash-app.modal.run",
        "https://flash-app.modal.run:443",
        "https://flash-app.modal.run:bad",
        "https://flash-app.modal.run/path",
        "https://flash-app.modal.run?query=1",
        "https://flash-app.modal.run#fragment",
        "https://flash-app.modal.run.evil.example",
        "https://flash_app.modal.run",
        "https://-flash.modal.run",
        "https://flash-.modal.run",
        "https://FLASH-APP.modal.run",
    ],
)
def test_modal_handle_rejects_noncanonical_provider_urls(url: str) -> None:
    spec = plan_deployment(_modal_request(_adapter(1)))
    with pytest.raises(ValueError, match="public_url"):
        replace(_modal_handle(spec), public_url=url)


def test_control_imports_without_runtime_or_provider_packages() -> None:
    probe = r"""
import builtins
import sys

blocked = ("torch", "vllm", "transformers", "PIL", "modal", "runpod", "runpod_flash", "httpx")
real_import = builtins.__import__


intercepted = []


def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name in blocked or name.startswith(tuple(item + "." for item in blocked)):
        intercepted.append(name)
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded
try:
    __import__("modal")
except ModuleNotFoundError:
    pass
assert intercepted == ["modal"]
intercepted.clear()
from flash.serve.control import DeploymentSpec, EngineIdentity, ModalCredentials, plan_deployment

assert DeploymentSpec
assert EngineIdentity
assert ModalCredentials
assert plan_deployment
assert intercepted == []
for name in blocked:
    assert name not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", probe],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_planning_accepts_only_exact_modal_placements() -> None:
    modal = plan_deployment(_modal_request(_adapter(1)))
    assert modal.provider == "modal"
    assert type(modal.placement) is ModalPlacement

    for provider in ("vast", "lambda", "custom"):
        with pytest.raises(PlanningError, match="provider must be modal"):
            plan_deployment(_modal_request(_adapter(1), provider=provider))


def test_planning_rejects_provider_placement_and_gpu_count_mismatches() -> None:
    with pytest.raises(PlanningError, match="provider must be modal"):
        plan_deployment(replace(_modal_request(_adapter(1)), provider="other"))
    request = _modal_request(_adapter(1))
    with pytest.raises(ValueError, match="workspace_name"):
        replace(request.placement, workspace_name="")

    engine = _engine(tensor_parallel_size=2)
    adapter = _adapter(1, engine)
    with pytest.raises(PlanningError, match=r"gpu_count.*tensor_parallel_size"):
        plan_deployment(_modal_request(adapter))
    assert plan_deployment(
        replace(_modal_request(adapter), placement=replace(request.placement, gpu_count=2))
    )


def test_provider_credentials_are_redacted_and_fail_closed_for_serialization() -> None:
    credentials = (ModalCredentials("token-id", SECRET_SENTINEL),)
    for credential in credentials:
        assert SECRET_SENTINEL not in repr(credential)
        assert "redacted" in repr(credential).lower()
        with pytest.raises(TypeError):
            asdict(credential)
        with pytest.raises(TypeError):
            json.dumps(credential)
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(credential)
        with pytest.raises(TypeError, match="cannot expose serialization state"):
            credential.__getstate__()
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(credential)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(credential)
        with pytest.raises(TypeError, match="cannot be subclassed"):
            type("CredentialSubclass", (type(credential),), {})
        with pytest.raises(TypeError):
            vars(credential)

    with pytest.raises(ValueError, match="token_secret") as exc_info:
        ModalCredentials(SECRET_SENTINEL, "")
    assert SECRET_SENTINEL not in str(exc_info.value)


def test_handles_and_results_expose_exact_secret_free_public_fields() -> None:
    spec = plan_deployment(_modal_request(_adapter(1)))
    handle = _modal_handle(spec)
    result = DeploymentResult.from_spec(spec, status="ready", handle=handle)

    assert spec.placement.workspace_name == "workspace-1"
    assert "workspace_id" not in {entry.name for entry in fields(spec.placement)}
    assert "volume_size_gb" not in {entry.name for entry in fields(spec.placement)}
    assert {entry.name for entry in fields(result)} == {
        "spec",
        "status",
        "handle",
        "error_code",
        "error_reason",
    }
    assert result.deployment_id == spec.deployment_id
    assert result.generation == spec.generation
    assert result.provider == "modal"
    assert result.placement == spec.placement
    assert result.engine_id == spec.engine.engine_id
    assert result.image_digest == handle.image_digest
    assert result.spec_id == spec.spec_id
    assert result.status == "ready"
    assert result.handle is handle
    assert result.error_code is None
    assert "group_id" not in {entry.name for entry in fields(handle)}
    assert handle.inference_secret_name == "flash-inference-secret"
    assert not any(entry.name.startswith("artifact_secret") for entry in fields(handle))
    assert SECRET_SENTINEL not in repr(
        (result.spec, result.status, result.handle, result.error_code)
    )
    assert handle.workspace_name == "workspace-1"
    assert "workspace_id" not in {entry.name for entry in fields(handle)}

    with pytest.raises(ValueError, match="app_id"):
        replace(handle, app_id="")
    with pytest.raises(ValueError, match="inference_secret_id"):
        replace(handle, inference_secret_id="")


def test_deployment_result_requires_an_exact_spec_and_matching_handle_provenance() -> None:
    spec = plan_deployment(_modal_request(_adapter(1)))
    handle = _modal_handle(spec)

    with pytest.raises(TypeError, match="constructed from an exact DeploymentSpec"):
        DeploymentResult()

    legitimate = DeploymentResult.from_spec(spec, status="ready", handle=handle)
    assert legitimate.spec_id == spec.spec_id

    raw_error = RuntimeError(SECRET_SENTINEL)
    with pytest.raises(ValueError, match="allowlisted deployment error") as exc_info:
        DeploymentResult.from_spec(spec, status="failed", error_code=raw_error)
    assert SECRET_SENTINEL not in str(exc_info.value)

    with pytest.raises(ValueError, match="allowlisted deployment reason") as exc_info:
        DeploymentResult.from_spec(
            spec,
            status="failed",
            error_code="resource_ambiguous",
            error_reason=SECRET_SENTINEL,
        )
    assert SECRET_SENTINEL not in str(exc_info.value)

    mismatches = (
        replace(handle, deployment_id="deployment-other"),
        replace(handle, generation=spec.generation + 1),
        replace(handle, engine_id="f" * 64),
        replace(handle, image_digest="sha256:" + "f" * 64),
    )
    for mismatched in mismatches:
        with pytest.raises(ValueError, match="provenance"):
            DeploymentResult.from_spec(spec, status="ready", handle=mismatched)

    with pytest.raises(ValueError, match="ModalProviderHandle"):
        DeploymentResult.from_spec(spec, status="ready", handle=object())


def test_deployment_result_validates_shared_placement_provenance() -> None:
    modal_spec = plan_deployment(_modal_request(_adapter(1)))
    modal_handle = _modal_handle(modal_spec)
    for field_name, value in (
        ("workspace_name", "workspace-other"),
        ("environment", "staging"),
        ("region", "us-west"),
    ):
        with pytest.raises(ValueError, match="placement"):
            DeploymentResult.from_spec(
                modal_spec,
                status="ready",
                handle=replace(modal_handle, **{field_name: value}),
            )


def test_control_records_have_no_credential_fields() -> None:
    records = (
        DeploymentRequest,
        DeploymentSpec,
        DeploymentResult,
        ModalProviderHandle,
    )
    forbidden = {"credentials", "provider_credentials", "inference_credentials", "api_key", "token"}
    for record in records:
        assert forbidden.isdisjoint(entry.name for entry in fields(record))
