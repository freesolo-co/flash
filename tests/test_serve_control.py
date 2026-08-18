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

import flash.serve.control as control
from flash.serve.control import (
    AdapterAliasIntent,
    DeploymentRequest,
    DeploymentResult,
    DeploymentSpec,
    EngineIdentity,
    ModalCredentials,
    ModalPlacement,
    ModalProviderHandle,
    PlanningError,
    ResolvedAdapter,
    RunPodCredentials,
    RunPodPlacement,
    RunPodProviderHandle,
    canonical_mapping_fingerprint,
    plan_deployment,
    sanitized_dict,
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
POD_ID = "abc123def4567"
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
        "runtime_family": "vllm-0.23.0",
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
        "swap_space_gb": 4.0,
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
        "checkpoint": "final",
        "adapter_revision": f"{run_id}@final.{artifact_revision}",
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
        "alias_intent": AdapterAliasIntent(activate=False, expected_adapter_revision=None),
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
            workspace_id="workspace-1",
            environment="main",
            gpu="B200",
            region="us-east",
            volume_size_gb=100,
        ),
        engine=_ADAPTER_ENGINES.get(id(selected[0]), _engine()),
        adapters=selected,
    )


def _runpod_spec() -> DeploymentSpec:
    adapter = _adapter(1)
    return plan_deployment(
        DeploymentRequest(
            deployment_id="deployment-1",
            generation=2,
            provider="runpod",
            placement=RunPodPlacement(
                account_id="account-1",
                gpu_type_id="NVIDIA B200",
                gpu_count=1,
                data_center_id="US-KS-2",
                container_disk_gb=50,
                volume_size_gb=100,
            ),
            engine=_ADAPTER_ENGINES[id(adapter)],
            adapters=(adapter,),
        )
    )


def test_control_imports_without_runtime_or_provider_packages() -> None:
    probe = r"""
import builtins
import sys

blocked = ("torch", "vllm", "transformers", "PIL", "modal", "runpod", "runpod_flash", "httpx")
real_import = builtins.__import__


def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == blocked or name.startswith(tuple(item + "." for item in blocked)):
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded
from flash.serve.control import DeploymentSpec, EngineIdentity, ModalCredentials, plan_deployment

assert DeploymentSpec
assert EngineIdentity
assert ModalCredentials
assert plan_deployment
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
        "swap_space_gb": 8.0,
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
        _engine(swap_space_gb=value, cpu_offload_gb=value, mm_processor_cache_gb=value)
        for value in (0, 0.0, -0.0)
    ]
    assert identities[0] == identities[1] == identities[2]
    assert len({identity.engine_id for identity in identities}) == 1
    assert all(identity.swap_space_gb == 0.0 for identity in identities)

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
    assert "group_id" not in sanitized_dict(forward)
    assert "group_name" not in sanitized_dict(forward)
    for legacy_name in ("DeploymentGroup", "DeploymentPlan", "GroupResult"):
        assert not hasattr(control, legacy_name)


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
        sanitized_dict(planned)


def test_request_is_the_single_engine_authority() -> None:
    engine = _engine(dtype="float16")
    adapter = _adapter(1, _engine())
    request = replace(_modal_request(adapter), engine=engine)

    spec = plan_deployment(request)

    assert spec.engine is engine
    assert sanitized_dict(spec)["engine"]["dtype"] == "float16"
    assert "engine" not in sanitized_dict(spec)["adapters"][0]


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
    first = _adapter(1, _engine(swap_space_gb=0))
    second = _adapter(2, _engine(swap_space_gb=-0.0))
    forward = plan_deployment(_modal_request(first, second))
    reverse = plan_deployment(_modal_request(second, first))
    assert forward == reverse


def test_planning_accepts_exactly_modal_and_persistent_runpod_placements() -> None:
    modal = plan_deployment(_modal_request(_adapter(1)))
    assert modal.provider == "modal"
    assert type(modal.placement) is ModalPlacement

    runpod = _runpod_spec()
    assert runpod.provider == "runpod"
    assert type(runpod.placement) is RunPodPlacement

    for provider in ("vast", "lambda", "runpod-serverless", "custom"):
        with pytest.raises(PlanningError, match="modal or runpod"):
            plan_deployment(_modal_request(_adapter(1), provider=provider))


def test_planning_rejects_provider_placement_and_gpu_count_mismatches() -> None:
    with pytest.raises(PlanningError, match="RunPodPlacement"):
        plan_deployment(replace(_modal_request(_adapter(1)), provider="runpod"))
    request = _modal_request(_adapter(1))
    with pytest.raises(ValueError, match="workspace_id"):
        replace(request.placement, workspace_id="")

    engine = _engine(tensor_parallel_size=2)
    adapter = _adapter(1, engine)
    with pytest.raises(PlanningError, match=r"gpu_count.*tensor_parallel_size"):
        plan_deployment(_modal_request(adapter))
    assert plan_deployment(
        replace(_modal_request(adapter), placement=replace(request.placement, gpu_count=2))
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"adapter_revision": "run-1/final"}, "full immutable"),
        ({"adapter_revision": "run-1@final.main"}, "full immutable"),
        ({"adapter_revision": f"other@final.{11:040x}"}, "does not belong"),
        ({"checkpoint": "step-1"}, "checkpoint"),
        ({"artifact_repo_id": "missing-owner"}, "owner/name"),
        ({"artifact_repo_type": "space"}, "model or dataset"),
        ({"artifact_revision": "A" * 40}, "lowercase"),
        ({"artifact_revision": "9" * 40}, "does not match"),
        ({"artifact_digest": "A" * 64}, "lowercase"),
        ({"artifact_digest": "a" * 63}, "lowercase"),
        ({"artifact_subfolder": "/sft/run-1"}, "safe relative"),
        ({"artifact_subfolder": "sft/../run-1"}, "unsafe"),
        ({"artifact_subfolder": "sft\\run-1"}, "safe relative"),
        ({"base_model": ""}, "base_model"),
        ({"base_model_revision": "A" * 40}, "lowercase"),
        ({"lora_rank": 0}, "positive"),
        ({"alias_intent": None}, "explicit"),
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


def test_duplicate_revisions_and_multiple_alias_activations_are_rejected() -> None:
    adapter = _adapter(1)
    with pytest.raises(PlanningError, match="duplicate adapter revision"):
        plan_deployment(_modal_request(adapter, adapter))

    first = replace(
        adapter,
        alias_intent=AdapterAliasIntent(activate=True, expected_adapter_revision=None),
    )
    artifact_revision = "d" * 40
    second = replace(
        first,
        checkpoint="step-2",
        adapter_revision=f"run-1@step-2.{artifact_revision}",
        artifact_revision=artifact_revision,
        artifact_digest="e" * 64,
        artifact_subfolder="sft/run-1-step-2",
    )
    with pytest.raises(PlanningError, match="at most one active alias intent per run"):
        plan_deployment(_modal_request(first, second))

    planned = plan_deployment(
        _modal_request(first, replace(second, alias_intent=AdapterAliasIntent(False, None)))
    )
    assert len(planned.adapters) == 2


def test_alias_compare_and_swap_revision_must_be_immutable_and_same_run() -> None:
    for expected in ("main", f"other@final.{12:040x}"):
        with pytest.raises(ValueError, match="same run"):
            _adapter(
                1,
                alias_intent=AdapterAliasIntent(
                    activate=True,
                    expected_adapter_revision=expected,
                ),
            )

    adapter = _adapter(
        1,
        alias_intent=AdapterAliasIntent(
            activate=True,
            expected_adapter_revision=f"run-1@step-2.{12:040x}",
        ),
    )
    assert plan_deployment(_modal_request(adapter)).adapters == (adapter,)


def test_provider_credentials_are_redacted_and_fail_closed_for_serialization() -> None:
    credentials = (
        ModalCredentials("token-id", SECRET_SENTINEL),
        RunPodCredentials(SECRET_SENTINEL),
    )
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


def _modal_handle(spec: DeploymentSpec) -> ModalProviderHandle:
    return ModalProviderHandle(
        deployment_id=spec.deployment_id,
        generation=spec.generation,
        engine_id=spec.engine.engine_id,
        workspace_id="workspace-1",
        app_id="app-1",
        app_name="flash-app",
        volume_id="volume-1",
        volume_name="flash-volume",
        environment="main",
        region="us-east",
        image_digest=spec.engine.image_digest,
        public_url="https://flash-app.modal.run",
    )


def _runpod_handle(spec: DeploymentSpec) -> RunPodProviderHandle:
    return RunPodProviderHandle(
        deployment_id=spec.deployment_id,
        generation=spec.generation,
        engine_id=spec.engine.engine_id,
        account_id="account-1",
        pod_id=POD_ID,
        pod_name="flash-pod",
        network_volume_id="volume-1",
        network_volume_name="flash-volume",
        data_center_id="US-KS-2",
        image_digest=spec.engine.image_digest,
        public_url=f"https://{POD_ID}-8000.proxy.runpod.net",
    )


def test_sanitized_handles_and_results_are_json_safe_secret_free_and_flat() -> None:
    modal_spec = plan_deployment(_modal_request(_adapter(1)))
    runpod_spec = _runpod_spec()
    handles_and_specs = (
        (_modal_handle(modal_spec), modal_spec),
        (_runpod_handle(runpod_spec), runpod_spec),
    )
    spec_encoded = json.dumps(sanitized_dict(modal_spec), sort_keys=True)
    assert SECRET_SENTINEL not in spec_encoded

    for handle, spec in handles_and_specs:
        result = DeploymentResult.from_spec(spec, status="ready", handle=handle)
        payload = sanitized_dict(result)
        encoded = json.dumps(payload, sort_keys=True)
        assert SECRET_SENTINEL not in encoded
        assert handle.public_url in encoded
        assert handle.image_digest in encoded
        assert spec.engine.engine_id in encoded
        assert set(payload) == {
            "deployment_id",
            "generation",
            "provider",
            "placement",
            "engine_id",
            "image_digest",
            "spec_id",
            "status",
            "handle",
            "error_code",
        }
        assert not any("group" in key for key in payload)
        assert "group_id" not in {entry.name for entry in fields(handle)}

    with pytest.raises(ValueError, match="pod_id"):
        replace(handles_and_specs[1][0], pod_id="")
    with pytest.raises(ValueError, match="app_id"):
        replace(handles_and_specs[0][0], app_id="")


def test_deployment_result_requires_factory_bound_exact_spec_provenance() -> None:
    spec = plan_deployment(_modal_request(_adapter(1)))
    handle = _modal_handle(spec)

    with pytest.raises(TypeError, match="constructed from an exact DeploymentSpec"):
        DeploymentResult()

    legitimate = DeploymentResult.from_spec(spec, status="ready", handle=handle)
    forged = object.__new__(DeploymentResult)
    for entry in fields(legitimate):
        object.__setattr__(forged, entry.name, getattr(legitimate, entry.name))
    assert sanitized_dict(legitimate)["spec_id"] == spec.spec_id
    with pytest.raises(ValueError, match="exact DeploymentResult factory"):
        sanitized_dict(forged)

    raw_error = RuntimeError(SECRET_SENTINEL)
    with pytest.raises(ValueError, match="allowlisted deployment error") as exc_info:
        DeploymentResult.from_spec(spec, status="failed", error_code=raw_error)
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

    runpod = _runpod_handle(_runpod_spec())
    with pytest.raises(ValueError, match="ModalProviderHandle"):
        DeploymentResult.from_spec(spec, status="ready", handle=runpod)


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
        ("tearing_down", True, None),
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
    assert sanitized_dict(result)["status"] == status


@pytest.mark.parametrize(
    ("status", "with_handle", "error_code"),
    [
        ("ready", False, None),
        ("ready", True, "conflict"),
        ("provisioning", False, "conflict"),
        ("failed", False, None),
        ("outcome_unknown", True, None),
        ("tearing_down", False, None),
        ("absent", True, None),
        ("absent", False, "not_found"),
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


def test_deployment_result_validates_shared_placement_provenance() -> None:
    modal_spec = plan_deployment(_modal_request(_adapter(1)))
    modal_handle = _modal_handle(modal_spec)
    for field_name, value in (
        ("workspace_id", "workspace-other"),
        ("environment", "staging"),
        ("region", "us-west"),
    ):
        with pytest.raises(ValueError, match="placement"):
            DeploymentResult.from_spec(
                modal_spec,
                status="ready",
                handle=replace(modal_handle, **{field_name: value}),
            )

    runpod_spec = _runpod_spec()
    runpod_handle = _runpod_handle(runpod_spec)
    for field_name, value in (
        ("account_id", "account-other"),
        ("data_center_id", "EU-RO-1"),
    ):
        with pytest.raises(ValueError, match="placement"):
            DeploymentResult.from_spec(
                runpod_spec,
                status="ready",
                handle=replace(runpod_handle, **{field_name: value}),
            )


def test_deployment_result_equality_and_serialization_include_public_provenance() -> None:
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
    assert sanitized_dict(first)["generation"] == first_spec.generation
    assert sanitized_dict(second)["generation"] == second_spec.generation
    assert sanitized_dict(first)["spec_id"] == first_spec.spec_id
    assert sanitized_dict(different_adapters)["spec_id"] == different_adapters_spec.spec_id
    assert sanitized_dict(first)["placement"] == sanitized_dict(first_spec)["placement"]


@pytest.mark.parametrize(
    "pod_id",
    [
        "abc/../def4567",
        "abc@def456789",
        "abc:def456789",
        "abc.def456789",
        "ABC123DEF4567",
        "abc123def456",
        "abc123def45678",
    ],
)
def test_runpod_handle_rejects_pod_id_injection(pod_id: str) -> None:
    with pytest.raises(ValueError, match="pod_id"):
        replace(_runpod_handle(_runpod_spec()), pod_id=pod_id)


def test_runpod_handle_requires_canonical_proxy_origin() -> None:
    handle = _runpod_handle(_runpod_spec())
    assert handle.public_url == f"https://{POD_ID}-8000.proxy.runpod.net"
    malformed = (
        "http://" + f"{POD_ID}-8000.proxy.runpod.net",
        "https://user@" + f"{POD_ID}-8000.proxy.runpod.net",
        "https://" + f"{POD_ID}-8000.proxy.runpod.net:443",
        "https://" + f"{POD_ID}-8000.proxy.runpod.net/path",
        "https://" + f"{POD_ID}-8000.proxy.runpod.net?query=1",
        "https://" + f"{POD_ID}-8000.proxy.runpod.net#fragment",
        "https://other-8000.proxy.runpod.net",
    )
    for url in malformed:
        with pytest.raises(ValueError, match="public_url"):
            replace(handle, public_url=url)


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


def test_serializers_reject_subclasses_nonrecords_and_malformed_nested_records() -> None:
    spec = plan_deployment(_modal_request(_adapter(1)))

    class SpecSubclass(type(spec)):
        pass

    with pytest.raises(ValueError, match="exact DeploymentSpec"):
        SpecSubclass(
            deployment_id=spec.deployment_id,
            generation=spec.generation,
            provider=spec.provider,
            placement=spec.placement,
            engine=spec.engine,
            adapters=spec.adapters,
        )

    subclass = object.__new__(SpecSubclass)
    for entry in fields(spec):
        object.__setattr__(subclass, entry.name, getattr(spec, entry.name))
    object.__setattr__(subclass, "secret", SECRET_SENTINEL)
    with pytest.raises(TypeError, match="exact sanitized"):
        sanitized_dict(subclass)
    for nonrecord in (spec.engine, _modal_request(_adapter(1)), {"deployment_id": "x"}):
        with pytest.raises(TypeError, match="exact sanitized"):
            sanitized_dict(nonrecord)

    malformed = replace(spec, adapters=spec.adapters)
    object.__setattr__(malformed, "placement", ModalCredentials("id", SECRET_SENTINEL))
    with pytest.raises(ValueError, match="ModalPlacement"):
        sanitized_dict(malformed)


def test_serializers_revalidate_exact_records_after_forced_mutation() -> None:
    spec = plan_deployment(_modal_request(_adapter(1)))
    object.__setattr__(spec.engine, "max_cpu_loras", spec.engine.max_loras - 1)
    with pytest.raises(ValueError, match="max_cpu_loras must be at least max_loras"):
        sanitized_dict(spec)

    result_spec = plan_deployment(_modal_request(_adapter(2)))
    result = DeploymentResult.from_spec(result_spec, status="absent")
    object.__setattr__(result, "status", "failed")
    with pytest.raises(ValueError, match="require an allowlisted error_code"):
        sanitized_dict(result)


def test_control_records_have_no_credential_fields() -> None:
    records = (
        DeploymentRequest,
        DeploymentSpec,
        DeploymentResult,
        ModalProviderHandle,
        RunPodProviderHandle,
    )
    forbidden = {"credentials", "provider_credentials", "inference_credentials", "api_key", "token"}
    for record in records:
        assert forbidden.isdisjoint(entry.name for entry in fields(record))
