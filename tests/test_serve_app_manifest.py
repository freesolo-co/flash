"""strict execution manifest identity, schema, and control binding."""

from __future__ import annotations

import copy
import json
from dataclasses import fields, replace

import pytest

from flash.serve.app.manifest import (
    AdapterExecutionInput,
    ArtifactFile,
    ExecutionInputs,
    ManifestError,
    aggregate_file_digest,
    build_serving_manifest,
    load_serving_manifest,
)
from flash.serve.control import (
    DeploymentRequest,
    EngineIdentity,
    ModalPlacement,
    ResolvedAdapter,
    canonical_mapping_fingerprint,
    plan_deployment,
)
from flash.serve.deployment.profiles import get_profile, supported_models
from flash.serve.provisioning import ServingImage

MODEL_REVISION = "1" * 40
TOKENIZER_REVISION = "2" * 40
BASE_REVISION = "3" * 40
SOURCE_REVISION = "4" * 40
QWEN38_MODEL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
QWEN38_TOKENIZER_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
IMAGE_DIGEST = "sha256:" + "5" * 64
ENGINE_ARGS = {"enforce_eager": True}
TOKENIZER_KWARGS = {"use_fast": True}
PROCESSOR_KWARGS = {"max_pixels": 1024}


def _engine(**overrides: object) -> EngineIdentity:
    values: dict[str, object] = {
        "served_model": "flash-owned/served-checkpoint",
        "model_revision": MODEL_REVISION,
        "tokenizer_model": "flash-owned/tokenizer",
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
        "max_loras": 1,
        "max_cpu_loras": 2,
        "max_lora_rank": 64,
        "gpu_memory_utilization": 0.9,
        "cpu_offload_gb": 0.0,
        "image_limit": None,
        "mm_processor_cache_gb": 0.0,
        "enable_tower_connector_lora": False,
        "reasoning_parser": None,
        "trust_remote_code": False,
        "engine_args_fingerprint": canonical_mapping_fingerprint(ENGINE_ARGS),
        "tokenizer_kwargs_fingerprint": canonical_mapping_fingerprint(TOKENIZER_KWARGS),
        "processor_kwargs_fingerprint": canonical_mapping_fingerprint(PROCESSOR_KWARGS),
    }
    values.update(overrides)
    return EngineIdentity(**values)


def _files() -> tuple[ArtifactFile, ...]:
    return (
        ArtifactFile("adapter_config.json", 123, "6" * 64),
        ArtifactFile("adapter_model.safetensors", 456, "7" * 64),
    )


def _spec_and_inputs(
    *,
    engine: EngineIdentity | None = None,
    files: tuple[ArtifactFile, ...] | None = None,
):
    identity = engine or _engine()
    files = files or _files()
    aggregate = aggregate_file_digest(files)
    adapter = ResolvedAdapter(
        run_id="run-1",
        checkpoint_id="run-1/final",
        artifact_repo_id="flash-owned/run-artifacts",
        artifact_repo_type="model",
        artifact_revision=SOURCE_REVISION,
        artifact_digest=aggregate,
        artifact_subfolder="sft/run-1/adapter",
        base_model="Qwen/Qwen3.5-9B",
        base_model_revision=BASE_REVISION,
        lora_rank=16,
        thinking_default=True,
        structured_outputs_default_json='{"json_object":true}',
    )
    spec = plan_deployment(
        DeploymentRequest(
            deployment_id="deployment-1",
            generation=3,
            provider="modal",
            placement=ModalPlacement(
                workspace_name="workspace",
                environment="main",
                gpu="B200",
                region=None,
            ),
            engine=identity,
            adapters=(adapter,),
        )
    )
    inputs = ExecutionInputs(
        expected_oci_digest=IMAGE_DIGEST,
        engine_args=ENGINE_ARGS,
        tokenizer_kwargs=TOKENIZER_KWARGS,
        processor_kwargs=PROCESSOR_KWARGS,
        adapters=(AdapterExecutionInput(adapter.checkpoint_id, files),),
    )
    return spec, inputs


def _profile_spec_and_inputs(model_id: str):
    original, inputs = _spec_and_inputs()
    profile = get_profile(model_id)
    image = ServingImage(
        reference=f"registry.example/flash/serve@{IMAGE_DIGEST}",
        digest=IMAGE_DIGEST,
    )
    model_revision = QWEN38_MODEL_REVISION if model_id == "Qwen/Qwen3.8-27B" else MODEL_REVISION
    tokenizer_revision = (
        QWEN38_TOKENIZER_REVISION if model_id == "Qwen/Qwen3.8-27B" else TOKENIZER_REVISION
    )
    engine = profile.engine(
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        image=image,
    )
    adapter = replace(
        original.adapters[0],
        base_model=model_id,
        base_model_revision=BASE_REVISION,
        lora_rank=min(original.adapters[0].lora_rank, profile.max_lora_rank),
    )
    spec = replace(original, engine=engine, adapters=(adapter,))
    inputs = replace(
        inputs,
        engine_args=dict(profile.engine_args),
        tokenizer_kwargs=dict(profile.tokenizer_kwargs),
        processor_kwargs=dict(profile.processor_kwargs),
    )
    return spec, inputs


@pytest.mark.parametrize("model_id", supported_models())
def test_every_profile_engine_round_trips_through_the_manifest(model_id: str) -> None:
    spec, inputs = _profile_spec_and_inputs(model_id)
    manifest = build_serving_manifest(spec, inputs)
    loaded = load_serving_manifest(manifest.canonical_json())

    assert loaded.engine == spec.engine
    assert loaded.logical_base_model == model_id
    assert loaded.logical_base_revision == BASE_REVISION
    if model_id == "Qwen/Qwen3.8-27B":
        assert loaded.engine.model_revision == QWEN38_MODEL_REVISION
        assert loaded.engine.tokenizer_revision == QWEN38_TOKENIZER_REVISION
        assert loaded.engine.model_revision != loaded.engine.tokenizer_revision


def test_manifest_round_trip_is_canonical_and_data_only() -> None:
    spec, inputs = _spec_and_inputs()
    manifest = build_serving_manifest(spec, inputs)
    loaded = load_serving_manifest(manifest.canonical_json())

    assert loaded == manifest
    assert loaded.spec_id == spec.spec_id
    assert loaded.expected_oci_digest == spec.engine.image_digest
    assert loaded.adapters[0].checkpoint_id == "run-1/final"
    assert loaded.adapters[0].repo_id == "flash-owned/run-artifacts"
    assert loaded.adapters[0].files == _files()
    payload = json.loads(manifest.canonical_json())
    assert payload["schema"] == "flash.serving.manifest"
    assert payload["version"] == 2
    assert "manifest_id" not in manifest.payload(False)
    forbidden = {"token", "signed_url", "capability", "local_path", "wheel", "plugin"}

    def keys(value):
        if isinstance(value, dict):
            return set(value) | {key for nested in value.values() for key in keys(nested)}
        if isinstance(value, list):
            return {key for nested in value for key in keys(nested)}
        return set()

    assert forbidden.isdisjoint(keys(payload))


def test_manifest_rejects_unknown_keys_at_every_level_and_duplicate_json_keys() -> None:
    manifest = build_serving_manifest(*_spec_and_inputs())
    payload = json.loads(manifest.canonical_json())
    mutations = []
    for path in (
        (),
        ("logical_base",),
        ("engine",),
        ("adapters", 0),
        ("adapters", 0, "source"),
        ("adapters", 0, "files", 0),
    ):
        changed = copy.deepcopy(payload)
        target = changed
        for component in path:
            target = target[component]
        target["unknown"] = True
        mutations.append(changed)
    for changed in mutations:
        with pytest.raises(ManifestError, match="unknown"):
            load_serving_manifest(changed)
    with pytest.raises(ManifestError, match="duplicate"):
        load_serving_manifest('{"schema":"flash.serving.manifest","schema":"other"}')


def test_manifest_recomputes_fingerprints_aggregate_and_ids() -> None:
    spec, inputs = _spec_and_inputs()
    with pytest.raises(ManifestError, match="fingerprints"):
        build_serving_manifest(
            spec,
            replace(inputs, engine_args={"enforce_eager": False}),
        )
    wrong_files = (
        ArtifactFile("adapter_config.json", 123, "8" * 64),
        ArtifactFile("adapter_model.safetensors", 456, "7" * 64),
    )
    with pytest.raises(ManifestError, match="aggregate"):
        build_serving_manifest(
            spec,
            replace(
                inputs,
                adapters=(AdapterExecutionInput(spec.adapters[0].checkpoint_id, wrong_files),),
            ),
        )

    payload = json.loads(build_serving_manifest(spec, inputs).canonical_json())
    payload["generation"] += 1
    with pytest.raises(ManifestError, match="manifest_id"):
        load_serving_manifest(payload)


def test_manifest_rejects_unsupported_runtime_keys_code_artifacts_and_remote_code() -> None:
    with pytest.raises(ManifestError, match="unsupported keys"):
        ExecutionInputs(
            expected_oci_digest=IMAGE_DIGEST,
            engine_args={"dtype": "float16"},
            tokenizer_kwargs=TOKENIZER_KWARGS,
            processor_kwargs=PROCESSOR_KWARGS,
            adapters=(),
        )
    with pytest.raises(ManifestError, match="exactly config"):
        AdapterExecutionInput(
            "run-1/final",
            (
                ArtifactFile("adapter_config.json", 1, "1" * 64),
                ArtifactFile("plugin.py", 1, "2" * 64),
            ),
        )
    spec, inputs = _spec_and_inputs(engine=_engine(trust_remote_code=True))
    with pytest.raises(ManifestError, match="trust_remote_code"):
        build_serving_manifest(spec, inputs)


def test_manifest_accepts_every_current_data_only_runtime_option() -> None:
    spec, inputs = _spec_and_inputs()
    manifest = build_serving_manifest(spec, inputs)

    assert dict(manifest.engine_args) == {"enforce_eager": True}
    assert dict(manifest.tokenizer_kwargs) == {"use_fast": True}
    assert dict(manifest.processor_kwargs) == {"max_pixels": 1024}
    assert load_serving_manifest(manifest.canonical_json()) == manifest


@pytest.mark.parametrize(
    ("field_name", "key"),
    [
        ("engine_args", "load_format"),
        ("engine_args", "io_processor_plugin"),
        ("engine_args", "model_loader_extra_config"),
        ("engine_args", "reasoning_parser_plugin"),
        ("engine_args", "worker_cls"),
        ("engine_args", "worker_extension_cls"),
        ("engine_args", "download_dir"),
        ("tokenizer_kwargs", "tokenizer_type"),
        ("tokenizer_kwargs", "chat_template"),
        ("tokenizer_kwargs", "cache_dir"),
        ("processor_kwargs", "processor_class"),
        ("processor_kwargs", "chat_template"),
        ("processor_kwargs", "local_files_only"),
    ],
)
def test_manifest_rejects_executable_template_and_local_path_selectors(
    field_name: str,
    key: str,
) -> None:
    _spec, inputs = _spec_and_inputs()

    with pytest.raises(ManifestError, match=rf"{field_name} contains unsupported keys: {key}"):
        replace(inputs, **{field_name: {key: "selected"}})


def test_manifest_loading_rejects_unknown_runtime_options_before_identity_checks() -> None:
    payload = json.loads(build_serving_manifest(*_spec_and_inputs()).canonical_json())
    payload["engine"]["engine_args"] = {"load_format": "custom-loader"}

    with pytest.raises(ManifestError, match="engine_args contains unsupported keys: load_format"):
        load_serving_manifest(payload)


def test_manifest_rejects_invalid_allowed_option_values() -> None:
    _spec, inputs = _spec_and_inputs()
    invalid = (
        ("engine_args", {"enforce_eager": 1}),
        ("tokenizer_kwargs", {"use_fast": "true"}),
        ("processor_kwargs", {"max_pixels": 0}),
    )
    for field_name, value in invalid:
        with pytest.raises(ManifestError):
            replace(inputs, **{field_name: value})


def test_control_adapter_has_one_exact_source_and_no_engine_or_execution_fields() -> None:
    spec, _ = _spec_and_inputs()
    adapter_fields = {entry.name for entry in fields(ResolvedAdapter)}
    assert {
        "artifact_repo_id",
        "artifact_repo_type",
        "artifact_revision",
        "artifact_subfolder",
    } <= adapter_fields
    assert "engine" not in adapter_fields
    assert {"files", "local_path", "token", "signed_url", "capability"}.isdisjoint(adapter_fields)
    assert spec.engine == _engine()
