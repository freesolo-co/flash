"""serving deployment composition, preview allowlist, and import purity."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from flash.serve.app import ManifestError
from flash.serve.control import DeploymentRequest, DeploymentSpec, ModalPlacement
from flash.serve.deployment.profiles import get_profile, placement_for, supported_models
from flash.serve.provisioning import ServingImage, serving_resource_names
from flash.server.domain.ops.serving_resources import dry_run_deployment, resolve_deployment_bundle
from tests.test_serve_app_manifest import IMAGE_DIGEST, _profile_spec_and_inputs, _spec_and_inputs

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_SECRET_SENTINEL = "provider-secret-sentinel"
OTHER_IMAGE_DIGEST = "sha256:" + "9" * 64


def _request(spec: DeploymentSpec) -> DeploymentRequest:
    return DeploymentRequest(
        deployment_id=spec.deployment_id,
        generation=spec.generation,
        provider=spec.provider,
        placement=spec.placement,
        engine=spec.engine,
        adapters=spec.adapters,
    )


def _image(digest: str = IMAGE_DIGEST) -> ServingImage:
    return ServingImage(
        reference=f"registry.example/flash/shared-serving-runtime@{digest}",
        digest=digest,
    )


def test_resolve_deployment_bundle_composes_matching_public_identities() -> None:
    expected_spec, execution_inputs = _spec_and_inputs()
    image = _image()

    bundle = resolve_deployment_bundle(_request(expected_spec), execution_inputs, image)

    assert bundle.spec == expected_spec
    assert bundle.spec.engine is expected_spec.engine
    assert bundle.manifest.spec_id == bundle.spec.spec_id
    assert bundle.manifest.engine is bundle.spec.engine
    assert bundle.manifest.engine.engine_id == bundle.spec.engine.engine_id
    assert bundle.manifest.expected_oci_digest == bundle.spec.engine.image_digest
    assert bundle.image is image
    assert bundle.image.digest == bundle.manifest.expected_oci_digest


def test_resolve_reuses_existing_image_binding_validators() -> None:
    spec, execution_inputs = _spec_and_inputs()
    request = _request(spec)

    with pytest.raises(ValueError, match="reference digest does not match"):
        ServingImage(
            reference=f"registry.example/flash/shared-serving-runtime@{IMAGE_DIGEST}",
            digest=OTHER_IMAGE_DIGEST,
        )

    with pytest.raises(ManifestError, match="execution image digest"):
        resolve_deployment_bundle(
            request,
            replace(execution_inputs, expected_oci_digest=OTHER_IMAGE_DIGEST),
            _image(),
        )

    with pytest.raises(ValueError, match="bundle image binding"):
        resolve_deployment_bundle(request, execution_inputs, _image(OTHER_IMAGE_DIGEST))


def test_json_preview_excludes_nonallowlisted_bundle_data() -> None:
    spec, execution_inputs = _spec_and_inputs()
    placement = ModalPlacement(
        workspace_name=PROVIDER_SECRET_SENTINEL,
        environment="placement-environment-sentinel",
        gpu="placement-gpu-sentinel",
        region="placement-region-sentinel",
    )
    engine = spec.engine
    request = DeploymentRequest(
        deployment_id=spec.deployment_id,
        generation=spec.generation,
        provider=spec.provider,
        placement=placement,
        engine=engine,
        adapters=spec.adapters,
    )
    image = _image()
    bundle = resolve_deployment_bundle(request, execution_inputs, image)
    preview = dry_run_deployment(bundle)
    encoded = json.dumps(preview, sort_keys=True)
    resource_names = serving_resource_names(
        bundle.spec.deployment_id,
        bundle.spec.generation,
        bundle.spec.engine.engine_id,
        workload_role="app",
    )

    forbidden_values = {
        PROVIDER_SECRET_SENTINEL,
        image.reference,
        bundle.spec.adapters[0].artifact_repo_id,
        bundle.spec.adapters[0].artifact_subfolder,
        bundle.spec.adapters[0].checkpoint_id,
        bundle.manifest.canonical_json(),
        placement.environment,
        placement.region,
        *tuple(getattr(resource_names, name) for name in resource_names.__slots__),
    }
    assert all(value not in encoded for value in forbidden_values)
    assert {
        "placement",
        "workspace_name",
        "environment",
        "region",
        "image_reference",
        "reference",
        "manifest",
        "adapters",
        "checkpoint_id",
        "artifact_repo_id",
        "artifact_subfolder",
        "resource_names",
    }.isdisjoint(preview)


@pytest.mark.parametrize("model_id", supported_models())
def test_dry_run_preview_has_exact_safe_plan_facts(model_id: str) -> None:
    spec, execution_inputs = _profile_spec_and_inputs(model_id)
    profile = get_profile(model_id)
    placement = placement_for(
        profile,
        "modal",
        workspace_name="workspace",
        environment="dev",
        region="us-east",
    )
    spec = replace(spec, provider="modal", placement=placement)
    bundle = resolve_deployment_bundle(_request(spec), execution_inputs, _image())

    preview = dry_run_deployment(bundle)

    assert preview == {
        "deployment_id": bundle.spec.deployment_id,
        "generation": bundle.spec.generation,
        "provider": "modal",
        "spec_id": bundle.spec.spec_id,
        "manifest_id": bundle.manifest.manifest_id,
        "engine_id": bundle.spec.engine.engine_id,
        "image_digest": bundle.image.digest,
        "served_model": spec.engine.served_model,
        "model_revision": spec.engine.model_revision,
        "tokenizer_model": spec.engine.tokenizer_model,
        "tokenizer_revision": spec.engine.tokenizer_revision,
        "runtime_family": spec.engine.runtime_family,
        "provider_gpu": profile.modal_gpu_request,
        "provider_gpu_count": profile.tensor_parallel_size,
        "max_model_len": profile.max_model_len,
        "max_num_seqs": profile.max_num_seqs,
        "max_num_batched_tokens": profile.max_num_batched_tokens,
        "max_loras": profile.max_loras,
        "max_cpu_loras": profile.max_cpu_loras,
        "max_lora_rank": profile.max_lora_rank,
        "modality": profile.modality,
        "image_limit": profile.image_limit,
        "adapter_count": len(bundle.spec.adapters),
        "adapter_capacity": spec.engine.adapter_capacity,
    }


def test_domain_module_import_does_not_load_provider_packages_or_lifecycle_modules() -> None:
    probe = r"""
import builtins
import sys

provider_roots = ("modal", "runpod", "runpod_flash")
lifecycle_prefixes = ("flash.serve.provisioning.modal",)
real_import = builtins.__import__
intercepted = []


def blocked(name):
    return (
        name in provider_roots
        or name.startswith(tuple(root + "." for root in provider_roots))
        or name.startswith(lifecycle_prefixes)
    )


def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if blocked(name):
        intercepted.append(name)
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded
from flash.server.domain.ops.serving_resources import dry_run_deployment, resolve_deployment_bundle

assert dry_run_deployment
assert resolve_deployment_bundle
assert intercepted == []
assert not any(blocked(name) for name in sys.modules)
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
