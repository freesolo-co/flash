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
from flash.serve.provisioning import ServingImage, serving_resource_names
from flash.server.domain.ops.serving_resources import dry_run_deployment, resolve_deployment_bundle
from tests.test_serve_app_manifest import IMAGE_DIGEST, _spec_and_inputs

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_SECRET_SENTINEL = "provider-secret-sentinel"
RUNTIME_SECRET_SENTINEL = "runtime-secret-sentinel"
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


def test_dry_run_preview_has_exact_keys_and_authoritative_values() -> None:
    spec, execution_inputs = _spec_and_inputs()
    bundle = resolve_deployment_bundle(_request(spec), execution_inputs, _image())

    preview = dry_run_deployment(bundle)

    assert set(preview) == {
        "deployment_id",
        "generation",
        "provider",
        "spec_id",
        "manifest_id",
        "engine_id",
        "image_digest",
        "adapter_count",
        "adapter_capacity",
    }
    assert preview == {
        "deployment_id": bundle.spec.deployment_id,
        "generation": bundle.spec.generation,
        "provider": bundle.spec.provider,
        "spec_id": bundle.spec.spec_id,
        "manifest_id": bundle.manifest.manifest_id,
        "engine_id": bundle.spec.engine.engine_id,
        "image_digest": bundle.image.digest,
        "adapter_count": len(bundle.spec.adapters),
        "adapter_capacity": bundle.spec.engine.adapter_capacity,
    }


def test_json_preview_excludes_nonallowlisted_bundle_data() -> None:
    spec, execution_inputs = _spec_and_inputs()
    placement = ModalPlacement(
        workspace_name=PROVIDER_SECRET_SENTINEL,
        environment="placement-environment-sentinel",
        gpu="placement-gpu-sentinel",
        region="placement-region-sentinel",
    )
    engine = replace(spec.engine, served_model=RUNTIME_SECRET_SENTINEL)
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
        RUNTIME_SECRET_SENTINEL,
        image.reference,
        bundle.spec.adapters[0].artifact_repo_id,
        bundle.spec.adapters[0].artifact_subfolder,
        bundle.spec.adapters[0].adapter_revision,
        bundle.manifest.canonical_json(),
        placement.environment,
        placement.gpu,
        placement.region,
        *tuple(getattr(resource_names, name) for name in resource_names.__slots__),
    }
    assert all(value not in encoded for value in forbidden_values)
    assert {
        "placement",
        "workspace_name",
        "environment",
        "gpu",
        "region",
        "image_reference",
        "reference",
        "manifest",
        "adapters",
        "adapter_revision",
        "artifact_repo_id",
        "artifact_subfolder",
        "resource_names",
    }.isdisjoint(preview)


def test_domain_module_import_does_not_load_provider_packages_or_lifecycle_modules() -> None:
    probe = r"""
import builtins
import sys

provider_roots = ("modal", "runpod", "runpod_flash")
lifecycle_prefixes = (
    "flash.serve.provisioning.modal",
    "flash.serve.provisioning.runpod",
)
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
