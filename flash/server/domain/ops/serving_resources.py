"""credential-free serving deployment composition and preview."""

from flash.serve.app import ExecutionInputs, build_serving_manifest
from flash.serve.control import DeploymentRequest, plan_deployment
from flash.serve.provisioning import DeploymentBundle, ServingImage


def resolve_deployment_bundle(
    request: DeploymentRequest,
    execution_inputs: ExecutionInputs,
    image: ServingImage,
) -> DeploymentBundle:
    """compose one validated deployment bundle from existing public boundaries."""

    spec = plan_deployment(request)
    manifest = build_serving_manifest(spec, execution_inputs)
    return DeploymentBundle(spec, manifest, image)


def dry_run_deployment(bundle: DeploymentBundle) -> dict[str, object]:
    """return a credential-free identity and capacity preview."""

    return {
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
