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
    """return a credential-free identity, runtime, and modal-plan preview."""

    engine = bundle.spec.engine
    placement = bundle.spec.placement
    return {
        "deployment_id": bundle.spec.deployment_id,
        "generation": bundle.spec.generation,
        "provider": bundle.spec.provider,
        "spec_id": bundle.spec.spec_id,
        "manifest_id": bundle.manifest.manifest_id,
        "engine_id": engine.engine_id,
        "image_digest": bundle.image.digest,
        "served_model": engine.served_model,
        "model_revision": engine.model_revision,
        "tokenizer_model": engine.tokenizer_model,
        "tokenizer_revision": engine.tokenizer_revision,
        "runtime_family": engine.runtime_family,
        "provider_gpu": placement.gpu,
        "provider_gpu_count": placement.gpu_count,
        "max_model_len": engine.max_model_len,
        "max_num_seqs": engine.max_num_seqs,
        "max_num_batched_tokens": engine.max_num_batched_tokens,
        "max_loras": engine.max_loras,
        "max_cpu_loras": engine.max_cpu_loras,
        "max_lora_rank": engine.max_lora_rank,
        "modality": engine.modality,
        "image_limit": engine.image_limit,
        "adapter_count": len(bundle.spec.adapters),
        "adapter_capacity": engine.adapter_capacity,
    }
