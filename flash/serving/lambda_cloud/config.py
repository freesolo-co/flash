"""Lambda placement identities; model and autoscaling settings remain shared."""

from flash.serving.src.engine.model_config import base_models, gpu_for

API_URL = "https://cloud.lambda.com/api/v1"
APP_NAME = "freesolo-lambda-serving-control"
LEDGER_NAME = "freesolo-lambda-serving-leases"
SSH_KEY_NAME = "freesolo-serving"
SERVER_NAME = "lambda-worker.internal"
PORT = 8443
HEARTBEAT_SECONDS = 30
# An orphan cannot be reaped before the full startup + request budget expires.
LEASE_SECONDS = 45 * 60 + 15 * 60 + 300
RUNTIME_IMAGE = (
    "ghcr.io/freesolo-co/freesolo-flash-serve@"
    "sha256:8fde9eec7ca01a2f0173be467beb753c7c107ced644f44a5f38fb61524c8060c"
)


def placement(model):
    if model not in base_models():
        raise ValueError("Lambda placement is only defined for the hosted Qwen catalog")
    return (
        ("gpu_1x_b200_sxm6", "B200") if gpu_for(model) == "H200" else ("gpu_1x_h100_sxm5", "H100")
    )


def select_region(catalog, model):
    instance_type, _ = placement(model)
    offer = catalog.get(instance_type, {})
    regions = sorted(
        region["name"]
        for region in offer.get("regions_with_capacity_available", [])
        if region["name"].startswith("us-")
    )
    if not regions:
        raise RuntimeError(f"Lambda has no US capacity for {instance_type}; no hardware fallback")
    return instance_type, regions[0]
