"""Lambda placement identities; model and autoscaling settings remain shared."""

import re

from flash.serving.app.inference_provider import policy_for
from flash.serving.src.engine.model_config import SMALL_QWEN_MODELS, base_models

API_URL = "https://cloud.lambda.com/api/v1"
APP_NAME = "freesolo-lambda-serving-control"
LEDGER_NAME = "freesolo-lambda-serving-leases"
SSH_KEY_NAME = "freesolo-serving"
SERVER_NAME = "lambda-worker.internal"
PORT = 8443
HEARTBEAT_SECONDS = 30
# An orphan cannot be reaped before the full startup + request budget expires.
LEASE_SECONDS = 45 * 60 + 15 * 60 + 300
# Published by publish-serving-image.yml from main d1397691; its /app/uv.lock is byte-identical
# to the repository lock since #1766 (uv lock revision 3, dnspython 2.9.2, opentelemetry 1.45.2).
# Workers verify that lock against their source revision at boot, so re-pin whenever uv.lock
# changes: take the digest the workflow reports for a commit with the new lock.
RUNTIME_IMAGE = (
    "ghcr.io/freesolo-co/freesolo-flash-serve@"
    "sha256:1b4bd9229621eaa0aeef8415608a2683c6eed5cf6703050973e65239fea2bc4c"
)


# Labels describe qualified shapes, not guessed provider instance-type IDs.
GPU_MEMORY_GIB = {
    "A10": 22,
    "A6000": 44,
    "A100-40GB": 35,
    "A100-80GB": 75,
    "H100": 75,
    "H200": 130,
    "B200": 170,
}
CATALOG_MEMORY_GB = {
    "A10": 24,
    "A6000": 48,
    "A100-40GB": 40,
    "A100-80GB": 80,
    "H100": 80,
    "H200": 141,
    "B200": 180,
}


def qualified_placements(model):
    if model not in base_models():
        raise ValueError("Lambda placement is only defined for the hosted Qwen catalog")
    if model == "Qwen/Qwen3.6-35B-A3B":
        return ("H200", "B200")
    # Every x86 single GPU that runs the model: Ampere (A10/A6000/A100) uses the
    # weight-only FP8 Marlin profile, Hopper/Blackwell the native FP8 one. The fleet
    # rents the cheapest shape with capacity. Turing (RTX 6000) has no FP8 kernels
    # and GH200 is arm64, so neither is listed.
    if model in SMALL_QWEN_MODELS:
        return ("A10", "A6000", "A100-40GB", "A100-80GB", "H100", "H200", "B200")
    if model in {"Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base"}:
        return ("A6000", "A100-40GB", "A100-80GB", "H100", "H200", "B200")
    return ("A100-80GB", "H100", "H200", "B200")


def placements(model):
    """Lambda shapes for a model: every qualified GPU unless the policy narrows it.

    Lambda is the only GPU provider for hosted Qwen, so any qualified shape with
    capacity is rented (cheapest first) instead of waiting on one GPU type.
    """
    qualified = qualified_placements(model)
    requested = policy_for(model).lambda_gpus
    if any(gpu not in qualified for gpu in requested):
        raise ValueError(f"Unqualified Lambda GPU for {model}: {requested}")
    return requested or qualified


def gpu_matches(gpu, description):
    family = gpu.split("-")[0]
    return re.search(rf"\b{family}\b", description, re.IGNORECASE) is not None


def validate_gpu(model, name, memory_bytes, count):
    for gpu in sorted(placements(model), key=GPU_MEMORY_GIB.get, reverse=True):
        if count == 1 and gpu_matches(gpu, name) and memory_bytes >= GPU_MEMORY_GIB[gpu] * 2**30:
            return gpu
    raise RuntimeError(f"Lambda worker received an unapproved GPU shape for {model}: {name}")


class NoCapacityError(RuntimeError):
    """No approved capacity is available; no VM was created."""


def select_region(catalog, model, *, excluded=()):
    allowed = placements(model)
    candidates = []
    for instance_type, offer in catalog.items():
        details = offer.get("instance_type", {})
        if details.get("specs", {}).get("gpus") != 1 or details.get("architecture") != "x86_64":
            continue
        description = details.get("gpu_description", "")
        if not any(
            gpu_matches(gpu, description)
            and re.search(rf"\b{CATALOG_MEMORY_GB[gpu]}\s*GB\b", description, re.IGNORECASE)
            for gpu in allowed
        ):
            continue
        price = details.get("price_cents_per_hour")
        if not isinstance(price, int) or price <= 0:
            continue
        candidates.extend(
            (price, instance_type, region["name"])
            for region in offer.get("regions_with_capacity_available", [])
            if region["name"].startswith("us-") and (instance_type, region["name"]) not in excluded
        )
    if not candidates:
        raise NoCapacityError(
            f"Lambda has no US capacity for approved shapes {allowed}; no hardware fallback"
        )
    _, instance_type, region = min(candidates)
    return instance_type, region
