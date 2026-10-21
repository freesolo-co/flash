"""Complete Modal attempt identities for runner tests."""

import hashlib


def modal_remote(instance_id="sb-worker", attempt=0, started_ts=1.0, **extra):
    instance_id = str(extra.pop("endpoint_id", instance_id))
    if not instance_id.startswith("sb-"):
        instance_id = f"sb-{instance_id}"
    for key in (
        "job_id",
        "key_fingerprint",
        "instance_type",
        "region",
        "offer_id",
        "machine_id",
        "label",
    ):
        extra.pop(key, None)
    return {
        "provider": "modal",
        "instance_id": instance_id,
        "attempt": attempt,
        "started_ts": started_ts,
        "gpu": extra.get("allocated_gpu", "L4"),
        "gpu_count": extra.get("allocated_gpu_count", 1),
        "hourly_usd": 1.0,
        "name": extra.pop("endpoint_name", f"flash-test-s0-a{attempt}"),
        "run_id": "flash-test",
        "phase": "sft",
        "hf_repo": "org/artifacts",
        "scope": hashlib.sha256(b"test-token-id:main").hexdigest(),
        **extra,
    }
