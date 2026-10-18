"""Complete Modal attempt identities for runner tests."""

import hashlib


def modal_remote(instance_id="sb-worker", attempt=0, started_ts=1.0, **extra):
    instance_id = str(instance_id)
    if not instance_id.startswith("sb-"):
        instance_id = f"sb-{instance_id}"
    return {
        "provider": "modal",
        "instance_id": instance_id,
        "attempt": attempt,
        "started_ts": started_ts,
        "gpu": "L4",
        "gpu_count": 1,
        "hourly_usd": 1.0,
        "name": f"flash-test-s0-a{attempt}",
        "run_id": "flash-test",
        "phase": "sft",
        "hf_repo": "org/artifacts",
        "scope": hashlib.sha256(b"test-token-id:main").hexdigest(),
        **extra,
    }
