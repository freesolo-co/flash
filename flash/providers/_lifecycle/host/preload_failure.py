"""publish preload-only terminal failure diagnostics from the rented host."""

import io
import json
import math
import sys
import time


def publish_preload_failure(payload: dict, detail: str, upload=None) -> bool:
    """upload one preload failure artifact, or no-op outside preload mode."""
    if payload.get("mode") != "preload":
        return False
    run_id = payload.get("run_id")
    attempt = payload.get("attempt")
    fence = payload.get("fence")
    now = time.time()
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("invalid run id")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("invalid attempt")
    if isinstance(fence, bool) or not isinstance(fence, int) or fence <= 0:
        raise ValueError("invalid fence")
    if not math.isfinite(now) or now <= 0:
        raise ValueError("invalid clock")
    artifact = {
        "run_id": run_id,
        "attempt": attempt,
        "fence": fence,
        "error": detail[:1000],
        "ts": now,
    }
    if upload is not None:
        path = "/tmp/preload_failure.json"
        with open(path, "w") as handle:
            json.dump(artifact, handle)
        upload(payload, path, "preload_failure.json")
    else:
        from huggingface_hub import HfApi

        HfApi(token=(payload.get("env") or {}).get("HF_TOKEN")).upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(artifact, sort_keys=True).encode()),
            path_in_repo=payload["hf_prefix"] + "/preload_failure.json",
            repo_id=payload["hf_repo"],
            repo_type="dataset",
        )
    return True


def main() -> None:
    """best-effort upload of one preload failure artifact with no training authority."""
    try:
        with open("/opt/flash/payload.json") as payload_file:
            payload = json.load(payload_file)
        detail = str(sys.argv[1] if len(sys.argv) > 1 else "preload host failed")
        publish_preload_failure(payload, detail)
    except Exception:
        pass


if __name__ == "__main__":
    main()
