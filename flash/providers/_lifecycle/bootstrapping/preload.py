"""Preload model weights into a mounted instance-worker cache."""

from __future__ import annotations

import os

if __package__:
    from flash.providers._lifecycle.bootstrapping.secrets import _payload_secrets, _safe_detail
else:
    from bootstrap_secrets import _payload_secrets, _safe_detail  # type: ignore[no-redef]


def run_preload(payload: dict) -> dict:
    """Download models into the bind-mounted weight cache. Fails if cache isn't mounted to avoid warming ephemeral disk."""
    env = payload.get("env") or {}
    cache_dir = env.get("FLASH_WEIGHT_CACHE_DIR") or ""
    token = env.get("HF_TOKEN")
    mount = os.path.dirname(os.path.dirname(cache_dir.rstrip("/"))) if cache_dir else ""
    if not cache_dir or not mount or not os.path.isdir(mount):
        return {
            "preloaded": [],
            "already_cached": [],
            "failed": {},
            "error": f"weight-cache not mounted (FLASH_WEIGHT_CACHE_DIR={cache_dir!r}); refusing to warm ephemeral disk",
        }
    # Sentinel written by cloud-init only onto a real mount; absent sentinel means Docker silently
    # auto-created an empty host dir (isdir passes) — we'd warm ephemeral disk. Must check.
    if payload.get("cache_mount_marker"):
        marker = os.path.join(mount, payload["cache_mount_marker"])
        if not os.path.exists(marker):
            return {
                "preloaded": [],
                "already_cached": [],
                "failed": {},
                "error": (
                    f"weight-cache NFS filesystem not mounted (no sentinel at {marker}); "
                    "refusing to warm ephemeral disk"
                ),
            }
    from huggingface_hub import snapshot_download

    ignore_patterns = ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]
    done, already, failed = [], [], {}
    for repo_id in payload.get("models") or []:
        try:
            # Probe with local_files_only first (HF's own resolution, not a dir-name guess).
            try:
                snapshot_download(
                    repo_id=repo_id,
                    token=token,
                    cache_dir=cache_dir,
                    ignore_patterns=ignore_patterns,
                    local_files_only=True,
                )
                already.append(repo_id)
                print(f"preload: {repo_id} -> {cache_dir} (cached)", flush=True)
                continue
            except Exception:
                pass
            snapshot_download(
                repo_id=repo_id, token=token, cache_dir=cache_dir, ignore_patterns=ignore_patterns
            )
            done.append(repo_id)
            print(f"preload: {repo_id} -> {cache_dir} (downloaded)", flush=True)
        except Exception as exc:
            failed[repo_id] = _safe_detail(exc, secrets=_payload_secrets(payload))
            print(f"preload FAILED {repo_id}: {failed[repo_id]}", flush=True)
    return {"preloaded": done, "already_cached": already, "failed": failed}
