"""Start a local Flash control plane that tolerates HF's daily repo-creation cap.

Why this exists: Flash's launcher (`providers/runpod/train.py:upload_code`) creates a
per-run HF *dataset* repo for code upload + checkpoint streaming. HuggingFace caps repo
*creation* at 300/day per user. When that cap is hit, every new run fails at submit with
a 429 from `https://huggingface.co/api/repos/create` — even for repos that already exist
(the create endpoint is rate-limited regardless of `exist_ok`).

For the benchmark we work around it by REUSING pre-existing (disposable) artifact repos:
this shim wraps `HfApi.create_repo` / `update_repo_settings` so a create-429 is swallowed
when the target repo already exists (commits via `upload_folder` are NOT daily-capped, so
code upload + checkpoint streaming still work). Flash core is left untouched.

Run it with the control plane's provider credentials in the environment, e.g.:

    FLASH_PLANE_PORT=8085 \
    RUNPOD_API_KEY=... HF_TOKEN=... PRIME_API_KEY=... \
    FREESOLO_INTERNAL_KEY=... FREESOLO_BASE_URL=... \
    /path/to/flash-benchmark/.venv/bin/python benchmark/run_flash_plane.py
"""
from __future__ import annotations

import os

from huggingface_hub import HfApi

_orig_create = HfApi.create_repo
_orig_update = HfApi.update_repo_settings


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _safe_create_repo(self, repo_id, *args, **kwargs):
    try:
        return _orig_create(self, repo_id, *args, **kwargs)
    except Exception as exc:
        if _is_rate_limit(exc):
            repo_type = kwargs.get("repo_type", "model")
            try:
                if self.repo_exists(repo_id, repo_type=repo_type):
                    print(f"[plane-shim] create_repo 429 but {repo_id} exists — reusing it")
                    return None
            except Exception:
                pass
        raise


def _safe_update_repo_settings(self, repo_id=None, *args, **kwargs):
    try:
        return _orig_update(self, repo_id, *args, **kwargs)
    except Exception as exc:
        if not _is_rate_limit(exc):
            raise
        # A swallowed 429 on update_repo_settings is NOT safe to ignore when it carries the
        # intended visibility: a reused artifact repo (run code / adapters / metrics) could
        # be left PUBLIC. Only swallow if the repo ALREADY matches the requested `private`;
        # otherwise re-raise so the caller can't proceed with the wrong visibility.
        want_private = kwargs.get("private")
        if want_private is None:
            print("[plane-shim] update_repo_settings 429 (no visibility change requested) "
                  "— leaving existing settings as-is")
            return None
        repo_type = kwargs.get("repo_type", "model")
        try:
            info = self.repo_info(repo_id, repo_type=repo_type)
            if bool(getattr(info, "private", None)) == bool(want_private):
                print(f"[plane-shim] update_repo_settings 429 but {repo_id} already "
                      f"private={want_private} — safe to continue")
                return None
        except Exception:
            pass
        # Could not confirm the intended visibility — fail loudly rather than risk exposure.
        raise RuntimeError(
            f"update_repo_settings rate-limited and could not ensure private={want_private} "
            f"on {repo_id}; refusing to continue with unverified visibility"
        ) from exc


HfApi.create_repo = _safe_create_repo
HfApi.update_repo_settings = _safe_update_repo_settings


def main() -> None:
    from flash.server.app import run_server

    port = int(os.environ.get("FLASH_PLANE_PORT", "8085"))
    print(f"[plane-shim] starting Flash control plane on 127.0.0.1:{port} "
          f"(HF create-cap tolerant)")
    run_server(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
