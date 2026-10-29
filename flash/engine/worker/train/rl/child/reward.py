"""Packaged reward callback: posts each completion to the Flash worker."""

import os

from flash.engine.worker.train.rl.child.multiturn import post_json


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    _URL = os.environ.get("FLASH_VERL_REWARD_URL", "")
    idx = (extra_info or {}).get("index")
    if idx is None:
        raise RuntimeError("flash reward bridge received no example index")
    if not _URL:
        raise RuntimeError("flash reward bridge url is not configured")
    if isinstance(idx, bool) or getattr(getattr(idx, "dtype", None), "kind", None) == "b":
        raise RuntimeError(f"flash reward bridge received an invalid example index: {idx!r}")
    try:
        exact_idx = int(idx)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"flash reward bridge received an invalid example index: {idx!r}"
        ) from exc
    if exact_idx != idx:
        raise RuntimeError(f"flash reward bridge received an invalid example index: {idx!r}")
    idx = exact_idx
    identity = (extra_info or {}).get("flash_rollout_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("flash reward bridge received no rollout identity")
    request = {"index": idx, "solution_str": solution_str or "", "identity": dict(identity)}
    if (extra_info or {}).get("flash_rollout_context") is not None:
        request["rollout_context"] = extra_info["flash_rollout_context"]
    payload = post_json(
        _URL,
        "/score",
        request,
        error_style="reward",
    )
    try:
        return float(payload["score"])
    except Exception as exc:
        raise RuntimeError(f"flash reward bridge returned an invalid response: {exc}") from exc
