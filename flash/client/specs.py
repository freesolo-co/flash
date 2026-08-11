"""Turn a locally validated JobSpec into the payload sent to the control plane."""

from __future__ import annotations

from flash.core.spec import JobSpec, require_project_id


def spec_payload(
    spec: JobSpec, authored_train_keys: frozenset[str] | set[str] | None = None
) -> dict:
    out = spec.to_dict()
    out["project"] = require_project_id(spec.project)
    # an omitted gpu.count must reach the server as omitted. `to_dict()` keeps the placeholder
    # `count: 1` for preparation-digest stability and strips the provenance marker, and the server
    # re-derives "auto" from the KEY'S ABSENCE (`spec_from_dict`) -- so sending the placeholder made
    # the server read an auto-sized run as an authored one-card pin and reject it at the pinned-count
    # preflight before a run existed. dropping the key here restores the author's intent over the
    # wire without touching the digest, which is computed server-side from the reparsed spec.
    if spec.gpu_count_auto and out.get("gpu", {}).get("count") == 1:
        out["gpu"] = {key: value for key, value in out["gpu"].items() if key != "count"}
    # [environment] pip is platform-managed and no longer travels in the payload: both provider
    # submit paths already resolve it from worker_pip_for_env when the spec carries none, so
    # sending it here only duplicated that constant, and the server's parser now rejects it as an
    # unauthorable key.
    if authored_train_keys is not None:
        out["train"] = {
            key: value for key, value in out["train"].items() if key in authored_train_keys
        }
    return out
