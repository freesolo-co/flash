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
    if spec.authored_gpu_count is None:
        out["gpu"] = {key: value for key, value in out["gpu"].items() if key != "count"}
    # [environment] pip DOES travel: it is the author's own scorer dependencies, which only they can
    # declare, and the submit paths append it to worker_pip_for_env rather than replacing it. What
    # travels is the extra requirements, not the worker's own.
    if authored_train_keys is not None:
        out["train"] = {
            key: value for key, value in out["train"].items() if key in authored_train_keys
        }
    return out
