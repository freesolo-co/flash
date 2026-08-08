"""Turn a locally validated JobSpec into the payload sent to the control plane."""

from __future__ import annotations

from flash.spec import JobSpec, require_project_id


def spec_payload(
    spec: JobSpec, authored_train_keys: frozenset[str] | set[str] | None = None
) -> dict:
    out = spec.to_dict()
    out["project"] = require_project_id(spec.project)
    # [environment] pip is platform-managed and no longer travels in the payload: both provider
    # submit paths already resolve it from worker_pip_for_env when the spec carries none, so
    # sending it here only duplicated that constant, and the server's parser now rejects it as an
    # unauthorable key.
    if authored_train_keys is not None:
        out["train"] = {
            key: value for key, value in out["train"].items() if key in authored_train_keys
        }
    return out
