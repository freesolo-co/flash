"""Turn a locally validated JobSpec into the payload sent to the control plane."""

from __future__ import annotations

from flash.spec import JobSpec


def spec_payload(
    spec: JobSpec, authored_train_keys: frozenset[str] | set[str] | None = None
) -> dict:
    out = spec.to_dict()
    if not spec.environment.pip:
        from flash.envs.registry import worker_pip_for_env

        pip = worker_pip_for_env(spec.environment.id)
        if pip:
            out["environment"]["pip"] = pip
    if authored_train_keys is not None:
        out["train"] = {
            key: value for key, value in out["train"].items() if key in authored_train_keys
        }
    return out
