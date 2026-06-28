"""Turn a locally validated JobSpec into the payload sent to the control plane."""

from __future__ import annotations

from flash.spec import JobSpec


def spec_payload(spec: JobSpec) -> dict:
    out = spec.to_dict()
    if not spec.environment.pip:
        from flash.envs.registry import worker_pip_for_env

        pip = worker_pip_for_env(spec.environment.id)
        if pip:
            out["environment"]["pip"] = pip
    return out
