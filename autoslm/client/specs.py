"""Turn a locally validated JobSpec into the payload sent to the control plane.

The one piece of client-local state a run needs is the pip requirements for installed
verifiers / Prime Hub environments (recorded in ``~/.autoslm/envs.json`` by
``slm env install``). The server has no access to that manifest, so the client resolves
it here and ships it inside the spec (``environment.pip``); a value already present in
the config (the documented escape hatch) wins.
"""

from __future__ import annotations

from autoslm.worker_spec import JobSpec


def spec_payload(spec: JobSpec) -> dict:
    out = spec.to_dict()
    if not spec.environment.pip:
        from autoslm.envs.registry import worker_pip_for_env

        pip = worker_pip_for_env(spec.environment.id, spec.environment.params)
        if pip:
            out["environment"]["pip"] = pip
    return out
