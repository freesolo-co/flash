"""Profile-only SFT worker that exits before model loading or cuda initialization."""

from __future__ import annotations

import time
from dataclasses import replace

from flash.engine.accounting import RunMetrics
from flash.engine.sft_workload import prepare_sft_workload
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.rng import seed_host_rngs
from flash.workload_profile import SFT_PROFILE_KIND, sft_profile_input_digest


def run_sft_profile() -> None:
    spec = _w.JOB_SPEC
    if spec is None or spec.workload_profile_kind != SFT_PROFILE_KIND:
        raise RuntimeError("sft profile worker requires an internal profile job spec")
    from flash import __version__

    expected = sft_profile_input_digest(
        spec,
        tokenizer_revision=spec.model_revision,
        producer_version=__version__,
    )
    if expected != spec.workload_profile_input_digest:
        raise ValueError("sft workload profile input digest does not match the worker spec")

    started_at = time.time()
    _w.heartbeat("profile_start")
    # load the environment BEFORE seeding, exactly as the training worker does. both sides reseed
    # python and numpy, so what has to match is not that they seed but WHERE: anything that draws on
    # the global generators during env load (the loader's retry jitter, a user module consuming
    # random/numpy at import) lands on the opposite side of the seed otherwise, the two workers build
    # different rows, and the drift guard rejects a profile the user already paid for.
    env = _w.require_active_env()
    # host generators only: environment code may consume them while building its dataset, so the
    # profile must reach the same rows training will. seeding torch would import it, and a job that
    # is quoted as cpu-only has no business pulling in the model stack.
    seed_host_rngs(spec.seed)
    prepared = prepare_sft_workload(
        spec,
        env,
        tokenizer_loader=lambda model_id, revision: _w.load_tokenizer(
            model_id,
            revision=revision,
        ),
        producer_version=__version__,
        allow_packing=True,
    )
    finished_at = time.time()
    # only a producer stamps provenance. prepare_sft_workload leaves created_at at 0.0 so the
    # training worker's recomputation stays comparable to this artifact.
    profile = replace(prepared.profile, created_at=finished_at)
    _w._finalize(
        RunMetrics(
            phase="profile",
            seed=spec.seed,
            model_id=spec.model,
            wall_seconds=finished_at - started_at,
            workload_profile=profile.to_dict(),
            notes={
                "profile_kind": SFT_PROFILE_KIND,
                "packing_mode": profile.packing_mode,
                "sample_policy": profile.sample_policy,
            },
        ),
        heartbeat_fields={"profile_kind": SFT_PROFILE_KIND},
    )
