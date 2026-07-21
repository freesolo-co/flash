"""Sequential SFT worker entrypoint with one opt-in resident frozen base."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from flash.engine.worker.resident import ResidentBase, ResidentJobState
from flash.engine.worker_entrypoint import WORKER_FAILURE_LINE
from flash.spec import JobSpec

_RESIDENT_SPECS_ENV = "FLASH_RESIDENT_JOB_SPECS_JSON"


def _load_specs(paths: Sequence[str]) -> list[JobSpec]:
    payloads: list[Any] = []
    if paths:
        for raw_path in paths:
            with Path(raw_path).open() as file:
                payload = json.load(file)
            payloads.extend(payload if isinstance(payload, list) else [payload])
    else:
        raw = os.environ.get(_RESIDENT_SPECS_ENV, "")
        if not raw:
            raise RuntimeError(
                f"provide job-spec json paths or set {_RESIDENT_SPECS_ENV} to a json array"
            )
        payload = json.loads(raw)
        payloads.extend(payload if isinstance(payload, list) else [payload])
    if not payloads:
        raise ValueError("resident worker requires at least one job spec")
    return [JobSpec.from_dict(payload) for payload in payloads]


def _activate_job(worker: Any, job_spec: JobSpec) -> None:
    if job_spec.algorithm != "sft":
        raise ValueError("resident worker supports SFT jobs only")
    if job_spec.train.init_from_adapter:
        raise ValueError("resident SFT adapter continuation is not supported")

    worker.JOB_SPEC = job_spec
    worker.SEED = int(job_spec.seed)
    worker.RUN_ID = job_spec.run_id
    worker.RUN_MODE = "sft"
    worker.PHASE = "sft"
    worker.HF_REPO = job_spec.train.hf_repo
    worker.THINKING = bool(job_spec.thinking)
    worker.ACTIVE_ENV = None
    worker.ATTEMPT = 0
    worker._HB_LAST_UPLOAD = 0.0
    worker._HB_LAST_PROGRESS_TS = 0.0
    worker._HB_PROGRESS_SEQ = 0
    worker._HB_PROGRESS_UPLOADED_SEQ = 0
    worker._HB_LAST_COMMITTED_STEP = 0
    worker._HB_LAST_FORCED_UPLOAD = 0.0
    worker._HB_TERMINAL_ONLY = False


def _prepare_process(worker: Any) -> None:
    worker._force_fla_triton_gdn_on_sm100()
    worker._ensure_fla_fastpath_on_hopper()
    worker._neutralize_tilelang_cudart_stub()
    worker._restrict_fla_gdn_autotune_on_blackwell()
    worker.heartbeat("boot", gpu=worker.gpu_diagnostics(include_torch=False))
    worker.finalize_alloc_conf_for_sleep()
    worker.load_mega_cache()


def _run_sft_job(worker: Any, resident: ResidentBase) -> ResidentJobState:
    result = worker.run_sft(resident_base=resident)
    if not isinstance(result, ResidentJobState):
        raise RuntimeError("resident SFT handler did not return cleanup state")
    return result


# TODO(resident-followup): add provider idempotency and wall-deadline preflight before wiring.
def run_jobs(
    job_specs: Sequence[JobSpec],
    *,
    resident: ResidentBase | None = None,
    run_job: Callable[[Any, ResidentBase], ResidentJobState] | None = None,
    prepare_process: bool = True,
) -> int:
    """Run SFT specs sequentially, abandoning the process after the first error."""

    try:
        from flash.engine import worker

        if not job_specs:
            raise ValueError("resident worker requires at least one job spec")
        holder = resident or ResidentBase()
        handler = run_job or _run_sft_job
        if prepare_process:
            _activate_job(worker, job_specs[0])
            _prepare_process(worker)
        for index, job_spec in enumerate(job_specs):
            if not prepare_process or index:
                _activate_job(worker, job_spec)
            state = handler(worker, holder)
            holder.reset_after_job(
                state.model,
                trainer=state.trainer,
                dataset=state.dataset,
            )
            state.model = None
            state.trainer = None
            state.dataset = None
            del state
            gc.collect()
            worker.wandb_finish(exit_code=0)
        return 0
    except BaseException:
        print(WORKER_FAILURE_LINE, file=sys.stderr, flush=True)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="run sequential SFT job specs with one resident frozen base"
    )
    parser.add_argument("job_specs", nargs="*", help="job-spec json files or json-array files")
    args = parser.parse_args(argv)
    try:
        specs = _load_specs(args.job_specs)
    except BaseException:
        print(WORKER_FAILURE_LINE, file=sys.stderr, flush=True)
        return 1
    return run_jobs(specs)


if __name__ == "__main__":
    raise SystemExit(main())
