"""Failure accounting and resume staging for the OPD orchestrator.

When the verl child dies, the reason is rarely in its exit code: the informative record is a
fallback file the child wrote before exiting (a teacher-scoring failure, an abandoned no-signal
batch, a mutation-callback error). These helpers read those records back, reconcile them against
what the parent observed, and turn the pair into one accurate raise. The resume-staging helpers sit
alongside them because a retry's contract is written from the same accounting.

Split out of `flash.engine.worker.train.entry.opd_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.perf as _worker_perf
import flash.engine.worker.runtime.state as _worker_state
from flash.engine.support.verl_policy import FsdpGeneration
from flash.engine.worker.train.entry.sft_train import (
    SHIM_FRAGMENT_FAILED_EXIT_CODE,
    _export_checkpoint_adapter,
    _VerlCheckpointWatcher,
)
from flash.engine.worker.train.opd.bridging.bridge import _TeacherAlignmentBridge
from flash.engine.worker.train.opd.child.bridge import _render_rollout_failure
from flash.engine.worker.train.opd.orchestration.protocol import (
    PERMANENT_TEACHER_EXIT,
    TRANSIENT_TEACHER_EXIT,
)
from flash.engine.worker.verl.checkpoints import (
    inspect_resume_checkpoint,
    resume_topology_matches,
    resume_upload_unavailable,
)
from flash.teacher.limits import OPD_NO_SIGNAL_ATTEMPTS
from flash.teacher.retry_contract import (
    OPD_RESUME_STATE_VERSION,
    validate_opd_resume_state_metadata,
)

# the verl bridge stores the empty-alignment skip under its own internal key,
# which the resume state also reads. trl publishes the same condition as
# alignment_empty, so translate it at the metadata boundary only.
_CANONICAL_SKIP_REASONS = {"empty_alignment": "alignment_empty"}


@dataclass(frozen=True)
class _TruncationWindow:
    truncated_rollouts: int
    samples_seen: int
    no_signal_skipped_steps: int
    max_completion: int

    def __post_init__(self) -> None:
        # in-flight bridge counters can expose truncations before samples_seen catches up. clamp at
        # the typed boundary so every consumer gets a coherent fraction.
        object.__setattr__(
            self,
            "truncated_rollouts",
            min(self.samples_seen, self.truncated_rollouts),
        )

    @property
    def indicates_completion_cap(self) -> bool:
        # no-signal batches can also come from teacher failures or empty alignments, so only a strict
        # majority justifies naming the cap instead of the generic child status.
        return (
            self.no_signal_skipped_steps > 0
            and self.samples_seen > 0
            and self.truncated_rollouts * 2 > self.samples_seen
        )


def _canonical_skip_reasons(skip_counts: dict) -> dict:
    canonical: dict[str, int] = {}
    for reason, count in skip_counts.items():
        count = int(count)
        # trl records skip reasons in a counter, so only reasons that actually
        # occurred appear. the verl snapshot always injects empty_alignment.
        if count <= 0:
            continue
        name = _CANONICAL_SKIP_REASONS.get(reason, reason)
        canonical[name] = canonical.get(name, 0) + count
    return dict(sorted(canonical.items()))


def _failure_accounting_metadata(accounting: dict) -> dict:
    return {
        "teacher_transient_failures": int(accounting["teacher_transient"]),
        "teacher_errors": int(accounting["teacher_error"]),
        "no_signal_resamples": int(accounting["no_signal_resamples"]),
        "no_signal_skipped_steps": int(accounting["no_signal_skipped_steps"]),
        "skip_reasons": _canonical_skip_reasons(accounting["skip_counts"]),
    }


def _read_failure_fallback_records(base_path: str) -> list[tuple[str, str]]:
    if not base_path:
        return []
    base = Path(base_path)
    failures: list[tuple[str, str]] = []
    for path in sorted(base.parent.glob(f"{base.name}.*.json")):
        try:
            with path.open(encoding="utf-8") as file:
                encoded = file.read(8193)
            if len(encoded) > 8192:
                continue
            record = json.loads(encoded)
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        classification = record.get("classification")
        message = record.get("message")
        if classification not in {"permanent", "transient"}:
            continue
        if not isinstance(message, str) or not message.strip():
            continue
        failures.append((classification, message.strip()))
    return failures


def _read_classified_failure_fallback(base_path: str) -> tuple[str, str] | None:
    failures = _read_failure_fallback_records(base_path)
    for classification in ("permanent", "transient"):
        for failure_classification, message in failures:
            if failure_classification == classification:
                return classification, message
    return None


def _reconcile_score_delivery_failure(
    bridge: _TeacherAlignmentBridge,
    failure: tuple[str, str] | None,
) -> tuple[str, str] | None:
    if failure is None or bridge.teacher_failure is not None:
        return None
    if bridge._promote_pending_teacher_failure():
        return None
    return failure


def _reconcile_no_signal_notification_failure(
    bridge: _TeacherAlignmentBridge,
    failures: tuple[tuple[str, str] | None, ...],
) -> tuple[str, str] | None:
    if bridge.teacher_failure is not None:
        return None
    selected = None
    for failure in failures:
        if failure is None:
            continue
        if failure[0] == "permanent" or selected is None:
            selected = failure
    if (
        selected is not None
        and selected[0] == "transient"
        and bridge._promote_pending_teacher_failure()
    ):
        return None
    return selected


def _raise_verl_failure(
    return_code: int,
    teacher_failure: tuple[str, str] | None,
    mutation_failure: tuple[str, str] | None = None,
    cycle_commit_failure: tuple[str, str] | None = None,
    no_signal_failure: tuple[str, str] | None = None,
    score_delivery_failure: tuple[str, str] | None = None,
    *,
    rollout_failure: dict[str, str] | None = None,
    truncation_window: _TruncationWindow | None = None,
) -> None:
    if return_code == 0:
        return
    if mutation_failure is not None:
        classification, message = mutation_failure
        if classification == "transient":
            raise _worker_perf.RetriableInfraError(f"optimizer marker failure: {message}")
        raise RuntimeError(f"permanent optimizer marker failure: {message}")
    if cycle_commit_failure is not None:
        classification, message = cycle_commit_failure
        if classification == "transient":
            raise _worker_perf.RetriableInfraError(
                f"pre-update cycle commitment failure: {message}"
            )
        raise RuntimeError(f"permanent pre-update cycle commitment failure: {message}")
    if no_signal_failure is not None:
        classification, message = no_signal_failure
        if classification == "transient":
            raise _worker_perf.RetriableInfraError(
                f"transient no-signal notification failure: {message}"
            )
        raise RuntimeError(f"permanent no-signal notification failure: {message}")
    if score_delivery_failure is not None:
        classification, message = score_delivery_failure
        if classification == "transient":
            raise _worker_perf.RetriableInfraError(
                f"transient teacher score delivery failure: {message}"
            )
        raise RuntimeError(f"permanent teacher score delivery failure: {message}")
    if teacher_failure is not None:
        classification, message = teacher_failure
        if classification == "transient":
            raise _worker_perf.RetriableInfraError(
                f"transient teacher failure after bounded retries: {message}"
            )
        raise RuntimeError(f"permanent teacher failure: {message}")
    if rollout_failure is not None:
        detail = _render_rollout_failure(rollout_failure)
        if rollout_failure["classification"] == "transient":
            raise _worker_perf.RetriableInfraError(
                f"transient multi-turn OPD rollout failure: {detail}"
            )
        raise RuntimeError(f"permanent multi-turn OPD rollout failure: {detail}")
    if return_code == TRANSIENT_TEACHER_EXIT:
        raise _worker_perf.RetriableInfraError("transient teacher bridge failure")
    if return_code == PERMANENT_TEACHER_EXIT:
        raise RuntimeError("permanent teacher bridge failure")
    if return_code == SHIM_FRAGMENT_FAILED_EXIT_CODE:
        # permanent, not retriable infra: the same interpreter fails the same fragment on retry.
        raise RuntimeError(
            f"verl OPD subprocess exited with status {return_code}: a required flash runtime "
            "patch failed to apply in the child interpreter (its traceback names the fragment in "
            "the flash log). "
            "the verl/transformers stack at the child python is incompatible with this flash "
            "version; rebuild the worker image or fix FLASH_VERL_PYTHON rather than retrying."
        )
    if truncation_window is not None and truncation_window.indicates_completion_cap:
        raise RuntimeError(
            f"verl OPD subprocess exited with status {return_code}: flash OPD produced no "
            f"aligned teacher signal after {OPD_NO_SIGNAL_ATTEMPTS} rollout attempts; "
            f"{truncation_window.truncated_rollouts}/{truncation_window.samples_seen} rollouts were "
            f"truncated at the configured max_completion_tokens={int(truncation_window.max_completion)}, "
            "so the completion cap is likely too small"
        )
    raise RuntimeError(f"verl OPD subprocess exited with status {return_code}")


def _find_checkpoint_file(checkpoint_dir: str, needles: tuple[str, ...]) -> str | None:
    for root, _dirs, files in os.walk(checkpoint_dir):
        for name in sorted(files):
            lowered = name.lower()
            if any(needle in lowered for needle in needles):
                return os.path.join(root, name)
    return None


def _stage_retry_contract(
    checkpoint_dir: str,
    *,
    step: int,
    seed: int,
    prompt_pool_fingerprint: str,
    prompts_per_step: int,
    group_size: int,
    adapter_dir: str,
    accounting_state: dict,
) -> None:
    for name in os.listdir(adapter_dir):
        if name == "adapter_config.json" or name.startswith("adapter_model"):
            shutil.copy2(os.path.join(adapter_dir, name), os.path.join(checkpoint_dir, name))
    optimizer_source = _find_checkpoint_file(checkpoint_dir, ("optim", "optimizer"))
    if optimizer_source is None:
        raise RuntimeError("verl OPD checkpoint has no optimizer state")
    shutil.copy2(optimizer_source, os.path.join(checkpoint_dir, "optimizer.pt"))
    rng_source = os.path.join(checkpoint_dir, "data.pt")
    if not os.path.isfile(rng_source):
        rng_source = _find_checkpoint_file(checkpoint_dir, ("extra", "rng"))
    if rng_source is None:
        raise RuntimeError("verl OPD checkpoint has no resumable dataloader or rng state")
    shutil.copy2(rng_source, os.path.join(checkpoint_dir, "rng_state.pth"))
    state = {
        **accounting_state,
        "contract_version": OPD_RESUME_STATE_VERSION,
        "seed": seed,
        "opt_steps": step,
        "step": step,
        "rollout_seed_ordinal": step * prompts_per_step * group_size,
        "prompt_pool_fingerprint": prompt_pool_fingerprint,
        "verl_checkpoint": True,
    }
    validate_opd_resume_state_metadata(state, expected_seed=seed, checkpoint_step=step)
    with open(os.path.join(checkpoint_dir, "opd_state.json"), "w", encoding="utf-8") as file:
        json.dump(state, file, sort_keys=True)


class _OpdVerlCheckpointWatcher(_VerlCheckpointWatcher):
    def __init__(
        self,
        *,
        seed: int,
        prompt_pool_fingerprint: str,
        prompts_per_step: int,
        group_size: int,
        accounting_state,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.seed = seed
        self.prompt_pool_fingerprint = prompt_pool_fingerprint
        self.prompts_per_step = prompts_per_step
        self.group_size = group_size
        self.accounting_state = accounting_state

    def _should_publish(self, step: int) -> bool:
        return True

    def _publishable(self, pending: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """keep every pending retry state because each checkpoint carries distinct accounting."""
        return pending

    def _publish(self, step: int, checkpoint_dir: str) -> None:
        actor_dir = os.path.join(checkpoint_dir, "actor")
        adapter_dir = os.path.join(self.export_root, f"step-{step}")
        # claimed before the work: `_pending` filters on the discovered set, so leaving a step
        # unclaimed while its export or retry-contract staging raises hands it straight back to the
        # next sweep.
        self.lifecycle.mark_discovered(step)
        _export_checkpoint_adapter(
            actor_dir,
            adapter_dir,
            model_id=self.model_id,
            model_revision=self.model_revision,
            exclude_modules=self.exclude_modules,
            python_bin=self.python_bin,
            preprocessor=self.preprocessor,
        )
        _stage_retry_contract(
            checkpoint_dir,
            step=step,
            seed=self.seed,
            prompt_pool_fingerprint=self.prompt_pool_fingerprint,
            prompts_per_step=self.prompts_per_step,
            group_size=self.group_size,
            adapter_dir=adapter_dir,
            accounting_state=self.accounting_state(step),
        )
        # the adapter and its retry contract are both on disk; opd keeps the export because
        # `_stage_retry_contract` above has already copied from it and the resume state references it.
        self.lifecycle.mark_staged(step)

        def publish_required_adapter() -> None:
            if step in self.required_steps:
                # opd publishes only required steps, and a required publish raises rather than
                # returning None, so reaching the next line IS the durable fact. sft needs a
                # returned-subfolder check because its `required` varies per step.
                _worker_hf.publish_deployable_checkpoint(
                    adapter_dir,
                    step,
                    required=True,
                    _provenance_ready=True,
                )
                self.lifecycle.mark_deployable_published(step)

        uploaded = _worker_hf.upload_resume_checkpoint(
            step,
            checkpoint_dir,
            before_upload=publish_required_adapter,
            # the callback, not the return value: see mark_resume_uploaded's contract.
            after_upload=lambda: self.lifecycle.mark_resume_uploaded(step),
        )
        if step in self.required_steps and not uploaded:
            self.lifecycle.mark_failed(step)
            resume_upload_unavailable(step, checkpoint_dir, job_label="opd")


def _restore_verl_resume(
    local_dir: str,
    *,
    prompt_pool_fingerprint: str,
    update_horizon: int,
    world_size: int,
    expected_fsdp_generation: FsdpGeneration,
) -> tuple[int, dict | None]:
    revision = _worker_state.OPD_RESUME_REVISION or None
    # no `prefer`: opd pins an exact commit via OPD_RESUME_REVISION with fail_closed, and
    # validate_opd_resume_state_metadata is keyed to that checkpoint's step, so silently picking a
    # different candidate here would violate the retry contract those enforce.
    resume = _worker_hf.hf_resume_checkpoint(fail_closed=bool(revision), revision=revision)
    if not resume:
        return 0, None
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume))
    if match is None:
        raise RuntimeError(f"invalid OPD resume checkpoint path {resume!r}")
    step = int(match.group(1))
    # before the state is read, not after: a checkpoint this attempt's rank count cannot load is
    # discarded whole, and the loop accounting it carries only describes steps that get redone.
    if revision:
        # a pinned revision means a prior attempt crossed optimizer.step(): the control-plane
        # gate (verify_opd_replacement_safe) authorized this replacement only to continue from
        # exactly this checkpoint. discarding it and restarting from step 0 would repeat
        # already-billed teacher work and optimizer steps outside what the gate approved, so an
        # incompatible native checkpoint fails closed here instead of starting fresh.
        inspection = inspect_resume_checkpoint(
            resume,
            world_size=world_size,
            expected_fsdp_generation=expected_fsdp_generation,
        )
        if not inspection.loadable:
            raise RuntimeError(
                f"permanent OPD resume failure: pinned resume revision {revision!r} names "
                f"checkpoint {os.path.basename(resume)}, rejected because "
                f"{inspection.diagnostic()}. restarting from step 0 would violate the pinned-resume "
                "contract, so this attempt refuses to train; relaunch from a compatible complete "
                f"fsdp{expected_fsdp_generation} checkpoint."
            )
    elif not resume_topology_matches(
        resume,
        world_size=world_size,
        expected_fsdp_generation=expected_fsdp_generation,
        job_label="OPD",
    ):
        return 0, None
    with open(os.path.join(resume, "opd_state.json"), encoding="utf-8") as file:
        state = validate_opd_resume_state_metadata(
            json.load(file), expected_seed=int(_worker_state.SEED), checkpoint_step=step
        )
    if state["prompt_pool_fingerprint"] != prompt_pool_fingerprint:
        raise RuntimeError("OPD resume prompt pool does not match the current run")
    if step > update_horizon:
        raise RuntimeError("OPD resume checkpoint is beyond the requested update horizon")
    target = os.path.join(local_dir, f"global_step_{step}")
    shutil.copytree(resume, target, dirs_exist_ok=True)
    with open(os.path.join(local_dir, "latest_checkpointed_iteration.txt"), "w") as file:
        file.write(str(step))
    return step, state
