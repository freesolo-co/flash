"""Checkpoint durability and gradient verification for the GRPO trainer."""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import threading
import time

import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.runtime.state as _worker_state
import flash.engine.worker.verl.checkpoints as verl_checkpoints
from flash._internal.fileio import reject_duplicate_keys
from flash.adapters.artifacts import MAX_ATTEMPT_ID, has_loadable_adapter_weights
from flash.engine.support.verl_checkpoint import FsdpCheckpointInspection
from flash.engine.support.verl_policy import FsdpGeneration
from flash.engine.worker.io.heartbeat import join_while_draining
from flash.engine.worker.train.core.lifecycle.ledger import CheckpointLedger
from flash.engine.worker.train.entry.backend_common import (
    completed_checkpoint_step,
    export_peft_adapter,
    stamp_adapter_dir_provenance,
    undiscovered_checkpoint_dirs,
)
from flash.engine.worker.verl.checkpoints import MergeDiskExhaustedError, MergeDiskHeadroomError

_RESUME_MANIFEST = "_flash_resume_manifest.json"
_RESUME_MANIFEST_VERSION = 2
_LOCAL_REQUIRED_ADAPTER_STORE = "_flash_required_adapter_store"
_HF_REQUIRED_ADAPTER_ROOT = "required-adapters"


def _validate_internal_adapter_dir(path: str, *, label: str) -> None:
    """reject malformed internal adapter trees before restoring or publishing them."""
    if not os.path.isdir(path) or os.path.islink(path):
        raise RuntimeError(f"invalid GRPO required adapter {label}: not a real directory")
    root = os.path.realpath(path)
    names = set(os.listdir(path))
    if "adapter_config.json" not in names or not has_loadable_adapter_weights(names):
        raise RuntimeError(
            f"invalid GRPO required adapter {label}: missing adapter config or weights"
        )
    for current, dirs, files in os.walk(path):
        if os.path.commonpath((root, os.path.realpath(current))) != root:
            raise RuntimeError(f"invalid GRPO required adapter {label}: path traversal")
        for name in (*dirs, *files):
            candidate = os.path.join(current, name)
            if os.path.islink(candidate):
                raise RuntimeError(f"invalid GRPO required adapter {label}: symlink {name!r}")
            if os.path.commonpath((root, os.path.realpath(candidate))) != root:
                raise RuntimeError(f"invalid GRPO required adapter {label}: path traversal")


def _strict_positive_step(value, *, field: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
        raise RuntimeError(
            f"invalid GRPO resume manifest {field}: expected an integer in 1..{maximum}"
        )
    return value


def _strict_attempt(value, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_ATTEMPT_ID:
        raise RuntimeError(
            f"invalid GRPO resume manifest {field}: expected an integer in 0..{MAX_ATTEMPT_ID}"
        )
    return value


class _DuplicateJsonKeyError(ValueError):
    pass


_REJECT_DUPLICATE_MANIFEST_KEYS = reject_duplicate_keys(
    lambda key: _DuplicateJsonKeyError(f"duplicate key {key!r}")
)


def _read_resume_manifest(
    checkpoint_dir: str,
    *,
    checkpoint_step: int,
    required_steps: frozenset[int] | set[int] | tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], int | None]:
    """read one exact current-contract resume manifest and reject every ambiguous shape."""
    path = os.path.join(checkpoint_dir, _RESUME_MANIFEST)
    if not os.path.isfile(path) or os.path.islink(path):
        raise RuntimeError(
            f"GRPO resume checkpoint {checkpoint_step} is missing a regular {_RESUME_MANIFEST}"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle, object_pairs_hook=_REJECT_DUPLICATE_MANIFEST_KEYS)
    except (OSError, UnicodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as error:
        raise RuntimeError(
            f"GRPO resume checkpoint {checkpoint_step} has an unreadable {_RESUME_MANIFEST}"
        ) from error
    expected_keys = {
        "version",
        "checkpoint_step",
        "checkpoint_attempt",
        "required_adapters",
        "first_positive_grad_step",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise RuntimeError("invalid GRPO resume manifest keys")
    if type(manifest["version"]) is not int or manifest["version"] != _RESUME_MANIFEST_VERSION:
        raise RuntimeError("invalid GRPO resume manifest version")
    declared_step = _strict_positive_step(
        manifest["checkpoint_step"], field="checkpoint_step", maximum=checkpoint_step
    )
    if declared_step != checkpoint_step:
        raise RuntimeError(
            f"invalid GRPO resume manifest checkpoint_step {declared_step}; expected {checkpoint_step}"
        )
    checkpoint_attempt = _strict_attempt(manifest["checkpoint_attempt"], field="checkpoint_attempt")
    raw_required = manifest["required_adapters"]
    if not isinstance(raw_required, list):
        raise RuntimeError("invalid GRPO resume manifest required_adapters")
    parsed_required = []
    for value in raw_required:
        if not isinstance(value, dict) or set(value) != {"step", "attempt"}:
            raise RuntimeError("invalid GRPO resume manifest required_adapters entry")
        parsed_required.append(
            (
                _strict_positive_step(
                    value["step"], field="required_adapters.step", maximum=checkpoint_step
                ),
                _strict_attempt(value["attempt"], field="required_adapters.attempt"),
            )
        )
    parsed_required = tuple(parsed_required)
    if parsed_required != tuple(sorted(set(parsed_required))):
        raise RuntimeError("invalid GRPO resume manifest noncanonical required_adapters")
    if any(attempt > checkpoint_attempt for _step, attempt in parsed_required):
        raise RuntimeError("invalid GRPO resume manifest future required adapter attempt")
    parsed_steps = tuple(step for step, _attempt in parsed_required)
    if len(parsed_steps) != len(set(parsed_steps)):
        raise RuntimeError("invalid GRPO resume manifest duplicate required adapter step")
    unknown = sorted(set(parsed_steps) - set(required_steps))
    if unknown:
        raise RuntimeError(f"invalid GRPO resume manifest unknown required adapter steps {unknown}")
    positive = manifest["first_positive_grad_step"]
    if positive is not None:
        positive = _strict_positive_step(
            positive, field="first_positive_grad_step", maximum=checkpoint_step
        )
    return parsed_required, positive


def _write_resume_manifest(
    checkpoint_dir: str,
    *,
    checkpoint_step: int,
    checkpoint_attempt: int,
    required_adapters: tuple[tuple[int, int], ...],
    first_positive_grad_step: int | None,
) -> None:
    """atomically write the small durability manifest included in a native checkpoint commit."""
    manifest = {
        "version": _RESUME_MANIFEST_VERSION,
        "checkpoint_step": checkpoint_step,
        "checkpoint_attempt": checkpoint_attempt,
        "required_adapters": [
            {"step": step, "attempt": attempt} for step, attempt in required_adapters
        ],
        "first_positive_grad_step": first_positive_grad_step,
    }
    path = os.path.join(checkpoint_dir, _RESUME_MANIFEST)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        raise


def _canonical_namespace_dir(root: str, *, prefix: str, value: int, label: str) -> str:
    canonical = f"{prefix}{value}"
    for entry in os.scandir(root):
        name = entry.name
        suffix = name.removeprefix(prefix)
        if (
            name.startswith(prefix)
            and suffix.isdigit()
            and int(suffix) == value
            and name != canonical
        ):
            raise RuntimeError(f"invalid GRPO required-adapter namespace collision {name!r}")
    path = os.path.join(root, canonical)
    if not os.path.isdir(path) or os.path.islink(path):
        raise RuntimeError(f"invalid GRPO required-adapter namespace {label} {canonical!r}")
    return path


def _internal_adapter_sources(
    checkpoint_dir: str, references: tuple[tuple[int, int], ...]
) -> dict[int, str]:
    """resolve immutable attempt-scoped adapters referenced by one native checkpoint."""
    if not references:
        return {}
    root = os.path.join(os.path.dirname(checkpoint_dir), _HF_REQUIRED_ADAPTER_ROOT)
    if not os.path.isdir(root) or os.path.islink(root):
        raise RuntimeError("GRPO resume is missing the required-adapter internal namespace")
    sources: dict[int, str] = {}
    for step, attempt in references:
        attempt_dir = _canonical_namespace_dir(
            root, prefix="attempt-", value=attempt, label="attempt"
        )
        step_dir = _canonical_namespace_dir(attempt_dir, prefix="step-", value=step, label="step")
        adapter = os.path.join(step_dir, "adapter")
        _validate_internal_adapter_dir(adapter, label=f"attempt-{attempt}/step-{step}")
        sources[step] = adapter
    return sources


def _stage_internal_adapters(
    checkpoint_dir: str,
    local_dir: str,
    references: tuple[tuple[int, int], ...],
) -> None:
    """copy manifest-referenced internal adapters to a deterministic local restore store."""
    sources = _internal_adapter_sources(checkpoint_dir, references)
    store = os.path.join(local_dir, _LOCAL_REQUIRED_ADAPTER_STORE)
    shutil.rmtree(store, ignore_errors=True)
    if not sources:
        return
    for step, source in sorted(sources.items()):
        destination = os.path.join(store, f"step-{step}", "adapter")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copytree(source, destination)
        _validate_internal_adapter_dir(destination, label=f"step-{step}")


class _VerlResumeUploader:
    """stream completed verl checkpoints without outrunning their durability evidence."""

    def __init__(
        self,
        local_dir: str,
        *,
        resume_step: int,
        required_steps: tuple[int, ...] = (),
        metric_evidence,
        export_root: str = "",
        python_bin: str = "",
        model_id: str = "",
        model_revision: str = "",
        exclude_modules: str | None = None,
        preprocessor=None,
    ) -> None:
        self.local_dir = local_dir
        self.resume_step = int(resume_step)
        self.metric_evidence = metric_evidence
        self.required_steps = frozenset(required_steps)
        self.lifecycle = CheckpointLedger()
        self.lifecycle.seed_resumed_step(resume_step, self.required_steps)
        self._resume_attempted: set[int] = {resume_step} if resume_step else set()
        self.staged_adapters: dict[int, str] = {}
        self._internal_adapter_attempts: dict[int, int] = {}
        self._attempt = _strict_attempt(_worker_state.ATTEMPT, field="attempt")
        self._reported_required_debt = False
        self._blocked_evidence_step: int | None = None
        self.export_root = export_root
        self.python_bin = python_bin
        self.model_id = model_id
        self.model_revision = model_revision
        self.exclude_modules = exclude_modules
        self.preprocessor = preprocessor
        self._publication_latch = threading.Event()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def credit_durable_required_steps(self, resume_step: int) -> None:
        """credit required deployables that are verifiably servable before this attempt."""
        for step in sorted(self.required_steps):
            if step <= int(resume_step) and _worker_hf._deployable_adapter_on_hf(step):
                self.lifecycle.mark_deployable_published(step)

    def restore_staged_adapters(self, resume_step: int) -> None:
        """validate the restored manifest, trust its metric fact, and hydrate owed adapters."""
        resume_step = int(resume_step)
        if not resume_step:
            self.metric_evidence.set_prior_positive_step(None, checkpoint_step=0)
            return
        checkpoint_dir = os.path.join(self.local_dir, f"global_step_{resume_step}")
        referenced, positive = _read_resume_manifest(
            checkpoint_dir,
            checkpoint_step=resume_step,
            required_steps=self.required_steps,
        )
        self.metric_evidence.set_prior_positive_step(positive, checkpoint_step=resume_step)
        self._internal_adapter_attempts.update(referenced)
        published = self.lifecycle.deployable_published_steps
        owed = set(self._internal_adapter_attempts) - published
        if not owed:
            return
        store = os.path.join(self.local_dir, _LOCAL_REQUIRED_ADAPTER_STORE)
        os.makedirs(self.export_root, exist_ok=True)
        restored: dict[int, str] = {}
        try:
            for step in sorted(owed):
                source = os.path.join(store, f"step-{step}", "adapter")
                _validate_internal_adapter_dir(source, label=f"step-{step}")
                name = f"step-{step}"
                destination = os.path.join(self.export_root, name)
                temporary = os.path.join(self.export_root, f".{name}.restoring")
                shutil.rmtree(temporary, ignore_errors=True)
                shutil.copytree(source, temporary)
                shutil.rmtree(destination, ignore_errors=True)
                os.replace(temporary, destination)
                restored[step] = destination
        except BaseException:
            for path in restored.values():
                shutil.rmtree(path, ignore_errors=True)
            for step in owed:
                shutil.rmtree(
                    os.path.join(self.export_root, f".step-{step}.restoring"), ignore_errors=True
                )
            raise
        for step, path in restored.items():
            self.staged_adapters[step] = path
            self.lifecycle.mark_staged(step)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        join_while_draining(self._thread, "verl resume uploader")

    def raise_if_incomplete(self) -> None:
        """fail clean completion when required durability or uploader work failed."""
        if isinstance(self._error, (MergeDiskHeadroomError, MergeDiskExhaustedError)):
            raise self._error
        if self._error is not None:
            raise RuntimeError("verl resume uploader failed") from self._error
        missing = self.lifecycle.missing_deployables(self.required_steps)
        if missing:
            raise RuntimeError(f"required saves were not durably published: {missing}")

    def allow_deployable_publication(self) -> None:
        self._publication_latch.set()

    def _deployable_allowed(self) -> bool:
        return self._publication_latch.is_set()

    def _completed_step(self) -> int:
        return completed_checkpoint_step(self.local_dir)

    def _pending(self) -> list[tuple[int, str]]:
        return undiscovered_checkpoint_dirs(
            self.local_dir, self._completed_step(), self.lifecycle.discovered_steps
        )

    def _stage_deployable(self, step: int, checkpoint_dir: str) -> str:
        """export a required step outside verl retention while publication remains closed."""
        actor_dir = os.path.join(checkpoint_dir, "actor")
        adapter_dir = os.path.join(self.export_root, f"step-{step}")
        shutil.rmtree(adapter_dir, ignore_errors=True)
        os.makedirs(adapter_dir, exist_ok=True)
        export_peft_adapter(
            actor_dir, adapter_dir, base_model_id=self.model_id, python_bin=self.python_bin
        )
        self.preprocessor.save_pretrained(adapter_dir)
        stamp_adapter_dir_provenance(
            adapter_dir,
            self.model_id,
            self.model_revision,
            exclude_modules=self.exclude_modules,
        )
        _worker_hf.write_base_model_provenance(adapter_dir, self.model_id, self.model_revision)
        return adapter_dir

    def _make_required_adapter_durable(self, step: int, adapter_dir: str) -> None:
        """upload one withheld adapter once under the non-serving singular namespace."""
        if step in self._internal_adapter_attempts:
            return
        _validate_internal_adapter_dir(adapter_dir, label=f"step-{step}")
        uploaded = _worker_hf.hf_upload_folder(
            adapter_dir,
            f"checkpoint/{_HF_REQUIRED_ADAPTER_ROOT}/attempt-{self._attempt}/step-{step}/adapter",
            required=True,
        )
        if not uploaded:
            raise RuntimeError(f"required internal adapter upload returned false at step {step}")
        self._internal_adapter_attempts[step] = self._attempt

    def _publish_staged(self, step: int, adapter_dir: str) -> None:
        _worker_hf.publish_deployable_checkpoint(
            adapter_dir, step, required=True, _provenance_ready=True
        )
        self.lifecycle.mark_deployable_published(step)

    def _publish_ready(self) -> None:
        if not self._deployable_allowed():
            return
        published = self.lifecycle.deployable_published_steps
        for step in sorted(self.staged_adapters):
            if step in self._internal_adapter_attempts and step not in published:
                self._publish_staged(step, self.staged_adapters[step])

    def _required_debt(self, checkpoint_step: int) -> list[int]:
        published = self.lifecycle.deployable_published_steps
        return sorted(
            step
            for step in self.required_steps
            if step <= checkpoint_step
            and step not in published
            and step not in self._internal_adapter_attempts
        )

    def _manifest_required_adapters(self, checkpoint_step: int) -> tuple[tuple[int, int], ...]:
        published = self.lifecycle.deployable_published_steps
        return tuple(
            (step, self._internal_adapter_attempts[step])
            for step in sorted(self._internal_adapter_attempts)
            if step <= checkpoint_step and step not in published
        )

    def _run(self) -> None:
        try:
            while True:
                stopping = self._stop.is_set()
                deferred_for_evidence = False
                if self._blocked_evidence_step is not None:
                    evidence_ready, _positive = self.metric_evidence.checkpoint_manifest_evidence(
                        self._blocked_evidence_step
                    )
                    if not evidence_ready:
                        if stopping:
                            return
                        time.sleep(0.5)
                        continue
                    self._blocked_evidence_step = None
                for step, path in self._pending():
                    if (
                        step in self.required_steps
                        and step not in self.lifecycle.deployable_published_steps
                        and step not in self.staged_adapters
                    ):
                        adapter_dir = self._stage_deployable(step, path)
                        self.staged_adapters[step] = adapter_dir
                        self.lifecycle.mark_staged(step)
                        try:
                            self._make_required_adapter_durable(step, adapter_dir)
                        except Exception as error:
                            print(
                                f"[rl-verl] required adapter durability failed at step {step}: {error}",
                                flush=True,
                            )
                        self._publish_ready()
                    if step not in self._resume_attempted:
                        evidence_ready, positive_step = (
                            self.metric_evidence.checkpoint_manifest_evidence(step)
                        )
                        if not evidence_ready:
                            self._blocked_evidence_step = step
                            deferred_for_evidence = True
                            break
                        self._blocked_evidence_step = None
                        debt = self._required_debt(step)
                        if debt:
                            if not self._reported_required_debt:
                                print(
                                    "[rl-verl] retaining the previous resume checkpoint because "
                                    f"checkpoint {step} has undurable required saves {debt}",
                                    flush=True,
                                )
                                self._reported_required_debt = True
                            self.lifecycle.mark_discovered(step)
                            continue
                        self._resume_attempted.add(step)
                        _write_resume_manifest(
                            path,
                            checkpoint_step=step,
                            checkpoint_attempt=self._attempt,
                            required_adapters=self._manifest_required_adapters(step),
                            first_positive_grad_step=positive_step,
                        )
                        try:
                            _worker_hf.upload_resume_checkpoint(
                                step,
                                path,
                                after_upload=lambda step=step: self.lifecycle.mark_resume_uploaded(
                                    step
                                ),
                            )
                        except Exception as error:
                            print(
                                f"[rl-verl] resume checkpoint upload failed at step {step}: {error}",
                                flush=True,
                            )
                    self.lifecycle.mark_discovered(step)
                self._publish_ready()
                if stopping and deferred_for_evidence:
                    return
                if stopping and not self._pending():
                    return
                time.sleep(0.5)
        except BaseException as error:
            self._error = error


def _restore_verl_resume(
    local_dir: str,
    *,
    world_size: int,
    expected_fsdp_generation: FsdpGeneration,
    required_steps: tuple[int, ...] = (),
) -> int:
    """stage this run's current-contract native checkpoint and its internal adapter store."""
    accepted: dict[str, tuple[tuple[int, int], ...]] = {}
    rejected: dict[str, RuntimeError] = {}

    def accept(path: str, _inspection: FsdpCheckpointInspection) -> bool:
        if path in accepted:
            return True
        if path in rejected:
            return False
        name = os.path.basename(path)
        if not name.startswith("checkpoint-") or not name[len("checkpoint-") :].isdigit():
            rejected[path] = RuntimeError(f"invalid GRPO resume checkpoint path {path!r}")
            return False
        step = int(name[len("checkpoint-") :])
        if step <= 0:
            rejected[path] = RuntimeError(f"invalid GRPO resume checkpoint path {path!r}")
            return False
        try:
            referenced, _positive = _read_resume_manifest(
                path,
                checkpoint_step=step,
                required_steps=required_steps,
            )
            _internal_adapter_sources(path, referenced)
        except RuntimeError as error:
            rejected[path] = error
            return False
        accepted[path] = referenced
        return True

    resume, inspection = verl_checkpoints.select_verl_resume_checkpoint(
        world_size=world_size,
        expected_fsdp_generation=expected_fsdp_generation,
        accept=accept,
    )
    if not resume:
        return 0
    if inspection is None:
        raise RuntimeError("selected GRPO resume checkpoint is missing its inspection")
    if not inspection.loadable:
        return verl_checkpoints.stage_verl_resume(
            resume,
            local_dir,
            job_label="GRPO",
            world_size=world_size,
            expected_fsdp_generation=expected_fsdp_generation,
            inspection=inspection,
        )
    referenced = accepted.get(resume)
    if referenced is None:
        error = rejected.get(resume)
        if error is not None:
            raise error
        raise RuntimeError(f"invalid GRPO resume checkpoint path {resume!r}")
    _stage_internal_adapters(resume, local_dir, referenced)
    return verl_checkpoints.stage_verl_resume(
        resume,
        local_dir,
        job_label="GRPO",
        world_size=world_size,
        expected_fsdp_generation=expected_fsdp_generation,
        inspection=inspection,
    )


def _check_grpo_had_a_gradient(
    reward_history: list[float],
    adv_spread_history: list[float],
    grad_norms: dict[int, float],
    *,
    expected_steps: range | tuple[int, ...],
    resume_step: int = 0,
    prior_positive_step: int | None = None,
    already_complete: bool = False,
) -> None:
    """require complete current metrics and positive-gradient evidence across all attempts."""
    resume_step = int(resume_step)
    if prior_positive_step is not None and (
        isinstance(prior_positive_step, bool)
        or not isinstance(prior_positive_step, int)
        or prior_positive_step <= 0
        or prior_positive_step > resume_step
    ):
        raise RuntimeError("invalid prior GRPO positive-gradient evidence")
    if not already_complete and not reward_history:
        raise RuntimeError(
            "verl reported no reward metrics for the whole run; the flash reward bridge was "
            "never consulted (wiring regression); refusing to publish a policy trained on "
            "default rewards"
        )
    if not already_complete and not adv_spread_history:
        raise RuntimeError(
            "verl reported reward metrics but no advantage metrics for any step; "
            "critic/advantages/max and /min could not be parsed (metric-format regression)"
        )
    expected = tuple(expected_steps)
    actual = tuple(sorted(grad_norms))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "GRPO gradient norms do not cover the executed optimizer steps: "
            f"missing={missing}, extra={extra}"
        )
    for step in actual:
        value = grad_norms[step]
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"GRPO gradient norm for step {step} is not finite and nonnegative")
    if already_complete:
        if prior_positive_step is None:
            raise RuntimeError(
                "GRPO resume has no durable positive actor gradient evidence; refusing publication"
            )
        return
    if not actual:
        raise RuntimeError("verl reported no actor/grad_norm evidence for newly executed steps")
    if prior_positive_step is None and not any(value > 0.0 for value in grad_norms.values()):
        raise RuntimeError(
            f"grpo reported zero actor gradient norm on all {len(grad_norms)} newly executed steps "
            "and no prior positive-gradient evidence; refusing to publish an unchanged adapter"
        )
