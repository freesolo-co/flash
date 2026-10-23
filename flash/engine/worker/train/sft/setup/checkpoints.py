"""Export and publish adapters alongside bounded native checkpoint persistence.

The watcher exports completed verl checkpoints while training continues. A separate
serial worker persists pinned native snapshots, so a recovery upload does not block
the next adapter publication while queue capacity remains. Shutdown drains both.

Split out of `flash.engine.worker.train.entry.sft_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import os
import shutil
import threading
import time

import flash.engine.worker.io.hf as _worker_hf
from flash.engine.worker.io.heartbeat import join_while_draining
from flash.engine.worker.runtime.state import WorkerContext
from flash.engine.worker.train.core.lifecycle.ledger import CheckpointLedger
from flash.engine.worker.train.core.lifecycle.persistence import (
    CheckpointPersistence,
    pin_checkpoint,
)
from flash.engine.worker.train.core.lifecycle.timing import checkpoint_phase
from flash.engine.worker.train.entry.backend_common import (
    completed_checkpoint_step,
    export_peft_adapter,
    resolve_checkpoint_actor_dir,
    stamp_adapter_dir_provenance,
    undiscovered_checkpoint_dirs,
)
from flash.engine.worker.verl.checkpoints import (
    MergeDiskExhaustedError,
    resume_upload_unavailable,
)


def _sft_train():
    """The trainer module, imported lazily because it imports this one.

    The watcher's export step is patched as `sft_train._export_checkpoint_adapter` by the sft tests,
    so `_publish` has to resolve it through that module rather than as a global here -- a direct
    call would bind this module's own function and run the real exporter under the patch.
    """
    from flash.engine.worker.train.entry import sft_train

    return sft_train


def _copy_processing_sidecars(actor_dir: str, adapter_dir: str) -> None:
    source = os.path.join(actor_dir, "huggingface")
    if not os.path.isdir(source):
        return
    prefixes = (
        "added_tokens",
        "chat_template",
        "merges",
        "preprocessor_config",
        "processor_config",
        "special_tokens_map",
        "tokenizer",
        "video_preprocessor_config",
        "vocab",
    )
    for name in os.listdir(source):
        if name.startswith(prefixes):
            src = os.path.join(source, name)
            dst = os.path.join(adapter_dir, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)


def _export_checkpoint_adapter(
    actor_dir: str,
    adapter_dir: str,
    *,
    model_id: str,
    model_revision: str,
    python_bin: str,
    exclude_modules: str | None = None,
    preprocessor=None,
) -> None:
    shutil.rmtree(adapter_dir, ignore_errors=True)
    export_peft_adapter(
        actor_dir,
        adapter_dir,
        base_model_id=model_id,
        python_bin=python_bin,
    )
    _copy_processing_sidecars(actor_dir, adapter_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(adapter_dir)
    stamp_adapter_dir_provenance(
        adapter_dir, model_id, model_revision, exclude_modules=exclude_modules
    )
    _worker_hf.write_base_model_provenance(adapter_dir, model_id, model_revision)


class _VerlCheckpointWatcher:
    """watch verl's completion marker and publish each completed flash checkpoint."""

    def __init__(
        self,
        *,
        local_dir: str,
        export_root: str,
        python_bin: str,
        model_id: str,
        model_revision: str,
        required_steps: tuple[int, ...],
        final_step: int | None = None,
        exclude_modules: str | None = None,
        preprocessor=None,
        context: WorkerContext,
    ) -> None:
        self.context = context
        self.local_dir = local_dir
        self.export_root = export_root
        # At most two pinned native snapshots, including the active recovery upload.
        # Their lifetime is independent of adapter publication and trainer retention.
        self.staging_root = os.path.join(export_root, "_staging")
        self.published_dir = os.path.join(export_root, "_published")
        self.python_bin = python_bin
        self.model_id = model_id
        self.model_revision = model_revision
        self.exclude_modules = exclude_modules
        self.preprocessor = preprocessor
        self.required_steps = frozenset(required_steps)
        self.final_step = final_step
        self.lifecycle = CheckpointLedger()
        self.persistence = CheckpointPersistence(context=context)
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        # Resumed steps already verified on HF must release the child's retention gate too.
        for step in self.lifecycle.deployable_published_steps:
            self._acknowledge_publication(step)
        self._thread.start()

    def _acknowledge_publication(self, step: int) -> None:
        os.makedirs(self.published_dir, exist_ok=True)
        with open(os.path.join(self.published_dir, f"step-{step}"), "w"):
            pass

    def raise_if_failed(self) -> None:
        if isinstance(self._error, MergeDiskExhaustedError):
            raise self._error
        if self._error is not None:
            raise RuntimeError("verl checkpoint watcher failed") from self._error

    def stop(self, *, require_complete: bool) -> None:
        self._stop.set()
        # The run deadline bounds both publication and persistence draining. A watcher
        # that failed surfaces its original error after accepted storage work drains.
        join_while_draining(self._thread, "verl checkpoint watcher", context=self.context)
        self.raise_if_failed()
        if require_complete:
            # the published adapters, not the steps this watcher handled: a required save is owed a
            # servable artifact, and handling a step covers outcomes that never produce one. the
            # boolean from upload_resume_checkpoint cannot stand in for it either -- it is also True
            # when there is no artifact repository, which returns before the publish callback runs.
            missing = self.lifecycle.missing_deployables(self.required_steps)
            if missing:
                raise RuntimeError(f"required saves were not durably published: {missing}")

    def _completed_step(self) -> int:
        return completed_checkpoint_step(self.local_dir)

    def _pending(self) -> list[tuple[int, str]]:
        """the completed checkpoint dirs this uploader has not handled yet, oldest first."""
        return undiscovered_checkpoint_dirs(
            self.local_dir, self._completed_step(), self.lifecycle.discovered_steps
        )

    def _should_publish(self, step: int) -> bool:
        return not self.required_steps or step in self.required_steps

    def _publishable(self, pending: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """coalesce a superseded OPTIONAL backlog to its newest save, required steps kept.

        the coalescing exists because each export writes a full model copy to the container disk,
        and the publisher runs on its own thread while training writes the next checkpoint. gating
        the whole thing on `self.required_steps` being empty disabled it for exactly the runs that
        need it most: `save_at_steps` makes every step required, so a backlog kept every optional
        step too and the disk held several full copies at once.

        a required step is still never dropped -- it is owed a durable artifact. only steps that
        `_should_publish` would skip anyway are coalesced away, so this changes no authored
        contract; it just stops claiming them one sweep later than it has to.
        """
        if len(pending) <= 1:
            return pending
        # a required step is owed a durable artifact and is never coalesced away. an optional step is
        # only worth publishing when it is the newest save in the backlog; anything older than that
        # is superseded by weights this same sweep is about to publish.
        keep = [
            entry
            for index, entry in enumerate(pending)
            if entry[0] in self.required_steps or index == len(pending) - 1
        ]
        superseded = [entry for entry in pending if entry not in keep]
        if not superseded:
            return pending
        # discovered only: these are claimed so the next sweep skips them, and deliberately gain no
        # durability fact. nothing was published for them, and the ledger has to say so.
        for step, _ in superseded:
            self.lifecycle.mark_discovered(step)
        print(
            f"[ckpt] publishing step(s) {', '.join(str(step) for step, _ in keep)} and skipping "
            f"superseded periodic checkpoint(s) {', '.join(str(step) for step, _ in superseded)}: "
            "the publisher is behind training, and each export writes a full model copy to the "
            "same disk",
            flush=True,
        )
        return keep

    def _staged_source(self, step: int, checkpoint_dir: str) -> str:
        """Pin completed files until the recovery uploader releases them."""
        self.persistence.wait_for_capacity()
        with checkpoint_phase(step, "snapshot_pin"):
            return pin_checkpoint(
                checkpoint_dir, os.path.join(self.staging_root, f"global_step_{step}")
            )

    def _publish(self, step: int, checkpoint_dir: str) -> None:
        if not self._should_publish(step):
            self.lifecycle.mark_discovered(step)
            return
        staged_dir = self._staged_source(step, checkpoint_dir)
        try:
            with checkpoint_phase(step, "adapter_publication"):
                self._publish_from(step, staged_dir)
            if self.lifecycle.facts(step).deployable_published:
                self._acknowledge_publication(step)
            self._persist_resume(step, staged_dir, job_label="sft")
        except BaseException:
            shutil.rmtree(staged_dir, ignore_errors=True)
            raise

    def _persist_resume(self, step: int, checkpoint_dir: str, *, job_label: str) -> None:
        def persist():
            started = time.monotonic()
            try:
                result = _worker_hf.upload_resume_checkpoint(
                    step, checkpoint_dir, final_step=self.final_step, context=self.context
                )
                if (
                    step in self.required_steps
                    and result.resume is None
                    and not result.resume_staged
                ):
                    resume_upload_unavailable(step, checkpoint_dir, job_label=job_label)
                return result
            finally:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
                print(
                    f"[ckpt] step={step} persistence_s={time.monotonic() - started:.3f}",
                    flush=True,
                )

        def complete(result):
            self.lifecycle.record_publication(step, result)
            if step in self.required_steps and result.resume is None and not result.resume_staged:
                self.lifecycle.mark_failed(step)

        self.persistence.submit(persist, complete)

    def _publish_from(self, step: int, checkpoint_dir: str) -> None:
        actor_dir = resolve_checkpoint_actor_dir(checkpoint_dir)
        adapter_dir = os.path.join(self.export_root, f"step-{step}")

        # claimed before the work, not after it: `_pending` filters on the discovered set, so a step
        # left unclaimed while its export raises is handed straight back on the next sweep. the
        # watcher thread dies on that exception anyway, and the retry would race its own teardown.
        self.lifecycle.mark_discovered(step)

        # the try opens before the export, so a run that dies partway through writing the adapter
        # still frees the partial directory. nothing in the sft path reads the export again --
        # `step` is discovered above, `_pending` filters on that, and no sweep or finalization walks
        # `export_root` -- so an adapter kept past this call has no reader and would just accumulate
        # one directory per save.
        #
        # not safe in the two sibling watchers, which keep their exports for a real later reader:
        # the rl uploader republishes from its staged adapters on subsequent sweeps, and the opd
        # watcher hands `adapter_dir` to `_stage_retry_contract`. sft is done with it inside this call.
        try:
            _sft_train()._export_checkpoint_adapter(
                actor_dir,
                adapter_dir,
                model_id=self.model_id,
                model_revision=self.model_revision,
                exclude_modules=self.exclude_modules,
                python_bin=self.python_bin,
                preprocessor=self.preprocessor,
            )
            self.lifecycle.mark_staged(step)
            published = _worker_hf.publish_deployable_checkpoint(
                adapter_dir,
                step,
                required=step in self.required_steps,
                _provenance_ready=True,
                context=self.context,
            )
            self.lifecycle.record_publication(step, published)
        finally:
            shutil.rmtree(adapter_dir, ignore_errors=True)

    def _run(self) -> None:
        self.persistence.start()
        try:
            while True:
                self.persistence.reap()
                for step, checkpoint_dir in self._publishable(self._pending()):
                    self._publish(step, checkpoint_dir)
                # re-read rather than reusing the sweep above: verl advances the tracker right up to
                # the moment the child exits, so a step can become visible during that sweep.
                if self._stop.is_set() and not self._pending():
                    return
                time.sleep(0.5)
        except BaseException as error:
            self._error = error
        finally:
            try:
                self.persistence.close()
            except BaseException as error:
                if self._error is None:
                    self._error = error
