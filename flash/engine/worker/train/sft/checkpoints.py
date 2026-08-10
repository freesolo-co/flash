"""Export a verl checkpoint into a deployable adapter while training continues.

verl writes a full fsdp checkpoint per save step; flash needs a peft adapter directory. Doing that
conversion inline would stall the trainer, so `_VerlCheckpointWatcher` runs it on a background
thread, one checkpoint at a time, and the parent joins it at the end of the run.

Split out of `flash.engine.worker.sft_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import os
import shutil
import threading
import time

from flash.engine.worker.backend_common import (
    completed_checkpoint_step,
    export_peft_adapter,
    resolve_checkpoint_actor_dir,
    stamp_adapter_dir_provenance,
    unprocessed_checkpoint_dirs,
)
from flash.engine.worker.io.heartbeat import join_while_draining
from flash.engine.worker.runtime.pkg_proxy import W as _w


def _sft_train():
    """The trainer module, imported lazily because it imports this one.

    The watcher's export step is patched as `sft_train._export_checkpoint_adapter` by the sft tests,
    so `_publish` has to resolve it through that module rather than as a global here -- a direct
    call would bind this module's own function and run the real exporter under the patch.
    """
    from flash.engine.worker import sft_train

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
) -> None:
    shutil.rmtree(adapter_dir, ignore_errors=True)
    export_peft_adapter(
        actor_dir,
        adapter_dir,
        base_model_id=model_id,
        python_bin=python_bin,
    )
    _copy_processing_sidecars(actor_dir, adapter_dir)
    stamp_adapter_dir_provenance(adapter_dir, model_id, model_revision)
    _w.write_base_model_provenance(adapter_dir, model_id, model_revision)


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
    ) -> None:
        self.local_dir = local_dir
        self.export_root = export_root
        # sibling of export_root so it shares its filesystem: _staged_source hardlinks into it, and
        # os.link cannot cross a mount. holds one step at a time and is cleaned up after each publish.
        self.staging_root = os.path.join(export_root, "_staging")
        self.python_bin = python_bin
        self.model_id = model_id
        self.model_revision = model_revision
        self.required_steps = frozenset(required_steps)
        self.processed_steps: set[int] = set()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("verl checkpoint watcher failed") from self._error

    def stop(self, *, require_complete: bool) -> None:
        self._stop.set()
        # bounded by lack of progress, not wall clock, so a big-model upload that is still moving
        # is never killed (VERL-131). a watcher that died mid-publish is already not alive, so the
        # join returns at once and raise_if_failed surfaces its real exception rather than a
        # generic "did not stop".
        join_while_draining(self._thread, "verl checkpoint watcher")
        self.raise_if_failed()
        if require_complete:
            missing = sorted(self.required_steps - self.processed_steps)
            if missing:
                raise RuntimeError(f"required saves were not durably published: {missing}")

    def _completed_step(self) -> int:
        return completed_checkpoint_step(self.local_dir)

    def _pending(self) -> list[tuple[int, str]]:
        """the completed checkpoint dirs this uploader has not handled yet, oldest first."""
        return unprocessed_checkpoint_dirs(
            self.local_dir, self._completed_step(), self.processed_steps
        )

    def _should_publish(self, step: int) -> bool:
        return not self.required_steps or step in self.required_steps

    def _staged_source(self, step: int, checkpoint_dir: str) -> str:
        """hardlink a completed checkpoint before verl retention can prune it.

        export and upload overlap training, while verl may delete ``global_step_N`` as soon as the
        next save completes. hardlinks preserve the data in o(1) without moving the path referenced by
        ``latest_checkpointed_iteration.txt``. copies can lose the same race; renames break resume.

        best-effort: on failure, return the original race-exposed path rather than fail the run.
        """
        staged = os.path.join(self.staging_root, f"global_step_{step}")
        try:
            shutil.rmtree(staged, ignore_errors=True)
            shutil.copytree(checkpoint_dir, staged, copy_function=os.link)
        except OSError as error:
            print(
                f"[ckpt] step {step} could not be staged out of verl's retention tree "
                f"({error}); publishing from verl's copy",
                flush=True,
            )
            shutil.rmtree(staged, ignore_errors=True)
            return checkpoint_dir
        return staged

    def _publish(self, step: int, checkpoint_dir: str) -> None:
        if not self._should_publish(step):
            self.processed_steps.add(step)
            return
        staged_dir = self._staged_source(step, checkpoint_dir)
        try:
            self._publish_from(step, staged_dir)
        finally:
            if staged_dir != checkpoint_dir:
                # the links have served their purpose once the upload returns; the originals and the
                # exported adapter are untouched by this.
                shutil.rmtree(staged_dir, ignore_errors=True)

    def _publish_from(self, step: int, checkpoint_dir: str) -> None:
        actor_dir = resolve_checkpoint_actor_dir(checkpoint_dir)
        adapter_dir = os.path.join(self.export_root, f"step-{step}")
        _sft_train()._export_checkpoint_adapter(
            actor_dir,
            adapter_dir,
            model_id=self.model_id,
            model_revision=self.model_revision,
            python_bin=self.python_bin,
        )

        def publish_adapter() -> None:
            _w.publish_deployable_checkpoint(
                adapter_dir,
                step,
                required=step in self.required_steps,
                _provenance_ready=True,
            )

        uploaded = _w.upload_resume_checkpoint(
            step,
            checkpoint_dir,
            before_upload=publish_adapter,
        )
        if step in self.required_steps and not uploaded:
            raise RuntimeError(f"required save step {step} full-state checkpoint was not published")
        self.processed_steps.add(step)

    def _run(self) -> None:
        try:
            while True:
                for step, checkpoint_dir in self._pending():
                    self._publish(step, checkpoint_dir)
                # re-read rather than reusing the sweep above: verl advances the tracker right up to
                # the moment the child exits, so a step can become visible during that sweep.
                if self._stop.is_set() and not self._pending():
                    return
                time.sleep(0.5)
        except BaseException as error:
            self._error = error
