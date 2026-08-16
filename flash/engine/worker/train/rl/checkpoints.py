"""Checkpoint upload and gradient verification for the GRPO trainer.

verl writes sharded checkpoints as training proceeds; `_VerlResumeUploader` watches for completed
ones and streams them to the Hub on a background thread so a retry can resume and required saves
are durable. `_check_grpo_had_a_gradient` is the publish-time guard that refuses an adapter
identical to its initialization.

Split out of `flash.engine.worker.rl_train` to keep that module under the file-size limit.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Callable

from flash.engine.worker.backend_common import (
    completed_checkpoint_step,
    stage_verl_resume,
    undiscovered_checkpoint_dirs,
)
from flash.engine.worker.io.heartbeat import join_while_draining
from flash.engine.worker.runtime.pkg_proxy import W as _w
from flash.engine.worker.train.core.checkpoint_lifecycle import CheckpointLedger
from flash.engine.worker.verl.checkpoints import (
    MergeDiskExhaustedError,
    MergeDiskHeadroomError,
    resume_checkpoint_is_loadable,
)


def _rl_train():
    """The orchestrator module, imported lazily because it imports this one.

    `export_peft_adapter`, `stamp_adapter_dir_provenance`, and `_deployable_adapter_on_hf` are
    defined elsewhere but patched on `rl_train` by the upload tests. Binding them here would
    capture the originals, so the patches would rebind an object this module never reads and the
    uploader would perform real exports and Hub reads under test.
    """
    from flash.engine.worker import rl_train

    return rl_train


class _VerlResumeUploader:
    """stream completed verl checkpoints to hf for resume and required saves.

    all checkpoints upload resume state. exact ``save_at_steps`` also exports deployable adapters.
    verl advances ``latest_checkpointed_iteration.txt`` only after writing ``global_step_N``, so the
    marker prevents half-written uploads.
    """

    def __init__(
        self,
        local_dir: str,
        *,
        resume_step: int,
        required_steps: tuple[int, ...] = (),
        export_root: str = "",
        python_bin: str = "",
        model_id: str = "",
        model_revision: str = "",
        preprocessor=None,
        had_gradient: Callable[[], bool] | None = None,
    ) -> None:
        self.local_dir = local_dir
        self.required_steps = frozenset(required_steps)
        self.lifecycle = CheckpointLedger()
        # seed the resumed step's resume state because hf already has it, but do not claim a required
        # step as discovered: a prior worker may have uploaded resume state while the gradient gate
        # withheld its deployable adapter, which this worker must still stage and publish.
        if resume_step:
            self.lifecycle.mark_resume_uploaded(resume_step)
            if resume_step not in self.required_steps:
                self.lifecycle.mark_discovered(resume_step)
        # attempted, NOT uploaded: a step joins this before the upload call so a permanently failing
        # upload cannot respin every 0.5s. durability is the ledger's resume_uploaded fact, which is
        # recorded from the transport's own callback.
        self._resume_attempted: set[int] = {resume_step} if resume_step else set()
        # required step -> staged adapter directory, populated the sweep a checkpoint appears and
        # drained by publication once the gradient gate opens. paths, not lifecycle state: this is
        # what decouples publication from verl's checkpoint retention.
        self.staged_adapters: dict[int, str] = {}
        self.export_root = export_root
        self.python_bin = python_bin
        self.model_id = model_id
        self.model_revision = model_revision
        # the processor on a multimodal job, else the tokenizer: an image model cannot be served
        # from an adapter dir that carries only tokenizer files, since the runtime needs the
        # preprocessor config to turn pixels back into tokens.
        self.preprocessor = preprocessor
        # gate deployable publication on real gradient evidence so zero-gradient runs cannot leave
        # servable per-step adapters before the final guard fails. resume uploads remain allowed because
        # they are internal retry state, not servable artifacts.
        self.had_gradient = had_gradient
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def credit_durable_required_steps(self, resume_step: int) -> None:
        """credit required saves already published before resume.

        verify the adapter on hf rather than trusting the step counter; a preempted worker can pass a
        required step without publishing its deployable artifact.
        """
        for step in sorted(self.required_steps):
            if step <= int(resume_step) and _rl_train()._deployable_adapter_on_hf(step):
                self.lifecycle.mark_deployable_published(step)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        # idempotent: the clean path stops the drain early to surface a missing required save, and
        # the finally block stops it again on every path.
        self._stop.set()
        # bounded by lack of progress, not wall clock: the uploads run on this thread, so a big
        # model's full-state drain can legitimately outlast any constant deadline (VERL-131).
        join_while_draining(self._thread, "verl resume uploader")

    def raise_if_incomplete(self) -> None:
        """fail the run when a required save never became durable.

        called only after training finished cleanly: on a crash or cancel the missing steps are a
        symptom of the real failure, and raising here would mask it.
        """
        if isinstance(self._error, (MergeDiskHeadroomError, MergeDiskExhaustedError)):
            raise self._error
        if self._error is not None:
            raise RuntimeError("verl resume uploader failed") from self._error
        missing = self.lifecycle.missing_deployables(self.required_steps)
        if missing:
            raise RuntimeError(f"required saves were not durably published: {missing}")

    def _deployable_allowed(self) -> bool:
        """return whether gradient evidence permits deployable publication.

        no callback means no gate. callback errors fail closed because this controls durable serving.
        """
        if self.had_gradient is None:
            return True
        try:
            return bool(self.had_gradient())
        except Exception:
            return False

    def _completed_step(self) -> int:
        return completed_checkpoint_step(self.local_dir)

    def _pending(self) -> list[tuple[int, str]]:
        """the completed checkpoint dirs this uploader has not handled yet, oldest first."""
        return undiscovered_checkpoint_dirs(
            self.local_dir, self._completed_step(), self.lifecycle.discovered_steps
        )

    def _stage_deployable(self, step: int, checkpoint_dir: str) -> str:
        """stage a required checkpoint as a servable adapter under ``export_root``.

        stage before the gradient gate because verl may prune ``checkpoint_dir`` while the gate is
        closed. ``export_root`` outlives verl retention; only later publication makes it servable.
        """
        actor_dir = os.path.join(checkpoint_dir, "actor")
        adapter_dir = os.path.join(self.export_root, f"step-{step}")
        shutil.rmtree(adapter_dir, ignore_errors=True)
        os.makedirs(adapter_dir, exist_ok=True)
        _rl_train().export_peft_adapter(
            actor_dir, adapter_dir, base_model_id=self.model_id, python_bin=self.python_bin
        )
        # the served adapter needs its preprocessor alongside it, exactly as the final publish does:
        # the processor on a multimodal job (tokenizer + image preprocessor), else the tokenizer.
        self.preprocessor.save_pretrained(adapter_dir)
        _rl_train().stamp_adapter_dir_provenance(adapter_dir, self.model_id, self.model_revision)
        _w.write_base_model_provenance(adapter_dir, self.model_id, self.model_revision)
        return adapter_dir

    def _publish_staged(self, step: int, adapter_dir: str) -> None:
        """make an already-staged adapter durable and servable."""
        _w.publish_deployable_checkpoint(adapter_dir, step, required=True, _provenance_ready=True)
        self.lifecycle.mark_deployable_published(step)

    def _publish_ready(self) -> None:
        """publish permitted staged steps oldest first.

        staged exports remain publishable after verl prunes their checkpoints. read the gate once so
        evidence appearing during export applies on the same pass.
        """
        if not self._deployable_allowed():
            return
        published = self.lifecycle.deployable_published_steps
        for step in sorted(self.staged_adapters):
            if step not in published:
                self._publish_staged(step, self.staged_adapters[step])

    def _run(self) -> None:
        # a failed resume upload must not fail the run: the policy is still trained and published at
        # the end, and the only loss is having to restart from an earlier step after a preemption.
        # a failed REQUIRED save is different -- the customer asked for a deployable at that step, so
        # it propagates and raise_if_incomplete() turns it into a run failure.
        try:
            while True:
                # sampled before the sweep so a stop() arriving mid-sweep still gets one full pass
                # over the checkpoints it made visible. verl advances
                # latest_checkpointed_iteration.txt right up to the moment the child exits, so the
                # newest resume checkpoint routinely appears in that window; exiting without
                # sweeping it would drop durable work a preemption then has to redo.
                stopping = self._stop.is_set()
                for step, path in self._pending():
                    # stage while verl's checkpoint still exists; publication happens below, once
                    # the gate opens. the two have different deadlines -- the source expires with
                    # verl's retention, the permission arrives with the first varying-reward group --
                    # so a step is staged on the sweep that finds it and published whenever it may be.
                    if (
                        step in self.required_steps
                        and step not in self.lifecycle.deployable_published_steps
                        and step not in self.staged_adapters
                    ):
                        self.staged_adapters[step] = self._stage_deployable(step, path)
                        self.lifecycle.mark_staged(step)
                        # published here, not once the sweep ends: staging a later step can raise,
                        # and the resume upload below can be interrupted by a preemption. either
                        # would leave an already-exported adapter local-only, so each step is made
                        # durable as soon as it is both staged and permitted.
                        self._publish_ready()
                    # resume uploads are ungated internal retry state and may be the only checkpoints before
                    # gradient evidence appears. mark attempts before upload so a permanent failure cannot
                    # spin every 0.5s, preserving the existing non-fatal upload semantics.
                    if step not in self._resume_attempted:
                        self._resume_attempted.add(step)
                        try:
                            _w.upload_resume_checkpoint(
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
                # a step staged while the gate was shut is published by a later sweep, so this runs
                # every pass and not only when something was staged: the gate opens on the first
                # varying-reward group, which is usually a sweep that finds no new checkpoint.
                self._publish_ready()
                # `stopping`, not a fresh read: the sweep above is the full pass over everything
                # stop() had made visible, and only after running it may the loop exit. staged
                # steps still awaiting the gate cannot hold the loop open -- with the gate shut they
                # never clear, and waiting would hang stop(); raise_if_incomplete() reports them.
                if stopping and not self._pending():
                    return
                time.sleep(0.5)
        except BaseException as error:
            self._error = error


def _restore_verl_resume(local_dir: str, *, world_size: int) -> int:
    """stage this run's streamed resume checkpoint into local_dir; return the step it resumes at.

    returns 0 when there is nothing to resume, which is the ordinary fresh-run path, and also when
    ``world_size`` does not match the shards' writer (``stage_verl_resume`` explains why). ``prefer``
    steers the fetch itself toward a lower checkpoint this attempt can load when a higher, later,
    incompatible one also streamed -- without it, a repeated discard would starve the compatible one
    every retry (the remote max-step pick never advances past the checkpoint this attempt rejects).
    """
    resume = _w.hf_resume_checkpoint(
        prefer=lambda path: resume_checkpoint_is_loadable(path, world_size=world_size)
    )
    if not resume:
        return 0
    return stage_verl_resume(resume, local_dir, job_label="GRPO", world_size=world_size)


def _check_grpo_had_a_gradient(
    reward_history: list[float],
    adv_spread_history: list[float],
    *,
    resumed: bool = False,
    already_complete: bool = False,
) -> None:
    """raise unless grpo rewards produced a nonzero policy gradient.

    grpo mean-centres rewards within each group, so equal rewards yield zero advantages and a zero
    gradient despite valid checkpoints and reward means. advantage spread is the decisive signal;
    pg_loss can also be near zero on a valid first step.

    abstain for resumed or already-complete runs because this worker did not observe the earlier
    productive steps. the guard still catches degenerate envs on their first, unresumed attempt.
    """
    if already_complete:
        # no step ran, so there is no metric stream to judge and nothing the parse could have
        # dropped. the restored policy's evidence lives in the worker that produced it.
        return
    if not reward_history:
        raise RuntimeError(
            "verl reported no reward metrics for the whole run — the flash reward bridge was "
            "never consulted (wiring regression); refusing to publish a policy trained on "
            "default rewards"
        )
    if not adv_spread_history:
        # advantages ride the same log line as critic/rewards/mean (both come from one
        # compute_data_metrics dict), so reward metrics without advantage metrics means the parse
        # broke, not that verl chose not to report. treat that as a regression rather than let the
        # spread check below degrade into a guard that cannot fire.
        raise RuntimeError(
            "verl reported reward metrics but no advantage metrics for any step — "
            "critic/advantages/max and /min could not be parsed (metric-format regression); "
            "refusing to publish because the zero-gradient check cannot run"
        )
    if resumed:
        # see the docstring: this worker's history is a suffix of the run's training, so an all-zero
        # suffix is not evidence the run never had a gradient. the parse checks above still apply --
        # they catch a wiring/format regression regardless of where training started.
        return
    # deliberately "no step ever had spread" rather than "some step had none": a run that genuinely
    # converges, or that draws one unlucky all-equal group, legitimately reports zero spread on
    # individual steps, and rejecting those would fail correct runs.
    if not any(spread > 0.0 for spread in adv_spread_history):
        raise RuntimeError(
            f"grpo saw zero advantage spread on all {len(adv_spread_history)} steps — every group's "
            "rewards were identical, so every advantage was 0 and the gradient was exactly 0; the "
            "environment's reward does not discriminate between completions. refusing to publish an "
            "adapter identical to its initialization"
        )
