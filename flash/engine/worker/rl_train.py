"""verl-backed grpo training path for the fine-tuning worker.

verl runs in a separate interpreter because its torch/vllm pins conflict with flash. rewards
cross a localhost rpc bridge into flash's live env.

supports non-tool, single/multi-turn, text/multimodal grpo. function-calling tool envs raise.
"""

from __future__ import annotations

import contextlib
import os
import subprocess

# several imports below are marked unused-ok on purpose: their call sites moved to
# `.train.rl.inputs`, but tests still read or patch them as attributes of THIS module and the
# resolver reads them back through it. an autofix that drops them breaks those tests.
from flash.engine.plan.steps import (
    final_save_due,
    on_policy_steps,  # noqa: F401
    resolve_update_horizon,  # noqa: F401
)
from flash.engine.worker.backend_common import (  # noqa: F401
    _ORPHANED_PIPE_GRACE_S,
    _TEARDOWN_GRACE_S,
    VERL_REQUIREMENT,
    ChildOutputTail,
    VerlChildSilenceWatchdog,
    _ChildExitWatchdog,
    clamp_engine_len,
    export_peft_adapter,
    fused_ce_backend,
    gdn_probe_module,
    kill_process_group,
    model_max_position_embeddings,
    probe_verl_capabilities,
    raise_for_classified_verl_exit,
    reap_stragglers,
    require_gdn_boundary_resets,
    resolve_blackwell_attention_backends,
    resolve_rollout_enforce_eager,
    resolve_verl_loggers,
    resolve_verl_python,
    rollout_fp8_kv,
    stamp_adapter_dir_provenance,
    verl_declares_rollout_field,
    verl_device_capability,
)
from flash.engine.worker.io.heartbeat import liveness_heartbeat

# no call site in this module since the uploader moved to `.train.rl.checkpoints`, but it is kept
# imported here on purpose: the resume tests patch `rl_train._deployable_adapter_on_hf`, and the
# uploader reads it back through this module. removing it as unused would silently break them.
from flash.engine.worker.io.hf import _deployable_adapter_on_hf  # noqa: F401
from flash.engine.worker.model.packing import model_is_gdn_hybrid
from flash.engine.worker.perf import gpu_diagnostics
from flash.engine.worker.runtime.pkg_proxy import W as _w
from flash.engine.worker.runtime.rng import seed_training_rngs  # noqa: F401
from flash.engine.worker.sft_train import (  # noqa: F401
    _build_verl_child_env,
    _cached_model_path,
    _NvidiaSmiPeakSampler,
)

# kept here purely as the patch surface the resolver tests replace: `_resolve_grpo_inputs` moved
# to `train/rl/inputs.py` and calls it back through this module, so the name must resolve here.
from flash.engine.worker.train.opd.gkd import (  # noqa: F401
    generation_eos_from_cached_config,
)
from flash.engine.worker.train.rl.reward_module import render_reward_module  # noqa: F401

DATA_SOURCE = "flash_env"


# --------------------------------------------------------------------------------------------
# pure helpers (no verl, no gpu, no network) -- unit tested directly.
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# orchestration.
# --------------------------------------------------------------------------------------------


class _GrpoSubprocessStream:
    """one grpo child stream and the evidence latched from that same stream."""

    def __init__(self, proc, *, tail=None, silence_watchdog=None) -> None:
        self._proc = proc
        # the caller uses start_new_session, so the leader pid remains the group's stable identity.
        self._process_group_id = proc.pid
        self._tail = tail if tail is not None else ChildOutputTail()
        self._silence_watchdog = silence_watchdog
        self._terminated = False
        self._orphaned_pipe = False

    def __iter__(self):
        assert self._proc.stdout is not None
        # the child's exit is watched independently of pipe EOF: verl's vllm EngineCore grandchild
        # inherits this same pipe, so a trainer that dies while it lives leaves a pipe nobody will
        # close and this loop would run forever on a paid gpu. the watchdog tears the group down,
        # which frees the cuda context AND closes the pipe, ending this loop.
        if self._silence_watchdog is not None:
            self._silence_watchdog.bind(
                child_alive=lambda: self._proc.poll() is None,
                teardown=self.terminate,
            )
            self._silence_watchdog.start()
        try:
            with _ChildExitWatchdog(
                self._proc, process_group_id=self._process_group_id, grace_s=_ORPHANED_PIPE_GRACE_S
            ) as watchdog:
                for line in self._proc.stdout:
                    # held across the yield. this is a generator, so the consumer's work for a line
                    # runs while suspended here; counting only arrival makes a long callback look idle.
                    with watchdog.handling_line():
                        self._tail.record(line)
                        if self._silence_watchdog is not None:
                            self._silence_watchdog.observe_line(line)
                        yield line
        finally:
            if self._silence_watchdog is not None:
                self._silence_watchdog.stop()
        if watchdog.tore_down:
            self._orphaned_pipe = True
            # the group is already gone, so teardown must not be attempted a second time.
            self._terminated = True

    def terminate(self) -> None:
        if self._terminated:
            return
        kill_process_group(self._proc, process_group_id=self._process_group_id)
        self._terminated = True

    def wait_and_classify(self) -> int:
        # BOUNDED. the iterator above ends on stdout EOF, which is not the direct child's exit: verl
        # spawns vllm's EngineCore as a grandchild holding this same merged pipe, so a trainer that
        # dies while the EngineCore lives keeps the pipe open and an unbounded wait parks the
        # attempt on a paid gpu. a child that has not exited within the grace after its own stdout
        # closed is not going to, so tear the group down; `terminate` waits on it.
        try:
            return_code = int(self._proc.wait(timeout=_TEARDOWN_GRACE_S))
        except subprocess.TimeoutExpired:
            self.terminate()
            # `terminate` waits, so the child is normally collected by now. it is still not
            # guaranteed -- a member wedged in uninterruptible io outlives even the SIGKILL -- and a
            # survivor is a failure however the exit reads, so an uncollected child is reported as
            # one rather than defaulted to zero.
            collected = self._proc.returncode
            return_code = int(collected) if collected is not None else 1
        try:
            raise_for_classified_verl_exit(return_code, self._tail)
            if self._silence_watchdog is not None:
                self._silence_watchdog.raise_if_failed()
        except BaseException:
            self.terminate()
            raise
        if return_code != 0:
            # an unclassified nonzero exit RETURNS from the classifier rather than raising, so this
            # is the failing path that reached no teardown. the direct child is gone but its group
            # need not be, and a surviving EngineCore strands the gpu for the next attempt.
            self.terminate()
        if self._orphaned_pipe and return_code == 0:
            # the trainer exited 0 but a descendant held the pipe open past the grace, so the group
            # was killed to release it. the trainer's status says nothing about that descendant, and
            # returning 0 here would publish a partial run as a completed one.
            raise RuntimeError(
                f"verl subprocess {self._proc.pid} exited 0 but a descendant held its output pipe "
                f"open for {_ORPHANED_PIPE_GRACE_S:.0f}s; the process group was torn down to "
                "release the gpu"
            )
        return return_code


def _write_terminal_metadata(
    *,
    inp,
    prompts,
    adapter_dir,
    steps_run,
    train_wall,
    setup_seconds,
    state,
    resume_step,
    download_seconds,
    device_peak_gpu_gb,
    fp8_kv,
    project_name,
    loggers,
    experiment_name,
    reward_runtime,
    gdn_hybrid,
):
    metrics_last = state.metrics_last
    _w.heartbeat(
        "rl_trained",
        train_wall=train_wall,
        step=steps_run,
        gpu=gpu_diagnostics(),
        metrics_last=list(metrics_last),
    )
    _w.write_train_meta(
        phase="rl",
        adapter_dir=adapter_dir,
        model_id=inp["model_id"],
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=0,
        # carry the completed step, as the sft and opd finalizers already do. without it the
        # rl_train_done and done heartbeats land stepless and overwrite the stepped rl_trained ping
        # above, so a cancel arriving between here and DONE reprices a fully trained run at zero
        # steps (actual_steps_run returns 0 for a non-training stage with no step).
        step=steps_run,
        framework_init_seconds=state.framework_init_seconds,
        reward_seconds=state.reward_seconds,
        heartbeat_fields={"metrics_last": list(metrics_last)},
        generated_tokens=int(
            sum(state.resp_len_history)
            / max(1, len(state.resp_len_history))
            * steps_run
            * inp["prompts_per_step"]
            * inp["group_size"]
        )
        if state.resp_len_history
        else 0,
        notes=_build_verl_train_notes(
            inp,
            steps_run=steps_run,
            retained_prompts=len(prompts),
            reward_history=state.reward_history,
            loss_curve=state.loss_curve,
            resumed=bool(resume_step),
            download_seconds=download_seconds,
            device_peak_gpu_gb=device_peak_gpu_gb,
            fp8_kv=fp8_kv,
            wandb_project=project_name if "wandb" in loggers else None,
            wandb_run_name=experiment_name if "wandb" in loggers else None,
            wandb_url=reward_runtime.wandb_link.get("wandb_url"),
            wandb_id=reward_runtime.wandb_link.get("wandb_id"),
            reward_bridge_batching=not inp["multi_turn"],
            gdn_boundary_resets=gdn_hybrid or None,
            host_census=state.host_census,
            rollout_identity_evidence=state.rollout_identity_evidence,
            advantage_spread_history=state.adv_spread_history,
            advantage_bounds=state.advantage_bounds_evidence,
            multi_turn_accounting=(
                reward_runtime.multi_turn_bridge.turn_accounting()
                if reward_runtime.multi_turn_bridge is not None
                else None
            ),
        ),
    )


def _configure_rl_child(
    *, inp, files, model_path_for_verl, t_start, python_bin, gdn_hybrid, gdn_module, caps
):
    # masking truncated completions is a fork-only rollout field. FLASH_VERL_PYTHON can point at a
    # stock verl, and hydra would compose the unknown key only to abort in dataclass conversion.
    # fail here with the cause instead, and never silently train on truncated completions.
    if inp["mask_truncated_completions"] and not verl_declares_rollout_field(
        caps, "mask_truncated_completions"
    ):
        raise RuntimeError(
            f"grpo requested mask_truncated_completions but the verl at {python_bin} does not "
            "support it. the worker image's baked interpreter predates the freesolo fork; "
            f"rebuild the worker image with '{VERL_REQUIREMENT}' installed."
        )
    # raises when a gdn child cannot honor resets: the padded fallback cannot complete a step on
    # verl's fsdp engine, so the final plugin config must carry the proven model type or stop here.
    gdn_reset_arch = require_gdn_boundary_resets(caps, gdn_module)
    expected_steps, loggers, project_name, experiment_name, cc_ok = _resolve_training_settings(
        inp, caps
    )
    _write_rl_plugin_config(
        inp,
        files,
        gdn_reset_arch=gdn_reset_arch,
        loggers=loggers,
    )
    # reuse the gdn answer resolved above rather than re-probing; see gdn_hybrid. a gdn hybrid keeps
    # fp8 only when the catalog pins its engine resident -- see rollout_fp8_kv for why sleep is the
    # thing that actually crashes, not gdn.
    fp8_kv = rollout_fp8_kv(cc_ok, gdn_hybrid, inp["model_id"])
    # one capability probe, both rollout decisions below. asked of the verl interpreter, whose
    # torch/vllm stack is the one that has to run the rollout.
    verl_cc = verl_device_capability(caps)
    # blackwell needs both rollout attention backends pinned; vllm 0.19.1's own defaults pick
    # flash-attn, which is PTX-unreliable on sm120 (silent empty rollouts) and routes the ViT
    # into an unimportable CUTE kernel on sm100/sm120. no-op off blackwell.
    attention_backend, mm_encoder_attn_backend = resolve_blackwell_attention_backends(caps, verl_cc)
    # sm86 is the one arch whose vllm 0.19.1 graph capture is a measured failure (completions
    # repeat to the token cap without emitting EOS), so only it runs the rollout eagerly. see
    # resolve_rollout_enforce_eager for the per-arch evidence and why one knob is enough.
    enforce_eager = resolve_rollout_enforce_eager(verl_cc)
    cfg = _build_verl_training_cfg(
        inp,
        train_files=files["train_pq"],
        val_files=files["val_pq"],
        model_path=model_path_for_verl,
        thinking=bool(_w.THINKING),
        loggers=loggers,
        fp8_kv=fp8_kv,
        enforce_eager=enforce_eager,
        attention_backend=attention_backend,
        mm_encoder_attn_backend=mm_encoder_attn_backend,
        reward_path=files["reward_py"],
        local_dir=files["local_dir"],
        project_name=project_name,
        experiment_name=experiment_name,
        gpu_type=(_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else ""),
        # the authored knobs the allocator sized this run's shape from, so the zero-2 gate the
        # worker applies is the same one the allocator admitted the shape under.
        train_spec=(_w.JOB_SPEC.train if _w.JOB_SPEC else None),
        # the ranks verl will actually run, not the cards rented: with ulysses pinned off every rank
        # is a dp rank, and verl chunks the step's sequences across them with an exact-divisibility
        # assert. a wider launch than the sequences divide aborts at step 0 on a paid box. resolved
        # in `_resolve_grpo_inputs` so the resume probe compares against this same width.
        n_gpus=int(inp["dp_cards"]),
        # resolved from the out-of-process capability probe, never by opening cuda in this
        # parent -- see fused_ce_backend.
        ce_backend=fused_ce_backend(caps),
    )
    setup_seconds, t_train = _announce_training(t_start, cfg)
    return {
        "expected_steps": expected_steps,
        "loggers": loggers,
        "project_name": project_name,
        "experiment_name": experiment_name,
        "fp8_kv": fp8_kv,
        "overrides": build_verl_overrides(cfg),
        "setup_seconds": setup_seconds,
        "t_train": t_train,
    }


def _shutdown_rl_runtime(resume_uploader, gpu_sampler, reward_runtime) -> float | None:
    # drain before the reward server goes down: on a cancel or crash the last completed checkpoint
    # is exactly the one a retry needs, so it is worth uploading on the way out.
    if resume_uploader is not None:
        with contextlib.suppress(Exception):
            resume_uploader.stop()
    device_peak_gpu_gb = None
    with contextlib.suppress(Exception):
        device_peak_gpu_gb = gpu_sampler.stop_gb()
    # bridge first: the scoring thread is what the server's routes block on, so stopping the server
    # before it would strand a scoring episode on an event nothing will ever set.
    if reward_runtime.multi_turn_bridge is not None:
        with contextlib.suppress(Exception):
            reward_runtime.multi_turn_bridge.shutdown()
    reward_runtime.server.shutdown()
    # every job boundary, not just the failing ones. `kill_process_group` runs on exceptions alone
    # here, so a straggler an earlier teardown sigkilled but could not drain in time would otherwise
    # be collected only by the next failing job.
    with contextlib.suppress(Exception):
        reap_stragglers()
    return device_peak_gpu_gb


def run_rl_train():
    """grpo training on verl, output-compatible with run_rl. see module docstring for scope."""
    # env load, prompt render and tokenization run for minutes on a large split and emit nothing of
    # their own; without the wrap the provider sees silence from rl_start until rl_train_start.
    # (the warm-start adapter pull nested inside carries its own rl_adapter_loading wrap.)
    with liveness_heartbeat("rl_data_loading"):
        t_start, inp, env, tok, preprocessor, prompts, download_seconds = _prepare_rl_inputs()
    # verl resolves model.path with HF_HUB_OFFLINE=1, so pass a real snapshot directory. bare ids can
    # miss pinned revisions or cache symlinks and fail after gpu rental. _cached_model_path handles
    # both and raises RetriableInfraError so the run can move to a healthy worker.
    model_path_for_verl = _cached_model_path(inp["model_id"], inp["model_revision"])
    files, reward_runtime = _prepare_rl_runtime(inp, env, tok, prompts)

    resume_uploader, gpu_sampler, device_peak_gpu_gb = _initialize_teardown_state()
    try:
        # provisioning and the cold verl capability probe can take minutes without step output. keep
        # liveness running so the stall watchdog does not fail healthy setup (#442). there is no
        # monotonic progress counter here, only the keepalive.
        with liveness_heartbeat("rl_configuring"):
            python_bin = resolve_verl_python(
                files["workdir"], install_wandb=bool(os.environ.get("WANDB_API_KEY"))
            )
            # gdn boundary resets need the child to have fla + causal_conv1d. resolve the model here
            # because the answer requires a hub/cache read the child must not repeat.
            gdn_hybrid = model_is_gdn_hybrid(inp["model_id"], inp["model_revision"])
            gdn_module = (
                gdn_probe_module(inp["model_id"], inp["model_revision"]) if gdn_hybrid else ""
            )
            # one child answers every independent capability question. each used to cost its own
            # interpreter, and the torch/verl import -- not the question -- was the price.
            caps = probe_verl_capabilities(python_bin, gdn_module)
        configured = _configure_rl_child(
            inp=inp,
            files=files,
            model_path_for_verl=model_path_for_verl,
            t_start=t_start,
            python_bin=python_bin,
            gdn_hybrid=gdn_hybrid,
            gdn_module=gdn_module,
            caps=caps,
        )
        expected_steps, loggers = configured["expected_steps"], configured["loggers"]
        _w.heartbeat("rl_step", step=0, initial=True)
        state = _StepMetricState(train_started_at=configured["t_train"])
        # equal within-group rewards produce zero advantages and gradients. collect per-step spread;
        # reward mean and pg_loss cannot prove a signal. declare this before the uploader because its
        # publication gate closes over the histories.
        adv_spread_history = state.adv_spread_history
        resume_uploader = _start_resume_uploader(
            local_dir=files["local_dir"],
            resume_step=files["resume_step"],
            inp=inp,
            workdir=files["workdir"],
            python_bin=python_bin,
            preprocessor=preprocessor,
            adv_spread_history=adv_spread_history,
        )
        env_for_verl = _build_rl_child_env(inp, files, loggers, reward_runtime.reward_url)
        metrics_last = state.metrics_last

        def _progress():
            return state.progress["step"]

        def _reward_observability() -> dict:
            """return reward observability and measured pace for one heartbeat."""
            return {
                **reward_runtime.observability.heartbeat_fields(),
                **_step_timing_fields(inp, state),
            }

        with liveness_heartbeat(
            "rl_step",
            progress=_progress,
            fields=lambda: {"metrics_last": list(metrics_last), **_reward_observability()},
            progress_step=True,
            sample_off_thread=True,
        ):
            rc = _execute_rl_child(
                python_bin=python_bin,
                overrides=configured["overrides"],
                env_for_verl=env_for_verl,
                inp=inp,
                state=state,
                reward_runtime=reward_runtime,
                _reward_observability=_reward_observability,
                files=files,
            )
        _validate_rl_child(
            rc,
            state,
            files["resume_step"],
            expected_steps,
            resume_uploader,
            files=files,
            reward_runtime=reward_runtime,
        )
    finally:
        device_peak_gpu_gb = _shutdown_rl_runtime(resume_uploader, gpu_sampler, reward_runtime)

    actor_dir, adapter_dir, steps_run, train_wall = _prepare_final_adapter(
        files["local_dir"], configured["t_train"]
    )
    if steps_run < expected_steps:
        raise RuntimeError(
            f"grpo completed {steps_run}/{expected_steps} requested optimizer updates"
        )
    with liveness_heartbeat(
        "rl_finalizing",
        progress=lambda: steps_run,
        fields=lambda: {"metrics_last": list(metrics_last)},
        progress_step=True,
        keepalive=True,
    ):
        _export_final_adapter(actor_dir, adapter_dir, inp, python_bin)
        preprocessor.save_pretrained(adapter_dir)
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        # preserve the final checkpoint only when exact save steps are not configured: with
        # save_at_steps set the customer asked for those steps and nothing else.
        if final_save_due(steps_run, inp["save_at_steps"]):
            _w.publish_deployable_checkpoint(adapter_dir, steps_run)
    _write_terminal_metadata(
        inp=inp,
        prompts=prompts,
        adapter_dir=adapter_dir,
        steps_run=steps_run,
        train_wall=train_wall,
        setup_seconds=configured["setup_seconds"],
        state=state,
        resume_step=files["resume_step"],
        download_seconds=download_seconds,
        device_peak_gpu_gb=device_peak_gpu_gb,
        fp8_kv=configured["fp8_kv"],
        project_name=configured["project_name"],
        loggers=loggers,
        experiment_name=configured["experiment_name"],
        reward_runtime=reward_runtime,
        gdn_hybrid=gdn_hybrid,
    )


# single-turn scoring, implemented in `.train.rl.single_turn`.
# verl config rendering, implemented in `.train.rl.verl_config`.
# the parent side of the multi-turn rollout loop, implemented in `.train.rl.multi_turn`.
# checkpoint upload and the zero-gradient publish guard, implemented in `.train.rl.checkpoints`.
# imported at the BOTTOM because that module reaches back here for the export helpers the upload
# tests patch on this module, so a top-level import would be circular.
from flash.engine.worker.rl_train_runner import (  # noqa: E402,F401
    _announce_training,
    _build_rl_child_env,
    _execute_rl_child,
    _export_final_adapter,
    _ingest_step_metrics,
    _initialize_teardown_state,
    _prepare_final_adapter,
    _prepare_rl_files,
    _prepare_rl_inputs,
    _prepare_rl_runtime,
    _resolve_training_settings,
    _RewardRuntime,
    _start_resume_uploader,
    _start_reward_runtime,
    _step_timing_fields,
    _StepMetricState,
    _validate_rl_child,
    _write_rl_plugin_config,
    _write_rl_shim,
)
from flash.engine.worker.train.rl.checkpoints import (  # noqa: E402,F401
    _check_grpo_had_a_gradient,
    _restore_verl_resume,
    _VerlResumeUploader,
)

# re-exported so `rl_train._resolve_grpo_inputs` keeps working: the resolver is called and
# source-inspected through this module by a large number of tests.
from flash.engine.worker.train.rl.inputs import _resolve_grpo_inputs  # noqa: E402,F401
from flash.engine.worker.train.rl.multi_turn import (  # noqa: E402,F401
    _MULTI_TURN_SCORE_BATCH_SIZE,
    _MULTI_TURN_SCORE_FLUSH_WAIT_S,
    _MULTI_TURN_SCORE_SHUTDOWN_WAIT_S,
    _MULTI_TURN_SESSION_LEASE_S,
    _REWARD_BRIDGE_MIN_REQUEST_BACKLOG,
    _SINGLE_TURN_SCORE_BATCH_SIZE,
    _SINGLE_TURN_SCORE_FLUSH_WAIT_S,
    _SINGLE_TURN_SCORE_SHUTDOWN_WAIT_S,
    GRPO_CHILD_MODULES,
    MultiTurnBridge,
    copy_grpo_child_modules,
    multi_turn_child_env,
    start_reward_server,
)
from flash.engine.worker.train.rl.single_turn import (  # noqa: E402,F401
    _finalize_single_turn_reward,
    _single_turn_scoring_state,
    score_single_turn,
    score_single_turn_batch,
)
from flash.engine.worker.train.rl.verl_config import (  # noqa: E402,F401
    _DEFAULT_GPU_MEM_UTIL,
    _build_verl_train_notes,
    _build_verl_training_cfg,
    _processor_expanded_prompt,
    _verl_epochs_for_horizon,
    _verl_grpo_parquet_features,
    build_verl_dataset_rows,
    build_verl_overrides,
    resolve_gpu_mem_util,
    write_verl_grpo_parquet,
)
