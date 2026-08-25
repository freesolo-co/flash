"""Execution phases for the verl-backed grpo worker.

Split out of ``flash.engine.worker.train.entry.rl_train`` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import flash.engine.worker.io.heartbeat as _worker_heartbeat
import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.io.wandb_log as _worker_wandb_log
import flash.engine.worker.runtime.state as _worker_state
import flash.engine.worker.train.rl.launch.checkpoints as rl_checkpoints
import flash.engine.worker.train.rl.launch.config as _worker_config
import flash.engine.worker.train.rl.launch.inputs as rl_inputs
import flash.engine.worker.train.rl.launch.verl_config as rl_verl_config
from flash.adapters.targets import resolve_lora_targeting
from flash.core.catalog import get_model
from flash.core.grpo import GRPO_NATIVE_THREAD_ENV
from flash.engine.profiling.sft_workload import _materialize_verl_images
from flash.engine.result.rollout_samples import sample_completion_text, sanitize_rollout_text
from flash.engine.worker.io.heartbeat import (
    GRPO_METRIC_HISTORY_LIMIT,
    LATEST_GRPO_METRICS_LAST,
    RewardObservabilityBuffer,
)
from flash.engine.worker.perf import gpu_diagnostics, wait_for_gpu
from flash.engine.worker.train.core.lifecycle.step_timing import StepTiming
from flash.engine.worker.train.entry.backend_common import (
    _ORPHANED_PIPE_GRACE_S,
    _TEARDOWN_GRACE_S,
    SHIM_FRAGMENT_FAILED_EXIT_CODE,
    ChildOutputTail,
    VerlChildSilenceWatchdog,
    _ChildExitWatchdog,
    adopt_orphaned_descendants,
    append_step_metrics,
    export_peft_adapter,
    kill_process_group,
    latest_global_step_dir,
    parse_verl_metric,
    parse_verl_step_metrics,
    parse_wandb_link,
    raise_for_classified_verl_exit,
    render_sitecustomize_bootstrap,
    resolve_verl_loggers,
    shim_marker_file,
    stamp_adapter_dir_provenance,
    verify_applied_shim_markers,
    verl_device_capability,
    verl_step_number,
)
from flash.engine.worker.train.entry.sft_train import (
    _build_verl_child_env,
    _NvidiaSmiPeakSampler,
)
from flash.engine.worker.train.rl.child.plugin import required_patch_names
from flash.engine.worker.train.rl.rollout.identity import RolloutIdentityLedger
from flash.engine.worker.train.rl.rollout.multi_turn import (
    MultiTurnBridge,
    copy_grpo_child_modules,
    multi_turn_child_env,
    start_reward_server,
)
from flash.engine.worker.train.rl.rollout.reward_module import render_reward_module
from flash.engine.worker.train.rl.rollout.single_turn import (
    score_single_turn,
    score_single_turn_batch,
)
from flash.engine.worker.verl.process_census import GrpoProcessCensus


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


@dataclass
class _StepMetricState:
    progress: dict[str, int] = field(default_factory=lambda: {"step": 0})
    reward_history: list[float] = field(default_factory=list)
    resp_len_history: list[float] = field(default_factory=list)
    loss_curve: list[float] = field(default_factory=list)
    adv_spread_history: list[float] = field(default_factory=list)
    advantage_bounds: dict[int, tuple[float, float]] = field(default_factory=dict)
    grad_norms: dict[int, float] = field(default_factory=dict)
    prior_positive_grad_step: int | None = None
    _metric_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # the last step a previous attempt already completed; 0 for a fresh run. a resumed verl child
    # replays this step's metrics line, which belongs to the earlier attempt, so its bounds are not
    # evidence for the steps THIS attempt executed.
    resume_step: int = 0
    advantage_bounds_evidence: list[dict[str, int | float]] = field(default_factory=list)
    grad_norm_evidence: list[dict[str, int | float]] = field(default_factory=list)
    last_dump_step: list[int] = field(default_factory=lambda: [-1])
    metrics_last: list[dict] = field(default_factory=list)
    step_timing: StepTiming = field(default_factory=StepTiming)
    host_census: dict[str, Any] = field(default_factory=dict)
    rollout_identity_evidence: dict[str, list] = field(
        default_factory=lambda: {"steps": [], "validation": []}
    )
    sent_first_metrics: bool = False
    sent_first_timing: bool = False

    def set_prior_positive_step(self, step: int | None, *, checkpoint_step: int) -> None:
        """install the strict durable positive-gradient fact before uploader start."""
        if step is not None and (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or step > int(checkpoint_step)
        ):
            raise RuntimeError("invalid prior GRPO positive-gradient evidence")
        with self._metric_lock:
            self.prior_positive_grad_step = step

    def record_grad_norm(self, step: int, value: float | None) -> None:
        """replace one current-attempt step's evidence under the uploader's lock."""
        with self._metric_lock:
            self.grad_norms.pop(int(step), None)
            if isinstance(value, float):
                self.grad_norms[int(step)] = value

    def checkpoint_manifest_evidence(self, checkpoint_step: int) -> tuple[bool, int | None]:
        """return readiness and first positive step after metric N has been ingested."""
        checkpoint_step = int(checkpoint_step)
        with self._metric_lock:
            if checkpoint_step <= self.resume_step:
                return True, self.prior_positive_grad_step
            expected = range(self.resume_step + 1, checkpoint_step + 1)
            if any(
                step not in self.grad_norms
                or not math.isfinite(self.grad_norms[step])
                or self.grad_norms[step] < 0.0
                for step in expected
            ):
                return False, None
            current_positive = min(
                (step for step in expected if self.grad_norms[step] > 0.0),
                default=None,
            )
            candidates = [
                step
                for step in (self.prior_positive_grad_step, current_positive)
                if step is not None
            ]
            return True, min(candidates) if candidates else None


@dataclass
class _RewardRuntime:
    observability: RewardObservabilityBuffer
    identity_ledger: RolloutIdentityLedger
    wandb_link: dict[str, str | None]
    multi_turn_bridge: object
    server: object
    reward_url: str


def _prepare_rl_inputs():
    t_start = time.time()
    _worker_heartbeat.heartbeat("rl_start", gpu=gpu_diagnostics())
    wait_for_gpu(
        _worker_state.JOB_SPEC.gpu.type if _worker_state.JOB_SPEC else None,
        gpu_type=_worker_state.JOB_SPEC.gpu.type if _worker_state.JOB_SPEC else "",
    )
    # no setup_perf_backends() here: torch's tf32 flags are per-process state that a subprocess does
    # not inherit, and this process trains nothing -- verl does, out of process. the child opts in
    # from its own sitecustomize instead (render_tf32_shim, wired into shim_source below).

    inp = rl_inputs._resolve_grpo_inputs()
    env, tok = inp["env"], inp["tok"]
    # what gets saved next to a published adapter. a multimodal adapter is unservable without its
    # image preprocessor, so save the whole processor there; a text run saves the tokenizer alone.
    preprocessor = inp["processor"] or tok
    prompts = inp["prompts"]

    # cache the base model before launching verl, then run verl fully offline so its vllm /
    # transformers never hit hf's (rate-limited) api. flash already owns model prefetch; the verl
    # subprocess simply reuses that cache.
    if inp["model_revision"]:
        download_seconds = _worker_hf.prefetch_model(
            inp["model_id"], revision=inp["model_revision"]
        )
    else:
        download_seconds = _worker_hf.prefetch_model(inp["model_id"])
    return t_start, inp, env, tok, preprocessor, prompts, download_seconds


def _prepare_rl_files(inp, prompts):
    # stable int index -> rollout example, exactly as the retired trl path (reward maps back via this).
    ds_rows, rollout_examples = _worker_config.build_grpo_prompt_dataset(prompts)
    message_prompts = [p["prompt"] for p in prompts]
    indices = [int(r["example_idx"]) for r in ds_rows]
    # ground_truth is a verl-schema placeholder only; the reward bridge scores by example_idx
    # against the live env and never reads it.
    ground_truths = [
        str(ex.get("answer", "") or "") if isinstance(ex, dict) else "" for ex in rollout_examples
    ]

    workdir = f"/tmp/rl_train_seed{_worker_state.SEED}"
    os.makedirs(workdir, exist_ok=True)
    local_dir = os.path.join(workdir, "ckpt")
    # a retry reuses the pod workdir; stale global_step_N dirs from a prior attempt would satisfy
    # latest_global_step_dir and publish an old policy as if this attempt trained it.
    shutil.rmtree(local_dir, ignore_errors=True)
    # restore after the wipe, never before: the wipe is what makes a stale local dir safe, and the
    # resume checkpoint is the one global_step_N this attempt is entitled to start from.
    os.makedirs(local_dir, exist_ok=True)
    # the same value `_build_verl_training_cfg` turns into n_gpus, so the width compared here is the
    # width this attempt actually launches verl at. NOT the rented card count: the clamp that bounds
    # the launch by the step's sequences can make those differ, and a mismatch here silently discards
    # a loadable checkpoint and restarts the run from step 0.
    resume_step = rl_checkpoints._restore_verl_resume(
        local_dir,
        world_size=int(inp["dp_cards"]),
        expected_fsdp_generation=inp["fsdp_generation"],
        required_steps=tuple(inp["save_at_steps"]),
    )
    train_pq = os.path.join(workdir, "train.parquet")
    val_pq = os.path.join(workdir, "val.parquet")
    reward_py = os.path.join(workdir, "reward.py")

    # multimodal: decode each prompt's images to png on disk and carry file:// uris in the parquet.
    # verl's dataset loads them through qwen_vl_utils.fetch_image, which reads file:// natively, so
    # the pixels never have to round-trip through arrow. same contract the opd verl path writes.
    image_uris = None
    if inp["multimodal"]:
        image_dir = os.path.join(workdir, "images")
        shutil.rmtree(image_dir, ignore_errors=True)
        image_uris = [
            _materialize_verl_images(
                list(prompt.get("images") or []), inp["package_root"], image_dir, index
            )
            for index, prompt in enumerate(prompts)
        ]

    rows = rl_verl_config.build_verl_dataset_rows(
        message_prompts, indices, ground_truths, image_uris
    )
    rl_verl_config.write_verl_grpo_parquet(rows, train_pq)
    rl_verl_config.write_verl_grpo_parquet(rows[: max(1, min(4, len(rows)))], val_pq)
    with open(reward_py, "w") as f:
        f.write(render_reward_module())

    # runtime patches for the verl interpreter. a stale shim from a prior attempt would otherwise
    # keep patching this one, so the file is rewritten every time.
    shim_dir = os.path.join(workdir, "shim")
    os.makedirs(shim_dir, exist_ok=True)
    # the marker file the child's wrapped fragments append to. a stale one from a prior attempt
    # would prove patches this attempt never applied, so it is removed alongside the rewrite.
    shim_markers = shim_marker_file(shim_dir)
    if os.path.exists(shim_markers):
        os.remove(shim_markers)
    # where each rank records the gpu uuid it opened, so a collision is caught before nccl init.
    # removed for the same reason as the marker file: a stale one from a prior attempt would show
    # ranks claiming devices this attempt never touched, and read as a collision that is not there.
    rank_device_claims = os.path.join(shim_dir, "rank_device_claims.txt")
    if os.path.exists(rank_device_claims):
        os.remove(rank_device_claims)
    return {
        "rollout_examples": rollout_examples,
        "message_prompts": message_prompts,
        "workdir": workdir,
        "local_dir": local_dir,
        "resume_step": resume_step,
        "train_pq": train_pq,
        "val_pq": val_pq,
        "reward_py": reward_py,
        "shim_dir": shim_dir,
        "shim_py": os.path.join(shim_dir, "sitecustomize.py"),
        "shim_markers": shim_markers,
        "rank_device_claims": rank_device_claims,
        "plugin_config_path": os.path.join(shim_dir, "flash_grpo_plugin_config.json"),
    }


def _write_rl_shim(inp, files) -> None:
    """write the minimal startup bootstrap and copy the complete GRPO plugin bundle."""
    with open(files["shim_py"], "w", encoding="utf-8") as file:
        file.write(render_sitecustomize_bootstrap())
    copy_grpo_child_modules(files["shim_dir"])


def _write_rl_plugin_config(inp, files, *, gdn_reset_arch: str | None, loggers) -> None:
    """serialize the final GRPO plugin configuration after capability resolution."""
    targeting = resolve_lora_targeting(
        inp["model_id"], algorithm="grpo", multimodal=bool(inp["multimodal"])
    )
    config = {
        "marker_file": files["shim_markers"],
        "dp_cards": int(inp["dp_cards"]),
        "reentrant_checkpointing": bool(inp["reentrant_checkpointing"]),
        "multimodal": bool(inp["multimodal"]),
        "entropy_quantile": inp["entropy_quantile"],
        "per_turn_credit": bool(inp["per_turn_credit"]),
        "stop_sequences": list(inp["stop_sequences"]),
        "image_pad_token_id": inp["image_pad_token_id"],
        "structured_outputs": inp["structured_outputs"],
        "save_at_steps": list(inp["save_at_steps"]),
        "total_steps": int(inp["steps"]),
        "kl_ref_adapter": bool(inp["warmstart_adapter"]) and float(inp["kl_coef"]) > 0,
        "multi_turn": bool(inp["multi_turn"]),
        "lora_language_prefix": (
            get_model(inp["model_id"]).lora_language_prefix if targeting.exclude_modules else ""
        ),
        "gdn_model_type": gdn_reset_arch,
        "wandb": "wandb" in loggers,
    }
    expected = required_patch_names(config)
    with open(files["plugin_config_path"], "w", encoding="utf-8") as handle:
        json.dump(config, handle, sort_keys=True, separators=(",", ":"))
    files["expected_shims"] = expected


def _prepare_rl_runtime(inp, env, tok, prompts):
    files = _prepare_rl_files(inp, prompts)
    _write_rl_shim(inp, files)
    return files, _start_reward_runtime(inp, env, tok, prompts, files)


def _start_reward_runtime(inp, env, tok, prompts, files) -> _RewardRuntime:
    rollout_examples = files["rollout_examples"]
    message_prompts = files["message_prompts"]
    # the localhost bridge carries rollouts and #607 reward components. generation_size closes each
    # generation when its last scoring call finishes, before a later stdout step line can mix in the
    # next generation. test_freq=-1 and val_before_train=false make every bridged completion training
    # data.
    observability = RewardObservabilityBuffer(
        generation_size=int(inp["prompts_per_step"]) * int(inp["group_size"]),
    )
    identity_ledger = RolloutIdentityLedger(
        int(inp["prompts_per_step"]),
        int(inp["group_size"]),
    )

    def _score_batch(requests: list[tuple[int, str]]) -> list[float]:
        # grade the whole batch before touching the observability lock. the env's scorer may block on
        # judge i/o, while record is intentionally a short per-result critical section.
        with observability.parent_work.busy():
            scored = score_single_turn_batch(
                env,
                [(solution_str, rollout_examples[int(index)]) for index, solution_str in requests],
                tok=tok,
                thinking=bool(_worker_state.THINKING),
                prompt_opened_thinking=inp["prompt_opened_thinking"],
                think_penalty=inp["think_penalty"],
            )
        results = []
        for (index, solution_str), (score, breakdowns) in zip(requests, scored, strict=True):
            observability.record(message_prompts[int(index)], solution_str, score, breakdowns)
            results.append(score)
        return results

    def _score_for_profile(index: int, solution_str: str) -> float:
        """score one bridge request without training observability side effects."""
        with observability.parent_work.busy():
            return score_single_turn(
                env,
                solution_str,
                rollout_examples[int(index)],
                tok=tok,
                thinking=bool(_worker_state.THINKING),
                prompt_opened_thinking=inp["prompt_opened_thinking"],
                think_penalty=inp["think_penalty"],
                raise_on_error=True,
            )

    multi_turn_bridge = (
        MultiTurnBridge(
            env,
            rollout_examples,
            # index-aligned with rollout_examples: build_grpo_prompt_dataset preserves order.
            env_prompts=[p["env_prompt"] for p in prompts],
            max_turns=int(inp["max_turns"]),
            prompt_ids=[p["prompt_ids"] for p in prompts],
            prompt_descriptors=[p.get("images", ()) for p in prompts],
            package_root=inp["package_root"],
            processor=inp["processor"],
            tokenizer=tok,
            thinking=bool(_worker_state.THINKING),
            per_turn_credit=bool(inp["per_turn_credit"]),
            on_episode_scored=observability.record,
            parent_work=observability.parent_work,
            identity_ledger=identity_ledger,
        )
        if inp["multi_turn"]
        else None
    )
    server, reward_url = start_reward_server(
        _score_for_profile,
        example_count=len(rollout_examples),
        multi_turn_bridge=multi_turn_bridge,
        rollout_batch=int(inp["prompts_per_step"]) * int(inp["group_size"]),
        score_batch=None if inp["multi_turn"] else _score_batch,
        identity_ledger=identity_ledger,
    )
    return _RewardRuntime(
        observability=observability,
        identity_ledger=identity_ledger,
        # filled from the child's marker line; stays empty when wandb is off.
        wandb_link={},
        multi_turn_bridge=multi_turn_bridge,
        server=server,
        reward_url=reward_url,
    )


def _resolve_training_settings(inp, caps):
    expected_steps = int(inp["steps"])
    # verl logs from its own interpreter; gate wandb on that env (see resolve_verl_loggers).
    loggers = resolve_verl_loggers(caps)
    spec = _worker_state.JOB_SPEC
    project_name = (spec.wandb.project if spec and spec.wandb else None) or "flash"
    experiment_name = _worker_wandb_log.wandb_run_name()
    # fp8 kv cache on ada/hopper+ (cc>=8.9), matching the sizing math in engine/vram.py. NOT for
    # hybrid linear-attention (GDN) models: vllm's fp8-kv wake path (init_fp8_kv_scales) assumes a
    # plain kv tensor and crashes on the hybrid cache ('list' has no zero_) under verl sleep/wake.
    # resolved from the out-of-process capability probe, never by opening cuda in this parent:
    # a context initialized here to answer one question is retained for the process lifetime, on
    # the same devices the verl child is about to own, which is unbudgeted vram against a reserve
    # sized without it (see fused_ce_backend). an unanswerable probe means conservative bf16 kv.
    verl_cc = verl_device_capability(caps)
    cc_ok = verl_cc is not None and verl_cc >= (8, 9)
    return expected_steps, loggers, project_name, experiment_name, cc_ok


def _initialize_teardown_state():
    # bind the uploader before the try so the finally can always ask whether it was started.
    # verl trains out-of-process, so nvidia-smi is the only reading that covers the child; stopping
    # the sampler in the finally keeps the reading even when the run crashes or is cancelled.
    return None, _NvidiaSmiPeakSampler().start(), None


def _announce_training(t_start: float, cfg) -> tuple[float, float]:
    # the executor budget is sized per run now, so print what this one actually asked for --
    # a vllm init failure reports the demand against the free memory, and the demand is
    # otherwise invisible in the log.
    print(f"[rl-verl] rollout gpu_memory_utilization={cfg['gpu_mem_util']:.4f}", flush=True)
    setup_seconds = time.time() - t_start
    _worker_heartbeat.heartbeat(
        "rl_train_start", setup_seconds=setup_seconds, gpu=gpu_diagnostics()
    )
    return setup_seconds, time.time()


def _start_resume_uploader(
    *, local_dir, resume_step, inp, workdir, python_bin, preprocessor, metric_evidence
):
    targeting = resolve_lora_targeting(
        inp["model_id"], algorithm="grpo", multimodal=bool(inp.get("multimodal"))
    )
    resume_uploader = rl_checkpoints._VerlResumeUploader(
        local_dir,
        resume_step=resume_step,
        metric_evidence=metric_evidence,
        required_steps=inp["save_at_steps"],
        export_root=os.path.join(workdir, "exports"),
        python_bin=python_bin,
        model_id=inp["model_id"],
        model_revision=inp["model_revision"],
        exclude_modules=targeting.exclude_modules,
        preprocessor=preprocessor,
    )
    resume_uploader.credit_durable_required_steps(resume_step)
    resume_uploader.restore_staged_adapters(resume_step)
    resume_uploader.start()
    return resume_uploader


def _build_rl_child_env(inp, files, loggers, reward_url):
    # allowlist the child env because ray fans it to every actor; scoring stays in the parent, so
    # the child needs no platform hf token, github token, or user secrets. FLA_ kernel settings
    # still cross through _CHILD_ENV_PREFIXES.
    env_for_verl = _build_verl_child_env(
        shim_dir=files["shim_dir"], wandb_enabled="wandb" in loggers
    )
    env_for_verl["VERL_USE_EXTERNAL_MODULES"] = "flash_grpo_plugin"
    env_for_verl["FLASH_GRPO_PLUGIN_CONFIG_PATH"] = files["plugin_config_path"]
    env_for_verl["FLASH_VERL_REWARD_URL"] = reward_url
    # where each rank records the gpu it opened. ray fans this env to every actor, which is what
    # makes the file a rendezvous point: the deferred plugin check compares ranks before a process
    # group exists.
    env_for_verl["FLASH_RANK_DEVICE_CLAIMS"] = files["rank_device_claims"]
    # the model is prefetched above; keep the subprocess off hf's rate-limited api.
    env_for_verl["HF_HUB_OFFLINE"] = "1"
    env_for_verl["TRANSFORMERS_OFFLINE"] = "1"
    env_for_verl["HF_HUB_DISABLE_XET"] = "1"
    if inp["multi_turn"]:
        env_for_verl.update(
            multi_turn_child_env(inp, reward_url=reward_url, thinking=bool(_worker_state.THINKING))
        )
    env_for_verl.update(GRPO_NATIVE_THREAD_ENV)
    return env_for_verl


def _step_timing_fields(inp, state: _StepMetricState) -> dict:
    return state.step_timing.heartbeat_fields(
        current_step=state.progress["step"],
        total_steps=int(inp["steps"]),
        remaining_wall_s=_worker_state._remaining_worker_wall_seconds(),
    )


def _ingest_step_metrics(
    line: str,
    inp,
    state: _StepMetricState,
    _reward_observability: Callable[[], dict],
) -> None:
    step_metrics = parse_verl_step_metrics(line)
    step_number = verl_step_number(line)
    pg_loss = parse_verl_metric(line, "actor/pg_loss") if step_number is not None else None
    if step_metrics is not None:
        state.step_timing.record_duration(parse_verl_metric(line, "timing_s/step"))
        # a run constant rather than a verl metric, so it is stamped here from
        # the resolved run config.
        step_metrics["max_completion_tokens"] = inp["max_completion"]
        step_metrics.update(
            {
                f"host_census/{key}": value
                for key, value in state.host_census.items()
                if isinstance(value, int)
            }
        )
        append_step_metrics(state.metrics_last, step_metrics, limit=GRPO_METRIC_HISTORY_LIMIT)
        # the worker's error path reads this global, so a run that dies mid-training
        # still reports the steps it did complete (worker/__init__.py:_err_metrics).
        LATEST_GRPO_METRICS_LAST[:] = state.metrics_last
        heartbeat_fields = _reward_observability()
        has_step_timing = "step_duration_s" in heartbeat_fields
        # rl_train_start arms a 900s throttle, so force until both the first backlog and the first
        # usable timing payload commit. the backlog commit also arms the force floor, so mark the first
        # timing attempt for the wrapper's dedicated floor bypass until that upload succeeds.
        if not state.sent_first_metrics or (has_step_timing and not state.sent_first_timing):
            heartbeat_committed = _worker_heartbeat.heartbeat(
                "rl_step",
                force=True,
                first_timing=has_step_timing and not state.sent_first_timing,
                step=step_metrics["step"],
                metrics_last=list(state.metrics_last),
                **heartbeat_fields,
                gpu=gpu_diagnostics(include_torch=False),
            )
            if heartbeat_committed:
                state.sent_first_metrics = True
                if has_step_timing:
                    state.sent_first_timing = True
        # per-step series for train_meta observability parity. these live on the same
        # line as everything else: verl's only console metric sink is LocalLogger,
        # which always prints "step:N - ..." (verl/utils/logger/aggregate_logger.py),
        # so a line without a step carries no metric to collect.
        for verl_key, sink in (
            ("critic/rewards/mean", state.reward_history),
            ("response_length/mean", state.resp_len_history),
        ):
            value = parse_verl_metric(line, verl_key)
            if value is not None:
                sink.append(value)
    if pg_loss is not None:
        state.loss_curve.append(pg_loss)
    # a structured training row or a renderable actor loss is authoritative for terminal evidence.
    # validation-only rows have neither, and a replay of the resume boundary belongs to the prior worker.
    authoritative_training_line = step_metrics is not None or pg_loss is not None
    if authoritative_training_line and step_number is not None and step_number > state.resume_step:
        state.advantage_bounds.pop(step_number, None)
        grad_norm = None
        if step_metrics is not None:
            grad_norm = step_metrics.get("grad_norm")
            adv_min = step_metrics.get("advantage_min")
            adv_max = step_metrics.get("advantage_max")
            if isinstance(adv_min, float) and isinstance(adv_max, float):
                state.advantage_bounds[step_number] = (adv_min, adv_max)
        state.adv_spread_history[:] = [
            maximum - minimum
            for minimum, maximum in (
                state.advantage_bounds[step] for step in sorted(state.advantage_bounds)
            )
        ]
        state.record_grad_norm(step_number, grad_norm if isinstance(grad_norm, float) else None)


def _execute_rl_child(
    *,
    python_bin,
    overrides,
    env_for_verl,
    inp,
    state,
    reward_runtime,
    _reward_observability,
    files=None,
) -> int:
    # claimed before the child exists, so a grandchild it orphans reparents here and can be
    # reaped at teardown. this process is not pid 1 (the runpod handler is), so without it
    # every wait answers ChildProcessError for a zombie nobody will collect.
    adopt_orphaned_descendants()
    resume_step = int((files or {}).get("resume_step", 0))
    state.resume_step = resume_step
    census = GrpoProcessCensus(
        os.getpid(),
        expected_steps=range(resume_step + 1, int(inp["steps"]) + 1),
    ).start()
    try:
        proc = subprocess.Popen(
            [python_bin, "-m", "flash_grpo_entry", *overrides],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env_for_verl,
            start_new_session=True,
        )
    except BaseException:
        state.host_census = census.stop()
        raise
    child_tail = ChildOutputTail()
    silence_watchdog = VerlChildSilenceWatchdog(
        child_tail,
        baseline_step=int((files or {}).get("resume_step", 0)),
        parent_work=reward_runtime.observability.parent_work,
    )
    child_stream = _GrpoSubprocessStream(
        proc,
        tail=child_tail,
        silence_watchdog=silence_watchdog,
    )
    progress, last_dump_step = state.progress, state.last_dump_step
    # verl replays its resume step's metrics line before producing the first NEW step
    # (`child_io.append_step_metrics` documents the same replay). the identity ledger only
    # registers steps `resume_step + 1 ..` horizon, so sealing the replayed step would raise
    # "no registered rollout identity set" and kill a resumed run at its first output line.
    # seeding the watermark at the resume boundary makes the replay a repeat of an
    # already-dumped step, which this loop skips, exactly as it skips any other repeat.
    if resume_step:
        last_dump_step[0] = resume_step
    shim_markers = (files or {}).get("shim_markers")
    expected_shims = (files or {}).get("expected_shims", ())
    shims_verified = shim_markers is None
    try:
        for line in child_stream:
            print(f"[verl] {line}", end="", flush=True)
            link = parse_wandb_link(line)
            if link is not None:
                reward_runtime.wandb_link.update(link)
            step_number = verl_step_number(line)
            if step_number is not None:
                progress["step"] = step_number
                silence_watchdog.observe_step(step_number)
                # the first step line is the training-start boundary: sitecustomize import is long
                # finished by then, so a marker still missing means this child is training with no
                # flash patch at all, so fail now rather than after the whole run is paid for. not
                # on the first output line: fragments print while later ones are still applying.
                if not shims_verified:
                    verify_applied_shim_markers(shim_markers, expected_shims)
                    shims_verified = True
                # dump one sample completion per new step to the flash log (#607).
                if progress["step"] != last_dump_step[0]:
                    census.sample_step(progress["step"])
                    state.host_census = census.summary()
                    # the generation boundary: verl logs this line once its step is scored, so
                    # everything the reward bridge buffered since the last one is that step's
                    # complete output. seal exact identities before publishing any step output.
                    reward_runtime.identity_ledger.seal(progress["step"])
                    reward_runtime.observability.close_generation(progress["step"])
                    last_dump_step[0] = progress["step"]
                    # asks for THIS step's rows, not merely the newest: when the line is spent on a
                    # generation the queue already dropped, nothing is published and the previous
                    # generation's text would print under this step.
                    samp = reward_runtime.observability.latest_for_step(progress["step"])
                    if samp:
                        _, completion, reward = samp
                        text = sanitize_rollout_text(sample_completion_text(completion))
                        preview = " ".join(text[:300].split())
                        print(
                            f"[rl-verl] step {progress['step']} sample "
                            f"(reward={reward:.3f}): {preview}",
                            flush=True,
                        )
            _ingest_step_metrics(line, inp, state, _reward_observability)
        return child_stream.wait_and_classify()
    except BaseException:
        # the stream loop died (upload error, cancel, oom in the parent): a still-running
        # verl child would keep burning the gpu unattended, so kill its whole process group.
        # this escalates to SIGKILL after the grace period, which a bare SIGTERM does not: a
        # vllm EngineCore that ignores the term keeps its cuda context and strands the gpu
        # for every later job on a reusable worker.
        child_stream.terminate()
        raise
    finally:
        state.host_census = census.stop()
        if state.metrics_last:
            state.metrics_last[-1].update(
                {
                    f"host_census/{key}": value
                    for key, value in state.host_census.items()
                    if isinstance(value, int)
                }
            )
            # multi-turn totals ride the same final metrics row. published in `finally` so a run
            # that dies mid-stream still reports the turns it did execute.
            bridge = getattr(reward_runtime, "multi_turn_bridge", None)
            accounting = bridge.turn_accounting() if bridge is not None else {}
            state.metrics_last[-1].update(
                {
                    f"multi_turn/{key}": value
                    for key, value in accounting.items()
                    if value is not None
                }
            )
            LATEST_GRPO_METRICS_LAST[:] = state.metrics_last


def _finalize_advantage_evidence(state, resume_step: int, expected_steps: int) -> None:
    expected = tuple(range(int(resume_step) + 1, int(expected_steps) + 1))
    actual = tuple(sorted(state.advantage_bounds))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            "GRPO advantage bounds do not cover the executed optimizer steps: "
            f"missing={missing}, extra={extra}"
        )
    rows: list[dict[str, int | float]] = []
    spreads: list[float] = []
    for step in expected:
        minimum, maximum = state.advantage_bounds[step]
        spread = maximum - minimum
        if not all(math.isfinite(value) for value in (minimum, maximum, spread)) or spread < 0.0:
            raise RuntimeError(f"GRPO advantage bounds for step {step} are not finite and ordered")
        row: dict[str, int | float] = {
            "step": step,
            "min": minimum,
            "max": maximum,
            "spread": spread,
        }
        if maximum - minimum != spread:
            raise RuntimeError(f"GRPO advantage spread for step {step} does not match its bounds")
        rows.append(row)
        spreads.append(spread)
    if state.adv_spread_history != spreads:
        raise RuntimeError("GRPO advantage spread history does not match the exact per-step bounds")
    state.advantage_bounds_evidence = rows


def _validate_rl_child(
    rc,
    state,
    resume_step,
    expected_steps,
    resume_uploader,
    *,
    files=None,
    reward_runtime=None,
):
    if rc == SHIM_FRAGMENT_FAILED_EXIT_CODE:
        # the wrapped fragment printed its traceback and named itself before exiting; classify
        # this as permanent, not retriable infra: the same interpreter fails identically on
        # retry. a foreign FLASH_VERL_PYTHON or a drifted verl/transformers is the usual cause.
        raise RuntimeError(
            f"verl.trainer.main_ppo exited {rc}: a required flash runtime patch failed to apply "
            "in the child interpreter (its traceback names the fragment in the flash log). the "
            "verl/transformers stack at the child python is incompatible with this flash "
            "version; rebuild the worker image or fix FLASH_VERL_PYTHON rather than retrying."
        )
    if rc != 0:
        raise RuntimeError(
            f"verl.trainer.main_ppo exited {rc}; see the flash log for the traceback"
        )
    rollout_identity_evidence = None
    if reward_runtime is not None:
        rollout_identity_evidence = reward_runtime.identity_ledger.finalize(
            range(int(resume_step) + 1, int(expected_steps) + 1)
        )
    # belt and braces behind the first-step check in _execute_rl_child: a run that exits 0 without
    # printing a step line (a resume already at the horizon) still may not pass unverified.
    shim_markers = (files or {}).get("shim_markers")
    if shim_markers is not None:
        verify_applied_shim_markers(shim_markers, (files or {}).get("expected_shims", ()))
    # the gradient verdict runs here, ahead of required-save completeness, because a zero-gradient
    # run withholds every required deployable by design. checking completeness first would report
    # the publication symptom instead of the missing actor update evidence that caused it.
    rl_checkpoints._check_grpo_had_a_gradient(
        state.reward_history,
        state.adv_spread_history,
        state.grad_norms,
        resume_step=int(resume_step),
        prior_positive_step=state.prior_positive_grad_step,
        # a resume already at the target runs zero steps and emits zero metrics. publication is
        # allowed only when the restored checkpoint carried trusted prior positive evidence.
        already_complete=bool(resume_step) and resume_step >= expected_steps,
        expected_steps=range(int(resume_step) + 1, int(expected_steps) + 1),
    )
    state.grad_norm_evidence = [
        {"step": step, "grad_norm": state.grad_norms[step]} for step in sorted(state.grad_norms)
    ]
    _finalize_advantage_evidence(state, resume_step, expected_steps)
    if resume_uploader is not None:
        # only the terminal verdict may make staged required adapters servable. open before stop so
        # its final sweep observes the latch and publishes every validated staged checkpoint.
        resume_uploader.allow_deployable_publication()
    # training finished cleanly, so a missing required save is a real defect rather than a
    # side effect of a crash. stop here (not in finally, which suppresses) to surface it.
    # only when exact saves were requested: without them the drain stays best-effort, and
    # letting a slow resume upload raise here would fail an otherwise-successful run.
    if resume_uploader is not None and resume_uploader.required_steps:
        resume_uploader.stop()
        resume_uploader.raise_if_incomplete()
    if rollout_identity_evidence is not None:
        state.rollout_identity_evidence = rollout_identity_evidence


def _prepare_final_adapter(local_dir: str, t_train: float):
    # collect verl's lora checkpoint -> flash-servable peft adapter, then reuse flash finalize.
    out_dir = f"/tmp/rl_seed{_worker_state.SEED}"
    adapter_dir = f"{out_dir}/adapter"
    shutil.rmtree(adapter_dir, ignore_errors=True)
    os.makedirs(adapter_dir, exist_ok=True)
    train_wall = time.time() - t_train
    # the zero-gradient verdict already ran inside the try above, ahead of required-save
    # completeness, so that a withheld deployable reports the reward cause rather than the
    # publication symptom.
    actor_dir, steps_run = latest_global_step_dir(local_dir)
    return actor_dir, adapter_dir, steps_run, train_wall


def _export_final_adapter(actor_dir, adapter_dir, inp, python_bin):
    export_peft_adapter(
        actor_dir, adapter_dir, base_model_id=inp["model_id"], python_bin=python_bin
    )
    targeting = resolve_lora_targeting(
        inp["model_id"], algorithm="grpo", multimodal=bool(inp.get("multimodal"))
    )
    stamp_adapter_dir_provenance(
        adapter_dir,
        inp["model_id"],
        inp["model_revision"],
        exclude_modules=targeting.exclude_modules,
    )
    _worker_hf.write_base_model_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
