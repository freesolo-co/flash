"""Execution phases for the verl-backed grpo worker.

Split out of ``flash.engine.worker.rl_train`` to keep that module under the file-size limit.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from flash.engine.profiling.sft_workload import _materialize_verl_images
from flash.engine.result.rollout_samples import sample_completion_text, sanitize_rollout_text
from flash.engine.worker.backend_common import (
    SHIM_FRAGMENT_FAILED_EXIT_CODE,
    ChildOutputTail,
    VerlChildSilenceWatchdog,
    adopt_orphaned_descendants,
    append_step_metrics,
    latest_global_step_dir,
    parse_verl_metric,
    parse_verl_step_metrics,
    parse_wandb_link,
    render_shim_marker_prologue,
    render_tf32_shim,
    render_tilelang_cudart_shim,
    render_wandb_link_shim,
    shim_marker_file,
    verify_applied_shim_markers,
    verl_device_capability,
    wrap_shim_fragment,
)
from flash.engine.worker.io.heartbeat import (
    _LIVENESS_TICK_S,
    GRPO_METRIC_HISTORY_LIMIT,
    LATEST_GRPO_METRICS_LAST,
    RewardObservabilityBuffer,
)
from flash.engine.worker.perf import gpu_diagnostics, wait_for_gpu
from flash.engine.worker.runtime.pkg_proxy import W as _w
from flash.engine.worker.train.rl.multi_turn import (
    MultiTurnBridge,
    copy_multi_turn_child_modules,
    multi_turn_child_env,
    start_reward_server,
)
from flash.engine.worker.train.rl.shims import (
    render_entropy_quantile_shim,
    render_exact_save_steps_shim,
    render_image_pad_ban_shim,
    render_kl_ref_adapter_shim,
    render_per_turn_credit_shim,
    render_reentrant_checkpointing_shim,
    render_reward_module,
    render_stop_sequences_shim,
    render_structured_outputs_shim,
)
from flash.engine.worker.train.rl.single_turn import (
    _log_reward_profile,
    score_single_turn,
    score_single_turn_batch,
)


def _rl_train():
    """return the parent module so runtime monkeypatch seams stay live."""
    from flash.engine.worker import rl_train

    return rl_train


@dataclass
class _StepMetricState:
    progress: dict[str, int] = field(default_factory=lambda: {"step": 0})
    reward_history: list[float] = field(default_factory=list)
    resp_len_history: list[float] = field(default_factory=list)
    loss_curve: list[float] = field(default_factory=list)
    adv_spread_history: list[float] = field(default_factory=list)
    last_dump_step: list[int] = field(default_factory=lambda: [-1])
    step_line_times: list[float] = field(default_factory=list)
    metrics_last: list[dict] = field(default_factory=list)
    sent_first_metrics: bool = False
    # the child's output tail and the watchdog reading it are per-run state like everything above,
    # so they travel with it rather than as two more parameters through the child call.
    child_tail: ChildOutputTail = field(default_factory=ChildOutputTail)
    silence_watchdog: VerlChildSilenceWatchdog | None = None
    # grpo's rewards are scored PARENT-side over the localhost bridge, so a child waiting on a slow
    # user scorer or a judge api prints nothing while real work is happening. without this the
    # watchdog cannot tell that from a wedge, exactly as opd cannot without its teacher counter.
    reward_activity: Callable[[], int] | None = None
    # and the counter alone is not enough: one coalesced grading call can outlast the silence budget
    # with nothing recorded until it returns, so the watchdog also asks whether one is in flight.
    reward_busy: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        self.silence_watchdog = VerlChildSilenceWatchdog(
            self.child_tail,
            tick_s=_LIVENESS_TICK_S,
            parent_activity=self.reward_activity,
            parent_busy=self.reward_busy,
        )

    @classmethod
    def for_reward_runtime(cls, reward_runtime: _RewardRuntime) -> _StepMetricState:
        """build the per-run state wired to the parent-side reward signals the watchdog needs.

        both probes come from the same buffer and are only ever passed together, so binding them
        here keeps the pair from being split up -- supplying one without the other reintroduces
        exactly the blind spot the other closes.
        """
        return cls(
            reward_activity=reward_runtime.observability.reward_activity_count,
            reward_busy=reward_runtime.observability.reward_grading_in_flight,
        )


@dataclass
class _RewardRuntime:
    observability: RewardObservabilityBuffer
    wandb_link: dict[str, str | None]
    reward_profile: object
    multi_turn_bridge: object
    server: object
    reward_url: str


def _prepare_rl_inputs():
    t_start = time.time()
    _w.heartbeat("rl_start", gpu=gpu_diagnostics())
    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        gpu_type=_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else "",
    )
    # no setup_perf_backends() here: torch's tf32 flags are per-process state that a subprocess does
    # not inherit, and this process trains nothing -- verl does, out of process. the child opts in
    # from its own sitecustomize instead (render_tf32_shim, wired into shim_source below).

    inp = _rl_train()._resolve_grpo_inputs()
    env, tok = inp["env"], inp["tok"]
    # what gets saved next to a published adapter. a multimodal adapter is unservable without its
    # image preprocessor, so save the whole processor there; a text run saves the tokenizer alone.
    preprocessor = inp["processor"] or tok
    prompts = inp["prompts"]

    # cache the base model before launching verl, then run verl fully offline so its vllm /
    # transformers never hit hf's (rate-limited) api. flash already owns model prefetch; the verl
    # subprocess simply reuses that cache.
    if inp["model_revision"]:
        download_seconds = _w.prefetch_model(inp["model_id"], revision=inp["model_revision"])
    else:
        download_seconds = _w.prefetch_model(inp["model_id"])
    return t_start, inp, env, tok, preprocessor, prompts, download_seconds


def _prepare_rl_files(inp, prompts):
    # stable int index -> rollout example, exactly as the retired trl path (reward maps back via this).
    ds_rows, rollout_examples = _w.build_grpo_prompt_dataset(prompts)
    message_prompts = [p["prompt"] for p in prompts]
    indices = [int(r["example_idx"]) for r in ds_rows]
    # ground_truth is a verl-schema placeholder only; the reward bridge scores by example_idx
    # against the live env and never reads it.
    ground_truths = [
        str(ex.get("answer", "") or "") if isinstance(ex, dict) else "" for ex in rollout_examples
    ]

    workdir = f"/tmp/rl_train_seed{_w.SEED}"
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
    resume_step = _rl_train()._restore_verl_resume(local_dir, world_size=int(inp["dp_cards"]))
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

    rows = _rl_train().build_verl_dataset_rows(message_prompts, indices, ground_truths, image_uris)
    _rl_train().write_verl_grpo_parquet(rows, train_pq)
    _rl_train().write_verl_grpo_parquet(rows[: max(1, min(4, len(rows)))], val_pq)
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
    }


def _write_rl_shim(inp, files) -> list[str]:
    """write the child sitecustomize; return the marker names it must prove applied.

    every correctness-critical fragment is wrapped fail-closed (see wrap_shim_fragment): cpython's
    execsitecustomize would otherwise swallow a fragment's exception and let the child train
    unpatched, with every fragment after the failing one silently skipped too.
    """
    # one sitecustomize holds every patch: python imports it once, so a second file would never be
    # loaded. each feature renderer returns "" when its feature is off; the tf32 fragment is
    # unconditional, so this source is never empty.
    required_fragments = [
        (
            "reentrant-checkpointing",
            render_reentrant_checkpointing_shim(
                inp["reentrant_checkpointing"], multimodal=bool(inp["multimodal"])
            ),
        ),
        ("entropy-quantile", render_entropy_quantile_shim(inp["entropy_quantile"])),
        ("per-turn-credit", render_per_turn_credit_shim(inp["per_turn_credit"])),
        ("stop-sequences", render_stop_sequences_shim(inp["stop_sequences"])),
        ("image-pad-ban", render_image_pad_ban_shim(inp["image_pad_token_id"])),
        ("structured-outputs", render_structured_outputs_shim(inp["structured_outputs"])),
        ("exact-save-steps", render_exact_save_steps_shim(inp["save_at_steps"], inp["steps"])),
        (
            "kl-ref-adapter",
            render_kl_ref_adapter_shim(
                bool(inp["warmstart_adapter"]) and float(inp["kl_coef"]) > 0
            ),
        ),
    ]
    shim_source = "".join(
        part
        for part in (
            # first: torch's matmul flags are process-wide state, and reading them back is how the
            # rest of the child sees the choice. nothing below depends on it, but a later fragment
            # that raised would otherwise cost the whole run its tensor-core throughput. tf32, the
            # tilelang cudart repoint and the wandb link stay unwrapped on purpose: all three
            # swallow their own failures by design and none may abort a paid run.
            render_tf32_shim(),
            # before anything that can import vllm -- so above the wrapped fragments too. the
            # fragment repoints tilelang's libcudart stub on disk, and vllm's CuMemAllocator binds
            # libcudart the first time a sleeping engine is built. after the stub is already mapped
            # into this process there is nothing left to fix.
            render_tilelang_cudart_shim(),
            render_shim_marker_prologue(files["shim_markers"]),
            *(wrap_shim_fragment(name, source) for name, source in required_fragments),
            # gated on the key rather than the resolved logger list: that list needs python_bin,
            # which is resolved after this file is written. the shim is inert either way -- it only
            # fires when verl actually calls wandb.init, which requires wandb in the logger list.
            render_wandb_link_shim() if os.environ.get("WANDB_API_KEY") else "",
        )
        if part
    )
    with open(files["shim_py"], "w") as f:
        f.write(shim_source)

    # multi-turn: copy the child-side agent loop next to the shim so the verl interpreter can
    # import it (see copy_multi_turn_child_modules for why it is a copy and not an import).
    if inp["multi_turn"]:
        copy_multi_turn_child_modules(files["shim_dir"])
    return [name for name, source in required_fragments if source]


def _prepare_rl_runtime(inp, env, tok, prompts):
    files = _prepare_rl_files(inp, prompts)
    files["expected_shims"] = _write_rl_shim(inp, files)
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

    def _score_batch(requests: list[tuple[int, str]]) -> list[float]:
        # grade the whole batch before touching the observability lock. the env's scorer may block on
        # judge i/o, while record is intentionally a short per-result critical section.
        # `grading` spans that blocking call so the silence watchdog can see the batch is running:
        # no completion is recorded until it returns, so this is the only liveness edge in between.
        with observability.grading():
            scored = score_single_turn_batch(
                env,
                [(solution_str, rollout_examples[int(index)]) for index, solution_str in requests],
                tok=tok,
                thinking=bool(_w.THINKING),
                prompt_opened_thinking=inp["prompt_opened_thinking"],
                think_penalty=inp["think_penalty"],
            )
        results = []
        for (index, solution_str), (score, breakdowns) in zip(requests, scored, strict=True):
            observability.record(message_prompts[int(index)], solution_str, score, breakdowns)
            results.append(score)
        return results

    def _score_for_profile(index: int, solution_str: str) -> float:
        """score for profiling without training observability side effects.

        do not seed rollout buffers. errors propagate so a broken grader is not measured as fast.
        """
        return score_single_turn(
            env,
            solution_str,
            rollout_examples[int(index)],
            tok=tok,
            thinking=bool(_w.THINKING),
            prompt_opened_thinking=inp["prompt_opened_thinking"],
            think_penalty=inp["think_penalty"],
            raise_on_error=True,
        )

    # the profiler times the single-turn grading path (env.reward / env.scores_breakdown on one
    # completion). a multi-turn env scores a terminal episode instead, so that timing would neither
    # describe nor even validly reach its reward path.
    reward_profile = (
        None
        if inp["multi_turn"]
        else _log_reward_profile(
            env,
            _score_for_profile,
            rollout_examples,
            int(inp["prompts_per_step"]) * int(inp["group_size"]),
        )
    )
    multi_turn_bridge = (
        MultiTurnBridge(
            env,
            rollout_examples,
            # index-aligned with rollout_examples: build_grpo_prompt_dataset preserves order.
            env_prompts=[p["env_prompt"] for p in prompts],
            max_turns=int(inp["max_turns"]),
            per_turn_credit=bool(inp["per_turn_credit"]),
            on_episode_scored=observability.record,
            grading=observability.grading,
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
    )
    return _RewardRuntime(
        observability=observability,
        # filled from the child's marker line; stays empty when wandb is off.
        wandb_link={},
        reward_profile=reward_profile,
        multi_turn_bridge=multi_turn_bridge,
        server=server,
        reward_url=reward_url,
    )


def _resolve_training_settings(inp, caps):
    expected_steps = int(inp["steps"])
    # verl logs from its own interpreter; gate wandb on that env (see resolve_verl_loggers).
    loggers = _rl_train().resolve_verl_loggers(caps)
    spec = _w.JOB_SPEC
    project_name = (spec.wandb.project if spec and spec.wandb else None) or "flash"
    experiment_name = _w.wandb_run_name()
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
    return None, _rl_train()._NvidiaSmiPeakSampler().start(), None


def _announce_training(t_start: float, cfg) -> tuple[float, float]:
    # the executor budget is sized per run now, so print what this one actually asked for --
    # a vllm init failure reports the demand against the free memory, and the demand is
    # otherwise invisible in the log.
    print(f"[rl-verl] rollout gpu_memory_utilization={cfg['gpu_mem_util']:.4f}", flush=True)
    setup_seconds = time.time() - t_start
    _w.heartbeat("rl_train_start", setup_seconds=setup_seconds, gpu=gpu_diagnostics())
    return setup_seconds, time.time()


def _start_resume_uploader(
    *, local_dir, resume_step, inp, workdir, python_bin, preprocessor, adv_spread_history
):
    resume_uploader = _rl_train()._VerlResumeUploader(
        local_dir,
        resume_step=resume_step,
        required_steps=inp["save_at_steps"],
        export_root=os.path.join(workdir, "exports"),
        python_bin=python_bin,
        model_id=inp["model_id"],
        model_revision=inp["model_revision"],
        preprocessor=preprocessor,
        # a resumed run's restored weights already carry the earlier steps' updates, so this
        # worker's own spread history cannot speak for them; let it publish as before and leave
        # the verdict to the same abstention _check_grpo_had_a_gradient makes.
        had_gradient=(
            None if resume_step else lambda: any(spread > 0.0 for spread in adv_spread_history)
        ),
    )
    resume_uploader.credit_durable_required_steps(resume_step)
    resume_uploader.start()
    return resume_uploader


def _build_rl_child_env(inp, files, loggers, reward_url):
    parent = _rl_train()
    # allowlist the child env because ray fans it to every actor; scoring stays in the parent, so
    # the child needs no platform hf token, github token, or user secrets. FLA_ kernel settings
    # still cross through _CHILD_ENV_PREFIXES.
    env_for_verl = parent._build_verl_child_env(
        shim_dir=files["shim_dir"], wandb_enabled="wandb" in loggers
    )
    env_for_verl["FLASH_VERL_REWARD_URL"] = reward_url
    # the model is prefetched above; keep the subprocess off hf's rate-limited api.
    env_for_verl["HF_HUB_OFFLINE"] = "1"
    env_for_verl["TRANSFORMERS_OFFLINE"] = "1"
    env_for_verl["HF_HUB_DISABLE_XET"] = "1"
    if inp["multi_turn"]:
        # the plugin named here and the loop it builds live in shim_dir, so PYTHONPATH must
        # carry it -- see below.
        env_for_verl.update(
            multi_turn_child_env(inp, reward_url=reward_url, thinking=bool(_w.THINKING))
        )
    # python imports sitecustomize automatically at startup, so the shim patches verl before
    # main_ppo runs. prepend rather than replace: an inherited PYTHONPATH may carry the verl
    # install itself, and ray workers inherit this env so every actor gets the same patch.
    # multi-turn needs the same entry for its copied-in agent loop modules. unconditional: the
    # shim always carries at least the tf32 fragment, so a skipped entry would silently drop it.
    env_for_verl["PYTHONPATH"] = os.pathsep.join(
        item for item in (files["shim_dir"], os.environ.get("PYTHONPATH", "")) if item
    )
    return env_for_verl


def _ingest_step_metrics(
    line: str,
    inp,
    state: _StepMetricState,
    _reward_observability: Callable[[], dict],
) -> None:
    sent_first_metrics = state.sent_first_metrics
    step_metrics = parse_verl_step_metrics(line)
    if step_metrics is not None:
        state.step_line_times.append(time.time())
        # a run constant rather than a verl metric, so it is stamped here from
        # the resolved run config.
        step_metrics["max_completion_tokens"] = inp["max_completion"]
        append_step_metrics(state.metrics_last, step_metrics, limit=GRPO_METRIC_HISTORY_LIMIT)
        # the worker's error path reads this global, so a run that dies mid-training
        # still reports the steps it did complete (worker/__init__.py:_err_metrics).
        LATEST_GRPO_METRICS_LAST[:] = state.metrics_last
        # rl_train_start arms a 900s throttle, so force the first backlog through like
        # heartbeat.py force_first_samples and retry until committed. later emissions remain
        # throttled to protect the hf commit cap.
        if not sent_first_metrics:
            sent_first_metrics = _w.heartbeat(
                "rl_step",
                force=True,
                step=step_metrics["step"],
                metrics_last=list(state.metrics_last),
                **_reward_observability(),
                gpu=gpu_diagnostics(include_torch=False),
            )
            state.sent_first_metrics = sent_first_metrics
        # per-step series for train_meta observability parity. these live on the same
        # line as everything else: verl's only console metric sink is LocalLogger,
        # which always prints "step:N - ..." (verl/utils/logger/aggregate_logger.py),
        # so a line without a step carries no metric to collect.
        for verl_key, sink in (
            ("critic/rewards/mean", state.reward_history),
            ("actor/pg_loss", state.loss_curve),
            ("response_length/mean", state.resp_len_history),
        ):
            value = parse_verl_metric(line, verl_key)
            if value is not None:
                sink.append(value)
        # advantages/max and /min are emitted for every step outside verl's
        # use_critic branch (trainer/ppo/metric_utils.py), so they are present under
        # grpo even though the key is namespaced critic/.
        adv_max = parse_verl_metric(line, "critic/advantages/max")
        adv_min = parse_verl_metric(line, "critic/advantages/min")
        if adv_max is not None and adv_min is not None:
            state.adv_spread_history.append(adv_max - adv_min)


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
    proc = subprocess.Popen(
        [python_bin, "-m", "verl.trainer.main_ppo", *overrides],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env_for_verl,
        start_new_session=True,
    )
    child_stream = _rl_train()._GrpoSubprocessStream(
        proc, tail=state.child_tail, silence_watchdog=state.silence_watchdog
    )
    step_re = re.compile(r"step:\s*(\d+)")
    progress, last_dump_step = state.progress, state.last_dump_step
    shim_markers = (files or {}).get("shim_markers")
    expected_shims = (files or {}).get("expected_shims", ())
    shims_verified = shim_markers is None
    try:
        for line in child_stream:
            print(f"[verl] {line}", end="", flush=True)
            link = parse_wandb_link(line)
            if link is not None:
                reward_runtime.wandb_link.update(link)
            m = step_re.search(line)
            if m:
                progress["step"] = int(m.group(1))
                # the first step line is the training-start boundary: sitecustomize import is long
                # finished by then, so a marker still missing means this child is training with no
                # flash patch at all, so fail now rather than after the whole run is paid for. not
                # on the first output line: fragments print while later ones are still applying.
                if not shims_verified:
                    verify_applied_shim_markers(shim_markers, expected_shims)
                    shims_verified = True
                # dump one sample completion per new step to the flash log (#607).
                if progress["step"] != last_dump_step[0]:
                    # the generation boundary: verl logs this line once its step is scored, so
                    # everything the reward bridge buffered since the last one is that step's
                    # complete output. seal it before the preview reads `latest`, so both the log
                    # line and the heartbeat describe the same generation.
                    reward_runtime.observability.close_generation(progress["step"])
                    # asks for THIS step's rows, not merely the newest: when the line is spent on a
                    # generation the queue already dropped, nothing is published and the previous
                    # generation's text would print under this step.
                    samp = reward_runtime.observability.latest_for_step(progress["step"])
                    if samp:
                        last_dump_step[0] = progress["step"]
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


def _validate_rl_child(rc, state, resume_step, expected_steps, resume_uploader, *, files=None):
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
    # belt and braces behind the first-step check in _execute_rl_child: a run that exits 0 without
    # printing a step line (a resume already at the horizon) still may not pass unverified.
    shim_markers = (files or {}).get("shim_markers")
    if shim_markers is not None:
        verify_applied_shim_markers(shim_markers, (files or {}).get("expected_shims", ()))
    # the gradient verdict runs here, ahead of required-save completeness, because a zero-spread
    # run withholds every required deployable BY DESIGN: checking completeness first would raise
    # on artifacts the gate is deliberately holding and report a checkpoint-publication failure
    # -- the symptom -- instead of the constant reward signal that caused it.
    _rl_train()._check_grpo_had_a_gradient(
        state.reward_history,
        state.adv_spread_history,
        resumed=bool(resume_step),
        # a resume already at the target runs zero steps and emits zero metrics; that is a
        # complete policy, not a broken reward bridge. steps_run below still has to reach
        # expected_steps, so this cannot excuse a run that stopped short.
        already_complete=bool(resume_step) and resume_step >= expected_steps,
    )
    # training finished cleanly, so a missing required save is a real defect rather than a
    # side effect of a crash. stop here (not in finally, which suppresses) to surface it.
    # only when exact saves were requested: without them the drain stays best-effort, and
    # letting a slow resume upload raise here would fail an otherwise-successful run.
    if resume_uploader is not None and resume_uploader.required_steps:
        resume_uploader.stop()
        resume_uploader.raise_if_incomplete()


def _prepare_final_adapter(local_dir: str, t_train: float):
    # collect verl's lora checkpoint -> flash-servable peft adapter, then reuse flash finalize.
    out_dir = f"/tmp/rl_seed{_w.SEED}"
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
    parent = _rl_train()
    parent.export_peft_adapter(
        actor_dir, adapter_dir, base_model_id=inp["model_id"], python_bin=python_bin
    )
    parent.stamp_adapter_dir_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
    _w.write_base_model_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
