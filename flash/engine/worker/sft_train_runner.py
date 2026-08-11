"""SFT training orchestration phases and the paid child-run boundary.

Split out of ``flash.engine.worker.sft_train`` to keep that module under the file-size limit.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import dataclass
from functools import reduce
from math import gcd

from flash.engine.worker import sft_train as _sft_train

RECIPE = _sft_train.RECIPE
_w = _sft_train._w
_MAX_ZERO_GRAD_STEPS = _sft_train._MAX_ZERO_GRAD_STEPS
_SFT_LORAPLUS_RATIO = _sft_train._SFT_LORAPLUS_RATIO
_LORAPLUS_READY_MARKER = _sft_train._LORAPLUS_READY_MARKER
_VERL_OPTIMIZER_IMPL = _sft_train._VERL_OPTIMIZER_IMPL
_VERL_OPTIMIZER_NAME = _sft_train._VERL_OPTIMIZER_NAME
build_sft_overrides = _sft_train.build_sft_overrides
gdn_reset_arch_from_caps = _sft_train.gdn_reset_arch_from_caps
render_gdn_varlen_shim = _sft_train.render_gdn_varlen_shim
render_wandb_link_shim = _sft_train.render_wandb_link_shim
sft_tokens_for_updates = _sft_train.sft_tokens_for_updates
sft_under_ran = _sft_train.sft_under_ran
validate_save_steps = _sft_train.validate_save_steps
_render_sft_dataset_module = _sft_train._render_sft_dataset_module
_render_sft_sitecustomize = _sft_train._render_sft_sitecustomize


@dataclass(frozen=True)
class _SftPaths:
    workdir: str
    data_dir: str
    image_dir: str
    local_dir: str
    export_root: str


@dataclass(frozen=True)
class _SftOptions:
    spec: object
    env: object
    started_at: float
    gpu_probe: dict
    model_id: str
    model_revision: str
    epochs: int
    learning_rate: float
    effective_batch: int
    max_steps: int
    save_at_steps: tuple[int, ...]
    save_every: int
    gpu_count: int
    paths: _SftPaths


@dataclass(frozen=True)
class _SftData:
    rows: list[dict]
    multimodal: bool
    profile: object
    max_length: int
    realized_max_length: int
    train_file: str


@dataclass(frozen=True)
class _SftModelSetup:
    download_seconds: float
    setup_seconds: float
    lora_rank: int
    lora_alpha: int
    target_modules: object
    warmstart_adapter: str | None
    fused_ce: bool
    train_batch_size: int
    micro_batch: int
    update_horizon: int
    loop_epochs: int
    save_freq: int
    gradient_checkpointing: bool
    reentrant_gradient_checkpointing: bool


@dataclass(frozen=True)
class _SftCapabilities:
    python_bin: str
    caps: dict
    gdn_hybrid: bool
    gdn_module: str


@dataclass(frozen=True)
class _SftChild:
    python_bin: str
    loggers: list[str]
    project_name: str
    experiment_name: str
    gdn_reset_arch: str | None
    gdn_hybrid: bool
    resume_step: int
    watcher: object
    child_env: dict[str, str]
    command: list[str]
    # ranks actually launched, which is the allocated card count only when the batch divides by it.
    # reported so a reader comparing realized step time against the quote sees the executed width.
    world_size: int


@dataclass
class _SftProgress:
    values: dict[str, float | int | None]
    zero_grad_steps: list[int]
    observed_grad_norms: list[float]
    loss_curve: list[float]
    train_tokens: int
    loraplus_applied: bool
    wandb_link: dict[str, str | None]


@dataclass(frozen=True)
class _SftVerified:
    actor_dir: str
    final_step: int
    train_tokens: int


@dataclass(frozen=True)
class _SftOutputs:
    """What the paid child produced: where the adapter landed and what it cost to get there."""

    adapter_dir: str
    train_wall: float
    device_peak_gpu_gb: float


def _resolve_sft_options(spec) -> _SftOptions:
    spec = spec or _w.JOB_SPEC
    env = _w.require_active_env()
    # the child trainer is seeded through its shim, but the environment's dataset/completion calls
    # run HERE in the parent. without this the documented top-level seed no longer reproduces sft
    # targets for any env whose row construction uses python/numpy randomness.
    _sft_train.seed_training_rngs(_w.SEED)
    started_at = time.time()
    _w.heartbeat("sft_start", gpu=_w.gpu_diagnostics(include_torch=False))
    gpu_probe = _sft_train._probe_gpu_in_subprocess(
        spec.gpu.type if spec else None,
        exact_type=spec.gpu.type if spec else "",
    )
    model_id = spec.model if spec else RECIPE.hf_model_id
    model_revision = getattr(spec, "model_revision", "") if spec else ""
    train_spec = spec.train if spec else None

    def train_opt(name, default):
        value = getattr(train_spec, name, None) if train_spec else None
        return value if value is not None else default

    workdir = os.path.join("/tmp", "flash-sft-verl", _w.RUN_ID, f"seed-{_w.SEED}")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)
    paths = _SftPaths(
        workdir=workdir,
        data_dir=os.path.join(workdir, "data"),
        image_dir=os.path.join(workdir, "images"),
        local_dir=os.path.join(workdir, "checkpoints"),
        export_root=os.path.join(workdir, "checkpoint-adapters"),
    )
    os.makedirs(paths.data_dir, exist_ok=True)
    os.makedirs(paths.local_dir, exist_ok=True)
    return _SftOptions(
        spec=spec,
        env=env,
        started_at=started_at,
        gpu_probe=gpu_probe,
        model_id=model_id,
        model_revision=model_revision,
        epochs=int(train_opt("epochs", RECIPE.sft.num_epochs)),
        learning_rate=float(train_opt("learning_rate", RECIPE.sft.learning_rate)),
        effective_batch=int(train_opt("batch_size", RECIPE.sft.effective_batch)),
        max_steps=int(train_opt("max_steps", 0) or 0),
        save_at_steps=tuple(getattr(train_spec, "save_at_steps", ()) or ()),
        save_every=int(train_opt("save_every", 50)),
        gpu_count=int(getattr(getattr(spec, "gpu", None), "count", 1) or 1),
        paths=paths,
    )


def _prepare_sft_data(options: _SftOptions) -> _SftData:
    from flash.engine.profiling.workload_profile import require_matching_sft_profile

    # carried on the spec, not read from `flash.__version__` here: the worker runs the plane's
    # source snapshot off PYTHONPATH with no flash distribution installed, so a locally derived
    # version is the "0+unknown" fallback and would reject every profile the plane ever froze.
    producer_version = options.spec.workload_profile_producer_version
    prepared_workload = _sft_train.prepare_sft_workload(
        options.spec,
        options.env,
        tokenizer_loader=lambda candidate, revision: _w.load_tokenizer(
            candidate,
            revision=revision,
        ),
        producer_version=producer_version,
        image_dir=options.paths.image_dir,
        allow_packing=True,
    )
    expected_profile = require_matching_sft_profile(
        options.spec.workload_profile,
        input_digest=options.spec.workload_profile_input_digest,
        producer_version=producer_version,
        tokenizer_revision=options.model_revision,
    )
    if prepared_workload.profile != expected_profile:
        raise ValueError("sft workload changed after the quote was frozen")

    rows = prepared_workload.rows
    profile = prepared_workload.profile
    # the context window comes from the profile, not a second reading of the train fields. the
    # rows were truncated at the profile's max_length and the quote was priced at it, so a
    # locally re-derived value could disagree with both while the parity check above still
    # passed: that check compares two values the workload module produced, so it cannot see a
    # third derivation living here.
    max_length = _sft_train._sft_profile_max_length(profile)
    dropped = profile.dropped_examples
    selected_count = profile.selected_examples
    sampled_texts = prepared_workload.sampled_texts
    multiturn_targets = prepared_workload.multiturn_targets
    if dropped:
        print(
            f"[sft] dropped {dropped} rows with no real completion target "
            "(sft_max_len truncated away the whole completion, or it was content-free)"
        )
    if multiturn_targets:
        print(
            f"[sft] multi-turn SFT: {multiturn_targets}/{selected_count} rows train on a full "
            "target transcript"
        )
    elif getattr(options.env, "multi_turn", False):
        print(
            "[sft][warn] this is a multi-turn environment but no row ships a multi-turn "
            "target completion"
        )
    if _w.THINKING and not any("<think>" in text for text in sampled_texts[:256]):
        print(
            "WARN: thinking mode is ON but no sampled SFT target contains a <think> trace; "
            "training on non-reasoning targets teaches the model to skip thinking"
        )

    total_tokens_per_epoch = profile.real_tokens_per_epoch
    realized_max_length = profile.realized_max_length
    masked_tokens = total_tokens_per_epoch - profile.supervised_tokens_per_epoch
    print(
        f"[sft] completion-only loss: masking {masked_tokens}/{total_tokens_per_epoch} "
        f"({masked_tokens / total_tokens_per_epoch:.0%}) prompt tokens"
    )
    train_file = os.path.join(options.paths.data_dir, "train.parquet")
    _sft_train._write_sft_parquet(rows, train_file)
    return _SftData(
        rows=rows,
        multimodal=prepared_workload.multimodal,
        profile=profile,
        max_length=max_length,
        realized_max_length=realized_max_length,
        train_file=train_file,
    )


def _prepare_sft_model(options: _SftOptions, data: _SftData) -> _SftModelSetup:
    from flash.core.catalog import MODELS
    from flash.engine.plan.vram import sft_chunked_nll_enabled

    download_seconds = _w.prefetch_model(options.model_id, revision=options.model_revision)
    setup_seconds = time.time() - options.started_at
    _w.heartbeat(
        "sft_model_load",
        setup_seconds=setup_seconds,
        gpu=_w.gpu_diagnostics(include_torch=False),
    )
    lora_config = _w.make_lora(options.model_id)
    lora_rank = int(lora_config.r)
    target_modules = lora_config.target_modules
    if isinstance(target_modules, set | frozenset):
        target_modules = sorted(target_modules)
    warmstart_adapter = _sft_train._warmstart_adapter_path(
        options.model_id, options.model_revision, lora_rank
    )
    vocab_size = _sft_train._resolve_sft_vocab_size(options.model_id, options.model_revision)
    fused_ce = sft_chunked_nll_enabled(options.model_id)
    per_device_batch, _ = _sft_train._resolve_sft_grad_accum(
        options.effective_batch,
        seq_len=data.realized_max_length,
        vocab=vocab_size,
        fused=fused_ce,
    )
    train_batch_size = data.profile.examples_per_update
    micro_batch = max(1, min(per_device_batch, train_batch_size))
    steps_per_epoch = max(1, math.ceil(len(data.rows) / train_batch_size))
    update_horizon = data.profile.authoritative_steps
    validate_save_steps(options.save_at_steps, update_horizon)
    loop_epochs = max(options.epochs, math.ceil(update_horizon / steps_per_epoch))
    save_freq = reduce(gcd, options.save_at_steps) if options.save_at_steps else options.save_every

    card_vram_gb = float(options.gpu_probe.get("memory_gb") or 0.0)
    raw_capability = options.gpu_probe.get("capability")
    capability = tuple(raw_capability) if raw_capability else None
    hidden, layers = _sft_train._model_arch_dims(options.model_id, revision=options.model_revision)
    info = MODELS.get(options.model_id)
    active_params_b = float(getattr(info, "active_params_b", 0.0) or 0.0) or None
    gradient_checkpointing = _sft_train._resolve_sft_gradient_checkpointing(
        options.model_id,
        data.realized_max_length,
        allow_disable=True,
        card_vram_gb=card_vram_gb,
        capability=capability,
        active_params_b=active_params_b,
        hidden=hidden,
        num_layers=layers,
        fused_ce=fused_ce,
        per_device_bs=micro_batch,
        lora_rank=lora_rank,
        revision=options.model_revision,
    )
    reentrant_gradient_checkpointing = bool(
        gradient_checkpointing
        and _sft_train._resolve_sft_reentrant_gradient_checkpointing(options.model_id)
    )
    return _SftModelSetup(
        download_seconds=download_seconds,
        setup_seconds=setup_seconds,
        lora_rank=lora_rank,
        lora_alpha=int(lora_config.lora_alpha),
        target_modules=target_modules,
        warmstart_adapter=warmstart_adapter,
        fused_ce=fused_ce,
        train_batch_size=train_batch_size,
        micro_batch=micro_batch,
        update_horizon=update_horizon,
        loop_epochs=loop_epochs,
        save_freq=save_freq,
        gradient_checkpointing=gradient_checkpointing,
        reentrant_gradient_checkpointing=reentrant_gradient_checkpointing,
    )


def sft_data_parallel_cards(gpu_count: int, train_batch_size: int) -> int:
    """Cards SFT can actually train on, given verl splits the batch across them.

    SFT runs data-parallel (see `_prepare_sft_child`), so verl derives
    `train_batch_size_per_dp = train_batch_size // dp_size` and hands that straight to a
    DataLoader. Two card counts are unusable and both fail loudly rather than train:
    a count ABOVE the batch floors the per-rank batch to 0, and `DataLoader(batch_size=0)`
    raises `ValueError`; a count that does not DIVIDE the batch silently changes the global
    batch, because verl's sampler drops the remainder (`drop_last=True` on both the sampler
    and the loader).

    So take the largest divisor of the batch that is <= the allocated cards. That keeps the
    realized global batch exactly `train_batch_size` on every width this returns, which is what
    makes the card count a pure throughput choice rather than a hyperparameter change.

    Returns 1 for an unpacked run (`examples_per_update` is 1 there), which is correct: one
    example cannot be split, so extra cards would have nothing to hold.
    """
    cards = max(1, int(gpu_count))
    batch = max(1, int(train_batch_size))
    for count in range(min(cards, batch), 0, -1):
        if batch % count == 0:
            return count
    return 1


def _prepare_sft_child(
    options: _SftOptions,
    data: _SftData,
    model: _SftModelSetup,
    capabilities: _SftCapabilities,
    use_remove_padding: bool,
    gdn_reset_arch: str | None,
) -> _SftChild:
    model_path = _sft_train._cached_model_path(options.model_id, options.model_revision)
    # verl logs from the verl interpreter, so gate wandb on THAT env (see resolve_verl_loggers).
    loggers = _sft_train.resolve_verl_loggers(capabilities.caps)
    project_name = (
        options.spec.wandb.project if options.spec and options.spec.wandb else None
    ) or "flash"
    experiment_name = _w.wandb_run_name()
    shim_dir = os.path.join(options.paths.workdir, "shim")
    os.makedirs(shim_dir, exist_ok=True)
    custom_dataset_path = os.path.join(shim_dir, "flash_verl_sft_dataset.py")

    # the gdn boundary shim resets conv and recurrent state at packed example boundaries, but only
    # when the verl child has the kernels that read seq_idx and cu_seqlens; the no-fla fallbacks
    # accept both and discard them. so the shim is installed only when the child proves it can reset.
    # `gdn_reset_arch` is resolved by the caller, inside the configuring liveness wrap, because the
    # probe is part of the setup silence that wrap exists to cover, and because a packed run must
    # take the RAISING gate there rather than the soft form.
    #
    # SFT shards by DATA, not by sequence: ulysses is pinned to 1 and fsdp splits the batch. verl's
    # ulysses support patches `_flash_attention_forward` and nothing else, but every catalog model
    # is a GatedDeltaNet hybrid whose layers are mostly LINEAR attention plus a causal conv -- and
    # both carry recurrent state along the sequence. slicing the sequence across ranks would run
    # each shard's recurrence and conv as if it were a whole sequence, with no cross-rank state, so
    # sequence parallelism is not merely unimplemented for this family, it is incorrect for it.
    # It also crashed outright. remove-padding flattens the batch to one `(1, total_nnz)` row, and
    # verl slices that row per rank, so the shapes reaching the GDN kernels stop agreeing and a
    # sharded run died on `seq_idx must have shape (batch_size, seqlen)` -- at any batch size,
    # including 1, because "remove padding" leaves no batch dimension to keep an example whole.
    world_size = sft_data_parallel_cards(options.gpu_count, model.train_batch_size)
    config = {
        "train_files": data.train_file,
        "train_batch_size": model.train_batch_size,
        "max_length": data.max_length,
        "micro_batch": model.micro_batch,
        "max_token_len_per_gpu": data.realized_max_length * model.micro_batch,
        "custom_dataset_path": custom_dataset_path,
        "model_path": model_path,
        "lora_rank": model.lora_rank,
        "lora_alpha": model.lora_alpha,
        "target_modules": model.target_modules,
        "target_parameters": _w.lora_target_parameters(options.model_id),
        "lora_adapter_path": model.warmstart_adapter,
        "ulysses_sp_size": 1,
        "lr": options.learning_rate,
        "warmup_ratio": RECIPE.sft.warmup_frac,
        "optimizer_impl": _VERL_OPTIMIZER_IMPL,
        "optimizer_name": _VERL_OPTIMIZER_NAME,
        "optimizer_kwargs": None,
        "local_dir": options.paths.local_dir,
        "save_freq": model.save_freq,
        "n_gpus_per_node": world_size,
        "seed": _w.backend_seed(_w.SEED),
        "project_name": project_name,
        "experiment_name": experiment_name,
        "loop_epochs": model.loop_epochs,
        "loggers": loggers,
        # liger produces a 0.0 lora grad norm under fsdp2 + peft + gradient checkpointing, versus
        # 7.02 off in a matched qwen3.5-9b test. fused linear ce remains provided by
        # use_fused_kernels with the impl_backend resolved below.
        **_sft_train._sft_liger_config(),
        "gradient_checkpointing": model.gradient_checkpointing
        and not model.reentrant_gradient_checkpointing,
        "total_training_steps": model.update_horizon if options.max_steps > 0 else None,
        "total_epochs": options.epochs if options.max_steps <= 0 else None,
        "use_remove_padding": use_remove_padding,
        # resolved from the out-of-process capability probe, never by opening cuda in this parent.
        "fused_ce_backend": _sft_train._resolve_sft_fused_ce_backend(capabilities.caps),
    }
    overrides = build_sft_overrides(config)
    shim_source = _render_sft_sitecustomize(
        seed=config["seed"],
        loraplus_ratio=_SFT_LORAPLUS_RATIO,
        save_at_steps=options.save_at_steps,
        total_steps=model.update_horizon,
        reentrant_gradient_checkpointing=model.reentrant_gradient_checkpointing,
    )
    # both shims, not either: this one patches DistributedSampler/StatefulDataLoader so the child's
    # row order matches the profile's byte for byte, and the gdn one patches the model's text
    # forward to reset linear-attention state at packed example boundaries. different objects,
    # no interaction -- a gdn hybrid needs both, and dropping either is a silent correctness bug.
    shim_source = _sft_train._append_exact_sft_dataloader_shim(shim_source)
    if gdn_reset_arch is not None:
        shim_source += render_gdn_varlen_shim(gdn_reset_arch)
    if "wandb" in loggers:
        shim_source += render_wandb_link_shim()
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as file:
        file.write(shim_source)
    with open(custom_dataset_path, "w", encoding="utf-8") as file:
        file.write(_render_sft_dataset_module())

    resume_step = _sft_train._restore_verl_resume(options.paths.local_dir)
    watcher = _sft_train._VerlCheckpointWatcher(
        local_dir=options.paths.local_dir,
        export_root=options.paths.export_root,
        python_bin=capabilities.python_bin,
        model_id=options.model_id,
        model_revision=options.model_revision,
        required_steps=options.save_at_steps,
    )
    watcher.processed_steps.update(
        _sft_train._durable_required_save_steps(options.save_at_steps, resume_step)
    )
    if resume_step >= model.update_horizon:
        missing = sorted(watcher.required_steps - watcher.processed_steps)
        if missing:
            raise RuntimeError(f"required saves were not durably published: {missing}")
    child_env = _sft_train._build_verl_child_env(
        shim_dir=shim_dir,
        wandb_enabled="wandb" in loggers,
    )
    command = [
        capabilities.python_bin,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        # the RESOLVED width, not the allocated card count: verl splits the global batch across
        # every rank torchrun starts, so a rank the batch cannot feed is the `batch_size=0` crash.
        f"--nproc-per-node={world_size}",
        "-m",
        "verl.trainer.sft_trainer",
        *overrides,
    ]
    return _SftChild(
        python_bin=capabilities.python_bin,
        loggers=loggers,
        project_name=project_name,
        experiment_name=experiment_name,
        gdn_reset_arch=gdn_reset_arch,
        gdn_hybrid=capabilities.gdn_hybrid,
        resume_step=resume_step,
        watcher=watcher,
        child_env=child_env,
        command=command,
        world_size=world_size,
    )


def _prepare_sft_progress(data: _SftData, model: _SftModelSetup, resume_step: int) -> _SftProgress:
    progress = {"step": resume_step, "loss": None, "grad_norm": None, "lr": None}
    # consecutive steps seen with grad_norm == 0.0 at a nonzero lr. one step can legitimately be
    # zero (a fully-masked micro-batch), so require a short run of them before failing.
    zero_grad_steps: list[int] = []
    # every grad_norm this session observed, so a horizon too short to trip the consecutive-run
    # guard above can still be rejected at the end. a one-update run appends exactly one zero and
    # never reaches _MAX_ZERO_GRAD_STEPS, which shipped the GRAD-001 failure the guard exists to
    # stop: done, billed, and an adapter identical to the base weights.
    observed_grad_norms: list[float] = []
    train_tokens = (
        sft_tokens_for_updates(
            data.rows,
            examples_per_update=model.train_batch_size,
            updates=resume_step,
            field="input_ids",
        )
        if resume_step > 0
        else 0
    )
    return _SftProgress(
        values=progress,
        zero_grad_steps=zero_grad_steps,
        observed_grad_norms=observed_grad_norms,
        loss_curve=[],
        train_tokens=train_tokens,
        loraplus_applied=resume_step >= model.update_horizon,
        wandb_link={},
    )


class _SftProgressCallbacks:
    def __init__(self, progress: _SftProgress):
        self.progress = progress

    def on_step(self, step: int) -> None:
        self.progress.values["step"] = step
        payload = {
            "step": step,
            "loss": self.progress.values["loss"],
            "grad_norm": self.progress.values["grad_norm"],
            "learning_rate": self.progress.values["lr"],
        }
        _w.heartbeat(
            "sft_step", **{key: value for key, value in payload.items() if value is not None}
        )

    def child_heartbeat(self) -> None:
        _w.heartbeat("sft_step", liveness=True, step=int(self.progress.values["step"] or 0))


def _invoke_sft_child(child: _SftChild, callbacks: _SftProgressCallbacks, on_line) -> int:
    return _sft_train.run_verl_training(
        child.command,
        env=child.child_env,
        on_step=callbacks.on_step,
        on_line=on_line,
        heartbeat=callbacks.child_heartbeat,
    )


def _consume_sft_marker_line(progress: _SftProgress, line: str) -> bool:
    """Fold a child log line's non-metric markers in; answer whether it is an optimizer step.

    False means the caller has nothing further to do with the line. A True answer additionally
    asserts the lora+ shim landed before the first step, because a run that reaches an optimizer
    update without it is training plain lora at the lora+ learning rate.
    """
    if _LORAPLUS_READY_MARKER in line:
        progress.loraplus_applied = True
    link = _sft_train.parse_wandb_link(line)
    if link is not None:
        progress.wandb_link.update(link)
    if _sft_train.verl_step_number(line) is None:
        return False
    if _SFT_LORAPLUS_RATIO > 1 and not progress.loraplus_applied:
        raise RuntimeError(
            "verl reached an optimizer step before the required lora+ shim succeeded"
        )
    return True


def _record_sft_step_metrics(
    progress: _SftProgress,
    loss: float | None,
    grad_norm: float | None,
    learning_rate_value: float | None,
) -> None:
    """Fold one step's parsed metrics into the progress carrier, raising on a dead gradient.

    Parsing stays in the caller's closure: `tests/test_sft_train.py` pins the three
    `parse_verl_metric` calls to `on_line` itself, and reads the values it hands over here.
    """
    values = progress.values
    if loss is not None:
        progress.loss_curve.append(round(loss, 4))
        values["loss"] = loss
    if grad_norm is not None:
        values["grad_norm"] = grad_norm
        progress.observed_grad_norms.append(grad_norm)
        # a 0.0 grad norm means backward produced nothing for every trainable parameter. fail the
        # run instead of billing and serving an unchanged adapter (GRAD-001).
        #
        # VERL-138: do not condition on lr. transformer_impl.py:683-688 computes grad_norm from
        # p.grad before optimizer.step() and scheduler advance, so lr cannot make the gradient zero.
        if grad_norm == 0.0:
            progress.zero_grad_steps.append(int(values["step"] or 0))
            if len(progress.zero_grad_steps) >= _MAX_ZERO_GRAD_STEPS:
                raise RuntimeError(
                    "verl reported train/grad_norm=0.0 on "
                    f"{len(progress.zero_grad_steps)} steps: no gradient is reaching the "
                    "lora parameters, so this run would train nothing. see GRAD-001"
                )
        else:
            progress.zero_grad_steps.clear()
    if learning_rate_value is not None:
        values["lr"] = learning_rate_value


def _finish_sft_child(
    gpu_sampler,
    train_started_at: float,
    return_code: int,
    progress: _SftProgress,
) -> tuple[float, float]:
    train_wall = time.time() - train_started_at
    device_peak_gpu_gb = gpu_sampler.stop_gb()
    if return_code != 0:
        raise RuntimeError(f"verl SFT subprocess exited with status {return_code}")
    if _SFT_LORAPLUS_RATIO > 1 and not progress.loraplus_applied:
        raise RuntimeError("required lora+ shim did not emit its success marker")
    return train_wall, device_peak_gpu_gb


def _verify_sft_run(
    options: _SftOptions,
    data: _SftData,
    model: _SftModelSetup,
    progress: _SftProgress,
    resume_step: int,
) -> _SftVerified:
    actor_dir, final_step = _sft_train.latest_global_step_dir(options.paths.local_dir)
    if sft_under_ran(final_step, model.update_horizon, options.max_steps):
        raise RuntimeError(
            f"sft completed {final_step}/{model.update_horizon} requested optimizer updates"
        )
    # short runs may finish before the consecutive zero-grad guard fires, so reject sessions where
    # every observed update had a dead gradient. tolerate isolated zeros; abstain on resume because
    # restored weights include unseen earlier updates.
    if not resume_step and progress.observed_grad_norms and not any(progress.observed_grad_norms):
        raise RuntimeError(
            "verl reported train/grad_norm=0.0 on every one of "
            f"{len(progress.observed_grad_norms)} observed optimizer updates: no gradient is reaching "
            "the lora parameters, so this run would train nothing. see GRAD-001"
        )
    train_tokens = sft_tokens_for_updates(
        data.rows,
        examples_per_update=model.train_batch_size,
        updates=final_step,
        field="input_ids",
    )
    return _SftVerified(actor_dir=actor_dir, final_step=final_step, train_tokens=train_tokens)
