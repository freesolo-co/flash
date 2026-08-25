"""SFT training orchestration phases and the paid child-run boundary.

Split out of ``flash.engine.worker.train.entry.sft_train`` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from functools import reduce
from math import gcd

import flash.engine.worker.io.heartbeat as _worker_heartbeat
import flash.engine.worker.io.hf as _worker_hf
import flash.engine.worker.io.wandb_log as _worker_wandb
import flash.engine.worker.model.adapter as _worker_adapter
import flash.engine.worker.perf as _worker_perf
import flash.engine.worker.runtime.rng as _worker_rng
import flash.engine.worker.runtime.state as _worker_state
from flash.adapters.fused_experts import lora_target_parameters
from flash.adapters.targets import resolve_lora_targeting
from flash.core.catalog import get_model
from flash.engine.plan.steps import sft_data_parallel_cards, widest_usable_dp_width
from flash.engine.support.verl_policy import _resolve_fsdp_generation
from flash.engine.worker.train.core.child.runtime import TEXT_LORA_TARGET_SHIM
from flash.engine.worker.train.entry import sft_train as _sft_train
from flash.engine.worker.verl.parallelism import ULYSSES_SEQUENCE_PARALLEL_SIZE
from flash.providers.core.base import rentable_gpu_counts

RECIPE = _sft_train.RECIPE
_MAX_ZERO_GRAD_STEPS = _sft_train._MAX_ZERO_GRAD_STEPS
_SFT_LORAPLUS_RATIO = _sft_train._SFT_LORAPLUS_RATIO
_LORAPLUS_READY_MARKER = _sft_train._LORAPLUS_READY_MARKER
_VERL_OPTIMIZER_IMPL = _sft_train._VERL_OPTIMIZER_IMPL
_VERL_OPTIMIZER_NAME = _sft_train._VERL_OPTIMIZER_NAME
build_sft_overrides = _sft_train.build_sft_overrides
# taken from the parent like every other name here, so the runner and `sft_train`'s own
# `sft_data_loading`/`sft_configuring` wraps use one object rather than two imports of it.
liveness_heartbeat = _sft_train.liveness_heartbeat
render_sitecustomize_bootstrap = _sft_train.render_sitecustomize_bootstrap
shim_marker_file = _sft_train.shim_marker_file
verify_applied_shim_markers = _sft_train.verify_applied_shim_markers
SHIM_FRAGMENT_FAILED_EXIT_CODE = _sft_train.SHIM_FRAGMENT_FAILED_EXIT_CODE
sft_tokens_for_updates = _sft_train.sft_tokens_for_updates
sft_under_ran = _sft_train.sft_under_ran
validate_save_steps = _sft_train.validate_save_steps
_render_sft_dataset_module = _sft_train._render_sft_dataset_module


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
    processor: object | None
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
    exclude_modules: str | None
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
    # the micro-batch verl RAN, which is `model.micro_batch` capped to one rank's share of the batch.
    # carried because the result file reports it: recording the uncapped request would tell a reader
    # reconstructing the token budget that each rank held twice the rows it did.
    micro_batch: int
    shim_markers: str
    expected_shims: tuple[str, ...]


@dataclass
class _SftProgress:
    values: dict[str, float | int | None]
    zero_grad_steps: list[int]
    observed_grad_norms: list[float]
    loss_curve: list[float]
    train_tokens: int
    loraplus_applied: bool
    wandb_link: dict[str, str | None]
    # what the child plugin must prove applied; verified at the first optimizer step and again at
    # child exit.
    shim_markers: str = ""
    expected_shims: tuple[str, ...] = ()
    shims_verified: bool = False


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
    spec = spec or _worker_state.JOB_SPEC
    env = _worker_state.require_active_env()
    # the child trainer is seeded through its shim, but the environment's dataset/completion calls
    # run HERE in the parent. without this the documented top-level seed no longer reproduces sft
    # targets for any env whose row construction uses python/numpy randomness.
    _sft_train.seed_training_rngs(_worker_state.SEED)
    started_at = time.time()
    _worker_heartbeat.heartbeat("sft_start", gpu=_worker_perf.gpu_diagnostics(include_torch=False))
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

    workdir = os.path.join(
        "/tmp", "flash-sft-verl", _worker_state.RUN_ID, f"seed-{_worker_state.SEED}"
    )
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
        tokenizer_loader=lambda candidate, revision: _worker_hf.load_tokenizer(
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
    rows = prepared_workload.rows
    realized_profile = prepared_workload.profile
    # the control-plane profile uses raw packaged record fields and deliberately does not execute
    # environment.py. training does execute the environment, so filtering and prompt construction may
    # change token totals. keep the quote profile as the packing and step contract while using the
    # recomputed measurements and rows for the environment-produced training data.
    if expected_profile.packing_mode == "packed" and realized_profile.packing_mode != "packed":
        raise RuntimeError(
            "the accepted sft quote requires packed execution, but this worker cannot reproduce its "
            "boundary-safe packing contract"
        )
    if expected_profile.examples_per_update > len(rows):
        raise RuntimeError(
            "the accepted sft quote requires more examples per update than the environment retained; "
            "refusing to change the accepted training and billing contract"
        )
    if realized_profile != expected_profile:
        print(
            "[sft][warn] environment processing changed the packaged-dataset token estimate; "
            "training continues on the environment-produced rows, and the accepted quote is unchanged"
        )
    profile = expected_profile
    max_length = _sft_train._sft_profile_max_length(realized_profile)
    dropped = realized_profile.dropped_examples
    selected_count = realized_profile.selected_examples
    retained_count = realized_profile.retained_examples
    sampled_texts = prepared_workload.sampled_texts
    multiturn_targets = prepared_workload.multiturn_targets
    coerced_singleturn_targets = prepared_workload.coerced_singleturn_targets
    role_aware_multiturn_targets = prepared_workload.role_aware_multiturn_targets
    fallback_multiturn_targets = prepared_workload.fallback_multiturn_targets
    if dropped:
        print(
            f"[sft] dropped {dropped} rows with no real completion target "
            "(sft_max_len truncated away the whole completion, or it was content-free)"
        )
    if role_aware_multiturn_targets:
        print(
            f"[sft] multi-turn SFT: {role_aware_multiturn_targets}/{retained_count} rows use "
            "assistant-body masking; interleaved environment/tool/user observations are masked "
            "out of the loss"
        )
    if fallback_multiturn_targets:
        print(
            f"[sft][warn] multi-turn SFT: {fallback_multiturn_targets}/{retained_count} rows use "
            "completion-only fallback because the rendered transcript was not parseable ChatML; "
            "interleaved observations are not proven masked"
        )
    if not multiturn_targets and getattr(options.env, "multi_turn", False):
        print(
            "[sft][warn] this is a multi-turn environment but no row ships a multi-turn "
            "target completion"
        )
    if selected_count > 1 and coerced_singleturn_targets == selected_count:
        print(
            f"[sft][warn] all {selected_count} selected rows use one bare assistant target coerced "
            "from raw output; reasoning blocks or multi-turn structure may have been lost. encode "
            "full target trajectories as message lists in output"
        )
    if _worker_state.THINKING and not any("<think>" in text for text in sampled_texts[:256]):
        print(
            "WARN: thinking mode is ON but no sampled SFT target contains a <think> trace; "
            "training on non-reasoning targets teaches the model to skip thinking"
        )
    # the partial case -- the template keeping only the last turn's reasoning and stripping the
    # rest -- renders a <think>, so the all-or-nothing check above stays quiet for it. it is
    # reported by `prepare_sft_workload`, which this function already called, so that one warning
    # reaches both the control-plane estimate and this worker log without being printed twice.

    total_tokens_per_epoch = realized_profile.real_tokens_per_epoch
    realized_max_length = realized_profile.realized_max_length
    masked_tokens = total_tokens_per_epoch - realized_profile.supervised_tokens_per_epoch
    print(
        f"[sft] completion-only loss: masking {masked_tokens}/{total_tokens_per_epoch} "
        f"({masked_tokens / total_tokens_per_epoch:.0%}) prompt tokens"
    )
    train_file = os.path.join(options.paths.data_dir, "train.parquet")
    _sft_train._write_sft_parquet(rows, train_file)
    return _SftData(
        rows=rows,
        multimodal=prepared_workload.multimodal,
        processor=prepared_workload.processor,
        profile=profile,
        max_length=max_length,
        realized_max_length=realized_max_length,
        train_file=train_file,
    )


def _prepare_sft_model(options: _SftOptions, data: _SftData) -> _SftModelSetup:
    from flash.core.catalog import MODELS
    from flash.engine.plan.vram import sft_chunked_nll_enabled

    download_seconds = _worker_hf.prefetch_model(options.model_id, revision=options.model_revision)
    setup_seconds = time.time() - options.started_at
    _worker_heartbeat.heartbeat(
        "sft_model_load",
        setup_seconds=setup_seconds,
        gpu=_worker_perf.gpu_diagnostics(include_torch=False),
    )
    # everything below reads adapter/tokenizer/architecture config from the hub or cache, which is
    # minutes on a cold mount and emits nothing of its own. without this the run's last ping is the
    # one-shot above, so `runs status` freezes on sft_model_load for the whole span and a healthy
    # cold cache is indistinguishable from a dead worker -- the exact ambiguity the stage was added
    # to resolve. same stage name, so the provider's setup-grace classification is unchanged.
    with liveness_heartbeat("sft_model_load"):
        lora_config = _worker_adapter.make_lora(options.model_id)
        targeting = resolve_lora_targeting(
            options.model_id, algorithm="sft", multimodal=data.multimodal
        )
        lora_rank = int(lora_config.r)
        target_modules = targeting.target_modules
        if isinstance(target_modules, set | frozenset):
            target_modules = sorted(target_modules)
        warmstart_adapter = _sft_train._warmstart_adapter_path(
            options.model_id,
            options.model_revision,
            lora_rank,
            int(lora_config.lora_alpha),
            targeting,
        )
        vocab_size = _sft_train._resolve_sft_vocab_size(options.model_id, options.model_revision)
        # hoisted into the span: on a PINNED revision this falls through to a live AutoConfig read
        # with no local_files_only, so it is the same cold-mount/hub stall as the reads above. it
        # only needs the model id, and everything between here and its old call site is arithmetic
        # on already-resolved values, so moving it up changes ordering but not results.
        hidden, layers = _sft_train._model_arch_dims(
            options.model_id, revision=options.model_revision
        )
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
    save_freq = (
        reduce(gcd, options.save_at_steps)
        if options.save_at_steps
        else max(1, min(options.save_every, update_horizon))
    )

    card_vram_gb = float(options.gpu_probe.get("memory_gb") or 0.0)
    raw_capability = options.gpu_probe.get("capability")
    capability = tuple(raw_capability) if raw_capability else None
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
        exclude_modules=targeting.exclude_modules,
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


def _widest_rentable_width(max_cards: int, train_batch_size: int, row_count: int) -> int:
    """Widest allocation that is both rentable and fully usable by SFT, at most ``max_cards``.

    Two constraints, and only counts meeting BOTH are worth advising: providers sell powers of two
    (`rentable_gpu_counts`), and SFT needs the width to divide the batch and the row count or it is
    back to idling cards. Neither implies the other, so this searches the rentable shapes rather
    than clamping a resolved width -- `sft_data_parallel_cards` walks downward to find a divisor and
    happily returns 3 or 5, which are not shapes anyone can allocate. 1 always qualifies.
    """
    return widest_usable_dp_width(rentable_gpu_counts(max_cards), train_batch_size, row_count)


def _idle_card_warning(
    world_size: int, gpu_count: int, train_batch_size: int, row_count: int
) -> str:
    """The live `[sft][warn]` line naming the binding input and a remedy that can actually work.

    The run is BILLED for every allocated card, so a card the batch cannot feed is money spent on an
    idle gpu. The notes carry the width too, but those are read after the fact -- say it while the
    run is live, and say what would actually use the card.

    The remedy differs by which input is binding, and a remedy that cannot be acted on is worse than
    none. BOTH inputs have to divide the allocation, so the batch is only worth naming as a knob
    when the rows already fit: at 4 cards with a batch of 6 over 10 rows, every multiple of 4 the
    batch could be raised still resolves to 2 ranks because 10 rows cannot be split 4 ways. Batch 1
    is a third case rather than a degenerate version of either -- it binds, but `sft_workload` fixes
    it at 1 for every unpacked run, so it is named as a fact and never as a knob.

    A card count BELOW the launched width is qualified rather than prescribed. The ranks that joined
    are what hold the model, so shrinking the allocation is a VRAM change and not only a billing one:
    a 27B at 32k needs 159 GB, runs on the 191.6 GB that 3 H100 ranks provide, and would be rejected
    at the 130.4 GB two cards give. This runs after allocation with no VRAM need in scope and cannot
    check that itself, so it says the smaller shape is worth checking instead of asserting it works.
    """
    rows_fit = row_count == 0 or row_count % gpu_count == 0
    batch_fits = train_batch_size % gpu_count == 0
    unpacked = train_batch_size <= 1
    # the batch is only worth naming as a knob when fixing it is sufficient, i.e. the rows already
    # fit. an unpacked batch of 1 is named as a fact instead: it binds, but it cannot be raised.
    batch_helps = rows_fit and not batch_fits and not unpacked
    rentable = _widest_rentable_width(world_size, train_batch_size, row_count)
    if unpacked:
        limiter = "an unpacked run's single example per update"
    elif batch_helps:
        limiter = f"a batch of {train_batch_size}"
    else:
        limiter = f"a dataset of {row_count} rows"
    # dropping below the launched width takes memory away from a run the fit gate already accepted
    # on this shape, so it is offered as a question rather than as the fix.
    card_advice = (
        f"allocate {rentable} card(s) instead"
        if rentable >= world_size
        else f"allocate {rentable} card(s) instead if the run still fits on {rentable}"
    )
    remedy = (
        f"raise [train] batch_size to a multiple of {gpu_count}, or {card_advice}"
        if batch_helps
        else card_advice
    )
    return (
        f"[sft][warn] training on {world_size} of {gpu_count} allocated cards: {limiter} "
        f"cannot be split across {gpu_count} ranks without starving one or dropping rows. "
        f"the idle cards are still billed -- {remedy}."
    )


def _resolve_sft_world_size(gpu_count: int, train_batch_size: int, row_count: int) -> int:
    """Ranks to launch, warning when that is fewer than the cards the run is paying for.

    SFT shards by DATA, not by sequence: ulysses is pinned to 1 and fsdp splits the batch. verl's
    ulysses support patches `_flash_attention_forward` and slices the qwen text model's inputs, but
    passes NO state between ranks -- and every catalog model is a GatedDeltaNet hybrid whose layers
    are mostly linear attention plus a causal conv, both of which carry state along the sequence. A
    sequence shard would run its recurrence and conv as if it were a whole sequence, so sequence
    parallelism is not merely unimplemented for this family, it is incorrect for it.

    It also crashed outright. remove-padding flattens the batch to one `(1, total_nnz)` row, and
    verl slices that row per rank, so the shapes reaching the GDN kernels stop agreeing and a
    sharded run died on `seq_idx must have shape (batch_size, seqlen)` -- at any batch size,
    including 1, because "remove padding" leaves no batch dimension to keep an example whole.

    The width must also divide the ROW count, or verl's sampler drops the remainder from every
    epoch while the quote still bills it. See `sft_data_parallel_cards`.
    """
    world_size = sft_data_parallel_cards(gpu_count, train_batch_size, row_count)
    if world_size < gpu_count:
        print(_idle_card_warning(world_size, gpu_count, train_batch_size, row_count))
    return world_size


def _resolve_sft_width_and_micro_batch(
    options: _SftOptions, data: _SftData, model: _SftModelSetup
) -> tuple[int, int]:
    """The data-parallel width and the micro-batch that fits one rank's share of the batch.

    Resolved together because the second depends on the first: the micro-batch was sized against
    the GLOBAL batch, but each rank only ever receives ``train_batch_size // world_size``. verl
    rejects a micro-batch larger than the per-rank batch before the first optimizer step, so batch
    8 across 4 ranks -- per-rank 2 -- cannot keep a micro-batch of 4. The token budget derives from
    the capped value too, or it reserves for rows the rank never receives.
    """
    world_size = _resolve_sft_world_size(options.gpu_count, model.train_batch_size, len(data.rows))
    micro_batch = max(1, min(model.micro_batch, model.train_batch_size // max(1, world_size)))
    return world_size, micro_batch


def _resolve_sft_run_identity(
    options: _SftOptions, capabilities: _SftCapabilities
) -> tuple[list[str], str, str]:
    """The child's ``(loggers, project_name, experiment_name)``."""
    # verl logs from the verl interpreter, so gate wandb on THAT env (see resolve_verl_loggers).
    loggers = _sft_train.resolve_verl_loggers(capabilities.caps)
    project_name = (
        options.spec.wandb.project if options.spec and options.spec.wandb else None
    ) or "flash"
    return loggers, project_name, _worker_wandb.wandb_run_name()


def _write_sft_child_shims(
    options: _SftOptions,
    model: _SftModelSetup,
    *,
    shim_dir: str,
    custom_dataset_path: str,
    seed: int,
    loggers: list[str],
    gdn_reset_arch: str | None,
    multimodal: bool = False,
) -> tuple[str, tuple[str, ...], str]:
    """write the SFT plugin bundle, startup bootstrap, and non-secret plugin config."""
    # the bundle paths below are relative to flash/engine/worker/; this module now lives in
    # worker/train/entry/, so climb back out of train/entry rather than anchoring here.
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(_sft_train.__file__)))
    copies = (
        ("train/core/child/runtime.py", "flash_verl_runtime.py"),
        ("train/sft/child/plugin.py", "flash_sft_plugin.py"),
        ("train/sft/child/entry.py", "flash_sft_entry.py"),
    )
    for source, target in copies:
        shutil.copy2(os.path.join(parent_dir, *source.split("/")), os.path.join(shim_dir, target))
    shim_markers = shim_marker_file(shim_dir)
    with open(os.path.join(shim_dir, "sitecustomize.py"), "w", encoding="utf-8") as file:
        file.write(render_sitecustomize_bootstrap())
    with open(custom_dataset_path, "w", encoding="utf-8") as file:
        file.write(_render_sft_dataset_module())
    text_only = bool(getattr(model, "exclude_modules", None))
    plugin_config = json.dumps(
        {
            "marker_file": shim_markers,
            "seed": int(seed),
            "loraplus_ratio": float(_SFT_LORAPLUS_RATIO),
            "loraplus_ready_marker": _LORAPLUS_READY_MARKER,
            "save_at_steps": list(options.save_at_steps),
            "total_steps": int(model.update_horizon),
            "reentrant_gradient_checkpointing": bool(model.reentrant_gradient_checkpointing),
            "lora_language_prefix": (
                get_model(options.model_id).lora_language_prefix if text_only else ""
            ),
            "multimodal": bool(multimodal),
            "gdn_model_type": gdn_reset_arch,
            "wandb": "wandb" in loggers,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = ("sft-core",)
    if text_only:
        expected += (TEXT_LORA_TARGET_SHIM,)
    if gdn_reset_arch:
        expected += ("gdn-varlen",)
    return shim_markers, expected, plugin_config


def _prepare_sft_child(
    options: _SftOptions,
    data: _SftData,
    model: _SftModelSetup,
    capabilities: _SftCapabilities,
    use_remove_padding: bool,
    gdn_reset_arch: str | None,
) -> _SftChild:
    model_path = _sft_train._cached_model_path(options.model_id, options.model_revision)
    loggers, project_name, experiment_name = _resolve_sft_run_identity(options, capabilities)
    shim_dir = os.path.join(options.paths.workdir, "shim")
    os.makedirs(shim_dir, exist_ok=True)
    custom_dataset_path = os.path.join(shim_dir, "flash_verl_sft_dataset.py")

    # the gdn boundary shim resets conv and recurrent state at packed example boundaries, but only
    # when the verl child has the kernels that read seq_idx and cu_seqlens; the no-fla fallbacks
    # accept both and discard them. so the shim is installed only when the child proves it can reset.
    # `gdn_reset_arch` is resolved by the caller, inside the configuring liveness wrap, because the
    # probe is part of the setup silence that wrap exists to cover, and because a packed run must
    # take the RAISING gate there rather than the soft form.
    world_size, micro_batch = _resolve_sft_width_and_micro_batch(options, data, model)
    target_parameters = lora_target_parameters(options.model_id)
    fsdp_generation = _resolve_fsdp_generation("sft", target_parameters)
    config = {
        "train_files": data.train_file,
        "train_batch_size": model.train_batch_size,
        "max_length": data.max_length,
        "micro_batch": micro_batch,
        "max_token_len_per_gpu": data.realized_max_length * micro_batch,
        "custom_dataset_path": custom_dataset_path,
        "model_path": model_path,
        "lora_rank": model.lora_rank,
        "lora_alpha": model.lora_alpha,
        "target_modules": model.target_modules,
        "exclude_modules": None,
        "target_parameters": target_parameters,
        "fsdp_generation": fsdp_generation,
        "lora_adapter_path": model.warmstart_adapter,
        "ulysses_sp_size": ULYSSES_SEQUENCE_PARALLEL_SIZE,
        "lr": options.learning_rate,
        "warmup_ratio": RECIPE.sft.warmup_frac,
        "optimizer_impl": _VERL_OPTIMIZER_IMPL,
        "optimizer_name": _VERL_OPTIMIZER_NAME,
        "optimizer_kwargs": None,
        "local_dir": options.paths.local_dir,
        "save_freq": model.save_freq,
        "n_gpus_per_node": world_size,
        "seed": _worker_rng.backend_seed(_worker_state.SEED),
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
        # the accepted quote's step count binds whether or not the user authored max_steps. the
        # plane profiles raw records without running environment.py, so an environment that expands
        # or replaces rows makes steps_per_epoch larger than the quote assumed; without an explicit
        # cap the loop would run a full realized epoch past the horizon the run was priced and
        # wall-budgeted for. build_sft_overrides rejects setting both, so the horizon replaces the
        # epoch count rather than joining it -- loop_epochs already covers the authored epochs.
        "total_training_steps": model.update_horizon,
        "total_epochs": None,
        "use_remove_padding": use_remove_padding,
        # resolved from the out-of-process capability probe, never by opening cuda in this parent.
        "fused_ce_backend": _sft_train._resolve_sft_fused_ce_backend(capabilities.caps),
    }
    overrides = build_sft_overrides(config)
    shim_markers, expected_shims, plugin_config = _write_sft_child_shims(
        options,
        model,
        shim_dir=shim_dir,
        custom_dataset_path=custom_dataset_path,
        seed=config["seed"],
        loggers=loggers,
        gdn_reset_arch=gdn_reset_arch,
        multimodal=data.multimodal,
    )

    # the RESOLVED width, not the allocated card count: it is what becomes --nproc-per-node below,
    # so the guard compares the checkpoint against the topology this attempt really launches. sft
    # shards by data and the width is bounded by the batch and the row count, so the two differ
    # whenever either fails to divide the allocation -- passing gpu_count here would discard a
    # checkpoint that matches the run about to start, and keep one that does not.
    resume_step = _sft_train._restore_verl_resume(
        options.paths.local_dir,
        world_size=world_size,
        expected_fsdp_generation=fsdp_generation,
    )
    watcher = _sft_train._VerlCheckpointWatcher(
        local_dir=options.paths.local_dir,
        export_root=options.paths.export_root,
        python_bin=capabilities.python_bin,
        model_id=options.model_id,
        model_revision=options.model_revision,
        exclude_modules=model.exclude_modules,
        required_steps=options.save_at_steps,
        preprocessor=data.processor,
    )
    # the staged resume checkpoint is already a pending global_step_N on disk, so an unseeded
    # watcher re-merges it and re-uploads full state hf already has, holding the resume-upload
    # lock while the first genuinely new checkpoint waits behind it.
    _sft_train._seed_resume_lifecycle(watcher, options.save_at_steps, resume_step)
    if resume_step >= model.update_horizon:
        missing = watcher.lifecycle.missing_deployables(watcher.required_steps)
        if missing:
            raise RuntimeError(f"required saves were not durably published: {missing}")
    child_env = _sft_train._build_verl_child_env(
        shim_dir=shim_dir,
        wandb_enabled="wandb" in loggers,
    )
    child_env["VERL_USE_EXTERNAL_MODULES"] = "flash_sft_plugin"
    child_env["FLASH_SFT_PLUGIN_CONFIG"] = plugin_config
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
        "flash_sft_entry",
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
        micro_batch=micro_batch,
        shim_markers=shim_markers,
        expected_shims=expected_shims,
    )


def _prepare_sft_progress(data: _SftData, model: _SftModelSetup, child: _SftChild) -> _SftProgress:
    resume_step = child.resume_step
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
        shim_markers=child.shim_markers,
        expected_shims=child.expected_shims,
        # like loraplus_applied above: a resume already at the horizon never launches the child,
        # so there is no marker file to verify and nothing left for the shims to patch.
        shims_verified=resume_step >= model.update_horizon,
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
        _worker_heartbeat.heartbeat(
            "sft_step", **{key: value for key, value in payload.items() if value is not None}
        )

    def child_heartbeat(self) -> None:
        _worker_heartbeat.heartbeat(
            "sft_step", liveness=True, step=int(self.progress.values["step"] or 0)
        )


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
    # the marker set first: a skipped sitecustomize also loses the lora+ shim, and "no fragment
    # ran at all" is the root cause worth reporting over its lora+ symptom.
    if progress.expected_shims and not progress.shims_verified:
        verify_applied_shim_markers(progress.shim_markers, progress.expected_shims)
        progress.shims_verified = True
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
    if return_code == SHIM_FRAGMENT_FAILED_EXIT_CODE:
        # permanent, not retriable infra: the same interpreter fails the same fragment on retry.
        raise RuntimeError(
            f"verl SFT subprocess exited {return_code}: a required flash runtime patch failed to "
            "apply in the child interpreter (its traceback names the fragment in the flash log). "
            "the verl/transformers stack at the child python is incompatible with this flash "
            "version; rebuild the worker image or fix FLASH_VERL_PYTHON rather than retrying."
        )
    if return_code != 0:
        raise RuntimeError(f"verl SFT subprocess exited with status {return_code}")
    # belt and braces behind the first-step check in _consume_sft_marker_line, for a child whose
    # step lines the parent never parsed. shims_verified starts True when the child is not
    # launched at all (resume already at the horizon), exactly like loraplus_applied above.
    if progress.expected_shims and not progress.shims_verified:
        verify_applied_shim_markers(progress.shim_markers, progress.expected_shims)
        progress.shims_verified = True
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
    if sft_under_ran(final_step, model.update_horizon):
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
