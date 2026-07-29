"""SFT training path (TRL SFTTrainer) for the fine-tuning worker."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import tempfile
import time
from pathlib import Path

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.engine.steps import (
    configure_trainer_save_schedule,
    final_save_due,
    resolve_update_horizon,
    sft_update_steps,
    validate_save_steps,
)
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.packing import (
    BlockDiagonalCollator,
    completion_mask_from_ids,
    gdn_packing_available,
    model_is_gdn_hybrid,
    model_is_pure_attention,
    pack_token_ids,
    packing_efficiency,
    tokenize_for_packing,
)
from flash.engine.worker.perf import (
    _arch_supports_attn_impl,
    _flash_attn_available,
    _GpuPeakSampler,
    _memory_mode,
    _metric_curve,
    _peak_gpu_gb,
    _reset_peak_gpu,
    _sdpa_cudnn_ctx,
    free_gpu,
    fused_optim_name,
    gpu_diagnostics,
    grad_checkpointing_on,
    grpo_use_reentrant,
    loraplus_optimizer_cls,
    make_multimodal_input_require_grads_callback,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.rng import backend_seed, seed_training_rngs

_SFT_PREP_CACHE_VERSION = 1
_SFT_TOKENIZE_MAX_PROCS = 8
_SFT_TOKENIZE_ROWS_PER_PROC = 512
_SFT_PREP_FINGERPRINT_FIELDS = (
    "env_resolved_sha",
    "dataset_prefix",
    "seed",
    "order",
    "model_revision",
    "tokenizer_identity",
    "chat_template",
    "thinking",
    "max_length",
)


def _stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sft_prep_fingerprint(**fields) -> str:
    missing = set(_SFT_PREP_FINGERPRINT_FIELDS) - fields.keys()
    extra = fields.keys() - set(_SFT_PREP_FINGERPRINT_FIELDS)
    if missing or extra:
        raise ValueError(
            f"invalid SFT prep fingerprint fields: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    payload = {"version": _SFT_PREP_CACHE_VERSION, **fields}
    return hashlib.sha256(_stable_json(payload).encode()).hexdigest()


def _tokenizer_identity(tokenizer) -> str:
    import transformers

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        tokenizer_state = backend.to_str().encode()
    else:
        try:
            tokenizer_state = pickle.dumps(tokenizer, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, TypeError, AttributeError) as exc:
            raise TypeError("slow tokenizer state must be picklable for SFT caching") from exc
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    init_config = json.dumps(
        init_kwargs,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=repr,
    )
    identity = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": str(getattr(tokenizer, "name_or_path", "") or ""),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}) or {},
        "transformers_version": transformers.__version__,
        "truncation_side": str(getattr(tokenizer, "truncation_side", "right") or "right"),
        "add_bos_token": bool(getattr(tokenizer, "add_bos_token", False)),
        "add_eos_token": bool(getattr(tokenizer, "add_eos_token", False)),
        "init_config_sha256": hashlib.sha256(init_config.encode()).hexdigest(),
        "tokenizer_state_sha256": hashlib.sha256(tokenizer_state).hexdigest(),
    }
    return hashlib.sha256(_stable_json(identity).encode()).hexdigest()


def _normalize_sft_records(
    env,
    examples,
    *,
    prefix_indices: list[int] | None = None,
    prompt_completion_rows: list[tuple] | None = None,
) -> tuple[list[dict], int]:
    if prefix_indices is None:
        prefix_indices = list(range(len(examples)))
    if len(prefix_indices) != len(examples):
        raise ValueError("prefix indices must match the selected SFT examples")
    if prompt_completion_rows is None:
        prompt_completion_rows = [
            (env.prompt_messages(example), env.sft_completion(example)) for example in examples
        ]
    if len(prompt_completion_rows) != len(examples):
        raise ValueError("normalized prompt rows must match the selected SFT examples")
    records = []
    multiturn_targets = 0
    for prefix_index, (prompt_messages, completion) in zip(
        prefix_indices, prompt_completion_rows, strict=True
    ):
        if len(completion) > 1:
            multiturn_targets += 1
        try:
            prompt_bytes = pickle.dumps(prompt_messages, protocol=pickle.HIGHEST_PROTOCOL)
            completion_bytes = pickle.dumps(completion, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, TypeError, AttributeError) as exc:
            raise TypeError(
                "normalized SFT prompt and completion messages must be picklable"
            ) from exc
        records.append(
            {
                "prefix_index": prefix_index,
                "prompt_messages": prompt_bytes,
                "completion_messages": completion_bytes,
            }
        )
    return records, multiturn_targets


def _normalized_records_digest(records: list[dict], *, prefix_order: bool) -> str:
    ordered = (
        sorted(records, key=lambda record: record["prefix_index"]) if prefix_order else records
    )
    digest = hashlib.sha256()
    for record in ordered:
        for name in ("prompt_messages", "completion_messages"):
            value = record[name]
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _sft_tokenize_num_proc(row_count: int) -> int | None:
    cpu_limit = max(1, (os.cpu_count() or 1) // 2)
    workers = min(row_count // _SFT_TOKENIZE_ROWS_PER_PROC, _SFT_TOKENIZE_MAX_PROCS, cpu_limit)
    return workers if workers >= 2 else None


def _render_tokenize_sft_batch(batch, *, tokenizer, max_length: int, thinking: bool):
    prompt_messages = [pickle.loads(value) for value in batch["prompt_messages"]]
    completion_messages = [pickle.loads(value) for value in batch["completion_messages"]]
    texts = [
        tokenizer.apply_chat_template(
            [*prompt, *completion],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=thinking,
        )
        for prompt, completion in zip(prompt_messages, completion_messages, strict=True)
    ]
    prompt_texts = [
        tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        for prompt in prompt_messages
    ]
    full_ids = tokenize_for_packing(texts, tokenizer, max_length)
    prompt_ids = tokenizer(prompt_texts, truncation=True, max_length=max_length)["input_ids"]
    completion_masks = [
        completion_mask_from_ids(prompt, full)
        for prompt, full in zip(prompt_ids, full_ids, strict=True)
    ]
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    has_real_target = [
        any(mask and token_id not in special_ids for token_id, mask in zip(ids, masks, strict=True))
        for ids, masks in zip(full_ids, completion_masks, strict=True)
    ]
    return {
        "text": texts,
        "prompt_text": prompt_texts,
        "input_ids": full_ids,
        "completion_mask": completion_masks,
        "has_real_target": has_real_target,
    }


def _sft_prep_cache_root() -> Path:
    from datasets import config as datasets_config

    configured = os.environ.get("HF_DATASETS_CACHE")
    return Path(configured or datasets_config.HF_DATASETS_CACHE) / "flash" / "sft-prep"


def _load_or_prepare_sft_dataset(
    records: list[dict],
    tokenizer,
    *,
    fingerprint: str,
    max_length: int,
    thinking: bool,
    cache_root: Path | None = None,
):
    import multiprocess
    from datasets import Dataset, load_from_disk

    root = cache_root or _sft_prep_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    cache_dir = root / fingerprint
    dataset_dir = cache_dir / "dataset"
    complete = cache_dir / "_SUCCESS"
    if complete.is_file():
        return load_from_disk(str(dataset_dir)), True

    lock_path = root / f"{fingerprint}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if complete.is_file():
            return load_from_disk(str(dataset_dir)), True
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        for stale_temp_dir in root.glob(f".{fingerprint}.*"):
            if stale_temp_dir.is_dir():
                shutil.rmtree(stale_temp_dir)

        temp_dir = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=root))
        try:
            num_proc = _sft_tokenize_num_proc(len(records))
            if num_proc is not None:
                # spawn keeps heartbeat threads and accelerator state out of tokenizer workers.
                multiprocess.set_start_method("spawn", force=True)
            source = Dataset.from_list(records)
            previous_tokenizer_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
            if num_proc is not None:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
            try:
                mapped = source.map(
                    _render_tokenize_sft_batch,
                    batched=True,
                    batch_size=256,
                    remove_columns=source.column_names,
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "max_length": max_length,
                        "thinking": thinking,
                    },
                    num_proc=num_proc,
                    cache_file_name=str(temp_dir / "map.arrow"),
                    new_fingerprint=fingerprint[:40],
                    desc="rendering and tokenizing SFT dataset",
                )
            finally:
                if previous_tokenizer_parallelism is None:
                    os.environ.pop("TOKENIZERS_PARALLELISM", None)
                else:
                    os.environ["TOKENIZERS_PARALLELISM"] = previous_tokenizer_parallelism
            mapped.save_to_disk(str(temp_dir / "dataset"))
            for map_shard in temp_dir.glob("map*.arrow"):
                map_shard.unlink()
            marker = temp_dir / "_SUCCESS"
            with marker.open("x") as marker_file:
                marker_file.write(f"{fingerprint}\n")
                marker_file.flush()
                os.fsync(marker_file.fileno())
            os.rename(temp_dir, cache_dir)
        except BaseException:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    return load_from_disk(str(dataset_dir)), False


def _prepare_sft_examples(
    env,
    examples,
    tokenizer,
    *,
    env_resolved_sha: str,
    seed: int,
    model_revision: str,
    thinking: bool,
    max_length: int,
    prefix_indices: list[int] | None = None,
    prompt_completion_rows: list[tuple] | None = None,
    cache_root: Path | None = None,
):
    normalized, multiturn_targets = _normalize_sft_records(
        env,
        examples,
        prefix_indices=prefix_indices,
        prompt_completion_rows=prompt_completion_rows,
    )
    if not normalized:
        return [], [], 0, multiturn_targets, False
    fingerprint = _sft_prep_fingerprint(
        env_resolved_sha=env_resolved_sha,
        dataset_prefix=_normalized_records_digest(normalized, prefix_order=True),
        seed=seed,
        order=_normalized_records_digest(normalized, prefix_order=False),
        model_revision=model_revision,
        tokenizer_identity=_tokenizer_identity(tokenizer),
        chat_template=str(getattr(tokenizer, "chat_template", "") or ""),
        thinking=thinking,
        max_length=max_length,
    )
    mapped, cache_hit = _load_or_prepare_sft_dataset(
        normalized,
        tokenizer,
        fingerprint=fingerprint,
        max_length=max_length,
        thinking=thinking,
        cache_root=cache_root,
    )
    rows = mapped.to_list()
    kept = [row for row in rows if row["has_real_target"]]
    texts = [{"text": row["text"], "prompt_text": row["prompt_text"]} for row in kept]
    pretok = [
        {"input_ids": row["input_ids"], "completion_mask": row["completion_mask"]} for row in kept
    ]
    return texts, pretok, len(rows) - len(kept), multiturn_targets, cache_hit


def _pretokenize_completion_only(texts, tokenizer, max_length):
    """Pre-tokenize SFT rows into ``{input_ids, completion_mask}``, dropping rows with no real completion target.

    Returns ``(kept_texts, pretok, n_dropped)``.
    """
    full_ids = tokenize_for_packing([t["text"] for t in texts], tokenizer, max_length)
    prompt_ids = tokenizer(
        [t["prompt_text"] for t in texts], truncation=True, max_length=max_length
    )["input_ids"]
    pretok = [
        {"input_ids": ids, "completion_mask": completion_mask_from_ids(pids, ids)}
        for ids, pids in zip(full_ids, prompt_ids, strict=True)
    ]
    special_ids = set(getattr(tokenizer, "all_special_ids", None) or [])

    def _has_real_target(row) -> bool:
        return any(
            m and tid not in special_ids
            for tid, m in zip(row["input_ids"], row["completion_mask"], strict=True)
        )

    kept = [(t, r) for t, r in zip(texts, pretok, strict=True) if _has_real_target(r)]
    return [t for t, _ in kept], [r for _, r in kept], len(pretok) - len(kept)


def _model_arch_dims(model_id: str, revision: str = "") -> tuple[int, int]:
    """``(hidden_size, num_hidden_layers)`` used to size the GC-off activation estimate.

    Prefer the CURATED catalog geometry (deterministic, no network/parse risk) for known models — a
    live B200 SFT showed the runtime ``AutoConfig`` probe returning (0, 0) on the 35B-A3B's
    multimodal-nested config, which silently kept GC on. For open-model-policy ids (no catalog dims)
    fall back to the HF config, handling the ``text_config`` nesting (config.json is already cached by
    the tokenizer load). Best-effort: ``(0, 0)`` if neither is available -> the GC-off gate
    conservatively keeps gradient checkpointing on."""
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    c_hidden = int(getattr(info, "hidden_size", 0) or 0) if info else 0
    c_layers = int(getattr(info, "num_layers", 0) or 0) if info else 0
    if c_hidden and c_layers and not revision:
        return c_hidden, c_layers
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            **_w.model_revision_kwargs(revision),
        )
        tc = getattr(cfg, "text_config", None) or cfg
        layers = int(
            getattr(tc, "num_hidden_layers", 0) or getattr(cfg, "num_hidden_layers", 0) or 0
        )
        hidden = int(getattr(tc, "hidden_size", 0) or getattr(cfg, "hidden_size", 0) or 0)
        # Only a NONZERO probe dim that disagrees with the catalog is a real revision mismatch. A (0, 0)
        # probe means "couldn't parse" (the 35B-A3B multimodal-nested config does exactly this), not
        # "differs" -- treat it like the unpinned path and fall back to catalog dims below, otherwise a
        # revision pin would spuriously fail such models after the GPU is already rented.
        if revision and (
            (c_hidden and hidden and hidden != c_hidden)
            or (c_layers and layers and layers != c_layers)
        ):
            raise ValueError("revision architecture does not match catalog geometry")
        return hidden or c_hidden, layers or c_layers
    except Exception as e:
        if revision:
            raise RuntimeError("could not validate revision-specific model architecture") from e
        print(f"[sft] arch-dims probe failed ({e}); GC decision stays conservative (keep GC on)")
        return c_hidden, c_layers


_CHUNKED_NLL_TEXT_CONFIG_FIELDS = (
    "final_logit_softcapping",
    "logit_scale",
)


def _prepare_chunked_nll_model(model, processing_class, peft_config) -> None:
    """validate trl's chunked-nll traversal and expose nested text-loss config on vl qwen models."""
    from transformers import PreTrainedTokenizerBase

    if isinstance(model, str):
        raise RuntimeError("chunked_nll requires a preloaded model so its lm head can be validated")
    if not isinstance(processing_class, PreTrainedTokenizerBase):
        raise RuntimeError("chunked_nll expects flash's text tokenizer processing path")

    output_head = model.get_output_embeddings()
    if output_head is None or not hasattr(output_head, "weight"):
        raise RuntimeError("chunked_nll requires a directly accessible output embedding weight")
    if model.base_model is model:
        raise RuntimeError("chunked_nll requires a distinct backbone that bypasses the lm head")

    targets = getattr(peft_config, "target_modules", None)
    modules_to_save = set(getattr(peft_config, "modules_to_save", None) or ())
    if targets != "all-linear" or "lm_head" in modules_to_save:
        raise RuntimeError(
            "chunked_nll requires PEFT all-linear targeting with the output layer excluded"
        )

    # flash intentionally passes an autotokenizer for text-only qwen training, so trl classifies the
    # full multimodal checkpoint as a text model. transformers 5 still traverses model.base_model
    # correctly, but trl reads logit transforms from the top-level config in that classification.
    # mirror only those transforms. the full qwen moe wrapper does not add router auxiliary loss on
    # the plain-nll path, so copying its nested router flag would change the training objective.
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        for name in _CHUNKED_NLL_TEXT_CONFIG_FIELDS:
            if hasattr(text_config, name):
                setattr(model.config, name, getattr(text_config, name))


def _select_indexed_sft_examples(train, max_examples, seed):
    if max_examples > 0:
        train = train[:max_examples]
    indexed_train = list(enumerate(train))
    rng = random.Random(seed)
    rng.shuffle(indexed_train)
    return indexed_train


def select_sft_examples(train, max_examples, seed):
    """Pick the SFT sample: the first ``max_examples`` rows of the dataset (file order), shuffled.

    The slice happens BEFORE the shuffle so ``max_examples`` is a deterministic prefix fence,
    not a random subsample. A train.jsonl that carries fully-labeled SFT rows first and
    prompt-only (empty-output) GRPO rows after can cap SFT to the labeled head — an empty
    completion can never be shuffled into the SFT sample and teach the model to emit nothing.
    """
    return [example for _, example in _select_indexed_sft_examples(train, max_examples, seed)]


def _safe_realized_sft_max_length(rows: list[dict], configured_max: int) -> int:
    """return the longest realized row while enforcing the configured token cap."""
    lengths = [len(row["input_ids"]) for row in rows]
    if not lengths:
        raise ValueError("sft sizing requires at least one tokenized row")
    realized_max = max(lengths)
    if realized_max > configured_max:
        raise ValueError(
            f"realized sft row length {realized_max} exceeds configured max {configured_max}"
        )
    return realized_max


def _sft_runtime_max_length(realized_max: int, *, pad_to_multiple_of: int = 1) -> int:
    """include collator padding in the sequence width used for memory decisions."""
    multiple = max(1, int(pad_to_multiple_of))
    return math.ceil(realized_max / multiple) * multiple


def _configure_unpacked_length_sampling(cfg_kwargs: dict, dataset, *, packed: bool):
    """attach exact row lengths and enable length grouping only for unpacked training."""
    if packed:
        return dataset
    lengths = [len(ids) for ids in dataset["input_ids"]]
    if "length" in dataset.column_names:
        dataset = dataset.remove_columns("length")
    dataset = dataset.add_column("length", lengths)
    cfg_kwargs["train_sampling_strategy"] = "group_by_length"
    cfg_kwargs["length_column_name"] = "length"
    return dataset


class _SFTTokenCountingCollator:
    """preserve real-token counts without exposing metadata to the model."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features: list[dict]) -> dict:
        import torch

        batch = self.collator(features)
        batch["_flash_seq_lengths"] = torch.tensor(
            [
                sum(feature["seq_lengths"])
                if "seq_lengths" in feature
                else len(feature["input_ids"])
                for feature in features
            ],
            dtype=torch.long,
        )
        return batch


def _sft_local_token_count(inputs: dict):
    """count real input tokens without reducing a quadratic attention mask."""
    import torch

    seq_lengths = inputs.get("_flash_seq_lengths")
    if seq_lengths is not None:
        return seq_lengths.sum()
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None and attention_mask.ndim <= 2:
        return attention_mask.sum()
    position_ids = inputs.get("position_ids")
    if position_ids is not None:
        return torch.tensor(position_ids.numel(), device=position_ids.device)
    if attention_mask is not None:
        raise ValueError("packed 4d attention masks require real sequence lengths")
    raise ValueError("expected attention_mask, position_ids, or packed sequence lengths")


def _sft_step_will_log(state, args) -> bool:
    """true throughout every microbatch of an optimizer step that will log."""
    next_step = int(state.global_step) + 1
    if bool(getattr(args, "logging_first_step", False)) and next_step == 1:
        return True
    logging_steps = int(getattr(args, "logging_steps", 0) or 0)
    return logging_steps > 0 and next_step % logging_steps == 0


def _sft_quality_metrics_due(state, args, accelerator) -> bool:
    """return true on the final microbatch of an optimizer step that will log."""
    return bool(accelerator.sync_gradients) and _sft_step_will_log(state, args)


def sft_completed_train_tokens(
    tokens_per_epoch: int,
    epochs: int,
    derived_steps: int,
    completed_steps: int,
) -> int:
    """Estimate tokens processed from completed updates while preserving epoch accounting at parity."""
    epoch_tokens = max(0, int(tokens_per_epoch)) * max(0, int(epochs))
    completed = max(0, int(completed_steps))
    derived = max(1, int(derived_steps))
    if completed == derived:
        return epoch_tokens
    if completed == 0 or epoch_tokens == 0:
        return 0
    return max(1, round(epoch_tokens * completed / derived))


def sft_under_ran(final_step: int, update_horizon: int, max_steps: int) -> bool:
    """True when a max_steps-authoritative run completed fewer updates than requested.

    with max_steps authoritative, trl's max_steps override lands a fresh run exactly on the horizon,
    and a resume from a checkpoint past a lowered horizon does zero new steps yet holds a fully-trained
    adapter (final_step >= horizon). fail loudly only on a genuine under-run, mirroring grpo
    (steps_run < steps) and opd (opt_steps < steps).
    """
    return int(max_steps) > 0 and int(final_step) < int(update_horizon)


def _reject_image_completion(completion) -> None:
    from flash.multimodal import record_has_images

    if record_has_images({}, completion):
        raise ValueError("image-bearing SFT completions are not supported")


def run_sft():
    from datasets import Dataset
    from transformers import AutoProcessor
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    seed_training_rngs(_w.SEED)
    env = _w.require_active_env()
    t_start = time.time()
    _w.heartbeat("sft_start", gpu=gpu_diagnostics())
    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        gpu_type=_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else "",
    )
    setup_perf_backends()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    model_revision = getattr(_w.JOB_SPEC, "model_revision", "") if _w.JOB_SPEC else ""
    download_seconds = (
        _w.prefetch_model(model_id, revision=model_revision)
        if model_revision
        else _w.prefetch_model(model_id)
    )

    _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None

    def _train_opt(name, default):
        val = getattr(_t, name, None) if _t else None
        return val if val is not None else default

    sft_max_len = _train_opt(
        "max_context_tokens",
        RECIPE.sft.max_seq_len_thinking if _w.THINKING else RECIPE.sft.max_seq_len,
    )
    # tokenizer, dataset download, parent normalization, and parallel arrow preparation can run for
    # minutes on a large dataset, so keep the worker heartbeat fresh across the full boundary.
    with liveness_heartbeat("sft_data_loading"):
        from flash.multimodal import (
            normalize_prompt_images,
            record_has_images,
            text_only_prompt_messages,
            validate_multimodal_training,
        )

        indexed_train = _select_indexed_sft_examples(
            env.dataset(), int(_train_opt("max_examples", 0) or 0), _w.SEED
        )
        prefix_indices = [index for index, _ in indexed_train]
        train = [example for _, example in indexed_train]
        prompt_rows = [(ex, env.prompt_messages(ex), env.sft_completion(ex)) for ex in train]
        multimodal = any(
            record_has_images(ex, prompt_messages)
            for ex, prompt_messages, _completion in prompt_rows
        )
        processor = None
        if multimodal:
            validate_multimodal_training(model_id, "sft")
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                **_w.model_revision_kwargs(model_revision),
            )
            tok = processor.tokenizer
        else:
            tok = _w.load_tokenizer(model_id, revision=model_revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        package_root = getattr(env, "package_root", None)
        vl_rows = []
        for _ex, _prompt_messages, completion in prompt_rows:
            _reject_image_completion(completion)
        if multimodal:
            texts = []
            multiturn_targets = 0
            for ex, prompt_messages, completion in prompt_rows:
                if len(completion) > 1:
                    multiturn_targets += 1
                normalized = normalize_prompt_images(ex, prompt_messages, package_root)
                prompt_messages = normalized.messages
                completion = text_only_prompt_messages(completion)
                vl_rows.append(
                    {
                        "prompt": prompt_messages,
                        "completion": completion,
                        "images": normalized.descriptors,
                        "chat_template_kwargs": {"enable_thinking": _w.THINKING},
                    }
                )
                msgs = [*prompt_messages, *completion]
                texts.append(
                    {
                        "text": tok.apply_chat_template(
                            msgs,
                            tokenize=False,
                            add_generation_prompt=False,
                            enable_thinking=_w.THINKING,
                        ),
                        "prompt_text": tok.apply_chat_template(
                            prompt_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=_w.THINKING,
                        ),
                    }
                )
        else:
            texts, _pretok, _dropped, multiturn_targets, cache_hit = _prepare_sft_examples(
                env,
                train,
                tok,
                env_resolved_sha=(_w.JOB_SPEC.environment.resolved_sha if _w.JOB_SPEC else ""),
                seed=_w.SEED,
                model_revision=model_revision,
                thinking=_w.THINKING,
                max_length=sft_max_len,
                prefix_indices=prefix_indices,
                prompt_completion_rows=[
                    (prompt_messages, completion)
                    for _ex, prompt_messages, completion in prompt_rows
                ],
            )
            _masked_tok = sum(m.count(0) for m in (row["completion_mask"] for row in _pretok))
            _total_tok = sum(len(row["input_ids"]) for row in _pretok)
            print(f"[sft] tokenized data cache: {'hit' if cache_hit else 'miss'}")
    if multiturn_targets:
        print(
            f"[sft] multi-turn SFT: {multiturn_targets}/{len(train)} rows train on a full target transcript"
        )
    elif getattr(env, "multi_turn", False):
        print(
            "[sft][warn] this is a multi-turn Freesolo environment but no row ships a multi-turn "
            "target completion; SFT collapses to a single assistant turn per row (tool/env turns "
            'ignored). Provide target transcripts (output={"messages": [...]}) for proper multi-turn SFT.'
        )
    if _w.THINKING and not any("<think>" in t["text"] for t in texts[:256]):
        print(
            "WARN: thinking mode is ON but no sampled SFT target contains a <think> "
            "trace — training on non-reasoning targets teaches the model to SKIP "
            "thinking. Use a dataset with reasoning traces, or set thinking = false."
        )
    setup_seconds = time.time() - t_start
    _w.heartbeat("sft_model_load", setup_seconds=setup_seconds, gpu=gpu_diagnostics())

    epochs = int(
        _w.JOB_SPEC.train.epochs
        if _w.JOB_SPEC and _w.JOB_SPEC.train.epochs is not None
        else RECIPE.sft.num_epochs
    )
    from flash.catalog import MODELS, resolve_vocab_size
    from flash.engine.vram import sft_chunked_nll_enabled, sft_grad_accum

    sft_lr = _train_opt("learning_rate", RECIPE.sft.learning_rate)
    if multimodal:
        with liveness_heartbeat("sft_pretokenizing"):
            from flash.multimodal import ArrowSafeVisionCollator, filter_vlm_sft_rows

            _vision_collator = ArrowSafeVisionCollator(processor, sft_max_len, package_root)
            vl_rows, _dropped, _masked_tok, _total_tok = filter_vlm_sft_rows(
                vl_rows,
                _vision_collator,
                set(getattr(tok, "all_special_ids", None) or []),
            )
            _pretok = []
    # everything from here to the trainer build is silent config work: arch probes that hit HF
    # (AutoConfig, resolve_vocab_size), the bin-packing preview, and the vram sizing math. the
    # sft_model_load ping above is ONE-SHOT, so unwrapped this whole span emits nothing -- a probe
    # wedged on a socket leaves the run frozen on a stage that has not even loaded the model yet:
    # status stops refreshing, the GPU reads idle, and the worker's stall stack-dump never arms
    # because no liveness thread is running. no progress= : there is no monotonic counter here, and
    # bare pings deliberately do NOT feed the provider's stall clock, so a real wedge still trips it.
    with liveness_heartbeat("sft_configuring"):
        if _dropped:
            print(
                f"[sft] dropped {_dropped} rows with no real completion target "
                "(sft_max_len truncated away the whole completion, or an empty/content-free completion)"
            )
        if not (vl_rows if multimodal else _pretok):
            raise ValueError(
                "every SFT example has an empty completion after sft_max_len truncation (nothing to "
                "train on); increase sft_max_len or shorten the prompts"
            )
        ds = Dataset.from_list(vl_rows) if multimodal else Dataset.from_list(_pretok)
        if _total_tok:
            print(
                f"[sft] completion-only loss: masking {_masked_tok}/{_total_tok} "
                f"({_masked_tok / _total_tok:.0%}) prompt tokens; training on the completion only"
            )
        effective_batch = _train_opt("batch_size", RECIPE.sft.effective_batch)
        # validated qwen models use trl 1.6 chunked_nll: ignored prompt positions are removed before
        # the lm head and valid tokens are projected in bounded chunks, so no dense [batch, seq, vocab]
        # tensor is materialized. unsupported model structures keep plain nll (which materializes full
        # fp32 logits). resolve vocab once so worker and cost quote use the same revision.
        _sft_vocab = resolve_vocab_size(model_id, model_revision)
        # chunked_nll's prep (_prepare_chunked_nll_model) and parity tests are text-only and require
        # flash's text tokenizer; a multimodal run's SFTTrainer uses the vision processor, so gate
        # chunked_nll off for image runs (they use standard nll) to avoid the prep/processing mismatch.
        # per_device/grad_accum are sized below in the unified packing-aware block using fused=_sft_fused.
        _sft_fused = sft_chunked_nll_enabled(model_id) and not multimodal
        sft_save_default = _train_opt("save_every", 50)
        out_dir = f"/tmp/sft_seed{_w.SEED}"
        resume_ckpt = _w.hf_resume_checkpoint()

        max_steps = int(_t.max_steps or 0) if _t else 0
        save_at_steps = tuple(getattr(_t, "save_at_steps", ()) or ())
        cfg_kwargs = {
            "output_dir": out_dir,
            "num_train_epochs": epochs,
            "learning_rate": sft_lr,
            "warmup_ratio": RECIPE.sft.warmup_frac,
            "logging_steps": 10,
            "save_steps": sft_save_default,
            "save_total_limit": 1,
            # save_only_model=False: saves optimizer/scheduler state so a resumed worker truly continues
            # instead of re-initializing Adam moments. The deployable snapshot strips trainer state separately.
            "save_only_model": False,
            "max_length": sft_max_len,
            "bf16": True,
            "report_to": _w.wandb_report_to(),
            "run_name": _w.wandb_run_name(),
            "dataloader_num_workers": 4,
            "dataloader_pin_memory": True,
            "dataloader_persistent_workers": True,
            "accelerator_config": {"non_blocking": True},
            "seed": backend_seed(_w.SEED),
            # MoE / GatedDeltaNet hybrids re-dispatch tokens (MoE router) or lay out custom-kernel
            # saved tensors differently on recompute, so non-reentrant checkpointing's metadata-equality
            # assert fires on the FIRST backward (Qwen3.6-35B-A3B). Mirror the GRPO path (rl.py, #429/#432)
            # and pick REENTRANT recompute for those; dense models keep the faster non-reentrant path.
            "gradient_checkpointing_kwargs": {"use_reentrant": grpo_use_reentrant(model_id)},
            "completion_only_loss": True,
            "loss_type": "chunked_nll" if _sft_fused else "nll",
            # remove_unused_columns=False: HF Trainer would otherwise drop completion_mask before
            # collation, silently reverting all paths to full-transcript loss.
            "remove_unused_columns": False,
            "optim": fused_optim_name(),
        }
        _sft_config_fields = set(getattr(TRLSFTConfig, "__dataclass_fields__", {}))
        if "use_liger_kernel" in _sft_config_fields:
            cfg_kwargs["use_liger_kernel"] = False
        if max_steps > 0:
            cfg_kwargs["max_steps"] = max_steps
        configure_trainer_save_schedule(cfg_kwargs, save_at_steps)
        # TRL 'bfd' packing: boundary-correct only under FA2/FA3 varlen (SDPA cross-contaminates).
        # GDN hybrids can't use bfd (no seq_idx to reset causal conv); they pack via the varlen collator.
        _pure_attn = model_is_pure_attention(model_id, revision=model_revision)
        _gdn = model_is_gdn_hybrid(model_id, revision=model_revision)
        _bfd_block_count: int | None = None
        _fa_ok = _flash_attn_available()
        if multimodal:
            _pure_attn = False
            _gdn = False
            _fa_ok = False
            print("[sft] packing disabled for vision-language batches")
        if _fa_ok and _pure_attn:
            cfg_kwargs["packing"] = True
            print("[sft] example packing enabled (FA2 varlen)")
        elif _fa_ok and _gdn:
            print(
                "[sft] TRL bfd packing NOT used for the GatedDeltaNet hybrid (bfd can't reset the conv); "
                "the cu_seqlens/seq_idx varlen collator handles its packing when both kernels are present."
            )
        else:
            _bfd_why = (
                "flash_attn not importable" if not _fa_ok else "arch not bfd-safe under FA2 varlen"
            )
            print(
                f"[sft] TRL bfd (FA2) packing not used ({_bfd_why}); the SDPA-mask path decides packing below."
            )
        if _memory_mode(model_id, sft_max_len, revision=model_revision):
            print("[sft] chalk standalone fused kernels scheduled after trainer build")
        _attn = optimal_attn_impl(model_id, model_revision)
        # When bfd packing is on, ensure a varlen-capable flash impl; sdpa cross-contaminates packed examples.
        # _attn=="sdpa" (Blackwell): disable bfd — don't force FA2 (unverified SASS). SDPA-mask path below still packs.
        # _attn is None (Hopper without FA3): force FA2 if available, else drop packing.
        if cfg_kwargs.get("packing"):
            if _attn in ("flash_attention_2", "flash_attention_3"):
                print(f"[sft] attn_implementation={_attn} (packing boundary-correct varlen)")
            elif _attn == "sdpa":
                cfg_kwargs["packing"] = False
                print(
                    "[sft] packing disabled: selected attn_implementation=sdpa (no varlen flash backend)"
                )
            elif _fa_ok and _arch_supports_attn_impl(model_id, "flash_attention_2", model_revision):
                _attn = "flash_attention_2"
                print("[sft] attn_implementation=flash_attention_2 (packing boundary-correct varlen)")
            else:
                cfg_kwargs["packing"] = False
                print(
                    "[sft] packing disabled: no varlen flash backend (FA2/FA3) available -> plain SDPA"
                )
        _bfd_rows: list[dict] | None = None
        _packed_examples_per_block: float | None = None
        if cfg_kwargs.get("packing"):
            from trl.data_utils import pack_dataset

            # preview with trl's exact bfd implementation so the realized maximum is a safe match for
            # the blocks the trainer will build, rather than an estimate from a different bin packer.
            _bfd_preview = pack_dataset(
                ds.select_columns(["input_ids", "completion_mask"]),
                sft_max_len,
                strategy="bfd",
                map_kwargs={"load_from_cache_file": False},
            )
            _bfd_rows = [
                {"input_ids": ids, "seq_lengths": seq_lengths}
                for ids, seq_lengths in zip(
                    _bfd_preview["input_ids"], _bfd_preview["seq_lengths"], strict=True
                )
            ]
            _bfd_block_count = max(1, len(_bfd_rows))
            _packed_examples_per_block = len(_pretok) / _bfd_block_count

        # 4D block-diagonal SDPA mask packing: for pure-attn models on plain SDPA (e.g. sm120 RTX 5090).
        # Flash varlen would silently IGNORE the 4D mask — so we downgrade to sdpa when using this path.
        # GDN hybrids need the next branch (mask alone can't reset their causal conv state).
        _collator = _vision_collator if multimodal else None
        _packed_rows: list[dict] | None = None
        _packing_kind = "bfd" if cfg_kwargs.get("packing") else "unpacked"
        # Cap at 16384: dense [B,1,T,T] mask is O(T^2) memory; above this packing gains little anyway.
        _PACK_MASK_MAX_LEN = 16384
        _mask_pack_ok = sft_max_len <= _PACK_MASK_MAX_LEN
        # both mask-packing paths below REQUIRE sdpa (a flash kernel ignores the 4D mask), so an arch
        # that ships no sdpa kernel cannot take them at all -- forcing it would only move the failure
        # to from_pretrained. skip the packing instead and train unpacked on the arch's own backend.
        _sdpa_ok = _arch_supports_attn_impl(model_id, "sdpa", model_revision)
        _sdpa_pack = bool(
            not cfg_kwargs.get("packing") and _pure_attn and _mask_pack_ok and _sdpa_ok
        )
        if _sdpa_pack:
            if _attn in ("flash_attention_2", "flash_attention_3"):
                print(
                    f"[sft] packing under SDPA: downgrading {_attn} -> sdpa (a flash kernel ignores the 4D mask)"
                )
            _attn = "sdpa"
            cfg_kwargs["packing"] = False  # we own the packing; TRL must not also pack
            _dk = dict(cfg_kwargs.get("dataset_kwargs") or {})
            _dk["skip_prepare_dataset"] = True
            cfg_kwargs["dataset_kwargs"] = _dk
            _ids = [r["input_ids"] for r in _pretok]
            _cmask = [r["completion_mask"] for r in _pretok]
            _packed_rows = pack_token_ids(_ids, sft_max_len, completion_masks=_cmask)
            ds = Dataset.from_list(_packed_rows)
            _collator = _SFTTokenCountingCollator(BlockDiagonalCollator(pad_token_id=tok.pad_token_id))
            _packed_examples_per_block = len(_ids) / max(1, len(_packed_rows))
            _packing_kind = "sdpa"
        elif (
            not cfg_kwargs.get("packing")
            and _gdn
            and gdn_packing_available(model_id, revision=model_revision)
            and _mask_pack_ok
            and _sdpa_ok
        ):
            # GDN hybrid: 4D mask for full-attn layers + cu_seqlens/seq_idx to reset DeltaNet recurrence.
            # Flash varlen would ignore the 4D mask — downgrade to sdpa for the full-attn layers.
            if _attn in ("flash_attention_2", "flash_attention_3"):
                print(
                    f"[sft] GDN packing under SDPA: downgrading {_attn} -> sdpa for the full-attn layers"
                )
            _attn = "sdpa"
            cfg_kwargs["packing"] = False
            _dk = dict(cfg_kwargs.get("dataset_kwargs") or {})
            _dk["skip_prepare_dataset"] = True
            cfg_kwargs["dataset_kwargs"] = _dk
            _ids = [r["input_ids"] for r in _pretok]
            _cmask = [r["completion_mask"] for r in _pretok]
            _packed_rows = pack_token_ids(_ids, sft_max_len, completion_masks=_cmask)
            ds = Dataset.from_list(_packed_rows)
            _collator = _SFTTokenCountingCollator(
                BlockDiagonalCollator(pad_token_id=tok.pad_token_id, emit_varlen=True)
            )
            _packed_examples_per_block = len(_ids) / max(1, len(_packed_rows))
            _packing_kind = "gdn"
        elif not cfg_kwargs.get("packing") and (_pure_attn or _gdn) and not _mask_pack_ok:
            print(
                f"[sft] packing stays OFF: max_length {sft_max_len} > {_PACK_MASK_MAX_LEN} — the dense "
                "O(T^2) block-diagonal mask gets too large at long context (unpacked is more memory-"
                "efficient there, and long rows already fill a block)."
            )
        elif not cfg_kwargs.get("packing") and (_pure_attn or _gdn) and not _sdpa_ok:
            print(
                f"[sft] packing stays OFF: {model_id} declares no sdpa kernel, and the block-diagonal "
                "mask paths need one (a flash kernel ignores the 4D mask)."
            )
        elif not multimodal and not cfg_kwargs.get("packing") and not _pure_attn:
            _why = (
                "hybrid GatedDeltaNet but the fla/causal_conv1d varlen kernels aren't both importable"
                if _gdn
                else "non-full-attention arch (e.g. sliding-window) a block-diagonal mask can't pack"
            )
            print(
                f"[sft] packing stays OFF: {_why}. (Pure full-attention models pack via the SDPA mask.)"
            )

        _sizing_rows = _bfd_rows if _packing_kind == "bfd" else _packed_rows or _pretok
        if multimodal:
            # vision rows are tokenized by the processor-backed collator, so retain configured-cap sizing.
            _realized_max_len = sft_max_len
            _runtime_max_len = sft_max_len
        else:
            _realized_max_len = _safe_realized_sft_max_length(_sizing_rows, sft_max_len)
            _runtime_max_len = _sft_runtime_max_length(
                _realized_max_len,
                pad_to_multiple_of=8 if _packing_kind == "sdpa" else 1,
            )
        per_device_bs, grad_accum = sft_grad_accum(
            effective_batch,
            seq_len=_runtime_max_len,
            vocab=_sft_vocab,
            fused=_sft_fused,
        )
        if _packing_kind == "sdpa":
            # cap the real dense mask allocation at 512 mb using its padded runtime width.
            per_device_bs = max(
                1,
                min(per_device_bs, (512 * 1024 * 1024) // (_runtime_max_len * _runtime_max_len)),
            )
        elif _packing_kind == "gdn":
            # the varlen recurrence metadata spans one block and requires a single block per microbatch.
            per_device_bs = 1
        if _packed_examples_per_block is not None:
            grad_accum = max(
                1,
                math.ceil(effective_batch / max(1.0, per_device_bs * _packed_examples_per_block)),
            )
        else:
            grad_accum = max(1, math.ceil(effective_batch / per_device_bs))
        cfg_kwargs["per_device_train_batch_size"] = per_device_bs
        cfg_kwargs["gradient_accumulation_steps"] = grad_accum

        _is_packed = _packing_kind != "unpacked"
        if not multimodal:
            ds = _configure_unpacked_length_sampling(cfg_kwargs, ds, packed=_is_packed)
        if multimodal:
            print(
                f"[sft] vision-language configured-cap sizing: seq={sft_max_len} "
                f"pd={per_device_bs} ga={grad_accum}"
            )
        elif _is_packed:
            assert _packed_examples_per_block is not None
            print(
                f"[sft] {_packing_kind} packing: {len(_pretok)} examples -> {len(_sizing_rows)} blocks "
                f"(~{_packed_examples_per_block:.1f} ex/block, "
                f"{packing_efficiency(_sizing_rows, sft_max_len):.0%} dense); "
                f"realized_max={_realized_max_len}/{sft_max_len} pd={per_device_bs} ga={grad_accum} "
                f"(effective batch kept ~{effective_batch} ex)"
            )
        else:
            print(
                f"[sft] unpacked length grouping enabled: realized_max={_realized_max_len}/{sft_max_len} "
                f"pd={per_device_bs} ga={grad_accum}"
            )
        if not _sft_fused and per_device_bs < min(effective_batch, 4):
            print(
                f"[sft] large-vocab logits cap: per_device={per_device_bs} grad_accum={grad_accum} "
                f"(realized_seq={_realized_max_len}, configured_seq={sft_max_len}, vocab={_sft_vocab}; "
                f"realized batch {per_device_bs * grad_accum} >= requested {effective_batch})"
            )

        _gc_card_gb, _gc_cap = 0.0, None
        try:
            import torch as _torch_gc

            if _torch_gc.cuda.is_available():
                _gc_card_gb = _torch_gc.cuda.get_device_properties(0).total_memory / 1e9
                _gc_cap = _torch_gc.cuda.get_device_capability(0)
        except Exception:
            _gc_card_gb, _gc_cap = 0.0, None
        _gc_hidden, _gc_layers = _model_arch_dims(model_id, revision=model_revision)
        _gc_active_b = float(getattr(MODELS.get(model_id), "active_params_b", 0.0) or 0.0) or None
        _gc_lora_rank = int(_t.lora_rank if _t and _t.lora_rank else RECIPE.lora.rank)
        _grad_ckpt = grad_checkpointing_on(
            model_id,
            _runtime_max_len,
            allow_disable=True,
            card_vram_gb=_gc_card_gb,
            capability=_gc_cap,
            active_params_b=_gc_active_b,
            hidden=_gc_hidden,
            num_layers=_gc_layers,
            fused_ce=_sft_fused,
            per_device_bs=per_device_bs,
            lora_rank=_gc_lora_rank,
            revision=model_revision,
        )
        cfg_kwargs["gradient_checkpointing"] = _grad_ckpt

        # Explicit bf16 + device_map=None: transformers-5 string loading otherwise falls back to fp32
        # (2x VRAM) or accelerate-offloads to meta ("expected device meta but got cuda:0" in backward).
        model_init_kwargs = {
            "dtype": "bfloat16",
            "device_map": None,
            **_w.model_revision_kwargs(model_revision),
        }
        if _attn:
            model_init_kwargs["attn_implementation"] = _attn
        cfg_kwargs["model_init_kwargs"] = model_init_kwargs
        examples_per_update = int(cfg_kwargs["per_device_train_batch_size"]) * int(
            cfg_kwargs["gradient_accumulation_steps"]
        )
        derived_steps = sft_update_steps(
            epochs=epochs,
            example_count=len(ds),
            examples_per_update=examples_per_update,
            packed_block_count=_bfd_block_count if cfg_kwargs.get("packing") else None,
        )
        update_horizon = resolve_update_horizon(derived_steps, max_steps)
        validate_save_steps(save_at_steps, update_horizon)
        cfg = TRLSFTConfig(**cfg_kwargs)

    # LoRA+ (arXiv 2402.12354): B-matrix LR ratio=16, measured -52% train loss. Must override
    # create_optimizer because TRL builds the model inside __init__ (can't pre-build the optimizer).
    _lp_ratio = 16
    _SFT = SFTTrainer
    if _lp_ratio > 1:

        class _SFT(SFTTrainer):  # local lora+ and metric-shaping subclass
            _loraplus_applied = False  # true only once the lora+ grouping actually installs
            _flash_total_train_tokens = None
            # quality-metric accumulators spanning the microbatches of one logging step
            _flash_qm_step = -1
            _flash_qm_entropy_sum = None
            _flash_qm_total_tokens = None
            _flash_qm_correct_tokens = None
            _flash_qm_aux_sum = None
            _flash_qm_aux_count = 0

            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
            ):
                if not model.training:
                    inputs.pop("_flash_seq_lengths", None)
                    return super().compute_loss(
                        model,
                        inputs,
                        return_outputs=return_outputs,
                        num_items_in_batch=num_items_in_batch,
                    )

                import torch
                from trl.trainer.sft_trainer import PeftType, entropy_from_logits

                local_tokens = _sft_local_token_count(inputs).detach()
                inputs.pop("_flash_seq_lengths", None)
                inputs.pop("_prediction_loss_only", None)
                labels = inputs["labels"] if "shift_labels" not in inputs else None
                inputs["use_cache"] = False
                loss, outputs = super(SFTTrainer, self).compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )

                if self._flash_total_train_tokens is None:
                    self._flash_total_train_tokens = local_tokens.clone()
                else:
                    self._flash_total_train_tokens += local_tokens

                if _sft_step_will_log(self.state, self.args):
                    # chunked_nll does not materialize outputs.logits, so the logits-based quality
                    # metrics (entropy / token accuracy) are skipped in that case; loss, num_tokens
                    # and aux_loss are still logged.
                    logits_available = outputs.logits is not None
                    # reset the per-logging-step accumulators when a new logging step begins (keyed on
                    # global_step) so logged metrics reflect the whole optimizer step, not only its
                    # final microbatch.
                    if self._flash_qm_step != int(self.state.global_step):
                        self._flash_qm_step = int(self.state.global_step)
                        self._flash_qm_entropy_sum = None
                        self._flash_qm_total_tokens = None
                        self._flash_qm_correct_tokens = None
                        self._flash_qm_aux_sum = None
                        self._flash_qm_aux_count = 0
                    if logits_available:
                        with torch.no_grad():
                            if "shift_labels" in inputs:
                                shift_logits = outputs.logits
                                shift_labels = inputs["shift_labels"]
                            else:
                                shift_logits = outputs.logits[..., :-1, :]
                                shift_labels = labels[..., 1:]
                            if (
                                self.num_virtual_tokens > 0
                                and model.peft_config[model.active_adapter].peft_type
                                != PeftType.PREFIX_TUNING
                            ):
                                shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                            mask = shift_labels != -100
                            entropy_sum_mb = (entropy_from_logits(shift_logits) * mask).sum()
                            total_tokens_mb = mask.sum()
                            correct_tokens_mb = (
                                (shift_logits.argmax(dim=-1) == shift_labels) & mask
                            ).sum()
                        if self._flash_qm_entropy_sum is None:
                            self._flash_qm_entropy_sum = entropy_sum_mb
                            self._flash_qm_total_tokens = total_tokens_mb
                            self._flash_qm_correct_tokens = correct_tokens_mb
                        else:
                            self._flash_qm_entropy_sum = self._flash_qm_entropy_sum + entropy_sum_mb
                            self._flash_qm_total_tokens = self._flash_qm_total_tokens + total_tokens_mb
                            self._flash_qm_correct_tokens = (
                                self._flash_qm_correct_tokens + correct_tokens_mb
                            )
                    if self.aux_loss_enabled:
                        aux_mb = outputs.aux_loss.detach()
                        self._flash_qm_aux_sum = (
                            aux_mb
                            if self._flash_qm_aux_sum is None
                            else self._flash_qm_aux_sum + aux_mb
                        )
                        self._flash_qm_aux_count += 1
                    if self.accelerator.sync_gradients:
                        if self._flash_qm_entropy_sum is not None:
                            entropy_sum = self.accelerator.gather_for_metrics(
                                self._flash_qm_entropy_sum
                            ).sum()
                            total_tokens = self.accelerator.gather_for_metrics(
                                self._flash_qm_total_tokens
                            ).sum()
                            correct_tokens = self.accelerator.gather_for_metrics(
                                self._flash_qm_correct_tokens
                            ).sum()
                            entropy = (entropy_sum / total_tokens).item() if total_tokens > 0 else 0.0
                            accuracy = (
                                (correct_tokens / total_tokens).item() if total_tokens > 0 else 0.0
                            )
                            self._metrics["train"]["entropy"].append(entropy)
                            self._metrics["train"]["mean_token_accuracy"].append(accuracy)
                        total_train_tokens = (
                            self.accelerator.gather_for_metrics(self._flash_total_train_tokens)
                            .sum()
                            .item()
                        )
                        self._total_train_tokens = total_train_tokens
                        self._metrics["train"]["num_tokens"] = [total_train_tokens]
                        if self.aux_loss_enabled and self._flash_qm_aux_count > 0:
                            aux_mean_local = self._flash_qm_aux_sum / self._flash_qm_aux_count
                            aux_loss = (
                                self.accelerator.gather_for_metrics(aux_mean_local).mean().item()
                            )
                            self._metrics["train"]["aux_loss"].append(aux_loss)

                return (loss, outputs) if return_outputs else loss

            def create_optimizer(self):
                if self.optimizer is None:
                    try:
                        from peft.optimizers import create_loraplus_optimizer

                        # Use .value not str(): OptimizerNames enum str() includes the class name.
                        opt_cls, extra = loraplus_optimizer_cls(
                            getattr(self.args.optim, "value", self.args.optim)
                        )
                        # Explicitly forward betas/eps/weight_decay — PEFT doesn't read TrainingArguments.
                        fwd = dict(extra)
                        _betas = (
                            getattr(self.args, "adam_beta1", None),
                            getattr(self.args, "adam_beta2", None),
                        )
                        if None not in _betas:
                            fwd.setdefault("betas", _betas)
                        _eps = getattr(self.args, "adam_epsilon", None)
                        if _eps is not None:
                            fwd.setdefault("eps", _eps)
                        lp_extra: dict[str, object] = {}
                        _wd = getattr(self.args, "weight_decay", None)
                        if _wd is not None:
                            lp_extra["loraplus_weight_decay"] = _wd
                        # lr kwarg name shifted across PEFT versions; fall back to top-level lr=.
                        try:
                            self.optimizer = create_loraplus_optimizer(
                                model=self.model,
                                optimizer_cls=opt_cls,
                                optimizer_kwargs={"lr": self.args.learning_rate, **fwd},
                                loraplus_lr_ratio=_lp_ratio,
                                **lp_extra,
                            )
                        except TypeError:
                            self.optimizer = create_loraplus_optimizer(
                                model=self.model,
                                optimizer_cls=opt_cls,
                                lr=self.args.learning_rate,
                                loraplus_lr_ratio=_lp_ratio,
                                **fwd,
                                **lp_extra,
                            )
                        self._loraplus_applied = True
                        print(
                            f"[lora+] optimizer enabled (B-matrix LR ratio={_lp_ratio}, "
                            f"cls={opt_cls.__name__})"
                        )
                        return self.optimizer
                    except Exception as e:  # never block training on LoRA+ wiring failure
                        print("[lora+] setup failed, falling back to default optimizer:", e)
                return super().create_optimizer()

    trainer_callbacks = [
        _w.make_sft_heartbeat_callback(),
        _w.make_checkpoint_upload_callback(save_at_steps),
    ]
    if multimodal:
        trainer_callbacks.append(make_multimodal_input_require_grads_callback())

    # SFTTrainer.__init__ can block 10-15 min (FA2 JIT). liveness_heartbeat keeps the control plane
    # from recycling the worker. include_torch=False: side-thread torch.cuda telemetry serializes on
    # the CUDA/allocator lock held by the init thread and can freeze the heartbeat itself.
    with liveness_heartbeat("sft_initializing"):
        seed_training_rngs(_w.SEED)
        sft_model = _w.prepare_fresh_lora_base(
            model_id,
            model_id,
            model_init_kwargs,
            force=_sft_fused,
            phase="sft",
            model_revision=model_revision,
        )
        if not isinstance(sft_model, str):
            cfg.model_init_kwargs = None
        _lora_config = _w.make_lora(model_id)
        if _sft_fused:
            _prepare_chunked_nll_model(sft_model, tok, _lora_config)
        trainer = _SFT(
            model=sft_model,
            args=cfg,
            train_dataset=ds,
            peft_config=_lora_config,
            processing_class=processor if multimodal else tok,
            data_collator=_collator,
            callbacks=trainer_callbacks,
        )
        if not multimodal and _collator is None:
            # trl rejects custom collators for padding-free bfd, so wrap its generated collator here.
            trainer.data_collator = _SFTTokenCountingCollator(trainer.data_collator)
        # chalk flce (fused_ce) stays OFF here regardless of _sft_fused: trl's SFTTrainer.compute_loss reads
        # outputs.logits (it only skips them under use_liger_kernel=True, which would make trl apply Liger
        # and clash with chalk), so chalk flce returning logits=None would crash large-vocab Qwen3.5 SFT with
        # "'NoneType' object is not subscriptable" (#421). When _sft_fused is True, trl's OWN chunked_nll
        # (loss_type="chunked_nll") owns the output-head loss through its supported logits=None path instead.
        # Either way the micro-batch / grad-accum / grad-checkpointing (and the allocator, vram.py) were sized
        # UP FRONT for the chosen path (fused=_sft_fused) — no post-init grad-accum fixup that would mutate it
        # after the trainer's Accelerator was built (codex[bot]). The custom GRPO/opd loops read the fused
        # loss directly, so they keep flce on.
        # inside the liveness wrap: chalk's kernel install can JIT-compile, silent for minutes.
        _chalk_report = install_chalk_kernels(getattr(trainer, "model", None), fused_ce=False)
    _chalk_active = active_kernels(_chalk_report)

    _reset_peak_gpu()
    _gpu_sampler = _GpuPeakSampler().start()
    t_train = time.time()
    # progress + progress_step: step advances emit REAL heartbeats (and stamp the step), so the
    # daemon can never starve the provider's stall clock by winning the throttled upload slot with
    # a bare liveness ping while training is healthy.
    with (
        liveness_heartbeat(
            "sft_step",
            progress=lambda: int(getattr(trainer.state, "global_step", 0) or 0),
            progress_step=True,
        ),
        _sdpa_cudnn_ctx(_attn),
    ):
        trainer.train(resume_from_checkpoint=resume_ckpt)
    train_wall = time.time() - t_train
    sft_peak_gpu_gb = _peak_gpu_gb()
    sft_device_peak_gpu_gb = _gpu_sampler.stop_gb()

    _final_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if sft_under_ran(_final_step, update_horizon, max_steps):
        raise RuntimeError(
            f"sft completed {_final_step}/{update_horizon} requested optimizer updates"
        )
    # adapter save + required upload can take minutes on a slow HF; keep the heartbeat fresh.
    # keepalive=True: _final_step is CONSTANT here (training is done), so without it every finalize
    # ping is a bare liveness that does NOT advance the provider's stall clock — a finalize outlasting
    # STALL_AFTER_S (1500s) would be killed at the finish line. keepalive forces a REAL heartbeat/tick.
    # progress_step stamps the final step on every finalize heartbeat so a cancel landing in this
    # window still bills the actual steps trained (actual_steps_run reads last_heartbeat.step).
    with liveness_heartbeat(
        "sft_finalizing", progress=lambda: _final_step, progress_step=True, keepalive=True
    ):
        adapter_dir = f"{out_dir}/adapter"
        _w.stamp_adapter_provenance(trainer.model, model_id, model_revision)
        trainer.model.save_pretrained(adapter_dir)
        (processor if multimodal else tok).save_pretrained(adapter_dir)
        _w.write_base_model_provenance(adapter_dir, model_id, model_revision)
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        # preserve the final checkpoint only when exact save steps are not configured.
        if final_save_due(_final_step, save_at_steps):
            _w.publish_deployable_checkpoint(adapter_dir, _final_step)
    _w.heartbeat("sft_trained", train_wall=train_wall, step=_final_step, gpu=gpu_diagnostics())

    train_tokens = sft_completed_train_tokens(
        _total_tok,
        epochs,
        derived_steps,
        _final_step,
    )
    _w.write_train_meta(
        phase="sft",
        adapter_dir=adapter_dir,
        model_id=model_id,
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=train_tokens,
        generated_tokens=0,
        notes={
            "epochs": epochs,
            "resumed": bool(resume_ckpt),
            "download_seconds": download_seconds,
            "hf_transfer": os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
            "thinking": _w.THINKING,
            "gradient_checkpointing": _grad_ckpt,
            "configured_max_length": sft_max_len,
            "realized_max_length": _realized_max_len,
            "runtime_max_length": _runtime_max_len,
            "per_device_train_batch_size": per_device_bs,
            "gradient_accumulation_steps": grad_accum,
            "packing": _packing_kind,
            "loss_curve": _metric_curve(trainer, "loss"),
            "peak_gpu_gb": sft_peak_gpu_gb,
            # device_peak_gpu_gb includes bnb managed optimizer pages; peak_gpu_gb does not.
            "device_peak_gpu_gb": sft_device_peak_gpu_gb,
            # Unwrap AcceleratedOptimizer (transformers 5.x) to get the underlying class name.
            "loraplus_optim": (
                type(getattr(trainer.optimizer, "optimizer", trainer.optimizer)).__name__
                if getattr(trainer, "optimizer", None) is not None
                else loraplus_optimizer_cls(fused_optim_name())[0].__name__
            ),
            "loraplus_applied": getattr(trainer, "_loraplus_applied", False),
            "chalk_kernels": _chalk_active or None,
            **_w.wandb_run_info(),
        },
    )
    free_gpu(trainer)
