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
from pathlib import Path

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.packing import (
    completion_mask_from_ids,
    tokenize_for_packing,
)

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
    """delegate SFT execution to the out-of-process verl worker."""
    from flash.engine.worker.sft_verl import run_sft_verl

    return run_sft_verl(_w.JOB_SPEC)
