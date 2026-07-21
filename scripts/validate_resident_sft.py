"""Validate SFT base residency on one real GPU.

The fresh arm uses Flash's production ``prepare_fresh_lora_base`` decision and TRL's
production string-model loader. The resident arm uses ``ResidentBase`` with identical
model kwargs, tokenizer, dataset, seed placement, LoRA configuration, and TRL SFTTrainer
constructor inputs. No reseed occurs between model loading and trainer construction. The
harness does not cover production environment rendering, the exact production SFTConfig,
completion-only packing/collation, the LoRA+ trainer subclass and optimizer, performance
backend selection, heartbeat, checkpoint resume, artifact upload, or provider lifecycle.
Those external steps are intentionally outside this model-residency gate.

The parent process launches one fresh baseline process and one resident process. The
resident process trains the same fresh LoRA adapter three or more times on one frozen
base. Each child writes per-step losses, final adapter tensors, load/setup timings, and
peak CUDA allocation. The parent compares resident job two with the fresh baseline and
fails if losses or adapter tensors exceed the requested floating-point tolerances.

Example:
    uv run --extra gpu python scripts/validate_resident_sft.py \
        --model-id HuggingFaceTB/SmolLM2-135M --steps 20 --resident-jobs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _model_kwargs(revision: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dtype": "bfloat16",
        "device_map": None,
    }
    if revision:
        kwargs["revision"] = revision
    return kwargs


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    from transformers import set_seed

    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dataset():
    from datasets import Dataset

    rows = [
        "Question: What is two plus two? Answer: Four.",
        "Question: Name a primary color. Answer: Blue.",
        "Question: Complete the sequence 1, 2, 3. Answer: 4.",
        "Question: What planet do humans live on? Answer: Earth.",
    ]
    return Dataset.from_dict({"text": rows * 8})


def _full_base_fingerprint(model) -> str:
    """Hash every frozen base parameter and persistent buffer without sampling."""

    import torch

    parameters = dict(model.named_parameters())
    digest = hashlib.sha256()
    included = 0
    for name, tensor in sorted(model.state_dict().items()):
        lowered = name.lower()
        if any(
            marker in lowered
            for marker in ("lora_", "adapter_", "modules_to_save", "prompt_encoder")
        ):
            continue
        parameter = parameters.get(name)
        if parameter is not None and parameter.requires_grad:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        included += 1
    if not included:
        raise ValueError("cannot fingerprint a model with no frozen base state")
    return digest.hexdigest()


def _adapter_state(model) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in get_peft_model_state_dict(model).items()
    }


def _loss_curve(trainer) -> list[float]:
    return [
        float(record["loss"])
        for record in trainer.state.log_history
        if record.get("loss") is not None
    ]


def _train_one(holder, args, job_index: int, output_dir: Path) -> dict[str, Any]:
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    from flash.engine.worker.adapter import prepare_fresh_lora_base
    from flash.engine.worker.hf import load_tokenizer
    from flash.engine.worker.resident import ResidentJobState, base_fingerprint

    dataset = _dataset()
    tokenizer = (
        holder.get_tokenizer(args.model_id, revision=args.revision)
        if holder is not None
        else load_tokenizer(args.model_id, revision=args.revision)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_init_kwargs = _model_kwargs(args.revision)
    config = SFTConfig(
        output_dir=str(output_dir / f"trainer-{job_index:02d}"),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        max_length=args.max_length,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        model_init_kwargs=model_init_kwargs,
    )
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        init_lora_weights=True,
        use_rslora=False,
        revision=args.revision or None,
    )

    # the faithful gpu a/b proved this is needed: vl checkpoints (qwen3.5) draw rng during model
    # construction, so a resident cache-hit diverges without a re-seed before adapter init. resolved by
    # the load-independent re-seed just before the trainer build below (mirrored in production run_sft).
    _seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    load_started = time.perf_counter()
    base_cache_hit = False
    raw_full_before = None
    fingerprint_before = None
    if holder is None:
        sft_model = prepare_fresh_lora_base(
            args.model_id,
            args.model_id,
            model_init_kwargs,
            phase="resident-validation",
            model_revision=args.revision,
        )
    else:
        cached_model = holder._model
        base, resident_tokenizer = holder.get_base(
            args.model_id,
            revision=args.revision,
            model_init_kwargs=model_init_kwargs,
            context_length=args.max_length,
        )
        if resident_tokenizer is not tokenizer:
            raise RuntimeError("resident tokenizer changed during validation setup")
        base_cache_hit = cached_model is not None and base is cached_model
        raw_full_before = _full_base_fingerprint(base)
        fingerprint_before = base_fingerprint(base)
        sft_model = base
        config.model_init_kwargs = None
    torch.cuda.synchronize()
    model_prepare_seconds = time.perf_counter() - load_started

    setup_started = time.perf_counter()
    # re-seed right before the trainer/adapter build (mirrors production run_sft): makes the lora
    # init deterministic and independent of any rng the model construction consumed, so a resident
    # cache-hit matches a fresh load for vl checkpoints (qwen3.5/3.6) whose construction draws rng.
    # no-op for standard checkpoints (smollm2 stays byte-identical).
    _seed_everything(args.seed)
    trainer = SFTTrainer(
        model=sft_model,
        args=config,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    torch.cuda.synchronize()
    trainer_setup_seconds = time.perf_counter() - setup_started
    load_seconds = (
        model_prepare_seconds + trainer_setup_seconds if holder is None else model_prepare_seconds
    )
    full_before = _full_base_fingerprint(trainer.model)

    train_started = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    full_after = _full_base_fingerprint(trainer.model)
    full_unchanged = full_after == full_before
    if not full_unchanged:
        raise RuntimeError(f"full frozen base state changed in job {job_index}")
    losses = _loss_curve(trainer)
    adapter = _adapter_state(trainer.model)
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated())

    job_dir = output_dir / f"job-{job_index:02d}"
    job_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter, job_dir / "adapter.pt")
    record = {
        "job_index": job_index,
        "base_cache_hit": base_cache_hit,
        "load_seconds": load_seconds,
        "model_prepare_seconds": model_prepare_seconds,
        "trainer_setup_seconds": trainer_setup_seconds,
        "train_seconds": train_seconds,
        "losses": losses,
        "peak_gpu_bytes": peak_gpu_bytes,
        "base_fingerprint_before": fingerprint_before,
        "base_full_fingerprint_before": full_before,
        "base_full_fingerprint_after": full_after,
        "base_full_unchanged": full_unchanged,
    }

    if holder is not None:
        state = ResidentJobState(
            model=trainer.model,
            trainer=trainer,
            dataset=trainer.train_dataset,
        )
        holder.reset_after_job(state.model, trainer=state.trainer, dataset=state.dataset)
        record["base_fingerprint_after"] = base_fingerprint(base)
        record["base_full_fingerprint_after_reset"] = _full_base_fingerprint(base)
        record["base_full_reset_unchanged"] = (
            record["base_full_fingerprint_after_reset"] == raw_full_before
        )
        if record["base_fingerprint_after"] != fingerprint_before:
            raise RuntimeError(f"sampled base fingerprint changed in job {job_index}")
        if not record["base_full_reset_unchanged"]:
            raise RuntimeError(f"full base state changed during reset in job {job_index}")
        state.model = None
        state.trainer = None
        state.dataset = None

    (job_dir / "metrics.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def _run_child(args) -> int:
    from flash.engine.worker.resident import ResidentBase

    if not __import__("torch").cuda.is_available():
        raise RuntimeError("resident SFT validation requires a CUDA GPU")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    holder = None if args.worker_mode == "fresh" else ResidentBase()
    count = 1 if args.worker_mode == "fresh" else args.resident_jobs
    records = [_train_one(holder, args, index + 1, output_dir) for index in range(count)]
    (output_dir / "summary.json").write_text(json.dumps(records, indent=2, sort_keys=True))
    return 0


def _child_command(args, mode: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--model-id",
        args.model_id,
        "--steps",
        str(args.steps),
        "--resident-jobs",
        str(args.resident_jobs),
        "--seed",
        str(args.seed),
        "--max-length",
        str(args.max_length),
        "--lora-rank",
        str(args.lora_rank),
        "--learning-rate",
        str(args.learning_rate),
    ]
    if args.revision:
        command.extend(("--revision", args.revision))
    return command


def _load_adapter(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def _compare(args, fresh_dir: Path, resident_dir: Path) -> dict[str, Any]:
    import torch

    fresh_records = json.loads((fresh_dir / "summary.json").read_text())
    resident_records = json.loads((resident_dir / "summary.json").read_text())
    fresh = fresh_records[0]
    resident_second = resident_records[1]
    fresh_losses = torch.tensor(fresh["losses"], dtype=torch.float64)
    resident_losses = torch.tensor(resident_second["losses"], dtype=torch.float64)
    losses_match = fresh_losses.shape == resident_losses.shape and torch.allclose(
        fresh_losses,
        resident_losses,
        rtol=args.rtol,
        atol=args.atol,
    )
    max_loss_delta = (
        float(torch.max(torch.abs(fresh_losses - resident_losses)).item())
        if fresh_losses.shape == resident_losses.shape and fresh_losses.numel()
        else float("inf")
    )

    fresh_adapter = _load_adapter(fresh_dir / "job-01" / "adapter.pt")
    resident_adapter = _load_adapter(resident_dir / "job-02" / "adapter.pt")
    keys_match = fresh_adapter.keys() == resident_adapter.keys()
    exact_adapter = keys_match and all(
        torch.equal(fresh_adapter[name], resident_adapter[name]) for name in fresh_adapter
    )
    tolerant_adapter = keys_match and all(
        torch.allclose(
            fresh_adapter[name],
            resident_adapter[name],
            rtol=args.rtol,
            atol=args.atol,
        )
        for name in fresh_adapter
    )
    max_adapter_delta = (
        max(
            (
                float(
                    torch.max(
                        torch.abs(fresh_adapter[name].float() - resident_adapter[name].float())
                    ).item()
                )
                for name in fresh_adapter
            ),
            default=0.0,
        )
        if keys_match
        else float("inf")
    )

    steady_peaks = [record["peak_gpu_bytes"] for record in resident_records[1:]]
    memory_spread = max(steady_peaks) - min(steady_peaks)
    memory_ok = memory_spread <= args.memory_tolerance_mb * 1024 * 1024
    second_skipped_load = bool(
        resident_second["base_cache_hit"]
        and resident_second["load_seconds"] < fresh["load_seconds"]
    )
    base_immutable = bool(
        fresh["base_full_unchanged"]
        and all(
            record["base_full_unchanged"] and record["base_full_reset_unchanged"]
            for record in resident_records
        )
    )
    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "steps": args.steps,
        "fresh": fresh,
        "resident_jobs": resident_records,
        "resident_second_skipped_load": second_skipped_load,
        "full_base_immutable": base_immutable,
        "losses_match": bool(losses_match),
        "max_loss_delta": max_loss_delta,
        "adapter_exact": exact_adapter,
        "adapter_within_tolerance": tolerant_adapter,
        "max_adapter_delta": max_adapter_delta,
        "steady_peak_gpu_spread_bytes": memory_spread,
        "memory_within_tolerance": memory_ok,
        "rtol": args.rtol,
        "atol": args.atol,
    }
    if not (
        second_skipped_load and base_immutable and losses_match and tolerant_adapter and memory_ok
    ):
        raise RuntimeError(f"resident SFT validation failed: {json.dumps(result, indent=2)}")
    return result


def _run_parent(args) -> int:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    fresh_dir = output_dir / "fresh"
    resident_dir = output_dir / "resident"
    subprocess.run(_child_command(args, "fresh", fresh_dir), check=True)
    subprocess.run(_child_command(args, "resident", resident_dir), check=True)
    result = _compare(args, fresh_dir, resident_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--revision", default="")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--resident-jobs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--memory-tolerance-mb", type=float, default=64.0)
    parser.add_argument("--output-dir", default="/tmp/flash-resident-sft-validation")
    parser.add_argument("--worker-mode", choices=("parent", "fresh", "resident"), default="parent")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.resident_jobs < 3:
        parser.error("--resident-jobs must be at least 3")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.worker_mode == "parent":
        return _run_parent(args)
    return _run_child(args)


if __name__ == "__main__":
    raise SystemExit(main())
