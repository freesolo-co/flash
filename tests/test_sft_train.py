"""cpu contracts for the sft to verl migration."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flash.engine.profiling.sft_workload import _serialize_multimodal_inputs
from flash.engine.worker.backend_common import parse_verl_metric, verl_step_number
from flash.engine.worker.entry.sft import _pretokenize_completion_only
from flash.engine.worker.sft_train import (
    _LORAPLUS_READY_MARKER,
    _MAX_ZERO_GRAD_STEPS,
    _VERL_OPTIMIZER_IMPL,
    _VERL_OPTIMIZER_NAME,
    _build_verl_child_env,
    _render_sft_dataset_module,
    _render_sft_sitecustomize,
    _write_sft_parquet,
    build_sft_overrides,
    render_exact_sft_dataloader_shim,
    render_loraplus_shim,
)

# distinct from `flash.__version__` on purpose: the worker resolves that to "0+unknown" (no flash
# distribution is installed there), so a fixture built from it could not catch a worker that
# re-derives the producer version instead of reading the one carried on the spec.
_PROFILE_PRODUCER_VERSION = "9.9.9"


def _cfg(**over):
    base = {
        # required: run_sft_train always resolves this from the capability probe.
        "fused_ce_backend": "torch",
        "train_files": "/w/train.parquet",
        "train_batch_size": 32,
        "max_length": 32768,
        "micro_batch": 1,
        "max_token_len_per_gpu": 8192,
        "custom_dataset_path": "/w/flash_verl_sft_dataset.py",
        "model_path": "Qwen/Qwen3-4B",
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "ulysses_sp_size": 2,
        "lr": 1e-4,
        "warmup_ratio": 0.03,
        "optimizer_impl": "bitsandbytes.optim",
        "optimizer_name": "PagedAdamW8bit",
        "optimizer_kwargs": {},
        "local_dir": "/w/ckpt",
        "save_freq": 50,
        "n_gpus_per_node": 2,
        "seed": 42,
        "project_name": "flash-sft",
        "experiment_name": "run-xyz",
        "loop_epochs": 4,
        "total_training_steps": 120,
    }
    base.update(over)
    return base


def _as_map(overrides):
    return dict(override.split("=", 1) for override in overrides)


def test_overrides_match_verl_0_8_sft_and_fsdp_config_surface():
    overrides = _as_map(build_sft_overrides(_cfg()))
    assert overrides == {
        "data.train_files": "/w/train.parquet",
        # hardcoded null, not a cfg value: see test_sft_ships_no_val_file_so_the_child_cannot_validate
        "data.val_files": "null",
        "data.train_batch_size": "32",
        "data.max_length": "32768",
        "data.micro_batch_size_per_gpu": "1",
        "data.use_dynamic_bsz": "true",
        "data.max_token_len_per_gpu": "8192",
        "data.truncation": "right",
        "data.num_workers": "4",
        "data.ignore_input_ids_mismatch": "false",
        "data.custom_cls.path": "/w/flash_verl_sft_dataset.py",
        "data.custom_cls.name": "FlashTokenizedSFTDataset",
        "model.path": "Qwen/Qwen3-4B",
        "model.trust_remote_code": "true",
        "model.lora_rank": "16",
        "model.lora_alpha": "32",
        "model.target_modules": "all-linear",
        "model.lora_adapter_path": "null",
        "model.use_remove_padding": "true",
        "model.use_liger": "false",
        "model.enable_gradient_checkpointing": "true",
        "engine.strategy": "fsdp2",
        "engine.model_dtype": "bfloat16",
        "engine.seed": "42",
        "engine.ulysses_sequence_parallel_size": "2",
        "optim.lr": "0.0001",
        "optim.lr_warmup_steps_ratio": "0.03",
        "optim.optimizer_impl": "bitsandbytes.optim",
        "optim.optimizer": "PagedAdamW8bit",
        "optim.weight_decay": "0.0",
        "optim.betas": "[0.9,0.999]",
        "optim.override_optimizer_config": "{eps:0.00000001}",
        "trainer.default_local_dir": "/w/ckpt",
        "trainer.save_freq": "50",
        "trainer.n_gpus_per_node": "2",
        "trainer.nnodes": "1",
        "trainer.seed": "42",
        "trainer.logger": "[console]",
        "trainer.project_name": "flash-sft",
        "trainer.experiment_name": "run-xyz",
        "trainer.total_epochs": "4",
        "trainer.test_freq": "-1",
        "trainer.resume_mode": "auto",
        "model.use_fused_kernels": "true",
        "model.fused_kernel_options.impl_backend": "torch",
        "trainer.max_ckpt_to_keep": "1",
        "trainer.total_training_steps": "120",
    }
    assert "optim.eps" not in overrides
    assert "optim.lr_scheduler_type" not in overrides
    assert "data.messages_key" not in overrides


def test_overrides_carry_fused_expert_target_parameters():
    overrides = _as_map(
        build_sft_overrides(
            _cfg(
                target_parameters=[
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                ]
            )
        )
    )

    assert overrides["++model.target_parameters"] == (
        "[mlp.experts.gate_up_proj,mlp.experts.down_proj]"
    )


def test_verl_sft_optimizer_is_dtensor_safe():
    """the fsdp2 engine hands DTensor params to the optimizer.

    bitsandbytes' 8-bit blockwise kernel is not a distributed operator and raises
    "got mixed torch.Tensor and DTensor" on the first step, so the verl SFT path must
    never select an 8-bit optimizer regardless of what the TRL memory profile prefers.
    """
    assert (_VERL_OPTIMIZER_IMPL, _VERL_OPTIMIZER_NAME) == ("torch.optim", "AdamW")
    assert "8bit" not in _VERL_OPTIMIZER_NAME.lower()
    assert "bitsandbytes" not in _VERL_OPTIMIZER_IMPL


def test_sft_engine_strategy_stays_fsdp2():
    """LoRA+ groups parameters by name ("lora_B" in name).

    fsdp1 flattens parameters into a 1-D flat_param, which would route every parameter
    into the 16x group B and silently corrupt the learning rates, so the DTensor problem
    above must not be "fixed" by downgrading the strategy.
    """
    assert _as_map(build_sft_overrides(_cfg()))["engine.strategy"] == "fsdp2"


def test_verl_packs_every_batch_so_the_batch_size_is_the_isolation_boundary():
    """verl concatenates a batch into one sequence, so `train_batch_size` decides what shares it.

    The worker sends `model.use_remove_padding=true` and leaves `data.pad_mode` at verl's
    `no_padding` default, which together make the fsdp engine hand the model a single
    ``(1, total_nnz)`` row with ``attention_mask=None`` and per-example ``position_ids`` restarts.
    Attention recovers its boundaries from those restarts; GatedDeltaNet layers do not, because
    they read ``seq_idx`` and ``cu_seq_lens_q`` out of kwargs the fsdp engine never sends. So on a
    gdn hybrid -- which every catalog model is -- the batch size is the only thing standing between
    one example and the next example's carried state, and a profile that grouped examples for
    costing convenience would silently corrupt training. Pin both halves of that coupling: neither
    override may drift without this failing.
    """
    overrides = _as_map(build_sft_overrides(_cfg(train_batch_size=1)))

    assert overrides["model.use_remove_padding"] == "true"
    assert "data.pad_mode" not in overrides
    assert overrides["data.train_batch_size"] == "1"


def test_sft_pins_ulysses_off_because_sequence_parallelism_breaks_gdn():
    """`ulysses_sp_size` must be the literal 1, never the card count.

    Two independent reasons, either sufficient. Correctness: every catalog model is a GDN hybrid
    whose layers are mostly linear attention plus a short causal conv, and both carry state ALONG
    the sequence. Pinned verl patches ulysses into `_flash_attention_forward` and slices the Qwen
    text model's inputs, but passes no recurrent or conv state between ranks -- so a sequence shard
    would run its recurrence as if it were a whole sequence. Liveness: it also crashed, because
    remove-padding leaves one `(1, total_nnz)` row and the slice desynchronizes the shapes the GDN
    kernels are handed (`seq_idx must have shape (batch_size, seqlen)`), at every batch size.

    Read the source: `build_sft_overrides` renders whatever it is given, so a test driving a cfg
    dict would assert on its own fixture and stay green if the caller went back to `gpu_count`.
    `_prepare_sft_child` itself downloads weights, so the source is what is reachable offline.

    Two assertions, because the site names a shared constant rather than a literal: the grep proves
    the WIRING (this site did not regress to the card count), and the constant proves the VALUE.
    Either alone would pass while the contract was broken.
    """
    import inspect

    from flash.engine.worker import sft_train_runner
    from flash.engine.worker.verl.parallelism import ULYSSES_SEQUENCE_PARALLEL_SIZE

    src = inspect.getsource(sft_train_runner._prepare_sft_child)
    line = next(ln.strip() for ln in src.splitlines() if ln.strip().startswith('"ulysses_sp_size"'))

    assert line == '"ulysses_sp_size": ULYSSES_SEQUENCE_PARALLEL_SIZE,', line
    assert ULYSSES_SEQUENCE_PARALLEL_SIZE == 1


def test_sft_card_count_never_starves_a_rank_of_its_batch():
    """The card count must divide the batch, because verl floor-divides the batch across ranks.

    With ulysses pinned off, `dp_size == world_size`, and verl computes
    `train_batch_size_per_dp = train_batch_size // dp_size` and hands that straight to a
    DataLoader. Two widths are unusable: one ABOVE the batch floors to 0, and
    `DataLoader(batch_size=0)` raises ValueError; one that does not DIVIDE the batch silently
    shrinks the global batch, because the sampler and loader both drop the remainder.

    The unpacked case is the one that matters most: `examples_per_update` is 1 for every
    exact-unpacked run, which is exactly what a GDN model without the boundary-reset contract
    gets -- so a 2-card allocation would otherwise compute 1 // 2 == 0 and die before step 1.
    """
    from flash.engine.worker.sft_train_runner import sft_data_parallel_cards

    # unpacked: one example cannot be split, so extra cards have nothing to hold.
    for cards in (1, 2, 4, 8):
        assert sft_data_parallel_cards(cards, 1) == 1

    # the batch divides the allocation: use every card.
    assert sft_data_parallel_cards(2, 32) == 2
    assert sft_data_parallel_cards(4, 32) == 4
    assert sft_data_parallel_cards(8, 32) == 8

    # it does not divide: fall to the largest divisor <= the allocation, never a remainder split.
    assert sft_data_parallel_cards(4, 6) == 3
    assert sft_data_parallel_cards(8, 12) == 6
    assert sft_data_parallel_cards(3, 4) == 2
    assert sft_data_parallel_cards(4, 10) == 2

    # batch smaller than the allocation: bounded by the batch, never 0.
    assert sft_data_parallel_cards(8, 2) == 2
    assert sft_data_parallel_cards(8, 3) == 3

    # exhaustive: never 0, never above the allocation, and always an exact divisor of the batch --
    # the three properties that together mean no rank is starved and the global batch is preserved.
    for cards in range(1, 9):
        for batch in range(1, 33):
            resolved = sft_data_parallel_cards(cards, batch)
            assert 1 <= resolved <= cards
            assert batch % resolved == 0, (cards, batch, resolved)


def test_sft_warns_while_the_run_is_live_when_it_leaves_cards_idle(monkeypatch, capsys):
    """An unused card is billed, so the run must say so while it is running.

    Reducing the width keeps the run correct, but it is not free: the allocation is charged whole.
    The notes record the executed width, and those are read afterwards -- the warning is what makes
    the waste visible in `flash runs log` in time to cancel and resubmit.
    """
    from flash.engine.worker import sft_train

    spec, captured = _stub_sft_run(monkeypatch)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        captured["command"] = command
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        on_line("step:2 - train/loss:1.0 - train/global_tokens:8\n")
        on_step(2)
        heartbeat()
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)
    sft_train.run_sft_train(spec)

    # the fixture allocates 2 cards and resolves to 1 (unpacked profile -> batch of 1).
    out = capsys.readouterr().out
    assert "[sft][warn] training on 1 of 2 allocated cards" in out, out
    assert "still billed" in out
    assert "--nproc-per-node=1" in captured["command"]


def test_sft_width_never_drops_a_profiled_row():
    """The width must divide the ROW count, not just the batch.

    verl builds `DistributedSampler(..., drop_last=True)` (`sft_trainer.py:237`) and Flash's
    exact-dataloader shim overrides `drop_last` on the LOADER only -- its sampler patch sets
    `shuffle` and nothing else. So a width that leaves a row remainder drops it from every epoch
    while the frozen quote still bills it: 11 rows on 2 ranks trains 10, on 4 ranks trains 8.

    This could not fire before Ulysses was pinned off, because `sp = gpu_count` forced `dp_size`
    to 1. Making multi-rank SFT reachable is what puts this in scope.
    """
    from flash.engine.plan.steps import sft_data_parallel_cards

    # 11 rows is prime: no width above 1 divides it, so every extra card would drop rows.
    assert sft_data_parallel_cards(4, 8, 11) == 1
    # 12 rows with batch 8 -> 4 divides both.
    assert sft_data_parallel_cards(4, 8, 12) == 4
    # rows divide but the batch does not: the batch still binds.
    assert sft_data_parallel_cards(4, 2, 12) == 2
    # batch divides but the rows do not: the rows now bind. 8 % 2 == 0, but 10 % 4 != 0.
    assert sft_data_parallel_cards(4, 8, 10) == 2

    # unknown row count (the cost path quotes before the dataset exists) must not constrain.
    assert sft_data_parallel_cards(4, 8) == 4
    assert sft_data_parallel_cards(4, 8, 0) == 4

    # exhaustive: whatever comes back divides BOTH, so no rank is starved and no row is dropped.
    for cards in range(1, 9):
        for batch in range(1, 17):
            for rows in range(1, 25):
                got = sft_data_parallel_cards(cards, batch, rows)
                assert 1 <= got <= cards
                assert batch % got == 0, (cards, batch, rows, got)
                assert rows % got == 0, (cards, batch, rows, got)


def test_sft_idle_card_warning_names_a_remedy_that_can_actually_work():
    """The advice has to move the width it is printed about, and name a shape you can rent.

    Two ways this line can be confidently wrong. An unpacked profile pins the batch to 1 in
    `sft_workload` regardless of `[train] batch_size`, so telling that operator to raise the batch
    points at a knob that cannot change the answer. And only powers of two are rentable, so naming
    an odd rank count as an allocation buys the next one DOWN -- "allocate 3" gets 2 cards, which
    can leave a run that only fit on 4 unplaceable.
    """
    from flash.engine.worker.sft_train_runner import _resolve_sft_world_size

    unpacked = io.StringIO()
    with contextlib.redirect_stdout(unpacked):
        assert _resolve_sft_world_size(2, 1, 12) == 1
    text = unpacked.getvalue()
    assert "[sft][warn] training on 1 of 2 allocated cards" in text, text
    assert "batch_size" not in text, "raising the batch cannot move an unpacked run off one card"
    assert "allocate 1 card(s)" in text, text

    # batch 6 on 4 cards resolves to 3 ranks, but 3 is not rentable -- recommend 2.
    odd = io.StringIO()
    with contextlib.redirect_stdout(odd):
        assert _resolve_sft_world_size(4, 6, 12) == 3
    text = odd.getvalue()
    assert "training on 3 of 4 allocated cards" in text, text
    assert "allocate 2 card(s)" in text, text
    assert "allocate 3 card(s)" not in text


def test_sft_quote_credits_only_the_ranks_that_will_execute():
    """A quote must not promise throughput from cards the batch cannot feed.

    `gpu_count` at the quote boundary is the BILLED shape. SFT shards by data, so the executed
    width is bounded by the batch: an unpacked run on 2 cards trains on one rank. Crediting the
    billed width there understates wall time against the run's own cap. GRPO and OPD also shard by
    data, but they bound work by TOKENS (`use_dynamic_bsz`), so the scheduler balances the batch
    across every rank instead of leaving one unfed -- their executed width is the allocation, and the
    clamp must stay SFT-only.
    """
    from flash.cost import analytical
    from flash.cost.types import RunConfig

    def speedup(method: str, batch: int, cards: int) -> float:
        config = RunConfig(model_id="Qwen/Qwen3.5-4B", method=method, steps=10, batch_size=batch)
        return analytical.method_card_speedup(config, cards, "H100", "runpod")

    one_card = speedup("sft", 1, 1)
    # unpacked sft: 2 billed cards, 1 executing rank -> quoted like the single card it is.
    assert speedup("sft", 1, 2) == one_card
    # a batch that divides the allocation keeps the full multi-card credit.
    assert speedup("sft", 8, 2) > one_card
    # grpo shards by data too, but token-balanced across every card, so it is untouched by the batch.
    assert speedup("grpo", 1, 2) > speedup("grpo", 1, 1)


def test_sft_stays_quiet_when_every_allocated_card_is_used(monkeypatch, capsys):
    """The warning must not fire on the normal path, or it trains readers to ignore it."""
    from flash.engine.worker import sft_train

    spec, _ = _stub_sft_run(monkeypatch)

    # one card allocated: the resolved width can only equal it, so there is nothing to warn about.
    spec.gpu.count = 1

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        on_line("step:2 - train/loss:1.0 - train/global_tokens:8\n")
        on_step(2)
        heartbeat()
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)
    sft_train.run_sft_train(spec)

    assert "allocated cards" not in capsys.readouterr().out


def test_sft_launches_the_resolved_width_not_the_allocated_cards():
    """torchrun must start the RESOLVED rank count.

    `sft_data_parallel_cards` is inert if the child is still launched with `--nproc-per-node` set
    to the allocated card count: verl would split the batch across every rank torchrun started, so
    the extra ranks would hit the batch_size=0 crash the resolver exists to prevent.
    """
    import inspect

    from flash.engine.worker import sft_train_runner

    src = inspect.getsource(sft_train_runner._prepare_sft_child)
    # the ARGUMENT line, not a comment that happens to name the flag -- match on the f-string so a
    # nearby comment mentioning `--nproc-per-node` cannot be picked up instead.
    line = next(
        ln.strip()
        for ln in src.splitlines()
        if "nproc-per-node" in ln and not ln.strip().startswith("#")
    )

    assert line == 'f"--nproc-per-node={world_size}",', line
    assert '"n_gpus_per_node": world_size,' in src
    # and the width is the RESOLVED one, not the raw allocation. the resolution lives in
    # `_resolve_sft_width_and_micro_batch` (it also caps the micro-batch, which needs the width),
    # so follow it there rather than pinning a call that moved.
    assert "_resolve_sft_width_and_micro_batch(options, data, model)" in src
    assert "options.gpu_count" in inspect.getsource(
        sft_train_runner._resolve_sft_width_and_micro_batch
    )


def test_remove_padding_is_unconditional():
    """Nothing may gate `use_remove_padding`: it is the only layout verl's sft_loss can consume.

    verl leaves `data.pad_mode` at its `no_padding` default, and that loss path reads
    ``log_prob.values()`` -- which only exists on the nested tensor the remove-padding branch
    builds. Send `use_remove_padding=false` and the fsdp engine hands the loss a strided tensor
    instead, so the first optimizer step dies with "values expected sparse tensor layout but got
    Strided" before a single update lands.

    Switching to verl's padded mode is not an escape either: `pad_mode: right` collates with
    ``default_collate`` (uniform rows only) and its loss branch reads ``response_mask``, and
    ``FlashTokenizedSFTDataset`` emits neither -- it yields variable-length rows carrying
    input_ids/position_ids/loss_mask. So `no_padding` + remove-padding is the only combination
    this dataset fits, and the flag has no legitimate false case.

    Two revisions learned this the hard way, each gating the flag on something real but
    irrelevant: first `packing_mode == "packed"` (the profile), then `not gdn_hybrid or
    gdn_boundary_resets` (the child probe). Both reasoned that a gdn hybrid without child-side
    resets must not pack -- true, but it never does: `_packing_mode` answers "exact-unpacked" for
    every gdn model, which pins `examples_per_update` to 1, so a gdn run has no packed neighbour
    to be contaminated by. Batch size is the isolation lever; this flag never was.

    Read the source rather than the rendered overrides: `build_sft_overrides` takes the flag
    already computed, so a test driving it through a cfg dict asserts on its own fixture and stays
    green no matter what the derivation does.
    """
    import inspect

    from flash.engine.worker import sft_train

    src = inspect.getsource(sft_train.run_sft_train)
    line = next(
        ln.strip() for ln in src.splitlines() if ln.strip().startswith("use_remove_padding =")
    )

    assert line == "use_remove_padding = True", line


@pytest.mark.parametrize(
    ("support", "expected_mode"),
    [
        (("gdn-hybrid", False), "exact-unpacked"),
        (("unsupported", False), "exact-unpacked"),
        (("pure-attention", True), "packed"),
    ],
)
def test_batch_is_the_isolation_lever_that_replaces_the_removed_flag(support, expected_mode):
    """Deleting the gate is only safe because no unsupported architecture ever packs.

    `use_remove_padding` used to be the (broken) isolation lever. The real one is the batch size,
    and it holds one step earlier: an architecture that cannot reset state at packed boundaries
    profiles as `exact-unpacked`, which pins `examples_per_update` to 1, so the run has no packed
    neighbour whose residue could bleed across. Pin the chain at its source -- if a future change
    let a gdn hybrid profile as "packed", the deleted flag would no longer be there to catch it.

    Read `_packing_mode` and the `examples_per_update` derivation directly rather than asserting
    on rendered overrides: `build_sft_overrides` defaults `use_remove_padding` to True when the key
    is absent, so an override-level assertion reads that default and passes no matter what the
    worker computed.
    """
    from flash.engine.profiling.sft_workload import _packing_mode

    packing_mode, architecture_mode = _packing_mode(
        "Qwen/Qwen3.5-4B",
        "rev",
        multimodal=False,
        allow_packing=True,
        packing_support=lambda _m, _r: support,
    )

    assert (packing_mode, architecture_mode) == (expected_mode, support[0])
    # an architecture that cannot pack must not be able to reach a batch > 1
    assert (expected_mode == "packed") is support[1]


def test_optimizer_eps_merges_into_override_config():
    overrides = _as_map(build_sft_overrides(_cfg(optimizer_kwargs={"amsgrad": True}, eps=1e-6)))
    assert overrides["optim.override_optimizer_config"] == "{amsgrad:true,eps:0.000001}"


def test_small_lr_renders_fixed_point_not_scientific():
    overrides = _as_map(build_sft_overrides(_cfg(lr=5e-5)))
    assert overrides["optim.lr"] == "0.00005"


def test_steps_xor_epochs_is_enforced():
    with pytest.raises(ValueError, match="exactly one"):
        build_sft_overrides(_cfg(total_training_steps=120, total_epochs=3))
    with pytest.raises(ValueError, match="exactly one"):
        build_sft_overrides(_cfg(total_training_steps=None, total_epochs=None))


class _ExactTokenizer:
    eos_token = "!"
    all_special_ids = (0,)

    def __call__(self, texts, *, truncation=False, max_length=None):
        ids = [[ord(char) for char in text] for text in texts]
        # the uncapped encode is how a truncated row reports its real size rather than the cap.
        if not truncation:
            return {"input_ids": ids}
        assert max_length is not None, "truncation=True requires an explicit max_length"
        return {"input_ids": [row[:max_length] for row in ids]}


def test_exact_mask_keeps_prompt_assistant_history_masked_and_full_target_active():
    tokenizer = _ExactTokenizer()
    prompt = "<user>q</user><assistant>history</assistant><assistant>"
    full = prompt + "first</assistant><user>tool</user><assistant>second</assistant>"
    texts = [{"text": full, "prompt_text": prompt}]

    kept, rows, dropped = _pretokenize_completion_only(texts, tokenizer, max_length=512)

    assert kept == texts
    assert dropped == 0
    split = len(prompt)
    assert rows[0]["completion_mask"][:split] == [0] * split
    assert all(rows[0]["completion_mask"][split:])
    assert len(rows[0]["input_ids"]) == len(rows[0]["completion_mask"])


def test_exact_mask_drops_right_truncated_completion_and_handles_thinking_prefix():
    tokenizer = _ExactTokenizer()
    prompt = "<think>prompt<assistant>"
    full = "<think>prompt<assistant>answer"
    kept, rows, dropped = _pretokenize_completion_only(
        [{"text": full, "prompt_text": prompt}],
        tokenizer,
        max_length=len(prompt),
    )
    assert kept == []
    assert rows == []
    assert dropped == 1

    kept, rows, dropped = _pretokenize_completion_only(
        [{"text": full, "prompt_text": prompt}],
        tokenizer,
        max_length=512,
    )
    assert kept
    assert dropped == 0
    assert any(rows[0]["completion_mask"])


def _load_custom_dataset_module(tmp_path):
    path = tmp_path / "flash_verl_sft_dataset.py"
    path.write_text(_render_sft_dataset_module())
    spec = importlib.util.spec_from_file_location("flash_verl_sft_dataset_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("image_first", [False, True])
def test_explicit_schema_and_custom_dataset_read_text_image_orders(
    monkeypatch, tmp_path, image_first
):
    np = pytest.importorskip("numpy")

    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def __len__(self):
            return len(self.value)

        def tolist(self):
            return self.value.tolist()

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.value, axis))

    fake_torch = _module("torch")
    fake_torch.long = np.int64
    fake_torch.tensor = lambda value, dtype=None: FakeTensor(value)
    fake_torch.arange = lambda length, dtype=None: FakeTensor(np.arange(length))
    fake_torch.ones_like = lambda value: FakeTensor(np.ones_like(value.value))
    fake_torch.from_numpy = FakeTensor
    fake_torch.cat = lambda values, dim=0: FakeTensor(
        np.concatenate([value.value for value in values], axis=dim)
    )
    image_row = {
        "input_ids": [20, 21, 22],
        "loss_mask": [0, 1, 1],
        "images": ["file:///tmp/image.png"],
        "multimodal_inputs": _serialize_multimodal_inputs(
            {"image_grid_thw": np.asarray([[1, 2, 3]], dtype=np.int64)}
        ),
    }
    text_row = {
        "input_ids": [10, 11],
        "loss_mask": [0, 1],
        "images": [],
        "multimodal_inputs": b"",
    }
    rows = [image_row, text_row] if image_first else [text_row, image_row]
    parquet = tmp_path / "mixed.parquet"
    _write_sft_parquet(rows, str(parquet))

    datasets = pytest.importorskip("datasets")
    raw = datasets.Dataset.from_parquet(str(parquet))
    assert raw.column_names == ["input_ids", "loss_mask", "images", "multimodal_inputs"]
    assert raw[0]["images"] == rows[0]["images"]
    assert raw[1]["images"] == rows[1]["images"]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    qwen_module = _module(
        "verl.models.transformers.qwen2_vl",
        get_rope_index=lambda processor, input_ids, **kwargs: FakeTensor(
            np.zeros((3, len(input_ids)), dtype=np.int64)
        ),
    )
    for name, injected in {
        "verl": _module("verl"),
        "verl.models": _module("verl.models"),
        "verl.models.transformers": _module("verl.models.transformers"),
        "verl.models.transformers.qwen2_vl": qwen_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, injected)

    class Qwen2VLImageProcessorFake:
        pass

    module = _load_custom_dataset_module(tmp_path)
    dataset = module.FlashTokenizedSFTDataset(
        parquet_files=str(parquet),
        tokenizer=SimpleNamespace(),
        processor=SimpleNamespace(image_processor=Qwen2VLImageProcessorFake()),
        config={"max_length": 8, "truncation": "right", "ignore_input_ids_mismatch": False},
    )
    first = dataset[0]
    second = dataset[1]
    assert first["input_ids"].tolist() == rows[0]["input_ids"]
    assert first["loss_mask"].tolist() == rows[0]["loss_mask"]
    assert second["input_ids"].tolist() == rows[1]["input_ids"]
    assert second["loss_mask"].tolist() == rows[1]["loss_mask"]
    assert len(first["position_ids"].tolist()) == 4
    assert len(second["position_ids"].tolist()) == 4
    image_item = first if image_first else second
    assert image_item["multi_modal_inputs"]["image_grid_thw"].tolist() == [[1, 2, 3]]


def test_mrope_processor_check_matches_every_image_training_model():
    """The mRoPE branch is selected by processor class NAME, so it must match the real classes.

    The dataset picks grid-aligned multi-axis positions only when the processor's image_processor
    class name contains a literal; anything else falls through to a 1-D `arange`, which a
    vision-language forward cannot use. The test above supplies a fake it names itself, so it
    proves the branch works but NOT that the literal matches what transformers actually builds --
    a reviewer read the Qwen3.5/3.6 catalog as using a distinct Qwen3 processor for exactly this
    reason.

    They do not: transformers maps the whole qwen3_5 family onto the Qwen2VL processor. This reads
    the real mapping rather than restating it, so a future transformers pin that introduces a
    genuine Qwen3 processor class fails here instead of silently training images at wrong
    positions.
    """
    from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES

    from flash.core.catalog import _IMAGE_TRAINING_MODELS

    # take the literal from the shipped source, so the guard cannot drift from the branch it guards.
    rendered = _render_sft_dataset_module()
    match = re.search(r'"([A-Za-z0-9_]+)" in self\.processor\.image_processor', rendered)
    assert match, "the mRoPE branch no longer selects on an image_processor class name"
    literal = match.group(1)

    # config model_type for the catalog's image-training models. Qwen3.6-* ship as qwen3_5 configs
    # (the product version is not the architecture), and the -A3B moe variant is a separate one.
    model_types = {"qwen3_5", "qwen3_5_moe"}
    assert len(_IMAGE_TRAINING_MODELS) == 6, (
        "the image-training catalog changed; confirm the new models' config model_type is covered"
    )

    for model_type in sorted(model_types):
        entry = IMAGE_PROCESSOR_MAPPING_NAMES.get(model_type)
        assert entry, f"transformers has no image processor mapped for {model_type}"
        # every variant is checked, not just the default: the substring test is what makes the
        # branch tolerate the Fast/Pil suffixes, and a real model advertises the suffixed name
        # (Qwen3.5-0.8B's preprocessor_config.json says Qwen2VLImageProcessorFast).
        names = list(entry.values()) if isinstance(entry, dict) else [entry]
        for name in names:
            assert literal in name, (
                f"{model_type} builds {name}, which the mRoPE check ({literal!r}) does not match; "
                "image SFT would fall through to 1-D arange positions on a paid run"
            )


@pytest.mark.parametrize("shape", ["listconfig", "list", "str"])
def test_custom_dataset_accepts_every_parquet_files_shape(tmp_path, shape):
    """verl passes data.train_files through from hydra, so the dataset sees a ListConfig."""
    module = _load_custom_dataset_module(tmp_path)
    parquet = tmp_path / "rows.parquet"
    rows = [{"input_ids": [1, 2], "loss_mask": [0, 1], "images": [], "multimodal_inputs": b""}]
    _write_sft_parquet(rows, str(parquet))

    if shape == "listconfig":
        # omegaconf is not a flash dependency (it lives in the verl venv), so stand in for
        # ListConfig with the property that actually broke: a sequence that is not a list/tuple.
        class _ListConfig:
            def __init__(self, items):
                self._items = list(items)

            def __iter__(self):
                return iter(self._items)

            def __len__(self):
                return len(self._items)

        assert not isinstance(_ListConfig([]), (list, tuple))
        parquet_files = _ListConfig([str(parquet)])
    elif shape == "list":
        parquet_files = [str(parquet)]
    else:
        parquet_files = str(parquet)

    dataset = module.FlashTokenizedSFTDataset(
        parquet_files=parquet_files,
        tokenizer=SimpleNamespace(),
        config={"max_length": 8, "truncation": "right", "ignore_input_ids_mismatch": False},
    )
    # constructing is the assertion: the ListConfig shape used to die here reading the parquet.
    assert len(dataset) == 1
    assert list(dataset.dataframe["input_ids"].iloc[0]) == [1, 2]


def test_custom_dataset_rejects_suppressed_input_id_checks(tmp_path):
    module = _load_custom_dataset_module(tmp_path)
    parquet = tmp_path / "one.parquet"
    _write_sft_parquet(
        [{"input_ids": [1], "loss_mask": [1], "images": [], "multimodal_inputs": b""}],
        str(parquet),
    )
    with pytest.raises(ValueError, match="mismatch checks"):
        module.FlashTokenizedSFTDataset(
            parquet_files=str(parquet),
            tokenizer=SimpleNamespace(),
            config={"max_length": 8, "ignore_input_ids_mismatch": True},
        )


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_generated_sitecustomize_installs_linear_scheduler_and_required_loraplus(
    monkeypatch, capsys
):
    class FakeLoader:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    class FakeTrainer:
        def _build_dataloader(self):
            self.train_sampler = SimpleNamespace(seed=None)
            self.val_sampler = None
            return "loader"

    class FakeCheckpointHandler:
        def save_checkpoint(self, step):
            return step

    class FakeEngine:
        rank = 0

        def _build_optimizer(self, module):
            return "plain"

        def _build_lr_scheduler(self, optimizer):
            return "constant"

        def _build_module(self):
            return SimpleNamespace(gradient_checkpointing_enable=lambda **kwargs: None)

    fake_torch = _module("torch")
    fake_torch.manual_seed = lambda seed: None
    fake_torch.set_float32_matmul_precision = lambda value: None
    fake_torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=False),
    )
    fake_numpy = _module("numpy", random=SimpleNamespace(seed=lambda seed: None))
    scheduler_calls = []
    fake_transformers = _module(
        "transformers",
        get_linear_schedule_with_warmup=lambda optimizer, **kwargs: (
            scheduler_calls.append((optimizer, kwargs)) or "linear"
        ),
    )
    fake_sft_module = _module(
        "verl.trainer.sft_trainer",
        StatefulDataLoader=FakeLoader,
        SFTTrainer=FakeTrainer,
    )
    fake_checkpoint_module = _module(
        "verl.utils.checkpoint.checkpoint_handler",
        CheckpointHandler=FakeCheckpointHandler,
    )
    fake_fsdp_module = _module(
        "verl.workers.engine.fsdp.transformer_impl",
        FSDPEngine=FakeEngine,
    )
    optimizer_calls = []
    fake_peft = _module(
        "peft.optimizers",
        create_loraplus_optimizer=lambda **kwargs: optimizer_calls.append(kwargs) or "lora+",
    )
    fake_optimizer_module = _module("fake_optimizer", AdamW=type("AdamW", (), {}))

    modules = {
        "torch": fake_torch,
        "numpy": fake_numpy,
        "transformers": fake_transformers,
        "verl": _module("verl"),
        "verl.trainer": _module("verl.trainer", sft_trainer=fake_sft_module),
        "verl.trainer.sft_trainer": fake_sft_module,
        "verl.utils": _module("verl.utils"),
        "verl.utils.checkpoint": _module("verl.utils.checkpoint"),
        "verl.utils.checkpoint.checkpoint_handler": fake_checkpoint_module,
        "verl.workers": _module("verl.workers"),
        "verl.workers.engine": _module("verl.workers.engine"),
        "verl.workers.engine.fsdp": _module("verl.workers.engine.fsdp"),
        "verl.workers.engine.fsdp.transformer_impl": fake_fsdp_module,
        "peft": _module("peft"),
        "peft.optimizers": fake_peft,
        "fake_optimizer": fake_optimizer_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = _render_sft_sitecustomize(
        seed=43,
        loraplus_ratio=16,
        save_at_steps=(3, 7),
        total_steps=9,
        reentrant_gradient_checkpointing=False,
    )
    exec(compile(source, "sitecustomize.py", "exec"), {})

    engine = FakeEngine()
    engine.optimizer_config = SimpleNamespace(
        optimizer_impl="fake_optimizer",
        optimizer="AdamW",
        lr=5e-5,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        override_optimizer_config={"eps": 1e-8},
        lr_warmup_steps=-1,
        lr_warmup_steps_ratio=0.1,
        total_training_steps=20,
    )
    assert engine._build_optimizer(SimpleNamespace()) == "lora+"
    assert _LORAPLUS_READY_MARKER in capsys.readouterr().out
    assert optimizer_calls[0]["optimizer_kwargs"]["eps"] == 1e-8
    assert engine._build_lr_scheduler("optimizer") == "linear"
    assert scheduler_calls == [("optimizer", {"num_warmup_steps": 2, "num_training_steps": 20})]


def _exec_dataloader_shim(monkeypatch):
    """Execute the real rendered dataloader shim against fake torch classes, and return them.

    The shim is a string concatenated into the shipped sitecustomize, so nothing else in the suite
    would notice it breaking: an import it cannot satisfy, or a patch that silently stops applying,
    surfaces on a rented gpu as rows arriving in a different order than the profile measured.
    """
    sampler_calls: list[dict] = []
    loader_calls: list[dict] = []

    class FakeDistributedSampler:
        def __init__(self, dataset, **kwargs):
            sampler_calls.append(kwargs)

    class FakeStatefulDataLoader:
        def __init__(self, dataset, **kwargs):
            loader_calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "torch.utils.data.distributed",
        _module("torch.utils.data.distributed", DistributedSampler=FakeDistributedSampler),
    )
    monkeypatch.setitem(
        sys.modules,
        "torchdata.stateful_dataloader",
        _module("torchdata.stateful_dataloader", StatefulDataLoader=FakeStatefulDataLoader),
    )

    exec(compile(render_exact_sft_dataloader_shim(), "sitecustomize.py", "exec"), {})
    return FakeDistributedSampler, FakeStatefulDataLoader, sampler_calls, loader_calls


def test_dataloader_shim_forces_the_row_order_the_profile_measured(monkeypatch):
    """Shuffle off and drop_last off, whatever verl asked for.

    Both are overrides, not defaults: verl passes shuffle=True and drop_last=True itself, so a shim
    that merely supplied them when absent would change nothing. Shuffling would reorder the rows
    the profile tokenized in a fixed order, and dropping the last partial batch would train on
    fewer examples than were measured and quoted -- neither of which fails loudly.
    """
    sampler, loader, sampler_calls, loader_calls = _exec_dataloader_shim(monkeypatch)

    sampler(["row"], shuffle=True, num_replicas=1, rank=0)
    loader(["row"], drop_last=True, batch_size=2)

    assert sampler_calls == [{"shuffle": False, "num_replicas": 1, "rank": 0}]
    assert loader_calls == [{"drop_last": False, "batch_size": 2}]


def test_dataloader_shim_sets_both_flags_when_the_caller_omits_them(monkeypatch):
    """A caller that passes neither must still get the exact-order behaviour.

    verl currently passes both, but the shim's guarantee cannot depend on that: a verl release that
    started relying on the library defaults (shuffle=True, drop_last=False for the sampler) would
    silently reintroduce shuffling through a shim that only rewrote what it was given.
    """
    sampler, loader, sampler_calls, loader_calls = _exec_dataloader_shim(monkeypatch)

    sampler(["row"])
    loader(["row"])

    assert sampler_calls == [{"shuffle": False}]
    assert loader_calls == [{"drop_last": False}]


def test_dataloader_shim_patches_the_classes_verl_imports(monkeypatch):
    """The patch must land on the shared class objects, not on a copy the shim made.

    A shim that rebound its own local name would execute cleanly and assert nothing -- verl
    constructs the sampler through its own import of the same class, so the identity of the patched
    attribute is the whole mechanism.
    """
    sampler, loader, _sampler_calls, _loader_calls = _exec_dataloader_shim(monkeypatch)

    from torch.utils.data.distributed import DistributedSampler
    from torchdata.stateful_dataloader import StatefulDataLoader

    assert DistributedSampler is sampler
    assert StatefulDataLoader is loader
    assert DistributedSampler.__init__.__name__ == "_flash_exact_sampler_init"
    assert StatefulDataLoader.__init__.__name__ == "_flash_exact_loader_init"


def test_shipped_shim_carries_the_exact_dataloader_patch(monkeypatch):
    """The rendered fragment is only worth testing if it reaches the file verl imports.

    ``run_sft_train`` concatenates it onto the sitecustomize source; asserting that here keeps the
    three tests above from passing against a fragment nothing writes out.
    """
    import pathlib

    import flash.engine.worker.sft_train as sft_train_module

    fragment = render_exact_sft_dataloader_shim()
    source = "".join(pathlib.Path(sft_train_module.__file__).read_text().split())
    assert "shim_source+=render_exact_sft_dataloader_shim()" in source
    assert "shuffle" in fragment
    assert "drop_last" in fragment


def test_reentrant_checkpointing_enables_input_grads_before_enabling_checkpointing():
    """GRAD-001: lora freezes the embeddings, so nothing entering the first checkpointed layer
    requires grad and reentrant checkpointing returns no gradient at all. the shim must call
    enable_input_require_grads() BEFORE gradient_checkpointing_enable(), or every sft run
    trains nothing while reporting done and billing."""
    calls = []

    class FakeModule:
        def enable_input_require_grads(self):
            calls.append("require_grads")

        def gradient_checkpointing_enable(self, **kwargs):
            calls.append(("gc_enable", kwargs))

    class FakeEngine:
        def _build_module(self):
            return FakeModule()

    source = _render_sft_sitecustomize(
        seed=1,
        loraplus_ratio=16,
        save_at_steps=(),
        total_steps=4,
        reentrant_gradient_checkpointing=True,
    )
    # execute only the reentrant block: the surrounding shim imports verl/torch at module scope.
    block = source[source.index("def _flash_build_reentrant_module") :]
    block = block[: block.index("_FlashFSDPEngine._build_module = _flash_build_reentrant_module")]
    namespace = {"_flash_original_build_module": FakeEngine._build_module}
    exec(compile(block, "shim.py", "exec"), namespace)

    namespace["_flash_build_reentrant_module"](FakeEngine())

    # order matters: enabling checkpointing first would capture the graph before any input
    # requires grad, so asserting mere presence would pass on a broken shim.
    assert calls[0] == "require_grads"
    assert calls[1] == ("gc_enable", {"gradient_checkpointing_kwargs": {"use_reentrant": True}})

    # the non-reentrant path never patches _build_module at all, so it must not appear.
    non_reentrant = _render_sft_sitecustomize(
        seed=1,
        loraplus_ratio=16,
        save_at_steps=(),
        total_steps=4,
        reentrant_gradient_checkpointing=False,
    )
    assert "enable_input_require_grads" not in non_reentrant


class _TolerantWatcher:
    """a watcher that permits an incomplete run, for tests where a guard is meant to raise.

    the default watcher in _stub_sft_run asserts require_complete is True, which is right for the
    happy path but masks the assertion under test here: a guard that raises from on_line unwinds
    before return_code is assigned, so the run is legitimately incomplete.
    """

    def __init__(self, **kwargs):
        self.processed_steps = set()

    def start(self):
        return None

    def raise_if_failed(self):
        return None

    def stop(self, *, require_complete):
        return None


@pytest.mark.parametrize(
    "lines",
    [
        # verbatim shape of the g4 gsm8k lines (flash-1785592071-e56cf3c6): loss barely moves on a
        # replayed identical batch because nothing is learning.
        pytest.param(
            [
                "step:1 - train/loss:0.5470 - train/grad_norm:0.0 - train/lr:0.0001",
                "step:2 - train/loss:0.5437 - train/grad_norm:0.0 - train/lr:0.0001",
                "step:3 - train/loss:0.5474 - train/grad_norm:0.0 - train/lr:0.0001",
                "step:4 - train/loss:0.5444 - train/grad_norm:0.0 - train/lr:0.0001",
            ],
            id="every-step-zero",
        ),
        # VERL-138: the same defect on a 2-step run, where the schedule decays lr to 0.0 on the
        # final step. the lr is not why the gradient is zero -- verl measures grad_norm off p.grad
        # before the optimizer and the scheduler run -- so this must fail exactly like the above.
        pytest.param(
            [
                "step:1 - train/loss:0.5464 - train/grad_norm:0.0 - train/lr:5e-05",
                "step:2 - train/loss:0.5477 - train/grad_norm:0.0 - train/lr:0.0",
            ],
            id="lr-decays-to-zero",
        ),
    ],
)
def test_zero_grad_norm_fails_the_run(monkeypatch, lines):
    """GRAD-001: four runs reported done and charged while grad_norm was 0.0 on every step.

    the number was parsed and recorded, never read. driven through run_sft_train so the assertion
    lands on the shipped guard -- this test used to define its own copy of the guard body and
    assert against that, which meant it could not fail no matter what the worker did.
    """
    from flash.engine.worker import sft_train

    spec, _ = _stub_sft_run(monkeypatch, watcher_cls=_TolerantWatcher)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        for step, line in enumerate(lines, start=1):
            on_line(line + "\n")
            on_step(step)
        raise AssertionError("the zero-grad guard should have stopped the run before this")

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    with pytest.raises(RuntimeError, match=re.escape("grad_norm=0.0")):
        sft_train.run_sft_train(spec)


@pytest.mark.parametrize(
    "lines",
    [
        pytest.param(
            [
                "step:1 - train/loss:0.9 - train/grad_norm:1.4 - train/lr:0.0001",
                "step:2 - train/loss:0.7 - train/grad_norm:0.9 - train/lr:0.0001",
            ],
            id="healthy",
        ),
        # an isolated zero is a legitimately fully-masked micro-batch, not a severed graph.
        pytest.param(
            [
                "step:1 - train/loss:0.9 - train/grad_norm:0.0 - train/lr:0.0001",
                "step:2 - train/loss:0.7 - train/grad_norm:1.1 - train/lr:0.0001",
                "step:3 - train/loss:0.6 - train/grad_norm:0.0 - train/lr:0.0001",
            ],
            id="isolated-zeros",
        ),
    ],
)
def test_healthy_grad_norms_do_not_trip_the_guard(monkeypatch, lines):
    """the guard must not fail a run that is training: any nonzero norm resets the count."""
    from flash.engine.worker import sft_train

    spec, _ = _stub_sft_run(monkeypatch)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        for step, line in enumerate(lines, start=1):
            on_line(line + "\n")
            on_step(step)
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)


def test_step_gate_admits_a_line_a_tqdm_bar_was_flushed_in_front_of():
    """VERL-134: the guard above never armed on the real run because on_line returned early.

    on_line gates every metric read on ``verl_step_number(line)``, and verl's LocalLogger shares its
    stream with tqdm, whose bar ends in "]" with no trailing newline. anchoring the left edge on
    whitespace matched step 1 and missed steps 2-4, so ``flash-1785598982-21827245`` reported done
    with train/grad_norm 0.0 on every step while the guard's counter sat at 1.

    the guard test above replays its own loop, so it cannot see this: the defect is in the gate the
    real on_line runs first, not in the counting.
    """
    # verbatim from the run log, tqdm prefix included.
    glued = (
        "Epoch 1/1:  25%|##        | 1/4 [01:21<04:04, 81.49s/it]"
        "step:2 - train/loss:1.0206047296524048 - train/grad_norm:0.0 - train/lr:5e-05"
    )

    assert verl_step_number(glued) == 2, "on_line would return before ever reading grad_norm"
    # and the metrics behind the gate are the ones the guard needs.
    assert parse_verl_metric(glued, "train/grad_norm") == 0.0
    assert parse_verl_metric(glued, "train/lr") == 5e-05


def test_loraplus_shim_has_no_plain_lora_fallback():
    source = render_loraplus_shim(16)
    assert _LORAPLUS_READY_MARKER in source
    assert "falling back" not in source
    assert "_flash_original_build_optimizer" not in source
    assert render_loraplus_shim(1) == ""


def test_child_environment_excludes_provider_and_control_plane_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("NCCL_DEBUG", "WARN")
    monkeypatch.setenv("HF_HOME", "/cache/hf")
    monkeypatch.setenv("FLASH_VERL_PYTHON", "/verl/python")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("PARASAIL_API_KEY", "teacher-secret")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "control-secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "provider-secret")
    monkeypatch.setenv("HF_TOKEN", "hub-secret")

    without_wandb = _build_verl_child_env(shim_dir=str(tmp_path), wandb_enabled=False)
    assert without_wandb["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert without_wandb["NCCL_DEBUG"] == "WARN"
    assert without_wandb["HF_HOME"] == "/cache/hf"
    assert without_wandb["FLASH_VERL_PYTHON"] == "/verl/python"
    assert "WANDB_API_KEY" not in without_wandb
    for secret in (
        "PARASAIL_API_KEY",
        "FREESOLO_INTERNAL_KEY",
        "RUNPOD_API_KEY",
        "HF_TOKEN",
    ):
        assert secret not in without_wandb

    with_wandb = _build_verl_child_env(shim_dir=str(tmp_path), wandb_enabled=True)
    assert with_wandb["WANDB_API_KEY"] == "wandb-secret"


def test_checkpoint_watcher_exports_and_uploads_required_step(monkeypatch, tmp_path):
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    checkpoint_dir = tmp_path / "checkpoints" / "global_step_5"
    actor_dir = checkpoint_dir / "actor"
    (actor_dir / "huggingface").mkdir(parents=True)
    exported = []
    published = []
    uploaded = []

    def fake_export(actor, adapter, **kwargs):
        exported.append((actor, adapter, kwargs))
        os.makedirs(adapter, exist_ok=True)

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", fake_export)
    monkeypatch.setattr(
        worker,
        "publish_deployable_checkpoint",
        lambda adapter, step, **kwargs: published.append((adapter, step, kwargs)),
    )

    def fake_upload(step, checkpoint, **kwargs):
        kwargs["before_upload"]()
        uploaded.append((step, checkpoint))
        return True

    monkeypatch.setattr(worker, "upload_resume_checkpoint", fake_upload)
    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(tmp_path / "checkpoints"),
        export_root=str(tmp_path / "exports"),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(5,),
    )

    watcher._publish(5, str(checkpoint_dir))

    # both read the STAGED copy rather than verl's directory (see _staged_source); what matters is
    # that they agree on one source and that it carries the `actor/` level this layout puts there.
    assert os.path.basename(exported[0][0]) == "actor"
    assert exported[0][0] == os.path.join(uploaded[0][1], "actor")
    assert published[0][1] == 5
    assert published[0][2]["required"] is True
    assert [step for step, _ in uploaded] == [5]
    assert watcher.processed_steps == {5}


def test_checkpoint_watcher_exports_the_sft_layout(monkeypatch, tmp_path):
    # this is the layout verl's sft trainer actually writes: shards + huggingface/ directly under
    # global_step_N. exporting <dir>/actor here hands the merger a path that does not exist.
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    checkpoint_dir = tmp_path / "checkpoints" / "global_step_5"
    (checkpoint_dir / "huggingface").mkdir(parents=True)
    exported = []

    def fake_export(actor, adapter, **kwargs):
        if not os.path.isdir(actor):
            raise AssertionError(f"exported a checkpoint dir that does not exist: {actor}")
        exported.append(actor)
        os.makedirs(adapter, exist_ok=True)

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", fake_export)
    monkeypatch.setattr(
        worker, "publish_deployable_checkpoint", lambda adapter, step, **kwargs: None
    )
    monkeypatch.setattr(
        worker,
        "upload_resume_checkpoint",
        lambda step, checkpoint, **kwargs: (kwargs["before_upload"](), True)[1],
    )
    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(tmp_path / "checkpoints"),
        export_root=str(tmp_path / "exports"),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(5,),
    )

    watcher._publish(5, str(checkpoint_dir))

    # the export reads a STAGED copy, not verl's own directory (see _staged_source), so assert the
    # layout this test is about -- global_step_N itself, no `actor/` level -- rather than the path.
    assert [os.path.basename(path) for path in exported] == ["global_step_5"]


def test_a_required_save_survives_verl_pruning_it_mid_publish(monkeypatch, tmp_path):
    """verl deleting global_step_N while it is being published must not fail the run.

    the watcher publishes on its own thread, so the merge and the multi-GB upload overlap training.
    under `trainer.max_ckpt_to_keep=1` verl's `register_checkpoint` rmtree's global_step_N from the
    CHILD process the moment N+1 finishes saving, and the parent has no way to pin it. with close
    `save_at_steps` (here [2, 3]) that lands squarely inside the publish window.

    this reproduces the interleave rather than asserting on the staging mechanism: the fake upload
    deletes verl's directory the way verl would, then reads the source it was handed. before staging,
    that read fails and takes the whole paid run down with a required-save error.
    """
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    local_dir = tmp_path / "checkpoints"
    checkpoint_dir = local_dir / "global_step_2"
    (checkpoint_dir / "actor" / "huggingface").mkdir(parents=True)
    (checkpoint_dir / "actor" / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(
        sft_train,
        "_export_checkpoint_adapter",
        lambda actor, adapter, **kwargs: os.makedirs(adapter, exist_ok=True),
    )
    monkeypatch.setattr(
        worker, "publish_deployable_checkpoint", lambda adapter, step, **kwargs: None
    )
    read_back = {}

    def fake_upload(step, checkpoint, **kwargs):
        kwargs["before_upload"]()
        # verl prunes global_step_2 because global_step_3 just landed. this is the whole race.
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        # the upload streams the folder AFTER that, so it must still have something to read.
        read_back["payload"] = (
            pathlib.Path(checkpoint, "actor", "model.safetensors").read_bytes()
            if os.path.isdir(checkpoint)
            else None
        )
        return True

    monkeypatch.setattr(worker, "upload_resume_checkpoint", fake_upload)
    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(local_dir),
        export_root=str(tmp_path / "exports"),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(2, 3),
    )

    watcher._publish(2, str(checkpoint_dir))

    assert read_back["payload"] == b"weights", (
        "the upload lost its source when verl pruned the checkpoint, so a required save would fail "
        "an otherwise successful paid run"
    )
    assert watcher.processed_steps == {2}
    # the staging links are transient: they must not outlive the publish and accumulate on the pod.
    assert not os.path.exists(os.path.join(str(tmp_path / "exports"), "_staging", "global_step_2"))


def test_resume_credits_only_required_saves_that_are_durable(monkeypatch):
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    class Api:
        def file_exists(self, *, filename, **kwargs):
            return "/step-3/" in filename

    monkeypatch.setattr(worker, "HF_REPO", "owner/artifacts")
    monkeypatch.setattr(worker, "hf_prefix", lambda: "sft/run")
    monkeypatch.setattr(worker, "hf_api", Api)

    assert sft_train._durable_required_save_steps((3, 5, 9), 5) == {3}
    # a resume step that is itself a required save is credited only when its adapter is on hf, so
    # step 5 stays publishable while the already-durable step 3 does not.
    assert sft_train._processed_resume_steps((3, 5, 9), 5) == {3}
    # a resume step that is not a required save is always credited: hf already holds its state.
    assert sft_train._processed_resume_steps((3, 9), 5) == {3, 5}


def test_a_resumed_sft_run_does_not_republish_the_step_it_resumed_from(monkeypatch, tmp_path):
    """the staged resume checkpoint is a pending ``global_step_N``; publishing it again is waste.

    ``stage_verl_resume`` copies the downloaded ``checkpoint-N`` into local_dir and points verl's
    tracker at it, so an unseeded watcher treats it as new work on its first sweep: it re-runs
    ``verl.model_merger`` and re-uploads multi-GB state hf already holds, holding the single
    resume-upload lock while the first genuinely new checkpoint waits behind it.
    """
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train
    from flash.engine.worker.backend_common import stage_verl_resume

    local_dir = tmp_path / "checkpoints"
    local_dir.mkdir()
    resume_dir = tmp_path / "downloaded" / "checkpoint-1"
    (resume_dir / "huggingface").mkdir(parents=True)
    # verl stamps every checkpoint with its writer's world size; staging demands a match.
    (resume_dir / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 1}))
    resume_step = stage_verl_resume(str(resume_dir), str(local_dir), job_label="SFT", world_size=1)
    # a checkpoint this attempt actually trained, which must still be exported and uploaded.
    (local_dir / "global_step_2" / "huggingface").mkdir(parents=True)
    (local_dir / "latest_checkpointed_iteration.txt").write_text("2")

    exported = []
    published = []
    uploaded = []

    def fake_export(actor, adapter, **kwargs):
        exported.append(actor)
        os.makedirs(adapter, exist_ok=True)

    def fake_upload(step, checkpoint, **kwargs):
        kwargs["before_upload"]()
        uploaded.append(step)
        return True

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", fake_export)
    monkeypatch.setattr(
        worker,
        "publish_deployable_checkpoint",
        lambda adapter, step, **kwargs: published.append(step),
    )
    monkeypatch.setattr(worker, "upload_resume_checkpoint", fake_upload)
    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(local_dir),
        export_root=str(tmp_path / "exports"),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(),
    )
    watcher.processed_steps.update(sft_train._processed_resume_steps((), resume_step))

    watcher.start()
    watcher.stop(require_complete=True)

    assert resume_step == 1
    assert [os.path.basename(path) for path in exported] == ["global_step_2"]
    assert published == [2]
    assert uploaded == [2]
    assert watcher.processed_steps == {1, 2}


def _stub_sft_run(
    monkeypatch,
    *,
    save_at_steps=(),
    watcher_cls=None,
    structured_targets=False,
    structured_singleturn=False,
    raw_output_fallback=False,
    missing_output=False,
):
    """monkeypatch every out-of-process dependency of run_sft_train and return (spec, captured).

    the caller supplies its own ``run_verl_training`` fake, which is the only remaining seam.
    """
    import flash.core.catalog as catalog
    import flash.engine.plan.vram as vram
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    monkeypatch.setattr(catalog, "resolve_vocab_size", lambda *_args, **_kwargs: 151936)

    spec = SimpleNamespace(
        model="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        algorithm="sft",
        seed=7,
        thinking=False,
        workload_profile_input_digest="",
        workload_profile_producer_version=_PROFILE_PRODUCER_VERSION,
        workload_profile={},
        environment=SimpleNamespace(
            id="owner/env",
            resolved_sha="b" * 40,
            params={},
            pip=(),
        ),
        gpu=SimpleNamespace(type="RTX 4090", exact_type="", count=2),
        train=SimpleNamespace(
            epochs=1,
            learning_rate=5e-5,
            batch_size=2,
            max_context_tokens=1024,
            max_examples=2,
            max_steps=2,
            save_at_steps=save_at_steps,
            save_every=50,
            init_from_adapter="",
        ),
        wandb=SimpleNamespace(project=None, run_name=None),
    )

    class Env:
        id = "owner/env"
        package_root = None
        multi_turn = False

        def dataset(self):
            if structured_targets:
                trajectory = [
                    {"role": "assistant", "content": "<think>reason</think>"},
                    {"role": "assistant", "content": "answer"},
                ]
                return [
                    {"prompt": "one", "output": trajectory},
                    {"prompt": "two", "output": trajectory},
                ]
            if structured_singleturn:
                # one assistant message, but explicitly structured -- NOT a scalar coercion.
                single = [{"role": "assistant", "content": "answer"}]
                return [
                    {"prompt": "one", "output": single},
                    {"prompt": "two", "output": {"messages": single}},
                ]
            rows = [
                {"prompt": "one", "output": "answer"},
                {"prompt": "two", "output": "answer"},
            ]
            if missing_output:
                rows[1].pop("output")
            return rows

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["prompt"]}]

        def sft_completion(self, example):
            output = example.get("output")
            if isinstance(output, list):
                return output
            return [{"role": "assistant", "content": "hook answer" if output is None else output}]

    class RawOutputFallbackEnv(Env):
        def sft_completion_with_provenance(self, example):
            output = example.get("output")
            return [{"role": "assistant", "content": "" if output is None else str(output)}], True

        def sft_completion(self, example):
            messages, _coerced_scalar_output = self.sft_completion_with_provenance(example)
            return messages

    class StructuredSingleTurnEnv(Env):
        """One assistant turn per row, reported as structured (not coerced).

        this is what the real adapter now returns for `output: [{...}]` and `{"messages": [...]}`:
        a single-message target whose provenance says no scalar coercion happened. it is the case
        the collapse warning must stay quiet for -- the row IS one assistant turn, so only the
        provenance flag separates it from a bare stringified answer.
        """

        def sft_completion_with_provenance(self, example):
            return [{"role": "assistant", "content": "answer"}], False

        def sft_completion(self, example):
            messages, _coerced_scalar_output = self.sft_completion_with_provenance(example)
            return messages

    EnvClass = Env
    if raw_output_fallback:
        EnvClass = RawOutputFallbackEnv
    elif structured_singleturn:
        EnvClass = StructuredSingleTurnEnv

    class Tokenizer(_ExactTokenizer):
        pad_token = None

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize,
            add_generation_prompt,
            enable_thinking,
        ):
            assert tokenize is False
            rendered = "".join(
                f"<{message['role']}>{message['content']}</{message['role']}>"
                for message in messages
            )
            if add_generation_prompt:
                rendered += "<assistant>"
            return rendered

    from flash.engine.profiling.sft_workload import prepare_sft_workload
    from flash.engine.profiling.workload_profile import sft_profile_input_digest

    # deliberately NOT `flash.__version__`: the worker has no flash distribution installed and
    # resolves that to "0+unknown", so building both sides from it would make this fixture agree
    # with a worker that re-derives the version -- the defect it must instead be able to catch.
    spec.workload_profile_input_digest = sft_profile_input_digest(
        spec,
        tokenizer_revision=spec.model_revision,
        producer_version=_PROFILE_PRODUCER_VERSION,
    )
    spec.workload_profile = prepare_sft_workload(
        spec,
        EnvClass(),
        tokenizer_loader=lambda _model, _revision: Tokenizer(),
        producer_version=_PROFILE_PRODUCER_VERSION,
        allow_packing=False,
        packing_support=lambda _model, _revision: ("unsupported", False),
    ).profile.to_dict()
    # the fixture pins the architecture above, but the WORKER re-derives it through the live probes,
    # which read the model config off the hub. pin them to the same answer so the parity check under
    # test compares workloads rather than network reachability -- an unresolvable probe now fails
    # closed instead of quietly labelling the model "unsupported".
    from flash.engine.profiling import sft_workload as _sft_workload

    monkeypatch.setattr(_sft_workload, "probe_is_pure_attention", lambda _m, revision="": False)
    monkeypatch.setattr(_sft_workload, "probe_is_gdn_hybrid", lambda _m, revision="": False)

    class LoraConfig:
        r = 16
        lora_alpha = 32
        target_modules = "all-linear"

    class PeakSampler:
        def start(self):
            return self

        def stop_gb(self):
            return 12.5

    class _DefaultWatcher:
        def __init__(self, **kwargs):
            self.required_steps = frozenset(kwargs["required_steps"])
            self.processed_steps = set()

        def start(self):
            return None

        def stop(self, *, require_complete):
            assert require_complete is True

        def raise_if_failed(self):
            return None

    Watcher = watcher_cls or _DefaultWatcher

    captured = {"heartbeats": [], "published": [], "uploads": []}
    real_sft_grad_accum = vram.sft_grad_accum

    def capture_sft_grad_accum(batch_size, **kwargs):
        captured["sft_grad_accum"] = {"batch_size": batch_size, **kwargs}
        return real_sft_grad_accum(batch_size, **kwargs)

    def capture_grad_checkpointing(model_id, max_length, **kwargs):
        captured["grad_checkpointing"] = {
            "model_id": model_id,
            "max_length": max_length,
            **kwargs,
        }
        return True

    monkeypatch.setattr(vram, "sft_grad_accum", capture_sft_grad_accum)
    # run_sft_train imports AutoProcessor at data-loading time and transformers is not installed in
    # the cpu test env. this used to pass only because some EARLIER test module left a transformers
    # stub in sys.modules, so running this file alone failed -- stub it here so the test stands on
    # its own. monkeypatch.setitem restores whatever was there (real module or nothing) afterwards.
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        sys.modules.get("transformers")
        or _module(
            "transformers",
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: None),
            # datasets' dill serializer issubclass()-checks against this while writing the
            # parquet, so it has to be a real class rather than a namespace attribute.
            PreTrainedTokenizerBase=type("PreTrainedTokenizerBase", (), {}),
        ),
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(worker, "SEED", 7)
    monkeypatch.setattr(worker, "RUN_ID", "test-sft-verl-orchestration")
    monkeypatch.setattr(worker, "THINKING", False)
    monkeypatch.setattr(worker, "JOB_SPEC", spec)
    monkeypatch.setattr(worker, "require_active_env", EnvClass)
    monkeypatch.setattr(
        worker,
        "heartbeat",
        lambda stage, **fields: captured["heartbeats"].append((stage, fields)),
    )
    monkeypatch.setattr(worker, "gpu_diagnostics", lambda **kwargs: {})
    monkeypatch.setattr(worker, "prefetch_model", lambda *args, **kwargs: 1.25)
    monkeypatch.setattr(worker, "load_tokenizer", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(worker, "make_lora", lambda model_id: LoraConfig())
    monkeypatch.setattr(worker, "grad_checkpointing_on", capture_grad_checkpointing)
    monkeypatch.setattr(worker, "grpo_use_reentrant", lambda model_id: False)
    monkeypatch.setattr(worker, "backend_seed", lambda seed: seed)
    monkeypatch.setattr(worker, "wandb_run_name", lambda: "flash-sft-test")
    monkeypatch.setattr(
        worker,
        "hf_upload_folder",
        lambda local, remote, required=False: captured["uploads"].append((local, remote, required)),
    )
    monkeypatch.setattr(
        worker,
        "publish_deployable_checkpoint",
        lambda adapter, step, **kwargs: captured["published"].append((adapter, step)),
    )
    monkeypatch.setattr(
        worker,
        "write_train_meta",
        lambda **kwargs: captured.__setitem__("meta", kwargs),
    )
    monkeypatch.setattr(
        sft_train,
        "liveness_heartbeat",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        sft_train,
        "_probe_gpu_in_subprocess",
        lambda *args, **kwargs: {"memory_gb": 24, "capability": [8, 9]},
    )
    monkeypatch.setattr(sft_train, "_model_arch_dims", lambda *args, **kwargs: (1024, 24))
    monkeypatch.setattr(sft_train, "resolve_verl_python", lambda *a, **k: "/venv/bin/python")
    monkeypatch.setattr(sft_train, "resolve_verl_loggers", lambda python_bin: ["console"])
    # torch is not installed in this test env; the real seeding is covered in test_training_controls.
    monkeypatch.setattr(sft_train, "seed_training_rngs", lambda seed: None)
    monkeypatch.setattr(sft_train, "_cached_model_path", lambda model, revision: model)
    monkeypatch.setattr(sft_train, "_restore_verl_resume", lambda local_dir, **_kwargs: 1)
    monkeypatch.setattr(sft_train, "_VerlCheckpointWatcher", Watcher)
    monkeypatch.setattr(sft_train, "_NvidiaSmiPeakSampler", PeakSampler)
    monkeypatch.setattr(
        sft_train,
        "latest_global_step_dir",
        lambda local_dir: (os.path.join(local_dir, "global_step_2"), 2),
    )

    def fake_export(actor_dir, adapter_dir, **kwargs):
        os.makedirs(adapter_dir, exist_ok=True)
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as file:
            file.write("{}")
        with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "wb") as file:
            file.write(b"adapter")

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", fake_export)

    # the fake run_verl_training each test supplies never executes the rendered sitecustomize, so
    # stand in for the child's marker writes: record every expected fragment as applied. the real
    # verifier still runs against this file, so a test that wants the missing-marker failure
    # restores the real _prepare_sft_child.
    real_prepare_child = sft_train._prepare_sft_child

    def prepare_child_with_applied_shims(*args, **kwargs):
        child = real_prepare_child(*args, **kwargs)
        with open(child.shim_markers, "w", encoding="utf-8") as handle:
            handle.write("".join(name + "\n" for name in child.expected_shims))
        captured["child"] = child
        return child

    monkeypatch.setattr(sft_train, "_prepare_sft_child", prepare_child_with_applied_shims)

    return spec, captured


def test_sft_warns_when_every_selected_row_is_a_coerced_singleturn_target(monkeypatch, capsys):
    from flash.engine.worker import sft_train, sft_train_runner

    spec, _captured = _stub_sft_run(monkeypatch, raw_output_fallback=True)
    monkeypatch.setattr(sft_train, "_write_sft_parquet", lambda _rows, _path: None)

    options = sft_train_runner._resolve_sft_options(spec)
    sft_train_runner._prepare_sft_data(options)

    output = capsys.readouterr().out
    assert "[sft][warn] all 2 selected rows use one bare assistant target coerced" in output
    assert "reasoning blocks or multi-turn structure may have been lost" in output


def test_sft_collapse_warning_stays_quiet_when_environment_hook_handles_raw_rows(
    monkeypatch, capsys
):
    from flash.engine.worker import sft_train, sft_train_runner

    spec, _captured = _stub_sft_run(monkeypatch, missing_output=True)
    monkeypatch.setattr(sft_train, "_write_sft_parquet", lambda _rows, _path: None)

    options = sft_train_runner._resolve_sft_options(spec)
    sft_train_runner._prepare_sft_data(options)

    output = capsys.readouterr().out
    assert "bare assistant target coerced" not in output


def test_sft_collapse_warning_stays_quiet_for_structured_multiturn_targets(monkeypatch, capsys):
    from flash.engine.worker import sft_train, sft_train_runner

    spec, _captured = _stub_sft_run(monkeypatch, structured_targets=True)
    monkeypatch.setattr(sft_train, "_write_sft_parquet", lambda _rows, _path: None)

    options = sft_train_runner._resolve_sft_options(spec)
    sft_train_runner._prepare_sft_data(options)

    output = capsys.readouterr().out
    assert "[sft] multi-turn SFT: 2/2 rows" in output
    assert "bare assistant target coerced" not in output


def test_sft_collapse_warning_stays_quiet_for_structured_singleturn_targets(monkeypatch, capsys):
    """A structured target is not a coercion even when it is a SINGLE assistant turn.

    the multi-turn case above is separated by length alone, so it cannot catch a provenance flag
    that marks parsed message lists as coerced. these rows are one assistant turn each, exactly
    like a stringified scalar, so only provenance keeps the warning quiet -- and firing here would
    tell users to encode message lists they have already encoded.
    """
    from flash.engine.worker import sft_train, sft_train_runner

    spec, _captured = _stub_sft_run(monkeypatch, structured_singleturn=True)
    monkeypatch.setattr(sft_train, "_write_sft_parquet", lambda _rows, _path: None)

    options = sft_train_runner._resolve_sft_options(spec)
    sft_train_runner._prepare_sft_data(options)

    output = capsys.readouterr().out
    assert "bare assistant target coerced" not in output


def test_run_sft_train_orchestrates_exact_dataset_and_resume_accounting(monkeypatch):
    from flash.engine.worker import sft_train

    spec, captured = _stub_sft_run(monkeypatch)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        captured["command"] = command
        captured["child_env"] = env
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        on_line("step:2 - train/loss:1.0 - train/global_tokens:8\n")
        on_step(2)
        heartbeat()
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)

    assert captured["command"][:3] == ["/venv/bin/python", "-m", "torch.distributed.run"]
    # ONE rank, on a 2-card spec: this fixture's probes answer "not gdn, not pure attention", so
    # the profile is exact-unpacked and `examples_per_update` is 1. verl would floor-divide that
    # single example across 2 dp ranks and hand a DataLoader batch_size=0, so the second card is
    # unusable for this run and must not be launched. See sft_data_parallel_cards.
    assert "--nproc-per-node=1" in captured["command"]
    assert "trainer.n_gpus_per_node=1" in captured["command"]
    assert "engine.ulysses_sequence_parallel_size=1" in captured["command"]
    assert "verl.trainer.sft_trainer" in captured["command"]
    custom_path = next(
        value.split("=", 1)[1]
        for value in captured["command"]
        if value.startswith("data.custom_cls.path=")
    )
    assert os.path.isfile(custom_path)
    assert captured["child_env"]["PYTHONPATH"].split(os.pathsep)[0].endswith("/shim")
    assert captured["child_env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["uploads"][0][1:] == ("adapter", True)
    assert captured["published"][0][1] == 2
    assert captured["meta"]["step"] == 2
    assert captured["meta"]["train_tokens"] > 0
    assert captured["meta"]["notes"]["loss_curve"] == [1.0]
    assert captured["meta"]["notes"]["loraplus_applied"] is True
    notes = captured["meta"]["notes"]
    realized_max_length = notes["realized_max_length"]
    assert 0 < realized_max_length < notes["configured_max_length"]
    assert notes["runtime_max_length"] == realized_max_length
    assert captured["sft_grad_accum"]["seq_len"] == realized_max_length
    assert captured["sft_grad_accum"]["fused"] is True
    assert captured["grad_checkpointing"]["max_length"] == realized_max_length
    assert captured["grad_checkpointing"]["fused_ce"] is True
    assert "data.max_length=1024" in captured["command"]
    assert (
        f"data.max_token_len_per_gpu={realized_max_length * notes['per_device_train_batch_size']}"
        in captured["command"]
    )


def test_the_sft_runner_seeds_the_watcher_with_the_step_it_resumed_from(monkeypatch):
    """the seed has to happen in the runner, before the watcher's thread takes its first sweep."""
    from flash.engine.worker import sft_train

    seeded = {}

    class Watcher:
        def __init__(self, **kwargs):
            self.required_steps = frozenset(kwargs["required_steps"])
            self.processed_steps = set()

        def start(self):
            seeded["at_start"] = set(self.processed_steps)

        def stop(self, *, require_complete):
            assert require_complete is True

        def raise_if_failed(self):
            return None

    spec, captured = _stub_sft_run(monkeypatch, watcher_cls=Watcher)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        on_line("step:2 - train/loss:1.0 - train/global_tokens:8\n")
        on_step(2)
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)

    # the stubbed resume lands at step 1 with no exact save steps: unseeded, the watcher would find
    # the staged global_step_1 pending and re-merge and re-upload the checkpoint it just downloaded.
    assert seeded["at_start"] == {1}
    # the new step is still published, so the seed cannot be a blanket skip of everything.
    assert captured["published"][0][1] == 2


def test_a_resume_at_the_horizon_still_publishes_the_final_deployable(monkeypatch):
    """the seeded resume step must not suppress the final publish.

    the previous attempt's per-step deployable publish is best-effort (``required=False``) while
    its resume upload is not, so hf can hold the resumable state without the servable adapter.
    """
    from flash.engine.worker import sft_train

    spec, captured = _stub_sft_run(monkeypatch)
    # max_steps is 2, so resuming at 2 means the watcher never runs and finalization is the only
    # path left that can publish the step.
    monkeypatch.setattr(sft_train, "_restore_verl_resume", lambda local_dir, *, world_size: 2)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        raise AssertionError("a run resumed at its horizon must not start the child")

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)

    assert [step for _adapter, step in captured["published"]] == [2]


def test_worker_uses_the_accepted_unpacked_quote_when_its_stack_can_pack(monkeypatch):
    """worker capability must not replace the packing contract the user accepted.

    the control plane lacks the gdn packing stack and freezes an exact-unpacked quote, while the gpu
    worker has that stack and would independently choose packed execution. rows still come from the
    worker recomputation, but the child batch and update horizon must remain the quoted shape.
    """
    from flash.engine.profiling import sft_workload
    from flash.engine.worker import sft_train, sft_train_runner

    spec, _captured = _stub_sft_run(monkeypatch)
    monkeypatch.setattr(sft_workload, "probe_is_gdn_hybrid", lambda _m, revision="": True)
    monkeypatch.setattr(
        sft_workload, "gdn_packing_contract_available", lambda _m, revision="": True
    )
    monkeypatch.setattr(sft_train, "_write_sft_parquet", lambda _rows, _path: None)
    recomputed_profiles = []
    prepare = sft_train.prepare_sft_workload

    def capture_recomputed_profile(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        recomputed_profiles.append(prepared.profile)
        return prepared

    monkeypatch.setattr(sft_train, "prepare_sft_workload", capture_recomputed_profile)

    options = sft_train_runner._resolve_sft_options(spec)
    data = sft_train_runner._prepare_sft_data(options)
    model = sft_train_runner._prepare_sft_model(options, data)

    assert recomputed_profiles[0].packing_mode == "packed"
    assert recomputed_profiles[0].examples_per_update == 2
    assert data.profile.packing_mode == "exact-unpacked"
    assert data.profile.examples_per_update == 1
    assert model.train_batch_size == 1
    assert model.update_horizon == data.profile.authoritative_steps


def test_the_child_caps_at_the_quoted_horizon_without_an_authored_max_steps(monkeypatch):
    """the accepted step count binds even when the user never authored max_steps.

    the plane profiles raw records without running environment.py, so an environment that expands
    the rows makes the realized epoch longer than the quote assumed. verl stops at
    total_training_steps, so leaving it unset would run past the horizon the run was priced for.
    """
    from flash.engine.worker import sft_train, sft_train_runner

    spec, _captured = _stub_sft_run(monkeypatch)
    # the shared fixture authors max_steps; this test is about the path where the user did not.
    spec.train.max_steps = 0
    monkeypatch.setattr(sft_train, "_write_sft_parquet", lambda _rows, _path: None)

    options = sft_train_runner._resolve_sft_options(spec)
    assert options.max_steps <= 0

    data = sft_train_runner._prepare_sft_data(options)
    model = sft_train_runner._prepare_sft_model(options, data)
    capabilities = sft_train_runner._SftCapabilities(
        python_bin="/venv/bin/python", caps={}, gdn_hybrid=False, gdn_module=""
    )
    child = sft_train_runner._prepare_sft_child(options, data, model, capabilities, True, None)

    # the horizon reaches verl as a rendered hydra override, so assert on what the child is
    # actually launched with rather than an intermediate dict. before this cap it rendered as
    # null whenever the user left max_steps unauthored, leaving the realized epoch as the only
    # bound; verl stops at whichever of the two limits it reaches first.
    horizon = data.profile.authoritative_steps
    assert f"trainer.total_training_steps={horizon}" in child.command
    assert "trainer.total_training_steps=null" not in child.command


def test_a_packed_quote_fails_closed_when_environment_filtering_leaves_less_than_one_batch(
    monkeypatch,
):
    """the worker must not silently shrink the accepted batch and change the billed contract."""
    from flash.engine.profiling.workload_profile import SftWorkloadProfile
    from flash.engine.worker import sft_train, sft_train_runner

    spec, _captured = _stub_sft_run(monkeypatch)
    quoted = SftWorkloadProfile.from_dict(spec.workload_profile)
    spec.workload_profile = replace(
        quoted,
        packing_mode="packed",
        architecture_mode="pure-attention",
        examples_per_update=2,
        packed_blocks=1,
    ).to_dict()
    monkeypatch.setattr(sft_train, "_write_sft_parquet", lambda _rows, _path: None)
    from flash.engine.profiling import sft_workload

    monkeypatch.setattr(sft_workload, "probe_is_pure_attention", lambda _m, revision="": True)
    prepared = sft_train.prepare_sft_workload

    def retain_one(*args, **kwargs):
        workload = prepared(*args, **kwargs)
        return replace(workload, rows=workload.rows[:1])

    monkeypatch.setattr(sft_train, "prepare_sft_workload", retain_one)

    options = sft_train_runner._resolve_sft_options(spec)
    with pytest.raises(
        RuntimeError, match="more examples per update than the environment retained"
    ):
        sft_train_runner._prepare_sft_data(options)


def test_a_packed_quote_fails_closed_when_the_worker_cannot_pack_safely(monkeypatch):
    """a worker without boundary resets must never execute a packed accepted quote."""
    from flash.engine.profiling.workload_profile import SftWorkloadProfile
    from flash.engine.worker import sft_train, sft_train_runner

    spec, _captured = _stub_sft_run(monkeypatch)
    quoted = SftWorkloadProfile.from_dict(spec.workload_profile)
    spec.workload_profile = replace(
        quoted,
        packing_mode="packed",
        architecture_mode="gdn-hybrid",
        examples_per_update=2,
        packed_blocks=1,
    ).to_dict()
    monkeypatch.setattr(sft_train, "_write_sft_parquet", lambda _rows, _path: None)

    options = sft_train_runner._resolve_sft_options(spec)
    with pytest.raises(RuntimeError, match="cannot reproduce its boundary-safe packing contract"):
        sft_train_runner._prepare_sft_data(options)


def test_environment_processing_may_change_the_static_estimate_without_repricing(
    monkeypatch, capsys
):
    """the worker reports estimate drift and trains the environment-produced rows.

    the control plane tokenizes only packaged input and output fields, while the worker executes
    environment prompt construction and filtering. those profiles are not expected to match. the
    accepted quote remains on the spec, and the worker uses its recomputed rows for training.
    """
    from flash.engine.profiling import sft_workload
    from flash.engine.worker import sft_train

    spec, _captured = _stub_sft_run(monkeypatch)
    frozen_quote = dict(spec.workload_profile)
    honest = sft_workload.prepare_sft_workload
    training_calls = []

    def drifted(*args, **kwargs):
        prepared = honest(*args, **kwargs)
        moved = replace(
            prepared.profile, realized_max_length=prepared.profile.realized_max_length - 1
        )
        return replace(prepared, profile=moved)

    def completed_training(command, *, env, on_step, on_line, heartbeat):
        training_calls.append(command)
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        on_line("step:2 - train/loss:1.0 - train/global_tokens:8\n")
        on_step(2)
        heartbeat()
        return 0

    monkeypatch.setattr(sft_train, "prepare_sft_workload", drifted)
    monkeypatch.setattr(sft_train, "run_verl_training", completed_training)

    sft_train.run_sft_train(spec)

    assert training_calls
    assert spec.workload_profile == frozen_quote
    assert (
        "environment processing changed the packaged-dataset token estimate"
        in capsys.readouterr().out
    )


def test_a_guard_failure_is_not_replaced_by_the_watcher_completeness_error(monkeypatch):
    """the zero-grad diagnosis must survive the finally block, not be overwritten by it.

    an on_line guard raises from INSIDE run_verl_training, so return_code is never assigned and
    keeps its initial 0. deriving require_complete from it would then demand every save_at_steps
    entry from a run that died at step 2, and the watcher's "required saves were not durably
    published" would unwind out of the finally in place of the real cause -- turning the one
    error GRAD-001 exists to surface into a checkpointing red herring.
    """
    from flash.engine.worker import sft_train

    stopped: list[bool] = []

    class Watcher:
        def __init__(self, **kwargs):
            self.processed_steps = set()
            self.required_steps = frozenset(kwargs.get("required_steps", ()))

        def start(self):
            return None

        def raise_if_failed(self):
            return None

        def stop(self, *, require_complete):
            stopped.append(require_complete)
            if require_complete:
                missing = sorted(self.required_steps - self.processed_steps)
                if missing:
                    raise RuntimeError(f"required saves were not durably published: {missing}")

    # a required save the run never durably publishes: the guard raises on the same step, before
    # the watcher has processed it. (the step is inside the 2-update horizon because
    # validate_save_steps rejects anything beyond it at config time.)
    spec, _ = _stub_sft_run(monkeypatch, save_at_steps=(2,), watcher_cls=Watcher)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        for step in range(1, _MAX_ZERO_GRAD_STEPS + 1):
            on_line(
                f"step:{step} - train/loss:1.0 - train/grad_norm:0.0 - train/lr:5e-05 "
                "- train/global_tokens:8\n"
            )
        raise AssertionError("the zero-grad guard should have stopped the run before this")

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    with pytest.raises(RuntimeError, match=re.escape("grad_norm=0.0")):
        sft_train.run_sft_train(spec)

    # the watcher still gets stopped -- it just is not asked to prove completeness for a run that
    # never finished. without this, stop() raises and the grad_norm error never reaches the caller.
    assert stopped == [False]


def test_zero_grad_guard_survives_an_lr_that_decays_to_zero(monkeypatch):
    """VERL-138: a decayed lr must not launder a run that trained nothing.

    replays the real 2-step shape of flash-1785606382-389d4630, which reported done and billed with
    grad_norm 0.0 on every step. the scheduler puts lr at 0.0 on the final step, so a guard that
    treats an lr of 0.0 as an excuse for a zero gradient never fires on the second step and the run
    bills for an adapter that learned nothing.

    the lr cannot cause this: verl computes grad_norm in optimizer_step (transformer_impl.py:683)
    by clipping over p.grad, before optimizer.step() and before lr_scheduler_step() advances the
    schedule. driven through run_sft_train rather than a local copy of the guard, so the assertion
    is about the shipped code and not about the test's own reimplementation of it.
    """
    from flash.engine.worker import sft_train

    spec, _ = _stub_sft_run(monkeypatch, watcher_cls=_TolerantWatcher)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        on_line(
            "step:1 - train/loss:0.5464 - train/grad_norm:0.0 - train/lr:5e-05 "
            "- train/global_tokens:6588\n"
        )
        on_line(
            "step:2 - train/loss:0.5477 - train/grad_norm:0.0 - train/lr:0.0 "
            "- train/global_tokens:6274\n"
        )
        raise AssertionError("the zero-grad guard should have stopped the run before this")

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    with pytest.raises(RuntimeError, match=re.escape("grad_norm=0.0")):
        sft_train.run_sft_train(spec)


def test_zero_grad_guard_clears_on_a_recovered_step(monkeypatch):
    """the guard must count consecutive steps, not keep a run-lifetime tally.

    a nonzero grad norm is proof the graph is intact, so evidence collected before it is stale and
    must be discarded. without this, one isolated zero-grad step early plus another much later
    would fail a run that is training normally in between.
    """
    from flash.engine.worker import sft_train

    spec, captured = _stub_sft_run(monkeypatch)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        # zero, healthy (clears), zero, healthy (clears). two zero-grad steps in total but never
        # two in a row, so the run must survive.
        for step, grad in enumerate([0.0, 1.4, 0.0, 0.9], start=1):
            on_line(
                f"step:{step} - train/loss:1.0 - train/grad_norm:{grad} - train/lr:5e-05 "
                "- train/global_tokens:8\n"
            )
            on_step(step)
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)
    assert captured["meta"]["notes"]["loss_curve"] == [1.0, 1.0, 1.0, 1.0]


def test_a_single_step_run_with_no_gradient_is_rejected(monkeypatch):
    """Regression (sft_train.py): the consecutive-run guard needs _MAX_ZERO_GRAD_STEPS
    steps to fire, so a horizon SHORTER than that could not trip it at all.

    a one-update horizon is ordinary (the retained rows fit a single batch). such a run appended its
    single grad_norm 0.0, never reached the threshold, and then published and billed an adapter
    identical to the base weights -- exactly the GRAD-001 outcome the guard exists to prevent, on the
    one horizon it could not see. driven through run_sft_train so the assertion is about the shipped
    code path, not a local reimplementation of the check.
    """
    from flash.engine.worker import sft_train

    spec, _ = _stub_sft_run(monkeypatch, watcher_cls=_TolerantWatcher)
    # a fresh run, not a resume: the guard abstains on a resume because the restored weights carry
    # earlier updates this session never observed.
    monkeypatch.setattr(sft_train, "_restore_verl_resume", lambda local_dir, **_kwargs: 0)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        # both updates report a dead gradient. the in-loop guard would also fire on two consecutive
        # zeros, so the horizon is cut to one below to isolate the end-of-run check.
        on_line(
            "step:1 - train/loss:0.5464 - train/grad_norm:0.0 - train/lr:5e-05 "
            "- train/global_tokens:6588\n"
        )
        on_step(1)
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    assert _MAX_ZERO_GRAD_STEPS > 1, "this regression only exists while the in-loop guard needs >1"
    with pytest.raises(RuntimeError, match=re.escape("grad_norm=0.0")):
        sft_train.run_sft_train(spec)


@pytest.mark.parametrize("grads", [[1.4], [0.0, 1.4], [1.4, 0.0]])
def test_a_fresh_run_with_any_real_gradient_still_completes(monkeypatch, grads):
    """The end-of-run check must reject only an ALL-zero session, never one that trained.

    pairs with the test above, and every case here is a FRESH run, so the guard is actually reached
    -- on a resume it abstains and the test could not fail however the check is spelled. `[1.4]`
    covers the short healthy horizon; the mixed cases pin the boundary, because the obvious
    over-broad spelling (`not all`, i.e. reject on any zero at all) fails a run that demonstrably
    trained. an isolated zero inside a longer run stays tolerated by contract.
    """
    from flash.engine.worker import sft_train

    spec, captured = _stub_sft_run(monkeypatch)
    monkeypatch.setattr(sft_train, "_restore_verl_resume", lambda local_dir, **_kwargs: 0)

    def fake_training(command, *, env, on_step, on_line, heartbeat):
        on_line(f"{_LORAPLUS_READY_MARKER} ratio=16 optimizer=AdamW\n")
        for step, grad in enumerate(grads, start=1):
            on_line(
                f"step:{step} - train/loss:1.0 - train/grad_norm:{grad} - train/lr:5e-05 "
                "- train/global_tokens:8\n"
            )
            on_step(step)
        return 0

    monkeypatch.setattr(sft_train, "run_verl_training", fake_training)

    sft_train.run_sft_train(spec)
    assert captured["meta"]["notes"]["loss_curve"] == [1.0] * len(grads)


def test_overrides_enable_fused_linear_ce_for_long_context():
    # 32k contexts must not materialize [tokens, vocab] logits; the fused torch-backend
    # linear-CE computes loss from hidden states in chunks (numerically exact CE).
    o = build_sft_overrides(_cfg(max_length=32768))
    assert "model.use_fused_kernels=true" in o
    assert "model.fused_kernel_options.impl_backend=torch" in o
    assert "data.max_length=32768" in o


def test_sft_line_handler_reads_metrics_through_the_shared_parser():
    """sft shares OPD's numpy-2-aware parser instead of keeping its own float() copy.

    sft's three metrics reach the logger as plain python floats today (engine_workers.py
    returns loss/grad_norm via .item() and lr via get_last_lr()), so unlike OPD's
    Metric(SUM) they do not currently print in numpy's np.float64(...) spelling. the
    duplicate parser was still removed: one upstream metric-type change would have
    reintroduced the same silent drop, and the shared helper additionally rejects nan/inf,
    which would otherwise serialize into the heartbeat as bare NaN.
    """
    import ast
    import inspect
    import textwrap

    import flash.engine.worker.sft_train as sv

    source = textwrap.dedent(inspect.getsource(sv.run_sft_train))
    handler = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "on_line"
    )
    calls = [
        node.func.id
        for node in ast.walk(handler)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("parse_verl_metric") == 3
    assert "_metric_value" not in calls
    # the duplicated helper and its regex are gone, not merely unused.
    assert not hasattr(sv, "_metric_value")
    assert not hasattr(sv, "_VERL_METRIC_RE")


def test_sft_drops_a_non_finite_loss_instead_of_poisoning_the_heartbeat():
    """a nan loss serializes as bare NaN, which strict json consumers reject."""
    import flash.engine.worker.sft_train as sv

    assert sv.parse_verl_metric("step:2 - train/loss:nan - train/lr:1e-05", "train/loss") is None
    assert sv.parse_verl_metric("step:2 - train/loss:inf", "train/loss") is None
    # a finite value on the same line is unaffected.
    assert sv.parse_verl_metric("step:2 - train/loss:nan - train/lr:1e-05", "train/lr") == 1e-05


def test_sft_never_enables_liger_because_it_zeroes_the_lora_gradient():
    """liger is off on the sft path, and the emitted verl override must say so.

    GRAD-001: a matched two-arm a/b on Qwen3.5-9B (identical data, seed, hardware and code,
    differing only in `model.use_liger`) measured train/grad_norm 0.0 with liger on and 7.02
    with liger off, at a loss identical to four decimal places. liger silently severed the
    gradient to the lora params under this fsdp2 + peft + gradient-checkpointing composition,
    so sft trained nothing while looking healthy. the grpo path never sets the key (verl
    defaults it false), which is why only sft was affected.

    this asserts the RENDERED override rather than the config literal, so re-enabling liger
    anywhere between the dict and the command line fails the test.
    """
    from flash.engine.worker.sft_train import build_sft_overrides

    base = {
        "fused_ce_backend": "torch",
        "train_files": "/w/train.parquet",
        "train_batch_size": 8,
        "max_length": 1024,
        "micro_batch": 1,
        "max_token_len_per_gpu": 1024,
        "custom_dataset_path": "/w/ds.py",
        "model_path": "Qwen/Qwen3.5-9B",
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": "all-linear",
        "lora_adapter_path": None,
        "ulysses_sp_size": 1,
        "lr": 1e-4,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "optimizer_impl": "bitsandbytes.optim",
        "optimizer_name": "PagedAdamW8bit",
        "optimizer_kwargs": None,
        "local_dir": "/w/ckpt",
        "save_freq": 50,
        "n_gpus_per_node": 1,
        "seed": 42,
        "project_name": "p",
        "experiment_name": "e",
        "loop_epochs": 1,
        "gradient_checkpointing": True,
        "total_training_steps": None,
        "total_epochs": 1,
        "loggers": ["console"],
    }

    # the default must be off: a caller that omits the key must not get liger.
    assert "model.use_liger=false" in build_sft_overrides(dict(base))
    assert "model.use_liger=true" not in build_sft_overrides(dict(base))

    # the dense-logit-free loss comes from fused kernels, not liger, so it survives.
    overrides = build_sft_overrides(dict(base))
    assert "model.use_fused_kernels=true" in overrides
    assert "model.fused_kernel_options.impl_backend=torch" in overrides


def test_drain_join_waits_out_a_slow_upload_until_the_run_deadline(monkeypatch):
    """VERL-131: a checkpoint drain is bounded by the RUN's wall deadline, not a constant.

    The measured failure was a 35B-A3B full-state upload that needed 607.6s against a fixed 600s
    join. It was healthy and still uploading -- it emitted another `checkpoint_uploading` heartbeat
    9s AFTER the join gave up -- and the timeout converted a run that had already trained and
    published into `failed`.

    The bound deliberately does NOT try to sample upload progress. `_HB_LAST_PROGRESS_TS` looks like
    a progress signal but is stamped unconditionally every 30s by the upload's own
    `liveness_heartbeat(keepalive=True)` daemon (heartbeat.py: `liveness=... and not keepalive`),
    so it advances whether or not bytes move -- a no-progress window keyed to it could never fire.
    The upload is already bounded from the inside by its retry budget and per-attempt deadline
    checks, so the only correct job here is to not impose a second, tighter deadline on top.
    """
    import threading
    import time as real_time

    # NB: the worker package rebinds the name `heartbeat` to the re-exported heartbeat
    # FUNCTION, so `import ... as hb` yields that function rather than this module.
    # import_module returns the real module object.
    hb = importlib.import_module("flash.engine.worker.io.heartbeat")
    from flash.engine.worker.runtime.pkg_proxy import W as _w

    # virtual clock: the test must not actually take an hour.
    now = [0.0]
    monkeypatch.setattr(hb.time, "monotonic", lambda: now[0])

    class _SlowUpload:
        """alive for 3600 virtual seconds -- six times the old fixed deadline."""

        def __init__(self) -> None:
            self.elapsed = 0.0

        def is_alive(self) -> bool:
            return self.elapsed < 3600.0

        def join(self, timeout=None) -> None:
            step = timeout or 5.0
            self.elapsed += step
            now[0] += step

    # the run still has budget left, so the drain must be allowed to finish. this raised under the
    # old fixed 600s join, which is exactly the reported failure.
    monkeypatch.setattr(_w, "_remaining_worker_wall_seconds", lambda: 7200.0, raising=False)
    hb.join_while_draining(_SlowUpload(), "slow uploader")

    # and the converse: once the RUN is out of time the drain must be cut off, or a wedged upload
    # holds the worker open past its own deadline.
    now[0] = 0.0
    budget = [120.0]

    class _Wedged:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout=None) -> None:
            step = timeout or 5.0
            now[0] += step
            budget[0] -= step

    monkeypatch.setattr(_w, "_remaining_worker_wall_seconds", lambda: budget[0], raising=False)
    with pytest.raises(RuntimeError, match="wall deadline expired"):
        hb.join_while_draining(_Wedged(), "wedged uploader")

    # a real finished thread returns immediately rather than waiting out a window.
    monkeypatch.setattr(_w, "_remaining_worker_wall_seconds", lambda: 7200.0, raising=False)
    done = threading.Thread(target=lambda: None)
    done.start()
    done.join()
    started = real_time.monotonic()
    hb.join_while_draining(done, "finished uploader")
    assert real_time.monotonic() - started < 5.0


def test_sft_ships_no_val_file_so_the_child_cannot_validate():
    """The child's last-step validation fires on the DATALOADER existing, not on test_freq.

    `verl/trainer/sft_trainer.py` gates it as
    `is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step)`.
    Python binds `and` tighter than `or`, so the left disjunct is true on the final step of every
    run that has a val dataloader at all -- `trainer.test_freq=-1` only suppresses the PERIODIC
    pass. Flash used to write a one-row `val.parquet` and pass it, buying a full inference forward
    per run whose `val/loss` nothing in flash reads. A null `data.val_files` makes the child's
    `val_dataset` (and hence `val_dataloader`) None, so the branch cannot execute.

    `data.val_files=null` is HARDCODED in `build_sft_overrides` rather than threaded through a cfg
    key, because null is the only supported value -- accepting one would let a caller recreate the
    inference forward this removes. So the override is asserted directly, and the worker source is
    asserted to name no val parquet at all.
    """
    import inspect

    from flash.engine.worker import sft_train

    # the override is unconditional: no fixture value can change it.
    assert "data.val_files=null" in build_sft_overrides(_cfg())
    src = inspect.getsource(build_sft_overrides)
    assert "cfg['val_files']" not in src
    assert 'cfg["val_files"]' not in src

    # comments are stripped first: the explanation mentions both val.parquet and val_file, and
    # matching those words in a comment would pass on a worker that still ships the file.
    worker = inspect.getsource(sft_train.run_sft_train)
    code = "\n".join(ln for ln in worker.splitlines() if not ln.strip().startswith("#"))
    assert "val.parquet" not in code
    assert "val_file" not in code


def test_sft_hardware_ranking_prices_the_profiled_batch_not_the_authored_one(monkeypatch):
    """Ranking must clamp on the batch the run EXECUTES, not the one the user typed.

    `sharded_step_seconds` credits SFT only the ranks `sft_data_parallel_cards` allows, and that
    reads `batch_size`. The workload profile reduces the authored batch to `examples_per_update`
    (1 for every exact-unpacked run), so ranking off the authored number would credit a 4-card
    candidate four ranks the worker will never launch -- picking a wider, costlier shape than the
    run can use, and disagreeing with the persisted quote, which does read the profile.
    """
    import types

    # a profile that reduces the authored batch of 8 to a single example per update.
    import flash.cost.spec as cost_spec
    from flash.core.spec import TrainSpec
    from flash.providers.base import run_config_for_ranking

    monkeypatch.setattr(
        cost_spec,
        "_sft_profile",
        lambda spec: types.SimpleNamespace(
            examples_per_update=1, retained_examples=10, max_length=1404
        ),
    )

    spec = types.SimpleNamespace(algorithm="sft", train=TrainSpec(batch_size=8))
    overrides = cost_spec.sft_ranking_overrides(spec)
    assert overrides["batch_size"] == 1

    # the row count binds the width too, and the MEASURED length is what a step is priced on --
    # ranking that reads the authored context length prices work the run will not do.
    assert overrides["sft_retained_examples"] == 10
    assert overrides["seq_len"] == 1404

    # the overrides must actually reach the config ranking prices, not just be computed.
    config = run_config_for_ranking(
        "Qwen/Qwen3.5-4B",
        "sft",
        train={"batch_size": 8, "max_context_tokens": 4096},
        overrides=overrides,
    )
    assert (config.batch_size, config.sft_retained_examples, config.seq_len) == (1, 10, 1404)

    # a non-sft run has no profile clamp and must pass its knobs through untouched.
    grpo = types.SimpleNamespace(algorithm="grpo", train=TrainSpec(batch_size=8))
    assert cost_spec.sft_ranking_overrides(grpo) == {}

    # an unreadable profile must not fail the submission -- rank on the authored knobs instead.
    # ranking runs BEFORE the quote, so raising here would fail a submission the quote would catch.
    def boom(spec):
        raise ValueError("digest mismatch")

    monkeypatch.setattr(cost_spec, "_sft_profile", boom)
    assert cost_spec.sft_ranking_overrides(spec) == {}
    fallback = run_config_for_ranking(
        "Qwen/Qwen3.5-4B",
        "sft",
        train={"batch_size": 8},
        overrides=cost_spec.sft_ranking_overrides(spec),
    )
    assert fallback.batch_size == 8


def test_sft_vram_sizing_uses_the_profiled_batch_not_the_authored_one(monkeypatch):
    """Submit must RESERVE for the work that runs, not the batch the user typed.

    Ranking takes the profile through `overrides`, but `required_vram_gb` sizes from `train`. Those
    are different vocabularies (`seq_len` vs `max_context_tokens`), so moving the profiled batch
    into `overrides` silently left sizing on the authored one: a 4B at the authored batch 8 / 4096
    reserves 23.0 GB while the run executes batch 1 / 1404 and needs 19.0 GB. That over-reserves by
    4 GB and can reject a card the run would have fit on -- the same authored-vs-executed split the
    ranking clamp exists to close.

    The assertion drives `allocate` and captures what sizing actually receives. Exercising
    `_overridden_train` alone cannot catch this: the helper keeps translating correctly whether or
    not the call site uses it, so a version that sizes off the authored `train` still passes.
    """
    from flash.core.spec import TrainSpec
    from flash.providers import allocator

    authored = {"batch_size": 8, "max_context_tokens": 4096}
    overrides = {"batch_size": 1, "seq_len": 1404, "sft_retained_examples": 10}

    sized_authored = allocator.required_vram_gb("Qwen/Qwen3.5-4B", "sft", train=authored)
    sized_executed = allocator.required_vram_gb(
        "Qwen/Qwen3.5-4B", "sft", train={"batch_size": 1, "max_context_tokens": 1404}
    )
    assert sized_executed < sized_authored, (
        f"sizing off the authored batch reserves {sized_authored} GB for work that needs "
        f"{sized_executed} GB"
    )

    captured = {}

    def capture(model_id, algorithm, *, train=None, thinking=False, model_revision=""):
        captured["train"] = train
        return sized_executed

    monkeypatch.setattr(allocator, "required_vram_gb", capture)
    # allocation itself may fail on provider availability; sizing runs first, which is the contract
    # under test.
    with contextlib.suppress(Exception):
        allocator.allocate("Qwen/Qwen3.5-4B", "sft", train=authored, overrides=overrides)
    assert captured["train"] == {"batch_size": 1, "max_context_tokens": 1404}, (
        f"allocate() sized VRAM from {captured.get('train')!r}; it must pass the profile-overridden "
        "knobs or submit reserves for a batch the run never executes"
    )

    # a dataclass train table must substitute the same way, and absent overrides must not touch it.
    spec_train = TrainSpec(batch_size=8, max_context_tokens=4096)
    with contextlib.suppress(Exception):
        allocator.allocate("Qwen/Qwen3.5-4B", "sft", train=spec_train, overrides=overrides)
    assert captured["train"].batch_size == 1
    assert captured["train"].max_context_tokens == 1404
    with contextlib.suppress(Exception):
        allocator.allocate("Qwen/Qwen3.5-4B", "sft", train=spec_train)
    assert captured["train"] is spec_train


def test_sft_idle_card_warning_only_recommends_widths_that_actually_work():
    """A remedy that cannot be acted on is worse than no remedy.

    Three ways the earlier wording failed. It routed the card advice through
    `largest_rentable_count(world_size)`, which is the next power of two DOWN and need not divide
    the batch or the rows either -- at 4 cards with a batch of 3 it named 2, and 2 does not divide
    3. And it advised raising `batch_size` whenever the batch was above 1, including when the batch
    already divided the allocation and the ROWS were what bound the width, where raising the batch
    changes nothing.

    Fixing the first by re-resolving under a rentable ceiling then broke rentability instead:
    `sft_data_parallel_cards` searches DOWNWARD for a divisor, so it walks back off the power-of-two
    grid and named 3 cards at 7/batch 6/rows 6. Divisibility and rentability are independent, so the
    sweep below asserts BOTH -- it passed on that revision while providers sold none of what it
    advised.
    """
    import contextlib
    import io
    import re

    from flash.engine.plan.steps import sft_data_parallel_cards
    from flash.engine.worker.sft_train_runner import _resolve_sft_world_size
    from flash.providers.base import rentable_gpu_counts

    def warn(cards, batch, rows):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            width = _resolve_sft_world_size(cards, batch, rows)
        return width, buf.getvalue()

    # rows bind (8 divides 4 cards cleanly), so raising the batch is not the remedy.
    width, text = warn(4, 8, 10)
    assert width == 2
    assert "a dataset of 10 rows" in text, text
    assert "batch_size" not in text, "rows bind here, so raising the batch cannot help"
    assert "allocate 2 card(s)" in text, text

    # the batch binds AND fixing it is sufficient (12 rows already divide 4), so the batch remedy
    # is legitimate and must still be offered.
    _, text = warn(4, 3, 12)
    assert "a batch of 3" in text, text
    assert "batch_size" in text, text

    # neither divides: raising the batch cannot reach full width because the rows still will not
    # split 4 ways, so name the dataset and the lower card count instead.
    _, text = warn(4, 6, 10)
    assert "a dataset of 10 rows" in text, text
    assert "batch_size" not in text, "rows block full width too, so the batch remedy is a dead end"

    # an unpacked run pins the batch to 1, which binds on its own -- but it is not the ROWS that
    # bind, and saying so is a false statement about the dataset. 12 divides 4 exactly here, so
    # blaming the rows sends the operator to reshape a dataset that was never the problem.
    width, text = warn(4, 1, 12)
    assert width == 1
    assert "single example per update" in text, text
    assert "12 rows" not in text, "12 divides 4 cleanly; the rows are not what bind at batch 1"
    assert "batch_size" not in text, "packing mode fixes the batch at 1, so it cannot be raised"
    assert "allocate 1 card(s)" in text, text

    # a batch remedy is only ever printed when acting on it actually restores the full allocation.
    for cards in range(1, 9):
        for batch in range(1, 17):
            for rows in range(32):
                _, text = warn(cards, batch, rows)
                if "batch_size" not in text:
                    continue
                raised = cards * (batch // cards + 1)
                assert sft_data_parallel_cards(cards, raised, rows) == cards, (
                    f"advised raising batch to {raised} at {cards}/{batch}/{rows}, which still "
                    "does not use every card"
                )

    # every width this warning recommends must be BOTH rentable and usable. neither implies the
    # other: the rentable count need not divide the batch (4 cards, batch 3 -> 2, and 3 % 2 != 0),
    # and a divisor need not be rentable (7 cards, batch 6, rows 6 -> 3, which nobody sells).
    for cards in range(1, 9):
        for batch in range(1, 17):
            for rows in range(1, 32):
                width, text = warn(cards, batch, rows)
                if width >= cards:
                    assert text == "", "must stay quiet when every allocated card is used"
                    continue
                found = re.search(r"allocate (\d+) card", text)
                assert found, text
                advised = int(found.group(1))
                assert advised in rentable_gpu_counts(cards), (
                    f"advised {advised} cards at {cards}/{batch}/{rows}: not a shape providers rent"
                )
                assert batch % advised == 0, (cards, batch, rows, advised)
                assert rows % advised == 0, (cards, batch, rows, advised)
                assert advised == sft_data_parallel_cards(advised, batch, rows)
                assert advised <= width, "advising more cards than the run can use is the same bug"


def test_sft_idle_card_advice_does_not_shrink_the_memory_the_run_is_running_on():
    """Advising FEWER cards than the run launched can take away memory it needs to exist.

    The idle-card warning fires on a rented shape the fit gate already accepted, and the ranks that
    joined are what hold the model. Dropping to the next rentable divisor is a VRAM change, not just
    a billing one: Qwen3.6-27B sft at 32k is sized at 159 GB, a 4x H100 rental launching 3 ranks
    provides 191.6 GB and runs, but batch 6 over 6 rows advised "allocate 2 card(s)" -- 130.4 GB,
    which the fit gate rejects. Acting on that remedy turns a working run into an unplaceable one.

    The worker cannot check the fit itself: it runs after allocation, on hardware already rented,
    and has no VRAM need in scope. So the advice is QUALIFIED rather than computed -- it stays a
    billing observation and never claims the smaller shape still holds the model. A width at or
    above the launched one needs no qualifier, because it takes no memory away.
    """
    import contextlib
    import io
    import re

    from flash.engine.worker.sft_train_runner import _resolve_sft_world_size

    def warn(cards, batch, rows):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            width = _resolve_sft_world_size(cards, batch, rows)
        return width, buf.getvalue()

    # the reported case: 3 ranks hold the run, the advised 2 cards would not.
    width, text = warn(4, 6, 6)
    assert width == 3, "batch 6 over 6 rows launches 3 of the 4 rented cards"
    assert "allocate 2 card(s)" in text, text
    assert "fits" in text, (
        "advice that drops below the launched width must be qualified as a fit question, "
        f"not presented as the remedy: {text!r}"
    )

    # whenever the advised width is BELOW the ranks actually running, the line must not present
    # the smaller shape as a straight remedy -- that shape may not hold the model at all.
    for cards in range(1, 9):
        for batch in range(1, 17):
            for rows in range(1, 32):
                width, text = warn(cards, batch, rows)
                if not text:
                    continue
                found = re.search(r"allocate (\d+) card", text)
                assert found, text
                advised = int(found.group(1))
                if advised < width:
                    assert "fits" in text, (
                        f"at {cards}/{batch}/{rows} advised {advised} cards below the {width} "
                        f"ranks running, unqualified: {text!r}"
                    )


def test_sft_quote_credits_the_width_the_rows_allow_not_just_the_batch():
    """Pricing must clamp on the rows too, or it re-opens the gap the batch clamp closed.

    A packed profile can leave a row count that narrows the width below what the batch alone
    permits: batch 8 with 10 retained rows on 4 cards launches 2 ranks, not 4. Crediting 4 there
    understates wall time and cost exactly as the authored-batch bug did.

    The row count must be carried explicitly rather than derived from `sft_packed_blocks`, which is
    `ceil(rows / examples_per_update)` and reconstructs those 10 rows as 16 -- an OVER-credit, i.e.
    the very failure this clamp exists to prevent.
    """
    from flash.cost import analytical
    from flash.cost.types import RunConfig
    from flash.engine.plan.steps import sft_data_parallel_cards

    def speedup(rows):
        config = RunConfig(
            model_id="Qwen/Qwen3.5-4B",
            method="sft",
            steps=10,
            batch_size=8,
            sft_retained_examples=rows,
        )
        return analytical.method_card_speedup(config, 4, "H100", "runpod")

    # the worker would launch 2 ranks on 10 rows and 4 on 16; the quote must agree with both.
    assert sft_data_parallel_cards(4, 8, 10) == 2
    assert sft_data_parallel_cards(4, 8, 16) == 4
    assert speedup(10) < speedup(16)
    assert speedup(10) == speedup(2 * 5)

    # an unknown row count must not constrain: the quote is built before the dataset exists on some
    # paths, and inventing a bound there would misprice every one of them.
    assert speedup(None) == speedup(16)


def test_sft_resume_guard_checks_the_launched_width_not_the_allocation():
    """The fsdp resume guard must compare against the width verl actually starts at.

    `_restore_verl_resume(..., world_size=...)` discards a checkpoint written at a different rank
    count. SFT shards by data, so the launched width is bounded by the batch and the row count and
    is NOT the allocated card count whenever either fails to divide it. Passing `options.gpu_count`
    there would discard a checkpoint that matches the run about to start, and keep one that does
    not -- the exact inversion the guard exists to prevent.
    """
    import inspect

    from flash.engine.worker import sft_train_runner

    src = inspect.getsource(sft_train_runner._prepare_sft_child)
    assert "_restore_verl_resume(options.paths.local_dir, world_size=world_size)" in src
    assert "world_size=options.gpu_count" not in src

    # and the resolved width must be established before the resume call that consumes it.
    assert src.index("world_size, micro_batch = _resolve_sft_width_and_micro_batch(") < src.index(
        "_restore_verl_resume("
    )


def test_reported_grad_accum_reconstructs_the_global_batch_under_data_parallelism():
    """notes must satisfy micro_batch x grad_accum x dp_size == the global batch.

    That product is how a reader reconstructs the token budget, and every factor in it is per-rank:
    `_resolve_sft_width_and_micro_batch` caps `micro_batch` to `train_batch_size // world_size`. So
    dividing the GLOBAL batch by the micro-batch -- the sequence-parallel formula, where each rank
    sees the whole batch -- reports an accumulation count world_size times too high, and the
    reconstruction lands world_size times over.
    """
    import math

    # (global batch, world size, requested micro-batch)
    for train_batch_size, world_size, requested in (
        (8, 4, 4),
        (8, 1, 2),
        (16, 2, 8),
        (6, 3, 4),
        (8, 8, 1),
    ):
        per_rank = max(1, train_batch_size // max(1, world_size))
        micro_batch = max(1, min(requested, per_rank))
        grad_accum = math.ceil((train_batch_size / max(1, world_size)) / micro_batch)
        assert micro_batch * grad_accum * world_size == train_batch_size, (
            f"batch {train_batch_size} over {world_size} ranks: "
            f"{micro_batch} x {grad_accum} x {world_size}"
        )


def test_publish_does_not_leave_every_step_adapter_on_the_container_disk(monkeypatch, tmp_path):
    """the exported adapter is deleted once it is durable on hf.

    the watcher exports each save to `export_root/step-N` and publishes it. nothing used to remove
    that directory, so a run kept one adapter per save for its whole lifetime, on the same container
    disk as the checkpoints. the rl path already drops its equivalent; sft did not.

    asserts the directory is gone AFTER publish rather than counting bytes: the leak is one
    undeleted directory per save, and its size is a property of the model, not of this bug.
    """
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    local_dir = tmp_path / "checkpoints"
    export_root = tmp_path / "exports"
    checkpoint_dir = local_dir / "global_step_7"
    (checkpoint_dir / "huggingface").mkdir(parents=True)
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")

    def fake_export(actor, adapter, **kwargs):
        os.makedirs(adapter, exist_ok=True)
        pathlib.Path(adapter, "adapter_model.safetensors").write_bytes(b"adapter")

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", fake_export)

    published = {}

    def fake_publish(adapter, step, **kwargs):
        # the adapter must still be readable AT publish time; it is only redundant afterwards.
        published["existed"] = os.path.isfile(os.path.join(adapter, "adapter_model.safetensors"))
        return f"step-{step}"

    monkeypatch.setattr(worker, "publish_deployable_checkpoint", fake_publish)
    monkeypatch.setattr(
        worker,
        "upload_resume_checkpoint",
        lambda step, checkpoint, **kwargs: (kwargs["before_upload"](), True)[1],
    )

    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(local_dir),
        export_root=str(export_root),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(7,),
    )
    watcher._publish(7, str(checkpoint_dir))

    assert published["existed"], "the adapter must exist while it is being published"
    assert not os.path.exists(export_root / "step-7"), (
        "the published adapter was left on the container disk"
    )


def test_a_failed_upload_still_frees_the_exported_adapter(monkeypatch, tmp_path):
    """a raising upload must not strand the adapter directory it exported.

    the disk pressure this cleanup exists to relieve is worst on the failure path -- a run that is
    retrying uploads is exactly the run that is short on space -- so once the adapter is durable on
    hf, a LATER failure in the same publish must not strand the now-redundant local copy.
    """
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    local_dir = tmp_path / "checkpoints"
    export_root = tmp_path / "exports"
    checkpoint_dir = local_dir / "global_step_3"
    (checkpoint_dir / "huggingface").mkdir(parents=True)

    monkeypatch.setattr(
        sft_train,
        "_export_checkpoint_adapter",
        lambda actor, adapter, **kwargs: os.makedirs(adapter, exist_ok=True),
    )
    # a successful publish returns the subfolder it committed to.
    monkeypatch.setattr(
        worker, "publish_deployable_checkpoint", lambda adapter, step, **kwargs: f"step-{step}"
    )

    def boom(step, checkpoint, **kwargs):
        # the adapter lands on hf, then the full-state upload beside it dies.
        kwargs["before_upload"]()
        raise RuntimeError("hf upload failed")

    monkeypatch.setattr(worker, "upload_resume_checkpoint", boom)

    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(local_dir),
        export_root=str(export_root),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(3,),
    )
    with pytest.raises(RuntimeError, match="hf upload failed"):
        watcher._publish(3, str(checkpoint_dir))

    assert not os.path.exists(export_root / "step-3"), (
        "a failed upload stranded the exported adapter on disk"
    )


def test_merge_is_refused_when_its_output_cannot_fit_beside_the_checkpoint(monkeypatch, tmp_path):
    """`verl.model_merger merge` writes the FULL model, so it needs room for a second copy.

    `save_hf_model_and_tokenizer` calls `model.save_pretrained(target_dir, state_dict=...)`, so
    exporting one 35b checkpoint materializes ~70 GB into `<adapter>_merge` beside the ~60 GB
    checkpoint it reads. When that does not fit, the merger dies partway through with ENOSPC and
    takes down a run whose training already succeeded.

    Asserts the subprocess is never launched: the value of the guard is failing BEFORE the
    expensive write, with a message naming the shortfall.
    """
    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    actor_dir = tmp_path / "global_step_9"
    (actor_dir / "huggingface").mkdir(parents=True)
    (actor_dir / "model_world_size_1_rank_0.pt").write_bytes(b"x" * 4096)

    launched = []
    monkeypatch.setattr(
        verl_checkpoints.subprocess, "run", lambda *a, **kw: launched.append(a) or None
    )
    monkeypatch.setattr(
        verl_checkpoints.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1 << 40, used=1 << 40, free=1024),
    )

    with pytest.raises(verl_checkpoints.MergeDiskHeadroomError, match=r"only 0\.0 GB is free"):
        verl_checkpoints.export_peft_adapter(
            str(actor_dir),
            str(tmp_path / "adapter"),
            base_model_id="org/model",
            python_bin="/verl/python",
        )
    assert not launched, "the merger ran even though its output could not fit"


def test_merge_sizing_ignores_shards_nested_below_the_checkpoint(tmp_path):
    """the estimate must count only the shards the merger actually opens.

    `_load_and_merge_state_dicts` reads exact top-level paths -- `Path(local_dir) /
    f"model_world_size_{W}_rank_{r}.pt"`, one per rank -- and never recurses. A nested directory
    that happens to hold shard-named files (a staging tree, a partially copied checkpoint) is not
    merge input, so adding its bytes would inflate the requirement and refuse a merge that fits.

    That is the same over-estimate the optimizer-state exclusion exists to prevent, in a different
    form, and it matters in the same direction: a guard that refuses a valid merge is a regression
    this PR would have introduced, whereas one that underestimates merely leaves today's behavior.
    """
    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    actor_dir = tmp_path / "ckpt"
    nested = actor_dir / "staged"
    nested.mkdir(parents=True)
    (actor_dir / "model_world_size_2_rank_0.pt").write_bytes(b"x" * 8192)
    (nested / "model_world_size_2_rank_0.pt").write_bytes(b"x" * 8192)

    assert verl_checkpoints._model_shard_bytes(str(actor_dir)) == 8192


def test_merge_headroom_allows_the_merge_when_there_is_room(monkeypatch, tmp_path):
    """paired control: the guard must not block a merge that fits.

    Without this, a guard that always raised would pass the refusal test above.
    """
    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    actor_dir = tmp_path / "global_step_9"
    (actor_dir / "huggingface").mkdir(parents=True)
    (actor_dir / "model_world_size_1_rank_0.pt").write_bytes(b"x" * 4096)

    launched = []
    monkeypatch.setattr(
        verl_checkpoints.subprocess,
        "run",
        lambda *a, **kw: launched.append(a) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        verl_checkpoints.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1 << 40, used=0, free=1 << 40),
    )
    # the merger is faked, so it produces no adapter; only reaching that error proves it ran.
    with pytest.raises(RuntimeError, match="did not produce a peft adapter"):
        verl_checkpoints.export_peft_adapter(
            str(actor_dir),
            str(tmp_path / "adapter"),
            base_model_id="org/model",
            python_bin="/verl/python",
        )
    assert launched, "the merger was refused even though its output fits"


def test_worker_disables_xet_upload_staging_before_importing_hf(monkeypatch):
    """uploads must stream from the checkpoint, not stage a second copy beside it.

    `hf_xet` is an unconditional dependency of `huggingface-hub` on x86_64, and Xet is selected
    merely because it imports -- so `upload_folder` chunks through a cache under `HF_XET_CACHE`
    (default `$HF_HOME/xet`), the same container disk holding the checkpoint. The legacy path
    streams from the source handle instead. A real 35b run died with ENOSPC under that staging dir.
    """
    from flash.engine.worker.io import hf

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    hf._disable_xet_upload_staging()
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_an_explicit_xet_choice_is_not_overridden(monkeypatch):
    """paired control: an operator who deliberately set the variable keeps their value.

    Proves the fix is a default, not a hardcode -- and that the assertion above is reading flash's
    own write rather than a value that was already there.
    """
    from flash.engine.worker.io import hf

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    hf._disable_xet_upload_staging()
    assert os.environ["HF_HUB_DISABLE_XET"] == "0"


def test_an_adapter_is_freed_even_when_before_upload_never_ran(monkeypatch, tmp_path):
    """the export must not survive a path that skipped the publish callback entirely.

    `upload_resume_checkpoint` can return WITHOUT running `before_upload`. Two of its early returns
    are reachable from this caller: the slot is already held by another upload (returns False), and
    HF_REPO is unset (returns True). Its third, `skip_upload`, is not -- the sft watcher never passes
    that argument -- so it is deliberately not claimed here.

    Retaining the adapter on either reachable path looks protective, but nothing in the sft watcher
    ever reads `export_root` again -- the step joins `processed_steps` and `_pending` filters it out
    forever -- so the directory would simply accumulate. The busy-slot branch is the one that fires
    on EVERY step once an upload is slow enough to hold the slot, which is exactly the busy-disk case
    this PR exists for, so that is the one simulated below.
    """
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    local_dir = tmp_path / "checkpoints"
    export_root = tmp_path / "exports"
    checkpoint_dir = local_dir / "global_step_5"
    (checkpoint_dir / "huggingface").mkdir(parents=True)

    monkeypatch.setattr(
        sft_train,
        "_export_checkpoint_adapter",
        lambda actor, adapter, **kwargs: os.makedirs(adapter, exist_ok=True),
    )
    monkeypatch.setattr(
        worker, "publish_deployable_checkpoint", lambda adapter, step, **kwargs: None
    )
    # the slot was busy: returns False having never called before_upload.
    monkeypatch.setattr(worker, "upload_resume_checkpoint", lambda step, ckpt, **kwargs: False)

    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(local_dir),
        export_root=str(export_root),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(),
    )
    watcher._publish(5, str(checkpoint_dir))

    assert not os.path.exists(export_root / "step-5"), (
        "an adapter with no reader was left on the container disk"
    )


def test_importing_the_worker_package_does_not_freeze_the_xet_default(monkeypatch):
    """the disable must still be able to take effect when the worker starts.

    `huggingface_hub.constants` reads HF_HUB_DISABLE_XET into a module constant at import time, so
    setting the variable afterwards changes nothing -- `is_xet_available()` keeps returning True and
    uploads keep staging through the xet cache. That makes this an ORDERING contract, not just a
    setenv: if anything ever pulls huggingface_hub in at `flash.engine.worker` import time, the fix
    silently becomes a no-op while every assertion about the env var still passes.

    Asserts the resulting behaviour rather than the call order, so a refactor that moves the call,
    or adds a module-level hf import above it, fails here instead of in production.
    """
    import subprocess
    import sys

    probe = (
        "import os, sys\n"
        "os.environ.pop('HF_HUB_DISABLE_XET', None)\n"
        "import flash.engine.worker\n"
        "assert not [m for m in sys.modules if m.startswith('huggingface_hub')], (\n"
        "    'huggingface_hub was imported during flash.engine.worker import; '\n"
        "    'the xet default is frozen before the worker can disable it'\n"
        ")\n"
        "flash.engine.worker._disable_xet_upload_staging()\n"
        "from huggingface_hub.utils._runtime import is_xet_available\n"
        "assert not is_xet_available(), 'xet is still selected for uploads'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr


def test_optimizer_state_is_not_counted_against_the_merge(tmp_path):
    """the estimate must cover what the merger WRITES, not what the checkpoint holds.

    `_load_and_merge_state_dicts` loads only `model_world_size_*_rank_*.pt`, and that merged dict is
    what `save_pretrained` writes back out. The `optim_*` and `extra_state_*` files beside it are
    read by resume and never materialized by the merger, so charging them to the merge inflates the
    requirement by the whole optimizer state -- about 7.6 GB of adam moments on a 35b rank-32 run.

    That direction of error is the dangerous one. Underestimating just lets the merger hit ENOSPC
    the way it already does today; overestimating fails a run that had room, which is a regression
    the guard itself would have introduced.
    """
    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    source = tmp_path / "global_step_9"
    source.mkdir()
    (source / "model_world_size_2_rank_0.pt").write_bytes(b"x" * 4096)
    (source / "model_world_size_2_rank_1.pt").write_bytes(b"x" * 4096)
    (source / "optim_world_size_2_rank_0.pt").write_bytes(b"x" * 65536)
    (source / "extra_state_world_size_2_rank_0.pt").write_bytes(b"x" * 2048)

    assert verl_checkpoints._model_shard_bytes(str(source)) == 8192, (
        "optimizer or extra state was charged to the merge, which can refuse a merge that fits"
    )


def test_the_adapter_is_moved_out_of_the_merge_tree_not_copied(monkeypatch, tmp_path):
    """the export must never hold two copies of the adapter at once.

    `export_peft_adapter` deletes `<adapter>_merge` only AFTER placing the adapter in its final
    directory. Copying the files there would put a second copy of every adapter file on the disk
    while the whole merge tree is still present, and that transient peak is not what
    `require_merge_headroom` reserved space for -- it is the same class of overshoot the guard
    exists to prevent.

    Both directories are siblings under one parent, hence one filesystem, so the placement can be a
    rename that transfers no bytes.

    The assertion has to be made at the moment `rmtree` runs, not after the call returns: by then
    the merge tree is gone under either implementation, so a post-hoc check cannot tell a move from
    a copy. What distinguishes them is whether the source files are still occupying space when the
    final adapter already exists.
    """
    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    actor_dir = tmp_path / "global_step_5"
    (actor_dir / "huggingface").mkdir(parents=True)
    (actor_dir / "model_world_size_1_rank_0.pt").write_bytes(b"x" * 512)
    out_adapter = tmp_path / "adapter"
    merge_out = tmp_path / "adapter_merge"
    lora_dir = merge_out / "lora_adapter"

    def fake_merge(*args, **kwargs):
        lora_dir.mkdir(parents=True)
        (lora_dir / "adapter_config.json").write_text("{}")
        (lora_dir / "adapter_model.safetensors").write_bytes(b"weights")
        return SimpleNamespace(returncode=0)

    duplicated: list[str] = []
    real_rmtree = verl_checkpoints.shutil.rmtree

    def watching_rmtree(path, *args, **kwargs):
        if str(path) == str(merge_out) and lora_dir.is_dir():
            duplicated.extend(sorted(os.listdir(lora_dir)))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(verl_checkpoints.subprocess, "run", fake_merge)
    monkeypatch.setattr(verl_checkpoints.shutil, "rmtree", watching_rmtree)
    monkeypatch.setattr(
        verl_checkpoints.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1 << 40, used=0, free=1 << 40),
    )

    verl_checkpoints.export_peft_adapter(
        str(actor_dir),
        str(out_adapter),
        base_model_id="org/model",
        python_bin="/verl/python",
    )

    assert (out_adapter / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (out_adapter / "adapter_config.json").read_text() == "{}"
    assert duplicated == [], (
        f"the merge tree still held {duplicated} after the adapter was placed, "
        "so both copies were on disk at once"
    )
    assert not merge_out.exists()


def test_repeated_swallowed_publish_failures_do_not_accumulate_adapters(monkeypatch, tmp_path):
    """the swallowed-failure path, asserted as an accumulation rather than a single free.

    `publish_deployable_checkpoint` raises for a REQUIRED step, but on an optional one it retries,
    prints a warning, and returns None -- the failure is swallowed. Retaining the export on that
    path would be pointless and actively harmful: no sweep, republish, or finalization in the sft
    path walks `export_root`, so a run whose deployable uploads keep failing transiently would leave
    a full adapter behind on every save and rebuild the exhaustion this class bounds. A required step
    does not depend on the retained copy either -- `required=True` raises rather than returning None,
    so it leaves through the exception path instead of continuing here.

    Driven over several steps because the property that matters is that N failed saves leave O(1)
    directories rather than N. A single-step assertion cannot distinguish "freed" from "freed this
    once": I mutated the cleanup to skip exactly one step and a single-step version still passed,
    while this one caught it.
    """
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    local_dir = tmp_path / "checkpoints"
    export_root = tmp_path / "exports"

    def fake_export(actor, adapter, **kwargs):
        os.makedirs(adapter, exist_ok=True)
        with open(os.path.join(adapter, "adapter_model.safetensors"), "wb") as fh:
            fh.write(b"w" * 4096)

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", fake_export)
    # every optional publish fails and swallows the error, the worst sustained case.
    monkeypatch.setattr(
        worker, "publish_deployable_checkpoint", lambda adapter, step, **kwargs: None
    )
    monkeypatch.setattr(
        worker,
        "upload_resume_checkpoint",
        lambda step, ckpt, **kwargs: (kwargs["before_upload"](), True)[1],
    )

    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(local_dir),
        export_root=str(export_root),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(),
    )

    for step in range(1, 9):
        checkpoint_dir = local_dir / f"global_step_{step}"
        (checkpoint_dir / "huggingface").mkdir(parents=True)
        watcher._publish(step, str(checkpoint_dir))

    left = sorted(p for p in os.listdir(export_root) if p.startswith("step-"))
    assert left == [], f"8 failed saves left {len(left)} adapters on disk: {left}"


def test_the_opd_watcher_still_keeps_its_export(monkeypatch, tmp_path):
    """the sft cleanup must not be generalized to the siblings that have a reader.

    `_OpdVerlCheckpointWatcher` subclasses the sft watcher, so a cleanup placed in a shared method
    would silently reach it. Both siblings hand their export to something that runs LATER -- rl
    republishes from `staged_steps` on a subsequent sweep, opd passes `adapter_dir` into
    `_stage_retry_contract` -- so for them the directory is live state, not garbage. The sft path is
    safe to clear precisely because it has no such consumer.

    Asserted by running the opd watcher's real `_publish` and looking at the disk afterwards. An
    earlier version of this test grepped `inspect.getsource` for `"rmtree(adapter_dir"`, which
    proved nothing: writing the deletion as `rmtree(os.fspath(adapter_dir))` does not match that
    substring, so the adapter could be destroyed with the assertion still green. Source text is not
    the contract; the surviving directory is.
    """
    from flash.engine.worker.train.opd import failures as opd_failures
    from flash.engine.worker.train.sft import checkpoints as sft_checkpoints

    assert (
        opd_failures._OpdVerlCheckpointWatcher._publish
        is not sft_checkpoints._VerlCheckpointWatcher._publish
    ), "opd no longer overrides _publish, so it now inherits the sft deletion"

    staged: dict[str, object] = {}
    monkeypatch.setattr(
        opd_failures,
        "_export_checkpoint_adapter",
        lambda actor, adapter, **kwargs: os.makedirs(adapter, exist_ok=True),
    )
    monkeypatch.setattr(
        opd_failures,
        "_stage_retry_contract",
        lambda checkpoint_dir, **kwargs: staged.update(kwargs),
    )
    monkeypatch.setattr(
        opd_failures._w, "publish_deployable_checkpoint", lambda adapter, step, **kw: f"step-{step}"
    )
    monkeypatch.setattr(
        opd_failures._w,
        "upload_resume_checkpoint",
        lambda step, ckpt, **kwargs: (kwargs["before_upload"](), True)[1],
    )

    export_root = tmp_path / "opd-exports"
    checkpoint_dir = tmp_path / "ckpts" / "global_step_2"
    (checkpoint_dir / "actor").mkdir(parents=True)

    watcher = opd_failures._OpdVerlCheckpointWatcher(
        local_dir=str(tmp_path / "ckpts"),
        export_root=str(export_root),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(2,),
        seed=0,
        prompt_pool_fingerprint="fp",
        prompts_per_step=1,
        group_size=1,
        accounting_state=lambda step: None,
    )
    watcher._publish(2, str(checkpoint_dir))

    # the retry contract recorded this exact path, so the directory has to still be there.
    assert staged["adapter_dir"] == str(export_root / "step-2")
    assert os.path.isdir(export_root / "step-2"), (
        "the opd watcher deleted an adapter its retry contract still points at"
    )


def test_the_rl_watcher_keeps_a_staged_adapter_until_a_later_sweep_publishes_it(
    monkeypatch, tmp_path
):
    """the rl half of the same contract, driven across the two sweeps that actually span it.

    The rl uploader stages an adapter on the sweep a checkpoint appears and publishes it on a LATER
    one, once the gradient gate opens -- `staged_steps` carries the path between them. That gap is
    the whole reason the sft deletion cannot be lifted into shared code.

    The gate is what opens the gap, so the test has to close it. An earlier version passed
    `had_gradient=lambda: True` and hand-called `_stage_deployable` then `_publish_ready`, which
    manufactured a two-sweep sequence that production would never produce: with the gate already
    open, `_run` stages and publishes in ONE iteration, so the window where the adapter must survive
    on disk unpublished never opened and a cleanup placed there would have stayed green. Bugbot
    caught that. Here the gate starts shut and the checkpoint is discovered through `_pending`, so
    the retention window is the real one.
    """
    from flash.engine.worker.train.rl import checkpoints as rl_checkpoints

    published: list[str] = []

    def fake_export(actor_dir, adapter_dir, **kwargs):
        os.makedirs(adapter_dir, exist_ok=True)
        with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "wb") as fh:
            fh.write(b"w" * 2048)

    rl_train_module = rl_checkpoints._rl_train()
    monkeypatch.setattr(rl_train_module, "export_peft_adapter", fake_export)
    monkeypatch.setattr(
        rl_train_module, "stamp_adapter_dir_provenance", lambda *a, **kw: None, raising=False
    )
    monkeypatch.setattr(rl_checkpoints._w, "write_base_model_provenance", lambda *a, **kw: None)
    monkeypatch.setattr(
        rl_checkpoints._w,
        "publish_deployable_checkpoint",
        lambda adapter_dir, step, **kw: published.append(adapter_dir),
    )

    monkeypatch.setattr(rl_checkpoints._w, "upload_resume_checkpoint", lambda *a, **kw: True)

    local_dir = tmp_path / "ckpts"
    export_root = tmp_path / "rl-exports"
    checkpoint_dir = local_dir / "global_step_4"
    (checkpoint_dir / "actor").mkdir(parents=True)
    # verl advances this marker only after the checkpoint is fully written, so `_pending` finds the
    # step exactly as it does in a real run.
    (local_dir / "latest_checkpointed_iteration.txt").write_text("4")

    gate_open = False
    uploader = rl_checkpoints._VerlResumeUploader(
        local_dir=str(local_dir),
        resume_step=0,
        required_steps=(4,),
        export_root=str(export_root),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        preprocessor=types.SimpleNamespace(save_pretrained=lambda path: None),
        had_gradient=lambda: gate_open,
    )

    # sweep one, gate SHUT: the step is staged because verl may prune its checkpoint at any time,
    # but nothing may be published yet. this is the window the adapter has to survive.
    for step, path in uploader._pending():
        if step in uploader.required_steps and step not in uploader.staged_steps:
            uploader.staged_steps[step] = uploader._stage_deployable(step, path)
            uploader._publish_ready()
        uploader.processed_steps.add(step)
    uploader._publish_ready()

    adapter_dir = uploader.staged_steps[4]
    assert published == [], "the gradient gate was shut, so nothing may have been published"
    assert os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")), (
        "the rl watcher discarded a staged adapter while the gradient gate was still shut"
    )

    # verl is free to prune its own checkpoint now; only `export_root` carries the step forward.
    shutil.rmtree(checkpoint_dir)

    # sweep two, gate OPEN: the surviving directory is read back out of `staged_steps` and published.
    gate_open = True
    uploader._publish_ready()

    assert published == [adapter_dir], "the staged adapter never reached publication"
    assert os.path.exists(os.path.join(adapter_dir, "adapter_model.safetensors")), (
        "the rl watcher lost the adapter weights between staging and publication"
    )


def test_a_failed_merge_does_not_strand_the_merge_tree(monkeypatch, tmp_path):
    """the largest transient on the disk must not survive its own failure.

    `merge_out` holds the full merged model, tens of gb on a real run. If the merger dies partway
    through -- or writes a layout this code refuses -- deleting it only on the success path would
    leave that tree behind on exactly the disk this module exists to protect, and the next save
    would start with less room than this one had. Cleanup has to cover the resource's whole
    lifetime, not just the happy path.
    """
    import subprocess

    from flash.engine.worker.verl import checkpoints as verl_checkpoints

    actor_dir = tmp_path / "global_step_3"
    (actor_dir / "huggingface").mkdir(parents=True)
    (actor_dir / "model_world_size_1_rank_0.pt").write_bytes(b"x" * 512)
    out_adapter = tmp_path / "adapter"
    merge_out = tmp_path / "adapter_merge"

    def dying_merge(*args, **kwargs):
        # the merger writes most of the model, then fails: the exact shape of a disk-full death.
        (merge_out / "model-00001-of-00002.safetensors").parent.mkdir(parents=True, exist_ok=True)
        (merge_out / "model-00001-of-00002.safetensors").write_bytes(b"w" * 65536)
        raise subprocess.CalledProcessError(1, "verl.model_merger")

    monkeypatch.setattr(verl_checkpoints.subprocess, "run", dying_merge)
    monkeypatch.setattr(
        verl_checkpoints.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1 << 40, used=0, free=1 << 40),
    )

    with pytest.raises(subprocess.CalledProcessError):
        verl_checkpoints.export_peft_adapter(
            str(actor_dir),
            str(out_adapter),
            base_model_id="org/model",
            python_bin="/verl/python",
        )

    assert not merge_out.exists(), "a failed merge left its partial model tree on the disk"


def test_a_failed_export_does_not_strand_a_partial_adapter(monkeypatch, tmp_path):
    """the watcher must free an export that died while being written.

    The cleanup used to begin after `_export_checkpoint_adapter` returned, so an export that raised
    partway through left its directory behind with no reader and no later sweep to collect it. On a
    run whose exports keep failing that is one partial adapter per save -- the accumulation this
    class exists to bound, reached through the failure path instead of the success one.
    """
    import flash.engine.worker as worker
    from flash.engine.worker import sft_train

    local_dir = tmp_path / "checkpoints"
    export_root = tmp_path / "exports"
    checkpoint_dir = local_dir / "global_step_7"
    (checkpoint_dir / "huggingface").mkdir(parents=True)

    def dying_export(actor, adapter, **kwargs):
        os.makedirs(adapter, exist_ok=True)
        with open(os.path.join(adapter, "adapter_model.safetensors"), "wb") as fh:
            fh.write(b"partial")
        raise RuntimeError("merger died")

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", dying_export)
    monkeypatch.setattr(
        worker, "publish_deployable_checkpoint", lambda adapter, step, **kwargs: f"step-{step}"
    )
    monkeypatch.setattr(worker, "upload_resume_checkpoint", lambda step, ckpt, **kwargs: True)

    watcher = sft_train._VerlCheckpointWatcher(
        local_dir=str(local_dir),
        export_root=str(export_root),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(),
    )

    with pytest.raises(RuntimeError, match="merger died"):
        watcher._publish(7, str(checkpoint_dir))

    assert not os.path.exists(export_root / "step-7"), (
        "a failed export left a partial adapter on the container disk"
    )


def test_the_worker_disables_xet_as_its_very_first_action():
    """the startup wiring itself, not just the helper.

    The other xet tests call `_disable_xet_upload_staging()` directly, or assert that importing the
    worker package does not freeze the default. Neither notices if the CALL is deleted from
    `_run_worker_mode` -- I verified that by removing it, and all three stayed green.

    Asserted over the AST rather than by running the worker. `_run_worker_mode` performs real boot
    work (heartbeat, kernel cache, gpu probe) and cannot be driven to a mode handler in a test
    process, and observing `os.environ` in-process cannot attribute the value anyway: an earlier
    test in the same session may already have set it.

    What actually has to hold is an ordering property, and ordering is exactly what the AST shows:
    the disable must be the FIRST executable statement, because `huggingface_hub.constants` captures
    `HF_HUB_DISABLE_XET` at import time and any import above it would freeze the default first.
    Matching the call by name also survives the alternate spellings a substring grep would miss.
    """
    import ast
    import inspect
    import textwrap

    import flash.engine.worker as worker

    tree = ast.parse(textwrap.dedent(inspect.getsource(worker._run_worker_mode)))
    body = ast.get_docstring(tree.body[0], clean=False) and tree.body[0].body[1:]
    first = (body or tree.body[0].body)[0]

    value = first.value if isinstance(first, ast.Expr) else None
    called = value.func if isinstance(value, ast.Call) else None
    assert isinstance(called, ast.Name | ast.Attribute), (
        f"the first statement of _run_worker_mode is {ast.dump(first)[:80]}, not a call; "
        "xet staging must be disabled before anything else can import huggingface_hub"
    )
    name = called.attr if isinstance(called, ast.Attribute) else called.id
    assert name == "_disable_xet_upload_staging", (
        f"_run_worker_mode starts by calling {name!r}, not _disable_xet_upload_staging; "
        "uploads would stage a second copy of every checkpoint through the xet cache"
    )


def test_sft_micro_batch_never_exceeds_a_ranks_share_of_the_batch():
    """The micro-batch is sized against the GLOBAL batch, but each rank only sees its slice.

    Pinning ulysses off makes SFT shard by DATA, so verl hands each rank
    `train_batch_size // world_size` and rejects a micro-batch larger than that before the first
    optimizer step. Batch 8 on 4 ranks is a per-rank batch of 2, so a micro-batch of 4 -- correct
    for the global batch -- is an incompatible geometry the run dies on.

    Asserted on the config the child actually receives, because the micro-batch is computed long
    before `world_size` is resolved; a test that only exercised the sizing helper would keep passing
    while the call site shipped the uncapped value.
    """
    from types import SimpleNamespace

    from flash.engine.worker import sft_train_runner as runner

    def capped(train_batch_size, micro_batch, gpu_count, rows):
        _world, mb = runner._resolve_sft_width_and_micro_batch(
            SimpleNamespace(gpu_count=gpu_count),
            SimpleNamespace(rows=[{}] * rows),
            SimpleNamespace(train_batch_size=train_batch_size, micro_batch=micro_batch),
        )
        return mb

    # codex's case: batch 8 across 4 ranks -> per-rank 2, so a global-derived 4 must come down
    assert capped(8, 4, 4, 64) == 2
    # and the token budget must follow the same number, not the uncapped one
    assert capped(8, 4, 8, 64) == 1
    # a micro-batch already within the rank's share is untouched
    assert capped(8, 2, 4, 64) == 2
    assert capped(8, 1, 4, 64) == 1
    # single card: the rank owns the whole batch, so nothing is capped away
    assert capped(8, 4, 1, 64) == 4
    # never zero -- a DataLoader with batch_size=0 raises
    assert capped(1, 4, 8, 64) >= 1


def test_sft_result_records_the_micro_batch_that_ran_not_the_one_requested():
    """The result file must report the EXECUTED per-rank micro-batch.

    Data parallelism caps the micro-batch to one rank's share of the batch (batch 8 over 4 ranks
    leaves 2), and verl rejects anything larger. Recording `model.micro_batch` instead reports the
    REQUEST: a completed run claimed 4 while every rank ran 2, so a reader reconstructing the token
    budget or reproducing the run doubles the rows each rank actually held. `gradient_accumulation_
    steps` is derived from the same number, so it inherits the error.

    Asserted against the source because the writer's inputs are a live verl child; the point is
    which object the value is read FROM, and that is what the fix changes.
    """
    import inspect

    from flash.engine.worker import sft_train

    src = inspect.getsource(sft_train._write_sft_result)
    assert '"per_device_train_batch_size": child.micro_batch,' in src
    assert "/ child.micro_batch" in src, "gradient accumulation must derive from the executed value"
    # over one rank's share, since `child.micro_batch` is itself per-rank: dividing the GLOBAL batch
    # by it is the sequence-parallel formula and over-counts by the world size.
    assert "(model.train_batch_size / max(1, child.world_size)) / child.micro_batch" in src
    assert "math.ceil(model.train_batch_size / child.micro_batch)" not in src
    # the uncapped request must not reach the result under either key.
    assert '"per_device_train_batch_size": model.micro_batch,' not in src
    assert "math.ceil(model.train_batch_size / model.micro_batch)" not in src

    # and the child must actually carry it, or the writer above cannot read it.
    from flash.engine.worker import sft_train_runner

    assert "micro_batch" in sft_train_runner._SftChild.__dataclass_fields__
