from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from flash.engine.worker import sft as sft_mod
from flash.engine.worker.sft_openrlhf import (
    _attention_implementation,
    _child_output_is_cuda_oom,
    _chunked_selected_token_nll,
    _OpenRLHFCheckpointWatcher,
    _probe_gpu_in_subprocess,
    _processor_tokenized_row,
    _register_zero3_external_output_head,
    _resolve_immutable_model_revision,
    _serialize_multimodal_inputs,
    _training_batch_shape,
    build_openrlhf_sft_child_env,
    build_sft_openrlhf_args,
    build_text_openrlhf_rows,
    filter_openrlhf_sft_rows,
    render_openrlhf_sft_runtime,
    validate_openrlhf_warmstart_adapter,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed in offline CI",
)


class _CharTokenizer:
    eos_token = "<eos>"
    eos_token_id = 999
    pad_token_id = 0
    all_special_ids = (0, 999)

    def __call__(self, texts, *, truncation=False, max_length=None, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            ids = [ord(char) + 1 for char in text]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            rows.append(ids)
        return {"input_ids": rows}


class _FixtureTokenizer:
    eos_token = "<eos>"
    eos_token_id = 999
    pad_token_id = 0
    all_special_ids = (0, 999)

    def __init__(self):
        self._rows = {
            "single-prompt": (11, 12),
            "single-full<eos>": (11, 12, 21, 22, 999),
            "multi-prompt": (31, 32),
            "multi-full<eos>": (31, 32, 41, 42, 51, 52, 999),
            "thinking-prompt": (61, 62, 63),
            "thinking-full<eos>": (61, 62, 71, 72, 999),
        }

    def __call__(self, texts, *, truncation=False, max_length=None, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = [list(self._rows[text]) for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        return {"input_ids": rows}


def _arg_config(tmp_path):
    return {
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "dataset_path": str(tmp_path / "dataset"),
        "epochs": 3,
        "gradient_checkpointing": True,
        "gradient_checkpointing_reentrant": True,
        "learning_rate": 2e-5,
        "lora_alpha": 64,
        "lora_rank": 32,
        "max_length": 4096,
        "max_num_checkpoints": 5,
        "micro_batch_size": 2,
        "model_path": str(tmp_path / "model"),
        "output_dir": str(tmp_path / "output"),
        "resume_enabled": False,
        "row_count": 17,
        "seed": 123,
        "train_batch_size": 8,
        "wandb_enabled": False,
    }


def test_run_sft_dispatches_to_openrlhf_without_entering_trl(monkeypatch):
    calls = []
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    monkeypatch.setenv("FLASH_SFT_BACKEND", "openrlhf")
    monkeypatch.setattr(openrlhf_mod, "run_sft_openrlhf", lambda: calls.append("openrlhf"))

    sft_mod.run_sft()

    assert calls == ["openrlhf"]


def test_run_sft_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("FLASH_SFT_BACKEND", "megatron")

    with pytest.raises(ValueError, match="not a known sft backend"):
        sft_mod.run_sft()


def test_worker_gate_allows_only_openrlhf_sft_adapter_continuation(monkeypatch):
    import flash.engine.worker as worker

    class ReachedOpenRLHF(BaseException):
        pass

    monkeypatch.setattr(worker, "RUN_MODE", "sft")
    monkeypatch.setattr(
        worker,
        "JOB_SPEC",
        SimpleNamespace(train=SimpleNamespace(init_from_adapter="owner/repo:sft/source")),
    )
    monkeypatch.setattr(worker, "HF_REPO", "")
    monkeypatch.setattr(worker, "flush_optional_uploads", lambda: True)
    monkeypatch.setattr(worker, "error_artifact_name", lambda *args: "error.txt")
    monkeypatch.setattr(worker, "hf_upload_file", lambda *args: None)
    monkeypatch.setattr(worker, "_force_fla_triton_gdn_on_sm100", lambda: None)
    monkeypatch.setattr(worker, "_ensure_fla_fastpath_on_hopper", lambda: None)
    monkeypatch.setattr(worker, "_neutralize_tilelang_cudart_stub", lambda: None)
    monkeypatch.setattr(worker, "_restrict_fla_gdn_autotune_on_blackwell", lambda: None)
    monkeypatch.setattr(worker, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "gpu_diagnostics", lambda **kwargs: {})
    monkeypatch.setattr(worker, "finalize_alloc_conf_for_sleep", lambda: None)
    monkeypatch.setattr(worker, "load_mega_cache", lambda: None)
    monkeypatch.setattr(worker, "run_sft", lambda: (_ for _ in ()).throw(ReachedOpenRLHF()))

    monkeypatch.delenv("FLASH_SFT_BACKEND", raising=False)
    with pytest.raises(ValueError, match="SFT adapter continuation is not supported"):
        worker.main()

    monkeypatch.setenv("FLASH_SFT_BACKEND", "openrlhf")
    with pytest.raises(ReachedOpenRLHF):
        worker.main()


def test_build_sft_openrlhf_args_maps_zero3_lora_gc_and_dataset(tmp_path):
    args = build_sft_openrlhf_args(_arg_config(tmp_path))

    assert args[args.index("--model.model_name_or_path") + 1] == str(tmp_path / "model")
    assert args[args.index("--data.dataset") + 1] == str(tmp_path / "dataset")
    assert args[args.index("--ds.zero_stage") + 1] == "3"
    assert args[args.index("--ds.ring_attn_size") + 1] == "1"
    assert args[args.index("--ds.param_dtype") + 1] == "bf16"
    assert args[args.index("--ds.lora.rank") + 1] == "32"
    assert args[args.index("--ds.lora.alpha") + 1] == "64"
    assert "--model.gradient_checkpointing_enable" in args
    assert "--model.gradient_checkpointing_reentrant" in args
    assert "--ds.lora.target_modules" not in args
    assert args[args.index("--lr_scheduler") + 1] == "linear"
    assert args[args.index("--adam.betas") + 1 : args.index("--adam.betas") + 3] == [
        "0.9",
        "0.999",
    ]
    assert args[args.index("--adam.weight_decay") + 1] == "0.0"
    assert "--train.full_determinism_enable" not in args
    assert "--ckpt.save_hf" in args
    assert "--ckpt.load_enable" not in args


def test_build_sft_openrlhf_args_enables_full_state_resume(tmp_path):
    config = _arg_config(tmp_path)
    config["resume_enabled"] = True

    args = build_sft_openrlhf_args(config)

    assert "--ckpt.load_enable" in args
    assert args[args.index("--ckpt.path") + 1] == str(tmp_path / "checkpoints")


def test_build_sft_openrlhf_args_requires_positive_lora_rank(tmp_path):
    config = _arg_config(tmp_path)
    config["lora_rank"] = 0

    with pytest.raises(ValueError, match="positive LoRA rank"):
        build_sft_openrlhf_args(config)


@pytest.mark.parametrize(
    ("prompt", "full", "expected_ids", "expected_mask"),
    [
        ("single-prompt", "single-full", [11, 12, 21, 22, 999], [0, 0, 1, 1, 1]),
        (
            "multi-prompt",
            "multi-full",
            [31, 32, 41, 42, 51, 52, 999],
            [0, 0, 1, 1, 1, 1, 1],
        ),
        ("thinking-prompt", "thinking-full", [61, 62, 71, 72, 999], [0, 0, 1, 1, 1]),
    ],
)
def test_exact_mask_rows_use_literal_single_multiturn_and_thinking_fixtures(
    prompt,
    full,
    expected_ids,
    expected_mask,
):
    rows, dropped = build_text_openrlhf_rows(
        [{"prompt_text": prompt, "text": full}],
        _FixtureTokenizer(),
        4096,
    )

    assert dropped == 0
    assert rows == [
        {
            "input_ids": expected_ids,
            "loss_mask": expected_mask,
            "multimodal_inputs": b"",
        }
    ]


@requires_torch
def test_shifted_loss_mask_selects_literal_predicted_completion_tokens():
    import torch

    input_ids = torch.tensor([[11, 12, 21, 22, 999]])
    loss_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)

    predicted_token_ids = input_ids[:, 1:][loss_mask[:, 1:]]

    assert predicted_token_ids.tolist() == [21, 22, 999]


def test_processor_rejects_multimodal_truncation_that_would_desync_features():
    class Processor:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            del messages, kwargs
            if add_generation_prompt:
                return {"input_ids": [[10, 11]]}
            return {
                "input_ids": [[10, 11, 20, 21]],
                "attention_mask": [[1, 1, 1, 1]],
                "pixel_values": [[1.0, 2.0]],
                "image_grid_thw": [[1, 2, 3]],
            }

    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text"}]}]
    completion = [{"role": "assistant", "content": "answer"}]

    with pytest.raises(ValueError, match="desynchronize"):
        _processor_tokenized_row(
            Processor(),
            prompt,
            completion,
            [object()],
            max_length=3,
            thinking=False,
        )


def test_filter_openrlhf_sft_rows_drops_special_only_target_and_fails_if_empty():
    rows = [
        {"input_ids": [10, 99], "loss_mask": [0, 1], "multimodal_inputs": b""},
        {"input_ids": [10, 11], "loss_mask": [0, 1], "multimodal_inputs": b""},
    ]

    kept, dropped = filter_openrlhf_sft_rows(rows, {99})

    assert kept == [rows[1]]
    assert dropped == 1
    with pytest.raises(ValueError, match="every SFT example"):
        filter_openrlhf_sft_rows(rows[:1], {99})


def test_validate_openrlhf_warmstart_adapter_checks_rank_model_and_revision(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 32,
                "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "revision": "a" * 40,
            }
        ),
        encoding="utf-8",
    )

    validate_openrlhf_warmstart_adapter(
        str(adapter),
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision="a" * 40,
        expected_rank=32,
    )

    with pytest.raises(ValueError, match="immutable target model revision"):
        validate_openrlhf_warmstart_adapter(
            str(adapter),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="",
            expected_rank=32,
        )
    with pytest.raises(ValueError, match="rank"):
        validate_openrlhf_warmstart_adapter(
            str(adapter),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="a" * 40,
            expected_rank=16,
        )
    with pytest.raises(ValueError, match="revision"):
        validate_openrlhf_warmstart_adapter(
            str(adapter),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="b" * 40,
            expected_rank=32,
        )
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config["revision"] = None
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        validate_openrlhf_warmstart_adapter(
            str(adapter),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision="a" * 40,
            expected_rank=32,
        )


def test_validate_openrlhf_warmstart_adapter_accepts_matching_snapshot_path(tmp_path):
    revision = "c" * 40
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.bin").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "base_model_name_or_path": str(
                    tmp_path / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / revision
                ),
                "revision": None,
            }
        ),
        encoding="utf-8",
    )

    validate_openrlhf_warmstart_adapter(
        str(adapter),
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision=revision,
        expected_rank=16,
    )


@pytest.mark.parametrize("peft_revision", [None, "main"])
def test_validate_openrlhf_warmstart_adapter_accepts_matching_provenance(tmp_path, peft_revision):
    revision = "f" * 40
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "revision": peft_revision,
            }
        ),
        encoding="utf-8",
    )
    (adapter / "base_model_provenance.json").write_text(
        json.dumps(
            {
                "model_id": "Qwen/Qwen3.5-0.8B",
                "requested_revision": None,
                "resolved_commit": revision,
            }
        ),
        encoding="utf-8",
    )

    validate_openrlhf_warmstart_adapter(
        str(adapter),
        model_id="Qwen/Qwen3.5-0.8B",
        model_revision=revision,
        expected_rank=16,
    )

    provenance = json.loads((adapter / "base_model_provenance.json").read_text(encoding="utf-8"))
    provenance["resolved_commit"] = "0" * 40
    (adapter / "base_model_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        validate_openrlhf_warmstart_adapter(
            str(adapter),
            model_id="Qwen/Qwen3.5-0.8B",
            model_revision=revision,
            expected_rank=16,
        )


def test_resolve_immutable_model_revision_uses_prefetched_snapshot(monkeypatch):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    commit = "d" * 40
    monkeypatch.setattr(
        openrlhf_mod,
        "resolve_cached_model_commit",
        lambda model_id, revision: commit,
    )

    assert _resolve_immutable_model_revision("org/model", "") == commit


def test_resolve_immutable_model_revision_retries_unpinned_cache_miss(monkeypatch):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    monkeypatch.setattr(
        openrlhf_mod,
        "resolve_cached_model_commit",
        lambda model_id, revision: "",
    )

    with pytest.raises(openrlhf_mod._w.RetriableInfraError, match="immutable commit"):
        _resolve_immutable_model_revision("org/model", "")


def test_resolve_immutable_model_revision_rejects_unresolved_pinned_revision(monkeypatch):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    monkeypatch.setattr(
        openrlhf_mod,
        "resolve_cached_model_commit",
        lambda model_id, revision: "",
    )

    with pytest.raises(RuntimeError, match="immutable commit"):
        _resolve_immutable_model_revision("org/model", "main")


@requires_torch
@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_chunked_selected_token_nll_matches_full_loss_and_gradients(chunk_size):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(17)
    hidden = torch.randn(2, 6, 5, dtype=torch.float32, requires_grad=True)
    output_head = torch.nn.Linear(5, 11, bias=True, dtype=torch.float32)
    input_ids = torch.randint(0, 11, (2, 6))
    loss_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 1, 0]],
        dtype=torch.bool,
    )

    shifted_mask = loss_mask[:, 1:]
    full_logits = functional.linear(
        hidden[:, :-1, :].float(),
        output_head.weight.float(),
        output_head.bias.float(),
    )
    full_loss = functional.cross_entropy(
        full_logits[shifted_mask],
        input_ids[:, 1:][shifted_mask],
    )
    full_loss.backward()
    full_hidden_grad = hidden.grad.detach().clone()
    full_weight_grad = output_head.weight.grad.detach().clone()
    full_bias_grad = output_head.bias.grad.detach().clone()

    hidden.grad = None
    output_head.weight.grad = None
    output_head.bias.grad = None
    chunked_loss = _chunked_selected_token_nll(
        hidden,
        output_head,
        input_ids,
        loss_mask,
        chunk_size=chunk_size,
    )
    chunked_loss.backward()

    assert chunked_loss.item() == pytest.approx(full_loss.item(), abs=1e-5)
    assert torch.allclose(hidden.grad, full_hidden_grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(output_head.weight.grad, full_weight_grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(output_head.bias.grad, full_bias_grad, atol=1e-5, rtol=1e-5)


@requires_torch
def test_zero3_chunked_nll_registers_output_head_and_uses_deepspeed_checkpoint(monkeypatch):
    import torch
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    calls = []

    def register_external_parameter(module, parameter):
        calls.append(("register", module, parameter))

    def checkpoint(function, *args):
        calls.append(("checkpoint", function))
        return torch_checkpoint(function, *args, use_reentrant=True)

    deepspeed = ModuleType("deepspeed")
    deepspeed.checkpointing = SimpleNamespace(checkpoint=checkpoint)
    deepspeed.zero = SimpleNamespace(register_external_parameter=register_external_parameter)
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)

    forward_module = torch.nn.Module()
    output_head = torch.nn.Linear(3, 7)
    output_head.weight.ds_id = 1
    output_head.weight.ds_process_group = object()
    hidden_states = torch.randn(1, 3, 3, requires_grad=True)
    input_ids = torch.tensor([[0, 1, 5]])
    loss_mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        expected_loss = torch.nn.functional.cross_entropy(
            output_head(hidden_states[:, :-1][loss_mask[:, 1:]]).float(),
            input_ids[:, 1:][loss_mask[:, 1:]],
            reduction="sum",
        ) / 2

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def all_reduce(chunk_count, op=None, group=None):
        assert op is torch.distributed.ReduceOp.MAX
        assert group is output_head.weight.ds_process_group
        chunk_count.fill_(3)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    assert _register_zero3_external_output_head(forward_module, output_head) is True
    loss = _chunked_selected_token_nll(
        hidden_states,
        output_head,
        input_ids,
        loss_mask,
        chunk_size=2,
        zero3_active=True,
    )
    loss.backward()

    assert calls[0] == ("register", forward_module, output_head.weight)
    assert [call[0] for call in calls].count("checkpoint") == 3
    assert loss.item() == pytest.approx(expected_loss.item(), abs=1e-6)
    assert hidden_states.grad is not None
    assert output_head.weight.grad is not None


@requires_torch
def test_zero3_chunked_nll_pads_empty_rank_to_global_chunk_count(monkeypatch):
    import torch
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    checkpoint_calls = []

    def checkpoint(function, *args):
        checkpoint_calls.append(function)
        return torch_checkpoint(function, *args, use_reentrant=True)

    deepspeed = ModuleType("deepspeed")
    deepspeed.checkpointing = SimpleNamespace(checkpoint=checkpoint)
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)

    output_head = torch.nn.Linear(3, 7)
    output_head.weight.ds_id = 1
    output_head.weight.ds_process_group = object()
    hidden_states = torch.randn(1, 3, 3, requires_grad=True)
    input_ids = torch.tensor([[0, 1, 5]])
    loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    def all_reduce(chunk_count, op=None, group=None):
        assert op is torch.distributed.ReduceOp.MAX
        assert group is output_head.weight.ds_process_group
        chunk_count.fill_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    loss = _chunked_selected_token_nll(
        hidden_states,
        output_head,
        input_ids,
        loss_mask,
        chunk_size=2,
        denominator=torch.tensor(4.0),
        zero3_active=True,
    )
    loss.backward()

    assert len(checkpoint_calls) == 2
    assert loss.item() == 0.0
    assert hidden_states.grad is not None
    assert output_head.weight.grad is not None
    assert torch.count_nonzero(output_head.weight.grad).item() == 0


@requires_torch
def test_zero3_chunked_nll_falls_back_to_torch_reentrant(monkeypatch):
    import torch
    import torch.utils.checkpoint as checkpoint_mod

    checkpoint_calls = []
    original_checkpoint = checkpoint_mod.checkpoint

    def checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs.get("use_reentrant"))
        return original_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(checkpoint_mod, "checkpoint", checkpoint)
    monkeypatch.setitem(sys.modules, "deepspeed", ModuleType("deepspeed"))

    output_head = torch.nn.Linear(3, 7)
    output_head.weight.ds_id = 1
    hidden_states = torch.randn(1, 3, 3, requires_grad=True)
    input_ids = torch.tensor([[0, 1, 5]])
    loss_mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)

    loss = _chunked_selected_token_nll(
        hidden_states,
        output_head,
        input_ids,
        loss_mask,
        chunk_size=2,
        zero3_active=True,
    )
    loss.backward()

    assert checkpoint_calls == [True]
    assert hidden_states.grad is not None


@requires_torch
def test_chunked_nll_uses_output_head_hooks_during_backward_recompute():
    import torch

    torch.manual_seed(29)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    output_head = torch.nn.Linear(4, 13)
    input_ids = torch.randint(0, 13, (1, 7))
    loss_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    hook_calls = {"pre": 0, "post": 0}

    def pre_hook(_module, _inputs):
        hook_calls["pre"] += 1

    def post_hook(_module, _inputs, _output):
        hook_calls["post"] += 1

    output_head.register_forward_pre_hook(pre_hook)
    output_head.register_forward_hook(post_hook)

    loss = _chunked_selected_token_nll(
        hidden,
        output_head,
        input_ids,
        loss_mask,
        chunk_size=2,
    )
    loss.backward()

    assert hook_calls == {"pre": 6, "post": 6}
    assert hidden.grad is not None
    assert output_head.weight.grad is not None


def test_rendered_runtime_embeds_canonical_vlm_gradient_hook():
    import inspect

    from flash.engine.worker.perf.memory import enable_multimodal_input_require_grads

    source = render_openrlhf_sft_runtime()

    assert inspect.getsource(enable_multimodal_input_require_grads) in source
    assert "def _enable_vlm_input_require_grads" not in source


def test_rendered_runtime_keeps_sdpa_context_through_backward():
    source = render_openrlhf_sft_runtime()
    micro_step = source.split("for micro_index, batch in enumerate(window):", 1)[1].split(
        "global_step += 1", 1
    )[0]

    assert "_maybe_gather_output_head" not in source
    attention = micro_step.index("with _attention_context():")
    forward = micro_step.index("output = engine(")
    backward = micro_step.index("self.strategy.backward(loss, self.model, self.optimizer)")
    optimizer_step = micro_step.index(
        "self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)"
    )
    assert attention < forward < backward < optimizer_step
    backward_line = next(
        line for line in micro_step.splitlines() if "self.strategy.backward" in line
    )
    optimizer_line = next(
        line for line in micro_step.splitlines() if "self.strategy.optimizer_step" in line
    )
    assert len(backward_line) - len(backward_line.lstrip()) > len(optimizer_line) - len(
        optimizer_line.lstrip()
    )


@requires_torch
def test_rendered_runtime_bypasses_full_lm_head_logits(monkeypatch, tmp_path):
    import torch
    import torch.nn.functional as functional

    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps({"chunked_loss_size": 2, "force_cudnn_sdpa": False}),
        encoding="utf-8",
    )
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    runtime = ModuleType("flash_openrlhf_sft_runtime_test")
    exec(compile(render_openrlhf_sft_runtime(), runtime.__name__, "exec"), runtime.__dict__)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(17, 5)

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(
                last_hidden_state=self.embedding(input_ids),
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )

    class BaseLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = Backbone()
            self.output_head = torch.nn.Linear(5, 17)
            self.config = SimpleNamespace(logit_scale=1.0)
            self.full_forward_calls = 0

        def get_output_embeddings(self):
            return self.output_head

        def forward(self, input_ids, **_kwargs):
            self.full_forward_calls += 1
            hidden = self.base_model(input_ids=input_ids).last_hidden_state
            return SimpleNamespace(logits=self.output_head(hidden))

    class PeftModel(torch.nn.Module):
        def __init__(self, base_lm):
            super().__init__()
            self.base_lm = base_lm

        def get_base_model(self):
            return self.base_lm

        def forward(self, **kwargs):
            return self.base_lm(**kwargs)

    class Engine(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, **kwargs):
            return self.module(**kwargs)

    torch.manual_seed(31)
    base_lm = BaseLM()
    engine = Engine(PeftModel(base_lm))
    actor = SimpleNamespace(model=engine)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    loss_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    labels = input_ids.masked_fill(~loss_mask, -100)
    hidden = base_lm.base_model(input_ids=input_ids).last_hidden_state
    shifted_mask = loss_mask[:, 1:]
    expected = functional.cross_entropy(
        base_lm.output_head(hidden[:, :-1, :])[shifted_mask],
        input_ids[:, 1:][shifted_mask],
    )

    patched_engine = runtime._install_chunked_loss_forward(actor)
    output = patched_engine(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        flash_loss_denominator=shifted_mask.sum(),
        flash_dp_size=1,
    )

    assert base_lm.full_forward_calls == 0
    assert output.logits is None
    assert output.loss.item() == pytest.approx(expected.item(), abs=1e-5)
    output.loss.backward()
    assert base_lm.base_model.embedding.weight.grad is not None
    assert base_lm.output_head.weight.grad is not None


@requires_torch
def test_rendered_runtime_uses_full_vlm_forward_without_full_head_projection(monkeypatch, tmp_path):
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps({"chunked_loss_size": 2, "force_cudnn_sdpa": False}),
        encoding="utf-8",
    )
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    runtime = ModuleType("flash_openrlhf_sft_vlm_runtime_test")
    exec(compile(render_openrlhf_sft_runtime(), runtime.__name__, "exec"), runtime.__dict__)

    class OutputHead(torch.nn.Linear):
        def __init__(self):
            super().__init__(5, 17)
            self.projection_shapes = []

        def forward(self, hidden_states):
            self.projection_shapes.append(tuple(hidden_states.shape))
            return super().forward(hidden_states)

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.direct_calls = 0

        def forward(self, **_kwargs):
            self.direct_calls += 1
            raise AssertionError("VLM training must use the full outer forward")

    class BaseLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = Backbone()
            self.embedding = torch.nn.Embedding(17, 5)
            self.visual = torch.nn.Module()
            self.visual.patch_embed = torch.nn.Linear(4, 5)
            self.visual.patch_embed.requires_grad_(False)
            self.vision_block = torch.nn.Linear(5, 5)
            self.output_head = OutputHead()
            self.config = SimpleNamespace(
                image_token_id=9,
                video_token_id=10,
                text_config=SimpleNamespace(logit_scale=1.0),
            )
            self.full_forward_calls = 0
            self.seen_mm_token_type_ids = None

        def get_output_embeddings(self):
            return self.output_head

        def forward(
            self,
            input_ids,
            pixel_values=None,
            image_grid_thw=None,
            mm_token_type_ids=None,
            **_kwargs,
        ):
            del image_grid_thw
            self.full_forward_calls += 1
            hidden = self.embedding(input_ids)
            if pixel_values is not None:
                self.seen_mm_token_type_ids = mm_token_type_ids.detach().clone()
                vision_hidden = self.visual.patch_embed(pixel_values)
                vision_hidden = checkpoint(self.vision_block, vision_hidden, use_reentrant=True)
                hidden = hidden + vision_hidden.sum()
            return SimpleNamespace(logits=self.output_head(hidden))

    class PeftModel(torch.nn.Module):
        def __init__(self, base_lm):
            super().__init__()
            self.base_lm = base_lm

        def get_base_model(self):
            return self.base_lm

        def forward(self, **kwargs):
            return self.base_lm(**kwargs)

    class Engine(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, **kwargs):
            return self.module(**kwargs)

    torch.manual_seed(37)
    base_lm = BaseLM()
    engine = Engine(PeftModel(base_lm))
    actor = SimpleNamespace(model=engine, is_vlm=True)
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 9, 6, 7]])
    loss_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    labels = input_ids.masked_fill(~loss_mask, -100)
    pixel_values = torch.tensor([[0.25, 0.5, 0.75, 1.0]])
    vision_hidden = functional.linear(
        pixel_values,
        base_lm.visual.patch_embed.weight,
        base_lm.visual.patch_embed.bias,
    )
    vision_hidden = functional.linear(
        vision_hidden,
        base_lm.vision_block.weight,
        base_lm.vision_block.bias,
    )
    hidden = base_lm.embedding(input_ids) + vision_hidden.sum()
    shifted_mask = loss_mask[:, 1:]
    expected = functional.cross_entropy(
        functional.linear(hidden[:, :-1, :], base_lm.output_head.weight, base_lm.output_head.bias)[
            shifted_mask
        ],
        input_ids[:, 1:][shifted_mask],
    )

    patched_engine = runtime._install_chunked_loss_forward(actor)
    output = patched_engine(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        flash_loss_denominator=shifted_mask.sum(),
        flash_dp_size=1,
        pixel_values=pixel_values,
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    )

    assert base_lm.full_forward_calls == 1
    assert base_lm.base_model.direct_calls == 0
    assert base_lm.output_head.projection_shapes == [(2, 5), (2, 5)]
    assert base_lm.seen_mm_token_type_ids.dtype == torch.int32
    assert base_lm.seen_mm_token_type_ids.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert output.loss.item() == pytest.approx(expected.item(), abs=1e-5)
    output.loss.backward()
    assert base_lm.embedding.weight.grad is not None
    assert base_lm.vision_block.weight.grad is not None
    assert base_lm.output_head.weight.grad is not None
    assert base_lm.output_head.projection_shapes == [(2, 5)] * 4
    assert len(base_lm.visual.patch_embed._forward_hooks) == 1

    base_lm.zero_grad(set_to_none=True)
    base_lm.output_head.projection_shapes.clear()
    text_hidden = base_lm.embedding(input_ids)
    text_expected = functional.cross_entropy(
        functional.linear(
            text_hidden[:, :-1, :], base_lm.output_head.weight, base_lm.output_head.bias
        )[shifted_mask],
        input_ids[:, 1:][shifted_mask],
    )
    text_output = patched_engine(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        flash_loss_denominator=shifted_mask.sum(),
        flash_dp_size=1,
    )

    assert base_lm.full_forward_calls == 2
    assert base_lm.base_model.direct_calls == 0
    assert text_output.loss.item() == pytest.approx(text_expected.item(), abs=1e-5)
    text_output.loss.backward()
    assert base_lm.embedding.weight.grad is not None
    assert base_lm.output_head.weight.grad is not None
    assert base_lm.output_head.projection_shapes == [(2, 5)] * 4


@requires_torch
def test_chunked_selected_token_nll_is_chunk_size_invariant():
    import torch

    torch.manual_seed(23)
    hidden = torch.randn(2, 7, 4, dtype=torch.float32, requires_grad=True)
    output_head = torch.nn.Linear(4, 13, bias=False, dtype=torch.float32)
    input_ids = torch.randint(0, 13, (2, 7))
    loss_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 0, 1, 1, 0]],
        dtype=torch.bool,
    )

    losses = [
        _chunked_selected_token_nll(
            hidden,
            output_head,
            input_ids,
            loss_mask,
            chunk_size=chunk_size,
        ).detach()
        for chunk_size in (1, 4, 7)
    ]

    assert torch.allclose(torch.stack(losses), losses[0].expand(3), atol=1e-5, rtol=1e-5)


def test_32k_uses_chunked_loss_without_sequence_parallel_rejection():
    import inspect

    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    worker_source = inspect.getsource(openrlhf_mod.run_sft_openrlhf)
    runtime_source = render_openrlhf_sft_runtime()

    assert "_validate_gdn_realized_length" not in worker_source
    assert "if realized_max_length >= 32768:" in worker_source
    assert "per_device_limit = 1" in worker_source
    assert "fused=True" in worker_source
    assert "fused_ce=True" in worker_source
    assert "_chunked_selected_token_nll(" in runtime_source
    assert "labels.ne(-100)" in runtime_source
    assert "loss_mask[:, 1:].bool()" in runtime_source
    assert "return_logprobs=True" not in runtime_source


def test_required_checkpoint_publish_writes_durability_marker(monkeypatch, tmp_path):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    ds_dir = checkpoint_dir / "global_step3"
    hf_dir = checkpoint_dir / "global_step3_hf"
    ds_dir.mkdir()
    hf_dir.mkdir()
    monkeypatch.setattr(openrlhf_mod, "_export_checkpoint_adapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        openrlhf_mod._w,
        "publish_deployable_checkpoint",
        lambda *args, **kwargs: "checkpoints/step-3/adapter",
    )

    def upload_resume_checkpoint(*_args, before_upload, **_kwargs):
        before_upload()
        return True

    monkeypatch.setattr(
        openrlhf_mod._w,
        "upload_resume_checkpoint",
        upload_resume_checkpoint,
    )
    watcher = _OpenRLHFCheckpointWatcher(
        checkpoint_dir=str(checkpoint_dir),
        export_root=str(tmp_path / "exports"),
        processing_dir=str(tmp_path / "processing"),
        python_bin="python",
        model_id="org/model",
        model_revision="a" * 40,
        required_steps=(3,),
    )

    watcher._publish(3, str(ds_dir), str(hf_dir))

    assert 3 in watcher.processed_steps
    assert 3 in watcher.published_steps
    assert (checkpoint_dir / ".flash-required-step-3.done").read_text() == "durable\n"
    assert (checkpoint_dir / ".flash-upload-step-3.done").read_text() == "uploaded\n"

    def fail_resume_checkpoint(*_args, before_upload, **_kwargs):
        before_upload()
        return False

    monkeypatch.setattr(
        openrlhf_mod._w,
        "upload_resume_checkpoint",
        fail_resume_checkpoint,
    )
    retry_watcher = _OpenRLHFCheckpointWatcher(
        checkpoint_dir=str(checkpoint_dir),
        export_root=str(tmp_path / "retry-exports"),
        processing_dir=str(tmp_path / "processing"),
        python_bin="python",
        model_id="org/model",
        model_revision="a" * 40,
        required_steps=(3,),
    )
    with pytest.raises(openrlhf_mod._w.RetriableInfraError, match="full-state checkpoint"):
        retry_watcher._publish(3, str(ds_dir), str(hf_dir))


def test_final_checkpoint_retry_uses_deployable_publication_state():
    source = Path(importlib.util.find_spec("flash.engine.worker.sft_openrlhf").origin).read_text(
        encoding="utf-8"
    )

    assert "final_step not in watcher.published_steps" in source
    assert "final_step not in watcher.processed_steps" not in source


def test_checkpoint_watcher_waits_for_authoritative_zero3_export(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    ds_dir = checkpoint_dir / "global_step3"
    hf_dir = checkpoint_dir / "global_step3_hf"
    ds_dir.mkdir(parents=True)
    hf_dir.mkdir()
    (hf_dir / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (hf_dir / "adapter_model.safetensors").write_bytes(b"transient")
    watcher = _OpenRLHFCheckpointWatcher(
        checkpoint_dir=str(checkpoint_dir),
        export_root=str(tmp_path / "exports"),
        processing_dir=str(tmp_path / "processing"),
        python_bin="python",
        model_id="org/model",
        model_revision="a" * 40,
        required_steps=(),
    )

    assert watcher._completed_checkpoints() == []
    (hf_dir / "adapter_model.bin").write_bytes(b"authoritative")
    assert watcher._completed_checkpoints() == []
    (checkpoint_dir / ".flash-hf-step-3.ready").write_text("ready\n", encoding="utf-8")

    assert watcher._completed_checkpoints() == [(3, str(ds_dir), str(hf_dir))]


def test_checkpoint_retention_waits_for_upload_before_pruning(monkeypatch, tmp_path):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for step in (1, 2):
        (checkpoint_dir / f"global_step{step}").mkdir()
        (checkpoint_dir / f"global_step{step}_hf").mkdir()
        (checkpoint_dir / f".flash-hf-step-{step}.ready").write_text("ready\n", encoding="utf-8")
    saw_old_checkpoint_during_second_upload = []

    def upload(step, ds_dir, *, before_upload):
        before_upload()
        if step == 2:
            saw_old_checkpoint_during_second_upload.append(
                (checkpoint_dir / "global_step1").is_dir()
            )
        return True

    monkeypatch.setattr(openrlhf_mod, "_export_checkpoint_adapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        openrlhf_mod._w, "publish_deployable_checkpoint", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(openrlhf_mod._w, "upload_resume_checkpoint", upload)
    watcher = _OpenRLHFCheckpointWatcher(
        checkpoint_dir=str(checkpoint_dir),
        export_root=str(tmp_path / "exports"),
        processing_dir=str(tmp_path / "processing"),
        python_bin="python",
        model_id="org/model",
        model_revision="a" * 40,
        required_steps=(),
        max_num_checkpoints=1,
    )

    watcher._publish(
        1, str(checkpoint_dir / "global_step1"), str(checkpoint_dir / "global_step1_hf")
    )
    watcher._publish(
        2, str(checkpoint_dir / "global_step2"), str(checkpoint_dir / "global_step2_hf")
    )

    assert saw_old_checkpoint_during_second_upload == [True]
    assert not (checkpoint_dir / "global_step1").exists()
    assert (checkpoint_dir / "global_step2").is_dir()


def test_checkpoint_watcher_failed_periodic_upload_does_not_hang(monkeypatch, tmp_path):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    checkpoint_dir = tmp_path / "checkpoints"
    ds_dir = checkpoint_dir / "global_step2"
    hf_dir = checkpoint_dir / "global_step2_hf"
    ds_dir.mkdir(parents=True)
    hf_dir.mkdir()
    (hf_dir / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (hf_dir / "adapter_model.bin").write_bytes(b"authoritative")
    (checkpoint_dir / ".flash-hf-step-2.ready").write_text("ready\n", encoding="utf-8")
    monkeypatch.setattr(openrlhf_mod, "_export_checkpoint_adapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(openrlhf_mod._w, "upload_resume_checkpoint", lambda *args, **kwargs: False)
    watcher = _OpenRLHFCheckpointWatcher(
        checkpoint_dir=str(checkpoint_dir),
        export_root=str(tmp_path / "exports"),
        processing_dir=str(tmp_path / "processing"),
        python_bin="python",
        model_id="org/model",
        model_revision="a" * 40,
        required_steps=(),
        max_num_checkpoints=1,
    )

    watcher.start()
    watcher.stop(require_complete=False)

    assert not watcher._thread.is_alive()
    assert 2 in watcher.processed_steps
    assert 2 not in watcher.published_steps
    assert (checkpoint_dir / ".flash-upload-step-2.failed").is_file()
    assert ds_dir.is_dir()
    assert hf_dir.is_dir()


def test_checkpoint_watcher_preserves_retriable_failure(tmp_path):
    import flash.engine.worker.sft_openrlhf as openrlhf_mod

    watcher = _OpenRLHFCheckpointWatcher(
        checkpoint_dir=str(tmp_path),
        export_root=str(tmp_path / "exports"),
        processing_dir=str(tmp_path / "processing"),
        python_bin="python",
        model_id="org/model",
        model_revision="a" * 40,
        required_steps=(3,),
    )
    error = openrlhf_mod._w.RetriableInfraError("transient checkpoint upload")
    watcher._error = error

    with pytest.raises(openrlhf_mod._w.RetriableInfraError) as raised:
        watcher.raise_if_failed()

    assert raised.value is error


def test_checkpoint_retention_prunes_matching_hf_sidecar(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for step in (1, 2):
        (checkpoint_dir / f"global_step{step}").mkdir()
        (checkpoint_dir / f"global_step{step}_hf").mkdir()
        (checkpoint_dir / f".flash-hf-step-{step}.ready").write_text("ready\n", encoding="utf-8")
        (checkpoint_dir / f".flash-upload-step-{step}.done").write_text(
            "uploaded\n", encoding="utf-8"
        )
    watcher = _OpenRLHFCheckpointWatcher(
        checkpoint_dir=str(checkpoint_dir),
        export_root=str(tmp_path / "exports"),
        processing_dir=str(tmp_path / "processing"),
        python_bin="python",
        model_id="org/model",
        model_revision="a" * 40,
        required_steps=(),
        max_num_checkpoints=1,
    )
    watcher.processed_steps.update({1, 2})

    watcher._prune_uploaded_checkpoints()

    assert not (checkpoint_dir / "global_step1_hf").exists()
    assert not (checkpoint_dir / ".flash-hf-step-1.ready").exists()
    assert not (checkpoint_dir / ".flash-upload-step-1.done").exists()
    assert (checkpoint_dir / "global_step2_hf").is_dir()


def test_checkpoint_watcher_stop_does_not_impose_upload_timeout(tmp_path):
    watcher = _OpenRLHFCheckpointWatcher(
        checkpoint_dir=str(tmp_path),
        export_root=str(tmp_path / "exports"),
        processing_dir=str(tmp_path / "processing"),
        python_bin="python",
        model_id="org/model",
        model_revision="a" * 40,
        required_steps=(),
    )
    join_calls = []
    watcher._thread = SimpleNamespace(
        join=lambda *args, **kwargs: join_calls.append((args, kwargs))
    )

    watcher.stop(require_complete=False)

    assert join_calls == [((), {})]


def test_openrlhf_child_env_excludes_training_and_provider_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setenv("FIREWORKS_API_KEY", "teacher-secret")
    monkeypatch.setenv("RUNPOD_API_KEY", "provider-secret")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "internal-secret")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("FLA_TILELANG", "0")

    child = build_openrlhf_sft_child_env(shim_dir=str(tmp_path), wandb_enabled=False)

    assert child["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert child["FLA_TILELANG"] == "0"
    assert "HF_TOKEN" not in child
    assert "FIREWORKS_API_KEY" not in child
    assert "RUNPOD_API_KEY" not in child
    assert "FREESOLO_INTERNAL_KEY" not in child
    assert "WANDB_API_KEY" not in child
    assert child["FLASH_OPENRLHF_SFT_CONFIG"] == str(tmp_path / "flash_sft_runtime.json")


def test_openrlhf_child_env_forwards_only_wandb_secrets_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_API_KEY", "wandb-secret")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("HF_TOKEN", "hf-secret")

    child = build_openrlhf_sft_child_env(shim_dir=str(tmp_path), wandb_enabled=True)

    assert child["WANDB_API_KEY"] == "wandb-secret"
    assert child["WANDB_MODE"] == "offline"
    assert "HF_TOKEN" not in child


def test_openrlhf_runtime_installs_fail_loud_loraplus_and_warmstart_checks():
    source = render_openrlhf_sft_runtime()

    compile(source, "flash_openrlhf_sft_runtime.py", "exec")
    assert "create_loraplus_optimizer" in source
    assert "PagedAdamW8bit" in source
    assert "zero_allow_untested_optimizer" in source
    assert "assert_adapter_load_clean" not in source
    assert "assert_adapter_delta_nonzero" in source
    assert "FLASH_OPENRLHF_LORAPLUS_READY" in source
    assert "loss_mask[:, 1:]" in source
    assert "loss_mask[:, :-1]" not in source
    assert 'kwargs.pop("pin_memory", None)' in source
    assert "non_blocking=True" in source
    assert "_install_attention_patch" not in source
    assert "_mark_hf_export_ready(args.ckpt.path, global_step)" in source
    assert "_wait_for_checkpoint_upload(args.ckpt.path, global_step)" in source
    assert "2**31 - 1" in source
    assert 'float("inf")' in source
    assert "falling back" not in source.lower()


def test_openrlhf_runtime_reapplies_blackwell_fla_safety(monkeypatch, tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    monkeypatch.delenv("FLA_TILELANG", raising=False)
    namespace = {"__name__": "flash_openrlhf_sft_blackwell_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda: (10, 0),
    )
    fla = ModuleType("fla")
    fla.__path__ = []
    fla.__spec__ = importlib.util.spec_from_loader("fla", loader=None, is_package=True)
    ops = ModuleType("fla.ops")
    ops.__path__ = []
    gated_delta_rule = ModuleType("fla.ops.gated_delta_rule")
    tuner = SimpleNamespace(
        configs=[
            SimpleNamespace(num_warps=4, num_stages=3),
            SimpleNamespace(num_warps=2, num_stages=4),
        ]
    )
    wy_fast = ModuleType("fla.ops.gated_delta_rule.wy_fast")
    wy_fast.prepare_wy_repr_bwd_kernel = tuner
    gated_delta_rule.wy_fast = wy_fast
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "fla", fla)
    monkeypatch.setitem(sys.modules, "fla.ops", ops)
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", gated_delta_rule)
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule.wy_fast", wy_fast)

    namespace["_apply_blackwell_fla_safety"]()

    assert os.environ["FLA_TILELANG"] == "0"
    assert tuner.configs == [SimpleNamespace(num_warps=2, num_stages=4)]


def test_openrlhf_runtime_disables_unpatchable_blackwell_fla(monkeypatch, tmp_path):
    import fcntl
    import shutil

    config_path = tmp_path / "runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_blackwell_fallback_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda: (10, 0),
    )
    fla_dir = tmp_path / "packages" / "fla"
    fla_dir.mkdir(parents=True)
    fla = ModuleType("fla")
    fla.__path__ = [str(fla_dir)]
    fla.__spec__ = importlib.util.spec_from_loader("fla", loader=None, is_package=True)
    fla.__spec__.submodule_search_locations = [str(fla_dir)]
    ops = ModuleType("fla.ops")
    ops.__path__ = []
    gated_delta_rule = ModuleType("fla.ops.gated_delta_rule")
    wy_fast = ModuleType("fla.ops.gated_delta_rule.wy_fast")
    wy_fast.prepare_wy_repr_bwd_kernel = SimpleNamespace(configs=[])
    gated_delta_rule.wy_fast = wy_fast
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "fla", fla)
    monkeypatch.setitem(sys.modules, "fla.ops", ops)
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", gated_delta_rule)
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule.wy_fast", wy_fast)
    specs = iter([fla.__spec__, fla.__spec__, None, None])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: next(specs))
    removed = []
    lock_operations = []
    monkeypatch.setattr(shutil, "rmtree", lambda path: removed.append(path))
    monkeypatch.setattr(
        fcntl, "flock", lambda lock_file, operation: lock_operations.append(operation)
    )

    namespace["_apply_blackwell_fla_safety"]()

    assert lock_operations == [fcntl.LOCK_EX]
    assert removed == [str(fla_dir)]
    assert not any(name == "fla" or name.startswith("fla.") for name in sys.modules)


def test_runtime_backpressures_only_required_checkpoints_until_upload_finishes():
    source = render_openrlhf_sft_runtime()
    save_block = source.split("if save_due:", 1)[1].split("consumed_in_epoch = 0", 1)[0]

    save_model = save_block.index("self.strategy.save_model(")
    export_ready = save_block.index("_mark_hf_export_ready(args.ckpt.path, global_step)")
    required_gate = save_block.index("if global_step in required:")
    upload_wait = save_block.index("_wait_for_checkpoint_upload(args.ckpt.path, global_step)")

    assert save_model < export_ready < required_gate < upload_wait


def test_rendered_warmstart_actor_patch_has_no_flash_child_import(monkeypatch, tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "lora_rank": 16,
                "model_id": "org/model",
                "model_revision": "a" * 40,
                "warmstart_adapter": "/adapter",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    source = render_openrlhf_sft_runtime()
    namespace = {"__name__": "flash_openrlhf_sft_warmstart_test"}
    exec(compile(source, "runtime.py", "exec"), namespace)

    class Weight:
        def detach(self):
            return self

        def ne(self, value):
            assert value == 0
            return self

        def any(self):
            return True

    class AdapterModel:
        def __init__(self):
            self.input_grads_enabled = False
            self.peft_config = {
                "default": SimpleNamespace(
                    base_model_name_or_path="org/model",
                    revision="main",
                )
            }

        def enable_input_require_grads(self):
            self.input_grads_enabled = True

        def named_modules(self):
            return [
                ("layer.lora_A.default", object()),
                ("layer.lora_B.default", SimpleNamespace(weight=Weight())),
            ]

        def load_adapter(self, *args, **kwargs):
            raise AssertionError("warm-start adapter must load exactly once")

    class OriginalActor:
        def __init__(self, pretrain_or_model, *args, lora_rank=0, **kwargs):
            del pretrain_or_model, args, lora_rank, kwargs
            self.model = SimpleNamespace(_checkpoint_conversion_mapping={"old": "new"})

    class PeftModel:
        @staticmethod
        def from_pretrained(base, warmstart, *, is_trainable, key_mapping):
            assert base is not None
            assert warmstart == "/adapter"
            assert is_trainable is True
            assert key_mapping == {"old": "new"}
            return AdapterModel()

    openrlhf = ModuleType("openrlhf")
    openrlhf.__path__ = []
    models = ModuleType("openrlhf.models")
    models.__path__ = []
    models.Actor = OriginalActor
    actor = ModuleType("openrlhf.models.actor")
    actor.Actor = OriginalActor
    peft = ModuleType("peft")
    peft.PeftModel = PeftModel
    openrlhf.models = models
    monkeypatch.setitem(sys.modules, "openrlhf", openrlhf)
    monkeypatch.setitem(sys.modules, "openrlhf.models", models)
    monkeypatch.setitem(sys.modules, "openrlhf.models.actor", actor)
    monkeypatch.setitem(sys.modules, "peft", peft)

    namespace["_install_warmstart_actor_patch"]()
    patched = models.Actor("base", lora_rank=16)

    assert isinstance(patched.model, AdapterModel)
    assert patched.model.input_grads_enabled is True
    assert patched.model.peft_config["default"].base_model_name_or_path == "org/model"
    assert patched.model.peft_config["default"].revision == "a" * 40
    assert "from flash.engine.worker.lora import" not in source


def test_short_accumulation_window_uses_realized_loss_norm_and_deepspeed_scale(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_accumulation_test"}
    source = render_openrlhf_sft_runtime()
    exec(compile(source, "runtime.py", "exec"), namespace)

    assert namespace["_accumulation_window_scale"](4, 2) == 2.0
    assert "masks, dp_group, dp_size, window_size" in source
    assert '"gpt_loss": step_loss / window_size' in source


def test_dataloader_patch_overrides_positional_pin_memory_without_duplication(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_dataloader_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)

    class FakeDeepspeedStrategy:
        def setup_dataloader(self, *args, **kwargs):
            return args, kwargs

        def prepare(self, *args):
            return args

        def load_ckpt(self, *args, **kwargs):
            return None, {}

    openrlhf = ModuleType("openrlhf")
    openrlhf.__path__ = []
    utils = ModuleType("openrlhf.utils")
    utils.__path__ = []
    deepspeed_package = ModuleType("openrlhf.utils.deepspeed")
    deepspeed_package.__path__ = []
    deepspeed_module = ModuleType("openrlhf.utils.deepspeed.deepspeed")
    deepspeed_module.DeepspeedStrategy = FakeDeepspeedStrategy
    monkeypatch.setitem(sys.modules, "openrlhf", openrlhf)
    monkeypatch.setitem(sys.modules, "openrlhf.utils", utils)
    monkeypatch.setitem(sys.modules, "openrlhf.utils.deepspeed", deepspeed_package)
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.utils.deepspeed.deepspeed",
        deepspeed_module,
    )

    namespace["_install_dataloader_and_scheduler_patches"]()
    positional_args, positional_kwargs = FakeDeepspeedStrategy().setup_dataloader(
        "dataset", 2, False, True
    )
    keyword_args, keyword_kwargs = FakeDeepspeedStrategy().setup_dataloader(
        "dataset", 2, pin_memory=False
    )

    assert positional_args == ("dataset", 2, True, True)
    assert positional_kwargs == {"drop_last": False}
    assert keyword_args == ("dataset", 2)
    assert keyword_kwargs == {"drop_last": False, "pin_memory": True}


def test_restored_client_state_reaches_runtime_trainer_state(monkeypatch, tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({"resume_step": 7}), encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_resume_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)

    states = {
        "consumed_samples": 31,
        "global_step": 7,
        "loss_curve": [2.5, 1.75],
        "token_count": 4096,
    }

    class FakeDeepspeedStrategy:
        def setup_dataloader(self, *args, **kwargs):
            return args, kwargs

        def prepare(self, *args):
            return args

        def load_ckpt(self, *args, **kwargs):
            return "/checkpoints/global_step7", states

    openrlhf = ModuleType("openrlhf")
    openrlhf.__path__ = []
    utils = ModuleType("openrlhf.utils")
    utils.__path__ = []
    deepspeed_package = ModuleType("openrlhf.utils.deepspeed")
    deepspeed_package.__path__ = []
    deepspeed_module = ModuleType("openrlhf.utils.deepspeed.deepspeed")
    deepspeed_module.DeepspeedStrategy = FakeDeepspeedStrategy
    monkeypatch.setitem(sys.modules, "openrlhf", openrlhf)
    monkeypatch.setitem(sys.modules, "openrlhf.utils", utils)
    monkeypatch.setitem(sys.modules, "openrlhf.utils.deepspeed", deepspeed_package)
    monkeypatch.setitem(
        sys.modules,
        "openrlhf.utils.deepspeed.deepspeed",
        deepspeed_module,
    )

    namespace["_install_dataloader_and_scheduler_patches"]()
    load_path, _loaded_states = FakeDeepspeedStrategy().load_ckpt(object(), "/checkpoints")
    trainer_state = namespace["_resume_training_state"](0)

    assert load_path == "/checkpoints/global_step7"
    assert namespace["CONFIG"]["_resume_states"] == states
    assert trainer_state == (7, 31, [2.5, 1.75], 4096)


@requires_torch
def test_torchrun_sitecustomize_patches_dataset_before_train_sft_import(tmp_path):
    shim_dir = tmp_path / "shim"
    package_dir = tmp_path / "packages"
    shim_dir.mkdir()
    (package_dir / "openrlhf" / "cli").mkdir(parents=True)
    (package_dir / "openrlhf" / "datasets").mkdir(parents=True)
    (package_dir / "openrlhf" / "models").mkdir(parents=True)
    (package_dir / "openrlhf" / "trainer").mkdir(parents=True)
    (package_dir / "openrlhf" / "utils" / "deepspeed").mkdir(parents=True)
    (package_dir / "peft").mkdir(parents=True)

    config_path = shim_dir / "flash_sft_runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    (shim_dir / "flash_openrlhf_sft_runtime.py").write_text(
        render_openrlhf_sft_runtime(),
        encoding="utf-8",
    )
    (shim_dir / "sitecustomize.py").write_text(
        "from flash_openrlhf_sft_runtime import apply_flash_openrlhf_sft_patches\n"
        "apply_flash_openrlhf_sft_patches()\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "openrlhf" / "cli" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "openrlhf" / "datasets" / "__init__.py").write_text(
        "class SFTDataset:\n    pass\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "cli" / "train_sft.py").write_text(
        "from openrlhf.datasets import SFTDataset\n"
        "assert SFTDataset.__name__ == 'FlashTokenizedSFTDataset'\n"
        "print('FLASH_DATASET_PATCHED_BEFORE_TRAIN_SFT', flush=True)\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "models" / "__init__.py").write_text(
        "class SFTLoss:\n    pass\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "trainer" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "openrlhf" / "trainer" / "sft_trainer.py").write_text(
        "class SFTTrainer:\n    pass\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "openrlhf" / "utils" / "deepspeed" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "utils" / "deepspeed" / "deepspeed.py").write_text(
        "class DeepspeedStrategy:\n"
        "    def setup_dataloader(self, *args, **kwargs):\n        pass\n"
        "    def prepare(self, *args):\n        pass\n"
        "    def load_ckpt(self, *args, **kwargs):\n        return None, {}\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "utils" / "distributed_sampler.py").write_text(
        "class DistributedSampler:\n    pass\n",
        encoding="utf-8",
    )
    (package_dir / "openrlhf" / "utils" / "loss_utils.py").write_text(
        "def _optimizer_step_loss_norm(*args, **kwargs):\n    return {}\n",
        encoding="utf-8",
    )
    (package_dir / "deepspeed.py").write_text(
        "def initialize(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (package_dir / "peft" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "peft" / "optimizers.py").write_text(
        "def create_loraplus_optimizer(*args, **kwargs):\n    return None\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["FLASH_OPENRLHF_SFT_CONFIG"] = str(config_path)
    project_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(shim_dir), str(package_dir), str(project_root)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=1",
            "-m",
            "openrlhf.cli.train_sft",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FLASH_DATASET_PATCHED_BEFORE_TRAIN_SFT" in result.stdout


def test_gpu_probe_uses_configured_openrlhf_interpreter(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=(
                'FLASH_GPU_PROBE={"name":"gpu","memory_gb":80.0,'
                '"capability":[9,0],"fa2_available":true,"fa3_available":true}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = _probe_gpu_in_subprocess(
        "/opt/openrlhf/bin/python",
        "H100",
        required_gpu_count=2,
    )

    assert calls[0][0][0] == "/opt/openrlhf/bin/python"
    assert json.loads(calls[0][0][3]) == ["H100", "", 2]
    assert "torch.cuda.device_count()" in calls[0][0][2]
    assert "for index in range(required_count)" in calls[0][0][2]
    assert result["capability"] == [9, 0]


def test_child_output_cuda_oom_detection_uses_explicit_child_markers():
    assert _child_output_is_cuda_oom("torch.OutOfMemoryError: CUDA out of memory")
    assert _child_output_is_cuda_oom("torch.cuda.OutOfMemoryError: allocation failed")
    assert not _child_output_is_cuda_oom("OpenRLHF SFT subprocess exited with status 1")


def test_attention_implementation_matches_gpu_capability():
    assert (
        _attention_implementation(
            {"capability": [9, 0], "fa2_available": True, "fa3_available": True}
        )
        == "flash_attention_3"
    )
    assert (
        _attention_implementation(
            {"capability": [8, 9], "fa2_available": True, "fa3_available": False}
        )
        == "flash_attention_2"
    )
    assert _attention_implementation({"capability": [12, 0]}) == "sdpa"


def test_training_batch_shape_respects_gpu_and_device_limits():
    micro, accumulation, global_batch = _training_batch_shape(
        row_count=100,
        effective_batch=32,
        per_device_limit=4,
        gpu_count=2,
    )

    assert (micro, accumulation, global_batch) == (4, 4, 32)


@requires_torch
def test_rendered_dataset_returns_exact_input_ids_and_loss_mask(monkeypatch, tmp_path):
    config_path = tmp_path / "runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_runtime_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)
    dataset_class = namespace["FlashTokenizedSFTDataset"]
    row = {"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 1, 1], "multimodal_inputs": b""}
    tokenizer = SimpleNamespace(pad_token_id=0)
    dataset = dataset_class([row], tokenizer, 8, strategy=object())

    input_ids, attention, loss_mask, mm_inputs = dataset[0]

    assert input_ids.tolist() == [[1, 2, 3, 4]]
    assert attention.tolist() == [[1, 1, 1, 1]]
    assert loss_mask.tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert mm_inputs == {}


@requires_torch
def test_rendered_dataset_collates_multimodal_tensors(tmp_path, monkeypatch):
    import numpy as np

    config_path = tmp_path / "runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_multimodal_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)
    dataset_class = namespace["FlashTokenizedSFTDataset"]
    rows = [
        {
            "input_ids": [1, 2, 3],
            "loss_mask": [0, 1, 1],
            "multimodal_inputs": _serialize_multimodal_inputs(
                {
                    "pixel_values": np.array([[1.0, 2.0]], dtype=np.float32),
                    "image_grid_thw": np.array([[1, 2, 3]], dtype=np.int64),
                }
            ),
        },
        {
            "input_ids": [4, 5],
            "loss_mask": [0, 1],
            "multimodal_inputs": _serialize_multimodal_inputs(
                {
                    "pixel_values": np.array([[3.0, 4.0]], dtype=np.float32),
                    "image_grid_thw": np.array([[4, 5, 6]], dtype=np.int64),
                }
            ),
        },
    ]
    tokenizer = SimpleNamespace(pad_token_id=0)
    dataset = dataset_class(rows, tokenizer, 8, strategy=object())

    inputs, attention, loss_mask, mm_inputs = dataset.collate_fn([dataset[0], dataset[1]])

    assert inputs.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert attention.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert loss_mask.tolist() == [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
    assert mm_inputs["pixel_values"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert mm_inputs["image_grid_thw"].tolist() == [[1, 2, 3], [4, 5, 6]]


@requires_torch
def test_rendered_dataset_collates_mixed_text_and_image_rows(tmp_path, monkeypatch):
    import numpy as np

    config_path = tmp_path / "runtime.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FLASH_OPENRLHF_SFT_CONFIG", str(config_path))
    namespace = {"__name__": "flash_openrlhf_sft_mixed_multimodal_test"}
    exec(compile(render_openrlhf_sft_runtime(), "runtime.py", "exec"), namespace)
    dataset_class = namespace["FlashTokenizedSFTDataset"]
    rows = [
        {"input_ids": [1, 2], "loss_mask": [0, 1], "multimodal_inputs": b""},
        {
            "input_ids": [3, 4, 5],
            "loss_mask": [0, 1, 1],
            "multimodal_inputs": _serialize_multimodal_inputs(
                {
                    "pixel_values": np.array([[1.0, 2.0]], dtype=np.float32),
                    "image_grid_thw": np.array([[1, 2, 3]], dtype=np.int64),
                }
            ),
        },
        {"input_ids": [6], "loss_mask": [1], "multimodal_inputs": b""},
    ]
    dataset = dataset_class(rows, SimpleNamespace(pad_token_id=0), 8, strategy=object())

    inputs, attention, loss_mask, mm_inputs = dataset.collate_fn(
        [dataset[0], dataset[1], dataset[2]]
    )

    assert inputs.tolist() == [[1, 2, 0], [3, 4, 5], [6, 0, 0]]
    assert attention.tolist() == [[1, 1, 0], [1, 1, 1], [1, 0, 0]]
    assert loss_mask.tolist() == [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
    assert mm_inputs["pixel_values"].tolist() == [[1.0, 2.0]]
    assert mm_inputs["image_grid_thw"].tolist() == [[1, 2, 3]]
