from __future__ import annotations

import copy

import pytest


def _dense_config():
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )


def _moe_config():
    from transformers import Qwen3MoeConfig

    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        output_router_logits=True,
        router_aux_loss_coef=0.001,
    )


def _vl_config(config_cls, *, moe: bool):
    text = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 2,
        "full_attention_interval": 2,
    }
    if moe:
        text.update(
            {
                "moe_intermediate_size": 16,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "output_router_logits": True,
                "router_aux_loss_coef": 0.001,
            }
        )
    vision = {
        "depth": 1,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    }
    return config_cls(text_config=text, vision_config=vision)


def _chunked_copy(model, *, is_vlm: bool = False):
    from trl.trainer.sft_trainer import _patch_chunked_ce_lm_head

    chunked = copy.deepcopy(model)
    _patch_chunked_ce_lm_head(chunked, chunk_size=2, is_vlm=is_vlm)
    return chunked


@pytest.mark.parametrize("packed", [False, True])
def test_chunked_nll_matches_plain_nll_dense_and_masked_packed(packed):
    torch = pytest.importorskip("torch")
    from transformers import Qwen3ForCausalLM

    from flash.engine.worker.packing import BlockDiagonalCollator, pack_token_ids

    torch.manual_seed(0)
    plain = Qwen3ForCausalLM(_dense_config()).eval()
    chunked = _chunked_copy(plain).eval()

    if packed:
        examples = [[5, 6, 7, 8], [9, 10, 11]]
        completion_masks = [[0, 0, 1, 1], [0, 1, 1]]
        rows = pack_token_ids(examples, 16, completion_masks=completion_masks)
        batch = BlockDiagonalCollator(pad_token_id=0, pad_to_multiple_of=1)(rows)
        assert batch["labels"][0].tolist() == [-100, -100, 7, 8, -100, 10, 11]
    else:
        input_ids = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 0]])
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            "labels": torch.tensor([[-100, -100, 7, 8], [-100, 10, 11, -100]]),
        }

    with torch.no_grad():
        plain_out = plain(**batch)
        chunked_out = chunked(**batch)

    expected_valid = (batch["labels"][..., 1:] != -100).sum()
    assert chunked_out.loss.item() == pytest.approx(plain_out.loss.item(), abs=1e-6)
    assert chunked_out.logits is None
    assert chunked_out.num_valid_tokens.item() == expected_valid.item()
    assert chunked_out.num_correct_tokens.item() <= expected_valid.item()


def test_chunked_nll_trainer_preserves_token_count_metrics(tmp_path):
    torch = pytest.importorskip("torch")
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, Qwen3ForCausalLM
    from trl import SFTConfig, SFTTrainer

    raw_tokenizer = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "a": 2}, unk_token="a"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer, pad_token="<pad>", eos_token="<eos>"
    )
    dataset = Dataset.from_list([{"input_ids": [5, 6, 7, 8], "completion_mask": [0, 0, 1, 1]}])
    trainer = SFTTrainer(
        model=Qwen3ForCausalLM(_dense_config()),
        args=SFTConfig(
            output_dir=str(tmp_path),
            loss_type="chunked_nll",
            completion_only_loss=True,
            remove_unused_columns=False,
            use_cpu=True,
            bf16=False,
            report_to="none",
            max_length=8,
            per_device_train_batch_size=1,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    batch = trainer.data_collator([dataset[0]])
    trainer.model.train()
    loss = trainer.compute_loss(trainer.model, batch)

    assert torch.isfinite(loss)
    assert trainer._total_train_tokens == 4
    assert trainer._metrics["train"]["num_tokens"] == [4]
    assert len(trainer._metrics["train"]["entropy"]) == 1
    assert len(trainer._metrics["train"]["mean_token_accuracy"]) == 1


def test_chunked_nll_preserves_moe_router_aux_loss():
    torch = pytest.importorskip("torch")
    from transformers import Qwen3MoeForCausalLM

    torch.manual_seed(0)
    plain = Qwen3MoeForCausalLM(_moe_config()).eval()
    chunked = _chunked_copy(plain).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    labels = input_ids.clone()
    labels[0, :2] = -100
    labels[1, 0] = -100
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        plain_out = plain(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        chunked_out = chunked(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert chunked_out.loss.item() == pytest.approx(plain_out.loss.item(), abs=1e-6)
    assert chunked_out.aux_loss.item() == pytest.approx(plain_out.aux_loss.item(), abs=1e-6)


def test_qwen_structures_keep_text_classification_and_safe_output_head():
    pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model
    from peft.tuners.tuners_utils import BaseTunerLayer
    from transformers import (
        PreTrainedTokenizerBase,
        Qwen3_5Config,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeConfig,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3ForCausalLM,
    )

    from flash.engine.worker.sft import _prepare_chunked_nll_model

    tokenizer = PreTrainedTokenizerBase()
    models = [
        Qwen3ForCausalLM(_dense_config()),
        Qwen3_5MoeForConditionalGeneration(_vl_config(Qwen3_5MoeConfig, moe=True)),
        Qwen3_5ForConditionalGeneration(_vl_config(Qwen3_5Config, moe=False)),
    ]
    assert models[2].config.text_config.layer_types == ["linear_attention", "full_attention"]

    for model in models:
        peft_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        _prepare_chunked_nll_model(model, tokenizer, peft_config)
        assert model.base_model is not model
        peft_model = get_peft_model(model, peft_config)
        output_head = peft_model.get_base_model().get_output_embeddings()
        assert not isinstance(output_head, BaseTunerLayer)
        assert "lm_head" not in peft_config.target_modules

    moe_config = models[1].config
    assert moe_config.output_router_logits is True
    assert moe_config.num_experts == moe_config.text_config.num_experts


def test_chunked_nll_rejects_trainable_output_head():
    from peft import LoraConfig
    from transformers import PreTrainedTokenizerBase, Qwen3ForCausalLM

    from flash.engine.worker.sft import _prepare_chunked_nll_model

    config = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules="all-linear",
        modules_to_save=["lm_head"],
        task_type="CAUSAL_LM",
    )
    with pytest.raises(RuntimeError, match="output layer excluded"):
        _prepare_chunked_nll_model(
            Qwen3ForCausalLM(_dense_config()), PreTrainedTokenizerBase(), config
        )
