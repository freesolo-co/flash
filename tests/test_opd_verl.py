"""CPU contracts for the OPD migration to verl 0.8.0."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.engine.worker.opd import _gkd_loss_from_logps
from flash.engine.worker.opd_verl import (
    _BridgePrompt,
    _build_opd_child_env,
    _OpdProgressState,
    _OpdVerlCheckpointWatcher,
    _processed_resume_steps,
    _processor_expanded_prompt_ids,
    _prompt_pool_fingerprint,
    _raise_verl_failure,
    _render_opd_sitecustomize,
    _restore_verl_resume,
    _stage_retry_contract,
    _TeacherAlignmentBridge,
    _trim_response_and_forced,
    _validate_forced_mask,
    _write_opd_parquet,
    build_opd_verl_overrides,
    encode_shifted_group_metadata,
)
from flash.engine.worker.opd_verl_plugin import (
    FlashTeacherBridgeError,
    _AllNoSignalBatch,
    _bridge_score_payload,
    _flash_groupwise_reverse_kl_values,
    _full_sequence_signal_sequences,
    _multi_modal_image_count,
    _post_json,
    _raw_prompt_has_image_block,
    _require_structured_runtime_versions,
    _resolve_image_token_id,
    _run_with_no_signal_replacements,
    _set_current_global_batch_info,
    _signal_sequences,
    deterministic_rollout_seed,
)
from flash.engine.worker.opd_verl_structured import (
    StructuredOutputReplay,
    _count_legal_tokens,
    canonical_structured_spec,
)
from flash.engine.worker.tokenizer_align import TeacherToken
from flash.opd_verl_validation import validate_opd_verl_structured_outputs


def _load_verl_rl_dataset(monkeypatch):
    test_root = os.environ.get("FLASH_VERL_TEST_ROOT", "").strip()
    if test_root:
        target = Path(test_root).resolve()
        root = target / "verl"
        monkeypatch.syspath_prepend(str(target))
    else:
        package_spec = importlib.util.find_spec("verl")
        if package_spec is None or not package_spec.submodule_search_locations:
            pytest.skip("verl 0.8.0 is not installed")
        root = Path(next(iter(package_spec.submodule_search_locations)))
    for name, path in {
        "verl": root,
        "verl.utils": root / "utils",
        "verl.utils.dataset": root / "utils" / "dataset",
    }.items():
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    try:
        import omegaconf  # noqa: F401
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "omegaconf",
            types.SimpleNamespace(DictConfig=dict, ListConfig=list),
        )
    module_spec = importlib.util.spec_from_file_location(
        "verl.utils.dataset.rl_dataset", root / "utils" / "dataset" / "rl_dataset.py"
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, module_spec.name, module)
    module_spec.loader.exec_module(module)
    return module


def _aggregate_seq_mean_token_mean(values, response_mask):
    counts = response_mask.sum(dim=-1)
    sequence_losses = (values * response_mask).sum(dim=-1) / counts
    return sequence_losses.mean()


def test_groupwise_reverse_kl_scalar_and_analytic_gradient_match_legacy():
    torch = pytest.importorskip("torch")
    student = torch.tensor(
        [[-0.4, -0.8, -1.2, -2.0], [-0.3, -0.9, -1.1, -1.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher_logsums = torch.tensor(
        [[-0.6, -2.3, -2.3, 0.0], [-0.5, -1.4, -1.4, 0.0]],
        dtype=torch.float64,
    )
    group_ids = torch.tensor([[0, 1, 1, -1], [0, 1, 1, -1]])
    response_mask = torch.ones_like(student, dtype=torch.bool)
    coef = 0.7

    values = _flash_groupwise_reverse_kl_values(
        student, teacher_logsums, group_ids, response_mask, coef
    )
    verl_loss = _aggregate_seq_mean_token_mean(values, response_mask)
    verl_gradient = torch.autograd.grad(verl_loss, student, retain_graph=True)[0]

    legacy_losses = [
        _gkd_loss_from_logps(
            student[row],
            [([0], float(teacher_logsums[row, 0])), ([1, 2], float(teacher_logsums[row, 1]))],
            kl_coef=coef,
        )
        for row in range(2)
    ]
    legacy_loss = torch.stack(legacy_losses).mean()
    legacy_gradient = torch.autograd.grad(legacy_loss, student)[0]

    assert torch.allclose(verl_loss, legacy_loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(verl_gradient, legacy_gradient, atol=1e-12, rtol=1e-12)
    assert torch.equal(verl_gradient[:, 3], torch.zeros(2, dtype=torch.float64))


def test_dynamic_microbatches_refresh_global_batch_info_and_match_flash_sequence_mean():
    torch = pytest.importorskip("torch")
    steps = [
        {
            "student": [
                [-0.4, -0.8, -1.2, -2.0],
                [-0.3, -0.9, -1.7, -2.1],
                [-0.6, -1.4, -2.2, -2.8],
            ],
            "teacher": [
                [-0.6, -2.3, -2.3, 0.0],
                [-0.5, -1.5, 0.0, 0.0],
                [-0.9, 0.0, 0.0, 0.0],
            ],
            "groups": [[0, 1, 1, -1], [0, 1, -1, -1], [0, -1, -1, -1]],
            "mask": [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]],
            "splits": [(0, 2), (2, 3)],
        },
        {
            "student": [[-0.2, -0.7, -1.3, -1.9], [-0.5, -1.0, -1.6, -2.4]],
            "teacher": [[-0.4, -1.8, -1.8, 0.0], [-0.8, -1.1, 0.0, 0.0]],
            "groups": [[0, 1, 1, -1], [0, 1, -1, -1]],
            "mask": [[1, 1, 1, 1], [1, 1, 1, 0]],
            "splits": [(0, 1), (1, 2)],
        },
    ]
    coef = 0.7
    observed = []
    expected = []

    for step in steps:
        student = torch.tensor(step["student"], dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor(step["teacher"], dtype=torch.float64)
        group_ids = torch.tensor(step["groups"])
        response_mask = torch.tensor(step["mask"], dtype=torch.bool)
        config = SimpleNamespace(
            global_batch_info={
                "dp_size": 99,
                "batch_num_tokens": 1,
                "global_batch_size": 99,
                "loss_scale_factor": 99,
            },
            loss_scale_factor=None,
        )
        micro_losses = []
        global_batch_size = student.shape[0]
        batch_num_tokens = int(response_mask.sum().item())
        for start, end in step["splits"]:
            data = {
                "dp_size": 1,
                "batch_num_tokens": batch_num_tokens,
                "global_batch_size": global_batch_size,
            }
            _set_current_global_batch_info(config, data)
            values = _flash_groupwise_reverse_kl_values(
                student[start:end],
                teacher[start:end],
                group_ids[start:end],
                response_mask[start:end],
                coef,
            )
            counts = response_mask[start:end].sum(dim=-1)
            sequence_losses = (values * response_mask[start:end]).sum(dim=-1) / counts
            micro_losses.append(
                sequence_losses.sum()
                / config.global_batch_info["global_batch_size"]
                * config.global_batch_info["dp_size"]
            )
            _set_current_global_batch_info(config, data)
        actual = torch.stack(micro_losses).sum()
        legacy = torch.stack(
            [
                _gkd_loss_from_logps(
                    student[row],
                    [
                        (
                            torch.nonzero(group_ids[row].eq(group_id), as_tuple=False)
                            .flatten()
                            .tolist(),
                            float(teacher[row][group_ids[row].eq(group_id)][0]),
                        )
                        for group_id in torch.unique(
                            group_ids[row][group_ids[row].ge(0)], sorted=True
                        )
                    ],
                    kl_coef=coef,
                )
                for row in range(student.shape[0])
            ]
        ).mean()
        actual_gradient = torch.autograd.grad(actual, student, retain_graph=True)[0]
        legacy_gradient = torch.autograd.grad(legacy, student)[0]

        assert config.global_batch_info == {
            "dp_size": 1,
            "batch_num_tokens": batch_num_tokens,
            "global_batch_size": global_batch_size,
            "loss_scale_factor": None,
        }
        assert torch.allclose(actual, legacy, atol=1e-12, rtol=1e-12)
        assert torch.allclose(actual_gradient, legacy_gradient, atol=1e-12, rtol=1e-12)
        observed.append(float(actual.detach()))
        expected.append(float(legacy.detach()))

    assert observed == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert len(steps[0]["student"]) != len(steps[1]["student"])


def test_no_signal_sequence_is_excluded_before_actor_training():
    torch = pytest.importorskip("torch")
    group_ids = torch.tensor([[0, -1, -1], [-1, -1, -1], [2, 2, -1]])
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    assert _signal_sequences(group_ids, response_mask).tolist() == [True, False, True]
    full_sequence_ids = torch.tensor(
        [[-1, -1, 0, -1], [-1, -1, -1, -1], [-1, 2, 2, -1]]
    ).unsqueeze(-1)
    assert _full_sequence_signal_sequences(full_sequence_ids).tolist() == [True, False, True]


def test_all_no_signal_rollout_dispatches_bounded_usable_replacement():
    attempts = []
    seeds = []
    cleaned = []
    prepared = []
    resamples = []
    abandoned = []

    def run_attempt(attempt_ordinal):
        attempts.append(attempt_ordinal)
        seeds.append(
            deterministic_rollout_seed(
                42,
                3,
                7,
                1,
                no_signal_attempt_ordinal=attempt_ordinal,
            )
        )
        if attempt_ordinal < 2:
            raise _AllNoSignalBatch(f"empty-batch-{attempt_ordinal}")
        return "usable-batch"

    result = _run_with_no_signal_replacements(
        run_attempt,
        cleaned.append,
        lambda: prepared.append(True),
        lambda: resamples.append(True),
        lambda: abandoned.append(True),
    )

    assert result == "usable-batch"
    assert attempts == [0, 1, 2]
    assert seeds[0] == deterministic_rollout_seed(42, 3, 7, 1)
    assert len(set(seeds)) == 3
    assert seeds[1] == deterministic_rollout_seed(
        42, 3, 7, 1, no_signal_attempt_ordinal=1
    )
    assert cleaned == ["empty-batch-0", "empty-batch-1"]
    assert prepared == [True, True]
    assert resamples == [True, True]
    assert abandoned == []


def test_shifted_group_metadata_uses_verl_prediction_layout():
    teacher_ids, teacher_logprobs = encode_shifted_group_metadata(
        prompt_length=3,
        response_length=4,
        groups=[([0], -0.5), ([1, 2], -1.75)],
    )
    assert teacher_ids == [-1, -1, 0, 1, 1, -1, -1]
    assert teacher_logprobs == [0.0, 0.0, -0.5, -1.75, -1.75, 0.0, 0.0]
    assert teacher_ids[2:6] == [0, 1, 1, -1]


def _structured_test_tokenizer():
    import tokenizers
    import transformers
    characters = list('{}[]":,0123456789truefalsenullabc xyz')
    vocab = {"[UNK]": 0, "</think>": 1, "<eos>": 2}
    for character in characters:
        if character not in vocab:
            vocab[character] = len(vocab)
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="<eos>",
        additional_special_tokens=["</think>"],
    )
    return tokenizer, vocab


def _structured_ids(vocab, text: str, *, eos: bool = True) -> list[int]:
    token_ids = [vocab[character] for character in text]
    if eos:
        token_ids.append(vocab["<eos>"])
    return token_ids


@pytest.mark.parametrize(
    ("spec", "text"),
    [
        ({"json": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}, '{"a":"b"}'),
        ({"json_object": True}, '{"a":1}'),
        ({"regex": "[0-9]+"}, "12"),
        ({"choice": ["ab", "ac"]}, "ab"),
    ],
)
def test_xgrammar_replay_accepts_every_flash_constraint_kind(spec, text):
    pytest.importorskip("xgrammar")
    tokenizer, vocab = _structured_test_tokenizer()
    replay = StructuredOutputReplay(tokenizer, len(vocab))
    response_ids = _structured_ids(vocab, text)

    forced = replay.forced_mask(
        [], response_ids, canonical_structured_spec(spec), thinking=False
    )

    assert len(forced) == len(response_ids)
    assert replay._compile(canonical_structured_spec(spec)) is replay._compile(
        canonical_structured_spec(spec)
    )


def test_xgrammar_bit_count_distinguishes_forced_and_free_positions():
    torch = pytest.importorskip("torch")
    forced_mask = torch.tensor([[0b1000]], dtype=torch.int32)
    free_mask = torch.tensor([[0b1010]], dtype=torch.int32)
    padded_mask = torch.tensor([[-1]], dtype=torch.int32)

    assert _count_legal_tokens(forced_mask, 4) == 1
    assert _count_legal_tokens(free_mask, 4) == 2
    assert _count_legal_tokens(padded_mask, 3) == 3


def test_xgrammar_replay_fails_closed_when_generated_token_is_rejected():
    pytest.importorskip("xgrammar")
    tokenizer, vocab = _structured_test_tokenizer()
    replay = StructuredOutputReplay(tokenizer, len(vocab))

    with pytest.raises(RuntimeError, match="rejected a generated structured-output token"):
        replay.forced_mask(
            [],
            _structured_ids(vocab, "b"),
            canonical_structured_spec({"choice": ["a"]}),
            thinking=False,
        )


def test_xgrammar_replay_masks_only_positions_after_thinking_boundary():
    pytest.importorskip("xgrammar")
    tokenizer, vocab = _structured_test_tokenizer()
    replay = StructuredOutputReplay(tokenizer, len(vocab))
    answer_ids = _structured_ids(vocab, "a")
    response_ids = [vocab["x"], vocab["</think>"], *answer_ids]

    generated_boundary = replay.forced_mask(
        [],
        response_ids,
        canonical_structured_spec({"choice": ["a"]}),
        thinking=True,
    )
    prompt_boundary = replay.forced_mask(
        [vocab["</think>"]],
        answer_ids,
        canonical_structured_spec({"choice": ["a"]}),
        thinking=True,
    )

    assert generated_boundary[:2] == [False, False]
    assert all(generated_boundary[2:])
    assert all(prompt_boundary)


@pytest.mark.parametrize(
    ("spec", "text"),
    [
        ({"choice": ["4"]}, "4"),
        ({"regex": "[0-9]+"}, "42"),
        (
            {
                "json": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                }
            },
            '{"answer":"4"}',
        ),
    ],
)
def test_xgrammar_replay_uses_real_qwen_padded_model_vocab(spec, text):
    pytest.importorskip("xgrammar")
    from transformers import AutoConfig, AutoTokenizer

    model_id = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    assert len(tokenizer.get_vocab()) == 248077
    assert config.text_config.vocab_size == 248320

    replay = StructuredOutputReplay(tokenizer, config.text_config.vocab_size)
    response_ids = tokenizer.encode(text, add_special_tokens=False)
    forced = replay.forced_mask(
        [],
        response_ids,
        canonical_structured_spec(spec),
        thinking=False,
    )

    assert len(forced) == len(response_ids)
    assert tuple(replay._bitmask.shape) == (1, 248320 // 32)


def test_image_prompt_positions_remain_outside_alignment_groups():
    prompt_ids = [1, 151655, 151655, 2]
    teacher_ids, _teacher_logprobs = encode_shifted_group_metadata(
        prompt_length=len(prompt_ids),
        response_length=2,
        groups=[([0], -0.4)],
    )
    assert teacher_ids[: len(prompt_ids) - 1] == [-1, -1, -1]
    assert teacher_ids[len(prompt_ids) - 1] == 0


@pytest.mark.parametrize("image_first", [False, True])
def test_multimodal_opd_parquet_round_trip_preserves_image_dicts(tmp_path, image_first):
    image_row = {
        "prompt": [{"role": "user", "content": "<image>describe"}],
        "images": [{"image": "file:///tmp/first.png"}],
        "data_source": "flash_opd",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"index": 1},
    }
    text_row = {
        "prompt": [{"role": "user", "content": "text only"}],
        "images": [],
        "data_source": "flash_opd",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"index": 0},
    }
    rows = [image_row, text_row] if image_first else [text_row, image_row]
    path = tmp_path / "mixed.parquet"

    _write_opd_parquet(rows, str(path))

    datasets = pytest.importorskip("datasets")
    restored = datasets.Dataset.from_parquet(str(path))
    assert restored[0]["images"] == rows[0]["images"]
    assert restored[1]["images"] == rows[1]["images"]
    assert restored.features["images"].feature["image"].dtype == "string"


@pytest.mark.parametrize("image_first", [False, True])
def test_verl_0_8_rlhf_dataset_builds_one_structured_block_per_image(
    monkeypatch, tmp_path, image_first
):
    rl_dataset = _load_verl_rl_dataset(monkeypatch)
    assert importlib.metadata.version("verl") == "0.8.0"
    image_row = {
        "prompt": [{"role": "user", "content": "before<image>after"}],
        "images": [{"image": "file:///tmp/only.png"}],
        "data_source": "flash_opd",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"index": 1},
    }
    text_row = {
        "prompt": [{"role": "user", "content": "text only"}],
        "images": [],
        "data_source": "flash_opd",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"index": 0},
    }
    rows = [image_row, text_row] if image_first else [text_row, image_row]
    path = tmp_path / "verl-row.parquet"
    _write_opd_parquet(rows, str(path))
    datasets = pytest.importorskip("datasets")
    restored = datasets.Dataset.from_parquet(str(path))
    row = dict(restored[0 if image_first else 1])
    dataset = object.__new__(rl_dataset.RLHFDataset)
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.processor = object()

    messages = dataset._build_messages(row, key="prompt")

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "image": "file:///tmp/only.png"},
                {"type": "text", "text": "after"},
            ],
        }
    ]


class _MockMultimodalProcessor:
    image_token_id = 151655

    def __init__(self):
        self.rendered = None
        self.images = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        self.rendered = messages
        return "<vision>describe"

    def __call__(self, **kwargs):
        self.images = kwargs["images"]
        assert kwargs["text"] == ["<vision>describe"]
        assert kwargs["videos"] is None
        assert kwargs["return_tensors"] == "pt"
        return {"input_ids": [[10, self.image_token_id, self.image_token_id, 11]]}


def test_processor_expanded_prompt_ids_enforce_visual_token_budget(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    image_path = dataset_dir / "image.png"
    image_module.new("RGB", (2, 2), "red").save(image_path)
    processor = _MockMultimodalProcessor()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "describe"}],
        }
    ]

    from flash.multimodal import normalize_image_source

    prompt_ids = _processor_expanded_prompt_ids(
        processor,
        messages,
        (normalize_image_source("dataset/image.png", tmp_path),),
        str(tmp_path),
        enable_thinking=False,
    )

    assert prompt_ids == (10, 151655, 151655, 11)
    assert len(prompt_ids) > 3
    assert processor.rendered[0]["content"][0]["image"].size == (2, 2)
    assert [image.size for image in processor.images] == [(2, 2)]


class _BridgeTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, *, skip_special_tokens):
        mapping = {65: "A", 66: "B", 99: "" if skip_special_tokens else "<eos>"}
        return "".join(mapping[int(token_id)] for token_id in token_ids)


def _resume_accounting(step=2):
    return {
        "contract_version": 2,
        "seed": 42,
        "opt_steps": step,
        "step": step,
        "rollout_seed_ordinal": 12,
        "prompt_pool_fingerprint": "a" * 64,
        "generated_tokens": 41,
        "teacher_input_tokens": 37,
        "truncated_rollouts": 3,
        "forced_tokens": 9,
        "dropped_forced_groups": 4,
        "granularity_n": 5,
        "samples_seen": 8,
        "teacher_ok": 6,
        "teacher_transient": 1,
        "teacher_error": 0,
        "no_signal_resamples": 2,
        "no_signal_skipped_steps": 1,
        "episodes_seen": 8,
        "mt_turn_records": 0,
        "granularity_sum": 3.5,
        "train_wall_seconds": 12.5,
        "loss_curve": [1.25, 0.75][:step],
        "coverage_curve": [0.5, 0.7][:step],
        "skip_counts": {"empty_alignment": 2},
        "opd_phase_seconds": {"teacher": 4.0},
        "opd_phase_counts": {"teacher": 6},
        "aligned_sequences": 5,
        "empty_alignments": 2,
        "coverage_sum": 3.5,
    }


class _BridgeTeacher:
    def score(self, prompt_text, completion_text):
        assert prompt_text == "User: question\nAssistant: "
        assert completion_text == "AB"
        return [
            TeacherToken(text="A", logprob=-0.4, start=0, end=1),
            TeacherToken(text="B", logprob=-0.7, start=1, end=2),
        ]


class _MergedBridgeTeacher:
    def score(self, prompt_text, completion_text):
        assert prompt_text == "User: question\nAssistant: "
        assert completion_text == "AB"
        return [TeacherToken(text="AB", logprob=-1.1, start=0, end=2)]


class _ScoreManyTeacherAdapter:
    def __init__(self, teacher):
        self.teacher = teacher

    def score(self, prompt_text, completion_text):
        return self.teacher.score(prompt_text, completion_text)

    def score_many(self, items):
        return [
            self.teacher.score(prompt_text, completion_text)
            for prompt_text, completion_text in items
        ]


def _text_bridge(teacher, *, mutation_callback=None):
    if not hasattr(teacher, "score_many"):
        teacher = _ScoreManyTeacherAdapter(teacher)
    return _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[{"role": "user", "content": "question"}],
                teacher_messages=[{"role": "user", "content": "question"}],
                prompt_ids=(10, 11),
                image_descriptors=(),
                package_root=None,
            )
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=teacher,
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=(
            mutation_callback if mutation_callback is not None else lambda: None
        ),
    )


def _batching_bridge(teacher, prompt_texts):
    return _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[{"role": "user", "content": prompt_text}],
                teacher_messages=[{"role": "user", "content": prompt_text}],
                prompt_ids=(10, 11),
                image_descriptors=(),
                package_root=None,
            )
            for prompt_text in prompt_texts
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=teacher,
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
    )


class _BatchingTeacher:
    def __init__(
        self,
        prompt_texts,
        *,
        failure: str | None = None,
        token_logprob: float | None = None,
    ):
        self.logprobs = {
            f"User: {prompt_text}\nAssistant: ": -float(index + 1)
            for index, prompt_text in enumerate(dict.fromkeys(prompt_texts))
        }
        self.failure = failure
        self.token_logprob = token_logprob
        self.batches = []
        self.called = threading.Event()
        self._lock = threading.Lock()

    def score_many(self, items):
        with self._lock:
            self.batches.append(list(items))
        self.called.set()
        if self.failure == "transient":
            from flash.engine.worker.teacher import TeacherError

            raise TeacherError("teacher unavailable", permanent=False)
        if self.failure == "permanent":
            from flash.engine.worker.teacher import TeacherError

            raise TeacherError("permanent teacher failure", permanent=True)
        if self.failure == "wrong_count":
            return []
        if self.failure == "malformed_result":
            return [[object()] for _item in items]
        if self.failure == "invalid_offsets":
            return [
                [TeacherToken(text="AB", logprob=-1.0, start=0, end=3)]
                for _item in items
            ]
        return [
            [
                TeacherToken(
                    text=completion_text,
                    logprob=(
                        self.token_logprob
                        if self.token_logprob is not None
                        else self.logprobs[prompt_text]
                    ),
                    start=0,
                    end=len(completion_text),
                )
            ]
            for prompt_text, completion_text in items
        ]


def _concurrent_bridge_scores(bridge, indexes, *, via_http=True):
    start = threading.Barrier(len(indexes))

    def score(index):
        start.wait(timeout=5.0)
        try:
            if via_http:
                result = _post_json(
                    bridge.url,
                    bridge.token,
                    "/score",
                    _bridge_score_payload(index, [10, 11], [65, 66, 99]),
                )
            else:
                result = bridge.score(index, 2, [10, 11, 65, 66, 99])
            return "ok", result
        except Exception as error:
            return "error", error

    with ThreadPoolExecutor(max_workers=len(indexes)) as executor:
        futures = [executor.submit(score, index) for index in indexes]
        return [future.result(timeout=10.0) for future in futures]


def _teacher_logsum(result):
    return result["teacher_logprobs"][1]


def _exhaust_bridge_no_signal(bridge):
    def run_attempt(attempt_ordinal):
        payload = _post_json(
            bridge.url,
            bridge.token,
            "/score",
            _bridge_score_payload(0, [10, 11], [65, 66, 99]),
        )
        assert payload["teacher_ids"] == [-1, -1, -1, -1, -1]
        raise _AllNoSignalBatch(attempt_ordinal)

    return _run_with_no_signal_replacements(
        run_attempt,
        lambda _batch: None,
        lambda: None,
        lambda: _post_json(bridge.url, bridge.token, "/no-signal/resample", {}),
        lambda: _post_json(bridge.url, bridge.token, "/no-signal/abandoned", {}),
    )


def test_bridge_verifies_prompt_and_serializes_aligned_native_fields():
    bridge = _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[{"role": "user", "content": "question"}],
                teacher_messages=[{"role": "user", "content": "question"}],
                prompt_ids=(10, 11),
                image_descriptors=(),
                package_root=None,
            )
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=_BridgeTeacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
    )
    encoded = bridge.score(0, 2, [10, 11, 65, 66, 99])
    assert encoded["teacher_ids"] == [-1, 0, 1, -1, -1]
    assert encoded["teacher_logprobs"] == [0.0, -0.4, -0.7, 0.0, 0.0]
    assert bridge.aligned_sequences == 1
    assert bridge.generated_tokens == 3
    with pytest.raises(ValueError, match="prompt ids"):
        bridge.score(0, 2, [10, 12, 65, 99])


def test_text_teacher_batcher_enforces_max_batch_size_across_concurrent_requests():
    prompt_texts = [f"question-{index}" for index in range(17)]
    teacher = _BatchingTeacher(prompt_texts)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(17))
    finally:
        bridge.close()

    assert all(status == "ok" for status, _result in outcomes)
    batch_sizes = [len(batch) for batch in teacher.batches]
    assert sum(batch_sizes) == 17
    assert max(batch_sizes) == 8
    assert all(size <= 8 for size in batch_sizes)


def test_text_teacher_batcher_flushes_final_partial_batch_within_bound():
    teacher = _BatchingTeacher(["question"])
    bridge = _batching_bridge(teacher, ["question"])
    bridge.start()
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _post_json,
                bridge.url,
                bridge.token,
                "/score",
                _bridge_score_payload(0, [10, 11], [65, 66, 99]),
            )
            assert teacher.called.wait(timeout=0.5)
            elapsed = time.monotonic() - started
            result = future.result(timeout=5.0)
    finally:
        bridge.close()

    assert elapsed < 0.5
    assert result["teacher_ids"] == [-1, 0, 0, -1, -1]
    assert [len(batch) for batch in teacher.batches] == [1]


def test_text_teacher_batcher_deduplicates_exact_pairs_and_scatters_to_all_waiters(
    monkeypatch,
):
    from flash.engine.worker import opd_verl as opd_verl_mod

    monkeypatch.setattr(opd_verl_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
    prompt_texts = ["same question"] * 8
    teacher = _BatchingTeacher(prompt_texts)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(8), via_http=False)
    finally:
        bridge.close()

    assert teacher.batches == [
        [("User: same question\nAssistant: ", "AB")]
    ]
    assert all(status == "ok" for status, _result in outcomes)
    assert [_teacher_logsum(result) for _status, result in outcomes] == [-1.0] * 8
    assert bridge.score_requests == 8
    assert bridge.teacher_ok == 8


def test_text_teacher_batcher_keeps_nonidentical_inputs_separate_and_ordered(monkeypatch):
    from flash.engine.worker import opd_verl as opd_verl_mod

    monkeypatch.setattr(opd_verl_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
    prompt_texts = [f"distinct-{index}" for index in range(8)]
    teacher = _BatchingTeacher(prompt_texts)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(8), via_http=False)
    finally:
        bridge.close()

    assert len(teacher.batches) == 1
    assert len(teacher.batches[0]) == 8
    for index, (status, result) in enumerate(outcomes):
        teacher_prompt = f"User: distinct-{index}\nAssistant: "
        assert status == "ok"
        assert _teacher_logsum(result) == teacher.logprobs[teacher_prompt]


def test_text_teacher_batch_accepts_positive_rounding_for_every_logical_waiter(monkeypatch):
    from flash.engine.worker import opd_verl as opd_verl_mod

    monkeypatch.setattr(opd_verl_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
    prompt_texts = ["rounding"] * 8
    teacher = _BatchingTeacher(prompt_texts, token_logprob=1e-9)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(8))
    finally:
        bridge.close()

    assert teacher.batches
    assert all(
        batch == [("User: rounding\nAssistant: ", "AB")]
        for batch in teacher.batches
    )
    assert all(status == "ok" for status, _result in outcomes)
    assert [_teacher_logsum(result) for _status, result in outcomes] == [1e-9] * 8
    assert bridge.score_requests == 8
    assert bridge.teacher_ok == 8
    assert bridge.teacher_error == 0
    assert bridge.teacher_transient == 0


def test_text_teacher_batch_rejects_positive_value_above_tolerance_for_every_waiter(
    monkeypatch,
):
    from flash.engine.worker import opd_verl as opd_verl_mod

    monkeypatch.setattr(opd_verl_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
    prompt_texts = ["invalid rounding"] * 8
    teacher = _BatchingTeacher(prompt_texts, token_logprob=2e-6)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(8))
    finally:
        bridge.close()

    assert teacher.batches
    assert all(
        batch == [("User: invalid rounding\nAssistant: ", "AB")]
        for batch in teacher.batches
    )
    assert all(status == "error" for status, _error in outcomes)
    errors = [error for _status, error in outcomes]
    assert all(isinstance(error, FlashTeacherBridgeError) for error in errors)
    assert all(error.classification == "permanent" for error in errors)
    assert all("invalid logprob" in str(error) for error in errors)
    assert bridge.score_requests == 8
    assert bridge.teacher_ok == 0
    assert bridge.teacher_error == 8
    assert bridge.teacher_transient == 0


def test_text_teacher_batch_transient_failure_recovers_each_logical_sample_once(monkeypatch):
    from flash.engine.worker import opd_verl as opd_verl_mod

    monkeypatch.setattr(opd_verl_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
    prompt_texts = [f"transient-{index}" for index in range(8)]
    teacher = _BatchingTeacher(prompt_texts, failure="transient")
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(8), via_http=False)
        _post_json(bridge.url, bridge.token, "/no-signal/abandoned", {})
    finally:
        bridge.close()

    assert len(teacher.batches) == 1
    assert all(status == "ok" for status, _result in outcomes)
    assert all(
        result["teacher_ids"] == [-1, -1, -1, -1, -1]
        for _status, result in outcomes
    )
    assert bridge.score_requests == 8
    assert bridge.teacher_transient == 8
    assert bridge.teacher_ok == 0
    assert bridge.teacher_error == 0
    assert bridge.teacher_failure == ("transient", "teacher unavailable")


@pytest.mark.parametrize(
    "failure",
    ["permanent", "wrong_count", "malformed_result", "invalid_offsets"],
)
def test_text_teacher_batch_failures_complete_every_waiter_fail_closed(failure):
    prompt_texts = [f"failure-{index}" for index in range(8)]
    teacher = _BatchingTeacher(prompt_texts, failure=failure)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(8))
    finally:
        bridge.close()

    assert teacher.batches
    assert sum(len(batch) for batch in teacher.batches) == 8
    assert all(len(batch) <= 8 for batch in teacher.batches)
    assert all(status == "error" for status, _error in outcomes)
    errors = [error for _status, error in outcomes]
    assert all(isinstance(error, FlashTeacherBridgeError) for error in errors)
    assert all(error.classification == "permanent" for error in errors)
    assert bridge.score_requests == 8
    assert bridge.teacher_error == 8
    assert bridge.teacher_transient == 0
    assert bridge.teacher_ok == 0


def test_text_teacher_batch_mixed_dedup_preserves_logical_accounting(monkeypatch):
    from flash.engine.worker import opd_verl as opd_verl_mod

    monkeypatch.setattr(opd_verl_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
    prompt_texts = [
        "duplicate",
        "unique-a",
        "duplicate",
        "unique-b",
        "duplicate",
        "unique-a",
        "duplicate",
        "unique-b",
    ]
    teacher = _BatchingTeacher(prompt_texts)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(8), via_http=False)
    finally:
        bridge.close()

    assert len(teacher.batches) == 1
    assert len(teacher.batches[0]) == 3
    for prompt_text, (status, result) in zip(prompt_texts, outcomes, strict=True):
        assert status == "ok"
        assert _teacher_logsum(result) == teacher.logprobs[
            f"User: {prompt_text}\nAssistant: "
        ]
    assert bridge.score_requests == 8
    assert bridge.teacher_ok == 8
    assert bridge.teacher_input_tokens == 40
    assert bridge.aligned_sequences == 8


def test_text_teacher_batcher_shutdown_cannot_strand_pending_bridge_waiter(monkeypatch):
    from flash.engine.worker import opd_verl as opd_verl_mod

    monkeypatch.setattr(opd_verl_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 10.0)
    teacher = _BatchingTeacher(["question"])
    bridge = _batching_bridge(teacher, ["question"])
    bridge.start()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        _post_json,
        bridge.url,
        bridge.token,
        "/score",
        _bridge_score_payload(0, [10, 11], [65, 66, 99]),
    )
    batcher = bridge._text_teacher_batcher
    assert batcher is not None
    with batcher._condition:
        assert batcher._condition.wait_for(lambda: bool(batcher._pending), timeout=1.0)

    bridge.close()
    try:
        with pytest.raises(FlashTeacherBridgeError) as error:
            future.result(timeout=2.0)
    finally:
        executor.shutdown(wait=True)

    assert error.value.classification == "permanent"
    assert not teacher.called.is_set()
    assert bridge.teacher_error == 1


def test_transient_teacher_sample_returns_no_signal_while_following_peer_trains():
    torch = pytest.importorskip("torch")
    from flash.engine.worker.teacher import TeacherError

    class FlakyTeacher:
        def __init__(self):
            self.calls = 0

        def score(self, prompt_text, completion_text):
            self.calls += 1
            if self.calls == 1:
                raise TeacherError("teacher unavailable", permanent=False)
            return _BridgeTeacher().score(prompt_text, completion_text)

    bridge = _text_bridge(FlakyTeacher())
    bridge.start()
    try:
        transient = _post_json(
            bridge.url,
            bridge.token,
            "/score",
            _bridge_score_payload(0, [10, 11], [65, 66, 99]),
        )
        signal = _post_json(
            bridge.url,
            bridge.token,
            "/score",
            _bridge_score_payload(0, [10, 11], [65, 66, 99]),
        )
    finally:
        bridge.close()

    assert transient == {
        "teacher_ids": [-1, -1, -1, -1, -1],
        "teacher_logprobs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    assert signal == {
        "teacher_ids": [-1, 0, 1, -1, -1],
        "teacher_logprobs": [0.0, -0.4, -0.7, 0.0, 0.0],
    }
    group_ids = torch.tensor([transient["teacher_ids"], signal["teacher_ids"]])
    teacher_logsums = torch.tensor(
        [transient["teacher_logprobs"], signal["teacher_logprobs"]],
        dtype=torch.float64,
    )
    response_mask = torch.tensor(
        [[False, True, True, True, False], [False, True, True, True, False]]
    )
    student_logprobs = torch.tensor(
        [[-0.3, -0.5, -0.8, -1.1, -0.2], [-0.4, -0.6, -0.9, -1.2, -0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    values = _flash_groupwise_reverse_kl_values(
        student_logprobs,
        teacher_logsums,
        group_ids,
        response_mask,
        0.5,
    )
    values.sum().backward()

    assert _full_sequence_signal_sequences(group_ids.unsqueeze(-1)).tolist() == [False, True]
    assert torch.equal(student_logprobs.grad[0], torch.zeros(5, dtype=torch.float64))
    assert student_logprobs.grad[1].abs().sum().item() > 0
    assert bridge.score_requests == 2
    assert bridge.teacher_transient == 1
    assert bridge.teacher_ok == 1
    assert bridge.teacher_error == 0
    assert bridge.teacher_failure is None


def test_recovered_transient_does_not_make_later_deterministic_exhaustion_retriable():
    from flash.engine.worker.teacher import TeacherError

    class TransientSignalThenEmptyTeacher:
        def __init__(self):
            self.calls = 0

        def score(self, prompt_text, completion_text):
            self.calls += 1
            if self.calls == 1:
                raise TeacherError("teacher unavailable", permanent=False)
            if self.calls == 2:
                return _BridgeTeacher().score(prompt_text, completion_text)
            return []

    mutations = []
    bridge = _text_bridge(
        TransientSignalThenEmptyTeacher(),
        mutation_callback=lambda: mutations.append(True),
    )
    bridge.start()
    try:
        transient = _post_json(
            bridge.url,
            bridge.token,
            "/score",
            _bridge_score_payload(0, [10, 11], [65, 66, 99]),
        )
        signal = _post_json(
            bridge.url,
            bridge.token,
            "/score",
            _bridge_score_payload(0, [10, 11], [65, 66, 99]),
        )
        _post_json(bridge.url, bridge.token, "/mutation", {})
        with pytest.raises(RuntimeError, match="after 3 rollout attempts"):
            _exhaust_bridge_no_signal(bridge)
        with pytest.raises(RuntimeError, match="subprocess exited with status 1"):
            _raise_verl_failure(1, bridge.teacher_failure)
    finally:
        bridge.close()

    assert transient["teacher_ids"] == [-1, -1, -1, -1, -1]
    assert signal["teacher_ids"] == [-1, 0, 1, -1, -1]
    assert bridge.teacher_transient == 1
    assert bridge.teacher_ok == 4
    assert bridge.empty_alignments == 3
    assert mutations == [True]


def test_recovered_transient_does_not_mask_unrelated_child_exit():
    from flash.engine.worker.teacher import TeacherError

    class TransientThenSignalTeacher:
        def __init__(self):
            self.calls = 0

        def score(self, prompt_text, completion_text):
            self.calls += 1
            if self.calls == 1:
                raise TeacherError("teacher unavailable", permanent=False)
            return _BridgeTeacher().score(prompt_text, completion_text)

    bridge = _text_bridge(TransientThenSignalTeacher())
    bridge.start()
    try:
        _post_json(
            bridge.url,
            bridge.token,
            "/score",
            _bridge_score_payload(0, [10, 11], [65, 66, 99]),
        )
        _post_json(
            bridge.url,
            bridge.token,
            "/score",
            _bridge_score_payload(0, [10, 11], [65, 66, 99]),
        )
        with pytest.raises(RuntimeError, match="subprocess exited with status 9"):
            _raise_verl_failure(9, bridge.teacher_failure)
    finally:
        bridge.close()

    assert bridge.teacher_transient == 1
    assert bridge.teacher_ok == 1


def test_fully_forced_groups_encode_as_no_signal_and_are_filtered():
    torch = pytest.importorskip("torch")
    bridge = _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[{"role": "user", "content": "question"}],
                teacher_messages=[{"role": "user", "content": "question"}],
                prompt_ids=(10, 11),
                image_descriptors=(),
                package_root=None,
            )
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=_BridgeTeacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
        structured=True,
    )

    encoded = bridge.score(
        0,
        2,
        [10, 11, 65, 66, 99],
        forced=[True, True, False],
    )

    assert encoded["teacher_ids"] == [-1, -1, -1, -1, -1]
    full_sequence_ids = torch.tensor([encoded["teacher_ids"]]).unsqueeze(-1)
    assert _full_sequence_signal_sequences(full_sequence_ids).tolist() == [False]
    assert bridge.forced_tokens == 2
    assert bridge.dropped_forced_groups == 2


def test_partially_forced_alignment_group_is_kept_whole():
    bridge = _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[{"role": "user", "content": "question"}],
                teacher_messages=[{"role": "user", "content": "question"}],
                prompt_ids=(10, 11),
                image_descriptors=(),
                package_root=None,
            )
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=_MergedBridgeTeacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
        structured=True,
    )

    encoded = bridge.score(
        0,
        2,
        [10, 11, 65, 66, 99],
        forced=[True, False, False],
    )

    assert encoded["teacher_ids"] == [-1, 0, 0, -1, -1]
    assert encoded["teacher_logprobs"] == [0.0, -1.1, -1.1, 0.0, 0.0]
    assert bridge.dropped_forced_groups == 0


def test_bridge_rejects_child_prompt_with_extra_token_as_permanent_error():
    bridge = _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[{"role": "user", "content": "question"}],
                teacher_messages=[{"role": "user", "content": "question"}],
                prompt_ids=(10, 11),
                image_descriptors=(),
                package_root=None,
            )
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=_BridgeTeacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
    )
    bridge.start()
    try:
        with pytest.raises(FlashTeacherBridgeError, match="exactly match") as error:
            _post_json(
                bridge.url,
                bridge.token,
                "/score",
                _bridge_score_payload(0, [10, 11, 77], [65, 99]),
            )
    finally:
        bridge.close()

    assert error.value.classification == "permanent"
    assert bridge.teacher_failure is not None
    assert bridge.teacher_failure[0] == "permanent"


def test_child_bridge_payload_carries_actual_prompt_boundary():
    payload = _bridge_score_payload(4, [10, 11, 77], [65, 99])

    assert payload == {
        "index": 4,
        "prompt_length": 3,
        "sequence_ids": [10, 11, 77, 65, 99],
        "image_count": 0,
    }


def test_structured_child_bridge_payload_includes_forced_mask():
    payload = _bridge_score_payload(
        4,
        [10, 11],
        [65, 99],
        forced=[True, False],
    )

    assert payload["forced"] == [True, False]


def test_structured_bridge_requires_exact_boolean_mask_for_untrimmed_response():
    with pytest.raises(ValueError, match="missing the forced mask"):
        _validate_forced_mask(None, 2, required=True)
    with pytest.raises(ValueError, match="only booleans"):
        _validate_forced_mask([True, 0], 2, required=True)
    with pytest.raises(ValueError, match="untrimmed response"):
        _validate_forced_mask([True], 2, required=True)
    assert _validate_forced_mask([True, False], 2, required=True) == [True, False]
    assert _validate_forced_mask([True], 2, required=False) == []


class _StopTokenizer:
    def decode(self, token_ids, *, skip_special_tokens):
        return "".join({1: "A", 2: "B", 3: "STOP"}[int(token_id)] for token_id in token_ids)


def test_stop_trimming_slices_response_ids_and_forced_mask_identically():
    kept_ids, completion_text, kept_forced = _trim_response_and_forced(
        _StopTokenizer(),
        [1, 2, 3],
        "ABSTOP",
        ("STOP",),
        [True, False, True],
    )

    assert kept_ids == [1, 2]
    assert completion_text == "AB"
    assert kept_forced == [True, False]


class _ScoredImageTokens(list):
    input_tokens = 17


class _ImageBridgeTeacher:
    def __init__(self):
        self.items = None

    def score_many_multimodal(self, items):
        self.items = items
        return [
            _ScoredImageTokens(
                [
                    TeacherToken(text="A", logprob=-0.4, start=0, end=1),
                    TeacherToken(text="B", logprob=-0.7, start=1, end=2),
                ]
            )
        ]


def test_multimodal_bridge_rebuilds_teacher_images_in_frozen_order_and_accounts_echo(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    from flash.multimodal import image_descriptors_to_data_uris, normalize_image_source

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    red = dataset_dir / "red.png"
    blue = dataset_dir / "blue.png"
    image_module.new("RGB", (2, 2), "red").save(red)
    image_module.new("RGB", (2, 2), "blue").save(blue)
    descriptors = (
        normalize_image_source("dataset/red.png", tmp_path),
        normalize_image_source("dataset/blue.png", tmp_path),
    )
    teacher = _ImageBridgeTeacher()
    bridge = _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": " then "},
                            {"type": "image"},
                        ],
                    }
                ],
                teacher_messages=[
                    {"role": "user", "content": "<|media_pad|> then <|media_pad|>"}
                ],
                prompt_ids=(10, 11),
                image_descriptors=descriptors,
                package_root=str(tmp_path),
            )
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=teacher,
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
    )

    bridge.start()
    try:
        encoded = bridge.score(0, 2, [10, 11, 65, 66, 99], image_count=2)
        with pytest.raises(ValueError, match="exactly match"):
            bridge.score(0, 3, [10, 11, 77, 65, 99], image_count=2)
    finally:
        bridge.close()

    expected_uris = image_descriptors_to_data_uris(descriptors, tmp_path)
    assert teacher.items == [
        ("User: <|media_pad|> then <|media_pad|>\nAssistant: ", "AB", expected_uris)
    ]
    assert encoded["teacher_ids"] == [-1, 0, 1, -1, -1]
    assert bridge.teacher_input_tokens == 17


def test_bridge_rejects_parent_child_image_count_mismatch_before_scoring():
    bridge = _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[{"role": "user", "content": [{"type": "image"}]}],
                teacher_messages=[{"role": "user", "content": "<|media_pad|>"}],
                prompt_ids=(10, 11),
                image_descriptors=("frozen-descriptor",),
                package_root=None,
            )
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=object(),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
    )

    with pytest.raises(ValueError, match=r"reported 0 image.*frozen prompt has 1"):
        bridge.score(0, 2, [10, 11, 65, 99], image_count=0)


def test_prompt_pool_fingerprint_freezes_image_descriptor_order_and_preserves_text():
    def fingerprint(descriptors):
        return _prompt_pool_fingerprint(
            [
                _BridgePrompt(
                    student_messages=[{"role": "user", "content": [{"type": "image"}]}],
                    teacher_messages=[{"role": "user", "content": "<|media_pad|>"}],
                    prompt_ids=(10, 11),
                    image_descriptors=descriptors,
                    package_root="/package",
                )
            ]
        )

    assert fingerprint(("red", "blue")) != fingerprint(("blue", "red"))

    text_messages = [{"role": "user", "content": "question"}]
    prompt = _BridgePrompt(
        student_messages=text_messages,
        teacher_messages=text_messages,
        prompt_ids=(10, 11),
        image_descriptors=(),
        package_root=None,
    )
    payload = json.dumps(
        [text_messages, [10, 11]],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    expected = hashlib.sha256(len(payload).to_bytes(8, "big") + payload).hexdigest()
    assert _prompt_pool_fingerprint([prompt]) == expected


def test_retry_sidecar_persists_real_accumulated_accounting(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    adapter = tmp_path / "adapter"
    checkpoint.mkdir()
    adapter.mkdir()
    (checkpoint / "optim_state.bin").write_bytes(b"optimizer")
    (checkpoint / "data.pt").write_bytes(b"rng")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    accounting = _resume_accounting()

    _stage_retry_contract(
        str(checkpoint),
        step=2,
        seed=42,
        prompt_pool_fingerprint="a" * 64,
        prompts_per_step=2,
        group_size=3,
        adapter_dir=str(adapter),
        accounting_state=accounting,
    )

    import json

    state = json.loads((checkpoint / "opd_state.json").read_text())
    assert state["loss_curve"] == [1.25, 0.75]
    assert state["coverage_curve"] == [0.5, 0.7]
    assert state["generated_tokens"] == 41
    assert state["teacher_input_tokens"] == 37
    assert state["teacher_ok"] == 6
    assert state["teacher_transient"] == 1
    assert state["forced_tokens"] == 9
    assert state["dropped_forced_groups"] == 4
    assert state["samples_seen"] == 8
    assert state["train_wall_seconds"] == 12.5
    assert state["rollout_seed_ordinal"] == 12


def test_resume_restores_bridge_counters_and_extends_full_curves():
    state = _resume_accounting()
    bridge = _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[{"role": "user", "content": "question"}],
                teacher_messages=[{"role": "user", "content": "question"}],
                prompt_ids=(10, 11),
                image_descriptors=(),
                package_root=None,
            )
        ],
        tokenizer=_BridgeTokenizer(),
        teacher=_BridgeTeacher(),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
        initial_state=state,
    )
    progress = _OpdProgressState(state)
    progress.start_training()
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    progress.record_step(3, 0.5, bridge)
    restored = progress.checkpoint_state(3, timeout_s=0.1)

    assert restored["loss_curve"] == [1.25, 0.75, 0.5]
    assert restored["coverage_curve"] == [0.5, 0.7, 1.0]
    assert restored["generated_tokens"] == 44
    assert restored["teacher_input_tokens"] == 42
    assert restored["teacher_ok"] == 7
    assert restored["forced_tokens"] == 9
    assert restored["dropped_forced_groups"] == 4
    assert restored["aligned_sequences"] == 6
    assert restored["empty_alignments"] == 2
    assert restored["train_wall_seconds"] >= 12.5


def test_restore_verl_resume_returns_validated_accounting(monkeypatch, tmp_path):
    from flash.engine.worker import opd_verl

    resume = tmp_path / "checkpoint-2"
    resume.mkdir()
    state = _resume_accounting()
    import json

    (resume / "opd_state.json").write_text(json.dumps(state))
    (resume / "payload.bin").write_bytes(b"checkpoint")
    monkeypatch.setattr(opd_verl._w, "OPD_RESUME_REVISION", "revision")
    monkeypatch.setattr(opd_verl._w, "SEED", 42)
    monkeypatch.setattr(
        opd_verl._w,
        "hf_resume_checkpoint",
        lambda **_kwargs: str(resume),
    )
    local_dir = tmp_path / "local"
    local_dir.mkdir()

    step, restored = _restore_verl_resume(
        str(local_dir), prompt_pool_fingerprint="a" * 64, update_horizon=3
    )

    assert step == 2
    assert restored == state
    assert (local_dir / "global_step_2" / "payload.bin").read_bytes() == b"checkpoint"


def test_resume_leaves_missing_required_companion_for_checkpoint_watcher(
    monkeypatch, tmp_path
):
    import flash.engine.worker as worker

    class Api:
        def file_exists(self, **_kwargs):
            return False

    monkeypatch.setattr(worker, "HF_REPO", "owner/artifacts")
    monkeypatch.setattr(worker, "hf_prefix", lambda: "opd/run")
    monkeypatch.setattr(worker, "hf_api", Api)
    local_dir = tmp_path / "checkpoints"
    checkpoint_dir = local_dir / "global_step_3"
    checkpoint_dir.mkdir(parents=True)
    (local_dir / "latest_checkpointed_iteration.txt").write_text("3")
    watcher = _OpdVerlCheckpointWatcher(
        local_dir=str(local_dir),
        export_root=str(tmp_path / "exports"),
        python_bin="/verl/python",
        model_id="org/model",
        model_revision="commit",
        required_steps=(3,),
        seed=42,
        prompt_pool_fingerprint="a" * 64,
        prompts_per_step=2,
        group_size=3,
        accounting_state=lambda _step: _resume_accounting(3),
    )

    watcher.processed_steps.update(_processed_resume_steps((3,), 3))
    published = []

    def publish(step, path):
        published.append((step, path))
        watcher.processed_steps.add(step)

    watcher._publish = publish
    watcher.start()
    watcher.stop(require_complete=True)

    assert published == [(3, str(checkpoint_dir))]
    assert watcher.processed_steps == {3}
    assert _processed_resume_steps((4,), 3) == {3}


def test_bridge_preserves_typed_permanent_teacher_failure():
    from flash.engine.worker.teacher import TeacherError

    class FailingTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("permanent teacher failure", permanent=True)

    bridge = _text_bridge(FailingTeacher())
    bridge.start()
    try:
        with pytest.raises(FlashTeacherBridgeError) as error:
            _post_json(
                bridge.url,
                bridge.token,
                "/score",
                _bridge_score_payload(0, [10, 11], [65, 66, 99]),
            )
    finally:
        bridge.close()

    assert error.value.classification == "permanent"
    assert str(error.value) == "permanent teacher failure"
    assert bridge.teacher_failure == ("permanent", "permanent teacher failure")
    assert bridge.teacher_error == 1
    assert bridge.teacher_transient == 0
    assert bridge.teacher_ok == 0


@pytest.mark.parametrize(
    "classifications",
    [("permanent", "transient"), ("transient", "permanent")],
)
def test_terminal_teacher_failure_preserves_permanent_precedence(classifications):
    bridge = _text_bridge(_BridgeTeacher())

    for classification in classifications:
        bridge._record_teacher_failure(
            classification,
            f"{classification} failure",
            terminal=True,
        )

    assert bridge.teacher_failure == ("permanent", "permanent failure")
    assert bridge.teacher_error == 1
    assert bridge.teacher_transient == 1


def test_all_transient_teacher_samples_exhaust_replacements_as_retriable():
    from flash.engine.worker.perf import RetriableInfraError
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    mutations = []
    bridge = _text_bridge(
        TransientTeacher(), mutation_callback=lambda: mutations.append(True)
    )
    bridge.start()
    try:
        with pytest.raises(RuntimeError, match="after 3 rollout attempts"):
            _exhaust_bridge_no_signal(bridge)
        with pytest.raises(RetriableInfraError, match="after bounded retries"):
            _raise_verl_failure(1, bridge.teacher_failure)
    finally:
        bridge.close()

    assert bridge.score_requests == 3
    assert bridge.teacher_transient == 3
    assert bridge.teacher_ok == 0
    assert bridge.teacher_error == 0
    assert bridge.teacher_failure == ("transient", "teacher unavailable")
    assert bridge.no_signal_resamples == 2
    assert bridge.no_signal_skipped_steps == 1
    assert mutations == []


def test_mixed_transient_and_truncated_exhaustion_remains_retriable():
    from flash.engine.worker.perf import RetriableInfraError
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(TransientTeacher())

    def run_attempt(attempt_ordinal):
        transient = _post_json(
            bridge.url,
            bridge.token,
            "/score",
            _bridge_score_payload(0, [10, 11], [65, 66, 99]),
        )
        truncated = _post_json(
            bridge.url,
            bridge.token,
            "/score",
            _bridge_score_payload(0, [10, 11], [65, 66]),
        )
        assert transient["teacher_ids"] == [-1, -1, -1, -1, -1]
        assert truncated["teacher_ids"] == [-1, -1, -1, -1]
        raise _AllNoSignalBatch(attempt_ordinal)

    bridge.start()
    try:
        with pytest.raises(RuntimeError, match="after 3 rollout attempts"):
            _run_with_no_signal_replacements(
                run_attempt,
                lambda _batch: None,
                lambda: None,
                lambda: _post_json(
                    bridge.url, bridge.token, "/no-signal/resample", {}
                ),
                lambda: _post_json(
                    bridge.url, bridge.token, "/no-signal/abandoned", {}
                ),
            )
        with pytest.raises(RetriableInfraError, match="after bounded retries"):
            _raise_verl_failure(1, bridge.teacher_failure)
    finally:
        bridge.close()

    assert bridge.score_requests == 6
    assert bridge.teacher_transient == 3
    assert bridge.teacher_ok == 0
    assert bridge.truncated_rollouts == 3


def test_mixed_transient_and_teacher_empty_alignment_exhaustion_remains_permanent():
    from flash.engine.worker.teacher import TeacherError

    class AlternatingTeacher:
        def __init__(self):
            self.calls = 0

        def score(self, _prompt_text, _completion_text):
            self.calls += 1
            if self.calls % 2:
                raise TeacherError("teacher unavailable", permanent=False)
            return []

    bridge = _text_bridge(AlternatingTeacher())
    bridge.start()
    try:
        with pytest.raises(RuntimeError, match="after 3 rollout attempts"):
            _exhaust_bridge_no_signal(bridge)
        with pytest.raises(RuntimeError, match="subprocess exited with status 1"):
            _raise_verl_failure(1, bridge.teacher_failure)
    finally:
        bridge.close()

    assert bridge.teacher_failure is None
    assert bridge.teacher_transient == 2
    assert bridge.teacher_ok == 1
    assert bridge.empty_alignments == 1


def test_deterministic_empty_alignment_exhaustion_remains_permanent():
    class EmptyAlignmentTeacher:
        def score(self, _prompt_text, _completion_text):
            return []

    bridge = _text_bridge(EmptyAlignmentTeacher())
    bridge.start()
    try:
        with pytest.raises(RuntimeError, match="after 3 rollout attempts"):
            _exhaust_bridge_no_signal(bridge)
        with pytest.raises(RuntimeError, match="subprocess exited with status 1"):
            _raise_verl_failure(1, bridge.teacher_failure)
    finally:
        bridge.close()

    assert bridge.teacher_failure is None
    assert bridge.teacher_ok == 3
    assert bridge.teacher_transient == 0
    assert bridge.teacher_error == 0
    assert bridge.empty_alignments == 3


def test_multiturn_transient_bridge_failure_latches_terminal_cause():
    from flash.engine.worker.perf import RetriableInfraError
    from flash.engine.worker.teacher import TeacherError

    bridge = _text_bridge(_BridgeTeacher())

    def fail_score(_session_id):
        raise TeacherError("multi-turn teacher unavailable", permanent=False)

    bridge.score_multiturn = fail_score
    bridge.start()
    try:
        with pytest.raises(FlashTeacherBridgeError) as actor_error:
            _post_json(
                bridge.url,
                bridge.token,
                "/multiturn/score",
                {"session_id": "session-1"},
            )
        with pytest.raises(RetriableInfraError, match="after bounded retries"):
            _raise_verl_failure(1, bridge.teacher_failure)
    finally:
        bridge.close()

    assert actor_error.value.classification == "transient"
    assert bridge.teacher_failure == (
        "transient",
        "multi-turn teacher unavailable",
    )
    assert bridge.teacher_transient == 1


def test_bridge_transport_failure_is_typed_retriable(monkeypatch):
    import urllib.error
    import urllib.request

    def offline(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(FlashTeacherBridgeError) as error:
        _post_json("http://127.0.0.1:1", "token", "/score", {})
    assert error.value.classification == "transient"


@pytest.mark.parametrize(
    ("classification", "expected_exit"),
    [("transient", 87), ("permanent", 86)],
)
def test_repeated_mutation_notice_maps_later_bridge_failure_to_child_exit(
    monkeypatch, classification, expected_exit
):
    import flash.engine.worker.opd_verl_plugin as plugin
    from flash.engine.worker.perf import RetriableInfraError

    posts = []

    def post_json(url, token, path, payload):
        posts.append((url, token, path, payload))
        if len(posts) == 2:
            raise FlashTeacherBridgeError(
                "mutation bridge failed",
                classification=classification,
            )
        return {"ok": True}

    class ChildExit(RuntimeError):
        def __init__(self, code):
            super().__init__(code)
            self.code = code

    def child_exit(code):
        raise ChildExit(code)

    monkeypatch.setattr(plugin, "_post_json", post_json)
    monkeypatch.setattr(plugin.os, "_exit", child_exit)

    plugin._post_mutation_notice("http://bridge", "token")
    with pytest.raises(ChildExit) as exit_error:
        plugin._post_mutation_notice("http://bridge", "token")

    assert posts == [
        ("http://bridge", "token", "/mutation", {}),
        ("http://bridge", "token", "/mutation", {}),
    ]
    assert exit_error.value.code == expected_exit
    if classification == "transient":
        with pytest.raises(RetriableInfraError, match="transient teacher bridge failure"):
            _raise_verl_failure(exit_error.value.code, None)
    else:
        with pytest.raises(RuntimeError, match="permanent teacher bridge failure"):
            _raise_verl_failure(exit_error.value.code, None)


@pytest.mark.parametrize(
    ("retriable", "expected_exit"),
    [(True, 87), (False, 86)],
)
def test_mutation_marker_failure_preserves_bridge_classification(
    monkeypatch, retriable, expected_exit
):
    import flash.engine.worker.opd_verl_plugin as plugin
    from flash.engine.worker.perf import RetriableInfraError

    marker_error = (
        RetriableInfraError("marker upload failed")
        if retriable
        else RuntimeError("invalid marker configuration")
    )
    callback_calls = []

    def fail_marker():
        callback_calls.append(True)
        raise marker_error

    bridge = _text_bridge(_BridgeTeacher(), mutation_callback=fail_marker)

    class ChildExit(RuntimeError):
        def __init__(self, code):
            super().__init__(code)
            self.code = code

    def child_exit(code):
        raise ChildExit(code)

    monkeypatch.setattr(plugin.os, "_exit", child_exit)
    bridge.start()
    try:
        with pytest.raises(ChildExit) as exit_error:
            plugin._post_mutation_notice(bridge.url, bridge.token)
    finally:
        bridge.close()

    assert callback_calls == [True]
    assert exit_error.value.code == expected_exit
    if retriable:
        with pytest.raises(RetriableInfraError, match="transient teacher bridge failure"):
            _raise_verl_failure(exit_error.value.code, None)
    else:
        with pytest.raises(RuntimeError, match="permanent teacher bridge failure"):
            _raise_verl_failure(exit_error.value.code, None)


def test_mutation_success_response_disconnect_does_not_latch_marker_failure():
    callback_calls = []
    bridge = _text_bridge(
        _BridgeTeacher(), mutation_callback=lambda: callback_calls.append(True)
    )
    bridge.start()
    handler = bridge._server.RequestHandlerClass

    def disconnect_during_response(_handler, _status, _payload):
        raise BrokenPipeError("client disconnected")

    handler._send_json = disconnect_during_response
    try:
        with pytest.raises(FlashTeacherBridgeError) as transport_error:
            _post_json(bridge.url, bridge.token, "/mutation", {})
    finally:
        bridge.close()

    assert transport_error.value.classification == "transient"
    assert callback_calls == [True]
    assert bridge.mutation_failure is None
    assert bridge.teacher_failure is None


@pytest.mark.parametrize("retriable", [True, False])
def test_mutation_marker_failure_survives_actor_exit_and_generic_driver_status(
    retriable,
):
    from flash.engine.worker.perf import RetriableInfraError

    marker_error = (
        RetriableInfraError("marker upload failed")
        if retriable
        else RuntimeError("invalid marker configuration")
    )

    def fail_marker():
        raise marker_error

    bridge = _text_bridge(_BridgeTeacher(), mutation_callback=fail_marker)
    bridge.start()
    try:
        with pytest.raises(FlashTeacherBridgeError) as actor_error:
            _post_json(bridge.url, bridge.token, "/mutation", {})
    finally:
        bridge.close()

    expected_classification = "transient" if retriable else "permanent"
    assert actor_error.value.classification == expected_classification
    assert bridge.teacher_failure is None
    if retriable:
        with pytest.raises(RetriableInfraError, match="optimizer marker failure"):
            _raise_verl_failure(
                1,
                bridge.teacher_failure,
                bridge.mutation_failure,
            )
    else:
        with pytest.raises(RuntimeError, match="permanent optimizer marker failure"):
            _raise_verl_failure(
                1,
                bridge.teacher_failure,
                bridge.mutation_failure,
            )


def test_parent_maps_teacher_failures_to_fatal_or_retriable_run_errors():
    from flash.engine.worker.perf import RetriableInfraError

    with pytest.raises(RuntimeError, match="permanent teacher failure"):
        _raise_verl_failure(86, ("permanent", "bad credentials"))
    with pytest.raises(RetriableInfraError, match="transient teacher failure"):
        _raise_verl_failure(87, ("transient", "service unavailable"))
    with pytest.raises(RetriableInfraError, match="transient teacher bridge failure"):
        _raise_verl_failure(87, None)
    with pytest.raises(RuntimeError, match="permanent teacher bridge failure"):
        _raise_verl_failure(86, None)
    with pytest.raises(RuntimeError, match="subprocess exited with status 9"):
        _raise_verl_failure(9, None)


def _config(**overrides):
    config = {
        "train_files": ["/w/train.parquet"],
        "val_files": ["/w/val.parquet"],
        "train_batch_size": 8,
        "max_prompt_length": 1024,
        "max_response_length": 512,
        "model_path": "/models/student",
        "lora_rank": 32,
        "lora_alpha": 64,
        "target_modules": "all-linear",
        "learning_rate": 1e-5,
        "local_dir": "/w/checkpoints",
        "save_freq": 20,
        "n_gpus_per_node": 4,
        "ulysses_sequence_parallel_size": 4,
        "seed": 42,
        "project_name": "flash",
        "experiment_name": "opd-test",
        "total_training_steps": 10,
        "group_size": 2,
        "bridge_url": "http://127.0.0.1:1234",
        "bridge_token": "token",
        "kl_penalty_coef": 0.5,
    }
    config.update(overrides)
    return config


def test_sitecustomize_saves_only_exact_required_steps(monkeypatch):
    saved = []

    class FakeCheckpointHandler:
        def save_checkpoint(self, step):
            saved.append(step)
            return step

    modules = {}
    for name in ("verl", "verl.utils", "verl.utils.checkpoint"):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
    modules["verl.utils.checkpoint.checkpoint_handler"] = types.ModuleType(
        "verl.utils.checkpoint.checkpoint_handler"
    )
    modules["verl.utils.checkpoint.checkpoint_handler"].CheckpointHandler = (
        FakeCheckpointHandler
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = _render_opd_sitecustomize(save_at_steps=(3, 7), total_steps=7)
    exec(compile(source, "sitecustomize.py", "exec"), {})
    handler = FakeCheckpointHandler()

    results = [handler.save_checkpoint(step) for step in range(1, 8)]

    assert saved == [3, 7]
    assert results == [None, None, 3, None, None, None, 7]


def test_overrides_match_verl_0_8_sync_distillation_contract():
    overrides = dict(value.split("=", 1) for value in build_opd_verl_overrides(_config()))
    assert overrides["distillation._target_"] == "flash_opd_verl_plugin.FlashRemoteDistillationConfig"
    assert overrides["distillation.distillation_loss.loss_mode"] == "flash_groupwise_reverse_kl"
    assert overrides["distillation.distillation_loss.use_policy_gradient"] == "false"
    assert overrides["distillation.distillation_loss.use_task_rewards"] == "false"
    assert overrides["actor_rollout_ref.actor.loss_agg_mode"] == "seq-mean-token-mean"
    assert overrides["actor_rollout_ref.actor.use_kl_loss"] == "false"
    assert overrides["algorithm.use_kl_in_reward"] == "false"
    assert overrides["actor_rollout_ref.model.use_remove_padding"] == "true"
    # 32k: fused linear-CE must be on so the actor never materializes [tokens, vocab] logits;
    # the distillation loss reads model_output["log_probs"], which the fused path emits exactly.
    assert overrides["actor_rollout_ref.model.use_fused_kernels"] == "true"
    assert overrides["actor_rollout_ref.model.fused_kernel_options.impl_backend"] == "torch"
    assert overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "4"
    assert overrides["actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size"] == "4"
    assert overrides["actor_rollout_ref.rollout.max_model_len"] == "32768"
    assert overrides["distillation.n_gpus_per_node"] == "0"
    assert overrides["distillation.nnodes"] == "0"
    assert overrides["distillation.teacher_key"] == "index"
    assert overrides["data.image_key"] == "images"
    assert overrides["data.return_raw_chat"] == "true"
    assert overrides["data.return_multi_modal_inputs"] == "false"
    assert overrides["actor_rollout_ref.rollout.limit_images"] == "8"
    assert "actor.engine.ulysses_sequence_parallel_size" not in overrides
    assert "ref_log_prob" not in " ".join(overrides)
    assert not any("structured_outputs_config" in key for key in overrides)

    multi_turn_overrides = dict(
        value.split("=", 1)
        for value in build_opd_verl_overrides(
            _config(multi_turn=True, max_sequence_length=1536, max_model_len=1536)
        )
    )
    assert (
        multi_turn_overrides["actor_rollout_ref.rollout.agent.default_agent_loop"]
        == "flash_multi_turn"
    )
    assert multi_turn_overrides["actor_rollout_ref.rollout.prompt_length"] == "1536"
    assert multi_turn_overrides["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == "1536"


def test_structured_overrides_pin_xgrammar_and_thinking_parser():
    overrides = dict(
        value.split("=", 1)
        for value in build_opd_verl_overrides(
            _config(
                thinking=True,
                structured_outputs={"choice": ["4"]},
            )
        )
    )

    assert (
        overrides[
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.structured_outputs_config"
        ]
        == "{backend:xgrammar,disable_any_whitespace:false,reasoning_parser:deepseek_r1}"
    )


def test_child_teacher_bridge_payload_sends_boundary_and_image_count_without_pixels():
    pixel_marker = "raw-image-pixels-must-not-cross-localhost"
    payload = _bridge_score_payload(
        7,
        [10, 11],
        [12],
        {"images": [pixel_marker, pixel_marker], "videos": [pixel_marker]},
    )

    assert payload == {
        "index": 7,
        "prompt_length": 2,
        "sequence_ids": [10, 11, 12],
        "image_count": 2,
    }
    assert pixel_marker not in json.dumps(payload)
    assert _multi_modal_image_count(None) == 0


def test_image_token_suppression_is_gated_on_structured_image_blocks():
    image_prompt = [
        {
            "role": "user",
            "content": [{"type": "image", "image": object()}, {"type": "text", "text": "x"}],
        }
    ]
    assert _raw_prompt_has_image_block(image_prompt)
    assert not _raw_prompt_has_image_block([{"role": "user", "content": "<image> text"}])
    processor = SimpleNamespace(image_token_id=151655)
    assert _resolve_image_token_id(processor, SimpleNamespace()) == 151655
    fallback = SimpleNamespace(
        convert_tokens_to_ids=lambda token: 42 if token == "<|image_pad|>" else None
    )
    assert _resolve_image_token_id(SimpleNamespace(), fallback) == 42


def test_child_environment_keeps_bridge_but_excludes_teacher_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FIREWORKS_API_KEY", "teacher-secret")
    monkeypatch.setenv("HF_TOKEN", "hub-secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    child = _build_opd_child_env(
        shim_dir=str(tmp_path),
        wandb_enabled=False,
        bridge_url="http://127.0.0.1:4444",
        bridge_token="bridge-token",
        seed=42,
        stop_sequences=("</answer>",),
        eos_token_ids=frozenset({1, 2}),
        structured_outputs=None,
        model_vocab_size=248320,
        thinking=False,
    )
    assert child["FLASH_OPD_BRIDGE_URL"] == "http://127.0.0.1:4444"
    assert child["FLASH_OPD_BRIDGE_TOKEN"] == "bridge-token"
    assert child["VERL_USE_EXTERNAL_MODULES"] == "flash_opd_verl_plugin"
    assert child["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "FIREWORKS_API_KEY" not in child
    assert "HF_TOKEN" not in child
    assert "FLASH_OPD_STRUCTURED_OUTPUTS" not in child
    assert "FLASH_OPD_MODEL_VOCAB_SIZE" not in child
    assert "FLASH_OPD_THINKING" not in child


def test_structured_child_environment_carries_only_canonical_replay_inputs(tmp_path):
    child = _build_opd_child_env(
        shim_dir=str(tmp_path),
        wandb_enabled=False,
        bridge_url="http://127.0.0.1:4444",
        bridge_token="bridge-token",
        seed=42,
        stop_sequences=(),
        eos_token_ids=frozenset({1}),
        structured_outputs={"choice": ["4"]},
        model_vocab_size=248320,
        thinking=True,
    )

    assert child["FLASH_OPD_STRUCTURED_OUTPUTS"] == '{"choice":["4"]}'
    assert child["FLASH_OPD_MODEL_VOCAB_SIZE"] == "248320"
    assert child["FLASH_OPD_THINKING"] == "1"


def test_multiturn_child_environment_carries_only_rollout_capabilities(tmp_path):
    child = _build_opd_child_env(
        shim_dir=str(tmp_path),
        wandb_enabled=False,
        bridge_url="http://127.0.0.1:4444",
        bridge_token="bridge-token",
        seed=42,
        stop_sequences=(),
        eos_token_ids=frozenset({1}),
        structured_outputs=None,
        model_vocab_size=248320,
        thinking=False,
        multi_turn=True,
        max_turns=6,
        max_model_len=4096,
    )

    assert child["FLASH_OPD_MAX_TURNS"] == "6"
    assert child["FLASH_OPD_MAX_MODEL_LEN"] == "4096"
    assert json.loads(child["FLASH_OPD_ENV_CAPABILITIES"]) == [
        "new_rollout_state",
        "record_model_turn",
        "env_reply",
        "rollout_done",
    ]
    assert "FIREWORKS_API_KEY" not in child








def test_structured_validator_rejects_vllm_mistral_tokenizer_models(monkeypatch):
    from flash import opd_verl_validation

    monkeypatch.setattr(
        opd_verl_validation,
        "_resolve_structured_model_metadata",
        lambda *_args: (32000, ("config.json", "tokenizer.model.v3")),
    )

    with pytest.raises(ValueError, match="does not support vLLM MistralTokenizer"):
        validate_opd_verl_structured_outputs(
            '{"choice":["4"]}',
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            compiler_vocab_size=32000,
        )


def test_unstructured_validator_does_not_resolve_model_metadata(monkeypatch):
    from flash import opd_verl_validation

    def unexpected(*_args, **_kwargs):
        pytest.fail("unstructured OPD must not resolve structured model metadata")

    monkeypatch.setattr(opd_verl_validation, "_resolve_compiler_vocab_size", unexpected)
    monkeypatch.setattr(opd_verl_validation, "_resolve_structured_model_metadata", unexpected)

    result = validate_opd_verl_structured_outputs(
        None,
        model_id="open-org/cached-model",
        model_revision="d" * 40,
        model_policy="allow",
    )

    assert result.constraint is None
    assert result.model_vocab_size == 0




def test_structured_runtime_version_mismatch_fails_before_generation(monkeypatch):
    versions = {"verl": "0.8.0", "vllm": "0.11.1", "xgrammar": "0.1.25"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match=r"vllm 0\.11\.0 exactly"):
        _require_structured_runtime_versions()


def test_structured_runtime_rejects_wrong_xgrammar_version(monkeypatch):
    versions = {"verl": "0.8.0", "vllm": "0.11.0", "xgrammar": "0.1.26"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match=r"xgrammar 0\.1\.25 exactly"):
        _require_structured_runtime_versions()


def test_plugin_initialization_checks_structured_versions_before_verl_install(monkeypatch):
    import importlib as importlib_module

    from flash.engine.worker import opd_verl_plugin as plugin

    versions = {"verl": "0.8.0", "vllm": "0.11.0", "xgrammar": "0.1.26"}
    monkeypatch.setenv("FLASH_OPD_STRUCTURED_OUTPUTS", '{"choice":["4"]}')
    monkeypatch.setattr(plugin.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(
        plugin.importlib.util,
        "find_spec",
        lambda _name: pytest.fail("verl discovery must follow the structured version gate"),
    )

    with pytest.raises(RuntimeError, match=r"xgrammar 0\.1\.25 exactly"):
        importlib_module.reload(plugin)

    monkeypatch.delenv("FLASH_OPD_STRUCTURED_OUTPUTS")
    monkeypatch.setattr(plugin.importlib.util, "find_spec", lambda _name: None)
    importlib_module.reload(plugin)










def test_deterministic_seed_uses_every_rollout_identity_component():
    baseline = deterministic_rollout_seed(42, 3, 7, 1)
    assert baseline == deterministic_rollout_seed(42, 3, 7, 1)
    assert len(
        {
            baseline,
            deterministic_rollout_seed(43, 3, 7, 1),
            deterministic_rollout_seed(42, 4, 7, 1),
            deterministic_rollout_seed(42, 3, 8, 1),
            deterministic_rollout_seed(42, 3, 7, 2),
            deterministic_rollout_seed(42, 3, 7, 1, no_signal_attempt_ordinal=1),
        }
    ) == 6
    assert 0 <= baseline < 2**63




def test_worker_structured_validator_runs_before_model_download():
    import inspect

    from flash.engine.worker.opd_verl import run_opd_verl

    source = inspect.getsource(run_opd_verl)
    assert source.index("validate_opd_verl_structured_outputs(") < source.index(
        "_w.prefetch_model("
    )
    assert "resolve_vocab_size" not in source


def test_plugin_registers_external_trainer_without_teacher_gpu_pool():
    import inspect

    import flash.engine.worker.opd_verl_plugin as plugin

    source = inspect.getsource(plugin)
    assert '@register_distillation_loss(' in source
    assert 'names=["flash_groupwise_reverse_kl"]' in source
    assert 'main_ppo_sync.TaskRunner = FlashTaskRunner' in source
    assert 'resource_pool_spec = {' in source
    assert '"global_pool"' in source
    assert "teacher_pool" not in source
    assert "Role.TeacherModel" not in source
    assert 'params["structured_outputs"]' in source
    assert "build_flash_multi_turn_agent_loop" in source
    assert "AgentLoopWorkerTQ._agent_loop_postprocess" not in source
    assert "_run_with_no_signal_replacements" in source
    assert 'params["logprobs"]' not in source


def test_plugin_identifiers_remain_provider_neutral():
    import inspect

    import flash.engine.worker.opd_verl_multiturn as multiturn
    import flash.engine.worker.opd_verl_plugin as plugin
    import flash.engine.worker.opd_verl_structured as structured

    source = (
        inspect.getsource(plugin)
        + inspect.getsource(structured)
        + inspect.getsource(multiturn)
    ).lower()
    forbidden = ("parasail", "fireworks")
    for name in forbidden:
        assert f"class {name}" not in source
        assert f"def {name}" not in source
        assert f"_{name}" not in source


def test_opd_backend_selector_routes_to_verl(monkeypatch):
    import flash.engine.worker.opd as opd_mod
    import flash.engine.worker.opd_verl as ov
    called = []
    monkeypatch.setattr(ov, "run_opd_verl", lambda *a, **k: called.append(True))
    monkeypatch.setenv("FLASH_OPD_BACKEND", "verl")
    opd_mod.run_opd()
    assert called == [True]
    monkeypatch.setenv("FLASH_OPD_BACKEND", "bogus")
    with pytest.raises(RuntimeError, match="not a known opd backend"):
        opd_mod.run_opd()


def test_worker_fails_closed_on_tool_env(monkeypatch):
    # multi-turn is now supported; native tool-calling OPD still fails closed at the worker layer.
    import flash.engine.worker.opd_verl as ov

    class FakeEnv:
        is_tool_env = True
        multi_turn = False

    monkeypatch.setattr(ov._w, "require_active_env", lambda: FakeEnv())
    with pytest.raises(RuntimeError, match="native tool-calling OPD environments are not supported"):
        ov.run_opd_verl(spec=object())
