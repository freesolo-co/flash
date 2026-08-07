"""CPU contracts for the OPD migration to verl 0.8.0."""

from __future__ import annotations

import hashlib
import http.client
import importlib.metadata
import importlib.util
import inspect
import json
import math
import multiprocessing
import os
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.engine.worker import opd_train, rl_train
from flash.engine.worker.opd_plugin import (
    FlashTeacherBridgeError,
    _AllNoSignalBatch,
    _bridge_score_payload,
    _flash_groupwise_reverse_kl_values,
    _full_sequence_signal_sequences,
    _init_transfer_queue,
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
from flash.engine.worker.opd_structured import (
    StructuredOutputReplay,
    _count_legal_tokens,
    canonical_structured_spec,
)
from flash.engine.worker.opd_train import (
    _OPD_PARQUET_WRITE_BATCH_ROWS,
    _BridgePrompt,
    _build_opd_child_env,
    _failure_accounting_metadata,
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
    _TextTeacherBatcher,
    _trim_response_and_forced,
    _validate_forced_mask,
    _write_opd_parquet,
    build_opd_overrides,
    encode_shifted_group_metadata,
)
from flash.engine.worker.teacher import TeacherScore
from flash.engine.worker.tokenizer_align import TeacherToken
from flash.opd_retry_contract import OPD_RESUME_STATE_VERSION
from flash.opd_validation import validate_opd_structured_outputs


def _reference_groupwise_reverse_kl(sp_t, groups, kl_coef=1.0):
    """Straight-line reference for the groupwise reverse-KL used by the equivalence tests below.

    Deliberately written as an INDEPENDENT formulation rather than imported from the worker: the
    point of the equivalence assertions is to pin verl's vectorized implementation to the objective
    on paper, and an oracle that shares code with the thing it checks can drift with it and still
    agree. Per group g over student token indices I_g, the coefficient is
    ``kl_coef * (sum_{i in I_g} sp_i - teacher_logsum_g) / |I_g|``, applied to each member token and
    averaged over all grouped tokens. The coefficient uses DETACHED student logprobs, so the
    gradient flows only through the trailing factor -- that asymmetry is the objective, not an
    optimization, so a reference that differentiates both factors would disagree by design.
    """
    import torch

    if sp_t is None or not groups:
        return None
    terms = []
    for s_idx, teacher_logsum in groups:
        if not s_idx:
            continue
        idx = [int(i) for i in s_idx]
        student_logsum = sum(sp_t[i].detach() for i in idx)
        coeff = kl_coef * (student_logsum - float(teacher_logsum)) / len(idx)
        terms.extend(coeff * sp_t[i] for i in idx)
    if not terms:
        return None
    return torch.stack(terms).mean()


def _write_mutation_failure_after_start(start, classification, message):
    from flash.engine.worker.opd_plugin import _write_mutation_failure_fallback

    start.wait()
    _write_mutation_failure_fallback(classification, message)


def _send_truncated_json_response(handler, status, payload):
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded) + 1))
    handler.end_headers()
    handler.wfile.write(encoded)
    handler.wfile.flush()
    handler.close_connection = True


def _send_malformed_json_response(handler, status, _payload):
    encoded = b"{"
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)
    handler.wfile.flush()


def _capture_client_only_score_delivery_loss(bridge, monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    failure_path = str(tmp_path / "score-delivery-failure")
    monkeypatch.setenv("FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH", failure_path)

    class ChildExit(RuntimeError):
        def __init__(self, code):
            super().__init__(code)
            self.code = code

    def child_exit(code):
        raise ChildExit(code)

    monkeypatch.setattr(plugin.os, "_exit", child_exit)
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    handler._send_json = _send_truncated_json_response
    try:
        with pytest.raises(FlashTeacherBridgeError) as transport_error:
            _post_json(
                bridge.url,
                bridge.token,
                "/score",
                _bridge_score_payload(0, [10, 11], [65, 66, 99]),
            )
        with pytest.raises(ChildExit) as actor_exit:
            plugin._exit_for_score_failure(transport_error.value)
        fallback = _read_classified_failure_fallback(failure_path)
        records = list(tmp_path.glob("score-delivery-failure.*.transient.json"))
    finally:
        bridge.close()

    assert [record.name for record in records] == [
        f"score-delivery-failure.{os.getpid()}.transient.json"
    ]
    assert not [path for path in tmp_path.iterdir() if path.name.endswith(".tmp")]
    return transport_error.value, actor_exit.value.code, fallback


class _ThreadedDistributedWorld:
    def __init__(self, size):
        self.size = size
        self.local = threading.local()
        self.readiness = threading.Barrier(size)
        self.broadcast_ready = threading.Barrier(size)
        self.broadcast_copied = threading.Barrier(size)
        self.confirmation = threading.Barrier(size)
        self.broadcast_value = None
        self.events = []

    def run(self, rank, operation):
        self.local.rank = rank
        self.local.barrier_index = 0
        return operation()

    def get_rank(self):
        return self.local.rank

    def barrier(self):
        phase = self.local.barrier_index
        self.local.barrier_index += 1
        self.events.append(("barrier", phase, self.local.rank))
        target = self.readiness if phase == 0 else self.confirmation
        target.wait(timeout=5)

    def broadcast_object_list(self, values, src):
        self.events.append(("broadcast", self.local.rank))
        if self.local.rank == src:
            self.broadcast_value = values[0]
        self.broadcast_ready.wait(timeout=5)
        if self.local.rank != src:
            values[0] = self.broadcast_value
        self.broadcast_copied.wait(timeout=5)


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


def test_groupwise_reverse_kl_scalar_and_analytic_gradient_match_reference():
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

    reference_losses = [
        _reference_groupwise_reverse_kl(
            student[row],
            [([0], float(teacher_logsums[row, 0])), ([1, 2], float(teacher_logsums[row, 1]))],
            kl_coef=coef,
        )
        for row in range(2)
    ]
    reference_loss = torch.stack(reference_losses).mean()
    reference_gradient = torch.autograd.grad(reference_loss, student)[0]

    assert torch.allclose(verl_loss, reference_loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(verl_gradient, reference_gradient, atol=1e-12, rtol=1e-12)
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
        reference = torch.stack(
            [
                _reference_groupwise_reverse_kl(
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
        reference_gradient = torch.autograd.grad(reference, student)[0]

        assert config.global_batch_info == {
            "dp_size": 1,
            "batch_num_tokens": batch_num_tokens,
            "global_batch_size": global_batch_size,
            "loss_scale_factor": None,
        }
        assert torch.allclose(actual, reference, atol=1e-12, rtol=1e-12)
        assert torch.allclose(actual_gradient, reference_gradient, atol=1e-12, rtol=1e-12)
        observed.append(float(actual.detach()))
        expected.append(float(reference.detach()))

    assert observed == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert len(steps[0]["student"]) != len(steps[1]["student"])


def test_no_signal_sequence_is_excluded_before_actor_training():
    torch = pytest.importorskip("torch")
    group_ids = torch.tensor([[0, -1, -1], [-1, -1, -1], [2, 2, -1]])
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    assert _signal_sequences(group_ids, response_mask).tolist() == [True, False, True]
    full_sequence_ids = torch.tensor([[-1, -1, 0, -1], [-1, -1, -1, -1], [-1, 2, 2, -1]]).unsqueeze(
        -1
    )
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
    assert seeds[1] == deterministic_rollout_seed(42, 3, 7, 1, no_signal_attempt_ordinal=1)
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
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]"))
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
        (
            {
                "json": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                }
            },
            '{"a":"b"}',
        ),
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

    forced = replay.forced_mask([], response_ids, canonical_structured_spec(spec), thinking=False)

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


def _opd_row(index: int, *, multimodal: bool) -> dict:
    row = {
        "prompt": [{"role": "user", "content": f"prompt {index}"}],
        "data_source": "flash_opd",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"index": index},
    }
    if multimodal:
        row["images"] = [{"image": f"file:///tmp/{index}.png"}]
    return row


@pytest.mark.parametrize("multimodal", [False, True])
def test_opd_parquet_spanning_write_batches_preserves_every_row_in_order(tmp_path, multimodal):
    """the horizon row list is written in batches, so a batch boundary must not drop or reorder.

    verl runs the dataloader with data.shuffle=false, so parquet row order IS the training order.
    """
    datasets = pytest.importorskip("datasets")
    # deliberately not a multiple of the batch size, so the final batch is a short one
    count = _OPD_PARQUET_WRITE_BATCH_ROWS * 2 + 3
    rows = [_opd_row(index, multimodal=multimodal) for index in range(count)]
    path = tmp_path / "horizon.parquet"

    _write_opd_parquet(rows, str(path))

    restored = datasets.load_dataset("parquet", data_files=str(path))["train"].to_list()
    assert restored == rows
    assert [row["extra_info"]["index"] for row in restored] == list(range(count))


def test_opd_parquet_repeated_prompt_references_survive_batching(tmp_path):
    """each row holds a shared reference to one pooled prompt; batching must copy, not alias."""
    datasets = pytest.importorskip("datasets")
    prompt = [{"role": "user", "content": "shared"}]
    rows = [
        {
            "prompt": prompt,
            "data_source": "flash_opd",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"index": ordinal % 3},
        }
        for ordinal in range(_OPD_PARQUET_WRITE_BATCH_ROWS + 5)
    ]
    path = tmp_path / "repeated.parquet"

    _write_opd_parquet(rows, str(path))

    restored = datasets.load_dataset("parquet", data_files=str(path))["train"].to_list()
    assert restored == rows


def test_opd_parquet_leaves_no_truncated_file_when_a_later_batch_fails(tmp_path):
    """closing a partly written parquet still emits a valid footer.

    without the atomic rename the failed write would leave a READABLE short file, and since the
    horizon parquet is the training schedule that is a silently truncated run rather than an error.
    """
    rows = [_opd_row(index, multimodal=False) for index in range(_OPD_PARQUET_WRITE_BATCH_ROWS + 5)]
    # a type the pinned schema cannot accept, landing in the second batch
    rows[-1]["extra_info"] = {"index": "not-an-int"}
    path = tmp_path / "doomed.parquet"

    pa = pytest.importorskip("pyarrow")
    with pytest.raises(pa.ArrowInvalid):
        _write_opd_parquet(rows, str(path))

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_opd_parquet_pins_the_image_type_when_the_first_batch_has_no_images(tmp_path):
    """a multimodal job may still round-robin a run of image-free prompts into the first batch.

    inferring the schema from that batch would type ``images`` as a list of nulls and reject the
    first row that actually carries one, so the declared multimodal features must pin it.
    """
    datasets = pytest.importorskip("datasets")
    rows = [_opd_row(index, multimodal=True) for index in range(_OPD_PARQUET_WRITE_BATCH_ROWS + 2)]
    for row in rows[:_OPD_PARQUET_WRITE_BATCH_ROWS]:
        row["images"] = []
    path = tmp_path / "late-image.parquet"

    _write_opd_parquet(rows, str(path))

    restored = datasets.load_dataset("parquet", data_files=str(path))["train"].to_list()
    assert restored == rows
    assert restored[_OPD_PARQUET_WRITE_BATCH_ROWS]["images"] == [
        {"image": f"file:///tmp/{_OPD_PARQUET_WRITE_BATCH_ROWS}.png"}
    ]


def test_opd_parquet_rejects_an_empty_row_list(tmp_path):
    """a zero-row write would leave verl pointing at a file with no training data."""
    with pytest.raises(ValueError, match="empty OPD parquet"):
        _write_opd_parquet([], str(tmp_path / "empty.parquet"))


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
        "contract_version": OPD_RESUME_STATE_VERSION,
        "seed": 42,
        "opt_steps": step,
        "step": step,
        "rollout_seed_ordinal": 12,
        "prompt_pool_fingerprint": "a" * 64,
        "generated_tokens": 41,
        "teacher_input_tokens": 37,
        "teacher_output_tokens": 6,
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


def test_failure_accounting_metadata_uses_canonical_train_meta_contract():
    accounting = {
        "teacher_transient": 3,
        "teacher_error": 2,
        "no_signal_resamples": 5,
        "no_signal_skipped_steps": 1,
        "skip_counts": {
            "truncated_rollout": 4,
            "empty_alignment": 2,
            "empty_completion": 3,
        },
    }

    metadata = _failure_accounting_metadata(accounting)

    assert metadata == {
        "teacher_transient_failures": 3,
        "teacher_errors": 2,
        "no_signal_resamples": 5,
        "no_signal_skipped_steps": 1,
        "skip_reasons": {
            "alignment_empty": 2,
            "empty_completion": 3,
            "truncated_rollout": 4,
        },
    }
    assert list(metadata["skip_reasons"]) == [
        "alignment_empty",
        "empty_completion",
        "truncated_rollout",
    ]
    counters = [
        metadata["teacher_transient_failures"],
        metadata["teacher_errors"],
        metadata["no_signal_resamples"],
        metadata["no_signal_skipped_steps"],
        *metadata["skip_reasons"].values(),
    ]
    assert all(type(counter) is int and math.isfinite(counter) for counter in counters)
    assert (
        not {
            "teacher_transient",
            "teacher_error",
            "mean_align_granularity",
        }
        & metadata.keys()
    )


def test_failure_accounting_metadata_omits_zero_skip_reasons():
    # the verl snapshot always injects empty_alignment, but trl's counter only
    # records reasons that occurred, so a clean run must persist no reasons.
    metadata = _failure_accounting_metadata(
        {
            "teacher_transient": 0,
            "teacher_error": 0,
            "no_signal_resamples": 0,
            "no_signal_skipped_steps": 0,
            "skip_counts": {"empty_alignment": 0, "truncated_rollout": 2},
        }
    )

    assert metadata["skip_reasons"] == {"truncated_rollout": 2}


def test_failure_accounting_metadata_uses_trl_skip_reason_names():
    # trl publishes this condition as alignment_empty via _sample_skip_reason,
    # so consumers aggregating across backends see one category, not two.
    metadata = _failure_accounting_metadata(
        {
            "teacher_transient": 0,
            "teacher_error": 0,
            "no_signal_resamples": 0,
            "no_signal_skipped_steps": 0,
            "skip_counts": {"empty_alignment": 3},
        }
    )

    assert metadata["skip_reasons"] == {"alignment_empty": 3}


def test_write_train_meta_integrates_canonical_failure_accounting_metadata():
    import inspect

    from flash.engine.worker.opd_train import run_opd_train

    source = inspect.getsource(run_opd_train)
    write_train_meta_source = source[source.index("_w.write_train_meta(") :]

    assert "**_failure_accounting_metadata(final_accounting)" in write_train_meta_source
    assert '"teacher_transient":' not in write_train_meta_source
    assert '"teacher_error":' not in write_train_meta_source
    # granularity IS reported, but only from its own accumulator. the snapshot's granularity_sum /
    # granularity_n are legacy aliases holding COVERAGE, so deriving the ratio from them would
    # publish coverage a second time under the name of the one signal coverage cannot provide.
    assert '"mean_align_granularity":' in write_train_meta_source
    assert 'final_accounting["align_group_sum"]' in write_train_meta_source
    assert 'final_accounting["align_group_n"]' in write_train_meta_source
    assert "granularity_sum" not in write_train_meta_source
    assert "granularity_n" not in write_train_meta_source


def _teacher_score(tokens, *, input_tokens=5, output_tokens=1):
    return TeacherScore(
        tokens=tuple(tokens),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class _BridgeTeacher:
    def score(self, prompt_text, completion_text):
        assert prompt_text == "User: question\nAssistant: "
        assert completion_text == "AB"
        return _teacher_score(
            [
                TeacherToken(text="A", logprob=-0.4, start=0, end=1),
                TeacherToken(text="B", logprob=-0.7, start=1, end=2),
            ]
        )


class _MergedBridgeTeacher:
    def score(self, prompt_text, completion_text):
        assert prompt_text == "User: question\nAssistant: "
        assert completion_text == "AB"
        return _teacher_score([TeacherToken(text="AB", logprob=-1.1, start=0, end=2)])


class _ScoreManyTeacherAdapter:
    def __init__(self, teacher):
        self.teacher = teacher

    @staticmethod
    def _with_usage(score):
        if isinstance(score, TeacherScore):
            return score
        return _teacher_score(score)

    def score(self, prompt_text, completion_text):
        return self._with_usage(self.teacher.score(prompt_text, completion_text))

    def score_many(self, items):
        return [
            self._with_usage(self.teacher.score(prompt_text, completion_text))
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
        mutation_callback=(mutation_callback if mutation_callback is not None else lambda: None),
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
                _teacher_score([TeacherToken(text="AB", logprob=-1.0, start=0, end=3)])
                for _item in items
            ]
        return [
            _teacher_score(
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
            )
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


def test_bridge_granularity_separates_a_collapsed_alignment_from_a_healthy_one():
    # both teachers cover every student token, so coverage is 1.0 for BOTH and cannot tell them
    # apart. the merged teacher returns one span where the healthy one returns two, which is the
    # degenerate collapse mean_coverage is blind to -- granularity is the signal that sees it.
    healthy = _text_bridge(_BridgeTeacher())
    healthy.score(0, 2, [10, 11, 65, 66, 99])
    collapsed = _text_bridge(_MergedBridgeTeacher())
    collapsed.score(0, 2, [10, 11, 65, 66, 99])

    assert healthy.coverage_sum == collapsed.coverage_sum == 1.0
    assert healthy.aligned_sequences == collapsed.aligned_sequences == 1
    # two student tokens over two groups vs two student tokens over one merged group.
    assert healthy.align_group_sum == 1.0
    assert collapsed.align_group_sum == 2.0
    assert healthy.align_group_n == collapsed.align_group_n == 1


def test_bridge_granularity_counts_only_sequences_that_reached_the_loss():
    # a teacher that returns no tokens still reaches the accounting, but its sequence carries no
    # groups and contributes nothing to the loss. folding its 0.0 into the mean would report a
    # collapse that never happened -- it belongs in empty_alignments only.
    class _EmptyAlignmentTeacher:
        def score(self, _prompt_text, _completion_text):
            return []

    bridge = _text_bridge(_BridgeTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    assert bridge.align_group_n == 1
    assert bridge.align_group_sum == 1.0

    bridge.teacher = _ScoreManyTeacherAdapter(_EmptyAlignmentTeacher())
    bridge._text_teacher_batcher = None
    bridge.score(0, 2, [10, 11, 65, 66, 99])

    assert bridge.empty_alignments == 1
    # the empty sequence moved empty_alignments, and left the granularity mean untouched.
    assert bridge.align_group_n == 1
    assert bridge.align_group_sum == 1.0


def test_bridge_granularity_survives_resume_without_reading_the_coverage_aliases():
    # granularity_sum / granularity_n are LEGACY ALIASES for coverage. resuming granularity from
    # them would silently restore coverage under the granularity name, so an old-format state must
    # restart the accumulator at zero rather than inherit the wrong quantity.
    resumed = _text_bridge(_BridgeTeacher())
    state = dict(_resume_accounting())
    state.update({"align_group_sum": 6.0, "align_group_n": 3})
    carried = _TeacherAlignmentBridge(
        prompts=list(resumed.prompts),
        tokenizer=_BridgeTokenizer(),
        teacher=_ScoreManyTeacherAdapter(_BridgeTeacher()),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
        initial_state=state,
    )
    assert carried.align_group_sum == 6.0
    assert carried.align_group_n == 3
    snapshot = carried.accounting_snapshot()
    assert snapshot["align_group_sum"] == 6.0
    assert snapshot["align_group_n"] == 3

    legacy_only = _TeacherAlignmentBridge(
        prompts=list(resumed.prompts),
        tokenizer=_BridgeTokenizer(),
        teacher=_ScoreManyTeacherAdapter(_BridgeTeacher()),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
        initial_state=_resume_accounting(),
    )
    # granularity_sum is 3.5 and granularity_n is 5 in that state; both must be ignored here.
    assert legacy_only.align_group_sum == 0.0
    assert legacy_only.align_group_n == 0
    # the coverage accumulators, whose aliases those really are, DO still resume from them.
    assert legacy_only.coverage_sum == 3.5
    assert legacy_only.aligned_sequences == 5


def test_text_teacher_batcher_enforces_max_batch_size_across_concurrent_requests():
    # the batcher's bound is OPD_TEACHER_SCORING_CONCURRENCY, a MEASURED provider ceiling that moves
    # when the provider is re-measured (it went 8 -> 32). so the prompt count is derived from it
    # rather than hardcoded: the invariant under test is "a full batch is reached and no batch ever
    # exceeds the bound", and a fixed 17 prompts stopped exercising that the moment the ceiling rose
    # above 17 -- the test would have passed while never once filling a batch.
    from flash.opd_limits import OPD_TEACHER_SCORING_CONCURRENCY as bound

    # one full batch plus a partial remainder, capped by the server's own listen backlog: every
    # prompt is posted concurrently, so more prompts than _TEXT_TEACHER_REQUEST_BACKLOG (64) cannot
    # be queued and the assertion above would fail on the harness rather than on the batcher. at the
    # measured ceiling of 32 that leaves exactly 2x headroom, so this is now a real adjacency: a
    # future ceiling above 32 must raise the backlog with it.
    total = min(bound + bound // 2, opd_train._TEXT_TEACHER_REQUEST_BACKLOG)
    assert total > bound, "prompt count must cross the batch bound for this test to mean anything"
    prompt_texts = [f"question-{index}" for index in range(total)]
    teacher = _BatchingTeacher(prompt_texts)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    assert bridge._server is not None
    assert bridge._server.request_queue_size >= len(prompt_texts)
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(total))
    finally:
        bridge.close()

    assert all(status == "ok" for status, _result in outcomes)
    batch_sizes = [len(batch) for batch in teacher.batches]
    # every prompt is scored exactly once, and no batch ever exceeds the bound. those are the
    # batcher's actual guarantees under concurrent posts.
    assert sum(batch_sizes) == total
    assert all(size <= bound for size in batch_sizes)
    # deliberately NOT asserted: that some batch reaches exactly `bound`. _take_batch waits at most
    # flush_wait_s (0.1s) for the batch to fill and then ships whatever arrived, so a full batch
    # depends on posts outrunning that window -- a race, not a contract. the old test asserted
    # max == 8 and passed only because 8 was small enough that 17 concurrent posts always beat the
    # flush; at the measured ceiling of 32 the same code produces 16/16/16 and the assertion fails
    # on scheduling rather than on any defect. more than one batch is still required, which is what
    # proves chunking happened at all.
    assert len(batch_sizes) > 1


def test_text_teacher_batcher_never_takes_more_than_max_batch_size():
    # the concurrent test above CANNOT observe the bound being exceeded: with a 0.1s flush window
    # the pending queue never accumulates past the ceiling, so widening the slice in _take_batch
    # leaves it green (verified by mutation). this pins the slice directly against a pre-filled
    # queue -- no threads, no timer -- so the bound is falsifiable rather than merely unreached.
    batcher = opd_train._TextTeacherBatcher(object(), max_batch_size=4, flush_wait_s=0.01)
    batcher._pending = [
        opd_train._TextTeacherWaiter((f"prompt-{index}", "completion"), enqueued_at=0.0)
        for index in range(10)
    ]

    batch = batcher._take_batch()

    assert batch is not None
    assert len(batch) == 4
    # the taken batch is removed from the queue, in order, so the remainder is the tail.
    assert [waiter.item[0] for waiter in batch] == [f"prompt-{index}" for index in range(4)]
    assert [waiter.item[0] for waiter in batcher._pending] == [
        f"prompt-{index}" for index in range(4, 10)
    ]


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
    from flash.engine.worker import opd_train as opd_train_mod

    monkeypatch.setattr(opd_train_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
    prompt_texts = ["same question"] * 8
    teacher = _BatchingTeacher(prompt_texts)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(8), via_http=False)
    finally:
        bridge.close()

    assert teacher.batches == [[("User: same question\nAssistant: ", "AB")]]
    assert all(status == "ok" for status, _result in outcomes)
    assert [_teacher_logsum(result) for _status, result in outcomes] == [-1.0] * 8
    assert bridge.score_requests == 8
    assert bridge.teacher_ok == 8
    assert bridge.teacher_input_tokens == 5
    assert bridge.teacher_output_tokens == 1


def test_text_teacher_batcher_marks_dedup_copy_as_explicit_unbilled_score():
    teacher = _BatchingTeacher(["same question"])
    batcher = _TextTeacherBatcher(teacher, max_batch_size=8, flush_wait_s=1.0)
    start = threading.Barrier(2)
    batcher.start()

    def score():
        start.wait(timeout=2.0)
        return batcher.score("User: same question\nAssistant: ", "AB")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [
                future.result(timeout=5.0) for future in [executor.submit(score) for _ in range(2)]
            ]
    finally:
        batcher.close()

    assert all(isinstance(result, TeacherScore) for result in results)
    assert all(isinstance(result.tokens, tuple) for result in results)
    assert results[0].tokens is results[1].tokens
    assert sorted((result.input_tokens, result.output_tokens) for result in results) == [
        (0, 0),
        (5, 1),
    ]


def test_text_teacher_batcher_keeps_nonidentical_inputs_separate_and_ordered(monkeypatch):
    from flash.engine.worker import opd_train as opd_train_mod

    monkeypatch.setattr(opd_train_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
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
    from flash.engine.worker import opd_train as opd_train_mod

    monkeypatch.setattr(opd_train_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
    prompt_texts = ["rounding"] * 8
    teacher = _BatchingTeacher(prompt_texts, token_logprob=1e-9)
    bridge = _batching_bridge(teacher, prompt_texts)
    bridge.start()
    try:
        outcomes = _concurrent_bridge_scores(bridge, range(8))
    finally:
        bridge.close()

    assert teacher.batches
    assert all(batch == [("User: rounding\nAssistant: ", "AB")] for batch in teacher.batches)
    assert all(status == "ok" for status, _result in outcomes)
    assert [_teacher_logsum(result) for _status, result in outcomes] == [1e-9] * 8
    assert bridge.score_requests == 8
    assert bridge.teacher_ok == 8
    assert bridge.teacher_error == 0
    assert bridge.teacher_transient == 0


def test_text_teacher_batch_rejects_positive_value_above_tolerance_for_every_waiter(
    monkeypatch,
):
    from flash.engine.worker import opd_train as opd_train_mod

    monkeypatch.setattr(opd_train_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
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
        batch == [("User: invalid rounding\nAssistant: ", "AB")] for batch in teacher.batches
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
    from flash.engine.worker import opd_train as opd_train_mod

    monkeypatch.setattr(opd_train_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
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
    assert all(result["teacher_ids"] == [-1, -1, -1, -1, -1] for _status, result in outcomes)
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
    from flash.engine.worker import opd_train as opd_train_mod

    monkeypatch.setattr(opd_train_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 1.0)
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
        assert _teacher_logsum(result) == teacher.logprobs[f"User: {prompt_text}\nAssistant: "]
    assert bridge.score_requests == 8
    assert bridge.teacher_ok == 8
    assert bridge.teacher_input_tokens == 15
    assert bridge.teacher_output_tokens == 3
    assert bridge.aligned_sequences == 8


def test_text_teacher_batcher_close_allows_inflight_scatter_within_bound():
    class BlockingTeacher:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.items = None

        def score_many(self, items):
            self.items = list(items)
            self.entered.set()
            assert self.release.wait(timeout=2.0)
            return [
                _teacher_score(
                    [
                        TeacherToken(
                            text=completion_text,
                            logprob=-float(int(prompt_text.removeprefix("prompt-")) + 1),
                            start=0,
                            end=len(completion_text),
                        )
                    ]
                )
                for prompt_text, completion_text in items
            ]

    teacher = BlockingTeacher()
    batcher = _TextTeacherBatcher(teacher, max_batch_size=8, flush_wait_s=5.0)
    batcher.start()
    start = threading.Barrier(8)

    def score(index):
        start.wait(timeout=2.0)
        return batcher.score(f"prompt-{index}", "AB")

    executor = ThreadPoolExecutor(max_workers=8)
    futures = [executor.submit(score, index) for index in range(8)]
    close_thread = None
    try:
        assert teacher.entered.wait(timeout=2.0)
        close_thread = threading.Thread(target=lambda: batcher.close(timeout_s=1.0))
        close_thread.start()
        with batcher._condition:
            assert batcher._condition.wait_for(lambda: batcher._closed, timeout=1.0)
        teacher.release.set()
        results = [future.result(timeout=2.0) for future in futures]
        close_thread.join(timeout=2.0)
        assert not close_thread.is_alive()
    finally:
        teacher.release.set()
        if close_thread is not None:
            close_thread.join(timeout=2.0)
        batcher.close(timeout_s=0.1)
        executor.shutdown(wait=True)

    assert teacher.items is not None
    assert len(teacher.items) == 8
    for index, result in enumerate(results):
        # the batcher hands back the full TeacherScore so authoritative provider usage stays
        # attached to the scored tokens; scatter must preserve per-caller token identity.
        assert result.tokens == (
            TeacherToken(text="AB", logprob=-float(index + 1), start=0, end=2),
        )


def test_text_teacher_batcher_shutdown_cannot_strand_pending_bridge_waiter(monkeypatch):
    from flash.engine.worker import opd_train as opd_train_mod

    monkeypatch.setattr(opd_train_mod, "_TEXT_TEACHER_FLUSH_WAIT_S", 10.0)
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


def test_recovered_transient_lost_response_promotes_transient_terminal_cause():
    from flash.engine.worker.perf import RetriableInfraError
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(TransientTeacher())
    bridge.start()
    handler = bridge._server.RequestHandlerClass

    def disconnect_during_response(_handler, _status, _payload):
        raise BrokenPipeError("client disconnected")

    handler._send_json = disconnect_during_response
    try:
        with pytest.raises(FlashTeacherBridgeError) as transport_error:
            _post_json(
                bridge.url,
                bridge.token,
                "/score",
                _bridge_score_payload(0, [10, 11], [65, 66, 99]),
            )
        with pytest.raises(RetriableInfraError, match="after bounded retries"):
            _raise_verl_failure(1, bridge.teacher_failure)
    finally:
        bridge.close()

    assert transport_error.value.classification == "transient"
    assert bridge.teacher_failure == ("transient", "teacher unavailable")
    assert bridge.teacher_transient == 1
    assert bridge.teacher_error == 0


def test_client_only_score_loss_promotes_recovered_transient_once(monkeypatch, tmp_path):
    from flash.engine.worker.opd_train import _reconcile_score_delivery_failure
    from flash.engine.worker.perf import RetriableInfraError
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(TransientTeacher())
    transport_error, exit_code, fallback = _capture_client_only_score_delivery_loss(
        bridge,
        monkeypatch,
        tmp_path,
    )
    score_delivery_failure = _reconcile_score_delivery_failure(bridge, fallback)

    assert transport_error.delivery_unknown
    assert exit_code == 87
    assert fallback is not None
    assert fallback[0] == "transient"
    assert http.client.IncompleteRead.__name__ in fallback[1]
    assert score_delivery_failure is None
    assert bridge.teacher_failure == ("transient", "teacher unavailable")
    assert bridge.score_requests == 1
    assert bridge.episodes_seen == 1
    assert bridge.teacher_transient == 1
    assert bridge.teacher_ok == 0
    assert bridge.teacher_error == 0
    assert bridge.no_signal_resamples == 0
    assert bridge.no_signal_skipped_steps == 0
    with pytest.raises(RetriableInfraError, match="teacher unavailable"):
        _raise_verl_failure(
            1,
            bridge.teacher_failure,
            score_delivery_failure=score_delivery_failure,
        )


def test_client_only_successful_score_loss_is_direct_retriable_once(monkeypatch, tmp_path):
    from flash.engine.worker.opd_train import _reconcile_score_delivery_failure
    from flash.engine.worker.perf import RetriableInfraError

    bridge = _text_bridge(_BridgeTeacher())
    transport_error, exit_code, fallback = _capture_client_only_score_delivery_loss(
        bridge,
        monkeypatch,
        tmp_path,
    )
    score_delivery_failure = _reconcile_score_delivery_failure(bridge, fallback)

    assert transport_error.delivery_unknown
    assert exit_code == 87
    assert score_delivery_failure == fallback
    assert bridge.teacher_failure is None
    assert bridge.score_requests == 1
    assert bridge.episodes_seen == 1
    assert bridge.teacher_transient == 0
    assert bridge.teacher_ok == 1
    assert bridge.teacher_error == 0
    assert bridge.no_signal_resamples == 0
    assert bridge.no_signal_skipped_steps == 0
    with pytest.raises(RetriableInfraError, match="teacher score delivery failure"):
        _raise_verl_failure(
            1,
            bridge.teacher_failure,
            score_delivery_failure=score_delivery_failure,
        )


def test_client_only_score_loss_preserves_authoritative_permanent_failure(monkeypatch, tmp_path):
    from flash.engine.worker.opd_train import _reconcile_score_delivery_failure

    bridge = _text_bridge(_BridgeTeacher())
    bridge._record_teacher_failure("permanent", "bad credentials", terminal=True)
    transport_error, exit_code, fallback = _capture_client_only_score_delivery_loss(
        bridge,
        monkeypatch,
        tmp_path,
    )
    score_delivery_failure = _reconcile_score_delivery_failure(bridge, fallback)

    assert transport_error.delivery_unknown
    assert exit_code == 87
    assert fallback is not None
    assert score_delivery_failure is None
    assert bridge.teacher_failure == ("permanent", "bad credentials")
    assert bridge.score_requests == 1
    assert bridge.episodes_seen == 1
    assert bridge.teacher_transient == 0
    assert bridge.teacher_ok == 1
    assert bridge.teacher_error == 1
    assert bridge.no_signal_resamples == 0
    assert bridge.no_signal_skipped_steps == 0
    with pytest.raises(RuntimeError, match="permanent teacher failure: bad credentials"):
        _raise_verl_failure(
            1,
            bridge.teacher_failure,
            score_delivery_failure=score_delivery_failure,
        )


def test_malformed_score_success_is_transient_delivery_unknown(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import (
        _read_classified_failure_fallback,
        _reconcile_score_delivery_failure,
    )
    from flash.engine.worker.perf import RetriableInfraError

    failure_path = str(tmp_path / "score-delivery-failure")
    monkeypatch.setenv("FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH", failure_path)

    class ChildExit(RuntimeError):
        def __init__(self, code):
            super().__init__(code)
            self.code = code

    def child_exit(code):
        raise ChildExit(code)

    monkeypatch.setattr(plugin.os, "_exit", child_exit)
    bridge = _text_bridge(_BridgeTeacher())
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    handler._send_json = _send_malformed_json_response
    try:
        with pytest.raises(FlashTeacherBridgeError) as transport_error:
            _post_json(
                bridge.url,
                bridge.token,
                "/score",
                _bridge_score_payload(0, [10, 11], [65, 66, 99]),
            )
        with pytest.raises(ChildExit) as actor_exit:
            plugin._exit_for_score_failure(transport_error.value)
        fallback = _read_classified_failure_fallback(failure_path)
    finally:
        bridge.close()

    score_delivery_failure = _reconcile_score_delivery_failure(bridge, fallback)
    assert transport_error.value.classification == "permanent"
    assert transport_error.value.delivery_unknown
    assert actor_exit.value.code == 87
    assert fallback == (
        "transient",
        "flash OPD bridge returned malformed success JSON",
    )
    assert score_delivery_failure == fallback
    assert bridge.teacher_failure is None
    assert bridge.score_requests == 1
    assert bridge.teacher_ok == 1
    assert bridge.teacher_transient == 0
    assert bridge.teacher_error == 0
    with pytest.raises(RetriableInfraError, match="teacher score delivery failure"):
        _raise_verl_failure(
            1,
            bridge.teacher_failure,
            score_delivery_failure=score_delivery_failure,
        )


def test_explicit_score_rejection_keeps_classification_without_delivery_fallback(
    monkeypatch, tmp_path
):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.teacher import TeacherError

    class PermanentTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("bad credentials", permanent=True)

    failure_path = str(tmp_path / "score-delivery-failure")
    monkeypatch.setenv("FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH", failure_path)

    class ChildExit(RuntimeError):
        def __init__(self, code):
            super().__init__(code)
            self.code = code

    def child_exit(code):
        raise ChildExit(code)

    monkeypatch.setattr(plugin.os, "_exit", child_exit)
    bridge = _text_bridge(PermanentTeacher())
    bridge.start()
    try:
        with pytest.raises(FlashTeacherBridgeError) as rejection:
            _post_json(
                bridge.url,
                bridge.token,
                "/score",
                _bridge_score_payload(0, [10, 11], [65, 66, 99]),
            )
        with pytest.raises(ChildExit) as actor_exit:
            plugin._exit_for_score_failure(rejection.value)
    finally:
        bridge.close()

    assert rejection.value.classification == "permanent"
    assert not rejection.value.delivery_unknown
    assert actor_exit.value.code == 86
    assert _read_classified_failure_fallback(failure_path) is None
    assert not list(tmp_path.iterdir())
    assert bridge.teacher_failure == ("permanent", "bad credentials")
    assert bridge.score_requests == 1
    assert bridge.teacher_ok == 0
    assert bridge.teacher_transient == 0
    assert bridge.teacher_error == 1
    with pytest.raises(RuntimeError, match="permanent teacher failure: bad credentials"):
        _raise_verl_failure(1, bridge.teacher_failure)


@pytest.mark.parametrize(
    ("error_type", "error_args"),
    [
        (BrokenPipeError, ("client disconnected",)),
        (ConnectionResetError, ("connection reset",)),
        (TimeoutError, ("delivery timed out",)),
        (OSError, ("socket failed",)),
        (http.client.IncompleteRead, (b"partial", 1)),
    ],
)
def test_singleturn_success_delivery_failure_is_terminal_retriable(
    error_type,
    error_args,
):
    from flash.engine.worker.perf import RetriableInfraError

    bridge = _text_bridge(_BridgeTeacher())
    bridge.start()
    handler = bridge._server.RequestHandlerClass

    def fail_delivery(_handler, _status, _payload):
        raise error_type(*error_args)

    handler._send_json = fail_delivery
    try:
        with pytest.raises(FlashTeacherBridgeError):
            _post_json(
                bridge.url,
                bridge.token,
                "/score",
                _bridge_score_payload(0, [10, 11], [65, 66, 99]),
            )
        with pytest.raises(RetriableInfraError, match="after bounded retries"):
            _raise_verl_failure(1, bridge.teacher_failure)
    finally:
        bridge.close()

    assert bridge.teacher_failure is not None
    assert bridge.teacher_failure[0] == "transient"
    assert error_type.__name__ in bridge.teacher_failure[1]
    assert bridge.teacher_ok == 1
    assert bridge.teacher_transient == 0
    assert bridge.teacher_error == 0


@pytest.mark.parametrize(
    ("error_type", "error_args"),
    [
        (BrokenPipeError, ("client disconnected",)),
        (ConnectionResetError, ("connection reset",)),
        (TimeoutError, ("delivery timed out",)),
        (OSError, ("socket failed",)),
        (http.client.IncompleteRead, (b"partial", 1)),
    ],
)
def test_multiturn_success_delivery_failure_is_terminal_retriable(
    error_type,
    error_args,
):
    from flash.engine.worker.perf import RetriableInfraError

    bridge = _text_bridge(_BridgeTeacher())
    bridge.score_multiturn = lambda _session_id: {"ok": True}
    accounting_before = bridge.accounting_snapshot()
    bridge.start()
    handler = bridge._server.RequestHandlerClass

    def fail_delivery(_handler, _status, _payload):
        raise error_type(*error_args)

    handler._send_json = fail_delivery
    try:
        with pytest.raises(FlashTeacherBridgeError):
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

    assert bridge.teacher_failure is not None
    assert bridge.teacher_failure[0] == "transient"
    assert error_type.__name__ in bridge.teacher_failure[1]
    assert bridge.accounting_snapshot() == accounting_before


def test_success_delivery_failure_does_not_overwrite_permanent_teacher_failure():
    bridge = _text_bridge(_BridgeTeacher())
    bridge._record_teacher_failure("permanent", "bad credentials")
    bridge.start()
    handler = bridge._server.RequestHandlerClass

    def fail_delivery(_handler, _status, _payload):
        raise BrokenPipeError("client disconnected")

    handler._send_json = fail_delivery
    try:
        with pytest.raises(FlashTeacherBridgeError):
            _post_json(
                bridge.url,
                bridge.token,
                "/score",
                _bridge_score_payload(0, [10, 11], [65, 66, 99]),
            )
    finally:
        bridge.close()

    assert bridge.teacher_failure == ("permanent", "bad credentials")
    assert bridge.teacher_ok == 1
    assert bridge.teacher_transient == 0
    assert bridge.teacher_error == 1


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


def test_dropped_forced_groups_renormalize_over_surviving_tokens_only():
    """A dropped forced span must re-normalize the reverse-KL, not leave a shrunken sum.

    Ported from the TRL OPD suite, where the same invariant was asserted over
    ``_prepare_gkd_groups`` / ``_gkd_loss_from_logps``. In verl the two halves live in one place:
    ``group_ids`` of -1 exclude the forced positions, and the ``response_count / selected_count``
    rescale (opd_plugin.py:135) restores the denominator. The load-bearing claim is that a row
    whose forced tokens were dropped scores EXACTLY as a row that only ever held the survivors --
    without that rescale, verl's seq-mean-token-mean aggregation would divide the surviving signal
    by the full response length and silently shrink the loss in proportion to how much was forced.
    """
    torch = pytest.importorskip("torch")

    # tokens 0 and 3 fully-forced (group -1 = no signal); tokens 1,2 form one surviving group.
    student = torch.tensor([[-0.5, -1.0, -1.5, -2.0]], dtype=torch.float64, requires_grad=True)
    teacher_logsums = torch.tensor([[0.0, -2.0, -2.0, 0.0]], dtype=torch.float64)
    group_ids = torch.tensor([[-1, 1, 1, -1]])
    response_mask = torch.ones_like(student, dtype=torch.bool)

    masked = _aggregate_seq_mean_token_mean(
        _flash_groupwise_reverse_kl_values(
            student, teacher_logsums, group_ids, response_mask, 0.25
        ),
        response_mask,
    )

    # the same group, in a row that never carried the forced positions at all.
    survivors = torch.tensor([[-1.0, -1.5]], dtype=torch.float64, requires_grad=True)
    survivor_mask = torch.ones_like(survivors, dtype=torch.bool)
    survivor_loss = _aggregate_seq_mean_token_mean(
        _flash_groupwise_reverse_kl_values(
            survivors,
            torch.tensor([[-2.0, -2.0]], dtype=torch.float64),
            torch.tensor([[1, 1]]),
            survivor_mask,
            0.25,
        ),
        survivor_mask,
    )

    assert torch.allclose(masked, survivor_loss, atol=1e-12, rtol=1e-12)
    # and the dropped positions carry no gradient -- they are excluded, not merely down-weighted.
    gradient = torch.autograd.grad(masked, student)[0]
    assert torch.equal(gradient[0, [0, 3]], torch.zeros(2, dtype=torch.float64))


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


class _RecordingEnv:
    """Environment that records every turn it is shown, and rejects an empty action.

    the rejection is the point: a real environment that parses each action is entitled to raise on
    a truncated or empty one, and that is what turns a no-signal rollout into a paid run failure.
    """

    def __init__(self):
        self.recorded: list[str] = []

    def new_rollout_state(self, _example):
        return {"messages": [{"role": "user", "content": "q"}], "prompt": None}

    def record_model_turn(self, state, content):
        if not content.strip():
            raise ValueError("environment rejects an empty action")
        self.recorded.append(content)
        return state

    def env_reply(self, _messages, _state):
        return [{"role": "user", "content": "next"}]

    def rollout_done(self, _state, _turn_limit):
        return False


class _MultiTurnBridgeTokenizer(_BridgeTokenizer):
    """``_BridgeTokenizer`` plus the chat-template surface the env-glue tokenizer needs."""

    def apply_chat_template(self, messages, **_kwargs):
        return "".join(str(message.get("content", "")) for message in messages)

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) % 64 for char in text]}


def _multiturn_bridge(env, *, max_turns=4):
    return _TeacherAlignmentBridge(
        prompts=[
            _BridgePrompt(
                student_messages=[{"role": "user", "content": "q"}],
                teacher_messages=[{"role": "user", "content": "q"}],
                prompt_ids=(10, 11),
                image_descriptors=(),
                package_root=None,
                example=object(),
            )
        ],
        tokenizer=_MultiTurnBridgeTokenizer(),
        teacher=object(),
        thinking_prefill="",
        eos_token_ids=frozenset({99}),
        stop_sequences=(),
        mutation_callback=lambda: None,
        active_env=env,
        multi_turn=True,
        max_turns=max_turns,
    )


def test_an_unusable_opd_turn_is_never_shown_to_the_environment():
    """A truncated or skipped turn must not reach ``record_model_turn``.

    such a turn is already excluded from teacher scoring, so showing it to the environment buys no
    signal and can cost the whole paid run: an env that validates each action may raise on it. the
    grpo bridge returns before its own ``record_model_turn`` on exactly this predicate.
    """
    env = _RecordingEnv()
    bridge = _multiturn_bridge(env)
    bridge.start_multiturn(
        index=0, session_id="s1", prompt_ids=[10, 11], raw_prompt=[{"role": "user", "content": "q"}]
    )

    response = bridge.step_multiturn(
        {
            "session_id": "s1",
            "turn_ordinal": 0,
            "accepted_prefix": [10, 11],
            "raw_response_ids": [],
            "response_ids": [],
            "completion_text": "",
            "termination": "truncated",
            "stop_reason": "length",
            "truncated": True,
            "skip_reason": "truncated_rollout",
        }
    )

    assert env.recorded == [], "an unusable turn was pushed into environment state"
    assert response["terminal"] is True
    # the turn is still accounted for -- it is skipped, not erased
    assert bridge.truncated_rollouts == 1
    assert bridge.skip_counts == {"truncated_rollout": 1}


def test_a_usable_opd_turn_still_reaches_the_environment():
    """The guard must key on the unusable predicate, not disable multi-turn recording."""
    env = _RecordingEnv()
    bridge = _multiturn_bridge(env)
    bridge.start_multiturn(
        index=0, session_id="s1", prompt_ids=[10, 11], raw_prompt=[{"role": "user", "content": "q"}]
    )

    response = bridge.step_multiturn(
        {
            "session_id": "s1",
            "turn_ordinal": 0,
            "accepted_prefix": [10, 11],
            "raw_response_ids": [65],
            "response_ids": [65],
            "completion_text": "A",
            "termination": "stop",
            "stop_reason": "stop",
            "truncated": False,
            "skip_reason": "",
        }
    )

    assert env.recorded == ["A"]
    assert response["terminal"] is False


def test_multimodal_bridge_rejects_managed_teacher_scoring():
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

    with pytest.raises(ValueError, match="not supported by managed Parasail teachers"):
        bridge.score(0, 2, [10, 11, 65, 99], image_count=1)


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
    assert state["teacher_output_tokens"] == 6
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
    assert restored["teacher_output_tokens"] == 7
    assert restored["teacher_ok"] == 7
    assert restored["forced_tokens"] == 9
    assert restored["dropped_forced_groups"] == 4
    assert restored["aligned_sequences"] == 6
    assert restored["empty_alignments"] == 2
    assert restored["train_wall_seconds"] >= 12.5


def test_restore_verl_resume_returns_validated_accounting(monkeypatch, tmp_path):
    from flash.engine.worker import opd_train

    resume = tmp_path / "checkpoint-2"
    resume.mkdir()
    state = _resume_accounting()
    import json

    (resume / "opd_state.json").write_text(json.dumps(state))
    (resume / "payload.bin").write_bytes(b"checkpoint")
    monkeypatch.setattr(opd_train._w, "OPD_RESUME_REVISION", "revision")
    monkeypatch.setattr(opd_train._w, "SEED", 42)
    monkeypatch.setattr(
        opd_train._w,
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


def test_resume_leaves_missing_required_companion_for_checkpoint_watcher(monkeypatch, tmp_path):
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


def test_the_watcher_marks_every_step_processed_but_publishes_only_required_ones():
    # the two facts that make the final-publish guard below wrong. processed_steps.add(step) is
    # unconditional, while the deployable publish is gated on `step in self.required_steps`, so a
    # default run (save_at_steps empty -> required_steps empty) processes its last step without ever
    # publishing a deployable for it.
    source = inspect.getsource(_OpdVerlCheckpointWatcher)
    publish_gate = "if step in self.required_steps:"
    assert publish_gate in source
    assert "self.processed_steps.add(step)" in source
    # the unconditional mark must not sit inside the required-only publish branch.
    assert source.index(publish_gate) < source.index("self.processed_steps.add(step)")


def test_the_final_deployable_publish_is_not_suppressed_by_the_processed_marker():
    # regression: the verl OPD path gated its final publish on `final_step not in
    # watcher.processed_steps`. that guard reads as "skip if already published", but the two publish
    # paths are PROVABLY DISJOINT -- final_save_due() is true only when save_at_steps is EMPTY, and
    # the watcher publishes deployables only for steps that are IN save_at_steps. so the clause can
    # never prevent a real double-publish; it can only suppress a legitimate one.
    #
    # live-confirmed on runpod a100 run flash-1785569991-7ce8c3c1 (4/4 steps, default save_at_steps):
    # hf carries checkpoint/checkpoint-4/ (the watcher DID process step 4) but no
    # checkpoints/step-4/adapter/, so `flash runs checkpoint` listed nothing and step-4 deploy was
    # impossible. the pre-verl path published this unconditionally
    # (opd.py: publish_checkpoint=final_save_due(opt_steps, knobs.save_at_steps)), and grpo's
    # rl_train.py still does, so this was a verl-migration regression and a grpo/opd parity break.
    source = inspect.getsource(opd_train.run_opd_train)
    assert "final_save_due(final_step, knobs.save_at_steps)" in source
    assert "final_step not in watcher.processed_steps" not in source


def test_final_save_due_and_the_watcher_publish_set_never_overlap():
    # the invariant the fix above rests on, asserted rather than assumed: if these two could ever be
    # true for the same step, dropping the processed_steps clause would double-publish.
    from flash.engine.steps import final_save_due

    for save_at_steps in ((), (1,), (4,), (2, 4), (1, 2, 3, 4)):
        for step in (1, 2, 3, 4):
            watcher_publishes = step in save_at_steps
            assert not (final_save_due(step, save_at_steps) and watcher_publishes), (
                f"step={step} save_at_steps={save_at_steps} is published by BOTH paths"
            )


def test_both_ray_trainers_publish_their_final_step_deployable_the_same_way():
    # grpo and opd must agree here: a default run's last step is what `flash models deploy
    # <run>/step-N` targets, and one trainer silently not publishing it is exactly the shape of
    # parity break this migration keeps producing.
    for module in (rl_train, opd_train):
        source = inspect.getsource(module)
        assert "final_save_due(" in source, module.__name__
        assert "publish_deployable_checkpoint(" in source, module.__name__


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
    bridge = _text_bridge(TransientTeacher(), mutation_callback=lambda: mutations.append(True))
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
                lambda: _post_json(bridge.url, bridge.token, "/no-signal/resample", {}),
                lambda: _post_json(bridge.url, bridge.token, "/no-signal/abandoned", {}),
            )
        with pytest.raises(RetriableInfraError, match="after bounded retries"):
            _raise_verl_failure(1, bridge.teacher_failure)
    finally:
        bridge.close()

    assert bridge.score_requests == 6
    assert bridge.teacher_transient == 3
    assert bridge.teacher_ok == 0
    assert bridge.truncated_rollouts == 3


def test_abandonment_transport_fallback_promotes_pending_teacher_failure(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(TransientTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    failure_path = str(tmp_path / "abandonment-failure")

    def fail_before_parent(*_args, **_kwargs):
        raise FlashTeacherBridgeError(
            "flash OPD bridge transport failed: ConnectionRefusedError",
            classification="transient",
        )

    monkeypatch.setenv("FLASH_OPD_ABANDONMENT_FAILURE_PATH", failure_path)
    monkeypatch.setattr(plugin, "_post_json", fail_before_parent)

    with pytest.raises(FlashTeacherBridgeError):
        plugin._post_no_signal_abandoned("http://bridge", "token")

    assert _read_classified_failure_fallback(failure_path)
    bridge._promote_pending_teacher_failure()
    assert bridge.teacher_failure == ("transient", "teacher unavailable")
    assert bridge.teacher_transient == 1
    assert bridge.no_signal_skipped_steps == 0


def test_accepted_abandonment_lost_response_does_not_duplicate_accounting(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(TransientTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    failure_path = str(tmp_path / "abandonment-failure")
    monkeypatch.setenv("FLASH_OPD_ABANDONMENT_FAILURE_PATH", failure_path)
    bridge.start()
    handler = bridge._server.RequestHandlerClass

    def disconnect_during_response(_handler, _status, _payload):
        raise BrokenPipeError("client disconnected")

    handler._send_json = disconnect_during_response
    try:
        with pytest.raises(FlashTeacherBridgeError):
            plugin._post_no_signal_abandoned(bridge.url, bridge.token)
        assert _read_classified_failure_fallback(failure_path)
        bridge._promote_pending_teacher_failure()
    finally:
        bridge.close()

    assert bridge.teacher_failure == ("transient", "teacher unavailable")
    assert bridge.teacher_transient == 1
    assert bridge.no_signal_skipped_steps == 1


def test_resample_transport_failure_before_acceptance_promotes_without_replacement(
    monkeypatch, tmp_path
):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(TransientTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    failure_path = str(tmp_path / "resample-failure")
    replacements = []

    def fail_before_parent(*_args, **_kwargs):
        raise FlashTeacherBridgeError(
            "flash OPD bridge transport failed: ConnectionRefusedError",
            classification="transient",
            delivery_unknown=True,
        )

    def all_no_signal(_attempt):
        raise _AllNoSignalBatch("batch")

    monkeypatch.setenv("FLASH_OPD_RESAMPLE_FAILURE_PATH", failure_path)
    monkeypatch.setattr(plugin, "_post_json", fail_before_parent)

    with pytest.raises(FlashTeacherBridgeError):
        _run_with_no_signal_replacements(
            all_no_signal,
            lambda _batch: None,
            lambda: replacements.append(True),
            lambda: plugin._post_no_signal_resample("http://bridge", "token"),
            lambda: None,
        )

    assert _read_classified_failure_fallback(failure_path)
    bridge._promote_pending_teacher_failure()
    assert bridge.teacher_failure == ("transient", "teacher unavailable")
    assert bridge.no_signal_resamples == 0
    assert bridge.no_signal_skipped_steps == 0
    assert replacements == []


def test_accepted_resample_lost_response_counts_once_and_promotes(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(TransientTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    failure_path = str(tmp_path / "resample-failure")
    monkeypatch.setenv("FLASH_OPD_RESAMPLE_FAILURE_PATH", failure_path)
    bridge.start()
    handler = bridge._server.RequestHandlerClass

    def disconnect_during_response(_handler, _status, _payload):
        raise BrokenPipeError("client disconnected")

    handler._send_json = disconnect_during_response
    try:
        with pytest.raises(FlashTeacherBridgeError):
            plugin._post_no_signal_resample(bridge.url, bridge.token)
        assert _read_classified_failure_fallback(failure_path)
        bridge._promote_pending_teacher_failure()
    finally:
        bridge.close()

    assert bridge.teacher_failure == ("transient", "teacher unavailable")
    assert bridge.no_signal_resamples == 1
    assert bridge.no_signal_skipped_steps == 0


def test_resample_fallback_without_pending_transient_does_not_promote(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    bridge = _text_bridge(_BridgeTeacher())
    bridge.score(0, 2, [10, 11, 65, 66])
    failure_path = str(tmp_path / "resample-failure")

    def fail_before_parent(*_args, **_kwargs):
        raise FlashTeacherBridgeError(
            "flash OPD bridge transport failed: ConnectionRefusedError",
            classification="transient",
            delivery_unknown=True,
        )

    monkeypatch.setenv("FLASH_OPD_RESAMPLE_FAILURE_PATH", failure_path)
    monkeypatch.setattr(plugin, "_post_json", fail_before_parent)

    with pytest.raises(FlashTeacherBridgeError):
        plugin._post_no_signal_resample("http://bridge", "token")

    assert _read_classified_failure_fallback(failure_path)
    bridge._promote_pending_teacher_failure()
    assert bridge.teacher_failure is None
    assert bridge.no_signal_resamples == 0
    assert bridge.no_signal_skipped_steps == 0


def test_successful_teacher_signal_suppresses_resample_fallback_promotion(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.teacher import TeacherError

    class TransientThenSuccessTeacher:
        def __init__(self):
            self.calls = 0

        def score(self, prompt_text, completion_text):
            self.calls += 1
            if self.calls == 1:
                raise TeacherError("teacher unavailable", permanent=False)
            return _BridgeTeacher().score(prompt_text, completion_text)

    bridge = _text_bridge(TransientThenSuccessTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    failure_path = str(tmp_path / "resample-failure")

    def fail_before_parent(*_args, **_kwargs):
        raise FlashTeacherBridgeError(
            "flash OPD bridge transport failed: ConnectionRefusedError",
            classification="transient",
            delivery_unknown=True,
        )

    monkeypatch.setenv("FLASH_OPD_RESAMPLE_FAILURE_PATH", failure_path)
    monkeypatch.setattr(plugin, "_post_json", fail_before_parent)

    with pytest.raises(FlashTeacherBridgeError):
        plugin._post_no_signal_resample("http://bridge", "token")

    assert _read_classified_failure_fallback(failure_path)
    bridge._promote_pending_teacher_failure()
    assert bridge.teacher_failure is None
    assert bridge.teacher_transient == 1
    assert bridge.teacher_ok == 1


def test_successful_resample_creates_no_marker_and_prepares_replacement(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    bridge = _text_bridge(_BridgeTeacher())
    failure_path = str(tmp_path / "resample-failure")
    monkeypatch.setenv("FLASH_OPD_RESAMPLE_FAILURE_PATH", failure_path)
    attempts = []
    replacements = []

    def run_attempt(attempt):
        attempts.append(attempt)
        if attempt == 0:
            raise _AllNoSignalBatch("batch")
        return "trained"

    bridge.start()
    try:
        result = _run_with_no_signal_replacements(
            run_attempt,
            lambda _batch: None,
            lambda: replacements.append(True),
            lambda: plugin._post_no_signal_resample(bridge.url, bridge.token),
            lambda: None,
        )
    finally:
        bridge.close()

    assert result == "trained"
    assert attempts == [0, 1]
    assert replacements == [True]
    assert bridge.no_signal_resamples == 1
    assert not _read_classified_failure_fallback(failure_path)


@pytest.mark.parametrize(
    "failures",
    [
        (("permanent", "resample rejected"), ("transient", "abandon timeout")),
        (("transient", "resample timeout"), ("permanent", "abandon rejected")),
    ],
)
def test_permanent_no_signal_notification_precedes_pending_transient(failures):
    from flash.engine.worker.opd_train import _reconcile_no_signal_notification_failure
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(TransientTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])

    failure = _reconcile_no_signal_notification_failure(bridge, failures)

    assert failure is not None
    assert failure[0] == "permanent"
    assert bridge.teacher_failure is None
    with pytest.raises(RuntimeError, match="permanent no-signal notification"):
        _raise_verl_failure(1, bridge.teacher_failure, no_signal_failure=failure)


@pytest.mark.parametrize(
    ("classification", "message"),
    [
        ("permanent", "bad credentials"),
        ("transient", "teacher unavailable"),
    ],
)
def test_authoritative_teacher_failure_precedes_transient_no_signal_evidence(
    classification,
    message,
):
    from flash.engine.worker.opd_train import _reconcile_no_signal_notification_failure
    from flash.engine.worker.perf import RetriableInfraError

    bridge = _text_bridge(_BridgeTeacher())
    bridge._record_teacher_failure(classification, message, terminal=True)

    failure = _reconcile_no_signal_notification_failure(
        bridge,
        (
            ("transient", "resample transport failed"),
            ("transient", "abandon transport failed"),
        ),
    )

    assert failure is None
    assert bridge.teacher_failure == (classification, message)
    if classification == "permanent":
        with pytest.raises(RuntimeError, match="permanent teacher failure"):
            _raise_verl_failure(1, bridge.teacher_failure, no_signal_failure=failure)
    else:
        with pytest.raises(RetriableInfraError, match="after bounded retries"):
            _raise_verl_failure(1, bridge.teacher_failure, no_signal_failure=failure)


@pytest.mark.parametrize(
    ("channel", "environment_key", "reader_name", "counter_name"),
    [
        (
            "resample",
            "FLASH_OPD_RESAMPLE_FAILURE_PATH",
            "_read_classified_failure_fallback",
            "no_signal_resamples",
        ),
        (
            "abandoned",
            "FLASH_OPD_ABANDONMENT_FAILURE_PATH",
            "_read_classified_failure_fallback",
            "no_signal_skipped_steps",
        ),
    ],
)
def test_malformed_accepted_no_signal_notification_is_transient_once(
    monkeypatch,
    tmp_path,
    channel,
    environment_key,
    reader_name,
    counter_name,
):
    import flash.engine.worker.opd_plugin as plugin
    import flash.engine.worker.opd_train as opd_train

    failure_path = str(tmp_path / f"{channel}-failure")
    monkeypatch.setenv(environment_key, failure_path)
    bridge = _text_bridge(_BridgeTeacher())
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    sends = []

    def malformed_response(request_handler, status, payload):
        sends.append(status)
        return _send_malformed_json_response(request_handler, status, payload)

    handler._send_json = malformed_response
    try:
        with pytest.raises(FlashTeacherBridgeError) as error:
            getattr(plugin, f"_post_no_signal_{channel}")(bridge.url, bridge.token)
        fallback = getattr(opd_train, reader_name)(failure_path)
    finally:
        bridge.close()

    assert error.value.classification == "permanent"
    assert error.value.delivery_unknown
    assert sends == [200]
    assert getattr(bridge, counter_name) == 1
    assert fallback == (
        "transient",
        "flash OPD bridge returned malformed success JSON",
    )


def test_malformed_accepted_cycle_commit_exhaustion_is_transient(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.perf import RetriableInfraError

    failure_path = str(tmp_path / "cycle-commit-failure")
    monkeypatch.setenv("FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH", failure_path)
    bridge = _text_bridge(_BridgeTeacher())
    accounting_before = bridge.accounting_snapshot()
    commit_calls = []
    original_commit = bridge.commit_teacher_cycle

    def record_commit():
        commit_calls.append(True)
        return original_commit()

    bridge.commit_teacher_cycle = record_commit
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    sends = []

    def malformed_response(request_handler, status, payload):
        sends.append(status)
        return _send_malformed_json_response(request_handler, status, payload)

    handler._send_json = malformed_response
    try:
        with pytest.raises(FlashTeacherBridgeError) as error:
            plugin._post_teacher_cycle_committed(bridge.url, bridge.token)
        fallback = _read_classified_failure_fallback(failure_path)
    finally:
        bridge.close()

    assert error.value.classification == "permanent"
    assert error.value.delivery_unknown
    assert sends == [200, 200]
    assert commit_calls == [True, True]
    assert bridge.accounting_snapshot() == accounting_before
    assert fallback == (
        "transient",
        "flash OPD bridge returned malformed success JSON",
    )
    with pytest.raises(RetriableInfraError, match="pre-update cycle commitment"):
        _raise_verl_failure(1, None, cycle_commit_failure=fallback)


@pytest.mark.parametrize(
    ("poster_name", "environment_key", "reader_name"),
    [
        (
            "_post_no_signal_resample",
            "FLASH_OPD_RESAMPLE_FAILURE_PATH",
            "_read_classified_failure_fallback",
        ),
        (
            "_post_no_signal_abandoned",
            "FLASH_OPD_ABANDONMENT_FAILURE_PATH",
            "_read_classified_failure_fallback",
        ),
        (
            "_post_teacher_cycle_committed",
            "FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH",
            "_read_classified_failure_fallback",
        ),
    ],
)
def test_explicit_notification_rejection_remains_permanent(
    monkeypatch,
    tmp_path,
    poster_name,
    environment_key,
    reader_name,
):
    import flash.engine.worker.opd_plugin as plugin
    import flash.engine.worker.opd_train as opd_train

    attempts = []
    failure_path = str(tmp_path / "notification-failure")

    def reject_notification(*_args, **_kwargs):
        attempts.append(True)
        raise FlashTeacherBridgeError(
            "notification rejected",
            classification="permanent",
        )

    monkeypatch.setenv(environment_key, failure_path)
    monkeypatch.setattr(plugin, "_post_json", reject_notification)

    with pytest.raises(FlashTeacherBridgeError, match="notification rejected"):
        getattr(plugin, poster_name)("http://bridge", "token")

    assert attempts == [True]
    assert getattr(opd_train, reader_name)(failure_path) == (
        "permanent",
        "notification rejected",
    )


@pytest.mark.parametrize(
    ("poster_name", "environment_key", "reader_name", "message_prefix"),
    [
        (
            "_post_no_signal_resample",
            "FLASH_OPD_RESAMPLE_FAILURE_PATH",
            "_read_classified_failure_fallback",
            "unexpected resample bridge failure",
        ),
        (
            "_post_no_signal_abandoned",
            "FLASH_OPD_ABANDONMENT_FAILURE_PATH",
            "_read_classified_failure_fallback",
            "unexpected abandonment bridge failure",
        ),
        (
            "_post_teacher_cycle_committed",
            "FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH",
            "_read_classified_failure_fallback",
            "unexpected cycle commitment bridge failure",
        ),
    ],
)
def test_unexpected_local_notification_error_remains_permanent(
    monkeypatch,
    tmp_path,
    poster_name,
    environment_key,
    reader_name,
    message_prefix,
):
    import flash.engine.worker.opd_plugin as plugin
    import flash.engine.worker.opd_train as opd_train

    failure_path = str(tmp_path / "notification-failure")

    def fail_locally(*_args, **_kwargs):
        raise ValueError("local failure")

    monkeypatch.setenv(environment_key, failure_path)
    monkeypatch.setattr(plugin, "_post_json", fail_locally)

    with pytest.raises(ValueError, match="local failure"):
        getattr(plugin, poster_name)("http://bridge", "token")

    fallback = getattr(opd_train, reader_name)(failure_path)
    assert fallback is not None
    assert fallback[0] == "permanent"
    assert fallback[1] == f"{message_prefix}: ValueError"


def test_transient_no_signal_notification_without_pending_is_retriable():
    from flash.engine.worker.opd_train import _reconcile_no_signal_notification_failure
    from flash.engine.worker.perf import RetriableInfraError

    bridge = _text_bridge(_BridgeTeacher())
    failure = _reconcile_no_signal_notification_failure(
        bridge,
        (("transient", "resample transport failed"),),
    )

    assert failure == ("transient", "resample transport failed")
    assert bridge.teacher_failure is None
    with pytest.raises(RetriableInfraError, match="no-signal notification"):
        _raise_verl_failure(1, None, no_signal_failure=failure)


def test_transient_no_signal_notification_promotes_causal_teacher_failure():
    from flash.engine.worker.opd_train import _reconcile_no_signal_notification_failure
    from flash.engine.worker.teacher import TeacherError

    class TransientTeacher:
        def score(self, _prompt_text, _completion_text):
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(TransientTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])

    failure = _reconcile_no_signal_notification_failure(
        bridge,
        (("transient", "abandon transport failed"),),
    )

    assert failure is None
    assert bridge.teacher_failure == ("transient", "teacher unavailable")


def test_successful_cycle_commit_allows_next_transient_cycle_to_promote():
    from flash.engine.worker.teacher import TeacherError

    class SuccessThenTransientTeacher:
        def __init__(self):
            self.calls = 0

        def score(self, prompt_text, completion_text):
            self.calls += 1
            if self.calls == 1:
                return _BridgeTeacher().score(prompt_text, completion_text)
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(SuccessThenTransientTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    bridge.commit_teacher_cycle()
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    bridge.record_no_signal_abandoned()

    assert bridge.teacher_ok == 1
    assert bridge.teacher_transient == 1
    assert bridge.teacher_failure == ("transient", "teacher unavailable")


def test_lost_cycle_commit_response_retries_without_mutation_failure(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.teacher import TeacherError

    class SuccessThenTransientTeacher:
        def __init__(self):
            self.calls = 0

        def score(self, prompt_text, completion_text):
            self.calls += 1
            if self.calls == 1:
                return _BridgeTeacher().score(prompt_text, completion_text)
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(SuccessThenTransientTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    mutation_failure_path = str(tmp_path / "mutation-failure")
    cycle_failure_path = str(tmp_path / "cycle-commit-failure")
    monkeypatch.setenv("FLASH_OPD_MUTATION_FAILURE_PATH", mutation_failure_path)
    monkeypatch.setenv("FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH", cycle_failure_path)

    def unexpected_exit(code):
        raise AssertionError(f"cycle commitment exited rank with status {code}")

    monkeypatch.setattr(plugin.os, "_exit", unexpected_exit)
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    original_send = handler._send_json
    sends = []

    def lose_first_response(request_handler, status, payload):
        sends.append(status)
        if len(sends) <= 2:
            raise BrokenPipeError("client disconnected")
        return original_send(request_handler, status, payload)

    handler._send_json = lose_first_response
    try:
        plugin._post_teacher_cycle_committed(bridge.url, bridge.token)
        bridge.score(0, 2, [10, 11, 65, 66, 99])
        bridge.record_no_signal_abandoned()
    finally:
        bridge.close()

    assert sends == [200, 422, 200]
    assert bridge.mutation_failure is None
    assert not list(tmp_path.glob("mutation-failure.*.json"))
    assert bridge.teacher_failure == ("transient", "teacher unavailable")


def test_incomplete_cycle_commit_response_retries_once_without_mutation_fallback(
    monkeypatch, tmp_path
):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.teacher import TeacherError

    class SuccessThenTransientTeacher:
        def __init__(self):
            self.calls = 0

        def score(self, prompt_text, completion_text):
            self.calls += 1
            if self.calls == 1:
                return _BridgeTeacher().score(prompt_text, completion_text)
            raise TeacherError("teacher unavailable", permanent=False)

    bridge = _text_bridge(SuccessThenTransientTeacher())
    bridge.score(0, 2, [10, 11, 65, 66, 99])
    mutation_failure_path = str(tmp_path / "mutation-failure")
    cycle_failure_path = str(tmp_path / "cycle-commit-failure")
    monkeypatch.setenv("FLASH_OPD_MUTATION_FAILURE_PATH", mutation_failure_path)
    monkeypatch.setenv("FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH", cycle_failure_path)

    def unexpected_exit(code):
        raise AssertionError(f"cycle commitment exited rank with status {code}")

    monkeypatch.setattr(plugin.os, "_exit", unexpected_exit)
    commit_calls = []
    original_commit = bridge.commit_teacher_cycle

    def record_commit():
        commit_calls.append(True)
        return original_commit()

    bridge.commit_teacher_cycle = record_commit
    actor_updates = []

    class ActorGroup:
        def update_actor(self, batch):
            actor_updates.append(batch)
            return {"updated": True}

    batch = object()
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    original_send = handler._send_json
    sends = []

    def truncate_first_response(request_handler, status, payload):
        sends.append(status)
        if len(sends) == 1:
            return _send_truncated_json_response(request_handler, status, payload)
        return original_send(request_handler, status, payload)

    handler._send_json = truncate_first_response
    try:
        result = plugin._update_actor_after_teacher_cycle_commit(
            ActorGroup(), batch, bridge.url, bridge.token
        )
        bridge.score(0, 2, [10, 11, 65, 66, 99])
        bridge.record_no_signal_abandoned()
    finally:
        bridge.close()

    assert sends == [200, 200]
    assert commit_calls == [True, True]
    assert result == {"updated": True}
    assert actor_updates == [batch]
    assert bridge.mutation_failure is None
    assert not list(tmp_path.glob("mutation-failure.*.json"))
    assert _read_classified_failure_fallback(cycle_failure_path) is None
    assert bridge.teacher_failure == ("transient", "teacher unavailable")


def test_transient_cycle_commit_failure_records_retriable_preupdate_cause(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.perf import RetriableInfraError

    attempts = []
    actor_updates = []
    failure_path = str(tmp_path / "cycle-commit-failure")

    class ActorGroup:
        def update_actor(self, batch):
            actor_updates.append(batch)

    def fail_before_parent(*_args, **_kwargs):
        attempts.append(True)
        raise FlashTeacherBridgeError(
            "cycle bridge unavailable",
            classification="transient",
            delivery_unknown=True,
        )

    monkeypatch.setenv("FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH", failure_path)
    monkeypatch.setattr(plugin, "_post_json", fail_before_parent)

    with pytest.raises(FlashTeacherBridgeError):
        plugin._update_actor_after_teacher_cycle_commit(
            ActorGroup(), object(), "http://bridge", "token"
        )

    fallback = _read_classified_failure_fallback(failure_path)
    assert attempts == [True, True]
    assert actor_updates == []
    assert fallback == ("transient", "cycle bridge unavailable")
    with pytest.raises(RetriableInfraError, match="pre-update cycle commitment"):
        _raise_verl_failure(1, None, cycle_commit_failure=fallback)


def test_explicit_cycle_commit_rejection_aborts_before_actor_update(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    attempts = []
    actor_updates = []
    failure_path = str(tmp_path / "cycle-commit-failure")

    class ActorGroup:
        def update_actor(self, batch):
            actor_updates.append(batch)

    def reject_commit(*_args, **_kwargs):
        attempts.append(True)
        raise FlashTeacherBridgeError(
            "cycle commit rejected",
            classification="permanent",
        )

    monkeypatch.setenv("FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH", failure_path)
    monkeypatch.setattr(plugin, "_post_json", reject_commit)

    with pytest.raises(FlashTeacherBridgeError, match="cycle commit rejected"):
        plugin._update_actor_after_teacher_cycle_commit(
            ActorGroup(), object(), "http://bridge", "token"
        )

    fallback = _read_classified_failure_fallback(failure_path)
    assert attempts == [True]
    assert actor_updates == []
    assert fallback == ("permanent", "cycle commit rejected")
    with pytest.raises(RuntimeError, match="permanent pre-update cycle commitment"):
        _raise_verl_failure(1, None, cycle_commit_failure=fallback)


def test_persistent_cycle_commit_failure_aborts_before_actor_update(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.perf import RetriableInfraError

    bridge = _text_bridge(_BridgeTeacher())
    commit_calls = []
    original_commit = bridge.commit_teacher_cycle

    def record_commit():
        commit_calls.append(True)
        return original_commit()

    bridge.commit_teacher_cycle = record_commit
    actor_updates = []
    accounting_before = bridge.accounting_snapshot()
    failure_path = str(tmp_path / "cycle-commit-failure")
    monkeypatch.setenv("FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH", failure_path)

    class ActorGroup:
        def update_actor(self, batch):
            actor_updates.append(batch)

    bridge.start()
    handler = bridge._server.RequestHandlerClass
    sends = []

    def truncate_response(request_handler, status, payload):
        sends.append(status)
        return _send_truncated_json_response(request_handler, status, payload)

    handler._send_json = truncate_response
    try:
        with pytest.raises(FlashTeacherBridgeError) as error:
            plugin._update_actor_after_teacher_cycle_commit(
                ActorGroup(), object(), bridge.url, bridge.token
            )
    finally:
        bridge.close()

    fallback = _read_classified_failure_fallback(failure_path)
    assert error.value.classification == "transient"
    assert error.value.delivery_unknown
    assert sends == [200, 200]
    assert commit_calls == [True, True]
    assert actor_updates == []
    assert bridge.mutation_failure is None
    assert bridge.accounting_snapshot() == accounting_before
    assert fallback is not None
    assert fallback[0] == "transient"
    with pytest.raises(RetriableInfraError, match="pre-update cycle commitment"):
        _raise_verl_failure(1, None, cycle_commit_failure=fallback)


@pytest.mark.parametrize(
    "message",
    [
        "é" * 9000,
        '"' * 9000,
        "\\" * 9000,
        ("\x00\n\r\t") * 3000,
        chr(0xD800) * 9000,
        chr(0xDC00) * 9000,
        bytes([0x80, 0xFF]).decode("utf-8", errors="surrogateescape") * 4500,
    ],
    ids=[
        "non-ascii",
        "quotes",
        "backslashes",
        "controls",
        "high-surrogate",
        "low-surrogate",
        "surrogateescape",
    ],
)
def test_failure_fallback_serialization_is_valid_and_within_reader_limit(tmp_path, message):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    failure_path = str(tmp_path / "cycle-commit-failure")
    plugin._write_failure_fallback(failure_path, "transient", message)

    records = list(tmp_path.glob("cycle-commit-failure.*.transient.json"))
    assert len(records) == 1
    encoded = records[0].read_bytes()
    serialized = encoded.decode("utf-8")
    record = json.loads(serialized)
    normalized_message = message.encode("utf-8", errors="replace").decode("utf-8")
    assert len(encoded) <= 8192
    assert len(serialized) <= 8192
    assert record["classification"] == "transient"
    assert record["message"]
    assert normalized_message.startswith(record["message"])
    assert _read_classified_failure_fallback(failure_path) == (
        "transient",
        record["message"].strip(),
    )


def test_cycle_commit_fallback_reader_ignores_incomplete_records(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    failure_path = str(tmp_path / "cycle-commit-failure")
    monkeypatch.setenv("FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH", failure_path)

    plugin._write_cycle_commit_failure_fallback("transient", "cycle timeout")
    Path(f"{failure_path}.malformed.permanent.json").write_text("{")
    Path(f"{failure_path}.incomplete.permanent.json.tmp").write_text(
        json.dumps({"classification": "permanent", "message": "incomplete"})
    )

    completed = list(tmp_path.glob("cycle-commit-failure.*.transient.json"))
    assert len(completed) == 1
    assert _read_classified_failure_fallback(failure_path) == (
        "transient",
        "cycle timeout",
    )


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


def test_multiturn_teacher_scores_are_chunked_and_ordered(monkeypatch):
    class OrderedTeacher:
        def __init__(self):
            self.batch_sizes = []
            self.next_index = 0

        def score_many(self, items):
            self.batch_sizes.append(len(items))
            results = []
            for _item in items:
                self.next_index += 1
                results.append(
                    _teacher_score(
                        [
                            TeacherToken(
                                text="AB",
                                logprob=-float(self.next_index),
                                start=0,
                                end=2,
                            )
                        ]
                    )
                )
            return results

    # turn count is derived from the measured concurrency ceiling for the same reason as the batcher
    # test above: this pins CHUNKING AND ORDER (one full chunk, then the remainder, with scores
    # still in turn order), and a hardcoded 15 turns would stop chunking at all once the ceiling
    # rose past it -- leaving an assertion that can no longer observe the behaviour it names.
    from flash.opd_limits import OPD_TEACHER_SCORING_CONCURRENCY as bound

    turns = bound + bound // 2
    teacher = OrderedTeacher()
    bridge = _text_bridge(teacher)
    monkeypatch.setattr(bridge, "_require_multiturn", lambda: None)
    bridge._sessions["session-1"] = {
        "turns": [
            {
                "prompt_ids": [10, 11],
                "response_ids": [65, 66],
                "completion_text": "AB",
                "context_messages": [{"role": "user", "content": "question"}],
                "truncated": False,
                "skip_reason": "",
            }
            for _index in range(turns)
        ],
        "score_cache": None,
        "score_lock": threading.Lock(),
        "lease_deadline": time.monotonic() + 60,
    }

    result = bridge.score_multiturn("session-1")

    assert teacher.batch_sizes == [bound, turns - bound]
    assert [_teacher_logsum(turn) for turn in result["turns"]] == [
        -float(index) for index in range(1, turns + 1)
    ]


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


def test_client_only_multiturn_score_loss_publishes_retriable_fallback_once(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_multiturn import _post_multiturn_score
    from flash.engine.worker.opd_train import (
        _read_classified_failure_fallback,
        _reconcile_score_delivery_failure,
    )
    from flash.engine.worker.perf import RetriableInfraError

    failure_path = str(tmp_path / "score-delivery-failure")
    monkeypatch.setenv("FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH", failure_path)

    class ChildExit(RuntimeError):
        def __init__(self, code):
            super().__init__(code)
            self.code = code

    def child_exit(code):
        raise ChildExit(code)

    score_calls = []
    bridge = _text_bridge(_BridgeTeacher())

    def score_multiturn(session_id):
        score_calls.append(session_id)
        return {"turns": []}

    bridge.score_multiturn = score_multiturn
    accounting_before = bridge.accounting_snapshot()
    monkeypatch.setattr(plugin.os, "_exit", child_exit)
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    handler._send_json = _send_truncated_json_response
    try:
        with pytest.raises(ChildExit) as actor_exit:
            _post_multiturn_score(
                _post_json,
                plugin._exit_for_score_failure,
                bridge.url,
                bridge.token,
                "session-1",
            )
        fallback = _read_classified_failure_fallback(failure_path)
    finally:
        bridge.close()

    score_delivery_failure = _reconcile_score_delivery_failure(bridge, fallback)
    assert actor_exit.value.code == 87
    assert score_calls == ["session-1"]
    assert bridge.accounting_snapshot() == accounting_before
    assert fallback is not None
    assert fallback[0] == "transient"
    assert http.client.IncompleteRead.__name__ in fallback[1]
    assert score_delivery_failure == fallback
    with pytest.raises(RetriableInfraError, match="teacher score delivery failure"):
        _raise_verl_failure(
            1,
            bridge.teacher_failure,
            score_delivery_failure=score_delivery_failure,
        )


def test_explicit_multiturn_score_rejection_bypasses_delivery_handler():
    from flash.engine.worker.opd_multiturn import _post_multiturn_score

    rejection = FlashTeacherBridgeError(
        "multi-turn score rejected",
        classification="permanent",
    )
    handled = []

    def reject_score(*_args, **_kwargs):
        raise rejection

    with pytest.raises(FlashTeacherBridgeError) as error:
        _post_multiturn_score(
            reject_score,
            handled.append,
            "http://bridge",
            "token",
            "session-1",
        )

    assert error.value is rejection
    assert handled == []


def test_bridge_transport_failure_is_typed_retriable(monkeypatch):
    import urllib.error
    import urllib.request

    def offline(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    with pytest.raises(FlashTeacherBridgeError) as error:
        _post_json("http://127.0.0.1:1", "token", "/score", {})
    assert error.value.classification == "transient"


def test_mutation_transport_failure_survives_actor_exit_and_generic_driver_status(
    monkeypatch, tmp_path
):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback
    from flash.engine.worker.perf import RetriableInfraError

    failure_path = str(tmp_path / "mutation-failure")

    def fail_post(*_args, **_kwargs):
        raise FlashTeacherBridgeError(
            "flash OPD bridge transport failed: ConnectionRefusedError",
            classification="transient",
        )

    class ChildExit(RuntimeError):
        def __init__(self, code):
            super().__init__(code)
            self.code = code

    def child_exit(code):
        raise ChildExit(code)

    monkeypatch.setenv("FLASH_OPD_MUTATION_FAILURE_PATH", failure_path)
    monkeypatch.setattr(plugin, "_post_json", fail_post)
    monkeypatch.setattr(plugin.os, "_exit", child_exit)

    with pytest.raises(ChildExit) as actor_exit:
        plugin._post_mutation_notice("http://bridge", "token")

    mutation_failure = _read_classified_failure_fallback(failure_path)
    assert actor_exit.value.code == 87
    assert mutation_failure == (
        "transient",
        "flash OPD bridge transport failed: ConnectionRefusedError",
    )
    with pytest.raises(RetriableInfraError, match="optimizer marker failure"):
        _raise_verl_failure(1, None, mutation_failure)


def test_mutation_failure_fallback_publishes_one_atomic_record_per_process(monkeypatch, tmp_path):
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    failure_path = str(tmp_path / "mutation-failure")
    messages = [f"bridge timeout {index}" for index in range(4)]
    context = multiprocessing.get_context("spawn")
    start = context.Event()

    monkeypatch.setenv("FLASH_OPD_MUTATION_FAILURE_PATH", failure_path)
    processes = [
        context.Process(
            target=_write_mutation_failure_after_start,
            args=(start, "transient", message),
        )
        for message in messages
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    records = sorted(tmp_path.glob("mutation-failure.*.transient.json"))
    assert len(records) == len(messages)
    assert {json.loads(record.read_text())["message"] for record in records} == set(messages)
    selected = _read_classified_failure_fallback(failure_path)
    expected_message = json.loads(records[0].read_text())["message"]
    assert selected == ("transient", expected_message)
    assert not [path for path in tmp_path.iterdir() if path.name.endswith(".tmp")]


def test_mutation_failure_fallback_removes_temp_when_publication_fails(monkeypatch, tmp_path):
    import flash.engine.worker.opd_plugin as plugin

    failure_path = str(tmp_path / "mutation-failure")

    def fail_publication(_source, _destination):
        raise OSError("publication failed")

    monkeypatch.setenv("FLASH_OPD_MUTATION_FAILURE_PATH", failure_path)
    monkeypatch.setattr(plugin.os, "replace", fail_publication)

    plugin._write_mutation_failure_fallback("transient", "bridge timeout")

    assert not list(tmp_path.iterdir())


def test_mutation_failure_fallback_selects_permanent_and_ignores_incomplete_records(
    tmp_path,
):
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    failure_path = str(tmp_path / "mutation-failure")
    Path(f"{failure_path}.100.transient.json").write_text(
        json.dumps({"classification": "transient", "message": "bridge timeout"})
    )
    Path(f"{failure_path}.200.permanent.json").write_text(
        json.dumps(
            {
                "classification": "permanent",
                "message": "invalid marker configuration",
            }
        )
    )
    Path(f"{failure_path}.300.permanent.json").write_text("{")
    Path(f"{failure_path}.400.transient.json.tmp").write_text(
        json.dumps({"classification": "transient", "message": "incomplete"})
    )

    assert _read_classified_failure_fallback(failure_path) == (
        "permanent",
        "invalid marker configuration",
    )


@pytest.mark.parametrize(
    ("classification", "expected_exit"),
    [("transient", 87), ("permanent", 86)],
)
def test_repeated_mutation_notice_maps_later_bridge_failure_to_child_exit(
    monkeypatch, classification, expected_exit
):
    import flash.engine.worker.opd_plugin as plugin
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

    expected_payload = {"process_id": os.getpid()}
    assert posts == [
        ("http://bridge", "token", "/mutation", expected_payload),
        ("http://bridge", "token", "/mutation", expected_payload),
    ]
    assert exit_error.value.code == expected_exit
    if classification == "transient":
        with pytest.raises(RetriableInfraError, match="transient teacher bridge failure"):
            _raise_verl_failure(exit_error.value.code, None)
    else:
        with pytest.raises(RuntimeError, match="permanent teacher bridge failure"):
            _raise_verl_failure(exit_error.value.code, None)


def test_optimizer_rank_posts_marker_once_across_multiple_steps(monkeypatch):
    import flash.engine.worker.opd_plugin as plugin

    coordination = []
    updates = []

    class Optimizer:
        def step(self):
            updates.append(True)
            return len(updates)

    def coordinate(url, token):
        coordination.append((url, token))

    monkeypatch.setattr(
        plugin,
        "_coordinate_first_mutation_notice",
        coordinate,
        raising=False,
    )
    optimizer = plugin._wrap_optimizer_with_mutation_notice(
        Optimizer(),
        "http://bridge",
        "token",
    )

    assert optimizer.step() == 1
    assert optimizer.step() == 2
    assert coordination == [("http://bridge", "token")]
    assert updates == [True, True]


def test_wrapped_optimizer_step_survives_the_lr_scheduler_rebind(monkeypatch):
    # verl builds a warmup lr scheduler immediately after the optimizer, and torch's
    # LRScheduler.__init__ patches optimizer.step through step_fn.__func__ followed by
    # func.__get__(opt, opt.__class__). both only exist on a bound method, so wrapping step with a
    # plain function crashed every opd run at engine construction, before any gpu work, with
    # "'function' object has no attribute '__func__'". this replays that exact sequence rather
    # than importing torch so it gates on machines without the training extras installed.
    import flash.engine.worker.opd_plugin as plugin

    coordination = []
    updates = []

    class Optimizer:
        def step(self, scale=1):
            updates.append(scale)
            return len(updates)

    monkeypatch.setattr(
        plugin,
        "_coordinate_first_mutation_notice",
        lambda url, token: coordination.append((url, token)),
        raising=False,
    )
    optimizer = plugin._wrap_optimizer_with_mutation_notice(
        Optimizer(),
        "http://bridge",
        "token",
    )

    function = optimizer.step.__func__
    optimizer.step = function.__get__(optimizer, type(optimizer))

    assert optimizer.step(scale=2) == 1
    assert updates == [2]
    assert coordination == [("http://bridge", "token")]


def test_each_optimizer_rank_acknowledges_marker_while_parent_publishes_once(
    monkeypatch,
):
    import flash.engine.worker.opd_plugin as plugin

    world = _ThreadedDistributedWorld(2)
    callback_calls = []
    rank_updates = []
    bridge = _text_bridge(
        _BridgeTeacher(),
        mutation_callback=lambda: (
            callback_calls.append(True),
            world.events.append(("callback", 0)),
        ),
    )

    class Optimizer:
        def __init__(self, rank):
            self.rank = rank

        def step(self):
            world.events.append(("step", self.rank))
            rank_updates.append(self.rank)

    monkeypatch.setattr(plugin, "_mutation_distributed", lambda: world, raising=False)
    bridge.start()
    try:
        optimizers = [
            plugin._wrap_optimizer_with_mutation_notice(Optimizer(rank), bridge.url, bridge.token)
            for rank in range(2)
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(
                pool.map(
                    lambda item: world.run(item[0], item[1].step),
                    enumerate(optimizers),
                )
            )
    finally:
        bridge.close()

    readiness = [event for event in world.events if event[:2] == ("barrier", 0)]
    confirmation = [event for event in world.events if event[:2] == ("barrier", 1)]
    callback_index = next(
        index for index, event in enumerate(world.events) if event[0] == "callback"
    )
    first_step_index = next(index for index, event in enumerate(world.events) if event[0] == "step")
    assert len(readiness) == 2
    assert len(confirmation) == 2
    assert callback_calls == [True]
    assert sorted(rank_updates) == [0, 1]
    assert callback_index > max(world.events.index(event) for event in readiness)
    assert first_step_index > max(world.events.index(event) for event in confirmation)


def test_failed_rank_readiness_prevents_marker_callback_and_optimizer_steps(
    monkeypatch,
):
    import flash.engine.worker.opd_plugin as plugin

    callbacks = []
    updates = []

    class UnreadyWorld:
        def barrier(self):
            raise RuntimeError("world readiness failed")

        def get_rank(self):
            return 0

        def broadcast_object_list(self, _values, _src):
            raise AssertionError("publication outcome must not broadcast")

    class Optimizer:
        def step(self):
            updates.append(True)

    monkeypatch.setattr(
        plugin,
        "_mutation_distributed",
        lambda: UnreadyWorld(),
        raising=False,
    )
    monkeypatch.setattr(
        plugin,
        "_publish_mutation_notice",
        lambda _url, _token: callbacks.append(True),
        raising=False,
    )
    optimizer = plugin._wrap_optimizer_with_mutation_notice(Optimizer(), "http://bridge", "token")

    with pytest.raises(RuntimeError, match="readiness failed"):
        optimizer.step()

    assert callbacks == []
    assert updates == []


def test_marker_publication_failure_reaches_all_ranks_before_optimizer_step(
    monkeypatch,
):
    import flash.engine.worker.opd_plugin as plugin

    world = _ThreadedDistributedWorld(2)
    callbacks = []
    updates = []

    class Optimizer:
        def __init__(self, rank):
            self.rank = rank

        def step(self):
            updates.append(self.rank)

    def fail_publication(_url, _token):
        callbacks.append(True)
        raise FlashTeacherBridgeError(
            "marker publication failed",
            classification="transient",
        )

    monkeypatch.setattr(plugin, "_mutation_distributed", lambda: world, raising=False)
    monkeypatch.setattr(
        plugin,
        "_publish_mutation_notice",
        fail_publication,
        raising=False,
    )
    optimizers = [
        plugin._wrap_optimizer_with_mutation_notice(Optimizer(rank), "http://bridge", "token")
        for rank in range(2)
    ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(world.run, rank, optimizer.step)
            for rank, optimizer in enumerate(optimizers)
        ]
        for future in futures:
            with pytest.raises(FlashTeacherBridgeError, match="publication failed"):
                future.result()

    assert callbacks == [True]
    assert updates == []


def test_bridge_publishes_the_marker_exactly_once_across_concurrent_ranks():
    """The bridge gate must collapse every rank's first-mutation notice into one marker publish.

    trl drove the marker from a single-process optimizer helper (_apply_opd_optimizer_update), so
    "exactly once" was trivially true and tested there. verl runs one worker per rank and each wraps
    its own optimizer, so N ranks race into notify_mutation at the first step -- and a second publish
    would commit a duplicate marker for the attempt. The _mutation_lock + _mutation_notified gate is
    what makes that safe, and it had no direct coverage: the tests above drive the plugin-side
    wrapper, which stops at one notice per rank, not one notice per run.
    """
    calls = []
    bridge = _text_bridge(_BridgeTeacher(), mutation_callback=lambda: calls.append(True))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _rank: bridge.notify_mutation(), range(8)))

    assert calls == [True]
    # a later notice is still a no-op: the marker is published once per attempt, not once per step.
    bridge.notify_mutation()
    assert calls == [True]
    assert bridge.mutation_failure is None


def test_first_parent_mutation_failure_is_replayed_to_all_ranks_without_steps():
    from flash.engine.worker.perf import RetriableInfraError

    callback_calls = []
    optimizer_steps = []

    def fail_once_then_succeed():
        callback_calls.append(True)
        if len(callback_calls) == 1:
            raise RetriableInfraError("first marker failure")

    bridge = _text_bridge(_BridgeTeacher(), mutation_callback=fail_once_then_succeed)
    bridge.start()

    def rank_step(_rank):
        try:
            _post_json(bridge.url, bridge.token, "/mutation", {})
        except FlashTeacherBridgeError as error:
            return error.classification, str(error)
        optimizer_steps.append(True)
        return "success", ""

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(rank_step, range(4)))
        bridge._record_mutation_failure("permanent", "fallback evidence")
    finally:
        bridge.close()

    expected_message = str(RetriableInfraError("first marker failure"))
    assert callback_calls == [True]
    assert optimizer_steps == []
    assert results == [("transient", expected_message)] * 4
    assert bridge.mutation_failure == ("transient", expected_message)


@pytest.mark.parametrize(
    ("retriable", "expected_exit"),
    [(True, 87), (False, 86)],
)
def test_mutation_marker_failure_preserves_bridge_classification(
    monkeypatch, retriable, expected_exit
):
    import flash.engine.worker.opd_plugin as plugin
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


@pytest.mark.parametrize(
    "classifications",
    [("permanent", "transient"), ("transient", "permanent")],
)
def test_mutation_failure_preserves_permanent_precedence(classifications):
    bridge = _text_bridge(_BridgeTeacher())

    for classification in classifications:
        bridge._record_mutation_failure(
            classification,
            f"{classification} failure",
        )

    assert bridge.mutation_failure == ("permanent", "permanent failure")


def test_mutation_success_response_disconnect_does_not_latch_marker_failure():
    callback_calls = []
    bridge = _text_bridge(_BridgeTeacher(), mutation_callback=lambda: callback_calls.append(True))
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


def test_mutation_lost_success_response_retries_once_without_republishing_marker(
    monkeypatch, tmp_path
):
    import flash.engine.worker.opd_plugin as plugin

    callback_calls = []
    optimizer_steps = []
    bridge = _text_bridge(_BridgeTeacher(), mutation_callback=lambda: callback_calls.append(True))
    failure_path = str(tmp_path / "mutation-failure")
    monkeypatch.setenv("FLASH_OPD_MUTATION_FAILURE_PATH", failure_path)

    class UnexpectedChildExit(RuntimeError):
        pass

    def child_exit(code):
        raise UnexpectedChildExit(code)

    monkeypatch.setattr(plugin.os, "_exit", child_exit)
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    original_send = handler._send_json
    sends = []

    def lose_first_response(request_handler, status, payload):
        sends.append(status)
        if len(sends) <= 2:
            raise BrokenPipeError("client disconnected")
        return original_send(request_handler, status, payload)

    handler._send_json = lose_first_response
    try:
        plugin._post_mutation_notice(bridge.url, bridge.token)
        optimizer_steps.append(True)
    finally:
        bridge.close()

    assert callback_calls == [True]
    assert optimizer_steps == [True]
    assert sends == [200, 422, 200]
    assert not list(tmp_path.glob("mutation-failure.*.json"))
    assert bridge.mutation_failure is None


def test_incomplete_mutation_response_retries_once_without_republishing_marker(
    monkeypatch, tmp_path
):
    import flash.engine.worker.opd_plugin as plugin

    callback_calls = []
    optimizer_steps = []
    bridge = _text_bridge(_BridgeTeacher(), mutation_callback=lambda: callback_calls.append(True))
    failure_path = str(tmp_path / "mutation-failure")
    monkeypatch.setenv("FLASH_OPD_MUTATION_FAILURE_PATH", failure_path)

    class UnexpectedChildExit(RuntimeError):
        pass

    def child_exit(code):
        raise UnexpectedChildExit(code)

    monkeypatch.setattr(plugin.os, "_exit", child_exit)
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    original_send = handler._send_json
    sends = []

    def truncate_first_response(request_handler, status, payload):
        sends.append(status)
        if len(sends) == 1:
            return _send_truncated_json_response(request_handler, status, payload)
        return original_send(request_handler, status, payload)

    handler._send_json = truncate_first_response
    try:
        plugin._post_mutation_notice(bridge.url, bridge.token)
        optimizer_steps.append(True)
    finally:
        bridge.close()

    assert callback_calls == [True]
    assert optimizer_steps == [True]
    assert sends == [200, 200]
    assert not list(tmp_path.glob("mutation-failure.*.json"))
    assert bridge.mutation_failure is None


def test_persistent_incomplete_mutation_response_fails_closed_after_one_retry(
    monkeypatch, tmp_path
):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    callback_calls = []
    optimizer_steps = []
    bridge = _text_bridge(_BridgeTeacher(), mutation_callback=lambda: callback_calls.append(True))
    failure_path = str(tmp_path / "mutation-failure")
    monkeypatch.setenv("FLASH_OPD_MUTATION_FAILURE_PATH", failure_path)
    bridge.start()
    handler = bridge._server.RequestHandlerClass
    sends = []

    def truncate_response(request_handler, status, payload):
        sends.append(status)
        return _send_truncated_json_response(request_handler, status, payload)

    class ChildExit(RuntimeError):
        def __init__(self, code):
            super().__init__(code)
            self.code = code

    def child_exit(code):
        raise ChildExit(code)

    handler._send_json = truncate_response
    monkeypatch.setattr(plugin.os, "_exit", child_exit)
    try:
        with pytest.raises(ChildExit) as exit_error:
            plugin._post_mutation_notice(bridge.url, bridge.token)
        fallback = _read_classified_failure_fallback(failure_path)
    finally:
        bridge.close()

    assert callback_calls == [True]
    assert optimizer_steps == []
    assert sends == [200, 200]
    assert exit_error.value.code == 87
    assert fallback is not None
    assert fallback[0] == "transient"
    assert http.client.IncompleteRead.__name__ in fallback[1]


def test_persistent_mutation_response_loss_writes_fallback_without_optimizer_step(
    monkeypatch, tmp_path
):
    import flash.engine.worker.opd_plugin as plugin
    from flash.engine.worker.opd_train import _read_classified_failure_fallback

    callback_calls = []
    optimizer_steps = []
    bridge = _text_bridge(_BridgeTeacher(), mutation_callback=lambda: callback_calls.append(True))
    failure_path = str(tmp_path / "mutation-failure")
    monkeypatch.setenv("FLASH_OPD_MUTATION_FAILURE_PATH", failure_path)
    bridge.start()
    handler = bridge._server.RequestHandlerClass

    def disconnect_during_response(_handler, _status, _payload):
        raise BrokenPipeError("client disconnected")

    class ChildExit(RuntimeError):
        pass

    def child_exit(code):
        raise ChildExit(code)

    handler._send_json = disconnect_during_response
    monkeypatch.setattr(plugin.os, "_exit", child_exit)
    try:
        with pytest.raises(ChildExit):
            plugin._post_mutation_notice(bridge.url, bridge.token)
        fallback = _read_classified_failure_fallback(failure_path)
        assert fallback is not None
        bridge._record_mutation_failure(*fallback)
    finally:
        bridge.close()

    assert callback_calls == [True]
    assert optimizer_steps == []
    assert bridge.mutation_failure is None


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
        "max_sequence_length": 1536,
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
        "reward_path": "/w/shim/flash_opd_reward.py",
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
    modules["verl.utils.checkpoint.checkpoint_handler"].CheckpointHandler = FakeCheckpointHandler
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = _render_opd_sitecustomize(save_at_steps=(3, 7), total_steps=7)
    exec(compile(source, "sitecustomize.py", "exec"), {})
    handler = FakeCheckpointHandler()

    results = [handler.save_checkpoint(step) for step in range(1, 8)]

    assert saved == [3, 7]
    assert results == [None, None, 3, None, None, None, 7]


def test_overrides_match_verl_0_8_sync_distillation_contract():
    overrides = dict(value.split("=", 1) for value in build_opd_overrides(_config()))
    assert overrides["distillation._target_"] == "flash_opd_plugin.FlashRemoteDistillationConfig"
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
    assert overrides["+ray_kwargs.ray_init.num_gpus"] == "4"
    # the engine is sized to the job's own sequence length, never a hardcoded context.
    assert overrides["actor_rollout_ref.rollout.max_model_len"] == "1536"
    assert overrides["distillation.n_gpus_per_node"] == "0"
    assert overrides["distillation.nnodes"] == "0"
    assert overrides["distillation.teacher_key"] == "index"
    assert overrides["data.image_key"] == "images"
    assert overrides["data.return_raw_chat"] == "true"
    assert overrides["data.return_multi_modal_inputs"] == "false"
    # `++`-prefixed: these keys are absent from the composed node, so a bare assignment would abort
    # the run at hydra composition. see build_opd_overrides for the per-key reasoning.
    assert overrides["++actor_rollout_ref.rollout.limit_images"] == "8"
    assert overrides["++actor_rollout_ref.rollout.engine_kwargs.vllm.seed"] == "42"
    assert "actor_rollout_ref.rollout.seed" not in overrides
    assert overrides["actor_rollout_ref.rollout.load_format"] == "safetensors"
    assert "actor.engine.ulysses_sequence_parallel_size" not in overrides
    assert "ref_log_prob" not in " ".join(overrides)
    assert not any("structured_outputs_config" in key for key in overrides)

    multi_turn_overrides = dict(
        value.split("=", 1)
        for value in build_opd_overrides(_config(multi_turn=True, max_sequence_length=1536))
    )
    assert (
        multi_turn_overrides["actor_rollout_ref.rollout.agent.default_agent_loop"]
        == "flash_multi_turn"
    )
    assert multi_turn_overrides["actor_rollout_ref.rollout.prompt_length"] == "1536"
    assert multi_turn_overrides["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == "1536"


def test_overrides_carry_fused_expert_target_parameters():
    overrides = dict(
        value.split("=", 1)
        for value in build_opd_overrides(
            _config(
                target_parameters=[
                    "mlp.experts.gate_up_proj",
                    "mlp.experts.down_proj",
                ]
            )
        )
    )

    assert overrides["++actor_rollout_ref.model.target_parameters"] == (
        "[mlp.experts.gate_up_proj,mlp.experts.down_proj]"
    )


def test_overrides_size_agent_loop_workers_to_the_opd_rollout_batch():
    # opd runs async rollout through verl's AgentLoopManager, which chunks
    # train_batch_size * group_size across agent.num_workers and asserts the split is exact.
    # verl's default of 8 aborts before the first step on e.g. 2 x 2 = 4.
    small = dict(
        value.split("=", 1)
        for value in build_opd_overrides(_config(train_batch_size=2, group_size=2))
    )
    assert small["actor_rollout_ref.rollout.agent.num_workers"] == "4"
    # the common case still gets the full worker pool.
    big = dict(
        value.split("=", 1)
        for value in build_opd_overrides(_config(train_batch_size=64, group_size=8))
    )
    assert big["actor_rollout_ref.rollout.agent.num_workers"] == "8"


def test_overrides_gate_rollout_enforce_eager_on_hardware():
    off = build_opd_overrides(_config())
    assert not any("enforce_eager" in value for value in off)
    on = build_opd_overrides(_config(enforce_eager=True))
    # plain override, not '+': enforce_eager is a declared verl RolloutConfig field
    # (workers/config/rollout.py:194), so appending it would be a duplicate-key error.
    assert "actor_rollout_ref.rollout.enforce_eager=True" in on


def test_the_resolved_eager_flag_reaches_the_opd_verl_config():
    # the string assertions above pass against a resolver whose answer is never carried into the
    # config, which is exactly the shape of the bug this fixes. pin the wiring.
    built = inspect.getsource(opd_train.run_opd_train)
    # one probe now feeds both eager and the attention pins, so the call is bound to a name rather
    # than nested inline. assert the resolver still consumes THAT probe, not a second one.
    assert "verl_cc = verl_device_capability(caps)" in built
    assert "resolve_rollout_enforce_eager(verl_cc)" in built
    assert '"enforce_eager": enforce_eager,' in built


def test_opd_pins_the_blackwell_attention_backends_like_grpo_does():
    """VERL-156: opd's rollout died on B200 because it never pinned the ViT attention backend.

    vllm 0.19.1 defaults the ViT to a CUTE flash-attn that is unimportable against every published
    nvidia-cutlass-dsl, so the engine aborts with `cutlass.cute.core has no attribute 'ThrMma'`.
    grpo has pinned both backends since the trl driver; opd never did, which is the same one-sided
    fix that produced the enforce_eager bug directly above. two attempts on two fresh B200s
    reproduced it byte-identically before this was found.
    """
    # off blackwell the resolver answers (None, None) and nothing may be emitted -- an unconditional
    # pin would change the rollout on every other card.
    off = build_opd_overrides(_config())
    assert not any("attention_backend" in value for value in off)

    on = build_opd_overrides(
        _config(attention_backend="FLASHINFER", mm_encoder_attn_backend="TORCH_SDPA")
    )
    # '+' appends under the existing engine_kwargs.vllm struct: these are AsyncEngineArgs fields
    # spread by verl, not declared RolloutConfig fields, so a bare assignment would not reach them.
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASHINFER" in on, on
    assert (
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_attn_backend=TORCH_SDPA" in on
    ), on


def test_both_ray_rollouts_pin_blackwell_attention_from_the_same_resolver():
    # the enforce_eager cross-trainer check below exists because a one-sided fix is how that bug
    # shipped. the attention pins are the same divergence one knob over, so assert them the same
    # way rather than trusting opd alone.
    for module, source in (
        ("rl_train", inspect.getsource(rl_train.run_rl_train)),
        ("opd_train", inspect.getsource(opd_train.run_opd_train)),
    ):
        # the resolver must be CALLED and its second return value must be carried onward. the two
        # trainers spell the handoff differently (rl_train passes it as a kwarg into
        # _rl_runconfig, opd_train builds its config dict inline), so assert the binding both
        # share rather than either one's syntax.
        assert "resolve_blackwell_attention_backends(" in source, module
        assert "attention_backend, mm_encoder_attn_backend = " in source, module
        assert "mm_encoder_attn_backend=mm_encoder_attn_backend" in source or (
            '"mm_encoder_attn_backend": mm_encoder_attn_backend,' in source
        ), module


def test_both_ray_rollouts_resolve_eager_from_the_same_hardware_probe():
    # asserted across BOTH trainers rather than in one file: grpo has resolved this since the trl
    # driver and opd did not, so opd captured 102 cuda graphs on an rtx 4090 and died on HOST ram
    # during the first weight sync. opd is the more exposed path, not the less -- it always runs
    # rollout.mode=async, whose server hardcodes cudagraph_mode=FULL_AND_PIECEWISE. a fix applied to
    # only one trainer is how this reached production in the first place.
    for module, source in (
        ("rl_train", inspect.getsource(rl_train.run_rl_train)),
        ("opd_train", inspect.getsource(opd_train.run_opd_train)),
    ):
        assert "resolve_rollout_enforce_eager(" in source, module
        # every capability question rides ONE child probe per run; the eager decision reads that
        # blob rather than spawning a torch import of its own.
        assert "caps = probe_verl_capabilities(python_bin, gdn_module)" in source, module
        assert "verl_cc = verl_device_capability(caps)" in source, module


def test_overrides_pin_the_rollout_resident_for_sleep_unsupported_models():
    off = build_opd_overrides(_config())
    assert not any("free_cache_engine" in value for value in off)
    assert not any("enable_sleep_mode" in value for value in off)
    on = build_opd_overrides(_config(sleep_unsupported=True))
    # both knobs, not one: free_cache_engine gates the sleep()/wake_up() rpcs
    # (vllm_async_server.py:626), enable_sleep_mode is what builds the sleep-capable engine (:265).
    # they need DIFFERENT hydra prefixes -- rollout_resident_overrides' docstring has the why.
    #
    # asserted EXACTLY rather than as a substring, because "x=false" is a substring of "+x=false":
    # the obvious assertion passes against the spelling that kills the run at parse, which is how
    # this shipped. see ISSUES.md VERL-148.
    assert "actor_rollout_ref.rollout.free_cache_engine=false" in on
    assert "+actor_rollout_ref.rollout.enable_sleep_mode=false" in on
    assert "actor_rollout_ref.rollout.enable_sleep_mode=false" not in on


def test_the_resolved_sleep_flag_reaches_the_opd_verl_config():
    # the assertions above pass against a resolver whose answer never reaches the config, which is
    # the exact shape of the bug this fixes. pin the wiring, not just the string.
    built = inspect.getsource(opd_train.run_opd_train)
    assert '"sleep_unsupported": rollout_sleep_unsupported(model_id),' in built


def test_both_ray_rollouts_honor_sleep_unsupported_from_the_same_catalog_flag():
    # opd is NOT exempt from verl's sleep path: main_ppo_sync.py:740 calls sleep_replicas() during
    # init_workers and again around validation, landing in the same vllm_async_server sleep(). the
    # flagged model declares algos including opd and the parse-time gate ADMITS it (routes to b200),
    # so a driver that ignores the flag wedges on wake. asserted across BOTH trainers because
    # one-trainer-only is precisely how the eager defect above reached production.
    for module, source in (
        ("rl_train", inspect.getsource(rl_train)),
        ("opd_train", inspect.getsource(opd_train)),
    ):
        assert "rollout_resident_overrides(" in source, module
        assert "rollout_sleep_unsupported(" in source, module


def test_a_flagged_model_is_actually_reachable_under_opd():
    # the guard above is only meaningful while some catalog model is both sleep_unsupported and
    # opd-capable. if that stops being true this test fails loudly rather than leaving a guard with
    # no subject silently passing forever.
    from flash.catalog import MODELS

    flagged = [m for m, i in MODELS.items() if getattr(i, "sleep_unsupported", False)]
    assert flagged, "no catalog model is sleep_unsupported; this guard now has no subject"
    assert any("opd" in (MODELS[m].algos or ()) for m in flagged), (
        "no sleep_unsupported model allows opd; the opd resident pin now has no subject"
    )


def test_overrides_size_the_engine_to_the_job_not_a_hardcoded_context():
    # regression: single-turn opd hardcoded rollout.max_model_len=32768 while the prompt budget was
    # carved out of the job's own max_context_tokens. above 32768 the prompt filter admitted prompts
    # the engine could not hold, so they died at rollout instead of training on a shorter context;
    # below it, vllm reserved kv cache for a context the job never uses. every length must descend
    # from one value.
    overrides = dict(
        value.split("=", 1)
        for value in build_opd_overrides(
            _config(max_prompt_length=65024, max_response_length=512, max_sequence_length=65536)
        )
    )
    assert overrides["actor_rollout_ref.rollout.max_model_len"] == "65536"
    # the token budgets track the same value, so a full-length sequence always fits a micro-batch.
    assert overrides["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == "65536"
    assert overrides["actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu"] == "65536"
    # and the prompt filter's budget is carved out of the engine, never larger than it.
    assert int(overrides["data.max_prompt_length"]) + int(
        overrides["data.max_response_length"]
    ) == int(overrides["actor_rollout_ref.rollout.max_model_len"])


def test_overrides_require_an_explicit_sequence_length():
    # the engine length is load-bearing for the kv cache, the prompt filter, and the token budget.
    # a defaulted one would silently size the run for a context the caller never asked for, which is
    # exactly the bug above. missing means fail loudly, before any gpu is paid for.
    config = _config()
    del config["max_sequence_length"]
    with pytest.raises(KeyError, match="max_sequence_length"):
        build_opd_overrides(config)


def test_overrides_bound_transfer_queue_storage_to_one_unit():
    # verl force-enables TransferQueue on the opd entry point and defaults SimpleStorage to 8
    # storage units. tq.init reserves them with a SPREAD placement group and blocks in
    # ray.get(pg.ready()) with no timeout, so any cluster with fewer free cpus than units hangs
    # the run forever before a gpu is touched. flash's trainer is single-node: one unit.
    overrides = dict(value.split("=", 1) for value in build_opd_overrides(_config()))
    assert overrides["transfer_queue.backend.SimpleStorage.num_data_storage_units"] == "1"


def test_overrides_route_reward_scoring_away_from_verls_data_source_registry():
    """OPD must configure a custom reward function even though it carries no task reward.

    `reward.reward_model.enable=false` disables the reward MODEL, not reward SCORING. verl's
    RewardLoopWorker (reward_loop.py:146-155) branches on custom_reward_function.path first and
    only falls through to the default rule-based scorer when it is None -- and that scorer
    dispatches on data_source against a builtin registry that has no "flash_opd" entry, so it
    raises NotImplementedError on every rollout and kills the run at step 0 (VERL-153).

    Both key forms are required. The legacy top-level key is migrated onto reward.* in the main
    process, but the RewardLoopWorker actor reads its own config copy and never sees that
    migration, so emitting only one leaves the actor on the failing branch.
    """
    overrides = dict(value.split("=", 1) for value in build_opd_overrides(_config()))

    for key in ("custom_reward_function.path", "reward.custom_reward_function.path"):
        assert overrides[key] == "/w/shim/flash_opd_reward.py"
    for key in ("custom_reward_function.name", "reward.custom_reward_function.name"):
        assert overrides[key] == "compute_score"

    # the branch this fix depends on: the reward model stays off, so a missing custom path would
    # land on the default scorer rather than on some other safe fallback.
    assert overrides["reward.reward_model.enable"] == "false"


def test_generated_opd_reward_shim_scores_every_rollout_zero():
    """The generated shim must be importable and return 0.0, matching use_task_rewards=false.

    OPD's gradient comes from the teacher distribution; a nonzero task reward here would silently
    add signal the objective never asked for.
    """
    namespace: dict = {}
    # exec, not import: the shim only exists on disk inside a live workdir, and the constant is
    # what actually ships to the worker.
    exec(opd_train._OPD_ZERO_REWARD_SOURCE, namespace)

    compute_score = namespace["compute_score"]
    assert compute_score("flash_opd", "any completion", "") == 0.0
    assert compute_score("flash_opd", "", "gt", {"index": 3}) == 0.0


def test_opd_rollout_reserves_the_fp8_kv_cache_its_sizing_assumes():
    """The OPD engine must ask vLLM for fp8 KV, because vram.py already sized the run for it.

    `_opd_fp8_adjust` halves an OPD requirement once it clears the largest non-fp8 card, on the
    premise that the colocated rollout reserves an fp8 cache. That premise was supplied by the trl
    engine (opd_vllm.py set kv_cache_dtype="fp8" on cc >= 8.9), which is deleted -- run_opd now
    delegates to verl unconditionally. Without this override the allocator reserves an fp8-sized
    pool while vLLM allocates a bf16 one: double the real KV, OOM at rollout init on a card sizing
    called sufficient. The two must move together or not at all.
    """
    on = dict(value.split("=", 1) for value in build_opd_overrides(_config(fp8_kv=True)))
    assert on["+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype"] == "fp8"

    # bf16 is the conservative default: no flag, no key (sizing only discounts when the run is
    # provably modern-card-only, so an un-probed run must not claim an fp8 pool it did not request).
    for cfg in (_config(), _config(fp8_kv=False)):
        off = dict(value.split("=", 1) for value in build_opd_overrides(cfg))
        assert not [k for k in off if "kv_cache_dtype" in k]


def test_remote_distillation_config_declares_every_field_post_init_assigns():
    # verl's BaseConfig.__setattr__ (base_config.py) raises FrozenInstanceError for any assignment to
    # an already-set field that is not in _mutable_fields. FlashRemoteDistillationConfig's __post_init__
    # normalizes distillation_loss and clears teacher_models, so BOTH must be declared mutable -- else
    # hydra's instantiate of distillation._target_ dies at worker startup, i.e. only on GPU, after the
    # run is already paid for. That is exactly how this shipped: the only other coverage of this class
    # asserts the _target_ STRING and never constructs it.
    #
    # Checked by AST rather than by instantiating: verl is not installed in CI (it lives in a separate
    # baked venv), so an importorskip test would skip here and prove nothing on the machine that gates
    # the merge. Reading the source needs no verl at runtime and still fails on a new unlisted field.
    import ast
    import inspect

    from flash.engine.worker import opd_plugin as plugin

    installer = ast.parse(inspect.getsource(plugin._install_verl_extensions))
    class_defs = [
        node
        for node in ast.walk(installer)
        if isinstance(node, ast.ClassDef) and node.name == "FlashRemoteDistillationConfig"
    ]
    assert len(class_defs) == 1, "expected exactly one FlashRemoteDistillationConfig definition"
    class_def = class_defs[0]

    declared: set[str] = set()
    post_init = None
    for node in class_def.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_mutable_fields" for t in node.targets
        ):
            # the value is `<Parent>._mutable_fields | {...}`; only the literal half is readable here,
            # so union in the parent's own set below rather than guessing at its contents.
            declared |= {
                el.value
                for el in ast.walk(node.value)
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            }
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__":
            post_init = node

    assert post_init is not None, "__post_init__ disappeared; this guard no longer covers anything"

    # verl's DistillationConfig already declares these, so flash's literal need not repeat them.
    inherited = {"teacher_models", "n_gpus_per_node", "nnodes"}
    assigned = {
        target.attr
        for node in ast.walk(post_init)
        for target in getattr(node, "targets", [])
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert assigned, "no self.<field> assignments found; the AST walk is not seeing the body"
    assert "distillation_loss" in assigned  # pin the specific field that broke

    undeclared = assigned - declared - inherited
    assert not undeclared, (
        f"__post_init__ assigns {sorted(undeclared)} but they are not in _mutable_fields; "
        "verl will raise FrozenInstanceError at worker startup"
    )


def test_agent_loops_are_registered_under_an_importable_qualname():
    # verl's `register` (agent_loop.py) stores f"{cls.__module__}.{cls.__qualname__}" into the agent
    # loop registry at decoration time, and hydra resolves that string later with a plain dotted-path
    # import. both flash loops are defined inside a function, so their natural qualname carries
    # `<locals>` and NOTHING can import it: the run dies at first rollout with
    # "Error locating target ... <locals>.FlashSingleTurnAgentLoop". the fix is ordering-sensitive --
    # rewriting __qualname__ after the decorator already ran does not help, because the registry
    # captured the broken string. so assert both dunders are rewritten BEFORE the register call.
    #
    # AST rather than import, for the same reason as the config test above: verl is not installed in
    # CI, so an importorskip guard would silently skip on the machine that gates the merge.
    import ast
    import inspect

    from flash.engine.worker import opd_multiturn, opd_plugin

    sources = {
        "flash_single_turn": inspect.getsource(opd_plugin._install_verl_extensions),
        "flash_multi_turn": inspect.getsource(opd_multiturn.build_flash_multi_turn_agent_loop),
    }

    for agent_name, source in sources.items():
        body = ast.unparse(ast.parse(source))
        cls = "FlashSingleTurnAgentLoop" if "single" in agent_name else "FlashMultiTurnAgentLoop"

        # a decorator would register the class before either dunder can be corrected
        assert f"@register('{agent_name}')" not in body, (
            f"{cls} is registered by decorator; verl freezes the `<locals>` qualname at that point"
        )

        register_at = body.find(f"register('{agent_name}')")
        assert register_at != -1, f"{cls} is no longer registered as {agent_name}"

        for dunder in ("__module__", "__qualname__"):
            set_at = body.find(f"{cls}.{dunder} =")
            assert set_at != -1, f"{cls}.{dunder} is not rewritten; hydra cannot locate the class"
            assert set_at < register_at, (
                f"{cls}.{dunder} is rewritten after register(); the registry already captured "
                "the unimportable name"
            )


def test_teacher_logprobs_patch_precedes_the_main_ppo_sync_import():
    # `@ray.remote` copies inherited methods onto the actor class it decorates, so
    # `AgentLoopWorkerTQ(AgentLoopWorker)` in verl.trainer.main_ppo_sync freezes whatever
    # `_compute_teacher_logprobs` resolves to at import time. importing that module before flash
    # patches AgentLoopWorker snapshots verl's original, which calls the teacher manager with
    # `sequence_ids=` while FlashBridgeTeacherManager takes prompt_ids/response_ids -- the first
    # rollout then dies with "unexpected keyword argument 'sequence_ids'". patching the parent is
    # only visible to the actor when it happens BEFORE the import, so assert that order.
    #
    # AST rather than import, for the same reason as the test above: verl is not installed in CI.
    import ast
    import inspect

    from flash.engine.worker import opd_plugin

    body = ast.unparse(ast.parse(inspect.getsource(opd_plugin._install_verl_extensions)))

    patch_at = body.find("AgentLoopWorker._compute_teacher_logprobs = ")
    assert patch_at != -1, "AgentLoopWorker._compute_teacher_logprobs is no longer patched"

    import_at = body.find("from verl.trainer.main_ppo_sync import")
    assert import_at != -1, "main_ppo_sync is no longer imported inside _install_verl_extensions"

    assert patch_at < import_at, (
        "main_ppo_sync is imported before AgentLoopWorker._compute_teacher_logprobs is patched; "
        "AgentLoopWorkerTQ froze verl's original method and will call the flash teacher manager "
        "with sequence_ids="
    )


def test_structured_overrides_pin_xgrammar_and_thinking_parser():
    overrides = dict(
        value.split("=", 1)
        for value in build_opd_overrides(
            _config(
                thinking=True,
                structured_outputs={"choice": ["4"]},
            )
        )
    )

    assert (
        overrides["+actor_rollout_ref.rollout.engine_kwargs.vllm.structured_outputs_config"]
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


def test_child_environment_keeps_bridge_but_excludes_teacher_transport(monkeypatch, tmp_path):
    monkeypatch.setenv("PARASAIL_API_KEY", "parasail-secret")
    monkeypatch.setenv("FLASH_CONTROL_PANEL_URL", "https://broker.example")
    monkeypatch.setenv("FLASH_TEACHER_CAPABILITY", "capability-secret")
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
        mutation_failure_path=str(tmp_path / "mutation-failure"),
        score_delivery_failure_path=str(tmp_path / "score-delivery-failure"),
        abandonment_failure_path=str(tmp_path / "abandonment-failure"),
        resample_failure_path=str(tmp_path / "resample-failure"),
        cycle_commit_failure_path=str(tmp_path / "cycle-commit-failure"),
    )
    assert child["FLASH_OPD_BRIDGE_URL"] == "http://127.0.0.1:4444"
    assert child["FLASH_OPD_BRIDGE_TOKEN"] == "bridge-token"
    assert child["FLASH_OPD_MUTATION_FAILURE_PATH"] == str(tmp_path / "mutation-failure")
    assert child["FLASH_OPD_SCORE_DELIVERY_FAILURE_PATH"] == str(
        tmp_path / "score-delivery-failure"
    )
    assert child["FLASH_OPD_ABANDONMENT_FAILURE_PATH"] == str(tmp_path / "abandonment-failure")
    assert child["FLASH_OPD_RESAMPLE_FAILURE_PATH"] == str(tmp_path / "resample-failure")
    assert child["FLASH_OPD_CYCLE_COMMIT_FAILURE_PATH"] == str(tmp_path / "cycle-commit-failure")
    assert child["VERL_USE_EXTERNAL_MODULES"] == "flash_opd_plugin"
    assert child["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "PARASAIL_API_KEY" not in child
    assert "FLASH_CONTROL_PANEL_URL" not in child
    assert "FLASH_TEACHER_CAPABILITY" not in child
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
    assert "PARASAIL_API_KEY" not in child


def test_structured_validator_rejects_vllm_mistral_tokenizer_models(monkeypatch):
    from flash import opd_validation

    monkeypatch.setattr(
        opd_validation,
        "_resolve_structured_model_metadata",
        lambda *_args: (32000, ("config.json", "tokenizer.model.v3")),
    )

    with pytest.raises(ValueError, match="does not support vLLM MistralTokenizer"):
        validate_opd_structured_outputs(
            '{"choice":["4"]}',
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            compiler_vocab_size=32000,
        )


def test_unstructured_validator_does_not_resolve_model_metadata(monkeypatch):
    from flash import opd_validation

    def unexpected(*_args, **_kwargs):
        pytest.fail("unstructured OPD must not resolve structured model metadata")

    monkeypatch.setattr(opd_validation, "_resolve_compiler_vocab_size", unexpected)
    monkeypatch.setattr(opd_validation, "_resolve_structured_model_metadata", unexpected)

    result = validate_opd_structured_outputs(
        None,
        model_id="open-org/cached-model",
        model_revision="d" * 40,
        model_policy="allow",
    )

    assert result.constraint is None
    assert result.model_vocab_size == 0


def test_structured_runtime_rejects_vllm_below_minimum(monkeypatch):
    versions = {"verl": "0.8.0", "vllm": "0.9.0", "xgrammar": "0.1.25"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match=r"vllm 0\.11\.0 or newer; found 0\.9\.0"):
        _require_structured_runtime_versions()


def test_structured_runtime_accepts_shipped_versions(monkeypatch):
    versions = {"verl": "0.8.0", "vllm": "0.19.1", "xgrammar": "0.1.25"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)

    _require_structured_runtime_versions()


def test_structured_runtime_rejects_wrong_verl_version(monkeypatch):
    versions = {"verl": "0.8.1", "vllm": "0.19.1", "xgrammar": "0.1.25"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match=r"verl 0\.8\.0 exactly"):
        _require_structured_runtime_versions()


def test_structured_runtime_rejects_wrong_xgrammar_version(monkeypatch):
    versions = {"verl": "0.8.0", "vllm": "0.19.1", "xgrammar": "0.1.26"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match=r"xgrammar 0\.1\.25 exactly"):
        _require_structured_runtime_versions()


def test_plugin_initialization_checks_structured_versions_before_verl_install(monkeypatch):
    import importlib as importlib_module

    from flash.engine.worker import opd_plugin as plugin

    versions = {"verl": "0.8.0", "vllm": "0.19.1", "xgrammar": "0.1.26"}
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
    assert (
        len(
            {
                baseline,
                deterministic_rollout_seed(43, 3, 7, 1),
                deterministic_rollout_seed(42, 4, 7, 1),
                deterministic_rollout_seed(42, 3, 8, 1),
                deterministic_rollout_seed(42, 3, 7, 2),
                deterministic_rollout_seed(42, 3, 7, 1, no_signal_attempt_ordinal=1),
            }
        )
        == 6
    )
    assert 0 <= baseline < 2**63


def test_train_meta_records_the_optimizer_steps_that_actually_produced_a_loss():
    import inspect

    from flash.engine.worker.opd_train import run_opd_train

    notes = inspect.getsource(run_opd_train)
    notes = notes[notes.index("_w.write_train_meta(") :]
    # `steps` is the REQUESTED horizon. a step whose batch carried no teacher signal never applies
    # an update, so reporting the horizon as the work done would overstate a partly-starved run.
    assert '"steps": update_horizon,' in notes
    assert '"opt_steps": len(final_accounting["loss_curve"]),' in notes


def test_worker_filters_over_budget_prompts_before_downloading_the_weights():
    """An all-over-budget dataset must fail before the multi-GB prefetch, not after it.

    The prompt budget comes from `max_context_tokens - max_completion` and the prompt ids come from
    the tokenizer, so "every prompt is over budget" is decidable from files that weigh kilobytes.
    Running `prefetch_model` first spends tens of GB of download and minutes of paid worker time to
    reach a verdict the tokenizer already had. The deleted trl trainer filtered first; this pins the
    verl path to the same order.

    `generation_eos_from_cached_config` reads the snapshot with local_files_only, so it is allowed
    to sit after the prefetch -- but the budget check must not.
    """
    import inspect

    from flash.engine.worker.opd_train import run_opd_train

    source = inspect.getsource(run_opd_train)
    budget_raise = source.index('raise RuntimeError("every OPD prompt exceeds the configured')
    prefetch = source.index("_w.prefetch_model(")
    assert budget_raise < prefetch
    # and the eos read, which needs the downloaded snapshot, must follow the prefetch.
    assert prefetch < source.index("generation_eos_from_cached_config(")


def test_worker_refuses_to_publish_a_loss_curve_shorter_than_the_final_checkpoint():
    import inspect

    from flash.engine.worker.opd_train import run_opd_train

    source = inspect.getsource(run_opd_train)
    # record_step only checks that each metric line FOLLOWS the previous one, so it cannot notice a
    # MISSING TRAILING metric: on_line silently skips any step-tagged line whose loss it cannot
    # parse, and no later step ever arrives to trip the sequence check. that leaves a curve of
    # length N-1 for a run verl actually checkpointed at step N, and opt_steps is published straight
    # from that curve. the guard must compare against final_step, not merely require non-emptiness.
    assert 'if len(final_accounting["loss_curve"]) != final_step:' in source
    guard = source[source.index('if len(final_accounting["loss_curve"]) != final_step:') :]
    assert "raise RuntimeError(" in guard[:600]
    # and it must sit BEFORE the publish, not after it.
    assert source.index('if len(final_accounting["loss_curve"]) != final_step:') < source.index(
        "_w.write_train_meta("
    )


def test_train_meta_reports_the_teacher_call_shape_only_where_one_is_enforced():
    import inspect

    from flash.engine.worker.opd_train import run_opd_train

    notes = inspect.getsource(run_opd_train)
    notes = notes[notes.index("_w.write_train_meta(") :]
    # only the single-turn text path runs through the batcher, and it holds one serial scoring
    # thread. multimodal scores one item per call and multi-turn batches an episode, both on the
    # bridge's own request threads -- neither has a constant to report.
    # the reported cap is bounded by the samples one step can actually produce, matching trl's
    # _opd_teacher_batch_size: a 1-rollout step can never fill a batch of 8, so publishing the
    # global cap there would describe a shape the run is structurally unable to reach.
    assert (
        '"opd_teacher_batch_size": (\n'
        "                    min(\n"
        "                        OPD_TEACHER_SCORING_CONCURRENCY, max(1, prompts_per_step * knobs.group_size)\n"
        "                    )\n"
        "                    if not multimodal and not multi_turn\n"
        "                    else None\n"
        "                ),"
    ) in notes
    assert ('"opd_teacher_workers": 1 if not multimodal and not multi_turn else None,') in notes
    # the engine length handed to vllm, not a hardcoded default: prompt filtering is carved out of
    # this same number, so a fabricated one would disagree with the budget the run actually used.
    assert '"vllm_max_model_len": max_model_len,' in notes


def test_text_teacher_batcher_scores_one_batch_at_a_time():
    # opd_teacher_workers reports 1 for the text path. that is only honest while the batcher scores
    # serially, so assert the observable property rather than the thread count: no two score_many
    # calls may ever overlap, however the batcher is wired internally.
    from flash.engine.worker.opd_train import _TextTeacherBatcher

    class _OverlapDetectingTeacher:
        def __init__(self):
            self.lock = threading.Lock()
            self.in_flight = 0
            self.max_in_flight = 0

        def score_many(self, items):
            with self.lock:
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            # hold the call open long enough that a second scoring thread would land inside it.
            time.sleep(0.05)
            with self.lock:
                self.in_flight -= 1
            return [
                _teacher_score(
                    [TeacherToken(text=completion, logprob=-1.0, start=0, end=len(completion))]
                )
                for _prompt, completion in items
            ]

    teacher = _OverlapDetectingTeacher()
    batcher = _TextTeacherBatcher(teacher, max_batch_size=1, flush_wait_s=0.01)
    batcher.start()
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(batcher.score, f"prompt-{index}", f"completion-{index}")
                for index in range(8)
            ]
            for future in futures:
                future.result(timeout=10.0)
    finally:
        batcher.close()

    assert teacher.max_in_flight == 1


def test_worker_structured_validator_runs_before_model_download():
    import inspect

    from flash.engine.worker.opd_train import run_opd_train

    source = inspect.getsource(run_opd_train)
    assert source.index("validate_opd_structured_outputs(") < source.index("_w.prefetch_model(")
    assert "resolve_vocab_size" not in source


def test_plugin_registers_external_trainer_without_teacher_gpu_pool():
    import inspect

    import flash.engine.worker.opd_plugin as plugin

    source = inspect.getsource(plugin)
    assert "@register_distillation_loss(" in source
    assert 'names=["flash_groupwise_reverse_kl"]' in source
    assert "main_ppo_sync.TaskRunner = FlashTaskRunner" in source
    assert "resource_pool_spec = {" in source
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

    import flash.engine.worker.opd_multiturn as multiturn
    import flash.engine.worker.opd_plugin as plugin
    import flash.engine.worker.opd_structured as structured

    source = (
        inspect.getsource(plugin) + inspect.getsource(structured) + inspect.getsource(multiturn)
    ).lower()
    forbidden = ("parasail", "fireworks")
    for name in forbidden:
        assert f"class {name}" not in source
        assert f"def {name}" not in source
        assert f"_{name}" not in source


def test_opd_delegates_to_verl_with_no_selector_left(monkeypatch):
    """run_opd has no backend selector: it calls verl unconditionally.

    Replaces the FLASH_OPD_BACKEND selector test. The TRL OPD body is gone, so there is nothing
    left to select between and no env key can route the phase anywhere else.
    """
    import flash.engine.worker.opd as opd_mod
    import flash.engine.worker.opd_train as ov

    called = []
    monkeypatch.setattr(ov, "run_opd_train", lambda *a, **k: called.append(True))
    monkeypatch.setenv("FLASH_OPD_BACKEND", "bogus")
    opd_mod.run_opd()
    assert called == [True]


def test_opd_spec_never_resolves_the_allocator_conf_that_kills_vllm(monkeypatch):
    """An OPD spec must build a NON-expandable allocator conf.

    verl leaves rollout.enable_sleep_mode defaulted True, so the engine always builds a
    CuMemAllocator, which asserts outright on expandable_segments (cumem.py, pytorch#147851) -- the
    run would die before step 1.

    A [worker_env] selector must not be able to reintroduce it either: run_opd delegates to verl
    unconditionally, so honoring a stale key would pick an allocator for a backend that cannot run.
    """
    from flash.providers._worker import build_worker_env
    from flash.spec import JobSpec

    def _spec(worker_env=None):
        payload = {
            "run_id": "r-alloc",
            "algorithm": "opd",
            "model": "Qwen/Qwen3.5-4B",
            "model_policy": "catalog",
            "environment": {"repo": "x/y", "name": "e"},
            "train": {"hf_repo": "a/b", "teacher_model": "", "max_examples": 1},
            "gpu": {"type": "B200", "count": 1, "provider": "runpod"},
        }
        if worker_env is not None:
            payload["worker_env"] = worker_env
        return JobSpec.from_dict(payload)

    teacher_runtime = {
        "FLASH_CONTROL_PANEL_URL": "https://broker.example",
        "FLASH_TEACHER_CAPABILITY": "capability-test-value",
    }
    for worker_env in (None, {"FLASH_OPD_BACKEND": "trl"}, {"FLASH_OPD_BACKEND": "bogus"}):
        spec = _spec(worker_env)
        assert spec.phase == "opd"
        alloc = build_worker_env(
            spec,
            spec.seed,
            runtime_secrets=teacher_runtime,
        )["PYTORCH_CUDA_ALLOC_CONF"]
        assert "expandable_segments" not in alloc, (worker_env, alloc)

    # sft is verl-only too, but it builds no rollout engine at all, so expandable segments stay.
    sft = JobSpec.from_dict(
        {
            "run_id": "r-sft",
            "algorithm": "sft",
            "model": "Qwen/Qwen3.5-4B",
            "model_policy": "catalog",
            "environment": {"repo": "x/y", "name": "e"},
            "train": {"hf_repo": "a/b", "teacher_model": "", "max_examples": 1},
            "gpu": {"type": "B200", "count": 1, "provider": "runpod"},
        }
    )
    assert build_worker_env(sft, sft.seed)["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_worker_fails_closed_on_tool_env(monkeypatch):
    # multi-turn is now supported; native tool-calling OPD still fails closed at the worker layer.
    import flash.engine.worker.opd_train as ov

    class FakeEnv:
        is_tool_env = True
        multi_turn = False

    monkeypatch.setattr(ov._w, "require_active_env", lambda: FakeEnv())
    with pytest.raises(
        RuntimeError, match="native tool-calling OPD environments are not supported"
    ):
        ov.run_opd_train(spec=object())


def test_on_line_parses_the_numpy2_distillation_loss_the_image_actually_prints():
    """the distillation loss must survive numpy 2's np.float64(...) repr.

    verl aggregates actor/distillation/loss with Metric(SUM) -> np.sum
    (verl/utils/metric/utils.py), and LocalLogger renders it through pprint. verl pins
    numpy<2, but Dockerfile.worker installs numpy 2.2.6 into the /opt/verl-venv interpreter
    flash actually launches, where that scalar prints as "np.float64(0.64)". a bare float()
    raises on that spelling, so every step is skipped, loss_curve ends empty, and the publish
    guard rejects a run that trained correctly.
    """
    import flash.engine.worker.opd_train as ov

    ray_prefixed = "(TaskRunner pid=3125) step:4 - actor/distillation/loss:np.float64(0.6421)"
    assert ov.parse_verl_metric(ray_prefixed, "actor/distillation/loss") == 0.6421
    # the unprefixed numpy-1 spelling verl's own pin produces still parses.
    assert (
        ov.parse_verl_metric("step:4 - actor/distillation/loss:0.6421", "actor/distillation/loss")
        == 0.6421
    )
    # the un-namespaced fallback key the handler tries second.
    assert (
        ov.parse_verl_metric("step:4 - distillation/loss:np.float32(0.25)", "distillation/loss")
        == 0.25
    )


def test_opd_line_handler_reads_the_loss_through_the_shared_parser():
    """pin the call site, not just the helper: a local float() here would reintroduce the drop."""
    import ast
    import inspect
    import textwrap

    import flash.engine.worker.opd_train as ov

    source = textwrap.dedent(inspect.getsource(ov.run_opd_train))
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
    assert "parse_verl_metric" in calls
    assert "_metric_value" not in calls


def test_transfer_queue_init_passes_the_config_through_and_returns():
    seen = []
    _init_transfer_queue(lambda conf: seen.append(conf), {"backend": "SimpleStorage"}, timeout_s=30)
    assert seen == [{"backend": "SimpleStorage"}]


def test_transfer_queue_init_that_never_returns_fails_with_the_resource_state(monkeypatch):
    # the wedge this bounds: tq.init reserves its controller and storage units through ray and waits
    # with no timeout, so an unsatisfiable reservation stops the run before a single gpu is touched --
    # silently, because tq's logger defaults to WARNING. without a deadline the run burns the whole
    # setup grace period at 0% gpu and is reported as a generic stall.
    import flash.engine.worker.opd_plugin as plugin

    monkeypatch.setattr(
        plugin,
        "_describe_ray_resources",
        lambda: "cluster CPU=1.0 GPU=1.0, free CPU=0.0 GPU=0.0",
    )
    release = threading.Event()
    with pytest.raises(RuntimeError, match="transfer_queue init did not finish") as excinfo:
        _init_transfer_queue(lambda conf: release.wait(60), None, timeout_s=0.2)
    # the message must carry the resource state: "it timed out" alone does not distinguish an
    # unschedulable reservation from a slow start, which is the whole diagnosis.
    assert "free CPU=0.0" in str(excinfo.value)
    # and it must name the frame the thread is parked in. resource counts separate the placement-group
    # wait from the other two, but the controller ray.get and the get_config spin look identical from
    # outside -- only the stack tells them apart. here the init is parked in Event.wait.
    assert "stalled at wait (threading.py:" in str(excinfo.value)
    # the message must NOT claim the cause is capacity: two of the three waits stall with ample free
    # resources, and asserting a scheduling failure would send an operator to the wrong remediation.
    assert "only the first is a capacity problem" in str(excinfo.value)
    assert "so this is a scheduling problem" not in str(excinfo.value)
    release.set()


def test_transfer_queue_init_failure_propagates_unwrapped():
    # a real tq.init error must not be reported as a timeout, and must not be swallowed by the worker
    # thread it now runs on.
    class Boom(RuntimeError):
        pass

    def _raise(conf):
        raise Boom("controller refused the sampler")

    with pytest.raises(Boom, match="controller refused the sampler"):
        _init_transfer_queue(_raise, None, timeout_s=30)


def test_ray_resource_probe_reports_the_cluster_and_free_counts(monkeypatch):
    # ray is only present inside the verl child image, so the probe is exercised against a stub. the
    # message must name both totals and free counts: a wedged reservation is diagnosed by the gap.
    import flash.engine.worker.opd_plugin as plugin

    stub = types.ModuleType("ray")
    stub.cluster_resources = lambda: {"CPU": 8.0, "GPU": 1.0}
    stub.available_resources = lambda: {"CPU": 2.0, "GPU": 1.0}
    monkeypatch.setitem(sys.modules, "ray", stub)
    described = plugin._describe_ray_resources()
    assert "cluster CPU=8.0 GPU=1.0" in described
    assert "free CPU=2.0 GPU=1.0" in described


def test_ray_resource_probe_reports_a_fully_consumed_resource_as_zero(monkeypatch):
    # ray DROPS a resource key from available_resources() once it is fully allocated rather than
    # reporting 0.0 -- verified against ray 2.56.1: consuming every cpu removes "CPU" from the mapping.
    # that is the exhaustion case this probe exists to name, so rendering it as "free CPU=None" would
    # read as a broken probe at the exact moment the diagnosis matters. an earlier version of this test
    # hardcoded zero-valued keys in the stub and so could not have caught it.
    import flash.engine.worker.opd_plugin as plugin

    stub = types.ModuleType("ray")
    stub.cluster_resources = lambda: {"CPU": 8.0}
    stub.available_resources = lambda: {"memory": 1.0}
    monkeypatch.setitem(sys.modules, "ray", stub)
    described = plugin._describe_ray_resources()
    assert "free CPU=0.0" in described
    # a node with no gpu omits the key from cluster_resources() too, and 0.0 describes that correctly.
    assert "cluster CPU=8.0 GPU=0.0" in described
    assert "None" not in described


def test_ray_resource_probe_never_raises_on_the_timeout_path(monkeypatch):
    # _describe_ray_resources only runs while building a failure message, so a probe error there would
    # replace the diagnosis it exists to produce.
    import flash.engine.worker.opd_plugin as plugin

    stub = types.ModuleType("ray")
    stub.cluster_resources = lambda: (_ for _ in ()).throw(RuntimeError("ray is gone"))
    stub.available_resources = dict
    monkeypatch.setitem(sys.modules, "ray", stub)
    assert "ray resources unreadable" in plugin._describe_ray_resources()


def test_stalled_thread_probe_names_the_frame_the_thread_is_parked_in():
    # the point of the probe: identify WHICH of tq.init's three unbounded waits is stuck. a placement
    # group that never fills, a ray.get on the controller, and the `while conf is None` get_config spin
    # are indistinguishable from ray's resource counts, so the parked frame is the only discriminator.
    import flash.engine.worker.opd_plugin as plugin

    entered = threading.Event()
    release = threading.Event()

    def _park_in_a_named_frame():
        entered.set()
        release.wait(60)

    thread = threading.Thread(target=_park_in_a_named_frame, daemon=True)
    thread.start()
    try:
        assert entered.wait(10)
        described = plugin._describe_stalled_thread(thread.ident)
    finally:
        release.set()
        thread.join(10)
    # innermost frame first, so the wait itself leads and its caller follows -- that ordering is what
    # makes the message readable in a log line.
    assert described.startswith("wait (threading.py:")
    assert "_park_in_a_named_frame" in described


def test_stalled_thread_probe_reports_a_thread_that_is_gone_instead_of_raising():
    # a thread can finish between the is_alive check and the probe. that must degrade to a note in the
    # failure message, not an exception that replaces the timeout diagnosis.
    import flash.engine.worker.opd_plugin as plugin

    thread = threading.Thread(target=lambda: None, daemon=True)
    thread.start()
    thread.join(10)
    assert plugin._describe_stalled_thread(thread.ident) == "stack unavailable"
    assert plugin._describe_stalled_thread(None) == "stack unavailable"


def test_opd_missing_managed_teacher_broker_fails_before_the_gpu_probe(monkeypatch):
    """A missing managed teacher key must fail before any paid GPU work starts.

    Ported from the TRL OPD suite (`test_opd_missing_teacher_key_raises_platform_managed_error`),
    where the same guard was asserted against `run_opd`. It is a COST guard, not just a validation
    one: the key is platform-injected, so its absence is a platform-side failure that can only ever
    end the run -- reaching `_probe_gpu_in_subprocess` would rent a card for a run guaranteed to
    fail, and `prefetch_model` would then spend minutes pulling weights before anything noticed.
    The monkeypatched probe asserts rather than returns, so ordering is proved, not assumed.
    """
    from flash.engine.worker import opd_train as opd_mod

    env = SimpleNamespace(
        is_tool_env=False,
        multi_turn=False,
        dataset=lambda: [{"question": "2+2"}],
        prompt_messages=lambda _record: [{"role": "user", "content": "2+2"}],
    )
    train = SimpleNamespace(
        init_from_adapter="",
        max_examples=1,
        teacher_model="",
        epochs=1,
        temperature=None,
        save_at_steps=(),
        stop_sequences=(),
        structured_outputs="",
    )
    monkeypatch.setattr(
        opd_mod,
        "_w",
        SimpleNamespace(
            SEED=0,
            THINKING=False,
            require_active_env=lambda: env,
            JOB_SPEC=SimpleNamespace(
                train=train,
                model="Qwen/Qwen3.5-4B",
                model_revision="",
                model_policy="catalog",
                gpu=SimpleNamespace(type=None),
            ),
            heartbeat=lambda *args, **kwargs: None,
            gpu_diagnostics=lambda **_kwargs: {},
            prefetch_model=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("model prefetch must not be reached")
            ),
        ),
    )
    monkeypatch.setattr(
        opd_mod,
        "_probe_gpu_in_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("gpu allocation must not be reached")
        ),
    )
    # torch is not installed in this test env; the real seeding is covered in test_training_controls.
    monkeypatch.setattr(opd_mod, "seed_training_rngs", lambda seed: None)
    monkeypatch.delenv("FLASH_CONTROL_PANEL_URL", raising=False)
    monkeypatch.delenv("FLASH_TEACHER_CAPABILITY", raising=False)

    with pytest.raises(RuntimeError, match="managed teacher control-panel transport is missing"):
        opd_mod.run_opd_train()


def test_opd_renders_each_prompt_once_so_a_stateful_environment_is_not_run_twice():
    """`prompt_messages` is user code, and it was called once to classify and again to build.

    Two consequences, and the second costs a paid run. The environment observes every prompt twice,
    so a stateful or seeded implementation advances its state on a pass whose output is discarded.
    And the two passes can DISAGREE: if the classifying pass is text-only for a record whose second
    rendering carries an image, `multimodal` is already latched false, no processor was built, and
    the build pass reaches a bare `assert processor is not None` (codex[bot]).

    Asserted on the source rather than by driving `run_opd_train`, because the build loop sits
    behind a gpu probe, a teacher client and a tokenizer load: a fixture that stubs its way there
    stops testing this and starts testing the stubs. What makes the mismatch UNCONSTRUCTIBLE is
    structural -- one call site, and the shuffle applied to the cached rows so the build pass has no
    reason to re-render -- and that is what this reads.
    """
    import ast
    import textwrap

    from flash.engine.worker import opd_train as opd_mod

    tree = ast.parse(textwrap.dedent(inspect.getsource(opd_mod.run_opd_train)))
    renders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "prompt_messages"
    ]
    assert len(renders) == 1, (
        f"run_opd_train calls env.prompt_messages() {len(renders)} times; each extra call re-runs "
        "user environment code on every example, and a second rendering that differs from the "
        "first reaches the build loop with multimodal already latched from the first"
    )

    shuffles = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "shuffle"
    ]
    assert len(shuffles) == 1, f"expected exactly one dataset shuffle, found {len(shuffles)}"
    # shuffling the raw examples is what forces the second rendering: the cached rows would no
    # longer line up with the training order, so the build pass has to ask the environment again.
    assert [a.id for a in shuffles[0].args if isinstance(a, ast.Name)] == ["prompt_rows"], (
        "the shuffle must reorder the already-rendered rows, not the raw examples"
    )


def test_the_opd_trainer_stores_the_frozen_base_in_bf16():
    """VERL-150: verl's fsdp.yaml default is fp32, which doubles the trainer's resident base.

    an un-overridden run loads the base at 4 bytes/param, because ``model_dtype: fp32``
    (``trainer/config/engine/fsdp.yaml:33``) is passed straight to ``from_pretrained`` and the
    ``if torch_dtype is None`` fallback below it never fires. at 27B that is 103 GB instead of 52,
    which left vLLM 71.9 GB against an 89.2 GB demand and killed g5's OPD leg at engine startup --
    reported 4 frames above the real error, as a bare ``ValueError`` from a child worker process.

    the sft driver has set an explicit dtype since it was written and the rl/opd drivers did not,
    which is exactly why g5's sft leg succeeded and its grpo and opd legs failed on the same model
    and the same card. the rl driver's half of this lives in test_rl_train.
    """
    overrides = build_opd_overrides(_config())
    want = "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
    # exact, not substring: "x=bfloat16" is a substring of "+x=bfloat16", so the obvious `in`
    # assertion passes against a spelling hydra REJECTS here ("Could not append to config. An item
    # is already at ..."). the key is declared in the yaml and takes a BARE override -- the inverse
    # of enable_sleep_mode above, which is dataclass-only and requires `+`. neighbouring keys,
    # opposite prefixes. see ISSUES.md VERL-150.
    assert want in overrides
    assert f"+{want}" not in overrides, "must not be + prefixed"
    # ref is deliberately NOT set. it reads like a second resident copy and is not one:
    # ray_trainer.py:897 aliases ref_policy_wg to the actor worker whenever ref_in_actor holds, and
    # flash parses lora_rank with minimum=1 (schema/__init__.py:485) so it always holds. setting it
    # would free nothing. this asserts the absence so a future reader has to re-derive the above
    # rather than pattern-match it back in.
    assert not [o for o in overrides if "ref.fsdp_config.model_dtype" in o]
