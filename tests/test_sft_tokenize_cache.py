"""Focused coverage for parallel cached SFT data preparation."""

from __future__ import annotations

import os
import pickle

import pytest

import flash.engine.worker.sft as sft_mod
from flash.engine.worker.sft import (
    _normalize_sft_records,
    _prepare_sft_examples,
    _pretokenize_completion_only,
    _sft_prep_fingerprint,
    _tokenizer_identity,
)


class _TestTokenizer:
    name_or_path = "test/tokenizer"
    chat_template = "test-template-v1"
    eos_token = "<eos>"
    truncation_side = "right"
    add_bos_token = False
    add_eos_token = False

    def __init__(self, *, fail=False, require_rayon_disabled=False):
        self.all_special_ids = [0]
        self.special_tokens_map = {"eos_token": "<eos>"}
        self.fail = fail
        self.require_rayon_disabled = require_rayon_disabled

    def get_vocab(self):
        return {"<eos>": 0, "text": 1}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is False
        if self.require_rayon_disabled:
            assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
        if self.fail:
            raise RuntimeError("intentional render failure")
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
        )
        if enable_thinking:
            rendered = f"<think>{rendered}"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, texts, *, truncation, max_length):
        assert truncation is True
        return {
            "input_ids": [
                [1 + (ord(character) % 251) for character in text][:max_length] for text in texts
            ]
        }


class _ParentOnlyEnv:
    multi_turn = False

    def __init__(self):
        self.parent_pid = os.getpid()

    def prompt_messages(self, example):
        assert os.getpid() == self.parent_pid
        return [{"role": "user", "content": example["input"]}]

    def sft_completion(self, example):
        assert os.getpid() == self.parent_pid
        return example["completion"]


def _serial_rows(env, examples, tokenizer, *, thinking, max_length):
    texts = []
    for example in examples:
        prompt = env.prompt_messages(example)
        completion = env.sft_completion(example)
        texts.append(
            {
                "text": tokenizer.apply_chat_template(
                    [*prompt, *completion],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=thinking,
                ),
                "prompt_text": tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                ),
            }
        )
    return _pretokenize_completion_only(texts, tokenizer, max_length)


def test_parallel_map_matches_serial_input_ids_and_reuses_complete_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_mod, "_sft_tokenize_num_proc", lambda _row_count: 2)
    env = _ParentOnlyEnv()
    tokenizer = _TestTokenizer()
    examples = [
        {
            "input": "alpha",
            "completion": [{"role": "assistant", "content": "one"}],
        },
        {
            "input": "beta",
            "completion": [
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "follow-up"},
                {"role": "assistant", "content": "three"},
            ],
        },
        {
            "input": "gamma",
            "completion": [{"role": "assistant", "content": "four"}],
        },
    ]
    thinking = True
    max_length = 512
    _, expected, expected_dropped = _serial_rows(
        env, examples, tokenizer, thinking=thinking, max_length=max_length
    )
    tokenizer.require_rayon_disabled = True
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")

    _, actual, dropped, multiturn, cache_hit = _prepare_sft_examples(
        env,
        examples,
        tokenizer,
        env_resolved_sha="a" * 40,
        seed=7,
        model_revision="b" * 40,
        thinking=thinking,
        max_length=max_length,
        cache_root=tmp_path,
    )
    assert [row["input_ids"] for row in actual] == [row["input_ids"] for row in expected]
    assert [row["completion_mask"] for row in actual] == [
        row["completion_mask"] for row in expected
    ]
    assert dropped == expected_dropped
    assert multiturn == 1
    assert cache_hit is False
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"
    assert not list(tmp_path.glob("*/map*.arrow"))

    _, cached, cached_dropped, cached_multiturn, cache_hit = _prepare_sft_examples(
        env,
        examples,
        tokenizer,
        env_resolved_sha="a" * 40,
        seed=7,
        model_revision="b" * 40,
        thinking=thinking,
        max_length=max_length,
        cache_root=tmp_path,
    )
    assert cached == actual
    assert cached_dropped == dropped
    assert cached_multiturn == multiturn
    assert cache_hit is True


def test_single_process_map_preserves_parent_tokenizer_parallelism(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_mod, "_sft_tokenize_num_proc", lambda _row_count: None)
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    env = _ParentOnlyEnv()
    examples = [
        {
            "input": "alpha",
            "completion": [{"role": "assistant", "content": "one"}],
        }
    ]

    _prepare_sft_examples(
        env,
        examples,
        _TestTokenizer(),
        env_resolved_sha="a" * 40,
        seed=7,
        model_revision="b" * 40,
        thinking=False,
        max_length=256,
        cache_root=tmp_path,
    )

    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"


def test_interrupted_map_never_publishes_a_partial_cache(tmp_path):
    env = _ParentOnlyEnv()
    tokenizer = _TestTokenizer(fail=True)
    examples = [
        {
            "input": "alpha",
            "completion": [{"role": "assistant", "content": "one"}],
        },
        {
            "input": "beta",
            "completion": [{"role": "assistant", "content": "two"}],
        },
    ]
    kwargs = {
        "env_resolved_sha": "a" * 40,
        "seed": 7,
        "model_revision": "b" * 40,
        "thinking": False,
        "max_length": 256,
        "cache_root": tmp_path,
    }

    with pytest.raises(RuntimeError, match="intentional render failure"):
        _prepare_sft_examples(env, examples, tokenizer, **kwargs)
    assert not list(tmp_path.glob("*/_SUCCESS"))
    assert not list(tmp_path.glob(".*"))

    tokenizer.fail = False
    _, rows, dropped, _, cache_hit = _prepare_sft_examples(env, examples, tokenizer, **kwargs)
    assert rows
    assert dropped == 0
    assert cache_hit is False
    assert len(list(tmp_path.glob("*/_SUCCESS"))) == 1


def test_tokenizer_process_count_is_amortized_and_bounded(monkeypatch):
    monkeypatch.setattr(sft_mod.os, "cpu_count", lambda: 64)
    assert sft_mod._sft_tokenize_num_proc(1023) is None
    assert sft_mod._sft_tokenize_num_proc(1024) == 2
    assert sft_mod._sft_tokenize_num_proc(100_000) == 8

    monkeypatch.setattr(sft_mod.os, "cpu_count", lambda: 4)
    assert sft_mod._sft_tokenize_num_proc(100_000) == 2


def test_tokenizer_identity_covers_behavior_affecting_runtime_settings():
    baseline = _TestTokenizer()
    changed = _TestTokenizer()
    changed.truncation_side = "left"
    assert _tokenizer_identity(changed) != _tokenizer_identity(baseline)

    changed.truncation_side = "right"
    changed.add_eos_token = True
    assert _tokenizer_identity(changed) != _tokenizer_identity(baseline)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("env_resolved_sha", "c" * 40),
        ("dataset_prefix", "changed-prefix"),
        ("seed", 8),
        ("order", "changed-order"),
        ("model_revision", "d" * 40),
        ("tokenizer_identity", "changed-tokenizer"),
        ("chat_template", "changed-template"),
        ("thinking", False),
        ("max_length", 1024),
    ],
)
def test_cache_fingerprint_invalidates_on_every_required_field(field, changed):
    fields = {
        "env_resolved_sha": "a" * 40,
        "dataset_prefix": "prefix-digest",
        "seed": 7,
        "order": "order-digest",
        "model_revision": "b" * 40,
        "tokenizer_identity": "tokenizer-digest",
        "chat_template": "template-v1",
        "thinking": True,
        "max_length": 512,
    }
    baseline = _sft_prep_fingerprint(**fields)
    changed_fields = {**fields, field: changed}
    assert _sft_prep_fingerprint(**changed_fields) != baseline


def test_parent_normalized_records_are_picklable():
    env = _ParentOnlyEnv()
    examples = [
        {
            "input": "question",
            "completion": [{"role": "assistant", "content": "answer"}],
        }
    ]
    records, multiturn = _normalize_sft_records(env, examples)
    assert pickle.loads(pickle.dumps(records)) == records
    assert multiturn == 0
