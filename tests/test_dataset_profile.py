from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest

from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
from flash.engine.profiling.dataset_profile import (
    PackagedDatasetUnavailable,
    profile_packaged_sft_dataset,
)
from flash.engine.profiling.workload_profile import sft_profile_input_digest


class FakeTokenizer:
    eos_token = "|"
    eos_token_id = 2
    pad_token = None
    pad_token_id = 0
    all_special_ids: ClassVar[list[int]] = [0, 1, 2]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        **_kwargs,
    ):
        assert not tokenize
        text = "".join(str(message.get("content") or "") for message in messages)
        return text + (">" if add_generation_prompt else "")

    def __call__(self, texts, *, truncation=False, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[3 + ord(char) % 89 for char in text] for text in texts]
        if truncation:
            assert max_length is not None
            ids = [row[:max_length] for row in ids]
        return {"input_ids": ids}


def _spec(
    *,
    params: dict | None = None,
    environment_id: str = "team/example",
    model: str = "test/model",
) -> JobSpec:
    base = JobSpec(
        model=model,
        model_revision="a" * 40,
        algorithm="sft",
        environment=EnvironmentSpec(
            id=environment_id,
            resolved_sha="b" * 40,
            params=params or {},
        ),
        train=TrainSpec(epochs=2, batch_size=2, max_context_tokens=128),
        seed=7,
    )
    digest = sft_profile_input_digest(
        base,
        tokenizer_revision=base.model_revision,
        producer_version="1.2.3",
    )
    return replace(
        base,
        workload_profile_input_digest=digest,
        workload_profile_producer_version="1.2.3",
    )


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    entrypoint = tmp_path / "environment.py"
    entrypoint.write_text("raise RuntimeError('environment code executed')\n", encoding="utf-8")
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return entrypoint


def _profile(entrypoint: Path, *, params: dict | None = None):
    spec = _spec(params=params, environment_id=str(entrypoint))
    profile = profile_packaged_sft_dataset(
        spec,
        producer_version="1.2.3",
        tokenizer_loader=lambda _model, _revision: FakeTokenizer(),
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )
    return spec, profile


def test_profile_reads_packaged_train_jsonl_without_executing_environment_code(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"one","output":"alpha"}\n{"input":"two","output":"beta"}\n'
            )
        },
    )

    spec, profile = _profile(entrypoint)

    assert profile.source_examples == 2
    assert profile.selected_examples == 2
    assert profile.retained_examples == 2
    assert profile.authoritative_steps == 2
    assert profile.input_digest == spec.workload_profile_input_digest


def test_profile_rejects_the_plural_dataset_layout_that_the_worker_never_reads(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {"datasets/train.jsonl": '{"input":"one","output":"alpha"}\n'},
    )

    with pytest.raises(PackagedDatasetUnavailable, match=r"dataset/train\.jsonl"):
        _profile(entrypoint)


def test_profile_prefers_canonical_dataset_over_plural_layout(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": '{"input":"canonical","output":"yes"}\n',
            "datasets/train.jsonl": (
                '{"input":"plural-one","output":"no"}\n{"input":"plural-two","output":"no"}\n'
            ),
        },
    )

    _spec_value, profile = _profile(entrypoint)

    assert profile.source_examples == 1


def test_profile_honors_split_and_dataset_path(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": '{"input":"train","output":"no"}\n',
            "dataset/sft.jsonl": '{"input":"split","output":"yes"}\n',
            "custom.json": json.dumps(
                [
                    {"input": "custom-one", "output": "yes"},
                    {"input": "custom-two", "output": "yes"},
                ]
            ),
        },
    )

    _split_spec, split_profile = _profile(entrypoint, params={"split": "sft"})
    _path_spec, path_profile = _profile(
        entrypoint, params={"dataset_path": "custom.json", "split": "sft"}
    )

    assert split_profile.source_examples == 1
    assert path_profile.source_examples == 2


def test_profile_refuses_missing_or_unreadable_dataset(tmp_path) -> None:
    entrypoint = _package(tmp_path, {})
    with pytest.raises(PackagedDatasetUnavailable, match=r"dataset/train\.jsonl"):
        _profile(entrypoint)

    unreadable = _package(tmp_path / "bad", {"dataset/train.jsonl": "not json\n"})
    with pytest.raises(PackagedDatasetUnavailable, match="not readable JSON"):
        _profile(unreadable)


def test_profile_includes_the_default_training_contract(tmp_path) -> None:
    with_contract = _package(
        tmp_path / "with-contract",
        {
            "dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n',
            "TRAINING_CONTRACT.md": "follow this fixed training contract",
        },
    )
    without_contract = _package(
        tmp_path / "without-contract",
        {"dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n'},
    )

    _spec_value, contracted = _profile(with_contract)
    _spec_value, raw = _profile(without_contract)

    assert contracted.real_tokens_per_epoch > raw.real_tokens_per_epoch
    assert contracted.supervised_tokens_per_epoch == raw.supervised_tokens_per_epoch


def test_profile_honors_training_contract_precedence(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n',
            "TRAINING_CONTRACT.md": "default contract is intentionally longest",
            "custom-contract.txt": "custom contract",
        },
    )

    _spec_value, default = _profile(entrypoint)
    _spec_value, custom_path = _profile(entrypoint, params={"contract_path": "custom-contract.txt"})
    _spec_value, authored = _profile(
        entrypoint,
        params={
            "contract_text": "x",
            "contract_path": "custom-contract.txt",
        },
    )

    assert default.real_tokens_per_epoch > custom_path.real_tokens_per_epoch
    assert custom_path.real_tokens_per_epoch > authored.real_tokens_per_epoch


def test_training_contract_participates_in_retention_and_truncation(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n',
            "TRAINING_CONTRACT.md": "c" * 256,
        },
    )

    with pytest.raises(ValueError, match="empty completion after"):
        _profile(entrypoint)


def test_profile_does_not_override_an_existing_nonblank_system_message(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":[{"role":"system","content":"record system"},'
                '{"role":"user","content":"prompt"}],"output":"answer"}\n'
            ),
            "TRAINING_CONTRACT.md": "package contract that must not replace the record system",
        },
    )

    _spec_value, contracted = _profile(entrypoint)
    _spec_value, same_record = _profile(entrypoint, params={"contract_text": "other contract"})

    assert contracted.real_tokens_per_epoch == same_record.real_tokens_per_epoch


def test_profile_rejects_dataset_path_outside_the_package(tmp_path) -> None:
    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"input":"outside","output":"no"}\n', encoding="utf-8")
    entrypoint = _package(tmp_path / "package", {})

    with pytest.raises(PackagedDatasetUnavailable, match="readable packaged dataset"):
        _profile(entrypoint, params={"dataset_path": str(outside)})


def test_profile_rejects_contract_path_outside_the_package_before_reading_it(tmp_path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("sensitive host content", encoding="utf-8")
    entrypoint = _package(
        tmp_path / "package",
        {"dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n'},
    )

    with pytest.raises(PackagedDatasetUnavailable, match=r"contract_path.*outside"):
        _profile(entrypoint, params={"contract_path": str(outside)})


def test_profile_tokenizes_raw_record_fields_and_message_shapes_only(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"short","output":"answer"}\n'
                '{"input":[{"role":"user","content":"message prompt"}],'
                '"output":{"messages":[{"role":"assistant","content":"message answer"}]}}\n'
            )
        },
    )

    _spec_value, profile = _profile(entrypoint)

    assert profile.source_examples == 2
    assert profile.real_tokens_per_epoch > profile.supervised_tokens_per_epoch > 0


def test_profile_rejects_mixed_message_lists(tmp_path) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":[{"role":"user","content":"prompt"},"invalid"],"output":"answer"}\n'
            )
        },
    )

    with pytest.raises(ValueError, match="non-object message entries"):
        _profile(entrypoint)


def test_control_plane_profile_refuses_image_rows_before_loading_a_processor(
    tmp_path, monkeypatch
) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"describe","output":"red","image":"dataset/red.png"}\n'
            )
        },
    )
    spec = _spec(environment_id=str(entrypoint), model="Qwen/Qwen3.5-4B")
    monkeypatch.setattr(
        "flash.engine.profiling.sft_workload._default_processor_loader",
        lambda *_args: (_ for _ in ()).throw(AssertionError("processor loader reached")),
    )

    with pytest.raises(PackagedDatasetUnavailable, match="torch-free control plane"):
        profile_packaged_sft_dataset(
            spec,
            producer_version="1.2.3",
            tokenizer_loader=lambda _model, _revision: FakeTokenizer(),
            packing_support=lambda _model, _revision: ("gdn-hybrid", True),
        )


def test_profile_streams_jsonl_and_tokenizes_only_the_deterministic_max_examples_prefix(
    tmp_path, monkeypatch
) -> None:
    from flash.engine.profiling import dataset_profile

    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": "".join(
                json.dumps({"input": f"prompt-{index}", "output": f"answer-{index}"}) + "\n"
                for index in range(100)
            )
        },
    )
    seen = {}
    real_prepare = dataset_profile.prepare_sft_workload

    def capture_rows(spec, env, **kwargs):
        seen["rows"] = list(env.rows)
        return real_prepare(spec, env, **kwargs)

    monkeypatch.setattr(dataset_profile, "prepare_sft_workload", capture_rows)
    spec = _spec(params={}, environment_id=str(entrypoint))
    spec = replace(spec, train=replace(spec.train, max_examples=3))

    profile = profile_packaged_sft_dataset(
        spec,
        producer_version="1.2.3",
        tokenizer_loader=lambda _model, _revision: FakeTokenizer(),
        packing_support=lambda _model, _revision: ("pure-attention", True),
    )

    assert profile.source_examples == 100
    assert profile.selected_examples == 3
    assert {row["input"] for row in seen["rows"]} == {"prompt-0", "prompt-1", "prompt-2"}


def test_jsonl_profile_never_slurps_the_dataset_into_one_control_plane_string(
    tmp_path, monkeypatch
) -> None:
    entrypoint = _package(
        tmp_path,
        {
            "dataset/train.jsonl": (
                '{"input":"one","output":"alpha"}\n{"input":"two","output":"beta"}\n'
            )
        },
    )
    dataset_path = tmp_path / "dataset/train.jsonl"
    read_text = Path.read_text

    def refuse_dataset_slurp(path, *args, **kwargs):
        if path == dataset_path:
            raise AssertionError("jsonl dataset was slurped with Path.read_text")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse_dataset_slurp)

    _spec_value, profile = _profile(entrypoint)

    assert profile.source_examples == 2


def test_profile_refuses_a_dataset_file_above_the_control_plane_memory_bound(
    tmp_path, monkeypatch
) -> None:
    from flash.engine.profiling import dataset_profile

    entrypoint = _package(
        tmp_path,
        {"dataset/train.jsonl": '{"input":"prompt","output":"answer"}\n'},
    )
    monkeypatch.setattr(dataset_profile, "_MAX_PROFILE_DATASET_BYTES", 1)

    with pytest.raises(PackagedDatasetUnavailable, match="profiling limit"):
        _profile(entrypoint)
