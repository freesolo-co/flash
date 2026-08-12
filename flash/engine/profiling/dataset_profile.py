"""control-plane sft profiling without executing environment code.

raw ``input`` and ``output`` fields are augmented with the statically readable training contract,
using the same ``contract_text`` then ``contract_path`` then ``TRAINING_CONTRACT.md`` precedence as
the environment loader. the shared tokenizer computes tokens, retention, truncation, and steps from
that one stream so the estimate stays internally consistent. environment-owned prompt construction,
few-shot examples, tool schemas, filtering, and other transformations remain unavailable without
executing user code. an environment that expands prompts heavily may therefore retain fewer rows,
truncate more rows, run fewer effective steps, and cost more than this estimate predicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash.engine.profiling.sft_workload import prepare_sft_workload
from flash.engine.profiling.tokenizer import load_tokenizer
from flash.envs.dataset_selection import (
    _packaged_dataset_file,
    _plural_dataset_file,
    _validate_packaged_dataset_split,
)
from flash.envs.loader import _load_contract_text, _resolve_environment_reference, _resolve_path_arg

_CANONICAL_INPUT_KEY = "input"
_CANONICAL_OUTPUT_KEY = "output"


class PackagedDatasetUnavailable(ValueError):
    """the pinned environment package has no readable dataset file for sft profiling."""


@dataclass
class _RawRecordEnvironment:
    rows: list[dict[str, Any]]
    package_root: Path
    contract_text: str = ""
    multi_turn: bool = False

    def dataset(self) -> list[dict[str, Any]]:
        return list(self.rows)

    def prompt_messages(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        value = row.get(_CANONICAL_INPUT_KEY)
        if _is_message_list(value):
            messages = _copied_messages(value, field="input")
        elif isinstance(value, dict) and set(value) == {"messages"}:
            messages = _copied_messages(value["messages"], field="input")
        else:
            messages = [{"role": "user", "content": "" if value is None else str(value)}]
        return _with_system_prompt(messages, self.contract_text)

    def sft_completion_with_provenance(
        self, row: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], bool]:
        value = row.get(_CANONICAL_OUTPUT_KEY)
        if _is_message_list(value):
            return _copied_messages(value, field="output"), False
        if isinstance(value, dict) and "messages" in value:
            sibling_keys = sorted(str(key) for key in value if key != "messages")
            if sibling_keys:
                raise ValueError(
                    "sft output has 'messages' alongside sibling keys "
                    f"{sibling_keys}; expected exactly {{'messages': [...]}}"
                )
            return _copied_messages(value["messages"], field="output"), False
        return [{"role": "assistant", "content": "" if value is None else str(value)}], True


def _is_message_list(value: object) -> bool:
    return isinstance(value, list) and any(isinstance(message, dict) for message in value)


def _with_system_prompt(messages: list[dict[str, Any]], contract_text: str) -> list[dict[str, Any]]:
    system_text = str(contract_text or "").strip()
    out = [dict(message) for message in messages]
    if not system_text:
        return out
    first_blank_system_index: int | None = None
    for index, message in enumerate(out):
        if str(message.get("role") or "").strip().lower() != "system":
            continue
        content = message.get("content")
        has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
        if has_content:
            return out
        if first_blank_system_index is None:
            first_blank_system_index = index
    if first_blank_system_index is not None:
        out[first_blank_system_index]["content"] = system_text
        return out
    return [{"role": "system", "content": system_text}, *out]


def _copied_messages(value: object, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"sft {field} messages must be a list of message objects")
    invalid = [index for index, message in enumerate(value) if not isinstance(message, dict)]
    if invalid:
        raise ValueError(f"sft {field} contains non-object message entries at indexes {invalid}")
    return [dict(message) for message in value]


def _contract_text(base_dir: Path, params: dict[str, Any]) -> str:
    authored = params.get("contract_text")
    if authored:
        return str(authored)
    contract_path = _resolve_path_arg(params.get("contract_path"), base_dir)
    if not isinstance(contract_path, str):
        contract_path = str(base_dir / "TRAINING_CONTRACT.md")
    return str(_load_contract_text(contract_path))


def _selected_dataset_path(base_dir: Path, params: dict[str, Any]) -> Path:
    split = params.get("split")
    split = split.strip() if isinstance(split, str) else ""
    wanted = _validate_packaged_dataset_split(split) if split else "train"
    dataset_path = params.get("dataset_path")
    if dataset_path:
        resolved = _resolve_path_arg(dataset_path, base_dir)
        path = Path(resolved) if isinstance(resolved, str) else None
        try:
            packaged = path is not None and path.resolve().is_relative_to(base_dir.resolve())
        except (OSError, RuntimeError):
            packaged = False
        if packaged and path is not None and path.is_file():
            return path
        raise PackagedDatasetUnavailable(
            f"[environment.params] dataset_path {dataset_path!r} is not a readable packaged dataset "
            "file. Add dataset/train.jsonl to the environment package or point dataset_path at a "
            "packaged .jsonl or .json file."
        )
    canonical = _packaged_dataset_file(base_dir, wanted)
    if canonical is not None:
        return canonical
    plural = _plural_dataset_file(base_dir, wanted)
    if plural is not None:
        return plural
    raise PackagedDatasetUnavailable(
        f"environment package has no readable dataset for split {wanted!r}. Add "
        "dataset/train.jsonl to the environment package, or package the selected split as "
        f"dataset/{wanted}.jsonl. The plural datasets/{wanted}.jsonl layout is also accepted."
    )


def _read_dataset_rows(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".jsonl":
            values = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            values = loaded.get("records") if isinstance(loaded, dict) else loaded
        else:
            raise ValueError("dataset file must end in .jsonl or .json")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PackagedDatasetUnavailable(
            f"environment dataset file {path.name!r} is not readable JSON. Add a valid "
            "dataset/train.jsonl to the environment package."
        ) from exc
    if not isinstance(values, list) or not all(isinstance(row, dict) for row in values):
        raise PackagedDatasetUnavailable(
            f"environment dataset file {path.name!r} must contain JSON object rows. Add a valid "
            "dataset/train.jsonl to the environment package."
        )
    if not values:
        raise PackagedDatasetUnavailable(
            f"environment dataset file {path.name!r} contains no rows. Add training rows to "
            "dataset/train.jsonl in the environment package."
        )
    missing = [index for index, row in enumerate(values) if _CANONICAL_INPUT_KEY not in row]
    if missing:
        raise ValueError(
            "Freesolo dataset records must contain an input field; missing at row indexes "
            f"{missing[:10]}"
        )
    return [dict(row) for row in values]


def profile_packaged_sft_dataset(
    spec,
    *,
    producer_version: str,
    tokenizer_loader=load_tokenizer,
    packing_support=None,
):
    """build an sft quote profile from the pinned package's raw dataset records.

    this function resolves and reads package contents only. it never imports ``environment.py`` or
    calls the freesolo environment loader. token totals, retention, truncation, and steps estimate
    the raw records plus the statically readable training contract on one consistent token stream.
    """
    if not spec.environment.id:
        raise PackagedDatasetUnavailable("sft workload profiling requires an environment id")
    if not spec.environment.resolved_sha:
        raise PackagedDatasetUnavailable(
            "sft workload profiling requires a pinned environment package revision"
        )
    reference = Path(
        _resolve_environment_reference(spec.environment.id, spec.environment.resolved_sha)
    )
    if not reference.is_file():
        raise PackagedDatasetUnavailable(
            "the pinned environment package did not resolve to an environment.py entrypoint"
        )
    base_dir = reference.parent
    params = dict(spec.environment.params or {})
    dataset_path = _selected_dataset_path(base_dir, params)
    rows = _read_dataset_rows(dataset_path)
    return prepare_sft_workload(
        spec,
        _RawRecordEnvironment(
            rows=rows,
            package_root=base_dir,
            contract_text=_contract_text(base_dir, params),
        ),
        tokenizer_loader=tokenizer_loader,
        producer_version=producer_version,
        allow_packing=True,
        packing_support=packing_support,
    ).profile
