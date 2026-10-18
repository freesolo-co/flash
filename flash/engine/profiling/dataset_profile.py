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
from flash.engine.profiling.tokenizer import load_control_plane_tokenizer
from flash.engine.worker.entry.sft import select_sft_examples
from flash.engine.worker.model.packing import worker_image_packing_support
from flash.envs.loading.base import with_system_prompt
from flash.envs.loading.loader import (
    _load_contract_text,
    _resolve_environment_reference,
    _resolve_path_arg,
)
from flash.envs.meta.dataset_selection import (
    _packaged_dataset_file,
    _validate_packaged_dataset_split,
    select_dataset_source,
)

_CANONICAL_INPUT_KEY = "input"
_CANONICAL_OUTPUT_KEY = "output"
_MAX_PROFILE_DATASET_BYTES = 32 * 1024 * 1024
# the contract rides in every row, so it is bounded far tighter than the dataset that carries it.
_MAX_PROFILE_CONTRACT_BYTES = 256 * 1024


class PackagedDatasetUnavailable(ValueError):
    """the pinned environment package has no readable dataset file for sft profiling."""


@dataclass
class _RawRecordEnvironment:
    rows: list[dict[str, Any]]
    package_root: Path
    contract_text: str = ""

    def dataset(self) -> list[dict[str, Any]]:
        return list(self.rows)

    def prompt_messages(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        # the worker never reads the raw input itself: it builds a TaskExample, whose .input the sdk
        # has already rendered through serialize_value, and the scaffolded env puts exactly that
        # string in one user turn (flash/envs/loading/adapter.py:_task_example). so call the sdk here too
        # rather than restating its rules. str() would tokenize a python repr, and reading a
        # message-shaped input as a transcript would quote a prompt the sdk never produces.
        from freesolo.datasets.records import task_example_from_record

        content = str(task_example_from_record(row).input)
        return with_system_prompt([{"role": "user", "content": content}], self.contract_text)

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
        # the adapter coerces a scalar output with str(), not serialize_value: unlike input, the
        # gold completion never passes through the sdk's record normalization.
        return [{"role": "assistant", "content": "" if value is None else str(value)}], True


def _is_message_list(value: object) -> bool:
    return isinstance(value, list) and any(isinstance(message, dict) for message in value)


def _copied_messages(value: object, *, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"sft {field} messages must be a list of message objects")
    invalid = [index for index, message in enumerate(value) if not isinstance(message, dict)]
    if invalid:
        raise ValueError(f"sft {field} contains non-object message entries at indexes {invalid}")
    return [dict(message) for message in value]


def _contract_text(base_dir: Path, params: dict[str, Any]) -> str:
    # the contract is rendered into the system turn of EVERY selected row, so its size is
    # multiplied across the dataset before tokenization. the package allows far more than the
    # plane should hold, and unlike the dataset file this text has no other bound.
    authored = params.get("contract_text")
    if authored:
        return _bounded_contract_text(str(authored), source="[environment.params] contract_text")
    configured_path = params.get("contract_path")
    if not isinstance(configured_path, str):
        # the loader defaults only when the key is absent or non-string (loader.py:838-841). an
        # empty string is a string, so it means "no contract" there and must mean that here too.
        path = base_dir / "TRAINING_CONTRACT.md"
    elif not configured_path:
        return ""
    else:
        # NOT _resolve_path_arg: it leaves a relative path unresolved when the file is absent, which
        # would read here as an escape from the package and refuse a config the worker trains on
        # (_load_contract_text treats a missing file as an empty contract). resolving it against the
        # package unconditionally keeps the containment check about escapes only.
        path = base_dir / configured_path
    try:
        packaged = path.resolve().is_relative_to(base_dir.resolve())
    except (OSError, RuntimeError):
        packaged = False
    if not packaged:
        raise PackagedDatasetUnavailable(
            f"[environment.params] contract_path {configured_path!r} is outside the environment "
            "package. Package the contract file with the environment."
        )
    try:
        oversized = path.is_file() and path.stat().st_size > _MAX_PROFILE_CONTRACT_BYTES
    except OSError:
        oversized = False
    if oversized:
        raise PackagedDatasetUnavailable(
            f"environment training contract {path.name!r} exceeds the "
            f"{_MAX_PROFILE_CONTRACT_BYTES // 1024} KiB control-plane profiling limit. Shorten the "
            "packaged training contract."
        )
    return _bounded_contract_text(_load_contract_text(str(path)), source=path.name)


def _reject_oversized_records(records: object) -> None:
    try:
        encoded = len(json.dumps(records, default=str).encode("utf-8", errors="ignore"))
    except (TypeError, ValueError):
        return
    if encoded > _MAX_PROFILE_DATASET_BYTES:
        raise PackagedDatasetUnavailable(
            "[environment.params] records exceed the "
            f"{_MAX_PROFILE_DATASET_BYTES // (1024 * 1024)} MiB control-plane profiling limit. "
            "Package the dataset as dataset/train.jsonl instead of inlining it."
        )


def _bounded_contract_text(text: str, *, source: str) -> str:
    if len(text.encode("utf-8", errors="ignore")) > _MAX_PROFILE_CONTRACT_BYTES:
        raise PackagedDatasetUnavailable(
            f"environment training contract {source!r} exceeds the "
            f"{_MAX_PROFILE_CONTRACT_BYTES // 1024} KiB control-plane profiling limit. Shorten the "
            "training contract."
        )
    return text


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
    raise PackagedDatasetUnavailable(
        f"environment package has no readable dataset for split {wanted!r}. Add "
        "dataset/train.jsonl to the environment package, or package the selected split as "
        f"dataset/{wanted}.jsonl."
    )


def _read_jsonl_rows(path: Path, *, max_examples: int) -> tuple[int, list[dict[str, Any]]]:
    source_examples = 0
    values = []
    missing = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("dataset row is not an object")
            if _CANONICAL_INPUT_KEY not in row and len(missing) < 10:
                missing.append(source_examples)
            source_examples += 1
            if max_examples <= 0 or len(values) < max_examples:
                values.append(row)
    if missing:
        raise ValueError(
            "Freesolo dataset records must contain an input field; missing at row indexes "
            f"{missing}"
        )
    return source_examples, values


def _validate_dataset_rows(
    values: object,
    *,
    invalid_message: str,
    empty_message: str,
    max_examples: int,
) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(values, list) or not all(isinstance(row, dict) for row in values):
        raise PackagedDatasetUnavailable(invalid_message)
    source_examples = len(values)
    missing = [index for index, row in enumerate(values) if _CANONICAL_INPUT_KEY not in row]
    if missing:
        raise ValueError(
            "Freesolo dataset records must contain an input field; missing at row indexes "
            f"{missing[:10]}"
        )
    selected = values[:max_examples] if max_examples > 0 else values
    if not selected:
        raise PackagedDatasetUnavailable(empty_message)
    return source_examples, [dict(row) for row in selected]


def _read_dataset_rows(path: Path, *, max_examples: int) -> tuple[int, list[dict[str, Any]]]:
    try:
        if path.stat().st_size > _MAX_PROFILE_DATASET_BYTES:
            raise PackagedDatasetUnavailable(
                f"environment dataset file {path.name!r} exceeds the 32 MiB control-plane profiling "
                "limit. Reduce the packaged training dataset."
            )
        if path.suffix.lower() == ".jsonl":
            source_examples, values = _read_jsonl_rows(path, max_examples=max_examples)
        elif path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            values = loaded.get("records") if isinstance(loaded, dict) else loaded
            source_examples, values = _validate_dataset_rows(
                values,
                invalid_message=(
                    f"environment dataset file {path.name!r} must contain JSON object rows. Add a "
                    "valid dataset/train.jsonl to the environment package."
                ),
                empty_message=(
                    f"environment dataset file {path.name!r} contains no rows. Add training rows to "
                    "dataset/train.jsonl in the environment package."
                ),
                max_examples=max_examples,
            )
        else:
            raise ValueError("dataset file must end in .jsonl or .json")
    except PackagedDatasetUnavailable:
        raise
    except TypeError as exc:
        raise PackagedDatasetUnavailable(
            f"environment dataset file {path.name!r} must contain JSON object rows. Add a valid "
            "dataset/train.jsonl to the environment package."
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackagedDatasetUnavailable(
            f"environment dataset file {path.name!r} is not readable JSON. Add a valid "
            "dataset/train.jsonl to the environment package."
        ) from exc
    if not values:
        raise PackagedDatasetUnavailable(
            f"environment dataset file {path.name!r} contains no rows. Add training rows to "
            "dataset/train.jsonl in the environment package."
        )
    return source_examples, [dict(row) for row in values]


def profile_packaged_sft_dataset(
    spec,
    *,
    producer_version: str,
    tokenizer_loader=load_control_plane_tokenizer,
    packing_support=worker_image_packing_support,
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
    records = params.pop("records", None)
    max_examples = int(spec.train.max_examples or 0)
    if records is not None:
        # inline records arrive in the request body rather than the package, so the packaged-file
        # size guard never sees them. bound them by the same limit before anything is copied or
        # tokenized in the shared plane process.
        _reject_oversized_records(records)
        selection = select_dataset_source(params, base_dir, records, _resolve_path_arg)
        source_examples, rows = _validate_dataset_rows(
            selection.source,
            invalid_message=(
                "[environment.params] records must contain JSON object rows. Add valid training "
                "records."
            ),
            empty_message=(
                "[environment.params] records contain no rows. Add training rows to records."
            ),
            max_examples=max_examples,
        )
    else:
        dataset_path = _selected_dataset_path(base_dir, params)
        source_examples, rows = _read_dataset_rows(dataset_path, max_examples=max_examples)
    rows = select_sft_examples(rows, 0, spec.seed)
    raw_environment = _RawRecordEnvironment(
        rows=rows,
        package_root=base_dir,
        contract_text=_contract_text(base_dir, params),
    )
    return prepare_sft_workload(
        spec,
        raw_environment,
        tokenizer_loader=tokenizer_loader,
        producer_version=producer_version,
        # no torch here, so no VL processor: image rows are counted arithmetically from the
        # published vision geometry rather than tokenized through the processor. see
        # flash/engine/profiling/image_tokens.py.
        require_processor=False,
        allow_packing=True,
        packing_support=packing_support,
        source_examples=source_examples,
        examples_preselected=True,
    ).profile
