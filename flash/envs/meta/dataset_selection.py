"""Packaged-dataset discovery and the split/dataset_path precedence rules.

Split out of loader.py, which imports this module; adapter.py and content.multimodal reach
these helpers through loader's re-exports, so the loader names stay the public surface.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

_DATASET_SPLIT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _packaged_dataset_file(base_dir: Path, name: str) -> Path | None:
    """First existing packaged dataset file for split `name`.

    ``dataset/`` is canonical for new environments because a top-level ``datasets/``
    directory shadows the Hugging Face ``datasets`` package in local scripts.
    """
    for rel in (
        f"dataset/{name}.jsonl",
        f"dataset/{name}.json",
        f"{name}.jsonl",
        f"{name}.json",
    ):
        candidate = base_dir / rel
        if candidate.is_file():
            return candidate
    return None


def _plural_dataset_file(base_dir: Path, name: str) -> Path | None:
    """Split `name` as packaged under the top-level ``datasets/`` (plural) directory.

    A package may legitimately carry BOTH directories: ``dataset/`` for the rows Flash reads and
    ``datasets/`` for raw or eval assets. The plural directory is only a problem when the split
    the operator asked for is exactly what sits unread inside it.
    """
    for rel in (f"datasets/{name}.jsonl", f"datasets/{name}.json"):
        candidate = base_dir / rel
        if candidate.is_file():
            return candidate
    return None


def _validate_packaged_dataset_split(split: str) -> str:
    if not _DATASET_SPLIT_RE.fullmatch(split):
        raise ValueError(
            "[environment.params] split must be a simple dataset name "
            "(letters, numbers, '.', '_', '-' only; no slashes or traversal)"
        )
    return split


def env_dataset_rows(sdk_env: object):
    """The rows an sdk env supplies in code, or None when it exposes no dataset attribute.

    ``dataset`` is probed first and ``examples`` only when it is absent or None: an attribute
    that is present but empty is a deliberate answer (a filter that matched nothing), not a
    reason to keep probing. Callers distinguish None (no in-code dataset, the packaged file is
    the source) from an empty sequence (the env rejected every row).
    """
    rows = getattr(sdk_env, "dataset", None)
    if rows is None:
        rows = getattr(sdk_env, "examples", None)
    return rows


class DatasetSelection(NamedTuple):
    source: object
    source_is_dataset_file: bool
    side_split: bool
    datasets_dir_unread: bool
    unread_split_hint: str


def select_dataset_source(
    params: dict,
    base_dir: Path,
    source: object,
    resolve_path: Callable[[object, Path], object],
) -> DatasetSelection:
    """Resolve which rows a run trains on and whether the env's own dataset may override them.

    Mutates ``params`` the way ``load_environment`` expects (normalized split, resolved
    dataset_path) and encodes the precedence rules: explicit records win outright, an explicit
    non-default dataset_path stays authoritative, the default packaged train file (and a probed
    split file for split="train") yields to the env's in-code dataset, and a requested side
    split must resolve to a real packaged file.
    """
    # [environment.params] split selects which packaged dataset file Flash trains on. it used to
    # be sdk-forwarded only, so training silently used train.jsonl even with a side split.
    split = params.get("split")
    split = split.strip() if isinstance(split, str) else None
    if split:
        split = _validate_packaged_dataset_split(split)
    if isinstance(params.get("split"), str):
        params["split"] = split or ""  # the sdk sees the validated name, not the raw padding
    side_split = bool(split) and split != "train"
    # true when source is the default packaged train file the env received as a param; its own
    # dataset then wins over re-reading the file. explicit records never reach the env.
    source_is_dataset_file = False
    dataset_path = params.get("dataset_path")
    if source is None and dataset_path:
        resolved_dataset_path = resolve_path(dataset_path, base_dir)
        params["dataset_path"] = resolved_dataset_path
        source = resolved_dataset_path
        # an explicit dataset_path names the rows to train on, and the scaffolded pattern is a
        # class-level `dataset = load_jsonl("dataset/train.jsonl")` that ignores the dataset_path
        # it is handed; letting that env dataset win would silently train on the default split
        # instead of the named file, so the explicit file stays authoritative. only the default
        # train file keeps env precedence (filtering the injected train file remains the headline
        # behavior), and only when no side split was requested alongside it: with dataset_path
        # AND split both explicit, env precedence would let a split-honoring env deliver the side
        # rows, so the codified rule (an explicit dataset_path wins over split) holds.
        default_train = _packaged_dataset_file(base_dir, "train")
        source_is_dataset_file = (
            not side_split
            and default_train is not None
            and isinstance(resolved_dataset_path, str)
            and Path(resolved_dataset_path).resolve() == default_train.resolve()
        )
    datasets_dir_unread = False
    unread_split_hint = ""
    if source is None:
        wanted = split if split and split != "train" else "train"
        found = _packaged_dataset_file(base_dir, wanted)
        # the requested split may be sitting unread under datasets/ (plural) while dataset/
        # holds other splits. that is the layout error the caller raises, and it names the
        # actual file, so it wins over the generic missing-split message.
        plural = _plural_dataset_file(base_dir, wanted) if found is None else None
        if plural is not None:
            unread_split_hint = (
                f" The {wanted!r} split is packaged at "
                f"{plural.relative_to(base_dir).as_posix()}, which Flash never reads; it looks "
                f"for dataset/{wanted}.jsonl or dataset/{wanted}.json."
            )
        if (
            found is None
            and plural is None
            and wanted != "train"
            and _packaged_dataset_file(base_dir, "train")
        ):
            # A default train.jsonl exists but the requested split file does not: refuse to fall
            # back silently (that trains on the wrong targets); envs with no packaged dataset at
            # all keep the SDK path, which may implement split itself.
            raise ValueError(
                f"[environment.params] split={split!r} was requested but no "
                f"dataset/{split}.jsonl or {split}.json exists in the environment; "
                "refusing to fall back to the default train split. Package the split file "
                "or drop the split param."
            )
        # a top-level datasets/ (plural) directory is never probed, so a package laid out that way
        # would otherwise fall through silently to whatever else resolved, often the wrong rows
        # entirely. only when the probe found nothing: a package may legitimately carry datasets/
        # for eval or other assets alongside a supported top-level <split>.jsonl. explicit
        # records/dataset_path params skip it because the user already said what to train on.
        # the verdict is deferred to after load_environment: only an env that cannot supply its
        # own rows actually needs the file this layout hid.
        # a singular dataset/ normally makes the plural directory unremarkable -- except when
        # the split the operator asked for is the file sitting unread inside it, which is the
        # silent-wrong-rows case this guard exists for.
        datasets_dir_unread = found is None and (
            plural is not None
            or ((base_dir / "datasets").is_dir() and not (base_dir / "dataset").is_dir())
        )
        if found is not None:
            params.setdefault("dataset_path", str(found))
            source = str(found)
            # an explicitly requested side split names the rows to train on, and the scaffolded
            # pattern is a CLASS-level `dataset = load_jsonl("dataset/train.jsonl")` that ignores
            # the dataset_path it is handed. letting that env's dataset win would silently train
            # on the default split -- the exact failure the split probe exists to stop -- so the
            # probed split file stays authoritative. the default train split keeps env precedence.
            source_is_dataset_file = wanted == "train"
    return DatasetSelection(
        source=source,
        source_is_dataset_file=source_is_dataset_file,
        side_split=side_split,
        datasets_dir_unread=datasets_dir_unread,
        unread_split_hint=unread_split_hint,
    )
