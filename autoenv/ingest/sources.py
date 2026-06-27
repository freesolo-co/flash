"""Resolve a dataset ref to canonical ``{"input", "output"}`` rows.

A ref is one of: a local ``.jsonl``/``.json`` path (loaded offline, no deps), an ``http(s)``
URL to a JSON/JSONL file (urllib), or an ``hf:<dataset_id>`` / bare ``org/name`` Hugging Face
dataset (lazy ``datasets`` import, behind the ``autoenv`` extra). Anything else raises a clear
``DatasetUnavailable``. ``fetch_rows`` returns normalised rows; ``can_fetch`` is the gate's
cheap availability probe (does the ref resolve to at least one row?) and never raises.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any


class DatasetUnavailable(RuntimeError):
    """A dataset ref could not be resolved to rows (missing file, network, bad format)."""


_REMOTE_PREFIXES = ("http://", "https://")
# Ref schemes that look remote but autoenv does not (yet) fetch — rejected with a clear error
# instead of falling through to the bare ``org/name`` Hugging Face path (a confusing HF failure).
_UNSUPPORTED_PREFIXES = ("github:", "git+", "git:", "ssh:", "s3:", "gs:", "file:")
_FETCH_TIMEOUT_S = 30.0


def _canonical_rows(
    raw_rows: list[dict], *, input_field: str = "input", output_field: str = "output"
) -> list[dict]:
    """Map source columns onto Freesolo's canonical ``input``/``output`` keys.

    Rows already in canonical shape pass through. A row missing the input column is dropped
    (with the rest kept) — a partially-malformed dump shouldn't sink the whole case.
    """
    out: list[dict] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        if input_field in row:
            value_in = row[input_field]
        elif "input" in row:
            value_in = row["input"]
        else:
            continue
        value_out = row.get(output_field, row.get("output", ""))
        out.append({"input": value_in, "output": value_out})
    return out


def _load_local(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded.get("records") or loaded.get("data") or []
        return loaded if isinstance(loaded, list) else []
    raise DatasetUnavailable(f"unsupported local dataset format: {path.suffix} ({path})")


def _load_http(url: str) -> list[dict]:
    try:
        with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:
        raise DatasetUnavailable(f"could not download {url}: {exc}") from exc
    text = body.strip()
    # Try the whole body as one JSON document first; only fall back to JSONL (one object per
    # line) if that fails — a pretty-printed JSON object spans lines too, so a newline heuristic
    # would mis-parse it line-by-line.
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(loaded, dict):
        return loaded.get("records") or loaded.get("data") or []
    return loaded if isinstance(loaded, list) else []


def _load_hf(spec: str, split: str) -> list[dict]:
    # ``spec`` is the part after ``hf:`` — ``<dataset_id>`` or ``<dataset_id>:<config>`` (a
    # dataset like GSM8K requires a config name, e.g. ``openai/gsm8k:main``).
    dataset_id, _, config = spec.partition(":")
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DatasetUnavailable(
            "loading a Hugging Face dataset needs the 'datasets' package "
            "(install the autoenv extra: pip install -e '.[autoenv]')"
        ) from exc
    try:
        ds = load_dataset(dataset_id, config or None, split=split)
    except Exception as exc:
        where = f"{dataset_id!r}" + (f" config {config!r}" if config else "")
        raise DatasetUnavailable(
            f"could not load HF dataset {where} split {split!r}: {exc}"
        ) from exc
    return [dict(row) for row in ds]


def fetch_rows(
    ref: str,
    *,
    split: str = "train",
    input_field: str = "input",
    output_field: str = "output",
    limit: int | None = None,
) -> list[dict]:
    """Resolve ``ref`` to a list of canonical ``{"input", "output"}`` rows."""
    if not ref:
        raise DatasetUnavailable("empty dataset ref")

    low = ref.lower()
    if low.startswith(_REMOTE_PREFIXES):
        raw = _load_http(ref)
    elif low.startswith("hf:"):
        raw = _load_hf(ref[3:], split)
    elif low.startswith(_UNSUPPORTED_PREFIXES):
        raise DatasetUnavailable(
            f"unsupported dataset ref scheme: {ref!r} (supported: a local .jsonl/.json path, "
            "an http(s) URL, hf:<dataset_id>, or a bare org/name Hugging Face id)"
        )
    else:
        path = Path(ref)
        if path.suffix.lower() in (".jsonl", ".json"):
            if not path.exists():
                raise DatasetUnavailable(f"local dataset not found: {path}")
            raw = _load_local(path)
        elif "/" in ref and not path.suffix:
            # Bare ``org/name`` -> a Hugging Face dataset id.
            raw = _load_hf(ref, split)
        else:
            raise DatasetUnavailable(f"unrecognised dataset ref: {ref!r}")

    rows = _canonical_rows(raw, input_field=input_field, output_field=output_field)
    return rows[:limit] if limit else rows


def can_fetch(ref: str, *, split: str = "train", **kwargs: Any) -> tuple[bool, str]:
    """Cheap availability probe for the gate: can ``ref`` be resolved to >=1 row? Never raises."""
    try:
        rows = fetch_rows(ref, split=split, limit=1, **kwargs)
    except DatasetUnavailable as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"unexpected error probing {ref!r}: {exc}"
    if not rows:
        return False, f"{ref!r} resolved but yielded no rows"
    return True, f"resolved {ref!r} (sampled {len(rows)} row)"
