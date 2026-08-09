"""Pinned local tokenizers for managed OPD teacher scoring."""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from flash.engine.plan.recipe import TeacherModel, teacher_for_model_id

_KIMI_PATTERN = (
    r"[\p{Han}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)


@dataclass(frozen=True)
class EncodedTeacherToken:
    """One exact local token id and its source-character span."""

    token_id: int
    start: int
    end: int


class TeacherTokenizer(Protocol):
    def encode(self, text: str) -> list[EncodedTeacherToken]: ...


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_files(model: TeacherModel) -> dict[str, str]:
    from huggingface_hub import hf_hub_download

    paths: dict[str, str] = {}
    for filename, expected_hash in model.tokenizer_files:
        path = hf_hub_download(
            model.tokenizer_repo,
            filename,
            revision=model.tokenizer_revision,
        )
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"managed teacher tokenizer hash mismatch for {model.alias} {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        paths[filename] = path
    return paths


def _exact_source_spans(
    text: str, token_ids: list[int], offsets: list[tuple[int, int]]
) -> list[EncodedTeacherToken]:
    if len(token_ids) != len(offsets):
        raise RuntimeError("managed teacher tokenizer returned inconsistent ids and offsets")
    if not text:
        if token_ids:
            raise RuntimeError("managed teacher tokenizer emitted tokens for empty text")
        return []
    if not token_ids:
        raise RuntimeError("managed teacher tokenizer emitted no ids for nonempty text")

    spans: list[list[int]] = []
    previous_start = 0
    for index, raw_span in enumerate(offsets):
        if not isinstance(raw_span, tuple) or len(raw_span) != 2:
            raise RuntimeError("managed teacher tokenizer returned an invalid source span")
        start, end = raw_span
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end > len(text)
            or (index and start < previous_start)
        ):
            raise RuntimeError("managed teacher tokenizer returned an invalid source span")
        spans.append([start, end])
        previous_start = start

    if spans[0][0] > 0:
        spans[0][0] = 0
    covered_end = spans[0][1]
    for index in range(1, len(spans)):
        start, end = spans[index]
        if start > covered_end:
            spans[index - 1][1] = start
        covered_end = max(covered_end, end)
    if covered_end < len(text):
        spans[-1][1] = len(text)

    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            raise RuntimeError("managed teacher tokenizer offsets do not cover the source text")
        cursor = max(cursor, end)
    if cursor != len(text):
        raise RuntimeError("managed teacher tokenizer offsets do not cover the source text")
    return [
        EncodedTeacherToken(token_id=int(token_id), start=start, end=end)
        for token_id, (start, end) in zip(token_ids, spans, strict=True)
    ]


class _TokenizerJson:
    def __init__(self, path: str) -> None:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]

        self._tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str) -> list[EncodedTeacherToken]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return _exact_source_spans(text, list(encoding.ids), list(encoding.offsets))


def _byte_piece_offsets(text: str, pieces: list[bytes]) -> list[tuple[int, int]]:
    encoded = text.encode("utf-8")
    if b"".join(pieces) != encoded:
        raise RuntimeError("managed tokenizer bytes do not exactly reproduce the source text")
    char_boundaries = [0]
    for character in text:
        char_boundaries.append(char_boundaries[-1] + len(character.encode("utf-8")))
    spans: list[tuple[int, int]] = []
    byte_cursor = 0
    for piece in pieces:
        byte_end = byte_cursor + len(piece)
        start = max(0, bisect.bisect_right(char_boundaries, byte_cursor) - 1)
        end = bisect.bisect_left(char_boundaries, byte_end)
        spans.append((start, end))
        byte_cursor = byte_end
    return spans


class _KimiTikToken:
    def __init__(self, config_path: str, model_path: str) -> None:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        decoder = config.get("added_tokens_decoder")
        if not isinstance(decoder, dict):
            raise RuntimeError("managed Kimi tokenizer config has no added token decoder")
        special_by_id: dict[int, str] = {}
        for raw_id, value in decoder.items():
            if not isinstance(value, dict) or not isinstance(value.get("content"), str):
                raise RuntimeError("managed Kimi tokenizer config has an invalid added token")
            special_by_id[int(raw_id)] = value["content"]
        mergeable_ranks = load_tiktoken_bpe(model_path)
        base_count = len(mergeable_ranks)
        special_tokens = {
            special_by_id.get(token_id, f"<|reserved_token_{token_id}|>"): token_id
            for token_id in range(base_count, base_count + 256)
        }
        self._encoding = tiktoken.Encoding(
            name="kimi-k3",
            pat_str=_KIMI_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

    def encode(self, text: str) -> list[EncodedTeacherToken]:
        token_ids = self._encoding.encode(text, allowed_special="all")
        pieces = [self._encoding.decode_single_token_bytes(token_id) for token_id in token_ids]
        return _exact_source_spans(text, token_ids, _byte_piece_offsets(text, pieces))


@lru_cache(maxsize=3)
def load_teacher_tokenizer(model_value: str) -> TeacherTokenizer:
    """Load and cache the exact pinned tokenizer for one managed teacher."""
    model = teacher_for_model_id(model_value)
    paths = _download_files(model)
    if model.tokenizer_kind == "tokenizer_json":
        return _TokenizerJson(paths["tokenizer.json"])
    if model.tokenizer_kind == "kimi_tiktoken":
        return _KimiTikToken(paths["tokenizer_config.json"], paths["tiktoken.model"])
    raise RuntimeError(f"unsupported managed teacher tokenizer kind {model.tokenizer_kind!r}")
