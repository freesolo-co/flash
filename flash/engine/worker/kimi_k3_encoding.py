"""Pinned, internal Kimi K3 prompt encoding for Parasail teacher scoring."""

from __future__ import annotations

import codecs
import hashlib
import importlib.metadata
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash.engine.worker.teacher import TeacherError, _TeacherMessage, _TeacherScoreRequest
from flash.engine.worker.tokenizer_align import TeacherToken

_PAT_STR = (
    r"[\p{Han}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|"
    r"\s*[\r\n]+|"
    r"\s+(?!\S)|"
    r"\s+"
)
_EXPECTED_SPECIAL_TOKENS = {
    "[BOS]": 163584,
    "[EOS]": 163585,
    "<|end_of_msg|>": 163586,
    "<|open|>": 163587,
    "<|close|>": 163588,
    "<|sep|>": 163589,
    "[UNK]": 163838,
    "[PAD]": 163839,
}
_INTERNAL_THINKING_EFFORT = (
    "`thinking_effort` guides on how much to think in your thinking channel "
    "(not including the response channel), supported values include `low`, "
    "`medium`, `high`, and `max`.\n"
    "Now the system is invoked with `thinking_effort=max`."
)


@dataclass(frozen=True)
class _Segment:
    text: str
    allow_special: bool = False
    label: str = ""


@dataclass(frozen=True)
class EncodedKimiK3Prompt:
    token_ids: tuple[int, ...]
    completion_positions: tuple[int, ...]
    completion_tokens: tuple[TeacherToken, ...]


class KimiK3Encoding:
    def __init__(self, encoding: Any) -> None:
        self._encoding = encoding

    def encode_request(self, request: _TeacherScoreRequest) -> EncodedKimiK3Prompt:
        messages = [
            *request.messages,
            _TeacherMessage("assistant", request.assistant_prefill + request.completion_text),
        ]
        segments = _render_segments(messages)
        token_ids: list[int] = []
        completion_positions: list[int] = []
        completion_tokens: list[TeacherToken] = []
        for segment in segments:
            start = len(token_ids)
            ids = self._encode(segment.text, allow_special=segment.allow_special)
            token_ids.extend(ids)
            if segment.label == "final_content":
                positions, tokens = self._completion_tokens(
                    ids,
                    request.assistant_prefill,
                    request.completion_text,
                    token_offset=start,
                )
                completion_positions.extend(positions)
                completion_tokens.extend(tokens)
        if request.completion_text and not completion_tokens:
            raise TeacherError(
                "Kimi K3 encoding produced no completion tokens",
                permanent=True,
            )
        return EncodedKimiK3Prompt(
            token_ids=tuple(token_ids),
            completion_positions=tuple(completion_positions),
            completion_tokens=tuple(completion_tokens),
        )

    def _encode(self, text: str, *, allow_special: bool) -> list[int]:
        if not text:
            return []
        if allow_special:
            return list(self._encoding.encode(text, allowed_special="all"))
        return list(self._encoding.encode(text, disallowed_special=()))

    def _completion_tokens(
        self,
        token_ids: list[int],
        assistant_prefill: str,
        completion_text: str,
        *,
        token_offset: int,
    ) -> tuple[list[int], list[TeacherToken]]:
        content = assistant_prefill + completion_text
        content_bytes = content.encode("utf-8")
        prefill_bytes = len(assistant_prefill.encode("utf-8"))
        token_bytes: list[bytes] = []
        for token_id in token_ids:
            try:
                value = self._encoding.decode_single_token_bytes(token_id)
            except Exception as exc:
                raise TeacherError(
                    "Kimi K3 tokenizer could not decode a prompt token to bytes",
                    permanent=True,
                ) from exc
            if not isinstance(value, bytes):
                raise TeacherError(
                    "Kimi K3 tokenizer returned non-byte token data",
                    permanent=True,
                )
            token_bytes.append(value)
        if b"".join(token_bytes) != content_bytes:
            raise TeacherError(
                "Kimi K3 tokenizer bytes do not reconstruct the final assistant content",
                permanent=True,
            )

        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        byte_cursor = 0
        char_cursor = 0
        positions: list[int] = []
        tokens: list[TeacherToken] = []
        completion_bytes = bytearray()
        completion_char_start = len(assistant_prefill)
        for local_index, value in enumerate(token_bytes):
            start_byte = byte_cursor
            start_char = char_cursor
            byte_cursor += len(value)
            try:
                decoded = decoder.decode(value, final=False)
            except UnicodeDecodeError as exc:
                raise TeacherError(
                    "Kimi K3 token bytes are not valid cumulative UTF-8",
                    permanent=True,
                ) from exc
            char_cursor += len(decoded)
            if byte_cursor <= prefill_bytes:
                continue
            overlap_start = max(start_byte, prefill_bytes)
            completion_bytes.extend(content_bytes[overlap_start:byte_cursor])
            start = max(0, start_char - completion_char_start)
            end = max(0, char_cursor - completion_char_start)
            positions.append(token_offset + local_index)
            tokens.append(
                TeacherToken(
                    text=completion_text[start:end],
                    logprob=0.0,
                    start=start,
                    end=end,
                )
            )
        try:
            tail = decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise TeacherError(
                "Kimi K3 token bytes end in incomplete UTF-8",
                permanent=True,
            ) from exc
        if tail:
            raise TeacherError(
                "Kimi K3 tokenizer emitted delayed trailing text",
                permanent=True,
            )
        try:
            reconstructed = bytes(completion_bytes).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TeacherError(
                "Kimi K3 completion bytes are not valid UTF-8",
                permanent=True,
            ) from exc
        if reconstructed != completion_text:
            raise TeacherError(
                "Kimi K3 token bytes do not reconstruct the exact completion",
                permanent=True,
            )
        return positions, tokens


_encoding_lock = threading.Lock()
_cached_encodings: dict[tuple[str, str, str, str], KimiK3Encoding] = {}


def load_kimi_k3_encoding(
    *,
    repo_id: str,
    revision: str,
    tokenizer_config_sha256: str,
    tokenizer_model_sha256: str,
) -> KimiK3Encoding:
    cache_key = (repo_id, revision, tokenizer_config_sha256, tokenizer_model_sha256)
    with _encoding_lock:
        cached = _cached_encodings.get(cache_key)
        if cached is not None:
            return cached
        encoding = _download_and_build(
            repo_id=repo_id,
            revision=revision,
            tokenizer_config_sha256=tokenizer_config_sha256,
            tokenizer_model_sha256=tokenizer_model_sha256,
        )
        _cached_encodings[cache_key] = encoding
        return encoding


def _download_and_build(
    *,
    repo_id: str,
    revision: str,
    tokenizer_config_sha256: str,
    tokenizer_model_sha256: str,
) -> KimiK3Encoding:
    try:
        if importlib.metadata.version("tiktoken") != "0.13.0":
            raise TeacherError("Kimi K3 scoring requires tiktoken==0.13.0", permanent=True)
    except importlib.metadata.PackageNotFoundError as exc:
        raise TeacherError("Kimi K3 scoring requires tiktoken==0.13.0", permanent=True) from exc

    try:
        from huggingface_hub import hf_hub_download

        config_path = Path(
            hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json", revision=revision)
        )
        model_path = Path(
            hf_hub_download(repo_id=repo_id, filename="tiktoken.model", revision=revision)
        )
    except TeacherError:
        raise
    except Exception as exc:
        raise TeacherError("pinned Kimi K3 tokenizer download failed") from exc

    _require_sha256(config_path, tokenizer_config_sha256, "tokenizer config")
    _require_sha256(model_path, tokenizer_model_sha256, "tiktoken model")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TeacherError("pinned Kimi K3 tokenizer config is invalid", permanent=True) from exc
    if config.get("tokenizer_class") != "TikTokenTokenizer":
        raise TeacherError("pinned Kimi K3 tokenizer class is invalid", permanent=True)
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict) or auto_map.get("AutoTokenizer") != [
        "tokenization_kimi.TikTokenTokenizer",
        None,
    ]:
        raise TeacherError("pinned Kimi K3 tokenizer auto-map is invalid", permanent=True)
    decoder = config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        raise TeacherError("pinned Kimi K3 special-token table is invalid", permanent=True)

    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        ranks = load_tiktoken_bpe(str(model_path))
        if len(ranks) != 163584:
            raise TeacherError(
                "pinned Kimi K3 tokenizer vocabulary size is invalid", permanent=True
            )
        added = {
            int(token_id): value["content"]
            for token_id, value in decoder.items()
            if isinstance(value, dict) and isinstance(value.get("content"), str)
        }
        special = {
            added.get(token_id, f"<|reserved_token_{token_id}|>"): token_id
            for token_id in range(len(ranks), len(ranks) + 256)
        }
        for token, token_id in _EXPECTED_SPECIAL_TOKENS.items():
            if special.get(token) != token_id:
                raise TeacherError(
                    "pinned Kimi K3 special-token ids are invalid",
                    permanent=True,
                )
        encoding = tiktoken.Encoding(
            name="flash-kimi-k3",
            pat_str=_PAT_STR,
            mergeable_ranks=ranks,
            special_tokens=special,
        )
    except TeacherError:
        raise
    except Exception as exc:
        raise TeacherError("pinned Kimi K3 tokenizer could not be loaded", permanent=True) from exc
    if encoding.n_vocab != 163840:
        raise TeacherError("pinned Kimi K3 tokenizer vocabulary is invalid", permanent=True)
    return KimiK3Encoding(encoding)


def _require_sha256(path: Path, expected: str, label: str) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TeacherError(f"pinned Kimi K3 {label} could not be read") from exc
    if digest.hexdigest() != expected:
        raise TeacherError(f"pinned Kimi K3 {label} hash mismatch", permanent=True)


def _segment(text: object, *, allow_special: bool = False, label: str = "") -> _Segment:
    return _Segment(str(text), allow_special=allow_special, label=label)


def _attr(key: str, value: object) -> list[_Segment]:
    escaped = str(value).replace("&", "&amp;").replace('"', "&quot;")
    return [_segment(f" {key}"), _segment('="'), _segment(escaped), _segment('"')]


def _open_tag(tag: str, attrs: list[tuple[str, object]] | None = None) -> list[_Segment]:
    result = [_segment("<|open|>", allow_special=True), _segment(tag)]
    for key, value in attrs or []:
        result.extend(_attr(key, value))
    result.append(_segment("<|sep|>", allow_special=True))
    return result


def _close_tag(tag: str) -> list[_Segment]:
    return [
        _segment("<|close|>", allow_special=True),
        _segment(tag),
        _segment("<|sep|>", allow_special=True),
    ]


def _internal_thinking_effort() -> list[_Segment]:
    result = _open_tag("message", [("role", "system"), ("type", "thinking-effort")])
    result.append(_segment(_INTERNAL_THINKING_EFFORT.strip()))
    result.extend(_close_tag("message"))
    result.append(_segment("<|end_of_msg|>", allow_special=True))
    return result


def _render_message(message: _TeacherMessage, *, final: bool) -> list[_Segment]:
    result = _open_tag("message", [("role", message.role)])
    if message.role == "assistant":
        result.extend(_open_tag("think"))
        if message.reasoning_content is not None and message.reasoning_content.strip():
            result.append(_segment(message.reasoning_content))
        result.extend(_close_tag("think"))
        result.extend(_open_tag("response"))
        if message.content:
            result.append(_segment(message.content, label="final_content" if final else ""))
        result.extend(_close_tag("response"))
    elif message.content:
        result.append(_segment(message.content))
    result.extend(_close_tag("message"))
    result.append(_segment("<|end_of_msg|>", allow_special=True))
    return result


def _render_segments(messages: list[_TeacherMessage]) -> list[_Segment]:
    if not messages or messages[-1].role != "assistant":
        raise TeacherError("Kimi K3 scoring requires a final assistant message", permanent=True)
    result = _internal_thinking_effort()
    for index, message in enumerate(messages):
        result.extend(_render_message(message, final=index == len(messages) - 1))
    result.extend(_open_tag("message", [("role", "assistant")]))
    result.extend(_open_tag("think"))
    return result
