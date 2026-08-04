"""Parasail Kimi K3 supplied-completion teacher adapter."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from flash.engine.worker.kimi_k3_encoding import KimiK3Encoding, load_kimi_k3_encoding
from flash.engine.worker.teacher import (
    TeacherClient,
    TeacherError,
    _ScoredTeacherTokens,
    _TeacherMessage,
    _TeacherScoreRequest,
)
from flash.engine.worker.tokenizer_align import TeacherToken

_MAX_CONCURRENCY = 8


class ParasailTeacherClient(TeacherClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        encoding_repo: str,
        encoding_revision: str,
        tokenizer_config_sha256: str,
        tokenizer_model_sha256: str,
        timeout: float = 90.0,
        max_retries: int = 4,
        encoding: KimiK3Encoding | None = None,
    ) -> None:
        super().__init__(
            api_key,
            base_url,
            model,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.encoding_repo = encoding_repo
        self.encoding_revision = encoding_revision
        self.tokenizer_config_sha256 = tokenizer_config_sha256
        self.tokenizer_model_sha256 = tokenizer_model_sha256
        self._encoding = encoding

    @property
    def encoding(self) -> KimiK3Encoding:
        if self._encoding is None:
            self._encoding = load_kimi_k3_encoding(
                repo_id=self.encoding_repo,
                revision=self.encoding_revision,
                tokenizer_config_sha256=self.tokenizer_config_sha256,
                tokenizer_model_sha256=self.tokenizer_model_sha256,
            )
        return self._encoding

    def preflight(self) -> None:
        _ = self.encoding

    def score_many(
        self, items: list[_TeacherScoreRequest | tuple[str, str]]
    ) -> list[list[TeacherToken]]:
        if not items:
            return []
        requests: list[_TeacherScoreRequest] = []
        for item in items:
            if not isinstance(item, _TeacherScoreRequest):
                raise TeacherError(
                    "Kimi K3 scoring requires structured teacher messages",
                    permanent=True,
                )
            requests.append(item)
        workers = min(_MAX_CONCURRENCY, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._score_request, requests))

    def score(
        self,
        request: _TeacherScoreRequest | str,
        completion_text: str | None = None,
    ) -> list[TeacherToken]:
        if not isinstance(request, _TeacherScoreRequest) or completion_text is not None:
            raise TeacherError(
                "Kimi K3 scoring requires one structured score request",
                permanent=True,
            )
        return self._score_request(request)

    def _score_request(self, request: _TeacherScoreRequest) -> _ScoredTeacherTokens:
        encoded = self.encoding.encode_request(request)
        messages = [message.to_payload() for message in request.messages]
        messages.append(
            _TeacherMessage(
                role="assistant",
                content=request.assistant_prefill + request.completion_text,
            ).to_payload()
        )
        response = self._post(
            "/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "prompt_logprobs": 1,
                "continue_final_message": True,
                "return_token_ids": True,
                "max_tokens": 1,
                "temperature": 0,
                "top_p": 1,
                "seed": 0,
            },
        )
        prompt_ids, prompt_scores, generated_ids = _validate_response(response)
        if tuple(prompt_ids) != encoded.token_ids:
            mismatch = _mismatch_index(encoded.token_ids, tuple(prompt_ids))
            raise TeacherError(
                "Kimi K3 prompt token ids do not match the pinned local encoding "
                f"at position {mismatch}",
                permanent=True,
            )
        output: list[TeacherToken] = []
        for position, token in zip(
            encoded.completion_positions,
            encoded.completion_tokens,
            strict=True,
        ):
            score = prompt_scores[position]
            if score is None:
                raise TeacherError(
                    "Kimi K3 response has no realized score for a completion token",
                    permanent=True,
                )
            output.append(
                TeacherToken(
                    text=token.text,
                    logprob=score,
                    start=token.start,
                    end=token.end,
                )
            )
        if request.completion_text and not output:
            raise TeacherError(
                "Kimi K3 response returned no scored completion token",
                permanent=True,
            )
        return _ScoredTeacherTokens(
            output,
            input_tokens=len(prompt_ids),
            output_tokens=len(generated_ids),
        )


def _validate_response(response: object) -> tuple[list[int], list[float | None], list[int]]:
    if not isinstance(response, dict):
        raise TeacherError("Kimi K3 response is not an object", permanent=True)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise TeacherError("Kimi K3 response must contain exactly one choice", permanent=True)
    choice = choices[0]
    prompt_ids = response.get("prompt_token_ids", choice.get("prompt_token_ids"))
    if not isinstance(prompt_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in prompt_ids
    ):
        raise TeacherError("Kimi K3 response prompt token ids are invalid", permanent=True)
    prompt_logprobs = response.get("prompt_logprobs")
    if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(prompt_ids):
        raise TeacherError(
            "Kimi K3 response prompt logprobs do not match the prompt token ids",
            permanent=True,
        )
    scores: list[float | None] = []
    for position, (token_id, entry) in enumerate(zip(prompt_ids, prompt_logprobs, strict=True)):
        if position == 0 and entry is None:
            scores.append(None)
            continue
        if not isinstance(entry, dict):
            raise TeacherError(
                "Kimi K3 response prompt logprob entry is invalid",
                permanent=True,
            )
        record = entry.get(str(token_id), entry.get(token_id))
        if not isinstance(record, dict):
            raise TeacherError(
                "Kimi K3 response lacks the realized token-id logprob",
                permanent=True,
            )
        value = record.get("logprob")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TeacherError(
                "Kimi K3 response realized logprob is not numeric",
                permanent=True,
            )
        score = float(value)
        if not math.isfinite(score) or score > 0.0:
            raise TeacherError(
                "Kimi K3 response realized logprob is invalid",
                permanent=True,
            )
        scores.append(score)
    generated_ids = choice.get("token_ids")
    if not isinstance(generated_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in generated_ids
    ):
        raise TeacherError("Kimi K3 response generated token ids are invalid", permanent=True)
    if len(generated_ids) != 1:
        raise TeacherError(
            "Kimi K3 response must contain exactly one generated token id",
            permanent=True,
        )
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise TeacherError("Kimi K3 response usage is missing", permanent=True)
    prompt_usage = usage.get("prompt_tokens")
    completion_usage = usage.get("completion_tokens")
    if (
        isinstance(prompt_usage, bool)
        or not isinstance(prompt_usage, int)
        or prompt_usage != len(prompt_ids)
    ):
        raise TeacherError("Kimi K3 prompt usage does not match token ids", permanent=True)
    if (
        isinstance(completion_usage, bool)
        or not isinstance(completion_usage, int)
        or completion_usage != len(generated_ids)
    ):
        raise TeacherError("Kimi K3 output usage does not match token ids", permanent=True)
    return prompt_ids, scores, generated_ids


def _mismatch_index(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right, strict=False)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))
