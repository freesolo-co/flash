"""cpu-only xgrammar replay for structured-output verl OPD rollouts."""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import re
from typing import Any

_XGRAMMAR_VERSION = "0.1.25"
_THINK_END_TOKEN = "</think>"


def _choice_as_grammar(choices: list[str]) -> str:
    """match vLLM 0.11.0's choice-to-ebnf conversion exactly."""

    def escape(value: str) -> str:
        return re.sub(r'(["\\])', r"\\\1", value)

    return "root ::= " + " | ".join(f'"{escape(choice)}"' for choice in choices)


def _count_legal_tokens(bitmask, vocab_size: int) -> int:
    """count enabled bits in one xgrammar mask, excluding word padding."""
    if vocab_size <= 0:
        raise ValueError("model vocabulary size must be positive")
    row = bitmask[0]
    expected_words = (vocab_size + 31) // 32
    if int(row.numel()) != expected_words:
        raise RuntimeError("xgrammar token bitmask does not match the model vocabulary size")
    total = 0
    remaining = vocab_size
    for raw_word in row.tolist():
        used = min(32, remaining)
        word = int(raw_word) & 0xFFFFFFFF
        if used < 32:
            word &= (1 << used) - 1
        total += word.bit_count()
        remaining -= used
    return total


class StructuredOutputReplay:
    """compile one canonical constraint and replay it with a fresh matcher per sequence."""

    def __init__(self, tokenizer, model_vocab_size: int, *, xgrammar_module=None) -> None:
        if xgrammar_module is None:
            if importlib.metadata.version("xgrammar") != _XGRAMMAR_VERSION:
                raise RuntimeError(f"structured OPD requires xgrammar {_XGRAMMAR_VERSION} exactly")
            import xgrammar as xgrammar_module

        self._xgrammar = xgrammar_module
        self._tokenizer = tokenizer
        self._model_vocab_size = int(model_vocab_size)
        tokenizer_info = xgrammar_module.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=self._model_vocab_size,
        )
        if int(tokenizer_info.vocab_size) != self._model_vocab_size:
            raise RuntimeError(
                "xgrammar compiler vocabulary size does not match the model vocabulary size"
            )
        self._compiler = xgrammar_module.GrammarCompiler(
            tokenizer_info,
            max_threads=8,
            cache_enabled=True,
        )
        self._compiled_grammars: dict[str, Any] = {}
        self._bitmask = xgrammar_module.allocate_token_bitmask(1, self._model_vocab_size)
        if tuple(self._bitmask.shape) != xgrammar_module.get_bitmask_shape(
            1, self._model_vocab_size
        ):
            raise RuntimeError(
                "xgrammar token bitmask shape does not match the model vocabulary size"
            )

    def _compile(self, canonical_spec: str):
        cached = self._compiled_grammars.get(canonical_spec)
        if cached is not None:
            return cached
        spec = json.loads(canonical_spec)
        disable_any_whitespace = bool(spec.get("disable_any_whitespace", False))
        if spec.get("json") is not None:
            compiled = self._compiler.compile_json_schema(
                json.dumps(spec["json"]),
                any_whitespace=not disable_any_whitespace,
            )
        elif spec.get("json_object") is not None:
            compiled = self._compiler.compile_json_schema(
                '{"type": "object"}',
                any_whitespace=not disable_any_whitespace,
            )
        elif spec.get("regex") is not None:
            compiled = self._compiler.compile_regex(str(spec["regex"]))
        elif spec.get("choice") is not None:
            compiled = self._compiler.compile_grammar(_choice_as_grammar(list(spec["choice"])))
        else:
            raise ValueError("structured OPD constraint has no supported grammar")
        if len(self._compiled_grammars) >= 32:
            self._compiled_grammars.pop(next(iter(self._compiled_grammars)))
        self._compiled_grammars[canonical_spec] = compiled
        return compiled

    def _think_end_token_id(self) -> int:
        token_id = self._tokenizer.get_vocab().get(_THINK_END_TOKEN)
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise RuntimeError(
                "structured thinking requires </think> as a tokenizer vocabulary token"
            )
        return token_id

    def forced_mask(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        canonical_spec: str,
        *,
        thinking: bool,
    ) -> list[bool]:
        """recover positions where exactly one model-vocabulary token was legal."""
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        response_ids = [int(token_id) for token_id in response_ids]
        grammar_start = 0
        if thinking:
            think_end_token_id = self._think_end_token_id()
            if think_end_token_id not in prompt_ids:
                try:
                    grammar_start = response_ids.index(think_end_token_id) + 1
                except ValueError:
                    # a thinking response that never closes </think> never entered the grammar
                    # region: vllm enforced the constraint only after the think block, so this
                    # completion is unconstrained text — reject it rather than silently training
                    # on it with an all-unforced mask.
                    raise RuntimeError(
                        "structured OPD thinking response never closed </think>; the grammar "
                        "region was never entered (truncated or runaway thinking)"
                    ) from None

        compiled = self._compile(canonical_spec)
        matcher = self._xgrammar.GrammarMatcher(compiled)
        forced = [False] * grammar_start
        for position, token_id in enumerate(response_ids[grammar_start:], start=grammar_start):
            self._xgrammar.reset_token_bitmask(self._bitmask)
            matcher.fill_next_token_bitmask(self._bitmask)
            forced.append(_count_legal_tokens(self._bitmask, self._model_vocab_size) == 1)
            if not matcher.accept_token(token_id):
                raise RuntimeError(
                    "xgrammar rejected a generated structured-output token at "
                    f"response position {position}"
                )
        if len(forced) != len(response_ids):
            raise AssertionError("structured OPD forced mask has the wrong response length")
        if grammar_start >= len(response_ids):
            # the response ended at (or before) the grammar region: no structured output was
            # generated at all — reject rather than replaying an empty region as valid.
            raise RuntimeError("structured OPD response contains no tokens in the grammar region")
        if not matcher.is_terminated():
            # a grammar like a bare integer is not "terminated" after `4` (it could continue as
            # `42`); completeness means STOPPING here is legal. accept the output iff the matcher
            # can take the stop token now — otherwise the structured output was truncated
            # mid-grammar and would train on garbage.
            stop_ok = False
            for _stop_id in list(getattr(matcher, "stop_token_ids", []) or []):
                with contextlib.suppress(Exception):
                    if matcher.accept_token(int(_stop_id)):
                        stop_ok = matcher.is_terminated()
                        break
            if not stop_ok:
                raise RuntimeError(
                    "structured OPD replay ended mid-grammar (truncated structured output)"
                )
        return forced


def canonical_structured_spec(spec: dict[str, Any]) -> str:
    """serialize a normalized flash constraint for stable compilation caching."""
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))
