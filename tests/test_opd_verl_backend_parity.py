"""pinned vllm xgrammar backend parity for structured OPD replay."""

from __future__ import annotations

import importlib.metadata
import json
from types import SimpleNamespace

import pytest

from flash.engine.worker.opd_verl_structured import (
    StructuredOutputReplay,
    _choice_as_grammar,
    canonical_structured_spec,
)


@pytest.mark.opd_verl_backend
def test_replay_bitmask_matches_vllm_0_11_xgrammar_backend():
    backend_module = pytest.importorskip("vllm.v1.structured_output.backend_xgrammar")
    backend_types = pytest.importorskip("vllm.v1.structured_output.backend_types")
    import torch
    import xgrammar
    from transformers import AutoTokenizer

    assert importlib.metadata.version("vllm") == "0.11.0"
    assert importlib.metadata.version("xgrammar") == "0.1.25"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
    model_vocab_size = 248320
    replay = StructuredOutputReplay(tokenizer, model_vocab_size)
    backend = backend_module.XgrammarBackend(
        SimpleNamespace(
            structured_outputs_config=SimpleNamespace(disable_any_whitespace=False),
            speculative_config=None,
        ),
        tokenizer,
        model_vocab_size,
    )
    cases = [
        (
            {"choice": ["4"]},
            backend_types.StructuredOutputOptions.GRAMMAR,
            _choice_as_grammar(["4"]),
            "4",
        ),
        (
            {"regex": "[0-9]+"},
            backend_types.StructuredOutputOptions.REGEX,
            "[0-9]+",
            "42",
        ),
        (
            {
                "json": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                }
            },
            backend_types.StructuredOutputOptions.JSON,
            json.dumps(
                {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                }
            ),
            '{"answer":"4"}',
        ),
    ]
    try:
        for spec, request_type, grammar_spec, text in cases:
            replay_matcher = xgrammar.GrammarMatcher(
                replay._compile(canonical_structured_spec(spec))
            )
            backend_grammar = backend.compile_grammar(request_type, grammar_spec)
            replay_mask = xgrammar.allocate_token_bitmask(1, model_vocab_size)
            backend_mask = backend.allocate_token_bitmask(1)
            for token_id in tokenizer.encode(text, add_special_tokens=False):
                xgrammar.reset_token_bitmask(replay_mask)
                xgrammar.reset_token_bitmask(backend_mask)
                replay_matcher.fill_next_token_bitmask(replay_mask)
                backend_grammar.fill_bitmask(backend_mask, 0)
                assert torch.equal(replay_mask, backend_mask)
                assert replay_matcher.accept_token(token_id)
                assert backend_grammar.accept_tokens("comparison", [token_id])
    finally:
        backend.destroy()
