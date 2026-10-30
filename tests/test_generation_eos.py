"""Generation termination metadata shared by the GRPO parent and child."""

from types import SimpleNamespace


def test_generation_eos_ids_unions_tokenizer_and_model_text_config():
    """every tokenizer, generation, top-level, and decoder text-config eos remains a halt id."""
    from flash.engine.worker.train.core.generation import _generation_eos_ids

    nested_config = SimpleNamespace(eos_token_id=248044)
    text_config_calls = []

    class _Config:
        eos_token_id = 151645

        def get_text_config(self, *, decoder=None, encoder=None):
            text_config_calls.append((decoder, encoder))
            return nested_config

    tok = SimpleNamespace(eos_token_id=248046)
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[248046, 73]),
        config=_Config(),
    )
    assert _generation_eos_ids(model, tok) == frozenset({73, 151645, 248044, 248046})
    assert text_config_calls == [(True, None)]

    # scalar-only tokenizer, model without generation config -> just the tokenizer id.
    assert _generation_eos_ids(SimpleNamespace(), SimpleNamespace(eos_token_id=5)) == frozenset({5})
    # nothing defines an eos -> empty set (the fail-open signal for _rollout_terminated).
    assert _generation_eos_ids(SimpleNamespace(), SimpleNamespace()) == frozenset()
    # bool must not leak in as a token id (true == 1 would poison the set).
    assert _generation_eos_ids(SimpleNamespace(), SimpleNamespace(eos_token_id=True)) == frozenset()


def test_generation_eos_ids_ignores_malformed_or_missing_nested_values():
    from flash.engine.worker.train.core.generation import _generation_eos_ids

    class _BrokenSupportedAccess:
        text_config = SimpleNamespace(eos_token_id=[248044, "248045", True, None])

        def get_text_config(self, *, decoder=None, encoder=None):
            raise ValueError("ambiguous text config")

    assert _generation_eos_ids(
        SimpleNamespace(config=_BrokenSupportedAccess()),
        SimpleNamespace(eos_token_id=248046),
    ) == frozenset({248044, 248046})
    assert (
        _generation_eos_ids(
            SimpleNamespace(
                config=SimpleNamespace(text_config=SimpleNamespace(eos_token_id="bad"))
            ),
            SimpleNamespace(),
        )
        == frozenset()
    )
