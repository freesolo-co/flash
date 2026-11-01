from types import SimpleNamespace

import pytest

from flash.content.stop_tokens import chat_stop_token_ids


@pytest.mark.parametrize("token_id", [None, -1, True, 7])
def test_base_chat_rejects_missing_or_unknown_turn_boundary(token_id):
    tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: token_id, unk_token_id=7)
    with pytest.raises(ValueError, match="missing its chat end token"):
        chat_stop_token_ids("Qwen/Qwen3.5-9B-Base", tokenizer, is_chat=True)
