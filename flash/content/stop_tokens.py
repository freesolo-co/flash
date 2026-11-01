"""Chat-turn boundaries for pretrained checkpoints with document-level EOS metadata."""

from typing import Any


def chat_stop_token_ids(model: str, tokenizer: Any, *, is_chat: bool) -> list[int] | None:
    """Stop Base chat at im_end without changing its tokenizer or raw completion EOS."""
    if model != "Qwen/Qwen3.5-9B-Base" or not is_chat:
        return None
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if type(token_id) is not int or token_id < 0 or token_id == tokenizer.unk_token_id:
        raise ValueError("Qwen3.5 9B Base tokenizer is missing its chat end token")
    return [token_id]
