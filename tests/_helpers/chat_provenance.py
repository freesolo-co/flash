"""shared managed chat provenance fixtures."""


def managed_chat_result(checkpoint_id: str, content: str = "ok") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "flash_provenance": {"checkpoint_id": checkpoint_id},
    }
