"""shared managed chat provenance fixtures."""


def managed_chat_result(checkpoint_id: str, content: str = "ok") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "flash_provenance": {"checkpoint_id": checkpoint_id},
    }


def managed_stream_headers(checkpoint_id: str) -> dict[str, str]:
    return {
        "content-type": "text/event-stream",
        "x-flash-checkpoint-id": checkpoint_id,
        "x-freesolo-lora-request-adapter": checkpoint_id,
    }
