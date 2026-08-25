"""shared managed chat provenance fixtures."""


def managed_chat_result(revision: str, content: str = "ok") -> dict:
    target, source_revision = revision.rsplit(".", 1)
    run_id, selector = target.rsplit("@", 1)
    checkpoint = run_id if selector == "final" else f"{run_id}/{selector}"
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "flash_provenance": {
            "adapter_revision": revision,
            "checkpoint": checkpoint,
            "source_revision": source_revision,
        },
    }


def managed_stream_headers(revision: str) -> dict[str, str]:
    target, source_revision = revision.rsplit(".", 1)
    run_id, selector = target.rsplit("@", 1)
    checkpoint = run_id if selector == "final" else f"{run_id}/{selector}"
    return {
        "content-type": "text/event-stream",
        "x-flash-adapter-revision": revision,
        "x-flash-checkpoint": checkpoint,
        "x-flash-source-revision": source_revision,
    }
