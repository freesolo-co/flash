"""Exercise prefill/decode before a newly allocated replica accepts customer traffic."""

import asyncio
import logging
from dataclasses import replace

from flash.serving.deepinfer.contract.config import (
    MODEL_ID,
    PREFILL_TOKENS,
    RANK_COUNT,
)
from flash.serving.deepinfer.contract.protocol import parse_request
from flash.serving.deepinfer.runtime.kernel_cache import verified_compiler_cache


async def warmup(runtime):
    runtime.compiled_kernel_cache_verified = await asyncio.to_thread(
        verified_compiler_cache, getattr(runtime, "provenance", {})
    )
    logging.getLogger(__name__).info(
        "Compiled kernel cache verified: %s", runtime.compiled_kernel_cache_verified
    )

    async def request(index, repetitions, concurrency, prefill_tokens=None, cache=False):
        parsed = parse_request(
            {
                "model": MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Warmup {prefill_tokens} batch {concurrency} request {index}. Write a numbered list of 100 detailed items. "
                            + "Count these words carefully and then continue the list. "
                            * repetitions
                        ),
                    }
                ],
                "max_tokens": 1 if prefill_tokens else 256,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        tokens = await runtime.prepare_tokens(parsed)
        if prefill_tokens is not None:
            # Synthetic token sequences exercise actual matrix shapes; no generated
            # text from these requests is exposed to users or billed.
            if len(tokens) < prefill_tokens:
                raise RuntimeError("Startup prompt is shorter than its prefill bucket")
            tokens = tokens[: prefill_tokens - 2] + tokens[-2:]
        # Warm the checkpoint path in an unreachable customer cache namespace too.
        parsed = replace(parsed, cache_namespace="0" * 64 if cache else None)
        parsed = replace(parsed, prefix_tokens=await runtime.prefix_length(parsed, tokens))
        stream = runtime.generate(parsed, tokens)
        try:
            finished = False
            async for event in stream:
                if event.get("type") in {"error", "fatal"}:
                    raise RuntimeError("DeepInfer startup warmup failed")
                finished = bool(event.get("finish_reason"))
            if not finished:
                raise RuntimeError("DeepInfer startup warmup did not complete")
        finally:
            await stream.aclose()

    # DeepGEMM selects different kernel layouts in 16-token M increments. Its
    # lazy CUDA compilation otherwise blocks live streams for seconds whenever
    # an unseen prompt shape arrives. The verified seed covers the full capacity;
    # exercise representative execution shapes, largest first for allocator reuse.
    # Missing or incompatible artifacts retain exhaustive shape coverage.
    lengths = (
        (PREFILL_TOKENS, 4096, 1024, 256, 16)
        if runtime.compiled_kernel_cache_verified
        else range(16, PREFILL_TOKENS + 1, 16)
    )
    for length in lengths:
        async with asyncio.TaskGroup() as group:
            for index in range(RANK_COUNT):
                group.create_task(request(index, 2048, RANK_COUNT, length))

    # Exercise representative decode batches up to the full two-rank capacity.
    for concurrency in [1, 2, 8, 16, 32, 64, 96, 128]:
        async with asyncio.TaskGroup() as group:
            for index in range(concurrency):
                group.create_task(request(index, 64 if index % 4 == 0 else 4, concurrency))

    # The first observation skips retention; the second captures and the third reuses.
    for _ in range(3):
        async with asyncio.TaskGroup() as group:
            for index in range(RANK_COUNT):
                group.create_task(request(index, 64, RANK_COUNT, cache=True))
