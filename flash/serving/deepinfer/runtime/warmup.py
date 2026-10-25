"""Exercise prefill/decode before a newly allocated replica accepts customer traffic."""

import asyncio
from dataclasses import replace

from flash.serving.deepinfer.contract.config import (
    GPU_COUNT,
    MODEL_ID,
    PREFILL_TOKENS,
    SEQUENCES_PER_RANK,
)
from flash.serving.deepinfer.contract.protocol import parse_request


async def warmup(runtime):
    async def request(index, repetitions, concurrency, prefill_tokens=None):
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
        parsed = replace(parsed, cache_namespace="0" * 64)
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
    # an unseen prompt shape arrives. Cover the full chunked-prefill capacity.
    for length in range(16, PREFILL_TOKENS + 1, 16):
        async with asyncio.TaskGroup() as group:
            for index in range(GPU_COUNT):
                group.create_task(request(index, 512, GPU_COUNT, length))

    # Every per-rank batch size has its own native kernels. Warming only the
    # smallest and largest batches still makes ordinary middle-sized bursts compile.
    for concurrency in [1, *(GPU_COUNT * n for n in range(1, SEQUENCES_PER_RANK + 1))]:
        async with asyncio.TaskGroup() as group:
            for index in range(concurrency):
                group.create_task(request(index, 64 if index % 4 == 0 else 4, concurrency))
