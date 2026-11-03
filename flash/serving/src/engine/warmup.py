"""Exercise representative Qwen prefill/decode paths before Modal admits traffic."""

import asyncio
import uuid


async def warmup(engine, tokenizer, max_sequences):
    from vllm import SamplingParams

    tokens = await asyncio.to_thread(
        tokenizer.encode,
        "Write a detailed numbered checklist. " * 2048,
        add_special_tokens=False,
    )

    async def request(index):
        request_id = "startup-" + uuid.uuid4().hex
        length = (32, 768, 2048, 4096)[index % 4]
        prompt = {"prompt_token_ids": tokens[:length], "cache_salt": request_id}
        stream = engine.generate(
            prompt,
            SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
            request_id,
        )
        finished = False
        try:
            async for output in stream:
                finished = output.finished
            if not finished:
                raise RuntimeError("Qwen startup generation did not finish")
        finally:
            await stream.aclose()
            if not finished:
                await engine.abort(request_id)

    # Cover intermediate decode batches too: jumping straight from four to a larger
    # cap leaves the first eight/sixteen-request burst paying for lazy kernel compilation.
    batch_sizes = [1]
    while batch_sizes[-1] < max_sequences:
        batch_sizes.append(min(max_sequences, batch_sizes[-1] * 2))
    for concurrency in batch_sizes:
        async with asyncio.TaskGroup() as group:
            for index in range(concurrency):
                group.create_task(request(index))
