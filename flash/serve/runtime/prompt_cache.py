"""Event-loop-owned prompt token caching shared by hosted and packaged engines."""

import asyncio
from collections import OrderedDict
from collections.abc import Callable, Hashable


async def cached_prompt_tokens(
    cache: OrderedDict | None,
    key: Hashable | None,
    tokenize: Callable[[], list[int]],
    *,
    max_entries: int,
) -> list[int]:
    """Keep hits inline, offload misses, and never expose mutable cached token IDs."""
    enabled = cache is not None and key is not None and max_entries > 0
    if enabled:
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return list(cached)
    tokens = await asyncio.to_thread(tokenize)
    if enabled:
        cache[key] = tuple(tokens)
        cache.move_to_end(key)
        while len(cache) > max_entries:
            cache.popitem(last=False)
    return tokens
