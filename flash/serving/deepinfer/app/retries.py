"""Retry only requests proven not to have been accepted; never replay a stream."""

import asyncio

import httpx

from flash.serving.deepinfer.contract.errors import DeepInferError

ATTEMPTS = 3
RETRY_SECONDS = 10


async def open_generation(client, targets, headers, body, deadline, *, start=0):
    loop = asyncio.get_running_loop()
    retry_deadline = min(deadline, loop.time() + RETRY_SECONDS)
    last = None
    failed = set()
    for attempt in range(ATTEMPTS):
        if attempt:
            remaining = retry_deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.25 * 2 ** (attempt - 1), remaining))
        try:
            endpoints = await asyncio.wait_for(asyncio.to_thread(targets), timeout=5)
        except Exception:
            # Discovery has not sent generation work and is also safe to retry.
            last = DeepInferError(503, "DeepInfer endpoint discovery failed")
            continue
        if not endpoints:
            last = DeepInferError(503, "DeepInfer has no ready replica")
            continue
        # Rotation is stable within an attempt and explicitly prefers a different
        # ready replica after failure. With one replica this is a bounded retry.
        candidates = endpoints[start % len(endpoints) :] + endpoints[: start % len(endpoints)]
        endpoint = next((url for url in candidates if url not in failed), candidates[0])
        failed.add(endpoint)
        request = client.build_request(
            "POST", endpoint + "/internal/generate", headers=headers, json=body
        )
        remaining = retry_deadline - loop.time()
        if remaining <= 0:
            break
        request.extensions["timeout"].update(connect=min(3, remaining), pool=min(3, remaining))
        try:
            response = await client.send(request, stream=True, follow_redirects=False)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            # No HTTP request was sent. Read/write/protocol errors are ambiguous
            # and intentionally propagate rather than duplicating native work.
            last = exc
            continue
        if response.next_request is not None:
            # A 303 may mean the POST was accepted and this is its result URL.
            # Failure connecting to that GET must NEVER replay the original POST.
            # The client's redirect hook has already validated the same origin.
            followup = response.next_request
            await response.aclose()
            return await client.send(followup, stream=True)
        if (
            response.status_code in {429, 503}
            and response.headers.get("X-DeepInfer-Retryable") == "not-started"
        ):
            code = response.status_code
            await response.aclose()
            last = DeepInferError(code, "DeepInfer replicas rejected admission")
            continue
        return response
    if last is not None:
        raise last
    raise DeepInferError(503, "DeepInfer admission retry budget expired")
