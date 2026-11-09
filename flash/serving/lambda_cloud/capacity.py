"""Wait outside the serialized fleet controller, retrying only definite capacity misses."""

import asyncio
import logging
import time

from flash.serving.app.timeouts import CAPACITY_POLL_SECONDS, CAPACITY_WAIT_SECONDS

from .config import NoCapacityError

logger = logging.getLogger(__name__)


async def launch_with_capacity_wait(launch, *args, clock=time.monotonic, sleep=asyncio.sleep):
    deadline = clock() + CAPACITY_WAIT_SECONDS
    # Bound remote calls as well as sleeps; no launch is started after the deadline.
    async with asyncio.timeout(CAPACITY_WAIT_SECONDS):
        while True:
            try:
                return await launch(*args)
            except NoCapacityError:
                remaining = deadline - clock()
                if remaining <= 0:
                    raise TimeoutError("Lambda capacity unavailable after one hour") from None
                logger.info("Lambda capacity unavailable; checking again in five minutes")
                await sleep(min(CAPACITY_POLL_SECONDS, remaining))
                if clock() >= deadline:
                    raise TimeoutError("Lambda capacity unavailable after one hour") from None
