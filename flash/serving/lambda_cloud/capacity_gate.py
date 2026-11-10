"""Wait for configured Lambda hardware before changing the deployed serving app."""

import asyncio
import logging
import os
import time

from flash.serving.app.inference_provider import MODEL_POLICIES
from flash.serving.app.timeouts import CAPACITY_POLL_SECONDS

from .api import LambdaAPI
from .config import NoCapacityError, select_region

DEPLOY_CAPACITY_WAIT_SECONDS = 5 * 60 * 60
logger = logging.getLogger(__name__)


async def wait_for_capacity(api, models, *, clock=time.monotonic, sleep=asyncio.sleep):
    """Poll all required shapes together; never rent a VM or retry an ambiguous launch."""
    if not models:
        return
    started = clock()
    deadline = started + DEPLOY_CAPACITY_WAIT_SECONDS
    async with asyncio.timeout(DEPLOY_CAPACITY_WAIT_SECONDS):
        while clock() < deadline:
            catalog = await api.call("GET", "instance-types")
            missing = []
            for model in models:
                try:
                    select_region(catalog, model)
                except NoCapacityError:
                    missing.append(model)
            remaining = deadline - clock()
            if remaining <= 0:
                break
            if not missing:
                logger.warning("Lambda capacity available for all configured models; proceeding")
                return
            logger.warning(
                "Lambda capacity check: unavailable for %s; elapsed %.0f min, remaining %.0f min; "
                "checking again in %.0f min",
                ", ".join(missing),
                (clock() - started) / 60,
                remaining / 60,
                min(CAPACITY_POLL_SECONDS, remaining) / 60,
            )
            await sleep(min(CAPACITY_POLL_SECONDS, remaining))
    raise TimeoutError("Lambda deployment capacity unavailable after five hours")


async def main():
    models = [model for model, policy in MODEL_POLICIES.items() if policy.provider != "modal"]
    if not models:
        logger.warning("No Lambda models configured; skipping capacity wait")
        return
    key = os.environ.get("LAMBDA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("LAMBDA_API_KEY is required for Lambda deployment")
    api = LambdaAPI(key)
    try:
        await wait_for_capacity(api, models)
    finally:
        await api.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    asyncio.run(main())
