"""Watch configured Lambda hardware alongside a serving deployment.

Every hosted model falls back to Modal when Lambda has no capacity, so the deployment does not wait
on this watch. It reports which shapes Lambda offered during the window and warns (without failing
the workflow) about any model that never had capacity.
"""

import asyncio
import logging
import os
import time

from flash.serving.app.inference_provider import lambda_models
from flash.serving.app.timeouts import CAPACITY_POLL_SECONDS

from .api import LambdaAPI
from .config import NoCapacityError, select_region

DEPLOY_CAPACITY_WAIT_SECONDS = 5 * 60 * 60
logger = logging.getLogger(__name__)


async def wait_for_capacity(api, models, *, clock=time.monotonic, sleep=asyncio.sleep):
    """Poll every five minutes until each model has been seen with capacity; never rent a VM.

    Returns the models Lambda never offered within the window (empty when all were seen). Shapes
    rarely free up together, so a model counts once any check found capacity for it.
    """
    pending = list(models)
    if not pending:
        return []
    started = clock()
    deadline = started + DEPLOY_CAPACITY_WAIT_SECONDS
    try:
        async with asyncio.timeout(DEPLOY_CAPACITY_WAIT_SECONDS):
            while clock() < deadline:
                catalog = await api.call("GET", "instance-types")
                still_missing = []
                for model in pending:
                    try:
                        select_region(catalog, model)
                    except NoCapacityError:
                        still_missing.append(model)
                for model in pending:
                    if model not in still_missing:
                        logger.warning("Lambda capacity seen for %s", model)
                pending = still_missing
                if not pending:
                    logger.warning("Lambda capacity seen for all configured models")
                    return []
                remaining = deadline - clock()
                if remaining <= 0:
                    break
                logger.warning(
                    "Lambda capacity check: not yet seen for %s; elapsed %.0f min, "
                    "remaining %.0f min; checking again in %.0f min",
                    ", ".join(pending),
                    (clock() - started) / 60,
                    remaining / 60,
                    min(CAPACITY_POLL_SECONDS, remaining) / 60,
                )
                await sleep(min(CAPACITY_POLL_SECONDS, remaining))
    except TimeoutError:
        # A stalled catalog call reached the overall bound; report what is still unseen.
        pass
    return pending


def _report_unavailable(models):
    message = (
        "Lambda capacity was not available within five hours for "
        f"{', '.join(models)}; these models serve on the Modal fallback"
    )
    logger.warning(message)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning title=Lambda capacity unavailable::{message}", flush=True)


async def main():
    models = lambda_models()
    key = os.environ.get("LAMBDA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("LAMBDA_API_KEY is required for Lambda deployment")
    api = LambdaAPI(key)
    try:
        unavailable = await wait_for_capacity(api, models)
    finally:
        await api.aclose()
    if unavailable:
        _report_unavailable(unavailable)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    asyncio.run(main())
