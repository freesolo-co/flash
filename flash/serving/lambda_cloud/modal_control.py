"""Keep Lambda mutations serialized and credentials out of GPU workers."""

import os
import sys

import modal

from .api import LambdaAPI
from .config import LEDGER_NAME
from .fleet import Fleet


class Ledger:
    def __init__(self):
        self.store = modal.Dict.from_name(LEDGER_NAME, create_if_missing=True)

    async def get(self, key):
        return await self.store.get.aio(key)

    async def put(self, key, value):
        await self.store.put.aio(key, value)

    def items(self):
        return self.store.items.aio()


def register_control(app, secret, image):
    class LambdaFleet:
        @modal.enter()
        async def start(self):
            environment = os.environ["FLASH_LAMBDA_ENVIRONMENT"]
            self.api = LambdaAPI(os.environ["LAMBDA_API_KEY"])
            self.fleet = Fleet(self.api, Ledger(), environment)

        @modal.exit()
        async def close(self):
            await self.api.aclose()

        @modal.method()
        async def launch(self, owner, model, user_data):
            return await self.fleet.launch(owner, model, user_data)

        @modal.method()
        async def inspect(self, owner):
            return await self.fleet.inspect(owner)

        @modal.method()
        async def heartbeat(self, owner):
            return await self.fleet.heartbeat(owner)

        @modal.method()
        async def release(self, owner):
            return await self.fleet.release(owner)

        @modal.method()
        async def reap(self):
            return await self.fleet.reap()

    async def reap_lambda_workers():
        return await control().reap.remote.aio()

    module = sys.modules["flash.serving.app.modal_app"]
    LambdaFleet.__module__ = module.__name__
    LambdaFleet.__qualname__ = "LambdaFleet"
    module.LambdaFleet = LambdaFleet
    control = app.cls(
        image=image,
        secrets=[secret],
        min_containers=1,
        max_containers=1,
        nonpreemptible=True,
        timeout=300,
    )(modal.concurrent(max_inputs=1)(LambdaFleet))
    module.LambdaFleet = control
    reap_lambda_workers.__module__ = module.__name__
    reap_lambda_workers.__qualname__ = "reap_lambda_workers"
    module.reap_lambda_workers = reap_lambda_workers
    reaper = app.function(
        image=image,
        schedule=modal.Period(seconds=60),
        timeout=300,
        max_containers=1,
        nonpreemptible=True,
    )(reap_lambda_workers)
    return control, reaper
