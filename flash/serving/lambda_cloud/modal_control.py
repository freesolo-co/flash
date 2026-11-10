"""Keep Lambda mutations serialized and credentials out of GPU workers."""

import os
import sys
import time

import modal

from .api import LambdaAPI
from .config import LEDGER_NAME, NoCapacityError, select_region
from .fleet import Fleet


class Ledger:
    def __init__(self, name=LEDGER_NAME):
        self.store = modal.Dict.from_name(name, create_if_missing=True)

    async def get(self, key):
        return await self.store.get.aio(key)

    async def put(self, key, value):
        await self.store.put.aio(key, value)

    async def delete(self, key):
        await self.store.pop.aio(key, None)

    def items(self):
        return self.store.items.aio()


def register_control(app, secret, image, module_name):
    class LambdaFleet:
        @modal.enter()
        async def start(self):
            environment = os.environ["FLASH_LAMBDA_ENVIRONMENT"]
            self.api = LambdaAPI(os.environ["LAMBDA_API_KEY"])
            self.fleet = Fleet(self.api, Ledger(), environment)
            from .pool_capacity import SharedCapacity, modal_pools

            self.capacity = SharedCapacity(
                self.fleet,
                modal_pools(environment),
                Ledger(LEDGER_NAME + "-pools"),
                os.environ.get("FREESOLO_DEPLOYMENT_SHA", ""),
            )
            self.fleet.shared_capacity = self.capacity

        @modal.exit()
        async def close(self):
            await self.api.aclose()

        @modal.method()
        async def has_capacity(self, model):
            if time.monotonic() >= getattr(self, "_catalog_expires", 0):
                self._catalog = await self.api.call("GET", "instance-types")
                self._catalog_expires = time.monotonic() + 30
            try:
                select_region(self._catalog, model)
            except NoCapacityError:
                return False
            return True

        @modal.method()
        async def launch(self, owner, model, user_data, revision):
            return await self.fleet.launch(owner, model, user_data, revision)

        @modal.method()
        async def inspect(self, owner):
            return await self.fleet.inspect(owner)

        @modal.method()
        async def heartbeat(self, owner, health=None):
            return await self.fleet.heartbeat(owner, health)

        @modal.method()
        async def activate(self, revision):
            return await self.capacity.activate(revision)

        @modal.method()
        async def refresh_capacity(self, model):
            if not await self.capacity.active():
                raise RuntimeError("Cannot refresh a superseded capacity controller")
            await self.capacity.reconcile_model(model)

        @modal.method()
        async def choose(self, model):
            return await self.capacity.choose(model)

        @modal.method()
        async def release(self, owner):
            return await self.fleet.release(owner)

        @modal.method()
        async def reap(self):
            result = await self.fleet.reap()
            await self.capacity.reconcile()
            return result

    async def reap_lambda_workers():
        return await control().reap.remote.aio()

    async def choose_provider(model):
        # Routing reads leases independently of the serialized VM mutation queue.
        # A slow cloud launch must not hold incoming requests behind it.
        from .pool_capacity import SharedCapacity, modal_pools

        environment = os.environ["FLASH_LAMBDA_ENVIRONMENT"]
        fleet = Fleet(None, Ledger(), environment)
        capacity = SharedCapacity(
            fleet,
            modal_pools(environment),
            Ledger(LEDGER_NAME + "-pools"),
            "",
            refresh=control().refresh_capacity.remote.aio,
        )
        return await capacity.choose(model)

    module = sys.modules[module_name]
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
    choose_provider.__module__ = module.__name__
    choose_provider.__qualname__ = "choose_provider"
    module.choose_provider = choose_provider
    module.choose_provider = app.function(
        image=image,
        secrets=[secret],
        timeout=30,
        min_containers=1,
        max_containers=2,
    )(modal.concurrent(max_inputs=32)(choose_provider))
    return control, reaper
