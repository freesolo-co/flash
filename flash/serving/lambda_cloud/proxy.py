"""CPU replica lifecycle: Lambda owns the GPU, Modal retains admission and autoscaling."""

import asyncio
import contextlib
import os
import uuid

from flash.serving.app.timeouts import STARTUP_TIMEOUT_SECONDS

from .bootstrap import identity, user_data
from .config import HEARTBEAT_SECONDS
from .transport import WorkerError, WorkerTransport


class LambdaEngineProxy:
    async def _load(self):
        import modal

        self._owner = uuid.uuid4().hex
        self._fleet = modal.Cls.from_name("freesolo-lora-serving", "LambdaFleet")()
        self._transport = None
        self._heartbeat_task = None
        self._health_cache = {"ok": False, "base_model": self.base_model}
        tls = await asyncio.to_thread(identity)
        revision = os.environ["FREESOLO_DEPLOYMENT_SHA"]
        bootstrap = user_data(self.base_model, revision, os.environ, tls)
        try:
            async with asyncio.timeout(STARTUP_TIMEOUT_SECONDS):
                await self._fleet.launch.remote.aio(self._owner, self.base_model, bootstrap)
                self._heartbeat_task = asyncio.create_task(self._heartbeat())
                await self._wait_ready(tls, revision)
                self.engine = self._transport  # stream-channel cancellation calls engine.abort
        except BaseException:
            await self._shutdown()
            raise

    async def _wait_ready(self, tls, revision):
        while True:
            instance = await self._fleet.inspect.remote.aio(self._owner)
            if instance["status"] in {"terminated", "terminating", "unhealthy"}:
                raise RuntimeError("Lambda worker failed during startup")
            if instance.get("ip") and self._transport is None:
                self._transport = WorkerTransport(instance["ip"], tls)
            if self._transport is not None:
                try:
                    async with asyncio.timeout(10):
                        health = await self._transport.call("/health")
                except (WorkerError, TimeoutError):
                    health = {}
                if health.get("ok") is True:
                    if (
                        health.get("base_model") != self.base_model
                        or health.get("deployment_sha") != revision
                    ):
                        raise RuntimeError("Lambda worker readiness identity mismatch")
                    self._health_cache = health
                    return
            await asyncio.sleep(15)

    async def _heartbeat(self):
        while True:
            try:
                await self._fleet.heartbeat.remote.aio(self._owner)
                if self._transport is not None:
                    async with asyncio.timeout(10):
                        health = await self._transport.call("/health")
                    self._health_cache = health
            except Exception:
                self._health_cache = {"ok": False, "base_model": self.base_model}
            await asyncio.sleep(HEARTBEAT_SECONDS)

    async def _shutdown(self):
        task = getattr(self, "_heartbeat_task", None)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        transport = getattr(self, "_transport", None)
        if transport is not None:
            await transport.aclose()
        fleet = getattr(self, "_fleet", None)
        if fleet is not None:
            # If cleanup loses its connection, the durable lease/reaper still owns the VM.
            with contextlib.suppress(Exception):
                async with asyncio.timeout(60):
                    await fleet.release.remote.aio(self._owner)

    def _replica_identifier(self):
        return self._owner

    async def _health(self):
        async with asyncio.timeout(10):
            self._health_cache = await self._transport.call("/health")
        return self._health_cache

    async def _register(self, record, deployment_generation=None):
        return await self._transport.call(
            "/register", {"record": record, "deployment_generation": deployment_generation}
        )

    async def _unregister(self, org_id, model_id, expected_generation=None):
        return await self._transport.call(
            "/unregister",
            {"org_id": org_id, "model_id": model_id, "expected_generation": expected_generation},
        )

    @staticmethod
    def _body(payload, record, expected_checkpoint, generation_id):
        return {
            "payload": payload,
            "record": record,
            "expected_checkpoint": expected_checkpoint,
            "generation_id": generation_id,
        }

    async def _generate(self, payload, record=None, expected_checkpoint=None, generation_id=None):
        return await self._transport.call(
            "/generate", self._body(payload, record, expected_checkpoint, generation_id)
        )

    async def _stream_generate(
        self,
        payload,
        record=None,
        expected_checkpoint=None,
        generation_id=None,
        *,
        pre_generate_check=None,
    ):
        if pre_generate_check is not None:
            await pre_generate_check()
        stream = self._transport.stream(
            self._body(payload, record, expected_checkpoint, generation_id)
        )
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()
