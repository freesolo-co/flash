"""Nonblocking local control with one ordered executor per GPU session.

The executor owns complete operations, including both passes of a custom loss. Cancelling
an await never cancels a submitted GPU mutation or allows another operation to overtake it.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class AsyncTrainingClient:
    def __init__(self, client):
        self._client = client
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-training")
        self._closing = False

    @property
    def step(self):
        return self._client._state["step"]

    @property
    def restored_state(self):
        return self._client._state.get("restored_state", {})

    @property
    def config(self):
        return self._client.config

    @property
    def model_version(self):
        return self._client.model_version

    @property
    def session_id(self):
        return self._client.session_id

    async def _call(self, method, *args, **kwargs):
        if self._closing:
            raise RuntimeError("custom client is closing")
        future = self._executor.submit(partial(getattr(self._client, method), *args, **kwargs))
        # shield preserves queued operations if their caller is cancelled. the sync protocol
        # retains ambiguous receipts, so subsequent queued calls fail closed until reconciled.
        return await asyncio.shield(asyncio.wrap_future(future))

    async def encode(self, text):
        return await self._call("encode", text)

    async def decode(self, tokens):
        return await self._call("decode", tokens)

    async def sample(self, prompt, **kwargs):
        return await self._call("sample", prompt, **kwargs)

    async def forward(self, data, **kwargs):
        return await self._call("forward", data, **kwargs)

    async def forward_backward(self, data, **kwargs):
        return await self._call("forward_backward", data, **kwargs)

    async def forward_backward_custom(self, data, loss_fn):
        return await self._call("forward_backward_custom", data, loss_fn)

    async def optim_step(self, **kwargs):
        return await self._call("optim_step", **kwargs)

    async def zero_grad(self):
        return await self._call("zero_grad")

    async def save_checkpoint(self, name, **kwargs):
        return await self._call("save_checkpoint", name, **kwargs)

    async def load_checkpoint(self, name):
        return await self._call("load_checkpoint", name)

    async def download_checkpoint(self, name, destination):
        return await self._call("download_checkpoint", name, destination)

    async def status(self):
        return await self._call("status")

    async def logs(self, after=0):
        return await self._call("logs", after)

    async def retry_pending(self):
        return await self._call("retry_pending")

    async def close(self):
        if self._closing:
            return
        self._closing = True
        future = self._executor.submit(self._client.close)
        try:
            await asyncio.shield(asyncio.wrap_future(future))
        finally:
            self._executor.shutdown(wait=False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
