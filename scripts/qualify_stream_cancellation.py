"""Bounded CPU-only check of real Modal transport cancellation; no model/GPU."""

import asyncio
import json
import os
import time
import uuid

import modal

app = modal.App("flash-stream-cancellation-qualification")
image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("flash")


@app.cls(image=image, max_containers=1, timeout=120, scaledown_window=60)
@modal.concurrent(max_inputs=8)
class Replica:
    @modal.enter()
    def setup(self):
        self.engine = self
        self.closed = []
        self.aborted = []

    def _replica_identifier(self):
        return os.environ["MODAL_TASK_ID"]

    async def abort(self, generation_id):
        self.aborted.append(generation_id)

    async def _stream_generate(
        self, payload_dict, record_dict, expected_checkpoint, generation_id, *, pre_generate_check
    ):
        await pre_generate_check()
        try:
            yield {"type": "ready", "engine_replica_id": self._replica_identifier()}
            for index in range(payload_dict["count"]):
                await asyncio.sleep(0.1)
                yield {"type": "delta", "index": index}
            yield {"type": "final", "ok": True}
        finally:
            self.closed.append(generation_id)

    @modal.method()
    async def stream_generate_call(self, *args):
        from flash.serving.src.stream_channel.engine import stream_generate_call

        return await stream_generate_call(self, *args)

    @modal.method()
    def health(self):
        return {
            "replica": self._replica_identifier(),
            "closed": self.closed,
            "aborted": self.aborted,
        }


@app.local_entrypoint()
async def main(report_path: str = ""):
    from flash.serving.src.stream_channel.client import CancellableStreamChannel

    replica = Replica()
    before = await replica.health.remote.aio()

    def channel(generation_id, count):
        return CancellableStreamChannel(
            spawn_method=replica.stream_generate_call,
            payload_dict={"count": count},
            record_dict=None,
            expected_checkpoint=None,
            generation_id=generation_id,
            dispatch_deadline_unix=time.time() + 90,
            invocation_nonce=uuid.uuid4().hex,
        )

    cancelled = "fsgen-" + uuid.uuid4().hex
    sibling = "fsgen-" + uuid.uuid4().hex
    first = channel(cancelled, 800)
    second = channel(sibling, 60)

    async def collect():
        return [event async for event in second]

    sibling_task = asyncio.create_task(collect())
    ready = await anext(first)
    assert ready["engine_replica_id"] == before["replica"]
    await anext(first)
    started = time.monotonic()
    await first.aclose()
    cancellation_seconds = time.monotonic() - started
    events = await asyncio.wait_for(sibling_task, 40)
    assert events[-1] == {"type": "final", "ok": True}
    assert sum(e["type"] == "delta" for e in events) == 60
    # Cross the 30s cancellation watchdog that killed the original GPU replicas.
    await asyncio.sleep(35)
    after = await replica.health.remote.aio()
    assert after["replica"] == before["replica"], (before, after)
    assert set(after["closed"]) == {cancelled, sibling}, after
    assert after["aborted"] == [cancelled], after
    result = {
        "passed": True,
        "same_replica": True,
        "sibling_completed": True,
        "exact_generation_aborted": True,
        "cancellation_seconds": cancellation_seconds,
        "scope": "Real Modal CPU transport; synthetic engine, no GPU/vLLM qualification.",
    }
    if report_path:
        from pathlib import Path

        await asyncio.to_thread(Path(report_path).write_text, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
