"""Paid isolated CPU-router qualification, with no customer traffic or credentials.

Run: modal run scripts/benchmark_router_resources.py
Optionally reuse a serving image with ROUTER_BENCH_IMAGE_ID=im-... .
Defaults to a 4 GiB reservation with normal memory bursting, as in production.
Set ROUTER_BENCH_MEMORY_LIMIT_MIB=4096 for a strict 4 GiB stress test, or set both
ROUTER_BENCH_MEMORY_MIB and ROUTER_BENCH_MEMORY_LIMIT_MIB to 8192 for an 8 GiB cap.
Uses the real HTTP router, Modal dispatch, leased queues, SSE framing and usage
serialization. GPU inference, authorization and database persistence are synthetic;
this measures router capacity, not model throughput or billing reliability.
Both routers have CPU limits and max_containers=1. The driver and
synthetic engine run in separate containers. `modal run` tears everything down.
"""

import asyncio
import json
import os
import secrets
import time
from pathlib import Path

import modal

from flash.serving.app.modal_app import MAX_INPUTS, TARGET_INPUTS
from flash.serving.app.modal_app import image as serving_image

CANDIDATE_MEMORY_MIB = int(os.environ.get("ROUTER_BENCH_MEMORY_MIB", "4096"))
CANDIDATE_MEMORY_LIMIT_MIB = int(os.environ.get("ROUTER_BENCH_MEMORY_LIMIT_MIB", "0"))
MEMORY_BURST = CANDIDATE_MEMORY_LIMIT_MIB == 0
app = modal.App("flash-router-resource-qualification" + ("-burst" if MEMORY_BURST else ""))
image = (
    modal.Image.from_id(os.environ["ROUTER_BENCH_IMAGE_ID"]).add_local_python_source("flash")
    if os.environ.get("ROUTER_BENCH_IMAGE_ID")
    else serving_image
)
test_secret = modal.Secret.from_dict({"ROUTER_BENCH_KEY": secrets.token_urlsafe(32)})
MODELS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-0.8B")
OUTPUT_TOKENS = 256


@app.cls(
    image=image,
    cpu=8,
    memory=4096,
    region="us-east",
    max_containers=1,
    scaledown_window=600,
    timeout=600,
)
@modal.concurrent(max_inputs=256)
class SyntheticEngine:
    def _replica_identifier(self):
        return os.environ["MODAL_TASK_ID"]

    async def abort(self, _generation_id):
        pass

    @property
    def engine(self):
        return self

    async def _stream_generate(
        self, payload_dict, record_dict, expected_checkpoint, generation_id, *, pre_generate_check
    ):
        await pre_generate_check()
        start = time.monotonic()
        counts = {
            "request_id": generation_id,
            "prompt_tokens": 512,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "thinking": False,
        }
        yield {"type": "ready", **counts}
        # Four synthetic tokens per event, paced to 128 tokens/s before queue overhead.
        for _ in range(OUTPUT_TOKENS // 4):
            await asyncio.sleep(4 / 128)
            yield {"type": "delta", "text": " test" * 4}
        yield {
            **counts,
            "type": "final",
            "finish_reason": "length",
            "completion_tokens": OUTPUT_TOKENS,
            "inference_time_seconds": time.monotonic() - start,
        }

    @modal.method()
    async def stream_generate_call(self, *args, **kwargs):
        from flash.serving.src.stream_channel.engine import stream_generate_call

        return await stream_generate_call(self, *args, **kwargs)

    @modal.method()
    async def generate(self, payload, record, expected_checkpoint, generation_id):
        await asyncio.sleep(0.05)
        return {
            "request_id": generation_id,
            "text": " test" * OUTPUT_TOKENS,
            "finish_reason": "length",
            "prompt_tokens": 512,
            "completion_tokens": OUTPUT_TOKENS,
            "reasoning_tokens": 0,
            "thinking": False,
        }


def resource_snapshot():
    import resource

    root = Path("/sys/fs/cgroup")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    values = {
        "rss_peak_bytes": usage.ru_maxrss * 1024,
        "process_cpu_seconds": usage.ru_utime + usage.ru_stime,
    }
    for name in ("memory.current", "memory.peak", "memory.max", "cpu.stat", "cpu.max"):
        path = root / name
        if path.exists():
            values[name] = path.read_text().strip()
    values["container_id"] = os.environ.get("MODAL_TASK_ID")
    return values


def build_benchmark_router():
    import hmac

    from fastapi import HTTPException, Request

    from flash.serving.app import modal_app
    from flash.serving.src.accounting.usage_outbox import OfflineUsageStore
    from flash.serving.src.http.router import AdapterRouter, build_serving_app
    from flash.serving.src.io.schemas import AdapterRecord

    # Only this isolated process is patched. Preserve the production dispatch and
    # queue protocol, replacing GPU work with a separately scheduled CPU fixture.
    modal_app._engine_cls_for = lambda _model: lambda **_kwargs: SyntheticEngine()
    key = os.environ["ROUTER_BENCH_KEY"]

    async def authorize(token, _model, _scope):
        if not hmac.compare_digest(token, key):
            raise HTTPException(401, "invalid benchmark key")
        return "router-benchmark"

    async def admit(_org, **_kwargs):
        await asyncio.sleep(0.005)

    class SerializedUsageStore(OfflineUsageStore):
        enabled = True

        async def capture(self, event):
            json.dumps(event.rpc_payload())
            await asyncio.sleep(0.005)

        finalize = capture

    records = [
        AdapterRecord(model_id=m, repo_id=m, base_model=m, serve_base_model=True, thinking=False)
        for m in MODELS
    ]
    api = build_serving_app(
        modal_app._ModalEnginePool(),
        AdapterRouter(records),
        usage_store=SerializedUsageStore(),
        rate_limiter=admit,
        chat_authorizer=authorize,
    )

    @api.get("/benchmark/resources")
    async def resources(request: Request):
        await authorize(request.headers.get("authorization", "").removeprefix("Bearer "), "", {})
        return resource_snapshot()

    return api


@app.function(
    image=image,
    secrets=[test_secret],
    region="us-east",
    cpu=(8, 8),
    memory=(16384, 16384),
    max_containers=1,
    buffer_containers=0,
    scaledown_window=600,
    timeout=600,
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
@modal.asgi_app()
def baseline():
    return build_benchmark_router()


@app.function(
    image=image,
    secrets=[test_secret],
    region="us-east",
    cpu=(2, 2),
    memory=(
        CANDIDATE_MEMORY_MIB if MEMORY_BURST else (CANDIDATE_MEMORY_MIB, CANDIDATE_MEMORY_LIMIT_MIB)
    ),
    max_containers=1,
    buffer_containers=0,
    scaledown_window=600,
    timeout=600,
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
@modal.asgi_app()
def candidate():
    return build_benchmark_router()


def percentile(values, quantile):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * quantile))] if ordered else None


async def request_stream(client, url, body):
    start = time.monotonic()
    first = None
    content = []
    usage = None
    done = False
    try:
        async with client.stream("POST", url + "/v1/chat/completions", content=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    done = True
                    continue
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content")
                    if text:
                        first = first or time.monotonic() - start
                        content.append(text)
                usage = event.get("usage") or usage
        if not done or "".join(content) != " test" * OUTPUT_TOKENS:
            raise RuntimeError("incomplete or corrupt stream")
        if usage is None or usage["completion_tokens"] != OUTPUT_TOKENS:
            raise RuntimeError("missing or incorrect usage")
        return {"ttft": first, "elapsed": time.monotonic() - start}
    except Exception as exc:
        return {"error": str(exc), "elapsed": time.monotonic() - start}


async def request_completion(client, url, body):
    start = time.monotonic()
    try:
        response = await client.post(url + "/v1/chat/completions", content=body)
        response.raise_for_status()
        result = response.json()
        if result["choices"][0]["message"]["content"] != " test" * OUTPUT_TOKENS:
            raise RuntimeError("incomplete or corrupt response")
        if result["usage"]["completion_tokens"] != OUTPUT_TOKENS:
            raise RuntimeError("incorrect usage")
        return {"ttft": time.monotonic() - start, "elapsed": time.monotonic() - start}
    except Exception as exc:
        return {"error": str(exc), "elapsed": time.monotonic() - start}


@app.function(
    image=image, secrets=[test_secret], region="us-east", cpu=8, memory=8192, timeout=3600
)
async def drive(urls, repeats, cases):
    import httpx

    results = []
    headers = {
        "authorization": "Bearer " + os.environ["ROUTER_BENCH_KEY"],
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(
        headers=headers,
        timeout=180,
        limits=httpx.Limits(max_connections=160, max_keepalive_connections=160),
    ) as client:
        # A completed request warms the router and the synthetic engine before measurement.
        for url in urls.values():
            body = make_body(1024, 0)
            warm = await request_stream(client, url, body)
            if "error" in warm:
                raise RuntimeError(f"Warmup failed: {warm}")
        for repeat in range(repeats):
            for size, concurrency, streaming in cases:
                # Alternate order to reduce placement/time bias. Pre-serialize outside timing.
                bodies = [make_body(size, i, streaming) for i in range(len(MODELS))]
                request = request_stream if streaming else request_completion
                order = list(urls.items())[:: 1 if repeat % 2 == 0 else -1]
                for name, url in order:
                    before = (await client.get(url + "/benchmark/resources")).json()
                    start = time.monotonic()
                    rows = await asyncio.gather(
                        *(request(client, url, bodies[i % len(bodies)]) for i in range(concurrency))
                    )
                    elapsed = time.monotonic() - start
                    after = (await client.get(url + "/benchmark/resources")).json()
                    good = [r for r in rows if "error" not in r]
                    row = {
                        "router": name,
                        "repeat": repeat,
                        "prompt_bytes": size,
                        "concurrency": concurrency,
                        "streaming": streaming,
                        "elapsed": elapsed,
                        "successes": len(good),
                        "errors": [r for r in rows if "error" in r],
                        "synthetic_output_tokens_per_second": len(good) * OUTPUT_TOKENS / elapsed,
                        "ttft_p50": percentile([r["ttft"] for r in good], 0.5),
                        "ttft_p95": percentile([r["ttft"] for r in good], 0.95),
                        "latency_p95": percentile([r["elapsed"] for r in good], 0.95),
                        "before": before,
                        "after": after,
                    }
                    results.append(row)
                    print("BENCH", json.dumps(row), flush=True)
    return results


def make_body(size, index, streaming=True):
    return json.dumps(
        {
            "model": MODELS[index % len(MODELS)],
            "messages": [{"role": "user", "content": ("test " * (size // 5 + 1))[:size]}],
            "stream": streaming,
            "max_tokens": OUTPUT_TOKENS,
            **({"stream_options": {"include_usage": True}} if streaming else {}),
        }
    ).encode()


@app.local_entrypoint()
def main(repeats: int = 3, smoke: bool = False, output: str = "results/router-sizing/results.json"):
    print(
        "FUNCTIONS", json.dumps({"baseline": baseline.object_id, "candidate": candidate.object_id})
    )
    cases = (
        [(4096, 1, True), (4096, 1, False)]
        if smoke
        else [
            (4096, 48, True),
            (4096, 64, True),
            (1024 * 1024, 64, True),
            (8 * 1024 * 1024, 64, True),
            (23 * 1024 * 1024, 64, True),
            (1024 * 1024, 128, True),
            (8 * 1024 * 1024, 64, False),
        ]
    )
    results = drive.remote(
        {
            "8cpu-16gib": baseline.get_web_url(),
            f"2cpu-{CANDIDATE_MEMORY_MIB // 1024}gib" + ("-burst" if MEMORY_BURST else ""): (
                candidate.get_web_url()
            ),
        },
        1 if smoke else repeats,
        cases,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    if any(
        row["errors"] or row["before"]["container_id"] != row["after"]["container_id"]
        for row in results
    ):
        raise RuntimeError(
            f"Benchmark had request failures or container restarts; inspect {output}"
        )
