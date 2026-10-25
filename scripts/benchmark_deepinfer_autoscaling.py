"""Measure a deployed private DeepInfer pool; never changes its autoscaler settings.

Run with MODAL_KEY and MODAL_SECRET set to environment-scoped Proxy Token values:
  uv run python scripts/benchmark_deepinfer_autoscaling.py --env dev \
      --concurrency 10 --seconds 600 --output /tmp/deepinfer-load.json

The output includes per-request timing and replica IDs plus five-second pool samples.
Use a quiet deployment and separately record its settings for each comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import itertools
import json
import math
import os
import time
import uuid
from pathlib import Path

import httpx
import modal

from flash.serving.deepinfer.contract.config import APP_NAME, MODEL_ID


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def summarize(requests, elapsed):
    completed = [row for row in requests if row.get("ok")]
    return {
        "requests": len(requests),
        "errors": len(requests) - len(completed),
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": sum(row["tokens"] for row in completed) / elapsed,
        "ttft_p50_seconds": percentile([row["ttft"] for row in completed], 0.5),
        "ttft_p95_seconds": percentile([row["ttft"] for row in completed], 0.95),
        "latency_p95_seconds": percentile([row["elapsed"] for row in completed], 0.95),
        "replicas": sorted({row["replica"] for row in completed}),
    }


async def generate(client, url, index, max_tokens):
    started = time.monotonic()
    # Unique cold requests exercise prefill as well as decoding, without claiming a
    # production cache-hit distribution. Every fourth request has a longer prompt.
    request_id = uuid.uuid4().hex
    context = "The service handles concurrent requests with bounded memory. " * (
        200 if index % 4 == 0 else 1
    )
    body = {
        "request_id": request_id,
        "chat": {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": f"Case {request_id}. {context}\nWrite a detailed 100-item numbered guide to operating this service. Explain each item in a full paragraph.",
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    row = {"started": time.time(), "long_prompt": index % 4 == 0, "ok": False}
    try:
        async with client.stream("POST", url + "/internal/generate", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "ready":
                    row["ttft"] = time.monotonic() - started
                elif event["type"] == "error":
                    raise RuntimeError("native_error")
                elif event["type"] == "final":
                    if "ttft" not in row or event["completion_tokens"] <= 0:
                        raise RuntimeError("invalid_final_event")
                    row.update(
                        ok=True,
                        tokens=event["completion_tokens"],
                        replica=event["engine_replica_id"],
                    )
        if not row["ok"]:
            raise RuntimeError("stream_ended_without_final")
    except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as exc:
        row["ok"] = False
        row["error"] = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):
            row["status"] = exc.response.status_code
    row["elapsed"] = time.monotonic() - started
    return row


async def run(args):
    function = modal.Function.from_name(args.app, "serve", environment_name=args.env)
    url = await function.get_web_url.aio()
    if not url:
        raise RuntimeError("The function has no web endpoint")
    origin = httpx.URL(url)

    async def same_origin(request):
        if (request.url.scheme, request.url.host, request.url.port) != (
            origin.scheme,
            origin.host,
            origin.port,
        ):
            raise RuntimeError("Refusing to send Proxy Tokens to another origin")

    headers = {"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"]}
    settings = dataclasses.asdict(await function.autoscaler.settings.aio())
    requests, samples = [], []
    request_numbers = itertools.count()
    started = time.monotonic()
    done = asyncio.Event()

    async def sample():
        while not done.is_set():
            stats = await function.get_current_stats.aio()
            samples.append({"elapsed": time.monotonic() - started, **dataclasses.asdict(stats)})
            args.output.write_text(json.dumps({"requests": requests, "samples": samples}))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(done.wait(), timeout=5)

    async with httpx.AsyncClient(
        headers=headers,
        timeout=900,
        follow_redirects=True,
        max_redirects=32,
        limits=httpx.Limits(
            max_connections=args.concurrency + 2, max_keepalive_connections=args.concurrency
        ),
        event_hooks={"request": [same_origin]},
    ) as client:

        async def worker():
            while time.monotonic() - started < args.seconds:
                row = await generate(client, url, next(request_numbers), args.max_tokens)
                requests.append(row)
                if not row["ok"]:
                    await asyncio.sleep(1)

        monitor = asyncio.create_task(sample())
        try:
            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        finally:
            done.set()
            await monitor
    elapsed = time.monotonic() - started
    result = {
        "app": args.app,
        "environment": args.env,
        "concurrency": args.concurrency,
        "seconds": args.seconds,
        "max_tokens": args.max_tokens,
        "autoscaler": settings,
        "summary": summarize(requests, elapsed),
        "requests": requests,
        "samples": samples,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    return result["summary"]["errors"] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--app", default=APP_NAME)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.concurrency, args.seconds, args.max_tokens) < 1:
        parser.error("concurrency, seconds, and max-tokens must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raise SystemExit(0 if asyncio.run(run(args)) else 1)


if __name__ == "__main__":
    main()
