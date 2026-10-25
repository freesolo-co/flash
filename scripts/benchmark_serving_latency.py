"""Measure public SSE TTFT and per-stream decode speed, separately from aggregate throughput.

OPENAI_API_KEY=... python scripts/benchmark_serving_latency.py \
    --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --concurrency 4 --requests 96 \
    --output /tmp/serving-latency.json

Uses unique prompts: one quarter approximately 750 tokens, the rest approximately 45.
Use --prompt-file to benchmark longer source context; actual input counts come from usage.
Warm-up requests are optional and saved separately, never silently discarded. Run
matched baseline/candidate/candidate/baseline rounds on otherwise quiet deployments.
MODAL_KEY/MODAL_SECRET optionally protect private qualification routers.
"""

import argparse
import asyncio
import json
import math
import os
import time
import uuid
from pathlib import Path

import httpx


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def summarize(rows, elapsed):
    good = [row for row in rows if row["ok"]]
    rates = [row["decode_tokens_per_second"] for row in good]
    return {
        "requests": len(rows),
        "errors": len(rows) - len(good),
        "elapsed_seconds": elapsed,
        "aggregate_output_tokens_per_second": sum(r["usage"]["completion_tokens"] for r in good)
        / elapsed,
        "ttft_p50_seconds": percentile([r["ttft_seconds"] for r in good], 0.5),
        "ttft_p95_seconds": percentile([r["ttft_seconds"] for r in good], 0.95),
        "per_stream_decode_tokens_per_second_p50": percentile(rates, 0.5),
        "per_stream_decode_tokens_per_second_p10": percentile(rates, 0.1),
        "prompt_tokens_p50": percentile([r["usage"]["prompt_tokens"] for r in good], 0.5),
        "prompt_tokens_p95": percentile([r["usage"]["prompt_tokens"] for r in good], 0.95),
    }


async def request(client, args, index):
    identity = "latency-" + uuid.uuid4().hex
    context = args.prompt_context or (
        "Service requests share bounded memory and must retain accurate usage records. "
        * (60 if index % 4 == 0 else 1)
    )
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": f"Case {identity}. {context}"
                "Write a numbered operational checklist with twenty detailed items.",
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.monotonic()
    row = {
        "correlation_id": identity,
        "started_unix": time.time(),
        "ok": False,
        "long_prompt": bool(args.prompt_context) or index % 4 == 0,
    }
    try:
        async with client.stream(
            "POST",
            args.url + "/v1/chat/completions",
            json=payload,
            headers={"X-Correlation-ID": identity},
        ) as response:
            row["status"] = response.status_code
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    row["done"] = True
                    continue
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError("stream returned an error")
                row["completion_id"] = event.get("id", row.get("completion_id"))
                if event.get("usage"):
                    row["usage"] = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                    ):
                        now = time.monotonic() - started
                        row.setdefault("ttft_seconds", now)
                        row["last_token_seconds"] = now
                    if choice.get("finish_reason"):
                        row["finish_reason"] = choice["finish_reason"]
        tokens = row.get("usage", {}).get("completion_tokens", 0)
        interval = row.get("last_token_seconds", 0) - row.get("ttft_seconds", 0)
        if not (row.get("done") and row.get("finish_reason") and tokens > 1 and interval > 0):
            raise RuntimeError("incomplete stream or insufficient output to measure decode speed")
        # Client-observed TPOT: excludes prefill and final accounting/network tail.
        # Native timing should also be captured when qualifying changes to buffering.
        row["decode_tokens_per_second"] = (tokens - 1) / interval
        row["ok"] = True
    except (httpx.HTTPError, RuntimeError, ValueError, KeyError) as exc:
        row["error"] = type(exc).__name__
    row["elapsed_seconds"] = time.monotonic() - started
    return row


async def run(args):
    origin = httpx.URL(args.url)

    async def same_origin(request):
        if (request.url.scheme, request.url.host, request.url.port) != (
            origin.scheme,
            origin.host,
            origin.port,
        ):
            raise RuntimeError("Refusing to forward credentials to a different origin")

    headers = {"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    if os.environ.get("MODAL_KEY"):
        headers.update(
            {"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"]}
        )
    async with httpx.AsyncClient(
        headers=headers,
        timeout=900,
        follow_redirects=True,
        max_redirects=32,
        event_hooks={"request": [same_origin]},
        limits=httpx.Limits(
            max_connections=args.concurrency + 2, max_keepalive_connections=args.concurrency
        ),
    ) as client:

        async def batch(count):
            numbers, rows = iter(range(count)), []

            async def worker():
                for index in numbers:
                    row = await request(client, args, index)
                    rows.append(row)

            started = time.monotonic()
            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
            return rows, time.monotonic() - started

        warmup, _ = await batch(args.warmup)
        rows, elapsed = await batch(args.requests)
    result = {
        "url": args.url,
        "model": args.model,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "warmup": warmup,
        "requests": rows,
        "summary": summarize(rows, elapsed),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    return all(row["ok"] for row in [*warmup, *rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://serve.freesolo.co")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--prompt-file", type=Path, help="UTF-8 source context; never stored in results"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.prompt_context = args.prompt_file.read_text() if args.prompt_file else None
    args.url = args.url.rstrip("/")
    if min(args.concurrency, args.requests) < 1 or args.max_tokens < 2 or args.warmup < 0:
        parser.error(
            "positive concurrency/requests, at least two output tokens, and nonnegative warmup required"
        )
    if httpx.URL(args.url).scheme != "https":
        parser.error("use an HTTPS endpoint")
    raise SystemExit(0 if asyncio.run(run(args)) else 1)


if __name__ == "__main__":
    main()
