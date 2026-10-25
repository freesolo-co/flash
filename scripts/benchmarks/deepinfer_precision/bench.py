import asyncio
import hashlib
import itertools
import json
import os
import signal
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path("/results")
ROOT.mkdir(exist_ok=True)
URL = "http://127.0.0.1:8000"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
SHORT = [
    {
        "role": "user",
        "content": "Explain how a B-tree index works. Include a detailed worked example and continue for at least 300 words.",
    }
]
SYSTEM = "You are a careful technical assistant.\n" + "".join(
    f"Reference {i}: Index pages store sorted keys and references. Splits preserve ordering. Transactions protect updates.\n"
    for i in range(240)
)
LONG = [
    {"role": "system", "content": SYSTEM},
    {
        "role": "user",
        "content": "Explain the B-tree reference with a detailed example and at least 300 words.",
    },
]


def percentile(values, p):
    return sorted(values)[min(len(values) - 1, int(len(values) * p))]


async def one(client, messages, namespace, max_tokens=256, started_event=None):
    started = time.monotonic()
    first = None
    first_count = None
    last = None
    events = []
    text = ""
    rid = uuid.uuid4().hex
    body = {
        "request_id": rid,
        "cache_namespace": namespace,
        "chat": {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    async with client.stream("POST", URL + "/internal/generate", json=body) as response:
        if response.status_code != 200:
            raise RuntimeError((response.status_code, (await response.aread()).decode()))
        async for line in response.aiter_lines():
            if not line:
                continue
            event = json.loads(line)
            now = time.monotonic()
            if event["type"] == "error":
                raise RuntimeError("Native error")
            if event["type"] == "delta" and event["text"]:
                if first is None:
                    first = now
                    first_count = event["completion_tokens"]
                    if started_event is not None:
                        started_event.set()
                last = now
                text += event["text"]
            events.append(
                {
                    "elapsed_s": now - started,
                    "type": event["type"],
                    "completion_tokens": event.get("completion_tokens", 0),
                }
            )
            if event["type"] == "final":
                final = event
    elapsed = time.monotonic() - started
    assert events[-1]["type"] == "final"
    assert final["text"] == text
    return {
        "latency_s": elapsed,
        "ttft_s": first - started if first else None,
        "events": len(events),
        "output_tokens": final["completion_tokens"],
        "prompt_tokens": final["prompt_tokens"],
        "cached_tokens": final["cached_tokens"],
        "cached_reported": final["cached_tokens_reported"],
        "decode_tps": (final["completion_tokens"] - first_count) / (last - first)
        if last > first
        else None,
        "end_to_end_tps": final["completion_tokens"] / elapsed,
        "first_event_tokens": first_count,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "finish": final["finish_reason"],
        "timeline": events,
    }


async def wave(client, kind, messages, mode, concurrency, repeat, variant, path):
    warm = hashlib.sha256(("ab-warm-" + kind).encode()).hexdigest()
    namespaces = [
        hashlib.sha256(f"ab-cold-{kind}-{repeat}-{concurrency}-{i}".encode()).hexdigest()
        if mode == "cold"
        else warm
        for i in range(concurrency)
    ]
    start = time.monotonic()
    values = await asyncio.gather(
        *(one(client, messages, n) for n in namespaces), return_exceptions=True
    )
    wall = time.monotonic() - start
    good = [v for v in values if isinstance(v, dict)]
    row = {
        "variant": variant,
        "workload": kind,
        "cache": mode,
        "concurrency": concurrency,
        "repeat": repeat,
        "wall_s": wall,
        "requests": good,
        "errors": [repr(v) for v in values if not isinstance(v, dict)],
    }
    row["aggregate_output_tps"] = sum(v["output_tokens"] for v in good) / wall
    for field in ["ttft_s", "decode_tps", "end_to_end_tps", "latency_s"]:
        vals = [v[field] for v in good if v[field] is not None]
        row[field + "_p50"] = statistics.median(vals) if vals else None
        row[field + "_p95"] = percentile(vals, 0.95) if vals else None
    row["cached_input_fraction"] = (
        sum(v["cached_tokens"] for v in good) / sum(v["prompt_tokens"] for v in good)
        if good
        else None
    )
    with (path / "measurements.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print("MEASURE", json.dumps({k: v for k, v in row.items() if k != "requests"}), flush=True)
    assert not row["errors"], row["errors"]
    if mode == "cold":
        assert all(v["cached_tokens"] == 0 for v in good)
    return row


async def quality(client, path):
    cases = []
    for a, b in [(17, 23), (121, 19), (43, 67), (89, 31), (27, 44), (53, 29), (101, 103), (73, 41)]:
        cases.append((f"Calculate {a} * {b}. Return only the integer answer.", str(a * b)))
    for index in range(8):
        target = index * 29 + 13
        context = "\n".join(f"Record {i}: code={1000 + i * 7}." for i in range(240))
        cases.append(
            (
                context + f"\nWhat is the code for Record {target}? Return only the integer code.",
                str(1000 + target * 7),
            )
        )
    for code, answer in [
        ("sum(range(10))", "45"),
        ("len(set([1,2,2,3,3,3]))", "3"),
        ('int("101101",2)', "45"),
        ("sum(x*x for x in range(5))", "30"),
        ('len("banana".replace("a",""))', "3"),
        ("max([7,19,3,11])-min([7,19,3,11])", "16"),
        ("2**10", "1024"),
        ("divmod(123,10)[1]", "3"),
    ]:
        cases.append(
            (
                f"What integer does this Python expression evaluate to: {code}? Return only the integer.",
                answer,
            )
        )
    rows = []
    for offset in range(0, len(cases), 8):
        results = await asyncio.gather(
            *(
                one(client, [{"role": "user", "content": prompt}], None, 64)
                for prompt, expected in cases[offset : offset + 8]
            )
        )
        for (prompt, expected), result in zip(cases[offset : offset + 8], results, strict=True):
            answer = result["text"].strip().strip("`").strip()
            rows.append(
                dict(prompt=prompt, expected=expected, correct=answer == expected, **result)
            )
    (path / "quality.json").write_text(json.dumps(rows, indent=2))
    print("QUALITY", path.name, sum(row["correct"] for row in rows), len(rows), flush=True)


async def run(variant):
    mode = variant.split("-")[0]
    path = ROOT / variant
    path.mkdir(exist_ok=True)
    (path / "started.json").write_text(json.dumps({"time": time.time(), "precision": mode}))
    log = (path / "server.log").open("w")
    process = await asyncio.to_thread(
        subprocess.Popen,
        ["python", "-u", "/tmp/server.py"],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=dict(
            os.environ, PRECISION=mode, PYTHONPATH="/root/" + mode + ":/opt/DeepInfer-" + mode
        ),
        start_new_session=True,
    )
    try:
        async with httpx.AsyncClient(
            timeout=650, limits=httpx.Limits(max_connections=64, max_keepalive_connections=32)
        ) as client:
            start = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Server exited during initialization")
                try:
                    r = await client.get(URL + "/health", timeout=5)
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() - start > 2400:
                    raise TimeoutError("Initialization")
                await asyncio.sleep(5)
            health = r.json()
            health["startup_seconds"] = time.monotonic() - start
            (path / "health.json").write_text(json.dumps(health, indent=2))
            print("HEALTH", variant, time.monotonic() - start, flush=True)
            await quality(client, path)
            for kind, messages in [("short", SHORT), ("long", LONG)]:
                # Warm kernels at every measured concurrency without retaining prefixes.
                for c in [1, 8, 24]:
                    await asyncio.gather(*(one(client, messages, None, 32) for _ in range(c)))
                for repeat in range(3):
                    for c in [1, 8, 24] if repeat % 2 == 0 else [24, 8, 1]:
                        await wave(client, kind, messages, "cold", c, repeat, variant, path)
                ns = hashlib.sha256(("ab-warm-" + kind).encode()).hexdigest()
                # Fill every rank before measuring a repeat-traffic workload.
                warm = await asyncio.gather(*(one(client, messages, ns, 32) for _ in range(32)))
                (path / (kind + "-cache-prime.json")).write_text(json.dumps(warm))
                for repeat in range(3):
                    for c in [1, 8, 24] if repeat % 2 == 0 else [24, 8, 1]:
                        await wave(client, kind, messages, "warm", c, repeat, variant, path)
            # Fresh prefills arriving while established streams decode: detect starvation.
            started = [asyncio.Event() for _ in range(4)]
            decoders = [
                asyncio.create_task(one(client, SHORT, None, 1024, event)) for event in started
            ]
            await asyncio.gather(*(event.wait() for event in started))
            cold = [
                asyncio.create_task(
                    one(client, LONG, hashlib.sha256(uuid.uuid4().bytes).hexdigest())
                )
                for _ in range(20)
            ]
            mixed = await asyncio.gather(*decoders, *cold)
            gaps = []
            for response in mixed[:4]:
                delta = [e["elapsed_s"] for e in response["timeline"] if e["type"] == "delta"]
                gaps.append(max(b - a for a, b in itertools.pairwise(delta)))
            (path / "mixed.json").write_text(
                json.dumps({"requests": mixed, "max_delivery_gaps_s": gaps}, indent=2)
            )
            print("MIXED_OK", variant, "gaps", gaps, flush=True)
            (path / "final-health.json").write_text(
                json.dumps((await client.get(URL + "/health")).json(), indent=2)
            )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.to_thread(process.wait, timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            await asyncio.to_thread(process.wait)
        log.close()
        (path / "stopped.json").write_text(
            json.dumps({"time": time.time(), "exit": process.returncode})
        )


async def memory():
    while True:
        p = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader",
            stdout=asyncio.subprocess.PIPE,
        )
        out, _ = await p.communicate()
        with (ROOT / "gpu-memory.jsonl").open("a") as f:
            f.write(json.dumps({"time": time.time(), "gpus": out.decode().splitlines()}) + "\n")
        await asyncio.sleep(2)


async def main():
    sampler = asyncio.create_task(memory())
    try:
        for variant in ["fp4-first", "fp8-first", "fp8-second", "fp4-second"]:
            await run(variant)
            await asyncio.sleep(3)
        (ROOT / "DONE").write_text("ok")
    finally:
        sampler.cancel()


if __name__ == "__main__":
    asyncio.run(main())
