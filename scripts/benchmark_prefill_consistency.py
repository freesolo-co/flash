"""Isolated paired H100 comparison; run with modal run (never deploy).

Two allocations run opposite budget orders on the same GPU, three randomized
repeats per phase. Prompts use repository code/docs and generated structured/chat
records. Unique salts disable cross-request prefix reuse. Results are JSON.
"""

import asyncio
import gc
import hashlib
import json
import math
import random
import subprocess
import time
import uuid
from pathlib import Path

import modal

from flash.serving.app.modal_app import engine_secrets, hf_cache_volume, image

app = modal.App("flash-prefill-consistency-20261107")
MODEL = "Qwen/Qwen3.5-9B"


def cases():
    matrix = [
        {
            "name": f"{kind}-{length}-c{concurrency}",
            "kind": kind,
            "length": length,
            "concurrency": concurrency,
            "output": 256,
            "ignore_eos": True,
        }
        for kind in ("code", "docs", "records", "conversation")
        for length, concurrency in (
            (512, 1),
            (512, 32),
            (4096, 1),
            (4096, 8),
            (4096, 32),
            (16384, 1),
            (16384, 32),
        )
    ]
    for name, kind, length, concurrency, output, ignore in (
        ("mixed-c64", "mixed", 0, 64, 256, True),
        ("code-32768-c8", "code", 32768, 8, 256, True),
        ("code-4096-long-output-c8", "code", 4096, 8, 1024, True),
        *(
            (f"{kind}-natural-c8", kind, 4096, 8, 512, False)
            for kind in ("code", "docs", "records", "conversation")
        ),
    ):
        matrix.append(
            {
                "name": name,
                "kind": kind,
                "length": length,
                "concurrency": concurrency,
                "output": output,
                "ignore_eos": ignore,
            }
        )
    return matrix


def corpus():
    def collect(paths):
        return [{"path": str(p), "text": p.read_text()} for p in paths]

    code = sorted(Path("flash/serving/src").rglob("*.py"))
    docs = [Path("SELF_HOSTING.md"), Path("flash/cli/scaffold/TRAINING.md")]
    return {"code": collect(code), "docs": collect(docs)}


def build_prompts(tokenizer, sources, selected_cases):
    bodies = {}
    for kind in ("code", "docs"):
        files = sources[kind]
        for index in range(64):
            rotated = files[index % len(files) :] + files[: index % len(files)]
            bodies[kind, index] = "\n\n".join(f"FILE: {f['path']}\n{f['text']}" for f in rotated)[
                :300000
            ]
    for index in range(64):
        rng = random.Random(5000 + index)
        bodies["records", index] = "\n".join(
            json.dumps(
                {
                    "id": j,
                    "team": rng.choice(["red", "blue", "green", "gold"]),
                    "duration_ms": rng.randrange(10, 10000),
                    "status": rng.choice(["ok", "failed", "pending"]),
                    "retries": rng.randrange(5),
                }
            )
            for j in range(3500)
        )
        bodies["conversation", index] = "\n".join(
            f"Turn {j}: {'User' if j % 2 == 0 else 'Assistant'}: "
            + rng.choice(
                [
                    "Schedule a project review after the integration tests complete on Friday.",
                    "The proposed budget is 12000 dollars; reserve twenty percent for support.",
                    "We need accessible navigation, keyboard shortcuts, and readable error messages.",
                    "The client asked for a migration plan and a rollback procedure before launch.",
                    "Compare weekly releases with monthly releases and explain the operational costs.",
                    "Assign the documentation review to Morgan and the launch checklist to Sam.",
                    "The staging environment has a separate database and realistic sample records.",
                    "Keep the decision log updated when requirements or deadlines change.",
                ]
            )
            + f" Reference ticket: {rng.randrange(10000, 99999)}."
            for j in range(2000)
        )
    tasks = {
        "code": "Review this code. Explain its control flow, identify concrete failure cases, "
        "and propose Python fixes with tests. Use details from the source.",
        "docs": "Summarize this documentation into a practical onboarding guide. Explain "
        "prerequisites, an example workflow, constraints and troubleshooting steps.",
        "records": "Write a Python function to summarize these records by team, computing "
        "counts, failure rates and p95 duration. Include validation and unit tests.",
        "conversation": "Summarize the discussion. List decisions, owners, deadlines, open "
        "questions and contradictions, then propose a follow-up plan.",
    }
    prompts = {}
    needed = {(c["kind"], c["length"]) for c in selected_cases}
    if any(c["kind"] == "mixed" for c in selected_cases):
        needed.update(
            (k, n) for k in ("code", "docs", "records", "conversation") for n in (512, 4096, 16384)
        )
    for (kind, index), body in bodies.items():
        if not any(k == kind for k, _ in needed):
            continue
        context = tokenizer.encode(body, add_special_tokens=False)
        for length in (512, 4096, 16384, 32768) if kind == "code" else (512, 4096, 16384):
            if (kind, length) not in needed:
                continue
            rendered = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": f"__BENCH_CONTEXT__\n{tasks[kind]}\nRequest reference: {index}.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            before, after = rendered.split("__BENCH_CONTEXT__")
            prefix = tokenizer.encode(before, add_special_tokens=False)
            suffix = tokenizer.encode(after, add_special_tokens=False)
            count = length - len(prefix) - len(suffix)
            if len(context) < count:
                raise ValueError(f"Insufficient corpus for {kind}: {len(context)} < {count}")
            prompts[kind, length, index] = prefix + context[:count] + suffix
    return prompts


def percentile(values, q):
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


async def wave(engine, prompts, case, repeat):
    from vllm import SamplingParams

    async def request(index):
        kind, length = case["kind"], case["length"]
        if kind == "mixed":
            kind = ("code", "docs", "records", "conversation")[index % 4]
            length = ([512] * 11 + [4096] * 4 + [16384])[index % 16]
        # Vary records between repeats, preserving identical inputs across budgets/GPUs.
        ids = prompts[kind, length, (index + repeat * 13) % 64]
        identity = uuid.uuid4().hex
        started = time.monotonic()
        first = last_at = None
        first_count = 0
        last = None
        stream = engine.engine.generate(
            {"prompt_token_ids": ids, "cache_salt": identity},
            SamplingParams(temperature=0, max_tokens=case["output"], ignore_eos=case["ignore_eos"]),
            identity,
        )
        try:
            async for last in stream:
                last_at = time.monotonic()
                tokens = last.outputs[0].token_ids
                if first is None and tokens:
                    first, first_count = last_at, len(tokens)
        finally:
            await stream.aclose()
            if last is None or not last.finished:
                await engine.engine.abort(identity)
        if last is None or not last.finished or first is None:
            raise RuntimeError("Incomplete generation")
        tokens = last.outputs[0].token_ids
        if case["ignore_eos"] and len(tokens) != case["output"]:
            raise RuntimeError("Wrong output length")
        return {
            "prompt_tokens": len(ids),
            "output_tokens": len(tokens),
            "ttft": first - started,
            "latency": last_at - started,
            "decode_tps": (len(tokens) - first_count) / max(1e-9, last_at - first),
            "finish_reason": last.outputs[0].finish_reason,
            "prompt_hash": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest(),
        }

    started = time.monotonic()
    async with asyncio.timeout(600):
        rows = await asyncio.gather(*(request(i) for i in range(case["concurrency"])))
    elapsed = time.monotonic() - started
    return {
        "case": case,
        "repeat": repeat,
        "elapsed": elapsed,
        "requests": rows,
        "aggregate_tps": sum(r["output_tokens"] for r in rows) / elapsed,
        "p50_stream_tps": percentile([r["decode_tps"] for r in rows], 0.5),
        "p05_stream_tps": percentile([r["decode_tps"] for r in rows], 0.05),
        "p95_ttft": percentile([r["ttft"] for r in rows], 0.95),
        "p95_latency": percentile([r["latency"] for r in rows], 0.95),
    }


def gpu_state():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


async def phase(budget, block, sources, focus="", repeats=3):
    import vllm

    from flash.serving.src.engine import model_config
    from flash.serving.src.engine.lora_engine import _LoraEngineImpl

    model_config._BY_MODEL[MODEL]["engine"].update(max_num_seqs=64, max_num_batched_tokens=budget)
    engine = _LoraEngineImpl()
    engine.base_model = MODEL
    await engine._load()
    try:
        matrix = [c for c in cases() if not focus or c["name"] == focus]
        if not matrix:
            raise ValueError(f"Unknown case: {focus}")
        warmups = [
            c for c in cases() if c["name"] in ("code-512-c1", "code-4096-c32", "code-16384-c32")
        ]
        prompts = build_prompts(engine.tokenizer, sources, matrix + warmups)
        metadata = {
            "vllm": vllm.__version__,
            "gpu": gpu_state(),
            "budget": budget,
            "block": block,
            "checkpoint": "Freesolo-Co/Qwen3.5-9B-FP8",
        }
        # Warm short/long prefill and batched decode before measuring either budget.
        for case in warmups:
            await wave(engine, prompts, case, 0)
        rows = []
        for repeat in range(repeats):
            ordered = list(matrix)
            random.Random(7000 + repeat + block * 31).shuffle(ordered)
            for case in ordered:
                row = await wave(engine, prompts, case, repeat)
                row.update(budget=budget, block=block, gpu=gpu_state())
                rows.append(row)
                print("PREFILL_WAVE " + json.dumps(row), flush=True)
        return {"metadata": metadata, "measurements": rows}
    finally:
        engine._liveness_task.cancel()
        await asyncio.gather(engine._liveness_task, return_exceptions=True)
        engine.engine.shutdown()


@app.function(
    image=image,
    gpu="H100!",
    region="us",
    max_containers=2,
    buffer_containers=0,
    single_use_containers=True,
    timeout=7200,
    secrets=engine_secrets or [modal.Secret.from_name("huggingface")],
    volumes={"/vol/hosting-cache": hf_cache_volume},
)
async def paired_block(block, sources, focus="", repeats=3):
    for budget in (4096, 8192) if block == 0 else (8192, 4096):
        yield await phase(budget, block, sources, focus, repeats)
        gc.collect()
        await asyncio.sleep(2)
        used = await asyncio.to_thread(
            subprocess.check_output,
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        if any(int(v) > 1024 for v in used.splitlines()):
            raise RuntimeError(f"GPU memory retained after engine shutdown: {used}")


@app.local_entrypoint()
def main(
    output: str = "results/prefill-consistency-20261107/results.json",
    focus: str = "",
    repeats: int = 3,
    blocks: int = 2,
):
    if blocks not in (1, 2) or not 1 <= repeats <= 30:
        raise ValueError("Use one or two blocks and between 1 and 30 repeats")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = corpus()
    (path.parent / "corpus-manifest.json").write_text(
        json.dumps(
            {
                k: [
                    {"path": f["path"], "sha256": hashlib.sha256(f["text"].encode()).hexdigest()}
                    for f in files
                ]
                for k, files in sources.items()
            },
            indent=2,
        )
    )
    results = []
    write_lock = asyncio.Lock()

    async def collect(block):
        async for result in paired_block.remote_gen.aio(block, sources, focus, repeats):
            async with write_lock:
                results.append(result)
                await asyncio.to_thread(path.write_text, json.dumps(results, indent=2) + "\n")
            print(f"SAVED_PHASE block={block} budget={result['metadata']['budget']}", flush=True)

    async def run():
        await asyncio.gather(*(collect(block) for block in range(blocks)))

    asyncio.run(run())
