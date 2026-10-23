"""Exercise the actual shared server, real teacher scoring, and process recovery on CUDA."""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from flash.training.shared.client import Client
from flash.training.shared.contracts import Limits
from flash.training.shared.journal import Journal
from flash.training.shared.recipes import grpo_step, opd_step, sft_step


class Server:
    def __init__(self, root, config, credentials):
        self.root, self.config, self.credentials = root, config, credentials
        self.process = None
        self.log = root / "server.log"
        self.config_path = root / "pool.json"
        self.config_path.write_text(json.dumps(config))
        self.url = "http://127.0.0.1:8090"

    def start(self):
        environment = {**os.environ, "FLASH_SHARED_CREDENTIALS": json.dumps(self.credentials)}
        with self.log.open("ab") as output:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "flash.training.shared.server",
                    "--config",
                    str(self.config_path),
                    "--state",
                    str(self.root / "state"),
                ],
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        client = Client(self.url, next(iter(self.credentials)), timeout=2)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(self.log.read_text()[-10000:])
            try:
                client.request("GET", "/usage")
                return
            except (OSError, RuntimeError):
                time.sleep(0.25)
        raise TimeoutError("shared server startup exceeded five minutes")

    def stop(self, *, kill=False):
        if self.process is None or self.process.poll() is not None:
            return
        if kill:
            self.process.kill()
        else:
            self.process.terminate()
        try:
            self.process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)


def expect_http(code, operation):
    import pytest

    with pytest.raises(RuntimeError, match=f"HTTP {code}:"):
        operation()


def wait_running(session, command):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        row = next(c for c in session.status()["commands"] if c["id"] == command)
        if row["state"] == "running":
            return
        assert row["state"] == "queued", row
        time.sleep(0.01)
    raise TimeoutError("command was not observed executing")


def equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal(a, b)
    else:
        assert left == right


def checkpoint(root, session):
    import torch

    saved = session.checkpoint()
    return torch.load(
        root / "state/checkpoints" / f"{saved['checkpoint']}.pt",
        weights_only=True,
        map_location="cpu",
    )


def make_teacher(model_id, revision, tokenizer):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    loader = (
        AutoModelForImageTextToText
        if getattr(config, "text_config", None)
        else AutoModelForCausalLM
    )
    model = (
        loader.from_pretrained(
            model_id, revision=revision, dtype=torch.bfloat16, trust_remote_code=False
        )
        .cuda()
        .eval()
    )
    model.requires_grad_(False)
    prefix = tokenizer.encode("Give a concise, accurate explanation. ", add_special_tokens=False)
    calls = []

    def teacher(prompt, completion):
        context = prefix + prompt
        tokens = torch.tensor([context + completion], device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=tokens[:, :-1], use_cache=False).logits[0].float()
            logs = logits.log_softmax(-1).gather(-1, tokens[0, 1:, None]).squeeze(-1)
            values = logs[len(context) - 1 :].cpu().tolist()
        assert len(values) == len(completion)
        assert all(math.isfinite(v) and v <= 0 for v in values)
        calls.append(len(completion))
        return list(range(len(completion))), values

    return teacher, calls


def continuation_payload(session, prompt, target, teacher):
    algorithm = session.algorithm
    if algorithm == "sft":
        completion = target
        sample = None
    else:
        sample = session.sample(prompt, max_tokens=16, seed=1234)
        completion = sample["tokens"]
    prefix = len(prompt) - 1
    row = {"tokens": prompt + completion, "mask": [False] * prefix + [True] * len(completion)}
    if algorithm == "grpo":
        row.update(
            old_logprobs=[0.0] * prefix + sample["logprobs"],
            advantages=[0.0] * prefix + [1.0] * len(completion),
        )
    if algorithm == "opd":
        groups, logs = teacher(prompt, completion)
        row.update(teacher_groups=[-1] * prefix + groups, teacher_logprobs=[0.0] * prefix + logs)
    payload = {"version": session.version, "rows": [row]}
    if algorithm == "grpo":
        payload["normalizer"] = 16
    return payload


def check_fairness(server, sessions, prompt, target):
    a, b, c = sessions
    slow = {"version": a.version, "prompt": prompt, "max_tokens": 256, "seed": 88}
    blocking = a.submit("sample", slow)
    wait_running(a, blocking)
    commands = []
    for index in range(3):
        payload = {
            "version": a.version + index,
            "rows": [
                {
                    "tokens": prompt + target,
                    "mask": [False] * (len(prompt) - 1) + [True] * len(target),
                }
            ],
        }
        command = a.submit("train", payload, key=f"backlog-{index}")
        assert a.submit("train", payload, key=f"backlog-{index}") == command
        commands.append(command)
    expect_http(429, lambda: a.submit("sample", slow))
    peers = [
        s.submit("sample", {"version": s.version, "prompt": prompt, "max_tokens": 1, "seed": 9})
        for s in (b, c)
    ]
    a.client.request("DELETE", a.path + "/commands/" + commands[-1])
    commands.pop()
    a.wait(blocking)
    for session, command in zip((b, c), peers, strict=True):
        session.wait(command)
    for command in commands:
        a.wait(command)
    with sqlite3.connect(server.root / "state/pool.sqlite") as db:
        starts = dict(db.execute("SELECT id,started_ns FROM commands WHERE started_ns IS NOT NULL"))
    assert all(starts[p] < starts[commands[0]] for p in peers), starts
    return {
        "peer_turns_before_backlog": True,
        "quota_429": True,
        "cancelled_queued_work": True,
        "idempotent_submission": True,
    }


def exercise(server, clients, tokenizer, teacher, teacher_calls):
    algorithms = ("sft", "grpo", "opd")
    sessions = [
        client.create(algorithm, seed=40 + i, learning_rate=1e-4)
        for i, (client, algorithm) in enumerate(zip(clients, algorithms, strict=True))
    ]
    for session, algorithm in zip(sessions, algorithms, strict=True):
        session.algorithm = algorithm
    prompt = tokenizer.encode(
        "Explain why the sky appears blue. Explanation:", add_special_tokens=False
    )
    target = tokenizer.encode(" Sunlight scatters in the atmosphere.", add_special_tokens=False)
    long_prompt = (
        tokenizer.encode("The atmosphere scatters light. " * 35, add_special_tokens=False) + prompt
    )
    assert 128 <= len(long_prompt) < 256
    initial = [checkpoint(server.root, session) for session in sessions]
    spare = clients[0].create("sft")
    expect_http(429, lambda: clients[0].create("sft"))
    spare.close()
    expect_http(404, lambda: clients[1].request("GET", sessions[0].path))
    expect_http(401, lambda: Client(server.url, secrets.token_hex(32)).request("GET", "/usage"))
    losses = {algorithm: [] for algorithm in algorithms}

    def train(index):
        session = sessions[index]
        for step in range(3):
            if index == 0:
                result = sft_step(session, [(long_prompt, target)])
            elif index == 1:
                # deterministic sequence reward exercises group variation, not a task-quality claim.
                result = grpo_step(
                    session,
                    [prompt],
                    lambda p, tokens: sum(tokens) / len(tokens),
                    group_size=4,
                    max_tokens=16,
                    seed=100 + 4 * step,
                )
            else:
                result = opd_step(session, [prompt], teacher, max_tokens=16, seed=200 + step)
            assert result["version"] == step + 1
            assert math.isfinite(result["loss"])
            assert result["grad_norm"] > 0
            losses[session.algorithm].append(result["loss"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(train, range(3)))
    after = [checkpoint(server.root, session) for session in sessions]
    for before, current in zip(initial, after, strict=True):
        import torch

        assert any(not torch.equal(before["adapter"][k], v) for k, v in current["adapter"].items())
    peer_before = checkpoint(server.root, sessions[1])
    sft_step(sessions[0], [(prompt, target)])
    peer_after = checkpoint(server.root, sessions[1])
    for key in ("adapter", "optimizer", "rng"):
        equal(peer_before[key], peer_after[key])
    fairness = check_fairness(server, sessions, prompt, target)
    print("SERVICE_PROGRESS concurrent recipes, tenant isolation, and fairness passed", flush=True)
    saved, payloads, expected, final = [], [], [], []
    for session in sessions:
        payload = continuation_payload(session, prompt, target, teacher)
        saved.append(session.checkpoint()["version"])
        payloads.append(payload)
        expected.append(session.call("train", payload))
        final.append(checkpoint(server.root, session))
    interrupted = sessions[0].submit(
        "sample", {"version": sessions[0].version, "prompt": prompt, "max_tokens": 256, "seed": 99}
    )
    wait_running(sessions[0], interrupted)
    server.stop(kill=True)
    server.start()
    assert all(s.status()["state"] == "lost" for s in sessions)
    row = next(c for c in sessions[0].status()["commands"] if c["id"] == interrupted)
    assert row["state"] == "unknown"
    assert clients[0].request("GET", "/usage")["unknown_executions"] == 1
    restored = []
    for i, (session, version) in enumerate(zip(sessions, saved, strict=True)):
        fresh = clients[i].create(session.algorithm, seed=40 + i, learning_rate=1e-4)
        if i == 1:
            expect_http(404, lambda fresh=fresh: fresh.restore(sessions[0].id, saved[0]))
        assert fresh.restore(session.id, version)["version"] == version
        assert fresh.call("train", payloads[i]) == expected[i]
        resumed = checkpoint(server.root, fresh)
        for key in ("adapter", "optimizer", "rng"):
            equal(final[i][key], resumed[key])
        restored.append(fresh)
    for session in restored:
        session.close()
    usage = [client.request("GET", "/usage") for client in clients]
    server.stop()
    journal = Journal(
        server.root / "state/pool.sqlite",
        limits=Limits(**server.config["limits"]),
        vocab_size=tokenizer.vocab_size,
        rate_microusd_per_hour=4_000_000,
    )
    pool = journal.pool_usage()
    journal.db.close()
    assert sum(u["execution_ns"] for u in usage) == pool["attributed_execution_ns"]
    assert (
        sum(u["resource_cost_estimate"]["numerator"] for u in usage)
        + pool["overhead_cost_estimate"]["numerator"]
        == pool["resource_cost_estimate"]["numerator"]
    )
    ids = [c["command"] for u in usage for c in u["commands"]]
    assert len(ids) == len(set(ids))
    return {
        "losses": losses,
        "sft_prompt_tokens": len(long_prompt),
        "teacher_calls": len(teacher_calls),
        "teacher_scored_tokens": sum(teacher_calls),
        "fairness": fairness,
        "independent_adapter_updates": True,
        "peer_state_unchanged": True,
        "sigkill_observed_running_command": True,
        "unknown_not_replayed": True,
        "exact_process_restart_continuation": list(algorithms),
        "cost_reconciles": True,
        "physical_execution_receipts": len(ids),
        "pool_usage": pool,
    }


def main():
    import torch
    from transformers import AutoTokenizer

    assert torch.cuda.is_available(), "qualification requires an actual CUDA device"
    started = time.monotonic()
    model_id, revision = (
        os.environ["FLASH_SHARED_MODEL_ID"],
        os.environ["FLASH_SHARED_MODEL_REVISION"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    teacher, calls = make_teacher(model_id, revision, tokenizer)
    print("SERVICE_PROGRESS actual CUDA teacher loaded", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        credentials = {
            secrets.token_hex(32): {"tenant": f"org-{i}", "owner": f"owner-{i}"} for i in range(3)
        }
        config = {
            "model": model_id,
            "revision": revision,
            "device": "cuda",
            "rank": 8,
            "alpha": 16,
            "rate_microusd_per_hour": 4_000_000,
            "limits": {
                "adapters": 6,
                "adapters_per_tenant": 2,
                "context_tokens": 512,
                "request_tokens": 2048,
                "queued_requests_per_tenant": 4,
            },
        }
        server = Server(Path(directory), config, credentials)
        try:
            server.start()
            clients = [Client(server.url, token) for token in credentials]
            result = exercise(server, clients, tokenizer, teacher, calls)
            result.update(
                model=model_id,
                revision=revision,
                gpu=torch.cuda.get_device_name(),
                wall_seconds=time.monotonic() - started,
            )
            print("SHARED_SERVICE_RESULT " + json.dumps(result), flush=True)
        except BaseException:
            if server.log.exists():
                print(server.log.read_text()[-16000:], file=sys.stderr, flush=True)
            raise
        finally:
            server.stop()


if __name__ == "__main__":
    main()
