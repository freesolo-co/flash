"""HEAVY real-training tests: actual GRPO LoRA gradient updates driven through the shared rollout
pool, including MULTIPLE concurrent training runs sharing one pool.

Unlike the other pool tests (which stub the GPU step), these run the REAL ``pool_policy`` path —
transformers + peft LoRA forward/backward + optimizer step on a tiny model on CPU — so they prove
the full distributed loop actually trains a model: rollout comes from the pool, reward from the
off-GPU reward worker, advantages are GRPO group-normalized, the LoRA is really updated, and the new
weights are synced back to the pool each step.

Skipped automatically where the GPU stack isn't installed (CI runs the offline extra), and where the
tiny test model can't be fetched. Run locally with torch+peft+transformers available:

    PYTHONPATH=<overlay>:. <py-with-torch> -m pytest tests/test_pool_training_real.py -v
"""

from __future__ import annotations

import os
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")

from flash.engine.pool_policy import build_lora_policy_update
from flash.engine.pool_trainer import GRPOPoolLoop
from flash.pool.client import RolloutPoolClient
from tests._helpers.pool_harness import build_harness

os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

GOOD = "the answer is paris"
BAD = "zzz"


def _prefers_good(prompts, completions, info):
    """Reward worker scorer: 1.0 for the target completion, else 0.0 (a clean learning signal)."""
    return [1.0 if (c or "").strip() == GOOD else 0.0 for c in completions]


@pytest.fixture(scope="module")
def tiny_model() -> str:
    mid = os.environ.get("FLASH_TEST_TINY_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
    try:  # warm the cache once; skip cleanly if offline
        from huggingface_hub import hf_hub_download

        hf_hub_download(mid, "config.json")
    except Exception as e:
        pytest.skip(f"tiny test model unavailable: {e}")
    return mid


def _train_one(
    router_url: str,
    tiny_model: str,
    name: str,
    out_dir: str,
    *,
    steps: int = 8,
    group_size: int = 4,
    replicas: int = 1,
    prompt: str | None = None,
) -> dict:
    """Run one REAL GRPO LoRA training run against the pool. Returns learning metrics."""
    prompt = prompt or f"question for {name}: capital of france?"
    client = RolloutPoolClient(router_url, adapter=name, base_model=tiny_model)
    policy_update, current_uri = build_lora_policy_update(
        tiny_model, out_dir=out_dir, lora_rank=8, lr=5e-3, max_len=48, device="cpu"
    )
    loop = GRPOPoolLoop(client, group_size=group_size, prefetch=2, max_tokens=16)
    loop.register(uri=current_uri(), replicas=replicas)

    margin_before = policy_update.logprob(prompt, GOOD) - policy_update.logprob(prompt, BAD)
    losses: list[float] = []
    versions: list[int] = []

    def _on_step(r):
        losses.append(policy_update.last_loss)
        versions.append(r.adapter_version)

    # Same prompt batch each step -> a clear "does it fit the high-reward completion" signal.
    batches = [[prompt, prompt] for _ in range(steps)]
    results = loop.run(batches, policy_update, on_step=_on_step)
    margin_after = policy_update.logprob(prompt, GOOD) - policy_update.logprob(prompt, BAD)
    client.close()
    return {
        "name": name,
        "margin_before": margin_before,
        "margin_after": margin_after,
        "losses": losses,
        "versions": versions,
        "mean_rewards": [r.mean_reward for r in results],
        "steps": len(results),
    }


def test_real_lora_training_single_run(tiny_model, tmp_path):
    """One real GRPO run: the model learns to prefer the high-reward completion, the advantage-
    weighted loss improves, and weights sync to the pool every step."""
    h = build_harness(
        [{"id": "gpu0", "base_model": tiny_model, "choices_pool": [GOOD, BAD]}],
        reward_workers=1,
        reward_scorer=_prefers_good,
    )
    try:
        m = _train_one(h.router_url, tiny_model, "run1", str(tmp_path / "run1"), steps=8)
        # 1) actually learned: logprob(GOOD) - logprob(BAD) rose
        assert m["margin_after"] > m["margin_before"] + 0.05, m
        # 2) the optimizer actually reduced the advantage-weighted loss
        assert m["losses"][-1] < m["losses"][0], m["losses"]
        # 3) the reward signal flowed (group has GOOD+BAD -> mean reward ~0.5)
        assert all(r == pytest.approx(0.5, abs=0.2) for r in m["mean_rewards"]), m["mean_rewards"]
        # 4) weights synced to the pool every step (version advanced 1..8)
        assert m["versions"] == list(range(1, 9)), m["versions"]
        # 5) rollout actually ran on the pool GPU (off the trainer) and the adapter is resident
        rec = h.record("gpu0")
        assert rec.chat_calls
        assert "run1" in rec.loaded
    finally:
        h.stop()


def test_multiple_concurrent_real_runs_share_pool(tiny_model, tmp_path):
    """THREE real GRPO runs train CONCURRENTLY on one shared pool (2 GPUs, multi-LoRA). Each learns
    independently, all adapters are served across the fleet, and the pool stays healthy."""
    h = build_harness(
        [
            {"id": "gpu0", "base_model": tiny_model, "choices_pool": [GOOD, BAD], "max_loras": 8},
            {"id": "gpu1", "base_model": tiny_model, "choices_pool": [GOOD, BAD], "max_loras": 8},
        ],
        reward_workers=2,
        reward_scorer=_prefers_good,
    )
    results: dict[str, dict] = {}
    errors: list[tuple[str, BaseException]] = []

    def _run(name: str) -> None:
        try:
            results[name] = _train_one(
                h.router_url, tiny_model, name, str(tmp_path / name), steps=5, replicas=2
            )
        except BaseException as e:
            errors.append((name, e))

    try:
        names = ["runA", "runB", "runC"]
        threads = [threading.Thread(target=_run, args=(n,)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=300)
        # If a run hung past the timeout, fail loudly here instead of letting a still-running thread
        # block teardown / mask the real failure.
        alive = [t.name for t in threads if t.is_alive()]
        assert not alive, f"training threads did not finish in time: {alive}"

        assert not errors, errors
        assert set(results) == set(names)
        # every run actually trained
        for name, m in results.items():
            assert m["steps"] == 5, (name, m)
            assert m["margin_after"] > m["margin_before"], (name, m)
            assert m["losses"][-1] < m["losses"][0], (name, m["losses"])
        # all three adapters were served across the shared GPUs (multi-LoRA, distributed)
        served = set(h.record("gpu0").loaded) | set(h.record("gpu1").loaded)
        assert set(names) <= served, served
        # the pool absorbed all the concurrent traffic and stayed healthy
        snap = h.status()
        assert all(b["healthy"] for b in snap["pool"]["backends"])
        assert sum(len(h.record(b).chat_calls) for b in ("gpu0", "gpu1")) >= 3 * 5 * 2
    finally:
        h.stop()


def test_concurrent_runs_keep_distinct_policies(tiny_model, tmp_path):
    """Two concurrent runs with OPPOSITE reward targets must learn OPPOSITE policies — proving the
    shared pool keeps each run's adapter (and weight-sync) isolated, no cross-contamination."""

    def prefers(target):
        return lambda prompts, completions, info: [1.0 if (c or "").strip() == target else 0.0 for c in completions]

    # run "good" is rewarded for GOOD; run "bad" is rewarded for BAD. Same base, same pool.
    h_good = build_harness(
        [{"id": "g", "base_model": tiny_model, "choices_pool": [GOOD, BAD]}],
        reward_workers=1,
        reward_scorer=prefers(GOOD),
    )
    h_bad = build_harness(
        [{"id": "g", "base_model": tiny_model, "choices_pool": [GOOD, BAD]}],
        reward_workers=1,
        reward_scorer=prefers(BAD),
    )
    try:
        out: dict[str, float] = {}

        def run(tag, harness, target):
            client = RolloutPoolClient(harness.router_url, adapter=tag, base_model=tiny_model)
            pu, cur = build_lora_policy_update(
                tiny_model, out_dir=str(tmp_path / tag), lora_rank=8, lr=5e-3, max_len=48, device="cpu"
            )
            loop = GRPOPoolLoop(client, group_size=4, prefetch=2, max_tokens=16)
            loop.register(uri=cur())
            prompt = "q: capital of france?"
            before = pu.logprob(prompt, target) - pu.logprob(prompt, GOOD if target == BAD else BAD)
            loop.run([[prompt, prompt] for _ in range(8)], pu)
            after = pu.logprob(prompt, target) - pu.logprob(prompt, GOOD if target == BAD else BAD)
            out[tag] = after - before
            client.close()

        t1 = threading.Thread(target=run, args=("good", h_good, GOOD))
        t2 = threading.Thread(target=run, args=("bad", h_bad, BAD))
        t1.start()
        t2.start()
        t1.join(300)
        t2.join(300)
        still_alive = [t.name for t in (t1, t2) if t.is_alive()]
        assert not still_alive, f"training threads did not finish in time: {still_alive}"
        # each run increased the logprob of ITS OWN target relative to the other completion
        assert out["good"] > 0, out
        assert out["bad"] > 0, out
    finally:
        h_good.stop()
        h_bad.stop()
