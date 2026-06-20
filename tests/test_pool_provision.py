"""Tests for pool provisioning: the vLLM serve-command contract, plan summary, dry-run, and a
fake-renter that registers backends with a real router."""

from __future__ import annotations

from flash.pool.config import PoolMember, PoolPlan
from flash.pool.provision import (
    build_vllm_serve_command,
    plan_summary,
    provision_pool,
)
from tests._helpers.pool_harness import build_harness


def test_vllm_serve_command_enables_dynamic_multilora():
    argv, env = build_vllm_serve_command("Qwen/Q", port=8001, max_loras=16, max_lora_rank=64)
    assert argv[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--enable-lora" in argv
    assert argv[argv.index("--max-loras") + 1] == "16"
    assert argv[argv.index("--max-lora-rank") + 1] == "64"
    assert argv[argv.index("--model") + 1] == "Qwen/Q"
    assert argv[argv.index("--port") + 1] == "8001"
    # runtime LoRA load/unload MUST be on or the router can't hot-swap adapters
    assert env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "1"


def test_plan_summary_capacity():
    plan = PoolPlan(
        members=[
            PoolMember(base_model="Q", gpu="RTX5090", count=2, max_loras=8),
            PoolMember(base_model="R", gpu="A100", count=1, max_loras=4),
        ]
    )
    s = plan_summary(plan)
    assert s["total_gpus"] == 3
    assert s["capacity_per_base_model"] == {"Q": 2, "R": 1}
    # 2 GPUs * 8 loras = 16 concurrent Q adapters; 1 * 4 = 4 R adapters
    assert s["max_concurrent_adapters"] == {"Q": 16, "R": 4}


def test_provision_dry_run_does_not_rent():
    plan = PoolPlan(members=[PoolMember(base_model="Q", gpu="RTX5090", count=2)])
    results = provision_pool(plan, "http://router", dry_run=True)
    assert len(results) == 2
    assert all(not r.registered and r.url == "(dry-run)" for r in results)


def test_provision_with_fake_renter_registers_backends():
    # A real router; the "renter" hands back the addresses of two already-running fake backends.
    h = build_harness([{"id": "pre0", "base_model": "Q"}, {"id": "pre1", "base_model": "Q"}])
    try:
        # Drop the auto-registered ones so we can re-register via provision_pool.
        with h.client() as c:
            c.delete("/pool/backends/pre0")
            c.delete("/pool/backends/pre1")
        urls = [h.backends["pre0"][0].url, h.backends["pre1"][0].url]

        def fake_rent(member, i):
            return (f"rented-{i}", urls[i], 0.88)

        plan = PoolPlan(members=[PoolMember(base_model="Q", gpu="RTX5090", count=2, max_loras=8)])
        results = provision_pool(plan, h.router_url, dry_run=False, rent=fake_rent)
        assert all(r.registered for r in results)
        snap = h.status()
        ids = {b["id"] for b in snap["pool"]["backends"]}
        assert ids == {"rented-0", "rented-1"}
        assert snap["pool"]["summary"]["cost_per_hour"] == 1.76  # 2 * 0.88
    finally:
        h.stop()


def test_reward_worker_app_scores():
    import httpx

    from flash.pool.rewards import create_reward_app
    from tests._helpers.pool_harness import start_server

    def scorer(prompts, completions, info):
        return [len(c) for c in completions]

    srv = start_server(create_reward_app(scorer, reward_id="lens"))
    try:
        r = httpx.post(srv.url + "/score", json={"completions": ["ab", "abcd"]}, timeout=10)
        r.raise_for_status()
        assert r.json() == {"scores": [2.0, 4.0], "reward_id": "lens"}
        assert httpx.get(srv.url + "/health", timeout=10).json()["reward_id"] == "lens"
    finally:
        srv.stop()
