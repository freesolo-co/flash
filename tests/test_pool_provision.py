"""Tests for pool provisioning: the vLLM serve-command contract, plan summary, dry-run, and a
fake-renter that registers backends with a real router."""

from __future__ import annotations

import pytest

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


def test_plan_summary_capacity_sums_mixed_max_loras_for_same_base():
    # A base model split across members with DIFFERENT max_loras: a big slice (1 GPU x 16) and a
    # small slice (2 GPUs x 4). True capacity is 1*16 + 2*4 = 24. The old code did
    # total_count * max(max_loras) = 3 * 16 = 48, crediting the small GPUs with the big slot count
    # and overstating what the fleet can actually hold.
    plan = PoolPlan(
        members=[
            PoolMember(base_model="Q", gpu="H200", count=1, max_loras=16),
            PoolMember(base_model="Q", gpu="RTX5090", count=2, max_loras=4),
        ]
    )
    s = plan_summary(plan)
    assert s["capacity_per_base_model"] == {"Q": 3}  # 3 GPUs total
    assert s["max_concurrent_adapters"] == {"Q": 24}  # 1*16 + 2*4, NOT 3*16


def test_from_toml_parses_int_counts(tmp_path):
    p = tmp_path / "pool.toml"
    p.write_text(
        '[[pool]]\nbase_model = "Q"\ngpu = "RTX5090"\ncount = 2\nmax_loras = 8\n'
    )
    plan = PoolPlan.from_toml(str(p))
    assert len(plan.members) == 1
    assert plan.members[0].count == 2
    assert plan.members[0].max_loras == 8


def test_from_toml_accepts_whole_number_float(tmp_path):
    # A whole-valued float (2.0) is unambiguous and accepted as 2.
    p = tmp_path / "pool.toml"
    p.write_text('[[pool]]\nbase_model = "Q"\ngpu = "RTX5090"\ncount = 2.0\n')
    plan = PoolPlan.from_toml(str(p))
    assert plan.members[0].count == 2
    assert isinstance(plan.members[0].count, int)


def test_from_toml_rejects_non_integer_float(tmp_path):
    # 2.9 must fail loudly, not silently truncate to 2 (provisioning a different fleet than intended).
    p = tmp_path / "pool.toml"
    p.write_text('[[pool]]\nbase_model = "Q"\ngpu = "RTX5090"\ncount = 2.9\n')
    with pytest.raises(ValueError, match="count"):
        PoolPlan.from_toml(str(p))


def test_from_toml_rejects_bool(tmp_path):
    # TOML true/false is not a valid GPU count; int(True) -> 1 would be a silent footgun.
    p = tmp_path / "pool.toml"
    p.write_text('[[pool]]\nbase_model = "Q"\ngpu = "RTX5090"\nmax_loras = true\n')
    with pytest.raises(ValueError, match="max_loras"):
        PoolPlan.from_toml(str(p))


def test_provision_dry_run_does_not_rent():
    plan = PoolPlan(members=[PoolMember(base_model="Q", gpu="RTX5090", count=2)])
    results = provision_pool(plan, "http://router", dry_run=True)
    assert len(results) == 2
    assert all(not r.registered and r.url == "(dry-run)" for r in results)


def test_provision_dry_run_ids_unique_for_same_gpu_and_base_members():
    # Two members sharing gpu + base (here split for different max_loras) — ``i`` restarts at 0 per
    # member, so a (gpu, base, i)-only id would collide across them. The dry-run backend_id must
    # incorporate the member index so every entry is uniquely identifiable.
    plan = PoolPlan(
        members=[
            PoolMember(base_model="org/Q", gpu="RTX5090", count=2, max_loras=8),
            PoolMember(base_model="org/Q", gpu="RTX5090", count=2, max_loras=4),
        ]
    )
    results = provision_pool(plan, "http://router", dry_run=True)
    ids = [r.backend_id for r in results]
    assert len(results) == 4
    assert len(set(ids)) == 4, f"dry-run backend_ids collided: {ids}"


def test_provision_live_without_rent_raises():
    # A live provision (dry_run=False) with no `rent` callable must fail loudly rather than silently
    # taking the dry-run path and returning unregistered "(dry-run)" backends that look like success.
    plan = PoolPlan(members=[PoolMember(base_model="Q", gpu="RTX5090", count=2)])
    with pytest.raises(ValueError, match="rent"):
        provision_pool(plan, "http://router", dry_run=False)


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
