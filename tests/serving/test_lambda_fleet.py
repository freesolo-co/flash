"""Ownership loss must not duplicate GPUs or terminate another deployment's workers."""

import copy

import httpx
import pytest

from flash.serving.lambda_cloud.config import LEASE_SECONDS, placements, select_region
from flash.serving.lambda_cloud.fleet import Fleet


@pytest.fixture
def anyio_backend():
    return "asyncio"


MODEL = "Qwen/Qwen3.5-0.8B"
OWNER = "a" * 32


class Ledger:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return copy.deepcopy(self.data.get(key))

    async def put(self, key, value):
        self.data[key] = copy.deepcopy(value)

    async def delete(self, key):
        self.data.pop(key, None)

    async def items(self):
        for item in list(copy.deepcopy(self.data).items()):
            yield item


class API:
    def __init__(self):
        self.instances = []
        self.launches = 0
        self.terminated = []
        self.lose_response = False

    async def call(self, method, path, body=None):
        if path == "instance-types":
            return {
                "gpu_1x_a10": {
                    "instance_type": {
                        "gpu_description": "A10 (24 GB PCIe)",
                        "architecture": "x86_64",
                        "specs": {"gpus": 1},
                        "price_cents_per_hour": 400,
                    },
                    "regions_with_capacity_available": [{"name": "us-south-2"}],
                }
            }
        if path in {"firewall-rulesets", "filesystems"}:
            return [] if method == "GET" else {"id": "resource"}
        if path == "instance-operations/launch":
            self.launches += 1
            assert body["instance_type_name"] == "gpu_1x_a10"
            assert body["file_system_mounts"][0]["mount_point"] == "/data/hosting-cache"
            assert body["firewall_rulesets"]
            self.instances.append(
                {"id": "vm1", "name": body["name"], "tags": body["tags"], "status": "active"}
            )
            if self.lose_response:
                raise httpx.ReadTimeout("lost launch response")
            return {"instance_ids": ["vm1"]}
        if path == "instances":
            return copy.deepcopy(self.instances)
        if path.startswith("instances/"):
            return copy.deepcopy(
                next(item for item in self.instances if item["id"] == path.split("/")[1])
            )
        if path == "instance-operations/terminate":
            self.terminated.extend(body["instance_ids"])
            return {}
        raise AssertionError(path)


@pytest.mark.anyio
async def test_ambiguous_launch_is_reconciled_without_duplicate():
    api, ledger = API(), Ledger()
    api.lose_response = True
    fleet = Fleet(api, ledger, "dev", clock=lambda: 100)
    with pytest.raises(httpx.ReadTimeout):
        await fleet.launch(OWNER, MODEL, "bootstrap", "b" * 40)
    assert (await fleet.launch(OWNER, MODEL, "bootstrap", "b" * 40))["id"] == "vm1"
    assert api.launches == 1
    assert ledger.data[OWNER]["instance_id"] == "vm1"


@pytest.mark.anyio
async def test_reaping_preserves_active_foreign_and_other_environment_instances():
    api, ledger = API(), Ledger()
    now = 100
    fleet = Fleet(api, ledger, "dev", clock=lambda: now)
    await fleet.launch(OWNER, MODEL, "bootstrap", "b" * 40)
    foreign = copy.deepcopy(api.instances[0])
    foreign.update(id="foreign", name="someone-elses-worker")
    api.instances.append(foreign)
    other_env = copy.deepcopy(api.instances[0])
    other_env["id"] = "prod"
    other_env["tags"][-1]["value"] = "main"
    api.instances.append(other_env)
    now += LEASE_SECONDS
    await fleet.reap()
    assert api.terminated == []
    await fleet.heartbeat(OWNER)
    now += LEASE_SECONDS + 1
    await fleet.reap()
    assert api.terminated == ["vm1"]
    with pytest.raises(RuntimeError, match="released"):
        await fleet.heartbeat(OWNER)


@pytest.mark.anyio
async def test_release_finds_vm_after_launch_response_loss():
    api, ledger = API(), Ledger()
    api.lose_response = True
    fleet = Fleet(api, ledger, "dev")
    with pytest.raises(httpx.ReadTimeout):
        await fleet.launch(OWNER, MODEL, "bootstrap", "b" * 40)
    await fleet.release(OWNER)
    assert api.terminated == ["vm1"]
    with pytest.raises(ValueError, match="already used"):
        await fleet.launch(OWNER, MODEL, "bootstrap", "b" * 40)
    assert api.launches == 1


def test_hardware_choices_and_capacity_fail_closed():
    assert placements(MODEL) == ("A10",)
    assert placements("Qwen/Qwen3.6-35B-A3B") == ("H200", "B200")
    with pytest.raises(ValueError, match="only defined"):
        placements("deepseek-ai/DeepSeek-V4.1-Flash-1017")
    with pytest.raises(RuntimeError, match="no US capacity"):
        select_region({}, MODEL)


@pytest.mark.anyio
async def test_uncertain_leases_count_toward_per_release_replica_limit():
    api, ledger = API(), Ledger()
    fleet = Fleet(api, ledger, "dev", clock=lambda: 100)
    for owner in ("a" * 32, "b" * 32):
        await ledger.put(
            owner,
            {
                "model": MODEL,
                "revision": "c" * 40,
                "heartbeat": 100,
                "created": 100,
                "instance_id": None,
            },
        )
    with pytest.raises(RuntimeError, match="replica limit"):
        await fleet.launch("d" * 32, MODEL, "bootstrap", "c" * 40)
    assert api.launches == 0
    # A new release may overlap the old one during the existing rolling handoff.
    await fleet.launch("d" * 32, MODEL, "bootstrap", "e" * 40)
    assert api.launches == 1


@pytest.mark.anyio
async def test_all_mode_can_fill_the_entire_shared_budget_with_lambda(monkeypatch):
    from flash.serving.app.inference_provider import MODEL_POLICIES, ModelPolicy

    monkeypatch.setitem(MODEL_POLICIES, MODEL, ModelPolicy("all"))
    api, ledger = API(), Ledger()
    fleet = Fleet(api, ledger, "dev", clock=lambda: 100)
    await fleet.launch(OWNER, MODEL, "bootstrap", "b" * 40)
    await fleet.launch("c" * 32, MODEL, "bootstrap", "b" * 40)
    with pytest.raises(RuntimeError, match="replica limit"):
        await fleet.launch("d" * 32, MODEL, "bootstrap", "b" * 40)
    assert api.launches == 2


@pytest.mark.anyio
async def test_custom_model_maximum_is_enforced_independently(monkeypatch):
    from flash.serving.app.inference_provider import MODEL_POLICIES, ModelPolicy

    monkeypatch.setitem(MODEL_POLICIES, MODEL, ModelPolicy("lambda", 1, 3, ("A10",)))
    api, ledger = API(), Ledger()
    fleet = Fleet(api, ledger, "dev", clock=lambda: 100)
    for owner in ("a" * 32, "b" * 32, "c" * 32):
        await fleet.launch(owner, MODEL, "bootstrap", "e" * 40)
    with pytest.raises(RuntimeError, match="replica limit"):
        await fleet.launch("d" * 32, MODEL, "bootstrap", "e" * 40)
    assert api.launches == 3
    with pytest.raises(RuntimeError, match="replica limit"):
        await fleet.launch("f" * 32, "Qwen/Qwen3.5-4B", "bootstrap", "e" * 40)
    assert api.launches == 3


@pytest.mark.anyio
async def test_explicit_capacity_rejection_retries_without_ambiguous_lease():
    from flash.serving.lambda_cloud.api import LambdaAPIError
    from flash.serving.lambda_cloud.config import NoCapacityError

    api, ledger = API(), Ledger()
    original = api.call
    attempts = []
    now = [100]

    async def reject(method, path, body=None):
        if path == "instance-operations/launch":
            attempts.append(body["region_name"])
            raise LambdaAPIError(400, "instance-operations/launch/insufficient-capacity")
        return await original(method, path, body)

    api.call = reject
    fleet = Fleet(api, ledger, "dev", clock=lambda: now[0])
    for instant in (100, 160, 220):
        now[0] = instant
        with pytest.raises(NoCapacityError):
            await fleet.launch(OWNER, MODEL, "bootstrap", "b" * 40)
        assert not ledger.data
    assert len(attempts) == 2  # Rejected offers cool down while other shapes remain eligible.


@pytest.mark.anyio
async def test_only_fresh_validated_health_counts_as_ready():
    ledger = Ledger()
    now = [100]
    fleet = Fleet(API(), ledger, "dev", clock=lambda: now[0])
    await fleet.launch(OWNER, MODEL, "bootstrap", "b" * 40)
    assert await fleet.ready_count(MODEL) == 0
    health = {"ok": True, "base_model": MODEL, "deployment_sha": "b" * 40, "configured_gpu": "A10"}
    await fleet.heartbeat(OWNER, health)
    assert await fleet.ready_count(MODEL) == 1
    now[0] += 91
    assert await fleet.ready_count(MODEL) == 0
    await fleet.heartbeat(OWNER, dict(health, deployment_sha="c" * 40))
    assert await fleet.ready_count(MODEL) == 0
    await fleet.heartbeat(OWNER, health)
    await fleet.release(OWNER)
    assert await fleet.ready_count(MODEL) == 0
