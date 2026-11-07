"""Ownership loss must not duplicate GPUs or terminate another deployment's workers."""

import copy

import httpx
import pytest

from flash.serving.lambda_cloud.config import LEASE_SECONDS, placement, select_region
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
                "gpu_1x_h100_sxm5": {"regions_with_capacity_available": [{"name": "us-south-2"}]}
            }
        if path in {"firewall-rulesets", "filesystems"}:
            return [] if method == "GET" else {"id": "resource"}
        if path == "instance-operations/launch":
            self.launches += 1
            assert body["instance_type_name"] == "gpu_1x_h100_sxm5"
            assert body["file_system_mounts"][0]["mount_point"] == "/vol/hosting-cache"
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
        await fleet.launch(OWNER, MODEL, "bootstrap")
    assert (await fleet.launch(OWNER, MODEL, "bootstrap"))["id"] == "vm1"
    assert api.launches == 1
    assert ledger.data[OWNER]["instance_id"] == "vm1"


@pytest.mark.anyio
async def test_reaping_preserves_active_foreign_and_other_environment_instances():
    api, ledger = API(), Ledger()
    now = 100
    fleet = Fleet(api, ledger, "dev", clock=lambda: now)
    await fleet.launch(OWNER, MODEL, "bootstrap")
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
        await fleet.launch(OWNER, MODEL, "bootstrap")
    await fleet.release(OWNER)
    assert api.terminated == ["vm1"]
    with pytest.raises(ValueError, match="already used"):
        await fleet.launch(OWNER, MODEL, "bootstrap")
    assert api.launches == 1


def test_hardware_choices_and_capacity_fail_closed():
    assert placement(MODEL) == ("gpu_1x_h100_sxm5", "H100")
    assert placement("Qwen/Qwen3.6-35B-A3B") == ("gpu_1x_b200_sxm6", "B200")
    with pytest.raises(ValueError, match="only defined"):
        placement("deepseek-ai/DeepSeek-V4.1-Flash-1017")
    with pytest.raises(RuntimeError, match="no US capacity"):
        select_region({}, MODEL)
