"""Durable ownership and orphan cleanup for one Lambda VM per Modal CPU replica."""

import re
import time

from .config import LEASE_SECONDS, SSH_KEY_NAME, select_region
from .resources import cache_filesystem, firewall


class Fleet:
    def __init__(self, api, ledger, environment, *, clock=time.time):
        if environment not in {"main", "dev"}:
            raise ValueError("Lambda fleet must use the main or dev environment")
        self.api, self.ledger, self.environment, self.clock = api, ledger, environment, clock

    def name(self, owner):
        if not re.fullmatch(r"[0-9a-f]{32}", owner):
            raise ValueError("invalid Lambda replica owner")
        return f"flash-{self.environment}-{owner}"

    def owns(self, instance, owner):
        tags = {tag["key"]: tag["value"] for tag in instance.get("tags", [])}
        return (
            instance.get("name") == self.name(owner)
            and tags.get("flash-owner") == owner
            and tags.get("flash-environment") == self.environment
        )

    async def launch(self, owner, model, user_data, revision):
        name = self.name(owner)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Lambda launch requires a source revision")
        previous = await self.ledger.get(owner)
        if previous is not None:
            # Even an ambiguous launch is at-most-once. Reconciliation discovers it later.
            if (
                previous["model"] != model
                or previous["revision"] != revision
                or previous.get("released")
            ):
                raise ValueError("Lambda owner already used")
            return await self.inspect(owner)
        await self._check_capacity(model, revision)
        instance_type, region = select_region(await self.api.call("GET", "instance-types"), model)
        record = {
            "model": model,
            "revision": revision,
            "created": self.clock(),
            "heartbeat": self.clock(),
            "instance_id": None,
        }
        firewall_id = await firewall(self.api, self.environment, region)
        filesystem_id = await cache_filesystem(self.api, self.environment, region)
        await self.ledger.put(owner, record)
        # A process failure after this write must never cause a blind second launch.
        result = await self.api.call(
            "POST",
            "instance-operations/launch",
            {
                "region_name": region,
                "instance_type_name": instance_type,
                "ssh_key_names": [SSH_KEY_NAME],
                "firewall_rulesets": [{"id": firewall_id}],
                "file_system_mounts": [
                    {"file_system_id": filesystem_id, "mount_point": "/vol/hosting-cache"}
                ],
                "name": name,
                "image": {"family": "gpu-base-24-04"},
                "user_data": user_data,
                "tags": [
                    {"key": "flash-owner", "value": owner},
                    {"key": "flash-environment", "value": self.environment},
                ],
            },
        )
        ids = result.get("instance_ids", [])
        if len(ids) != 1:
            raise RuntimeError("Lambda launch did not return exactly one instance")
        record["instance_id"] = ids[0]
        await self.ledger.put(owner, record)
        return {"id": ids[0], "status": "booting"}

    async def _check_capacity(self, model, revision):
        # Modal normally enforces this. Include uncertain/orphan launches so a
        # replacement CPU replica cannot silently allocate a third GPU.
        instances = await self.api.call("GET", "instances")
        occupied = 0
        async for owner, record in self.ledger.items():
            if record["model"] != model or record["revision"] != revision:
                continue
            live = any(
                self.owns(instance, owner) and instance["status"] != "terminated"
                for instance in instances
            )
            uncertain = (
                not record.get("instance_id")
                and self.clock() - record["heartbeat"] <= LEASE_SECONDS
            )
            if not record.get("released") or live or uncertain:
                occupied += 1
        if occupied >= 2:
            raise RuntimeError("Lambda model/release replica limit reached; pending cleanup")

    async def inspect(self, owner):
        record = await self.ledger.get(owner)
        if record is None or record.get("released"):
            raise RuntimeError("Lambda replica has no live ownership lease")
        if record.get("instance_id"):
            instance = await self.api.call("GET", f"instances/{record['instance_id']}")
            if not self.owns(instance, owner):
                raise RuntimeError("Lambda instance ownership mismatch")
            return instance
        instances = await self.api.call("GET", "instances")
        found = [instance for instance in instances if self.owns(instance, owner)]
        if len(found) > 1:
            raise RuntimeError("duplicate Lambda ownership")
        if found:
            record["instance_id"] = found[0]["id"]
            await self.ledger.put(owner, record)
            return found[0]
        return {"status": "pending"}

    async def heartbeat(self, owner):
        self.name(owner)
        record = await self.ledger.get(owner)
        if record is None or record.get("released"):
            raise RuntimeError("Lambda replica lease has been released")
        record["heartbeat"] = self.clock()
        await self.ledger.put(owner, record)

    async def release(self, owner):
        self.name(owner)
        record = await self.ledger.get(owner)
        if record is None:
            return
        record["released"] = True
        await self.ledger.put(owner, record)
        # Reconcile by ownership, including a launch whose response was lost.
        instances = await self.api.call("GET", "instances")
        await self._terminate_owned(instances, owner)

    async def _terminate_owned(self, instances, owner):
        ids = [
            instance["id"]
            for instance in instances
            if self.owns(instance, owner)
            and instance["status"] not in {"terminating", "terminated"}
        ]
        if ids:
            await self.api.call("POST", "instance-operations/terminate", {"instance_ids": ids})

    async def reap(self):
        instances = await self.api.call("GET", "instances")
        count = 0
        async for owner, record in self.ledger.items():
            if not record.get("released") and self.clock() - record["heartbeat"] <= LEASE_SECONDS:
                continue
            # A snapshot may span a heartbeat; reread before claiming an expired lease.
            record = await self.ledger.get(owner)
            if not record.get("released") and self.clock() - record["heartbeat"] <= LEASE_SECONDS:
                continue
            if (
                record.get("released")
                and self.clock() - record["heartbeat"] > 86400
                and not any(self.owns(instance, owner) for instance in instances)
            ):
                await self.ledger.delete(owner)
                continue
            record["released"] = True
            await self.ledger.put(owner, record)
            await self._terminate_owned(instances, owner)
            count += 1
        return {"expired_owners": count}
