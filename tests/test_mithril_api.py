"""Mithril wire contracts, bid ownership and cancellation without renting GPUs."""

import base64
import hashlib
import io
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from flash.providers.core.base import (
    AllocationConstraints,
    CapacityLookupError,
    UnreconciledCreateError,
)
from flash.providers.core.registry import get_provider
from flash.providers.mithril import api, bootstrap, client, config, prices, queue
from flash.providers.mithril.prices import Offer


@pytest.fixture(autouse=True)
def mithril_env(monkeypatch):
    monkeypatch.setenv("FLASH_MITHRIL_API_KEY", "test-secret")
    monkeypatch.setenv("FLASH_MITHRIL_PROJECT_ID", "proj_test")
    monkeypatch.delenv("FLASH_MITHRIL_REGION", raising=False)
    monkeypatch.delenv("FLASH_MITHRIL_IMAGE_VERSION", raising=False)
    monkeypatch.delenv("FLASH_MITHRIL_SSH_KEY_IDS", raising=False)


def payload():
    return {
        "run_id": "test",
        "attempt": 0,
        "phase": "sft",
        "hf_repo": "owner/repo",
        "job_spec_json": "{}",
        "source_snapshot": {},
        "env": {
            "FLASH_ARTIFACT_BLOB_SAS_URL": "https://example.blob.core.windows.net/private?sig=secret"
        },
    }


def owned_bid(**changes):
    labels = {
        "flash-owner": "flash-training",
        "flash-run": "test",
        "flash-attempt": "0",
        "flash-spec": hashlib.sha256(b"{}").hexdigest(),
        "flash-source": hashlib.sha256(b"{}").hexdigest(),
    }
    owner = api._OWNER_PREFIX + base64.b64encode(json.dumps(labels).encode()).decode()
    return {
        "fid": "bid_test",
        "name": "flash-test-a0",
        "project": "proj_test",
        "region": "us-central2-b",
        "status": "Allocated",
        "instances": ["inst_test"],
        "instance_type": "it_test",
        "instance_quantity": 1,
        "limit_price": "$2.51",
        "launch_specification": {"startup_script": "#!/bin/bash\n" + owner + "\n"},
        **changes,
    }


def test_configuration_requires_api_key_and_namespaces_scope(monkeypatch):
    assert config.configured()
    before = config.scope_fingerprint()
    monkeypatch.setenv("FLASH_MITHRIL_PROJECT_ID", "proj_other")
    assert config.scope_fingerprint() != before
    # A leftover region setting is ignored; it no longer changes ownership.
    monkeypatch.setenv("FLASH_MITHRIL_PROJECT_ID", "proj_test")
    monkeypatch.setenv("FLASH_MITHRIL_REGION", "us-central2-b")
    assert config.scope_fingerprint() == before
    monkeypatch.delenv("FLASH_MITHRIL_API_KEY")
    assert not config.configured()
    with pytest.raises(RuntimeError, match="Configure"):
        config.scope_fingerprint()


@pytest.mark.parametrize("status", [401, 422, 429, 500])
def test_http_auth_and_errors_never_expose_secrets(monkeypatch, status):
    def request(self, method, url, **kwargs):
        assert kwargs["headers"]["Authorization"] == "Bearer test-secret"
        assert url == "https://api.mithril.ai/v2/spot/bids"
        return httpx.Response(status, json={"detail": "test-secret"})

    monkeypatch.setattr(httpx.Client, "request", request)
    with pytest.raises(client.MithrilApiError) as error:
        client.call("POST", "/spot/bids", body={"startup_script": "private"})
    assert error.value.status == status
    assert "test-secret" not in str(error.value)


def test_http_timeout_remains_ambiguous_and_deadline_stops_requests(monkeypatch):
    calls = []

    def request(*args, **kwargs):
        calls.append(1)
        raise httpx.ReadTimeout("secret")

    monkeypatch.setattr(httpx.Client, "request", request)
    with pytest.raises(client.MithrilApiError) as error:
        client.call("POST", "/spot/bids")
    assert error.value.status == 0
    with pytest.raises(TimeoutError):
        client.call("POST", "/spot/bids", deadline_at=0)
    assert calls == [1]


def test_pagination_follows_cursor_and_rejects_cycles(monkeypatch):
    queries = []

    def call(*args, **kwargs):
        queries.append(dict(kwargs["params"]))
        return {"data": [{"fid": "bid_test"}], "next_cursor": "same"}

    monkeypatch.setattr(client, "call", call)
    with pytest.raises(client.MithrilApiError, match="cursor"):
        list(client.pages("/spot/bids", params={"project": "proj_test"}))
    assert queries[1]["next_cursor"] == "same"
    assert queries[1]["project"] == "proj_test"


def catalog(
    monkeypatch,
    *,
    price="$6.40",
    count=8,
    capacity=1,
    region="us-central2-b",
    socket="SXM5",
    gpu="H200",
    memory=141,
):
    shapes = [
        {
            "fid": "it_test",
            "gpu_type": gpu,
            "num_gpus": count,
            "gpu_memory_gb": memory,
            "gpu_socket": socket,
            "local_storage_gb": 1500,
        }
    ]
    auctions = [
        {
            "instance_type": "it_test",
            "region": region,
            "capacity": capacity,
            "last_instance_price": price,
        }
    ]
    monkeypatch.setattr(
        prices, "call", lambda method, path, **kw: shapes if path == "/instance-types" else auctions
    )


@pytest.mark.parametrize("count", [1, 8])
def test_exact_dollar_per_gpu_auction_is_admitted(monkeypatch, count):
    catalog(monkeypatch, price=f"${count:.2f}", count=count)
    assert prices.affordable_offers() == [
        Offer("H200", count, "it_test", 1.0, 1500, "us-central2-b")
    ]


def test_offers_use_only_advertised_shapes_and_per_gpu_prices(monkeypatch):
    catalog(monkeypatch)
    assert prices.affordable_offers() == [Offer("H200", 8, "it_test", 0.8, 1500, "us-central2-b")]


def test_allocation_offers_fitting_catalog_shapes_at_the_quote_rate(monkeypatch):
    # live prices are the queue's concern; allocation never reads them.
    monkeypatch.setattr(prices, "affordable_offers", lambda **kw: pytest.fail("no live prices"))
    candidates = get_provider("spot").live_candidates(80, AllocationConstraints(max_gpu_count=8))
    assert [(c.gpu, c.gpu_count, c.hourly_usd) for c in candidates] == [
        ("H100", 1, 2.5),
        ("H100", 8, 2.5),
        ("H200", 1, 2.5),
        ("H200", 8, 2.5),
    ]
    one = get_provider("spot").live_candidates(100, AllocationConstraints(max_gpu_count=1))
    assert [(c.gpu, c.gpu_count) for c in one] == [("H200", 1)]  # 100 GB only fits an H200


@pytest.mark.parametrize(
    "change",
    [
        {"price": "$20.08"},
        {"price": "$21.00"},
        {"capacity": 0},
        {"region": ""},
        {"socket": "PCIe"},
        {"gpu": "L40S"},
        {"count": 4},
        {"memory": 80},
    ],
)
def test_unfit_or_expensive_auctions_do_not_admit(monkeypatch, change):
    catalog(monkeypatch, **change)
    assert prices.affordable_offers() == []


@pytest.mark.parametrize("price", ["EUR 0.50", "$NaN", "$Infinity", "$-1", "$0", None])
def test_invalid_prices_fail_closed(monkeypatch, price):
    catalog(monkeypatch, price=price)
    with pytest.raises(CapacityLookupError):
        prices.affordable_offers()


def test_queue_waits_for_exact_count_and_disk_without_provisioning(monkeypatch):
    monkeypatch.setattr(queue, "quota_available", lambda *a, **kw: True)
    from flash.runner.lifecycle import status

    now = [100.0]
    states = []
    monkeypatch.setattr(queue.time, "time", lambda: now[0])
    monkeypatch.setattr(queue.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(status, "get_status", lambda _: SimpleNamespace(state="queued"))
    monkeypatch.setattr(status, "_update", lambda _, state: states.append(state))
    answers = iter(
        [
            [Offer("H200", 8, "it_8", 0.8, 1500, "us-central2-b")],
            [Offer("H200", 1, "it_1", 0.9, 100, "us-central2-b")],
            [Offer("H200", 1, "it_1", 0.9, 1500, "us-central2-b")],
        ]
    )
    monkeypatch.setattr(queue, "affordable_offers", lambda *a, **kw: next(answers))
    offer = queue.wait_for_price("H200", "test", disk_gb=500, deadline_at=300, log=io.StringIO())
    assert offer.instance_type == "it_1"
    assert now[0] == 220
    assert states == ["queued", "queued", "provisioning"]


def launch(monkeypatch, *, count=1, gpu="H200"):
    monkeypatch.setattr(
        bootstrap,
        "startup_script",
        lambda *a, owner_line, **kw: "#!/bin/bash\n" + owner_line + "\n",
    )
    return api.create(
        name="flash-test-a0",
        offer=Offer(gpu, count, "it_test", 0.8, 1500, "us-central2-b"),
        payload=payload(),
        image_reference="worker",
        gpu=gpu,
        gpu_count=count,
        capsule=b"capsule",
        disk_gb=500,
        deadline_at=time.time() + 100,
    )


@pytest.mark.parametrize("count", [1, 8])
@pytest.mark.parametrize(("gpu", "cap"), [("H100", 2.01), ("H200", 2.51)])
def test_create_bids_for_one_instance_at_per_instance_hold_cap(monkeypatch, count, gpu, cap):
    monkeypatch.setattr(api, "pages", lambda *a, **kw: [])
    captured = []

    def call(method, path, **kw):
        captured.append(kw["body"])
        return {"fid": "bid_test"}

    monkeypatch.setattr(api, "call", call)
    assert launch(monkeypatch, count=count, gpu=gpu) == "bid_test"
    body = captured[0]
    assert body["instance_quantity"] == 1
    assert body["limit_price"] == f"${cap * count:.2f}"
    assert body["launch_specification"]["volumes"] == []
    assert "api_key" not in json.dumps(body)
    assert api._ownership(body)["flash-run"] == "test"


@pytest.mark.parametrize("reconcile", [True, False])
def test_ambiguous_create_is_reconciled_without_a_second_post(monkeypatch, reconcile):
    listings = iter([[], [owned_bid()] if reconcile else []])
    monkeypatch.setattr(api, "pages", lambda *a, **kw: next(listings))
    posted = []

    def call(*a, **kw):
        posted.append(1)
        raise client.MithrilApiError("timeout")

    monkeypatch.setattr(api, "call", call)
    if reconcile:
        assert launch(monkeypatch) == "bid_test"
    else:
        with pytest.raises(UnreconciledCreateError):
            launch(monkeypatch)
    assert posted == [1]


@pytest.mark.parametrize(
    "change",
    [
        {"project": "foreign"},
        {"limit_price": "$3.00"},
        {"status": "Terminated"},
        {"instance_quantity": 2},
    ],
)
def test_existing_name_with_wrong_ownership_or_policy_is_never_adopted(monkeypatch, change):
    monkeypatch.setattr(api, "pages", lambda *a, **kw: [owned_bid(**change)])
    with pytest.raises(UnreconciledCreateError):
        launch(monkeypatch)


def test_foreign_bid_cannot_be_cancelled(monkeypatch):
    calls = []

    def call(method, path, **kw):
        calls.append(method)
        return owned_bid(project="foreign")

    monkeypatch.setattr(api, "call", call)
    with pytest.raises(client.MithrilApiError, match="does not belong"):
        api.terminate("bid_test")
    assert calls == ["GET"]


def test_cancel_waits_for_bid_and_instances_even_after_delete_timeout(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(api.time, "time", lambda: now[0])
    monkeypatch.setattr(api.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    bids = iter([owned_bid(), owned_bid(status="Terminated"), owned_bid(status="Terminated")])
    monkeypatch.setattr(api, "_bid", lambda *a, **kw: next(bids))
    rows = iter([[{"status": "STATUS_STOPPING"}], [{"status": "STATUS_TERMINATED"}]])
    monkeypatch.setattr(api, "_instances", lambda *a, **kw: next(rows))
    methods = []

    def call(method, *a, **kw):
        methods.append(method)
        raise client.MithrilApiError("timeout")

    monkeypatch.setattr(api, "call", call)
    api.terminate("bid_test")
    assert methods == ["DELETE"]
    assert now[0] == 102


@pytest.mark.parametrize(
    ("bid_status", "instance_status", "expected"),
    [
        ("Open", None, "starting"),
        ("Allocated", "STATUS_RUNNING", "running"),
        ("Preempting", "STATUS_PREEMPTING", "preempting"),
        ("Open", "STATUS_PREEMPTED", "preempting"),
        ("Allocated", "STATUS_RELOCATING", "preempting"),
        ("Terminated", "STATUS_STOPPING", "stopping"),
        ("Terminated", "STATUS_TERMINATED", "stopped"),
    ],
)
def test_bid_and_vm_states_drive_attempt_lifecycle(
    monkeypatch, bid_status, instance_status, expected
):
    monkeypatch.setattr(api, "_bid", lambda *a, **kw: owned_bid(status=bid_status))
    monkeypatch.setattr(
        api, "_instances", lambda *a, **kw: [{"status": instance_status}] if instance_status else []
    )
    assert api.get_instance("bid_test")["status"] == expected


@pytest.mark.parametrize(
    ("destination", "expected"),
    [
        ("ubuntu@8.8.4.4", "8.8.4.4"),
        ("ubuntu@8.8.4.4:22", "8.8.4.4"),
        ("ubuntu@10.0.0.5", None),
        ("invalid", None),
    ],
)
def test_public_address_comes_from_running_owned_instance(monkeypatch, destination, expected):
    monkeypatch.setattr(api, "_bid", lambda *a, **kw: owned_bid())
    monkeypatch.setattr(
        api,
        "_instances",
        lambda *a, **kw: [{"status": "STATUS_RUNNING", "ssh_destination": destination}],
    )
    assert api.public_ip("bid_test") == expected


def test_handle_rejects_legacy_provider_identity():
    from flash.providers.mithril.handles import SpotJobHandle

    with pytest.raises(ValueError, match="invalid Mithril Spot"):
        SpotJobHandle._coerce_instance_id("computeinstance-old")


def test_startup_script_offloads_large_payload_and_compiles(monkeypatch):
    published = []

    def publish(payload, data, suffix, **kw):
        published.append((data, suffix))
        return {"blob": "private/" + suffix, "sha256": hashlib.sha256(data).hexdigest()}

    monkeypatch.setattr(bootstrap, "_publish", publish)
    raw = payload()
    raw["env"]["LARGE"] = "x" * 100_000
    script = bootstrap.startup_script(
        raw, b"capsule", "worker", owner_line="# owner", deadline_at=time.time() + 100
    )
    assert len(script) < 10_000
    assert raw["env"]["LARGE"] not in script
    assert [suffix for _, suffix in published] == ["pyz", "py"]
    compile(published[1][0], "launch.py", "exec")
    compile(
        script.split("<<'FLASH_BOOTSTRAP'\n")[1].split("\nFLASH_BOOTSTRAP")[0],
        "download.py",
        "exec",
    )
    assert "If-None-Match" in published[1][0].decode()
    assert "test-secret" not in script


def test_missing_bid_does_not_hide_a_still_stopping_instance(monkeypatch):
    monkeypatch.setattr(api, "_bid", lambda *a, **kw: None)
    monkeypatch.setattr(api, "_instances", lambda *a, **kw: [{"status": "STATUS_STOPPING"}])
    assert api.get_instance("bid_test")["status"] == "stopping"


@pytest.mark.parametrize("valid", [True, False])
def test_private_launch_download_checks_digest_before_execution(monkeypatch, tmp_path, valid):
    import subprocess
    import urllib.request

    data = b"print('launch')"
    ref = {
        "sas": "https://example.blob.core.windows.net/private?sig=secret",
        "blob": "repo/launch.py",
        "sha256": hashlib.sha256(data).hexdigest() if valid else "wrong",
    }
    seen = []

    def download(url, timeout):
        assert url == "https://example.blob.core.windows.net/private/repo/launch.py?sig=secret"
        return io.BytesIO(data)

    monkeypatch.setattr(urllib.request, "urlopen", download)
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: seen.append(command))
    monkeypatch.setattr("os.umask", lambda value: None)
    code = bootstrap.DOWNLOAD.replace("'/opt/flash'", repr(str(tmp_path)))
    if valid:
        exec(code, {"REF": ref})
        assert (tmp_path / "launch.py").read_bytes() == data
        assert (tmp_path / "launch.py").stat().st_mode & 0o777 == 0o600
        assert len(seen) == 1
    else:
        with pytest.raises(SystemExit, match="download failed"):
            exec(code, {"REF": ref})
        assert seen == []


@pytest.mark.parametrize("expired", [True, False])
def test_worker_never_starts_on_expired_deadline_or_duplicate_claim(monkeypatch, expired, capsys):
    import subprocess
    import urllib.error
    import urllib.request

    commands = []
    requests = []

    def claim(request, timeout):
        requests.append(request)
        assert request.get_method() == "PUT"
        assert request.get_header("If-none-match") == "*"
        raise urllib.error.HTTPError(request.full_url, 412, "secret", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", claim)
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: commands.append(command))
    monkeypatch.setattr("os.umask", lambda value: None)
    bundle = {
        "payload": payload(),
        "deadline_at": 0 if expired else time.time() + 100,
        "claim_blob": "repo/run/claim",
    }
    exec(bootstrap.WORKER, {"BUNDLE": bundle})
    assert commands == [["shutdown", "-h", "now"]]
    assert len(requests) == (0 if expired else 1)
    assert "secret" not in capsys.readouterr().out


@pytest.mark.parametrize("algorithm", ["sft", "custom"])
def test_worker_launches_capsule_on_ephemeral_docker_storage(monkeypatch, tmp_path, algorithm):
    import pathlib
    import subprocess
    import urllib.request

    root = tmp_path / "flash"
    root.mkdir()
    daemon = tmp_path / "daemon.json"
    daemon.write_text('{"default-runtime":"nvidia"}')
    capsule = b"test capsule"
    raw = payload()
    raw["job_spec_json"] = json.dumps({"algorithm": algorithm})
    bundle = {
        "payload": raw,
        "deadline_at": time.time() + 3600,
        "claim_blob": "repo/run/claim",
        "capsule": {"blob": "repo/capsule", "sha256": hashlib.sha256(capsule).hexdigest()},
        "image": "worker@sha256:digest",
    }

    def download(request, timeout):
        return io.BytesIO(b"" if isinstance(request, urllib.request.Request) else capsule)

    commands = []
    forwarded = []
    monkeypatch.setattr(urllib.request, "urlopen", download)
    monkeypatch.setattr(pathlib.Path, "is_mount", lambda _: True)
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: commands.append(command))
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kw: forwarded.append(command))
    monkeypatch.setattr("os.umask", lambda _: None)
    code = bootstrap.WORKER.replace("'/opt/flash'", repr(str(root))).replace(
        "'/etc/docker/daemon.json'", repr(str(daemon))
    )
    exec(code, {"BUNDLE": bundle})
    assert json.loads(daemon.read_text()) == {
        "default-runtime": "nvidia",
        "data-root": "/mnt/local/docker",
    }
    command = commands[1]
    assert command[:3] == ["timeout", "--signal=TERM", "--kill-after=30"]
    assert 0 < int(command[3]) <= 3600
    assert "worker@sha256:digest" in command
    assert command[-1].endswith("exec python -u /opt/flash/capsule.pyz bootstrap")
    assert (root / "capsule.pyz").read_bytes() == capsule
    assert json.loads((root / "payload.env").read_text().split("=", 1)[1]) == raw
    assert ("-p" in command) == (algorithm == "custom")
    assert bool(forwarded) == (algorithm == "custom")
    assert commands[-1] == ["shutdown", "-h", "now"]


@pytest.mark.parametrize(
    ("states", "remaining"),
    [
        (["STATUS_RUNNING"], True),
        (["STATUS_STOPPING"], True),
        (["STATUS_TERMINATED", "STATUS_STOPPING"], True),
        (["STATUS_UNKNOWN"], True),
        (["STATUS_TERMINATED"], False),
        (["STATUS_PREEMPTED", "STATUS_PAUSED", "STATUS_REPLACED", "STATUS_ERROR"], False),
        ([], False),
    ],
)
def test_terminated_bids_remain_discoverable_until_every_vm_stops(monkeypatch, states, remaining):
    deadline = time.time() + 30

    def pages(path, *, params, deadline_at):
        assert deadline_at == deadline
        assert params["project"] == "proj_test"
        if path == "/spot/bids":
            return [owned_bid(status="Terminated")]
        assert path == "/instances"
        assert params["bid_fid_in"] == "bid_test"
        return [{"bid": "bid_test", "project": "proj_test", "status": state} for state in states]

    monkeypatch.setattr(api, "pages", pages)
    found = api.list_instances(deadline_at=deadline)
    assert [item["metadata"]["id"] for item in found] == (["bid_test"] if remaining else [])


def test_instance_shutdown_checks_ignore_unowned_bids(monkeypatch):
    monkeypatch.setattr(
        api, "pages", lambda *a, **kw: [owned_bid(status="Terminated", project="foreign")]
    )
    monkeypatch.setattr(
        api, "_instances", lambda *a, **kw: pytest.fail("must not inspect foreign resources")
    )
    assert api.list_instances() == []


def test_handleless_recovery_blocks_until_terminated_bid_instances_stop(monkeypatch):
    from flash.providers.core import registry
    from flash.providers.modal import auth
    from flash.server.platform import runtime

    provider = get_provider("spot")
    monkeypatch.setattr(registry, "configured_providers", lambda: [provider])
    monkeypatch.setattr(
        runtime, "get_status", lambda _: SimpleNamespace(submitted_instance_providers=["spot"])
    )
    monkeypatch.setattr(auth, "require_recorded_scope", lambda _: None)
    monkeypatch.setattr(api, "pages", lambda *a, **kw: [owned_bid(status="Terminated")])
    current = ["STATUS_STOPPING"]

    def instances(*args, **kwargs):
        if current[0] == "unavailable":
            raise api.MithrilApiError("instance lookup failed")
        return [{"status": current[0]}]

    monkeypatch.setattr(api, "_instances", instances)
    cancelled = []

    def terminate(bid_id):
        cancelled.append(bid_id)
        raise api.MithrilApiError("shutdown not confirmed")

    monkeypatch.setattr(api, "terminate", terminate)
    spec = SimpleNamespace(run_id="test")
    assert provider.run_instances_remaining(spec.run_id) == ["bid_test"]
    assert runtime._confirm_run_clear(spec) is False
    assert cancelled == ["bid_test"]
    current[0] = "unavailable"
    assert runtime._confirm_run_clear(spec) is False
    current[0] = "STATUS_TERMINATED"
    assert runtime._confirm_run_clear(spec) is True
    assert provider.run_instances_remaining(spec.run_id) == []


def test_orphan_sweep_reaps_terminated_bid_with_stopping_instance(monkeypatch):
    monkeypatch.setattr(api, "pages", lambda *a, **kw: [owned_bid(status="Terminated")])
    monkeypatch.setattr(api, "_instances", lambda *a, **kw: [{"status": "STATUS_STOPPING"}])
    terminated = []
    monkeypatch.setattr(api, "terminate", terminated.append)
    assert get_provider("spot").sweep_orphans(active_labels=set(), known_labels={"test"}) == [
        "bid_test"
    ]
    assert terminated == ["bid_test"]


@pytest.mark.parametrize("status", [400, 403, 409, 422, 429, 0, 500])
def test_capacity_refusal_requeues_but_unknown_create_outcomes_do_not(monkeypatch, status):
    from flash.providers.core.base import CapacityUnavailableError
    from flash.providers.mithril import capacity

    monkeypatch.setattr(api, "pages", lambda *a, **kw: [])
    monkeypatch.setattr(capacity, "quota_available", lambda *a, **kw: False)

    def refuse(*args, **kwargs):
        raise client.MithrilApiError(status=status)

    monkeypatch.setattr(api, "call", refuse)
    error = UnreconciledCreateError if status in {0, 500} else CapacityUnavailableError
    with pytest.raises(error):
        launch(monkeypatch)


def test_refused_create_reconciles_existing_bid_before_capacity_retry(monkeypatch):
    from flash.providers.mithril import capacity

    listings = iter([[], [owned_bid()]])
    monkeypatch.setattr(api, "pages", lambda *a, **kw: next(listings))
    monkeypatch.setattr(capacity, "quota_available", lambda *a, **kw: pytest.fail("existing bid"))

    def refuse(*args, **kwargs):
        raise client.MithrilApiError(status=429)

    monkeypatch.setattr(api, "call", refuse)
    assert launch(monkeypatch) == "bid_test"


# --- every region (no region setting) ---------------------------------------------------------


def test_offers_cover_every_region_and_keep_each_region(monkeypatch):
    shapes = [
        {
            "fid": "it_test",
            "gpu_type": "H200",
            "num_gpus": 1,
            "gpu_memory_gb": 141,
            "gpu_socket": "SXM5",
            "local_storage_gb": 1500,
        }
    ]
    auctions = [
        {"instance_type": "it_test", "region": r, "capacity": 3, "last_instance_price": p}
        for r, p in (("us-central2-b", "$3.50"), ("ap-x", "$0.90"), ("eu-x", "$1.00"))
    ]
    monkeypatch.setattr(
        prices, "call", lambda method, path, **kw: shapes if path == "/instance-types" else auctions
    )
    assert {o.region for o in prices.affordable_offers()} == {"ap-x", "eu-x"}
    # An excluded region is never offered, however cheap.
    auctions.append(
        {
            "instance_type": "it_test",
            "region": "af-south1-a",
            "capacity": 3,
            "last_instance_price": "$0.50",
        }
    )
    assert {o.region for o in prices.affordable_offers()} == {"ap-x", "eu-x"}


def test_queue_picks_the_cheapest_region(monkeypatch):
    monkeypatch.setattr(queue, "quota_available", lambda *a, **kw: True)
    monkeypatch.setattr(
        queue,
        "affordable_offers",
        lambda **kw: [
            Offer("H200", 1, "it_a", 1.0, 1500, "eu-x"),
            Offer("H200", 1, "it_b", 0.9, 1500, "af-south1-a"),
        ],
    )
    from flash.runner.lifecycle import status

    monkeypatch.setattr(status, "get_status", lambda run: SimpleNamespace(state="queued"))
    monkeypatch.setattr(status, "_update", lambda run, state: None)
    offer = queue.wait_for_price("H200", "test", deadline_at=time.time() + 60, log=io.StringIO())
    assert offer.region == "af-south1-a"


def test_bid_uses_the_offer_region(monkeypatch):
    monkeypatch.setattr(api, "pages", lambda *a, **kw: [])
    captured = []
    monkeypatch.setattr(
        api, "call", lambda m, p, **kw: captured.append(kw["body"]) or {"fid": "bid_x"}
    )
    monkeypatch.setattr(
        bootstrap,
        "startup_script",
        lambda *a, owner_line, **kw: "#!/bin/bash\n" + owner_line + "\n",
    )
    assert (
        api.create(
            name="flash-test-a0",
            offer=Offer("H200", 1, "it_test", 0.9, 1500, "af-south1-a"),
            payload=payload(),
            image_reference="worker",
            gpu="H200",
            gpu_count=1,
            capsule=b"capsule",
            disk_gb=500,
            deadline_at=time.time() + 100,
        )
        == "bid_x"
    )
    assert captured[-1]["region"] == "af-south1-a"


def test_ownership_is_the_project_in_any_region():
    assert api._ownership(owned_bid(region="af-south1-a")) is not None
    assert api._ownership(owned_bid(region="")) is None
    assert api._ownership(owned_bid(project="other")) is None


def test_attempts_from_region_pinned_releases_stay_recognized(monkeypatch):
    from flash.providers.mithril import handles

    legacy = config.region_scope_fingerprint("proj_test", "us-central2-b")
    handle = SimpleNamespace(scope=legacy, instance_id="bid_test")
    monkeypatch.setattr(
        client, "call", lambda m, p, **kw: {"project": "proj_test", "region": "us-central2-b"}
    )
    handles.require_scope(handle)  # verified against its own bid's region
    monkeypatch.setattr(
        client, "call", lambda m, p, **kw: {"project": "proj_test", "region": "eu-x"}
    )
    with pytest.raises(RuntimeError, match="differ"):
        handles.require_scope(handle)
    handles.require_scope(SimpleNamespace(scope=config.scope_fingerprint(), instance_id="bid_x"))


def test_a_failed_scope_lookup_is_retryable_not_a_mismatch(monkeypatch):
    from flash.providers.mithril import handles

    legacy = config.region_scope_fingerprint("proj_test", "us-central2-b")
    handle = SimpleNamespace(scope=legacy, instance_id="bid_test")
    for status in (0, 429, 503):

        def fail(m, p, status=status, **kw):
            raise client.MithrilApiError(status=status)

        monkeypatch.setattr(client, "call", fail)
        with pytest.raises(client.MithrilApiError):
            handles.require_scope(handle)

    def gone(m, p, **kw):
        raise client.MithrilApiError(status=404)

    monkeypatch.setattr(client, "call", gone)
    with pytest.raises(RuntimeError, match="differ"):
        handles.require_scope(handle)


def test_offers_skip_a_region_cooling_down_after_a_boot_timeout(monkeypatch):
    from flash.providers.mithril import region_health

    shapes = [
        {
            "fid": "it_test",
            "gpu_type": "H200",
            "num_gpus": 1,
            "gpu_memory_gb": 141,
            "gpu_socket": "SXM5",
            "local_storage_gb": 1500,
        }
    ]
    auctions = [
        {"instance_type": "it_test", "region": r, "capacity": 3, "last_instance_price": "$2.00"}
        for r in ("af-south1-a", "us-central2-b")
    ]
    monkeypatch.setattr(
        prices, "call", lambda method, path, **kw: shapes if path == "/instance-types" else auctions
    )
    region_health.mark_unbootable("af-south1-a")
    assert {o.region for o in prices.affordable_offers()} == {"us-central2-b"}
