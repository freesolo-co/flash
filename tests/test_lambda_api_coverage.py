"""Lambda Cloud REST client behavior: envelope unwrap, instance-type cache/TTL, capacity &
pricing readers, non-idempotent launch, filesystem asymmetry, per-id terminate isolation.

CPU-only + offline: most tests replace the module-level ``request_with_retries`` seam (every api
function reaches the network through that single global), so no key and no urlopen are needed. Two
tests exercise the REAL wrapper: one asserts the missing-key error (the autouse offline fixture
deletes LAMBDA_API_KEY), one drives urlopen to check the Cloudflare User-Agent + envelope unwrap.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _reset_types_cache():
    """The instance-type cache is a module global; reset it around every test for determinism."""
    from flash.providers.lambda_.client import api as lambda_api

    lambda_api._types_cache.update(ts=0.0, data=None)
    yield
    lambda_api._types_cache.update(ts=0.0, data=None)


class _FakeResponse:
    """Minimal urlopen() stand-in: context manager whose read() yields a JSON body."""

    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _seed_types(monkeypatch, data):
    """Prime the instance-type cache fresh (ts == now) so list_instance_types serves it without a
    request — lets the capacity/pricing readers run against a fixed catalog."""
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(lambda_api.time, "time", lambda: 1000.0)
    lambda_api._types_cache.update(ts=1000.0, data=data)
    return lambda_api


# ---------------------------------------------------------------------------
# _data envelope unwrap
# ---------------------------------------------------------------------------
def test_data_unwraps_only_the_data_envelope():
    from flash.providers.lambda_.client.api import _data

    # a {"data": ...} envelope is unwrapped to its payload (even a falsy one)
    assert _data({"data": [1, 2]}) == [1, 2]
    assert _data({"data": None}) is None
    # anything without a "data" key passes straight through, unchanged
    assert _data([1, 2]) == [1, 2]
    assert _data({"other": 1}) == {"other": 1}
    assert _data("raw") == "raw"


# ---------------------------------------------------------------------------
# list_instance_types: cache-hit / force / TTL expiry / bad shape
# ---------------------------------------------------------------------------
def test_list_instance_types_caches_forces_and_expires(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    clock = {"t": 1000.0}
    monkeypatch.setattr(lambda_api.time, "time", lambda: clock["t"])
    calls = {"n": 0}

    def fake(path, **kw):
        calls["n"] += 1
        return {"data": {"gpu_1x_a10": {"regions_with_capacity_available": []}}}

    monkeypatch.setattr(lambda_api, "request_with_retries", fake)

    first = lambda_api.list_instance_types()
    assert "gpu_1x_a10" in first
    assert calls["n"] == 1
    # a second call inside the TTL is served from cache (no new request)
    lambda_api.list_instance_types()
    assert calls["n"] == 1
    # force=True bypasses the fresh cache
    lambda_api.list_instance_types(force=True)
    assert calls["n"] == 2
    # once the TTL (45s) lapses, even a non-forced call refetches
    clock["t"] = 1000.0 + 46.0
    lambda_api.list_instance_types()
    assert calls["n"] == 3


def test_list_instance_types_rejects_non_dict_payload(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(lambda_api.time, "time", lambda: 1000.0)
    monkeypatch.setattr(
        lambda_api, "request_with_retries", lambda path, **kw: {"data": ["not", "a", "dict"]}
    )
    with pytest.raises(lambda_api.LambdaApiError, match="unexpected /instance-types response"):
        lambda_api.list_instance_types(force=True)


# ---------------------------------------------------------------------------
# capacity readers
# ---------------------------------------------------------------------------
def test_regions_with_capacity_filters_names_and_unknown_type(monkeypatch):
    lambda_api = _seed_types(
        monkeypatch,
        {
            "gpu_1x_a10": {
                "regions_with_capacity_available": [
                    {"name": "us-east-1"},
                    {"name": None},  # null name dropped
                    {"description": "no name key"},  # missing name dropped
                    {"name": "us-west-2"},
                ]
            }
        },
    )
    assert lambda_api.regions_with_capacity("gpu_1x_a10") == ["us-east-1", "us-west-2"]
    # an unknown instance type -> the `or {}` guard -> empty, never a KeyError
    assert lambda_api.regions_with_capacity("does-not-exist") == []


def test_all_regions_unions_sorts_and_dedups(monkeypatch):
    lambda_api = _seed_types(
        monkeypatch,
        {
            "gpu_1x_a10": {
                "regions_with_capacity_available": [{"name": "us-west-2"}, {"name": "us-east-1"}]
            },
            "gpu_8x_h100": {
                "regions_with_capacity_available": [
                    {"name": "us-east-1"},  # duplicate across types
                    {"name": None},  # dropped
                    {"name": "eu-central-1"},
                ]
            },
            "gpu_broken": None,  # None info -> (info or {}) guard
            "gpu_empty": {},  # no regions key
        },
    )
    assert lambda_api.all_regions() == ["eu-central-1", "us-east-1", "us-west-2"]


def test_instance_type_price_usd_hr_variants(monkeypatch):
    lambda_api = _seed_types(
        monkeypatch,
        {
            "gpu_1x_a10": {"instance_type": {"price_cents_per_hour": 129}},
            "gpu_free": {"instance_type": {"price_cents_per_hour": 0}},  # 0 -> falsy -> None
            "gpu_noprice": {"instance_type": {}},  # no price key
            "gpu_notype": {},  # no instance_type key
        },
    )
    assert lambda_api.instance_type_price_usd_hr("gpu_1x_a10") == 1.29
    assert lambda_api.instance_type_price_usd_hr("gpu_free") is None
    assert lambda_api.instance_type_price_usd_hr("gpu_noprice") is None
    assert lambda_api.instance_type_price_usd_hr("gpu_notype") is None
    assert lambda_api.instance_type_price_usd_hr("missing") is None


# ---------------------------------------------------------------------------
# list_ssh_keys / list_filesystems / list_instances: list-or-empty + paths
# ---------------------------------------------------------------------------
def test_list_ssh_keys_returns_list_or_empty(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(
        lambda_api, "request_with_retries", lambda path, **kw: {"data": [{"name": "jk"}]}
    )
    assert lambda_api.list_ssh_keys() == [{"name": "jk"}]
    # a non-list payload collapses to [] (never surfaced as a dict to callers)
    monkeypatch.setattr(
        lambda_api, "request_with_retries", lambda path, **kw: {"data": {"not": "a list"}}
    )
    assert lambda_api.list_ssh_keys() == []


def test_list_filesystems_uses_hyphenated_path_and_list_or_empty(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    seen = {}

    def fake(path, **kw):
        seen["path"] = path
        return {"data": [{"id": "fs-1", "name": "w"}]}

    monkeypatch.setattr(lambda_api, "request_with_retries", fake)
    assert lambda_api.list_filesystems() == [{"id": "fs-1", "name": "w"}]
    assert seen["path"] == "/file-systems"  # LIST is hyphenated (asymmetric with create/delete)
    monkeypatch.setattr(lambda_api, "request_with_retries", lambda path, **kw: {"data": {"x": 1}})
    assert lambda_api.list_filesystems() == []


def test_list_instances_returns_list_or_empty(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(
        lambda_api, "request_with_retries", lambda path, **kw: {"data": [{"id": "i-1"}]}
    )
    assert lambda_api.list_instances() == [{"id": "i-1"}]
    monkeypatch.setattr(lambda_api, "request_with_retries", lambda path, **kw: {"data": {"x": 1}})
    assert lambda_api.list_instances() == []


# ---------------------------------------------------------------------------
# launch_instance: non-idempotent POST, body shape, missing-id guard
# ---------------------------------------------------------------------------
def test_launch_instance_builds_body_and_returns_first_id(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    seen = {}

    def fake(path, method="GET", body=None, retries=4, base_delay=2.0):
        seen.update(path=path, method=method, body=body, retries=retries)
        return {"data": {"instance_ids": ["i-abc"]}}

    monkeypatch.setattr(lambda_api, "request_with_retries", fake)
    iid = lambda_api.launch_instance(
        region_name="us-east-1",
        instance_type_name="gpu_1x_a10",
        ssh_key_names=["jk"],
        name="flash-x",
        user_data="#cloud-config",
        file_system_names=["flash-weights"],
    )
    assert iid == "i-abc"  # first id, stringified
    assert seen["path"] == "/instance-operations/launch"
    assert seen["method"] == "POST"
    assert seen["retries"] == 0  # NON-IDEMPOTENT: never retried (blind retry = double-provision)
    assert seen["body"] == {
        "region_name": "us-east-1",
        "instance_type_name": "gpu_1x_a10",
        "ssh_key_names": ["jk"],
        "name": "flash-x",
        "quantity": 1,
        "user_data": "#cloud-config",
        "file_system_names": ["flash-weights"],
    }

    # No filesystems -> the key is omitted entirely (not sent as None/[]).
    seen.clear()
    lambda_api.launch_instance(
        region_name="us-east-1",
        instance_type_name="gpu_1x_a10",
        ssh_key_names=["jk"],
        name="flash-x",
        user_data="#cloud-config",
    )
    assert "file_system_names" not in seen["body"]


def test_launch_instance_raises_when_no_instance_id(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    kwargs = {
        "region_name": "us-east-1",
        "instance_type_name": "gpu_1x_a10",
        "ssh_key_names": ["jk"],
        "name": "flash-x",
        "user_data": "#cloud-config",
    }
    # success-shaped dict but no instance_ids -> LambdaApiError
    monkeypatch.setattr(lambda_api, "request_with_retries", lambda *a, **k: {"data": {}})
    with pytest.raises(lambda_api.LambdaApiError, match="returned an invalid instance identity"):
        lambda_api.launch_instance(**kwargs)
    # a non-dict payload (ids can't be read) -> same guard
    monkeypatch.setattr(lambda_api, "request_with_retries", lambda *a, **k: {"data": []})
    with pytest.raises(lambda_api.LambdaApiError, match="returned an invalid instance identity"):
        lambda_api.launch_instance(**kwargs)


# ---------------------------------------------------------------------------
# filesystem create/delete/ensure (path asymmetry + idempotent ensure)
# ---------------------------------------------------------------------------
def test_create_filesystem_posts_to_unhyphenated_path_and_dict_or_empty(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    seen = {}
    deadline_at = 1_000_000_000_000.0

    def fake(path, method="GET", body=None, retries=4, base_delay=2.0, deadline_at=None):
        seen.update(
            path=path,
            method=method,
            body=body,
            retries=retries,
            deadline_at=deadline_at,
        )
        return {"data": {"mount_point": "/lambda/nfs/w"}}

    monkeypatch.setattr(lambda_api, "request_with_retries", fake)
    out = lambda_api.create_filesystem("w", "us-east-1", deadline_at=deadline_at)
    assert out == {"mount_point": "/lambda/nfs/w"}
    assert seen["path"] == "/filesystems"  # CREATE is NOT hyphenated
    assert seen["method"] == "POST"
    assert seen["body"] == {"name": "w", "region": "us-east-1"}
    assert seen["retries"] == 0
    assert seen["deadline_at"] == deadline_at
    monkeypatch.setattr(lambda_api, "request_with_retries", lambda *a, **k: {"data": ["x"]})
    with pytest.raises(lambda_api.LambdaApiError, match="returned no verifiable object"):
        lambda_api.create_filesystem("w", "us-east-1", deadline_at=deadline_at)


def test_delete_filesystem_true_on_success_false_on_error(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    seen = {}

    def ok(path, method="GET", body=None, retries=4, base_delay=2.0):
        seen.update(path=path, method=method)
        return {}

    monkeypatch.setattr(lambda_api, "request_with_retries", ok)
    assert lambda_api.delete_filesystem("fs-1") is True
    assert seen["path"] == "/filesystems/fs-1"
    assert seen["method"] == "DELETE"

    def boom(*a, **k):
        raise lambda_api.LambdaApiError("filesystem quota exceeded")

    monkeypatch.setattr(lambda_api, "request_with_retries", boom)
    assert lambda_api.delete_filesystem("fs-1") is False  # best-effort: swallows + logs


def test_ensure_filesystem_reuses_same_name_and_region(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    deadline_at = 1_000_000_000_000.0
    monkeypatch.setattr(
        lambda_api,
        "list_filesystems",
        lambda **_kwargs: [
            {"name": "other", "region": {"name": "us-east-1"}, "mount_point": "/other"},
            {"name": "w", "region": {"name": "us-west-2"}, "mount_point": "/wrong-region"},
            {"name": "w", "region": None, "mount_point": "/null-region"},  # (region or {}) guard
            {"name": "w", "region": {"name": "us-east-1"}, "mount_point": "/lambda/nfs/w"},
        ],
    )

    def no_create(*a, **k):
        raise AssertionError("must reuse an existing same-name/region filesystem, not create")

    monkeypatch.setattr(lambda_api, "create_filesystem", no_create)
    assert (
        lambda_api.ensure_filesystem("w", "us-east-1", deadline_at=deadline_at) == "/lambda/nfs/w"
    )

    # a matching filesystem with no mount_point falls back to the default host path
    monkeypatch.setattr(
        lambda_api,
        "list_filesystems",
        lambda **_kwargs: [{"name": "w", "region": {"name": "us-east-1"}}],
    )
    assert (
        lambda_api.ensure_filesystem("w", "us-east-1", deadline_at=deadline_at) == "/lambda/nfs/w"
    )


def test_ensure_filesystem_creates_when_absent(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    deadline_at = 1_000_000_000_000.0
    monkeypatch.setattr(lambda_api, "list_filesystems", lambda **_kwargs: [])
    created = {}
    monkeypatch.setattr(
        lambda_api,
        "create_filesystem",
        lambda n, r, *, deadline_at: (
            created.update(n=n, r=r, deadline_at=deadline_at) or {"mount_point": "/mnt/new"}
        ),
    )
    assert lambda_api.ensure_filesystem("w", "us-east-1", deadline_at=deadline_at) == "/mnt/new"
    assert created == {"n": "w", "r": "us-east-1", "deadline_at": deadline_at}
    # a create that returns no mount_point falls back to the default host path
    monkeypatch.setattr(lambda_api, "create_filesystem", lambda n, r, *, deadline_at: {})
    assert (
        lambda_api.ensure_filesystem("w", "us-east-1", deadline_at=deadline_at) == "/lambda/nfs/w"
    )


# ---------------------------------------------------------------------------
# get_instance: dict / 404-None / non-404 re-raise / non-dict-None
# ---------------------------------------------------------------------------
def test_get_instance_dict_none_on_404_and_reraises_others(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(
        lambda_api,
        "request_with_retries",
        lambda path, **kw: {"data": {"id": "i-1", "status": "active"}},
    )
    assert lambda_api.get_instance("i-1") == {"id": "i-1", "status": "active"}

    # a genuine HTTP 404 (terminated instance) -> None, not an error
    def not_found(*a, **k):
        raise lambda_api.LambdaApiError("GET /instances/i-1 -> HTTP 404: not found")

    monkeypatch.setattr(lambda_api, "request_with_retries", not_found)
    assert lambda_api.get_instance("i-1") is None

    # a non-404 error is a real fault -> re-raised
    def server_err(*a, **k):
        raise lambda_api.LambdaApiError("GET /instances/i-1 -> HTTP 500: server error")

    monkeypatch.setattr(lambda_api, "request_with_retries", server_err)
    with pytest.raises(lambda_api.LambdaApiError, match="HTTP 500"):
        lambda_api.get_instance("i-1")

    # a 200 whose payload isn't a dict -> None
    monkeypatch.setattr(lambda_api, "request_with_retries", lambda *a, **k: {"data": ["x"]})
    assert lambda_api.get_instance("i-1") is None


# ---------------------------------------------------------------------------
# terminate_instances: per-id isolation + falsy filtering + stringify
# ---------------------------------------------------------------------------
def test_terminate_instances_isolates_per_id_and_filters(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    bodies = []

    def fake(path, method="GET", body=None, retries=4, base_delay=2.0):
        bodies.append(body)
        if body["instance_ids"][0] == "bad":
            raise lambda_api.LambdaApiError("HTTP 400: invalid instance id")
        return {}

    monkeypatch.setattr(lambda_api, "request_with_retries", fake)
    deleted = lambda_api.terminate_instances(["good1", "", None, "bad", 42])
    # falsy ids dropped, ints stringified, and the bad id's failure never blocks the others
    assert deleted == ["good1", "42"]
    # each surviving id was terminated ONE AT A TIME (batch endpoint 400s the whole set on one bad id)
    assert [b["instance_ids"] for b in bodies] == [["good1"], ["bad"], ["42"]]


def test_terminate_instance_confirmed_requires_acceptance_and_disappearance(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: list(ids))
    monkeypatch.setattr(lambda_api, "get_instance", lambda iid, *, strict: None)
    lambda_api.terminate_instance_confirmed("i-1")

    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: [])
    with pytest.raises(lambda_api.LambdaApiError, match="was not confirmed"):
        lambda_api.terminate_instance_confirmed("i-1")

    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: list(ids))
    monkeypatch.setattr(
        lambda_api,
        "get_instance",
        lambda iid, *, strict: {"id": iid, "status": "terminating"},
    )
    with pytest.raises(lambda_api.LambdaApiError, match="remains"):
        lambda_api.terminate_instance_confirmed("i-1")


def test_strict_instance_reads_reject_malformed_success_payloads(monkeypatch):
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(lambda_api, "request_with_retries", lambda *a, **k: {"data": {}})
    with pytest.raises(lambda_api.LambdaApiError, match=r"listing.*malformed"):
        lambda_api.list_instances(strict=True)

    monkeypatch.setattr(lambda_api, "request_with_retries", lambda *a, **k: {"data": []})
    with pytest.raises(lambda_api.LambdaApiError, match=r"lookup.*malformed"):
        lambda_api.get_instance("i-1", strict=True)


# ---------------------------------------------------------------------------
# the REAL request_with_retries wrapper (missing key + Cloudflare UA + unwrap)
# ---------------------------------------------------------------------------
def test_request_with_retries_requires_configured_key():
    # the autouse offline fixture deletes LAMBDA_API_KEY; the real wrapper must surface the
    # control-plane-specific missing-key message before touching the network.
    from flash.providers.lambda_.client import api as lambda_api

    with pytest.raises(lambda_api.LambdaApiError, match="LAMBDA_API_KEY not configured"):
        lambda_api.request_with_retries("/instance-types")


def test_real_request_sends_user_agent_and_unwraps_envelope(monkeypatch):
    from flash.providers._lifecycle.net import http as _http
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setenv("LAMBDA_API_KEY", "lk-test")
    monkeypatch.setattr(lambda_api.time, "time", lambda: 1000.0)
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["ua"] = req.get_header("User-agent")  # urllib title-cases the stored key
        captured["auth"] = req.get_header("Authorization")
        captured["url"] = req.full_url
        return _FakeResponse({"data": {"gpu_1x_a10": {"regions_with_capacity_available": []}}})

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    out = lambda_api.list_instance_types(force=True)
    assert out == {"gpu_1x_a10": {"regions_with_capacity_available": []}}  # envelope unwrapped
    # Cloudflare 403s the stdlib UA -> a real UA must be sent, plus the bearer key.
    assert captured["ua"] == "flash-lambda/1.0 (+https://freesolo.co)"
    assert captured["auth"] == "Bearer lk-test"
    assert captured["url"] == "https://cloud.lambdalabs.com/api/v1/instance-types"
