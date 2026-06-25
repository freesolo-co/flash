"""Regression locks for the teardown/robustness hardening from the adversarial review — the
"won't randomly die / no billing leak" guarantees:

- is_not_found keys off the HTTP status CODE, not a bare "404" substring (a transient 5xx on a
  resource whose id contains "404" must NOT be read as "gone").
- Lambda terminate_instances is PER-ID isolated: one bad/stale id can't abort teardown of the rest.
- Hyperstack delete_vms returns only the ids ACTUALLY deleted (truthful teardown accounting).
"""

from __future__ import annotations

import io
import urllib.error


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x/y", code, "msg", {}, io.BytesIO(b""))


def _chained(code: int, text: str) -> Exception:
    e = RuntimeError(text)
    e.__cause__ = _http_error(code)
    return e


def test_is_not_found_keys_off_status_code_not_bare_404():
    from flash.providers._http import is_not_found

    # genuine 404 (chained HTTPError) -> gone
    assert is_not_found(_chained(404, "GET /core/virtual-machines/7 -> HTTP 404: Not Found")) is True
    # a 5xx on a VM whose id CONTAINS 404 -> NOT gone (the old bare-substring bug)
    assert is_not_found(_chained(500, "GET /core/virtual-machines/1404 failed: HTTP 500")) is False
    assert is_not_found(_chained(403, "GET /x/404 -> HTTP 403: Forbidden")) is False
    # no chained cause: only an unambiguous "HTTP 404" token counts, never a bare "404"
    assert is_not_found(RuntimeError("GET /x -> HTTP 404: Not Found")) is True
    assert is_not_found(RuntimeError("GET /core/virtual-machines/1404 failed after 5 attempts")) is False


def test_lambda_terminate_is_per_id_isolated(monkeypatch):
    """One bad id must NOT abort teardown of the others (the crash-backstop sweep passes many)."""
    from flash.providers.lambdalabs import api as la_api

    calls = []

    def fake_req(path, method="GET", body=None, retries=4, base_delay=2.0):
        iid = body["instance_ids"][0]
        calls.append(iid)
        if iid == "bad":
            raise la_api.LambdaApiError("instance bad does not exist")
        return {}

    monkeypatch.setattr(la_api, "request_with_retries", fake_req)
    deleted = la_api.terminate_instances(["i-1", "bad", "i-3"])
    assert deleted == ["i-1", "i-3"]  # the bad id is skipped, the rest still terminate
    assert calls == ["i-1", "bad", "i-3"]  # each tried independently (per-id POST)


def test_hyperstack_delete_vms_returns_only_actually_deleted(monkeypatch):
    from flash.providers.hyperstack import api as hs_api

    monkeypatch.setattr(hs_api, "delete_vm", lambda vid: vid != "bad")  # "bad" fails to delete
    assert hs_api.delete_vms(["vm-1", "bad", "vm-3"]) == ["vm-1", "vm-3"]


def test_hyperstack_sweep_reports_only_truly_reaped(monkeypatch):
    """sweep_orphans must log/return only the VMs that ACTUALLY deleted — never a still-billing VM."""
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    vms = [
        {"id": "vm-a", "name": "flash-1700-aaaa-s0-a0"},
        {"id": "vm-b", "name": "flash-1700-bbbb-s0-a0"},
    ]
    monkeypatch.setattr(hs_api, "list_vms", lambda: vms)
    monkeypatch.setattr(hs_api, "delete_vm", lambda vid: vid == "vm-a")  # vm-b delete fails
    monkeypatch.setattr(hs_api, "delete_vms", lambda ids: [i for i in ids if hs_api.delete_vm(i)])
    out = jobs.sweep_orphans(active_labels=set())
    assert out == ["vm-a"]  # vm-b still billing -> NOT reported as reaped
