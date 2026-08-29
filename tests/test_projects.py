from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest
from fastapi import HTTPException

from flash.client import ClientError, create_project, get_project, list_projects
from flash.server.domain.registry.projects import (
    require_project_access,
    require_project_access_slug,
)
from tests._helpers.wire_headers import sent_headers


class _Response:
    status = 200

    def __init__(self, payload: Any):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


def test_project_client_uses_bearer_api_without_caller_org(monkeypatch) -> None:
    seen = []

    def urlopen(req, timeout=None):
        body = json.loads(req.data) if req.data else None
        seen.append((req.method, req.full_url, sent_headers(req), body))
        if req.method == "POST":
            return _Response({"id": " 33333333-3333-4333-8333-333333333333 "})
        if req.full_url.endswith("/api/projects/11111111-1111-4111-8111-111111111111"):
            return _Response({"id": "11111111-1111-4111-8111-111111111111", "name": "One"})
        return _Response([{"id": "11111111-1111-4111-8111-111111111111", "name": "One"}])

    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")
    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    assert list_projects("fslo-user") == [
        {"id": "11111111-1111-4111-8111-111111111111", "name": "One"}
    ]
    assert (
        get_project(" 11111111-1111-4111-8111-111111111111 ", "fslo-user")["id"]
        == "11111111-1111-4111-8111-111111111111"
    )
    assert (
        create_project(" New project ", " description ", "fslo-user")["id"]
        == "33333333-3333-4333-8333-333333333333"
    )

    assert [call[0] for call in seen] == ["GET", "GET", "POST"]
    assert all(call[2]["Authorization"] == "Bearer fslo-user" for call in seen)
    assert seen[2][3] == {"name": "New project", "description": "description"}
    assert "orgId" not in seen[2][3]
    assert "org_id" not in seen[2][3]


def test_project_client_rejects_malformed_or_mismatched_responses(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response({"projects": []}))
    with pytest.raises(ClientError, match="invalid project list"):
        list_projects("fslo-user")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _Response({"id": "22222222-2222-4222-8222-222222222222"}),
    )
    with pytest.raises(ClientError, match="mismatched project id"):
        get_project("11111111-1111-4111-8111-111111111111", "fslo-user")


def test_project_create_rejects_invalid_returned_id(monkeypatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response({"id": "   "}))
    with pytest.raises(ClientError, match="invalid project id"):
        create_project("name", None, "fslo-user")


def test_server_project_validation_uses_authenticated_bearer_and_org(monkeypatch) -> None:
    seen = {}

    def urlopen(req, timeout=None):
        seen.update(
            method=req.get_method(),
            url=req.full_url,
            headers=sent_headers(req),
            body=req.data,
            timeout=timeout,
        )
        return _Response(
            {"project": {"id": "11111111-1111-4111-8111-111111111111", "orgId": "org-one"}}
        )

    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")
    monkeypatch.setattr("flash.server.domain.registry.projects.urllib.request.urlopen", urlopen)

    assert (
        require_project_access(
            project_id=" 11111111-1111-4111-8111-111111111111 ",
            key={"auth_kind": "freesolo_api_key", "org_id": "org-one"},
            authorization="Bearer fslo-user",
        )
        == "11111111-1111-4111-8111-111111111111"
    )
    assert seen["method"] == "GET"
    assert seen["url"] == "https://freesolo.test/api/projects/11111111-1111-4111-8111-111111111111"
    assert seen["headers"]["Authorization"] == "Bearer fslo-user"
    assert seen["body"] is None
    assert "X-freesolo-org-id" not in seen["headers"]


def test_internal_project_validation_uses_internal_service_endpoint(monkeypatch) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    with pytest.raises(HTTPException, match="X-Freesolo-Org-Id is required"):
        require_project_access(
            project_id=project_id,
            key={"auth_kind": "internal"},
            authorization="Bearer incoming-internal-key",
        )

    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "service-internal-key")
    seen = {}

    def urlopen(req, timeout=None):
        seen.update(
            method=req.get_method(),
            url=req.full_url,
            headers=sent_headers(req),
            body=json.loads(req.data),
            timeout=timeout,
        )
        return _Response({"ok": True, "orgId": "org-one", "projectId": project_id})

    monkeypatch.setattr("flash.server.domain.registry.projects.urllib.request.urlopen", urlopen)
    assert (
        require_project_access(
            project_id=f"  {project_id}  ",
            key={"auth_kind": "internal"},
            authorization="Bearer incoming-internal-key",
            org_id="org-one",
        )
        == project_id
    )
    assert seen == {
        "method": "POST",
        "url": "https://freesolo.test/api/flash/projects/validate/internal",
        "headers": {
            "Authorization": "Bearer service-internal-key",
            "Content-Type": "application/json",
        },
        "body": {"orgId": "org-one", "projectId": project_id},
        "timeout": 10.0,
    }


def test_internal_project_validation_requires_service_token(monkeypatch) -> None:
    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    monkeypatch.setattr(
        "flash.server.domain.registry.projects.urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("missing token must fail before transport"),
    )

    with pytest.raises(HTTPException) as excinfo:
        require_project_access(
            project_id="11111111-1111-4111-8111-111111111111",
            key={"auth_kind": "internal"},
            authorization="Bearer incoming-internal-key",
            org_id="org-one",
        )
    assert excinfo.value.status_code == 503
    assert "FREESOLO_INTERNAL_KEY" in excinfo.value.detail


@pytest.mark.parametrize(
    ("http_status", "expected_status"),
    [(401, 502), (404, 403), (500, 502)],
)
def test_internal_project_validation_fails_closed_on_http_errors(
    monkeypatch, http_status, expected_status
) -> None:
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "service-internal-key")
    error = urllib.error.HTTPError(
        "https://freesolo.test/api/flash/projects/validate/internal",
        http_status,
        "failed",
        {},
        io.BytesIO(json.dumps({"detail": "validation failed"}).encode()),
    )
    monkeypatch.setattr(
        "flash.server.domain.registry.projects.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(HTTPException) as excinfo:
        require_project_access(
            project_id="11111111-1111-4111-8111-111111111111",
            key={"auth_kind": "internal"},
            authorization="Bearer incoming-internal-key",
            org_id="org-one",
        )
    assert excinfo.value.status_code == expected_status


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"ok": False, "orgId": "org-one", "projectId": "11111111-1111-4111-8111-111111111111"},
        {"ok": True, "orgId": "org-two", "projectId": "11111111-1111-4111-8111-111111111111"},
        {"ok": True, "orgId": "org-one", "projectId": "22222222-2222-4222-8222-222222222222"},
        {"ok": True, "orgId": "org-one", "projectId": " 11111111-1111-4111-8111-111111111111 "},
    ],
)
def test_internal_project_validation_rejects_malformed_or_mismatched_success(
    monkeypatch, payload
) -> None:
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "service-internal-key")
    monkeypatch.setattr(
        "flash.server.domain.registry.projects.urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(payload),
    )

    with pytest.raises(HTTPException) as excinfo:
        require_project_access(
            project_id="11111111-1111-4111-8111-111111111111",
            key={"auth_kind": "internal"},
            authorization="Bearer incoming-internal-key",
            org_id="org-one",
        )
    assert excinfo.value.status_code == 502


def test_internal_project_validation_rejects_invalid_json(monkeypatch) -> None:
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "service-internal-key")

    class _InvalidResponse:
        status = 200

        def read(self):
            return b"not-json"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        "flash.server.domain.registry.projects.urllib.request.urlopen",
        lambda *_args, **_kwargs: _InvalidResponse(),
    )

    with pytest.raises(HTTPException) as excinfo:
        require_project_access(
            project_id="11111111-1111-4111-8111-111111111111",
            key={"auth_kind": "internal"},
            authorization="Bearer incoming-internal-key",
            org_id="org-one",
        )
    assert excinfo.value.status_code == 502


def test_internal_project_validation_fails_closed_on_transport_error(monkeypatch) -> None:
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "service-internal-key")
    monkeypatch.setattr(
        "flash.server.domain.registry.projects.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    with pytest.raises(HTTPException) as excinfo:
        require_project_access(
            project_id="11111111-1111-4111-8111-111111111111",
            key={"auth_kind": "internal"},
            authorization="Bearer incoming-internal-key",
            org_id="org-one",
        )
    assert excinfo.value.status_code == 503


def test_server_project_validation_rejects_cross_org_project(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "https://freesolo.test/api/projects/22222222-2222-4222-8222-222222222222",
        404,
        "not found",
        {},
        io.BytesIO(json.dumps({"detail": "not found"}).encode()),
    )
    monkeypatch.setattr(
        "flash.server.domain.registry.projects.urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(error),
    )
    with pytest.raises(HTTPException) as excinfo:
        require_project_access(
            project_id="22222222-2222-4222-8222-222222222222",
            key={"auth_kind": "freesolo_api_key", "org_id": "org-one"},
            authorization="Bearer fslo-user",
        )
    assert excinfo.value.status_code == 403
    assert "authenticated organization" in excinfo.value.detail


def _no_network(monkeypatch) -> None:
    """Any outbound call from a standalone path is the bug under test, so make one fail loudly."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("standalone must not call the Freesolo backend")

    monkeypatch.setattr("urllib.request.urlopen", _boom)


def test_standalone_accepts_a_well_formed_project_without_a_backend(monkeypatch) -> None:
    """A self-hosted plane has no org directory to validate against, so the id is taken as given.

    This is the gate that makes self-hosting work at all: without it every submit resolves
    api.freesolo.co, fails, and is rejected 503.
    """
    monkeypatch.setenv("FLASH_STANDALONE", "1")
    _no_network(monkeypatch)

    project_id = "11111111-1111-4111-8111-111111111111"
    assert (
        require_project_access(
            project_id=f" {project_id} ",
            key={"auth_kind": "internal"},
            authorization=None,
        )
        == project_id
    )


def test_standalone_still_requires_a_well_formed_project_id(monkeypatch) -> None:
    """Standalone relaxes OWNERSHIP, not the requirement itself: runs stay grouped by project.

    A malformed id must fail the same 400 it would against the backend, or self-hosting would
    become the one mode where the explicit-project contract silently does not apply.
    """
    monkeypatch.setenv("FLASH_STANDALONE", "1")
    _no_network(monkeypatch)

    for bad in ("", "   ", "not-a-uuid"):
        with pytest.raises(HTTPException) as excinfo:
            require_project_access(
                project_id=bad,
                key={"auth_kind": "internal"},
                authorization=None,
            )
        assert excinfo.value.status_code == 400


def test_standalone_refuses_to_resolve_a_publish_slug(monkeypatch) -> None:
    """Standalone has no project directory, so a slug caller must fail HERE, naming that.

    The id-only path above still succeeds: standalone runs are the reason it must. Only the
    caller that needs a slug is refused, and it is refused where the reason is known -- publish
    used to run on the empty string and fail later blaming a stale login key, which no re-login
    can fix.
    """
    monkeypatch.setenv("FLASH_STANDALONE", "1")
    _no_network(monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        require_project_access_slug(
            project_id="11111111-1111-4111-8111-111111111111",
            key={"auth_kind": "internal"},
            authorization=None,
        )
    assert excinfo.value.status_code == 501
    assert "standalone plane does not have" in excinfo.value.detail
    assert "flash login" not in excinfo.value.detail


def _install_project_validation_responses(
    monkeypatch,
    *,
    public_payloads: list[dict[str, Any]],
    internal_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "service-internal-key")
    public = iter(public_payloads)
    internal = iter(internal_payloads)
    requests: list[dict[str, Any]] = []

    def urlopen(req, timeout=None):
        method = req.get_method()
        requests.append(
            {
                "method": method,
                "url": req.full_url,
                "authorization": req.get_header("Authorization"),
                "body": json.loads(req.data) if req.data else None,
            }
        )
        if method == "GET":
            return _Response(next(public))
        return _Response(next(internal))

    monkeypatch.setattr("flash.server.domain.registry.projects.urllib.request.urlopen", urlopen)
    return requests


def test_public_name_only_response_uses_internal_canonical_slug(monkeypatch) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    requests = _install_project_validation_responses(
        monkeypatch,
        public_payloads=[{"id": project_id, "name": "Foo Bar"}],
        internal_payloads=[
            {
                "ok": True,
                "orgId": "org-one",
                "projectId": project_id,
                "projectSlug": "foo-bar-2",
            }
        ],
    )

    assert require_project_access_slug(
        project_id=project_id,
        key={"auth_kind": "freesolo_api_key", "org_id": "org-one"},
        authorization="Bearer fslo-user",
    ) == (project_id, "foo-bar-2")
    assert [request["method"] for request in requests] == ["GET", "POST"]
    assert requests[1]["body"] == {"orgId": "org-one", "projectId": project_id}


def test_colliding_project_names_keep_distinct_canonical_slugs(monkeypatch) -> None:
    first_id = "11111111-1111-4111-8111-111111111111"
    second_id = "22222222-2222-4222-8222-222222222222"
    requests = _install_project_validation_responses(
        monkeypatch,
        public_payloads=[
            {"id": first_id, "name": "Foo Bar"},
            {"id": second_id, "name": "foo-bar"},
        ],
        internal_payloads=[
            {
                "ok": True,
                "orgId": "org-one",
                "projectId": first_id,
                "projectSlug": "foo-bar",
            },
            {
                "ok": True,
                "orgId": "org-one",
                "projectId": second_id,
                "projectSlug": "foo-bar-2",
            },
        ],
    )
    key = {"auth_kind": "freesolo_api_key", "org_id": "org-one"}

    first = require_project_access_slug(
        project_id=first_id, key=key, authorization="Bearer fslo-user"
    )
    second = require_project_access_slug(
        project_id=second_id, key=key, authorization="Bearer fslo-user"
    )

    assert first == (first_id, "foo-bar")
    assert second == (second_id, "foo-bar-2")
    assert first[1] != second[1]
    assert [request["method"] for request in requests] == ["GET", "POST", "GET", "POST"]


def test_explicit_public_slug_passes_through_without_internal_lookup(monkeypatch) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    requests = _install_project_validation_responses(
        monkeypatch,
        public_payloads=[
            {"project": {"id": project_id, "slug": "checkout-bot", "name": "Something Else"}}
        ],
        internal_payloads=[],
    )

    assert require_project_access_slug(
        project_id=project_id,
        key={"auth_kind": "freesolo_api_key", "org_id": "org-one"},
        authorization="Bearer fslo-user",
    ) == (project_id, "checkout-bot")
    assert [request["method"] for request in requests] == ["GET"]


def test_internal_key_canonical_slug_passes_through(monkeypatch) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    requests = _install_project_validation_responses(
        monkeypatch,
        public_payloads=[],
        internal_payloads=[
            {
                "ok": True,
                "orgId": "org-one",
                "projectId": project_id,
                "projectSlug": "checkout-bot",
            }
        ],
    )

    assert require_project_access_slug(
        project_id=project_id,
        key={"auth_kind": "internal"},
        authorization="Bearer incoming-internal-key",
        org_id="org-one",
    ) == (project_id, "checkout-bot")
    assert [request["method"] for request in requests] == ["POST"]


def test_public_authorization_failure_prevents_internal_lookup(monkeypatch) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    calls: list[str] = []
    error = urllib.error.HTTPError(
        f"https://freesolo.test/api/projects/{project_id}",
        404,
        "not found",
        {},
        io.BytesIO(b"{}"),
    )

    def urlopen(req, timeout=None):
        calls.append(req.get_method())
        if req.get_method() != "GET":
            pytest.fail("internal lookup must not run before public authorization succeeds")
        raise error

    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "service-internal-key")
    monkeypatch.setattr("flash.server.domain.registry.projects.urllib.request.urlopen", urlopen)

    with pytest.raises(HTTPException) as excinfo:
        require_project_access_slug(
            project_id=project_id,
            key={"auth_kind": "freesolo_api_key", "org_id": "org-one"},
            authorization="Bearer fslo-user",
        )
    assert excinfo.value.status_code == 403
    assert calls == ["GET"]


def test_missing_org_identity_fails_before_internal_lookup(monkeypatch) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    requests = _install_project_validation_responses(
        monkeypatch,
        public_payloads=[{"id": project_id, "name": "Foo Bar"}],
        internal_payloads=[],
    )

    with pytest.raises(HTTPException) as excinfo:
        require_project_access_slug(
            project_id=project_id,
            key={"auth_kind": "freesolo_api_key"},
            authorization="Bearer fslo-user",
        )
    assert excinfo.value.status_code == 502
    assert "organization id" in excinfo.value.detail
    assert [request["method"] for request in requests] == ["GET"]


def test_missing_internal_key_fails_after_public_authorization(monkeypatch) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    requests = _install_project_validation_responses(
        monkeypatch,
        public_payloads=[{"id": project_id, "name": "Foo Bar"}],
        internal_payloads=[],
    )
    monkeypatch.delenv("FREESOLO_INTERNAL_KEY")

    with pytest.raises(HTTPException) as excinfo:
        require_project_access_slug(
            project_id=project_id,
            key={"auth_kind": "freesolo_api_key", "org_id": "org-one"},
            authorization="Bearer fslo-user",
        )
    assert excinfo.value.status_code == 503
    assert "FREESOLO_INTERNAL_KEY" in excinfo.value.detail
    assert [request["method"] for request in requests] == ["GET"]


@pytest.mark.parametrize(
    ("ok", "expected_detail"),
    [(True, "canonical project slug"), (False, "malformed response")],
)
def test_missing_or_malformed_internal_slug_response_fails_closed(
    monkeypatch, ok, expected_detail
) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    requests = _install_project_validation_responses(
        monkeypatch,
        public_payloads=[{"id": project_id, "name": "Foo Bar"}],
        internal_payloads=[{"ok": ok, "orgId": "org-one", "projectId": project_id}],
    )

    with pytest.raises(HTTPException) as excinfo:
        require_project_access_slug(
            project_id=project_id,
            key={"auth_kind": "freesolo_api_key", "org_id": "org-one"},
            authorization="Bearer fslo-user",
        )
    assert excinfo.value.status_code == 502
    assert expected_detail in excinfo.value.detail
    assert [request["method"] for request in requests] == ["GET", "POST"]


def test_internal_transport_failure_after_public_authorization_fails_closed(monkeypatch) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    calls: list[str] = []

    def urlopen(req, timeout=None):
        calls.append(req.get_method())
        if req.get_method() == "GET":
            return _Response({"id": project_id, "name": "Foo Bar"})
        raise urllib.error.URLError("offline")

    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", "service-internal-key")
    monkeypatch.setattr("flash.server.domain.registry.projects.urllib.request.urlopen", urlopen)

    with pytest.raises(HTTPException) as excinfo:
        require_project_access_slug(
            project_id=project_id,
            key={"auth_kind": "freesolo_api_key", "org_id": "org-one"},
            authorization="Bearer fslo-user",
        )
    assert excinfo.value.status_code == 503
    assert calls == ["GET", "POST"]


def test_id_only_project_access_does_not_require_canonical_slug(monkeypatch) -> None:
    project_id = "11111111-1111-4111-8111-111111111111"
    requests = _install_project_validation_responses(
        monkeypatch,
        public_payloads=[{"id": project_id, "name": "Foo Bar"}],
        internal_payloads=[],
    )
    monkeypatch.delenv("FREESOLO_INTERNAL_KEY")

    assert (
        require_project_access(
            project_id=project_id,
            key={"auth_kind": "freesolo_api_key", "org_id": "org-one"},
            authorization="Bearer fslo-user",
        )
        == project_id
    )
    assert [request["method"] for request in requests] == ["GET"]
