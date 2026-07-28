from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest
from fastapi import HTTPException

from flash.client import ClientError, create_project, get_project, list_projects
from flash.server.projects import require_project_access


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
        seen.append((req.method, req.full_url, dict(req.headers), body))
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
            headers=dict(req.headers),
            body=req.data,
            timeout=timeout,
        )
        return _Response(
            {"project": {"id": "11111111-1111-4111-8111-111111111111", "orgId": "org-one"}}
        )

    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")
    monkeypatch.setattr("flash.server.projects.urllib.request.urlopen", urlopen)

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
            headers=dict(req.headers),
            body=json.loads(req.data),
            timeout=timeout,
        )
        return _Response({"ok": True, "orgId": "org-one", "projectId": project_id})

    monkeypatch.setattr("flash.server.projects.urllib.request.urlopen", urlopen)
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
            "Content-type": "application/json",
        },
        "body": {"orgId": "org-one", "projectId": project_id},
        "timeout": 10.0,
    }


def test_internal_project_validation_requires_service_token(monkeypatch) -> None:
    monkeypatch.delenv("FREESOLO_INTERNAL_KEY", raising=False)
    monkeypatch.setattr(
        "flash.server.projects.urllib.request.urlopen",
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
        "flash.server.projects.urllib.request.urlopen",
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
        "flash.server.projects.urllib.request.urlopen",
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
        "flash.server.projects.urllib.request.urlopen",
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
        "flash.server.projects.urllib.request.urlopen",
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
        "flash.server.projects.urllib.request.urlopen",
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
