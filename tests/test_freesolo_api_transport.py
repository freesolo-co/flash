"""Transport-level failure handling for the Freesolo backend calls.

These helpers sit between the CLI and a network the CLI cannot control, so almost everything
that can go wrong here arrives as an exception rather than a bad value: a timeout, a refused
connection, a 401 from the wrong issuer, an HTML error page where JSON was promised. The
callers all catch ``ClientError`` to print a clean message and keep their own verdict, so any
exception that escapes as something else -- a bare ``TimeoutError``, a ``JSONDecodeError`` --
reaches the user as a traceback instead.

tests/test_projects.py covers the parsing of successful responses. This file covers the
failure translation, and the request each call actually puts on the wire (method, url, query,
timeout), since those are invisible to a response-shape assertion.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
from typing import Any

import pytest

from flash.client.freesolo_api import (
    _freesolo_request,
    create_project,
    export_trace_records,
    get_project,
    list_trace_projects,
    upload_eval_run,
    verify_freesolo_key,
)
from flash.client.http import ApiError, ClientError, RequestTimeoutError
from tests._helpers.wire_headers import sent_headers

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"


class _Response:
    status = 200

    def __init__(self, payload: Any, *, raw: bytes | None = None):
        self._body = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


def _http_error(code: int, body: bytes = b'{"detail": "nope"}') -> urllib.error.HTTPError:
    import io

    return urllib.error.HTTPError(
        "https://freesolo.test/api/projects", code, "err", {}, io.BytesIO(body)
    )


def _raise(exc):
    def _urlopen(*_args, **_kwargs):
        raise exc

    return _urlopen


@pytest.fixture(autouse=True)
def _pinned_base_url(monkeypatch):
    """Every test names its own backend, so none of them can reach the real default."""
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://freesolo.test")


# --- key verification ------------------------------------------------------------------------


@pytest.mark.parametrize("code", [401, 403])
def test_verify_key_names_the_service_that_rejected_it(monkeypatch, code):
    # the same 401 is what a VALID key gets when a stale --api-url sent it to the wrong issuer,
    # so the url has to appear or the message blames the one thing the user copied correctly.
    monkeypatch.setattr("urllib.request.urlopen", _raise(_http_error(code)))

    with pytest.raises(ClientError) as excinfo:
        verify_freesolo_key("fslo-user")

    assert "https://freesolo.test" in str(excinfo.value)


def test_verify_key_reports_a_non_auth_status_as_an_api_error(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _raise(_http_error(500)))

    with pytest.raises(ApiError) as excinfo:
        verify_freesolo_key("fslo-user")

    assert excinfo.value.status == 500


def test_verify_key_translates_an_unreachable_backend(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", _raise(urllib.error.URLError("connection refused"))
    )

    with pytest.raises(ClientError) as excinfo:
        verify_freesolo_key("fslo-user")

    message = str(excinfo.value)
    assert "cannot reach the freesolo backend" in message
    assert "connection refused" in message
    assert "FREESOLO_BASE_URL" in message


def test_verify_key_sends_a_bearer_get_and_accepts_a_2xx(monkeypatch):
    seen = {}

    def urlopen(req, timeout=None):
        seen.update(method=req.method, url=req.full_url, headers=sent_headers(req))
        return _Response({})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    verify_freesolo_key("fslo-user")

    assert seen["method"] == "GET"
    assert seen["url"] == "https://freesolo.test/api/auth/verify"
    assert seen["headers"]["Authorization"] == "Bearer fslo-user"


# --- shared request helper -------------------------------------------------------------------


def test_request_translates_401_into_a_login_hint(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _raise(_http_error(401)))

    with pytest.raises(ClientError, match="flash login"):
        _freesolo_request("GET", "/api/projects", "fslo-user")


def test_request_preserves_a_non_401_status_and_detail(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", _raise(_http_error(422, b'{"detail": {"code": "bad_shape"}}'))
    )

    with pytest.raises(ApiError) as excinfo:
        _freesolo_request("GET", "/api/projects", "fslo-user")

    assert excinfo.value.status == 422
    assert excinfo.value.code == "bad_shape"


def test_request_translates_a_socket_timeout_into_request_timeout_error(monkeypatch):
    # a socket timeout surfaces as a bare TimeoutError, NOT a URLError, so without its own
    # except clause it escapes this module entirely.
    monkeypatch.setattr("urllib.request.urlopen", _raise(TimeoutError("timed out")))

    with pytest.raises(RequestTimeoutError) as excinfo:
        _freesolo_request("GET", "/api/projects", "fslo-user", timeout=12.5)

    message = str(excinfo.value)
    assert "https://freesolo.test" in message
    assert "/api/projects" in message
    assert "12.5" in message


def test_timeout_message_names_the_backend_without_reconstructing_a_route(monkeypatch):
    """The base is reduced to scheme and host, and the path is named separately.

    Two reasons the message is built this way, and a plain f"{base}{path}" would break both: a
    base URL may carry credentials in its authority, and a base carrying a reverse-proxy prefix
    would print an endpoint that was never requested.
    """
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://user:secret@freesolo.test/proxy")
    monkeypatch.setattr("urllib.request.urlopen", _raise(TimeoutError("timed out")))

    with pytest.raises(RequestTimeoutError) as excinfo:
        _freesolo_request("GET", "/api/projects", "fslo-user")

    message = str(excinfo.value)
    assert "secret" not in message
    # the proxy prefix belongs to the base, so the message must not read as /proxy/api/projects
    # nor as a bare https://freesolo.test/api/projects that was never the requested URL.
    assert "https://freesolo.test/api/projects" not in message
    assert "https://freesolo.test" in message
    assert "/api/projects" in message


def test_unreachable_backend_message_also_hides_credentials_in_the_base(monkeypatch):
    monkeypatch.setenv("FREESOLO_BASE_URL", "https://user:secret@freesolo.test")
    monkeypatch.setattr("urllib.request.urlopen", _raise(urllib.error.URLError("no route")))

    with pytest.raises(ClientError) as excinfo:
        _freesolo_request("GET", "/api/projects", "fslo-user")

    message = str(excinfo.value)
    assert "secret" not in message
    assert "no route" in message


def test_request_timeout_error_is_catchable_as_client_error():
    # callers catch ClientError to report a failure without changing their verdict; a timeout
    # that is not one of those would bypass every such handler.
    assert issubclass(RequestTimeoutError, ClientError)


def test_request_translates_an_unreachable_backend(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", _raise(urllib.error.URLError("no route")))

    with pytest.raises(ClientError, match="cannot reach the freesolo backend"):
        _freesolo_request("GET", "/api/projects", "fslo-user")


def test_request_rejects_a_2xx_body_that_is_not_json(monkeypatch):
    # an HTML error page from a proxy is a 200 with a body this client cannot use; a raw
    # JSONDecodeError has no CLI handler and would print as a traceback.
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _Response(None, raw=b"<html>502</html>")
    )

    with pytest.raises(ClientError, match="invalid JSON"):
        _freesolo_request("GET", "/api/projects", "fslo-user")


def test_request_treats_an_empty_body_as_an_empty_object(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(None, raw=b""))

    assert _freesolo_request("DELETE", "/api/projects/x", "fslo-user") == {}


def test_request_sends_the_body_as_json_and_omits_it_when_absent(monkeypatch):
    seen = []

    def urlopen(req, timeout=None):
        seen.append((req.method, req.data, sent_headers(req), timeout))
        return _Response({})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    _freesolo_request("GET", "/api/projects", "fslo-user")
    _freesolo_request("POST", "/api/projects", "fslo-user", body={"name": "n"}, timeout=7.0)

    assert seen[0][1] is None
    assert json.loads(seen[1][1]) == {"name": "n"}
    assert seen[1][2]["Content-Type"] == "application/json"
    assert seen[1][3] == 7.0


# --- project calls ---------------------------------------------------------------------------


def test_get_project_rejects_a_malformed_requested_id_without_calling_out(monkeypatch):
    def urlopen(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("a malformed id must be rejected before any request")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(ClientError, match="project id"):
        get_project("not-a-uuid", "fslo-user")


def test_get_project_url_encodes_the_requested_id(monkeypatch):
    seen = {}

    def urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return _Response({"project": {"id": _PROJECT_ID}})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    assert get_project(_PROJECT_ID, "fslo-user")["id"] == _PROJECT_ID
    assert seen["url"] == f"https://freesolo.test/api/projects/{urllib.parse.quote(_PROJECT_ID)}"


def test_get_project_unwraps_either_response_envelope(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _Response({"project": {"id": _PROJECT_ID, "name": "Wrapped"}}),
    )
    assert get_project(_PROJECT_ID, "fslo-user")["name"] == "Wrapped"

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: _Response({"id": _PROJECT_ID, "name": "Bare"}),
    )
    assert get_project(_PROJECT_ID, "fslo-user")["name"] == "Bare"


@pytest.mark.parametrize("payload", [{"project": {"id": "  "}}, {"project": {}}, {"id": None}])
def test_get_project_rejects_a_response_without_a_usable_id(monkeypatch, payload):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(payload))

    with pytest.raises(ClientError, match="no valid project"):
        get_project(_PROJECT_ID, "fslo-user")


def test_get_project_rejects_a_list_payload(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response([{"id": _PROJECT_ID}]))

    with pytest.raises(ClientError, match="no valid project"):
        get_project(_PROJECT_ID, "fslo-user")


@pytest.mark.parametrize("name", ["", "   ", None])
def test_create_project_requires_a_nonblank_name_before_calling_out(monkeypatch, name):
    def urlopen(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("a blank name must be rejected before any request")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(ClientError, match="nonblank"):
        create_project(name, None, "fslo-user")


def test_create_project_normalizes_a_blank_description_to_null(monkeypatch):
    seen = {}

    def urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return _Response({"id": _PROJECT_ID})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    create_project(" Name ", "   ", "fslo-user")

    assert seen["body"] == {"name": "Name", "description": None}


def test_create_project_rejects_a_non_dict_response(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response([{"id": _PROJECT_ID}]))

    with pytest.raises(ClientError, match="invalid project response"):
        create_project("Name", None, "fslo-user")


# --- trace projects and export ---------------------------------------------------------------


def test_list_trace_projects_returns_the_projects_array(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: _Response({"projects": [{"id": _PROJECT_ID}]})
    )

    assert list_trace_projects("fslo-user") == [{"id": _PROJECT_ID}]


@pytest.mark.parametrize("payload", [{"projects": None}, {"projects": {}}, {}])
def test_list_trace_projects_degrades_to_empty_on_an_unusable_payload(monkeypatch, payload):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(payload))

    assert list_trace_projects("fslo-user") == []


def test_export_sends_only_the_project_id_when_nothing_else_is_requested(monkeypatch):
    seen = {}

    def urlopen(req, timeout=None):
        seen["query"] = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
        seen["timeout"] = timeout
        return _Response({"records": []})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    export_trace_records(_PROJECT_ID, "fslo-user")

    assert seen["query"] == {"project_id": [_PROJECT_ID]}
    # a whole project's traces is a large read, so it must not inherit the 60s default.
    assert seen["timeout"] == 300.0


def test_export_passes_limit_and_format_through_the_query(monkeypatch):
    seen = {}

    def urlopen(req, timeout=None):
        seen["query"] = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
        return _Response({"records": []})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    export_trace_records(_PROJECT_ID, "fslo-user", limit=25, export_format="openai")

    assert seen["query"] == {
        "project_id": [_PROJECT_ID],
        "limit": ["25"],
        "format": ["openai"],
    }


def test_export_coerces_a_float_limit_to_an_integer_query_value(monkeypatch):
    seen = {}

    def urlopen(req, timeout=None):
        seen["query"] = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
        return _Response({"records": []})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    export_trace_records(_PROJECT_ID, "fslo-user", limit=10.0)

    assert seen["query"]["limit"] == ["10"]


# --- eval run upload -------------------------------------------------------------------------


def _upload(**overrides):
    body = {
        "project_id": _PROJECT_ID,
        "suite_name": "suite",
        "environment_reference": "org/env",
        "model": "m",
        "status": "completed",
        "error": None,
        "started_at": "2026-01-01T00:00:00Z",
        "cases": [{"name": "c", "score": 1.0}],
        "api_key": "fslo-user",
    }
    body.update(overrides)
    return upload_eval_run(**body)


@pytest.mark.parametrize("project_id", ["", "   ", "not-a-uuid", None])
def test_upload_eval_run_requires_an_explicit_valid_project(monkeypatch, project_id):
    # an api key identifies an org, not a project, so a missing id must fail rather than let
    # results land under a project the caller never named.
    def urlopen(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("an invalid project id must be rejected before any request")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    with pytest.raises(ClientError, match="project id"):
        _upload(project_id=project_id)


def test_upload_eval_run_posts_the_suite_with_a_long_timeout(monkeypatch):
    seen = {}

    def urlopen(req, timeout=None):
        seen.update(method=req.method, url=req.full_url, body=json.loads(req.data), timeout=timeout)
        return _Response({"id": "run-1"})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    assert _upload() == {"id": "run-1"}

    assert seen["method"] == "POST"
    assert seen["url"] == "https://freesolo.test/api/evals/runs"
    assert seen["body"]["project_id"] == _PROJECT_ID
    assert seen["body"]["cases"] == [{"name": "c", "score": 1.0}]
    # a large suite is a bigger write than a normal control-plane call.
    assert seen["timeout"] == 300.0


def test_upload_eval_run_canonicalizes_a_padded_project_id(monkeypatch):
    seen = {}

    def urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return _Response({"id": "run-1"})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    _upload(project_id=f"  {_PROJECT_ID.upper()}  ")

    assert seen["body"]["project_id"] == _PROJECT_ID


def test_upload_eval_run_rejects_a_non_dict_response(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response(["run-1"]))

    with pytest.raises(ClientError, match="invalid eval run response"):
        _upload()
