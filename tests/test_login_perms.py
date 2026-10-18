"""`flash login` verifies the freesolo key, then stores it with private (0600) permissions."""

from __future__ import annotations

import importlib
import json
import os
import stat
import tempfile
import types

import pytest

import flash.cli.commands.ops.account as cli_account
from flash.serve.contract.urls import is_freesolo_hosted_url


def test_login_writes_private_config(monkeypatch):
    saved_home = os.environ.get("HOME")
    try:
        _check_login(monkeypatch)
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home
        import flash.client.config as client_config

        importlib.reload(client_config)


def _check_login(monkeypatch):
    with tempfile.TemporaryDirectory() as home:
        os.environ["HOME"] = home
        import flash.client.config as client_config

        importlib.reload(client_config)

        # Login verifies the freesolo key against the freesolo backend, then stores it. Stub
        # the network verify so the test stays offline; an invalid key would raise instead.
        verified: dict = {}
        monkeypatch.setattr(
            cli_account,
            "verify_freesolo_key",
            lambda api_key, base_url=None: verified.update(api_key=api_key, base_url=base_url),
        )
        # After storing the key, login fetches the identity card straight from the key it just
        # verified (not ambient credentials); stub the client so the test stays offline and
        # capture the key it was built with.
        identity = {"kind": "freesolo_api_key", "key_prefix": "fs-secr", "email": "me@example.com"}
        built_with: dict = {}

        class _FakeApi:
            def __init__(self, api_url, api_key=None, timeout=None):
                built_with.update(api_url=api_url, api_key=api_key)

            def me(self):
                return identity

        monkeypatch.setattr(cli_account, "ApiClient", _FakeApi)
        args = types.SimpleNamespace(api_key="fs-secret-123", api_url=None, freesolo_url=None)
        rc = cli_account.cmd_login(args)
        assert rc == 0
        assert verified["api_key"] == "fs-secret-123"  # the key was actually verified
        assert built_with["api_key"] == "fs-secret-123"  # ...and that exact key built the card

        cfg = client_config.CONFIG_PATH
        assert cfg.exists()
        mode = stat.S_IMODE(os.stat(cfg).st_mode)
        assert mode == 0o600, f"expected 0600, got {oct(mode)}"
        # Directory should be private too.
        dir_mode = stat.S_IMODE(os.stat(client_config.CONFIG_DIR).st_mode)
        assert dir_mode == 0o700, f"expected 0700, got {oct(dir_mode)}"
        assert json.loads(cfg.read_text())["api_key"] == "fs-secret-123"
        # And the stored key resolves through the normal credential lookup.
        os.environ.pop("FREESOLO_API_KEY", None)
        _, key = client_config.load_credentials()
        assert key == "fs-secret-123"


def test_login_warns_when_env_key_will_override_saved_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FREESOLO_API_KEY", "fs-secret-env")

    import flash.client.config as client_config

    importlib.reload(client_config)

    monkeypatch.setattr(cli_account, "verify_freesolo_key", lambda api_key, base_url=None: None)
    args = types.SimpleNamespace(api_key="fs-secret-arg", api_url=None, freesolo_url=None)

    assert cli_account.cmd_login(args) == 0
    captured = capsys.readouterr()
    assert "logged in to flash" in captured.out
    assert "FREESOLO_API_KEY is set and will override this saved login" in captured.err
    assert client_config.load_credentials()[1] == "fs-secret-env"


SELF_HOSTED_PLANE_URLS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://my-plane:8080",
    "https://flash.example.internal",
    "https://plane.acme.com/v1",
    # the suffix traps: these are NOT freesolo.co, so they must be treated as self-hosted.
    "https://notfreesolo.co",
    "https://freesolo.co.attacker.test",
]

HOSTED_PLANE_URLS = [
    "https://api.freesolo.co",
    "https://flash-dev.freesolo.co",
    "https://FREESOLO.CO",
    "https://api.freesolo.co./v1",
]


def _login_capturing_the_wire(monkeypatch, tmp_path, *, api_url, freesolo_url=None):
    """Run cmd_login with the network stubbed, returning every URL a credential was sent to.

    Asserts at the TRANSPORT layer rather than on `verify_freesolo_key` being called, because the
    leak is defined by where the key travels. A test that only checked "did not raise" passed
    while the bug was live.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)
    monkeypatch.delenv("FREESOLO_BASE_URL", raising=False)
    monkeypatch.delenv("FLASH_API_URL", raising=False)

    import flash.client.config as client_config

    importlib.reload(client_config)
    import flash.client.http as client_http

    sent: list[tuple[str, str]] = []

    class _NoBody:
        """Minimal urlopen result: a verify call only reads and discards the body."""

        status = 200

        def read(self, *a, **k):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _capture(req, *a, **k):
        # record, then succeed. Raising here would make the hosted path fail for a reason
        # unrelated to what is under test, hiding whether the request was made at all.
        sent.append((req.full_url, req.headers.get("Authorization") or ""))
        return _NoBody()

    monkeypatch.setattr(client_http.urllib.request, "urlopen", _capture)

    class _FakeApi:
        def __init__(self, api_url, api_key=None, timeout=None):
            sent.append((api_url, f"Bearer {api_key}"))

        def me(self):
            # an ACCEPTING plane: the self-hosted path verifies the key by calling this, so
            # returning an identity is what "the plane accepted it" looks like on the wire.
            return {"kind": "internal", "key_prefix": "operator"}

    monkeypatch.setattr(cli_account, "ApiClient", _FakeApi)
    args = types.SimpleNamespace(
        api_key="operator-root-key", api_url=api_url, freesolo_url=freesolo_url
    )
    rc = cli_account.cmd_login(args)
    return rc, sent


@pytest.mark.parametrize("api_url", SELF_HOSTED_PLANE_URLS)
def test_login_against_a_self_hosted_plane_never_contacts_freesolo(monkeypatch, tmp_path, api_url):
    """The plane-root credential must not leave the operator's own infrastructure.

    `flash login --api-url <own-plane> --api-key $FREESOLO_INTERNAL_KEY` is the DOCUMENTED
    self-hosting quickstart. It used to verify against https://api.freesolo.co regardless of
    --api-url, so the key that controls the plane was sent to a third party, which then rejected
    it: the credential leaked AND the documented setup could not complete.
    """
    rc, sent = _login_capturing_the_wire(monkeypatch, tmp_path, api_url=api_url)

    assert rc == 0, "login against a self-hosted plane must succeed"
    # matched on the parsed HOST, not a substring: `https://notfreesolo.co` CONTAINS "freesolo.co"
    # while not being ours, so a substring check would fail its own must-allow case.
    leaked = [(url, auth) for url, auth in sent if is_freesolo_hosted_url(url)]
    assert not leaked, f"credential sent to Freesolo-operated infrastructure: {leaked}"
    # and it did reach the operator's own plane, so this is not vacuously clean.
    assert any(api_url.split("://", 1)[1].split("/")[0] in url for url, _ in sent), sent


@pytest.mark.parametrize("api_url", HOSTED_PLANE_URLS)
def test_login_against_the_hosted_plane_still_verifies(monkeypatch, tmp_path, api_url):
    """The managed path is unchanged: a hosted control plane still verifies the key upstream."""
    _rc, sent = _login_capturing_the_wire(monkeypatch, tmp_path, api_url=api_url)

    verified = [url for url, _ in sent if "/api/auth/verify" in url]
    assert verified, f"hosted login must still verify the key upstream; saw {sent}"


def test_login_honors_an_explicit_freesolo_url_on_a_self_hosted_plane(monkeypatch, tmp_path):
    """An operator running their own Freesolo-compatible auth backend keeps verification.

    The opt-out is keyed on --api-url, so an explicit --freesolo-url must still win; otherwise
    the fix would silently disable verification for someone who asked for it.
    """
    _rc, sent = _login_capturing_the_wire(
        monkeypatch,
        tmp_path,
        api_url="https://flash.example.internal",
        freesolo_url="https://auth.example.internal",
    )

    verified = [url for url, _ in sent if "/api/auth/verify" in url]
    assert verified, f"explicit --freesolo-url must still be verified; saw {sent}"
    assert all(not is_freesolo_hosted_url(url) for url, _ in sent), sent


def test_login_against_a_self_hosted_plane_rejects_a_bad_key(monkeypatch, tmp_path, capsys):
    """Not phoning home must not become "accept anything".

    Skipping the freesolo verify without putting the plane's own check in its place made
    `flash login --api-key <typo>` exit 0 and SAVE the bad key: credentials are stored before the
    identity card is fetched, and `_identity_or_none` swallows errors on purpose so a hiccup can't
    fail an already-verified login. The 401 then surfaced later on an unrelated command.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)

    import flash.client.config as client_config

    importlib.reload(client_config)
    from flash.client import ApiError

    class _RejectingPlane:
        def __init__(self, *a, **k):
            pass

        def me(self):
            raise ApiError(401, "invalid or missing API key")

    monkeypatch.setattr(cli_account, "ApiClient", _RejectingPlane)
    args = types.SimpleNamespace(
        api_key="wrong-key", api_url="http://my-plane:8080", freesolo_url=None
    )

    assert cli_account.cmd_login(args) == 1, "a key the plane rejects must fail the login"
    assert "login failed" in capsys.readouterr().err
    # and nothing was persisted, so the next command doesn't run on a key known to be bad.
    assert client_config.load_credentials()[1] is None


def _login_against_plane(monkeypatch, tmp_path, plane_cls, *, api_key="a-key"):
    """Run cmd_login against a self-hosted plane stubbed by `plane_cls`; return (rc, saved_key)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FREESOLO_API_KEY", raising=False)

    import flash.client.config as client_config

    importlib.reload(client_config)

    monkeypatch.setattr(cli_account, "ApiClient", plane_cls)
    args = types.SimpleNamespace(api_key=api_key, api_url="http://my-plane:8080", freesolo_url=None)
    try:
        rc = cli_account.cmd_login(args)
    except Exception as exc:  # a crash must not be the thing that "stops" a bad login
        rc = f"raised {type(exc).__name__}"
    return rc, client_config.load_credentials()[1]


def test_plane_verification_gets_a_real_request_timeout(monkeypatch, tmp_path):
    """A valid plane that is merely SLOW must not have its key rejected.

    Verification reused `_IDENTITY_LOOKUP_TIMEOUT_S` (5s), which exists for the OPTIONAL identity
    card where abandoning a slow lookup is harmless. As the timeout for MANDATORY auth it turned a
    cold-starting plane -- the documented quickstart case -- into a hard login failure, while the
    hosted path allowed 30s for the very same decision.
    """
    from flash.client import RequestTimeoutError

    class _SlowButValidPlane:
        def __init__(self, api_url, api_key=None, timeout=None):
            self.timeout = timeout

        def me(self):
            # answers in 8s: comfortable within a real request budget, fatal under the 5s one.
            if self.timeout is not None and self.timeout < 8.0:
                raise RequestTimeoutError("request timed out")
            return {"kind": "internal", "key_prefix": "operator"}

    rc, saved = _login_against_plane(monkeypatch, tmp_path, _SlowButValidPlane, api_key="good-key")
    assert rc == 0, "a slow but valid plane must not fail the login"
    assert saved == "good-key"


@pytest.mark.parametrize(
    "response",
    [
        pytest.param({}, id="empty-2xx-body"),
        pytest.param("not-an-identity", id="truthy-non-object"),
        pytest.param({"detail": "ok"}, id="object-without-identity-fields"),
        pytest.param({"kind": "internal"}, id="missing-key_prefix"),
    ],
)
def test_a_plane_that_does_not_return_an_identity_fails_the_login(monkeypatch, tmp_path, response):
    """Reaching the endpoint is not proof the key was accepted.

    `_request` turns an empty 2xx body into `{}`, so a misrouted --api-url (a proxy, a health
    responder, an older service without /v1/me) returns success having never consulted
    `require_key`. Treating that as verified persisted an ARBITRARY key and printed "logged in" --
    the same "I reached it, so it must have authenticated me" mistake this path exists to fix.
    """

    class _NotAFlashPlane:
        def __init__(self, *a, **k):
            pass

        def me(self):
            return response

    rc, saved = _login_against_plane(monkeypatch, tmp_path, _NotAFlashPlane, api_key="unverified")
    assert rc == 1, f"a non-identity response must fail the login, got rc={rc}"
    # the key must not be persisted -- including in the crash case, where an exception AFTER
    # save_credentials would still leave an unusable key on disk.
    assert saved is None, f"an unverified key was persisted: {saved!r}"


def test_read_json_or_empty_returns_dict_for_non_object(tmp_path):
    """read_json_or_empty honors its ``-> dict`` contract: valid-but-non-object JSON (list,
    scalar, null) and unreadable/empty content all yield ``{}`` so config/credential callers
    can ``.get(...)`` / item-assign without an AttributeError/TypeError bricking every command."""
    from flash._internal.fileio import read_json_or_empty

    p = tmp_path / "config.json"
    for content in ("[]", "5", '"x"', "null", "[1,2,3]", "not json at all", ""):
        p.write_text(content)
        assert read_json_or_empty(p) == {}, content
    p.write_text('{"api_url": "https://x", "api_key": "k"}')
    assert read_json_or_empty(p) == {"api_url": "https://x", "api_key": "k"}
    assert read_json_or_empty(tmp_path / "missing.json") == {}
