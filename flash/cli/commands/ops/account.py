"""account, identity, and project command handlers."""

from __future__ import annotations

import os
import sys
import uuid

from flash import __version__
from flash._internal.channel import BRAND_NAME, CLI_NAME
from flash.cli.ui import render, tables
from flash.client import (
    ApiClient,
    ClientError,
    client_from_config,
    save_credentials,
    verify_freesolo_key,
)
from flash.client.config import load_credentials, load_credentials_with_source
from flash.client.http import freesolo_base_url, has_freesolo_backend
from flash.serve.contract.urls import is_freesolo_hosted_url


def cmd_version(args) -> int:
    if render.styled():
        print(render.version(__version__))
    else:
        print(f"{BRAND_NAME} {__version__}")
    return 0


def unavailable_without_a_freesolo_backend(what: str, *, because: str, instead: str) -> ClientError:
    """The refusal for a command backed only by the hosted Freesolo backend.

    These commands are pure client->api.freesolo.co calls that never touch the plane. Left alone
    they send the operator's plane credential to a service they have no relationship with, which
    answers 401 -- the same failure `flash env setup` hit on the documented quickstart. Naming the
    reason and the way forward is what separates "not built for your deployment" from "your key is
    wrong", which is what a bare 401 says.
    """
    return ClientError(f"{what} is not available on a self-hosted plane: {because}. {instead}.")


def _verifies_against_freesolo(api_url: str, freesolo_url: str | None) -> bool:
    """Whether ``flash login`` should check this key against the hosted Freesolo backend.

    A self-hosted plane verifies ``FREESOLO_INTERNAL_KEY`` itself, so sending that credential to
    the hosted backend would leak it. The client can infer standalone mode only from ``--api-url``;
    an explicit ``--freesolo-url`` still requests external verification. False routes verification
    to ``_verify_key_against_plane`` rather than accepting the key unchecked.
    """
    if freesolo_url:
        return True
    return is_freesolo_hosted_url(api_url)


def _plaintext_transport_warning(api_url: str) -> str:
    """Warn when the api url would carry the key over unencrypted, non-loopback HTTP.

    The key travels as an ``Authorization: Bearer`` header on login and on every command after it,
    and on a standalone plane it is the entire authorization boundary and owns every run -- so an
    observer can submit billed GPU jobs and read or cancel existing ones. Loopback never leaves the
    machine, so it stays quiet: local development is the one place plaintext is fine.

    A warning rather than a refusal: an operator may front the plane with a TLS-terminating proxy
    reached over a private link, and this cannot see that. It must be printed BEFORE the key is
    transmitted, so the caller can still abort.
    """
    from urllib.parse import urlparse

    parsed = urlparse((api_url or "").strip())
    if parsed.scheme != "http":
        return ""
    host = (parsed.hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".localhost"):
        return ""
    return (
        f"{api_url} is plaintext HTTP, so your API key is sent unencrypted and anyone who can "
        "observe the connection can reuse it to submit billed GPU jobs and control your runs. "
        "Use https:// for a remote plane; plaintext is safe only on localhost."
    )


def _plaintext_login_warnings(api_url: str | None, freesolo_url: str | None) -> list[str]:
    """Warn for every url `flash login` may send the key to, not just the plane's.

    Login has two possible destinations: the plane, which the saved credential targets for every
    later command, and the Freesolo identity backend that ``verify_freesolo_key`` checks the key
    against. The second is RESOLVED rather than passed -- ``freesolo_base_url`` falls back to
    ``FREESOLO_BASE_URL`` and then to the hosted default -- so reading the ``--freesolo-url`` arg
    alone still sent the bearer key to an ``http://`` identity backend named by that env var with no
    warning, the case where the operator is most likely to believe they are protected.

    The identity url is resolved only when `_verifies_against_freesolo` says login will really
    contact it: on a self-hosted plane the key goes to the plane alone, and warning about an env var
    nothing reads is noise. The hosted default is https, so it never warns on its own.
    """
    destinations = [api_url]
    if _verifies_against_freesolo(api_url or "", freesolo_url):
        destinations.append(freesolo_base_url(freesolo_url))
    seen: set[str] = set()
    warnings: list[str] = []
    for url in destinations:
        normalized = (url or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        warning = _plaintext_transport_warning(normalized)
        if warning:
            warnings.append(warning)
    return warnings


def cmd_login(args) -> int:
    api_url = args.api_url or load_credentials()[0]
    identity: dict | None = None
    if args.api_key:
        print(
            "warning: --api-key is visible in process listings; prefer FREESOLO_API_KEY",
            file=sys.stderr,
        )
    # before any request: the warning is worthless once the key has already gone over the wire.
    # both destinations are checked here rather than at the branch below, because which one receives
    # the key is decided after this point and the caller needs the warning while it can still abort.
    for transport_warning in _plaintext_login_warnings(
        api_url, getattr(args, "freesolo_url", None)
    ):
        print(
            render.warn(transport_warning) if render.styled() else f"warning: {transport_warning}",
            file=sys.stderr,
        )
    try:
        env_api_key = os.environ.get("FREESOLO_API_KEY")
        api_key = args.api_key or env_api_key
        if not api_key:
            raise ClientError(
                "no API key provided: pass `--api-key <key>` or set FREESOLO_API_KEY. "
                "Create or copy a key at https://freesolo.co/sign-in."
            )
        freesolo_url = getattr(args, "freesolo_url", None)
        if _verifies_against_freesolo(api_url, freesolo_url):
            verify_freesolo_key(api_key, base_url=freesolo_url)
        else:
            # a self-hosted plane is its own key issuer, so it is the only service that CAN
            # verify this key -- and the only one allowed to see it.
            identity = _verify_key_against_plane(api_key, api_url)
    except ClientError as exc:
        if getattr(args, "debug", False):
            raise
        print(render.login_failed(str(exc)), file=sys.stderr)
        return 1
    save_credentials(api_key, api_url=api_url)
    if args.api_key and env_api_key and env_api_key != args.api_key:
        msg = (
            "FREESOLO_API_KEY is set and will override this saved login for future "
            "commands; unset FREESOLO_API_KEY to use the saved key."
        )
        print(render.warn(msg) if render.styled() else f"warning: {msg}", file=sys.stderr)
    # Show who they are right away (the same identity `flash whoami` prints) so they don't
    # have to run a second command. Never echo the key itself. The identity lookup is
    # best-effort: the key is already verified and stored, so a momentary control-plane
    # hiccup must not turn a successful login into a failure.
    print(
        render.login_ok(identity if identity is not None else _identity_or_none(api_key, api_url))
    )
    return 0


_IDENTITY_LOOKUP_TIMEOUT_S = 5.0
# Verification is MANDATORY, so it gets a normal request budget rather than the 5s the optional
# identity card uses: abandoning a slow lookup is harmless when the card is decoration, but here it
# would reject a valid key. Matches the hosted path's 30s (`verify_freesolo_key`) so the same
# decision gets the same budget whoever answers it -- a self-hosted plane that is cold-starting,
# behind a slow proxy, or under database contention is exactly the documented quickstart case.
_LOGIN_VERIFY_TIMEOUT_S = 30.0


def _verify_key_against_plane(api_key: str, api_url: str) -> dict:
    """Verify a key through the self-hosted plane's authenticated identity endpoint.

    Hosted verification would leak the plane-root credential, while skipping verification stores
    invalid keys. ``/v1/me`` both authenticates and returns identity. Require ``kind`` and
    ``key_prefix`` because ``_request`` converts an empty 2xx body to ``{}``; those fields are
    unconditional in ``flash/server/routes/meta.py``.
    """
    identity = ApiClient(api_url, api_key, timeout=_LOGIN_VERIFY_TIMEOUT_S).me()
    if not isinstance(identity, dict) or not identity.get("kind") or not identity.get("key_prefix"):
        raise ClientError(
            f"{api_url} answered the login check but did not return a Flash identity, so the key "
            "could not be verified and was not saved. Check that --api-url points at your Flash "
            'control plane (its /v1/health should report "service": "flash") rather than at a '
            "proxy or another service."
        )
    return identity


def _identity_or_none(api_key: str, api_url: str) -> dict | None:
    # Don't use client_from_config(): ambient FREESOLO_API_KEY would win and show wrong identity.
    try:
        return ApiClient(api_url, api_key, timeout=_IDENTITY_LOOKUP_TIMEOUT_S).me()
    except (ClientError, OSError, ValueError):
        return None


def cmd_whoami(args) -> int:
    _, _, key_source = load_credentials_with_source()
    print(render.whoami(client_from_config().me(), key_source))
    return 0


def cmd_projects_create(args) -> int:
    from flash.client import create_project

    api_url, api_key = load_credentials()
    if not has_freesolo_backend(api_url):
        # no directory to create a row in, and none needed: the plane accepts any well-shaped
        # uuid (server/projects.py under standalone()), so minting one locally IS the create.
        project_id = str(uuid.uuid4())
    else:
        if not api_key:
            raise ClientError(f"not logged in. Run `{CLI_NAME} login` before creating a project")
        project_id = create_project(args.name, getattr(args, "description", None), api_key)["id"]

    # both ids are reported the same way: where it came from is the only difference above.
    if render.styled():
        print(render.project_created(project_id, str(args.name).strip()))
    else:
        print(project_id)
    return 0


def cmd_projects_list(args) -> int:
    from flash.client import list_projects

    api_url, api_key = load_credentials()
    if not has_freesolo_backend(api_url):
        raise unavailable_without_a_freesolo_backend(
            "listing projects",
            because="a self-hosted plane keeps no project directory to enumerate",
            instead=(
                f"`{CLI_NAME} projects create <name>` mints a project id you record yourself; "
                "any id you already use in a config keeps working"
            ),
        )
    if not api_key:
        raise ClientError(f"not logged in. Run `{CLI_NAME} login` before listing projects")
    projects = list_projects(api_key)
    if render.styled():
        print(tables.projects_table(projects))
        return 0
    for project in projects:
        project_id = str(project.get("id") or "").strip()
        name = str(project.get("name") or "").strip()
        print(f"{project_id}\t{name}")
    return 0
