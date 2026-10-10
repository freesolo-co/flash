"""Opt-in gate and shared client for the serving conformance suite.
The suite runs `docs/serving-contract.md` against a REAL backend, so it is skipped entirely unless
`--serving-url` names one. That keeps it free in the offline run while making one command the
answer to "does my backend work with flash".
It WRITES: registering, activating, and deleting adapters are real state changes. Point it at a
backend you own.
"""

from __future__ import annotations

import os
import re
import uuid

import pytest

from tests.conftest import PLACEHOLDER_HF_REVISION

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def _option(request, name: str, env: str, default: str | None = None) -> str | None:
    value = request.config.getoption(name) or os.environ.get(env) or default
    return str(value).strip() if value else None


@pytest.fixture(scope="session")
def serving_url(request) -> str:
    url = _option(request, "--serving-url", "FLASH_CONFORMANCE_SERVING_URL")
    if not url:
        pytest.skip("serving conformance disabled; pass --serving-url to run it against a backend")
    return url.rstrip("/")


@pytest.fixture(scope="session")
def serving_key() -> str:
    """the bearer token the serving app authenticates every request with.

    this is the deployment's own inference credential, the value `flash serve deploy` read from
    FLASH_SERVING_KEY. it is deliberately NOT FREESOLO_INTERNAL_KEY: that key belongs to the
    freesolo control plane and must never be sent to a customer-owned endpoint.
    """

    return (os.environ.get("FLASH_SERVING_KEY") or "").strip()


@pytest.fixture(scope="session")
def adapter_source(request) -> dict:
    repo_id = _option(request, "--conformance-repo", "FLASH_CONFORMANCE_REPO")
    subfolder = _option(request, "--conformance-subfolder", "FLASH_CONFORMANCE_SUBFOLDER")
    base_model = _option(request, "--conformance-base-model", "FLASH_CONFORMANCE_BASE_MODEL")
    hf_revision = _option(request, "--conformance-hf-revision", "FLASH_CONFORMANCE_HF_REVISION")
    missing = [
        flag
        for flag, value in (
            ("--conformance-repo", repo_id),
            ("--conformance-subfolder", subfolder),
            ("--conformance-base-model", base_model),
            ("--conformance-hf-revision", hf_revision),
        )
        if not value
    ]
    if missing:
        pytest.fail(
            f"serving conformance needs a real adapter to register; missing {missing}. "
            f"Without one the suite can only check /healthz, which does not prove the contract. "
            f"Pass --serving-url alone only if you meant to run nothing."
        )
    if not _COMMIT_SHA.fullmatch(hf_revision or "") or hf_revision == PLACEHOLDER_HF_REVISION:
        pytest.fail(
            f"--conformance-hf-revision must be a real 40-character commit sha, got "
            f"{hf_revision!r}. It is both the artifact's pinned commit and part of the immutable "
            f"revision id, so a branch name or placeholder cannot stand in for it."
        )
    return {
        "repo_id": repo_id,
        "subfolder": subfolder,
        "base_model": base_model,
        "repo_type": _option(request, "--conformance-repo-type", "FLASH_CONFORMANCE_REPO_TYPE")
        or "dataset",
        "hf_revision": hf_revision,
    }


@pytest.fixture(scope="session")
def ready_timeout(request, adapter_source) -> float:
    from flash.serve.deployment.readiness import revision_ready_budget_seconds

    override = request.config.getoption("--conformance-ready-timeout")
    if override is not None:
        return float(override)
    return revision_ready_budget_seconds(adapter_source["base_model"])


def _require_httpx() -> None:
    try:
        import httpx  # noqa: F401
    except ImportError as exc:
        pytest.fail(
            "the conformance suite needs httpx to talk to the backend, and it is not installed: "
            f"{exc}. install it with `pip install httpx`, or drop --serving-url to skip the suite."
        )


def _build_client(serving_url, serving_key):
    import httpx

    # the serving app reads only `authorization` and compares a sha-256 of the bearer value
    # against the inference token it launched with. it never reads X-Freesolo-Internal-Key --
    # that header belonged to the freesolo-gateway app this path replaces, so sending it here
    # authenticates nothing and every request comes back 401.
    headers = {"Authorization": f"Bearer {serving_key}"} if serving_key else {}

    def _origin(url) -> tuple[str, str, int | None]:
        return (url.scheme.lower(), (url.host or "").rstrip(".").lower(), url.port)

    target = _origin(httpx.URL(serving_url))

    def _strip_key_off_origin(request) -> None:
        if "Authorization" not in request.headers:
            return
        if _origin(request.url) != target:
            del request.headers["Authorization"]

    return httpx.Client(
        base_url=serving_url,
        headers=headers,
        timeout=60.0,
        follow_redirects=True,
        max_redirects=100,
        event_hooks={"request": [_strip_key_off_origin]},
    )


CHAT_REQUEST_TIMEOUT_SECONDS = 30 * 60.0


@pytest.fixture(scope="session")
def chat_timeout():
    return CHAT_REQUEST_TIMEOUT_SECONDS


@pytest.fixture(scope="session")
def serving_client_factory(serving_url, serving_key):
    _require_httpx()

    def _factory():
        return _build_client(serving_url, serving_key)

    return _factory


@pytest.fixture(scope="session")
def http(serving_url, serving_key):
    _require_httpx()
    with _build_client(serving_url, serving_key) as client:
        yield client


@pytest.fixture
def run_id() -> str:
    return f"conformance-{uuid.uuid4().hex[:12]}"
