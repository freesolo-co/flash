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

# `pytest_addoption` for these flags lives in `tests/conftest.py`, not here: pytest only calls that
# hook on INITIAL conftests, so registering it in this subdirectory makes `--serving-url` an
# unrecognized-argument error for every invocation that does not name this directory explicitly
# (including a bare `pytest`, which resolves testpaths to `tests`).


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
def internal_key() -> str:
    """The key every request carries. Absent is legitimate: a local backend may not require one."""
    return (os.environ.get("FREESOLO_INTERNAL_KEY") or "").strip()


@pytest.fixture(scope="session")
def adapter_source(request) -> dict:
    """The real artifact to register. Every field has to be right or the backend cannot load it."""
    repo_id = _option(request, "--conformance-repo", "FLASH_CONFORMANCE_REPO")
    subfolder = _option(request, "--conformance-subfolder", "FLASH_CONFORMANCE_SUBFOLDER")
    base_model = _option(request, "--conformance-base-model", "FLASH_CONFORMANCE_BASE_MODEL")
    # Required, not defaulted. It is the commit the backend downloads AND the suffix of the
    # revision id, so the placeholder shape satisfies the grammar while naming a commit that does
    # not exist -- registration would fail minutes later, on the GPU, as an unresolvable revision
    # rather than as the missing argument it actually is.
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
        # Fail, do not skip. `--serving-url` was passed, so the suite is explicitly ON, and skipping
        # here leaves only /healthz and the 404 checks running -- a green exit that proves none of
        # registration, readiness, activation, chat, or teardown. The shorter command in
        # docs/serving-contract.md lands exactly here, so the skip made the documented invocation
        # silently vacuous.
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
def ready_timeout(request) -> float:
    return float(request.config.getoption("--conformance-ready-timeout"))


@pytest.fixture(scope="session")
def http(serving_url, internal_key):
    """A client bound to the target, carrying the internal key exactly as flash sends it."""
    # FAILS rather than skips. Reaching this fixture means the suite was explicitly enabled with
    # --serving-url, and a skip there is a false green: pytest exits 0 having checked not one
    # endpoint, which reads as "the backend conforms". The whole point of this suite is that its
    # green means something.
    try:
        import httpx
    except ImportError as exc:
        pytest.fail(
            "the conformance suite needs httpx to talk to the backend, and it is not installed: "
            f"{exc}. install it with `pip install httpx`, or drop --serving-url to skip the suite."
        )
    headers = {"X-Freesolo-Internal-Key": internal_key} if internal_key else {}

    def _origin(url) -> tuple[str, str, int | None]:
        return (url.scheme.lower(), (url.host or "").rstrip(".").lower(), url.port)

    target = _origin(httpx.URL(serving_url))

    def _strip_key_off_origin(request) -> None:
        # The same hook the shipped client installs, for the same reason: httpx strips only
        # `Authorization` and `Cookie` across an origin change, so the plane credential rides a
        # custom header straight to whatever host a redirect names. Following redirects without
        # this would make the suite leak the key that a real deploy would not.
        if "X-Freesolo-Internal-Key" not in request.headers:
            return
        if _origin(request.url) != target:
            del request.headers["X-Freesolo-Internal-Key"]

    # follow_redirects mirrors `_new_serving_client`. Modal answers a slow registration, chat, or
    # delete with a 303 to a same-origin async-result poll url, and the shipped client follows it;
    # a suite that does not would see the bare 303 and fail a backend that `flash models deploy`
    # drives successfully -- the suite rejecting behavior the client handles.
    with httpx.Client(
        base_url=serving_url,
        headers=headers,
        timeout=120.0,
        follow_redirects=True,
        max_redirects=100,
        event_hooks={"request": [_strip_key_off_origin]},
    ) as client:
        yield client


@pytest.fixture
def run_id() -> str:
    """A fresh run id per test.

    Aliases and revisions are permanent records on the target, so reusing one id across tests would
    make each run depend on the leftovers of the last. The suffix keeps it inside flash's run-id
    grammar while staying unique.
    """
    return f"conformance-{uuid.uuid4().hex[:12]}"
