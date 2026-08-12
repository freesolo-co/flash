"""Opt-in gate and shared client for the serving conformance suite.

The suite runs `docs/serving-contract.md` against a REAL backend, so it is skipped entirely unless
`--serving-url` names one. That keeps it free in the offline run while making one command the
answer to "does my backend work with flash".

It WRITES: registering, activating, and deleting adapters are real state changes. Point it at a
backend you own.
"""

from __future__ import annotations

import os
import uuid

import pytest

from tests.conftest import PLACEHOLDER_HF_REVISION

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
    missing = [
        flag
        for flag, value in (
            ("--conformance-repo", repo_id),
            ("--conformance-subfolder", subfolder),
            ("--conformance-base-model", base_model),
        )
        if not value
    ]
    if missing:
        pytest.skip(f"serving conformance needs a real adapter to register; missing {missing}")
    return {
        "repo_id": repo_id,
        "subfolder": subfolder,
        "base_model": base_model,
        "repo_type": _option(request, "--conformance-repo-type", "FLASH_CONFORMANCE_REPO_TYPE")
        or "dataset",
        "hf_revision": _option(
            request,
            "--conformance-hf-revision",
            "FLASH_CONFORMANCE_HF_REVISION",
            PLACEHOLDER_HF_REVISION,
        ),
    }


@pytest.fixture(scope="session")
def ready_timeout(request) -> float:
    return float(request.config.getoption("--conformance-ready-timeout"))


@pytest.fixture(scope="session")
def http(serving_url, internal_key):
    """A client bound to the target, carrying the internal key exactly as flash sends it."""
    httpx = pytest.importorskip("httpx")
    headers = {"X-Freesolo-Internal-Key": internal_key} if internal_key else {}
    with httpx.Client(base_url=serving_url, headers=headers, timeout=120.0) as client:
        yield client


@pytest.fixture
def run_id() -> str:
    """A fresh run id per test.

    Aliases and revisions are permanent records on the target, so reusing one id across tests would
    make each run depend on the leftovers of the last. The suffix keeps it inside flash's run-id
    grammar while staying unique.
    """
    return f"conformance-{uuid.uuid4().hex[:12]}"
