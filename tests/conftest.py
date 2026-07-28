"""Offline-by-default test harness.

There is no "skip the network" env flag; instead this autouse fixture stubs the network
boundaries the production code would otherwise reach, so the whole suite stays hermetic (no
real RunPod / Hugging Face calls) without any env switch. A test that exercises one of
these boundaries monkeypatches it itself — applied after this fixture, so the test's patch
wins. (The freesolo auth verify isn't stubbed here because tests that touch it either patch
``_freesolo_verify`` or ``urllib.request.urlopen`` directly; a global ``urlopen`` stub would
also break the client tests, which talk to a real loopback server.)
"""

from __future__ import annotations

import contextlib

import pytest


@pytest.fixture(scope="session", autouse=True)
def _close_status_reporter_after_suite():
    yield
    import flash.runner as runner

    runner._shutdown_status_reporter(close=True)


@pytest.fixture(autouse=True)
def _reset_status_reporter_before_test():
    import flash.runner as runner

    runner._shutdown_status_reporter(close=True)
    runner._open_status_reporter()


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # HF param probe -> offline: sizing for an unlisted model falls back to the heuristic
    # (24 GB tier) instead of hitting the Hugging Face API.
    import flash.engine.vram as vram

    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda model_id: None, raising=False)

    # RunPod endpoint listing -> offline: the idle-endpoint sweep (deploy-time quota reclaim
    # and the startup/post-run orphan sweep) lists account endpoints. Default to "no endpoints"
    # so a sweep never reaches the real API; sweep tests monkeypatch it after this fixture.
    import flash.providers.runpod.api as runpod_api

    monkeypatch.setattr(runpod_api, "list_endpoints", lambda: [], raising=False)

    # Credential scrubbing. Importing ``runpod_flash`` runs ``load_dotenv(find_dotenv(usecwd=True))``
    # at module scope, which walks UP out of the repo and loads whatever .env it finds -- so on an
    # operator box real keys appear in os.environ partway through the suite, as soon as some test
    # first imports that package. Deleting a key BEFORE that import is worthless: load_dotenv skips
    # names already set, so a name we just unset is exactly the one it fills back in.
    #
    # Force the import first, then delete. One ordering, one deletion pass, no name that can be
    # scrubbed on the wrong side of the import.
    with contextlib.suppress(Exception):  # package absent in a client-only checkout
        import runpod_flash  # noqa: F401

    # Secrets flash forwards AUTOMATICALLY (no per-job declaration) are the dangerous ones: a
    # submit-path test asserting on the outgoing payload picks them up from the operator's ambient
    # env and fails, and a less careful test would ship a real key into a fixture. Derived from the
    # production constant so a key added there is scrubbed here without touching this file.
    #
    # Lambda and Vast are OPT-IN instance-based complements: leaving their keys set makes the
    # provider "available" and pulls live capacity/pricing into offline tests (allocation
    # candidates, registry). A provider test opts back in with ``monkeypatch.setenv(...)``.
    from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS

    for _key in {"RUNPOD_API_KEY", "LAMBDA_API_KEY", "VAST_API_KEY"} | set(
        DEFAULT_RUNTIME_SECRET_KEYS
    ):
        monkeypatch.delenv(_key, raising=False)

    # The RunPod key pool caches the parsed RUNPOD_API_KEY at module level (so collapsing
    # it to a single active key never loses the rest of the pool). Reset it around every
    # test so a key set/collapsed by one test can't leak into the next.
    import flash.providers.runpod.keys as rp_keys

    rp_keys.reset()

    # Always-on artifact GC: the control-plane lifespan sweeps ONCE on startup (when an operator
    # HF_TOKEN is set). Stub it to a no-op so offline TestClient startups never reach HF/serving;
    # tests/test_repo_cleanup.py restores the real function to exercise the genuine sweep.
    import flash.server.repo_cleanup as _rc

    monkeypatch.setattr(_rc, "run_scheduled_cleanup", lambda *a, **k: 0, raising=False)

    yield
    rp_keys.reset()


@pytest.fixture(autouse=True)
def _fast_serving_readback(monkeypatch):
    """Zero the deploy read-back backoff so verification polls don't slow the suite."""
    import flash.serve.deploy as _deploy

    monkeypatch.setattr(_deploy, "READBACK_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(_deploy, "ACTIVATION_READBACK_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(_deploy, "SMOKE_RETRY_FALLBACK_DELAY_SECONDS", 0.0)


@pytest.fixture
def stub_serving_registry(monkeypatch):
    """Patch GET /adapters to return the given records (deploy read-back verification)."""

    def _stub(*records: dict):
        import flash.serve.deploy as _deploy

        class _RegistryResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"ok": True, "adapters": list(records)}

        monkeypatch.setattr(_deploy.httpx, "get", lambda *a, **k: _RegistryResp())

    return _stub
