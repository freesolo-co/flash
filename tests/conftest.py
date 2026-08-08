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

    # The cost estimator memoizes open-model sizes for the life of the process (see
    # cost.facts._SIZE_MEMO). Clear it per test so one test's stubbed size cannot answer another
    # test's lookup, which would make a size-dependent assertion pass for the wrong reason.
    import flash.cost.facts as cost_facts

    cost_facts._SIZE_MEMO.clear()

    # RunPod endpoint listing -> offline: the idle-endpoint sweep (deploy-time quota reclaim
    # and the startup/post-run orphan sweep) lists account endpoints. Default to "no endpoints"
    # so a sweep never reaches the real API; sweep tests monkeypatch it after this fixture.
    import flash.providers.runpod.api as runpod_api

    monkeypatch.setattr(runpod_api, "list_endpoints", list, raising=False)

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
    # FREESOLO_API_KEY is scrubbed for a different reason than the rest: it is not forwarded to a
    # job, it selects the ORGANIZATION every CLI command runs against. A box with a .env above the
    # repo (the dotenv walk-up above) gets it refilled mid-suite, and the CLI then prepends its
    # "this overrides your saved login" warning to stderr -- so tests asserting on exact stderr fail
    # on a developer machine while passing in CI, which reads as a broken test rather than a leaked
    # credential. Every test that wants the var sets it itself after this fixture.
    #
    # FLASH_STANDALONE and FLASH_HF_NAMESPACE are not credentials but they switch global MODE, and a
    # developer shell that followed SELF_HOSTING.md exports both. Left set, managed-mode tests take
    # the standalone auth path, the disabled billing loop, a raising ``serving_base_url``, or a
    # non-default artifact namespace -- and CI, whose env is clean, still passes. Every test that
    # wants either one sets it itself (see tests/test_server_standalone.py).
    from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS

    for _key in {
        "FREESOLO_API_KEY",
        "LAMBDA_API_KEY",
        "VAST_API_KEY",
        "PARASAIL_API_KEY",
        "FLASH_CONTROL_PANEL_URL",
        "FLASH_TEACHER_CAPABILITY",
        "FLASH_STANDALONE",
        "FLASH_HF_NAMESPACE",
    } | set(DEFAULT_RUNTIME_SECRET_KEYS):
        monkeypatch.delenv(_key, raising=False)

    # RunPod is OVERWRITTEN, not deleted: same hygiene (no operator key ever reaches a test) with
    # the default substrate intact. is_configured() is gated on the key pool, so deleting the var
    # leaves the harness with NO provider at all -- every allocator test then fails "no allocatable
    # GPU" for want of a credential rather than for the reason it is testing. Two fake accounts so
    # the multi-account failover paths have a pool to walk. A test that wants runpod unconfigured
    # (or a specific pool) sets its own value after this fixture.
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test-a,rp-test-b")

    # Same hazard, one layer up: the client discovers runtime secrets from the process env at submit
    # time, so an operator shell that exports WANDB_API_KEY makes every dry-run test assert against
    # that real value instead of the fixture's. CI has no such key and passes; the box that does have
    # one fails, which reads as a broken test rather than a leaked env. Drive the scrub off the real
    # constant so a new default key cannot reintroduce the gap. A test that wants one sets it back.
    from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS

    for _key in DEFAULT_RUNTIME_SECRET_KEYS:
        monkeypatch.delenv(_key, raising=False)

    # The RunPod key pool caches the parsed RUNPOD_API_KEY at module level (so collapsing
    # it to a single active key never loses the rest of the pool). Reset it around every
    # test so a key set/collapsed by one test can't leak into the next.
    import flash.providers.runpod.keys as rp_keys

    rp_keys.reset()

    # Provider singletons are @cache'd per name, and is_configured() now reads the env, so a test
    # that sets or clears a provider key would otherwise see the PREVIOUS test's answer. Clear
    # around every test: the cache exists to avoid re-importing, not to freeze configuration.
    import flash.providers as _providers

    _providers._get_provider.cache_clear()

    # Always-on artifact GC: the control-plane lifespan sweeps ONCE on startup (when an operator
    # HF_TOKEN is set). Stub it to a no-op so offline TestClient startups never reach HF/serving;
    # tests/test_repo_cleanup.py restores the real function to exercise the genuine sweep.
    import flash.server.repo_cleanup as _rc

    monkeypatch.setattr(_rc, "run_scheduled_cleanup", lambda *a, **k: 0, raising=False)

    yield
    rp_keys.reset()
    _providers._get_provider.cache_clear()


@pytest.fixture(autouse=True)
def _child_subreaper_does_not_leak_between_tests():
    """Restore PR_SET_CHILD_SUBREAPER after every test, not just the ones that opt in.

    ``adopt_orphaned_descendants`` sets a PROCESS-global flag, and the production entry points call
    it unconditionally: any test reaching ``run_verl_training`` or ``kill_process_group`` flips this
    pytest process from 0 to 1 for the rest of the session. From then on every later test adopts
    orphaned grandchildren it never waits on, so it accumulates zombies and its result depends on
    what ran before it. The ``subreaper`` fixture in ``test_backend_common`` restores only the tests
    that ask for it, which is not where the flag is now set.

    Autouse and here rather than in that module: five test files reach those entry points, and the
    tests harmed are any that follow -- so the guarantee has to cover the whole suite. The module's
    own ``_child_subreaper_enabled`` still nests inside this untouched.

    ``_ADOPTS_ORPHANS`` is reset alongside the kernel flag. It is the module's memory of having
    claimed adoption; leaving it True while clearing the flag would leave the module believing a
    claim the kernel no longer holds, and the next call would skip the prctl it needs to make.
    """
    import ctypes

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        current = ctypes.c_int(0)
        # PR_GET_CHILD_SUBREAPER / PR_SET_CHILD_SUBREAPER, from linux/prctl.h.
        if libc.prctl(37, ctypes.byref(current), 0, 0, 0) != 0:
            yield  # pragma: no cover - linux 3.4+
            return
    except (OSError, AttributeError, ValueError):  # pragma: no cover - not reachable on linux
        yield  # no libc: nothing can set the flag either, so there is nothing to restore
        return

    previous = current.value
    try:
        yield
    finally:
        libc.prctl(36, previous, 0, 0, 0)
        import flash.engine.worker.backend_common as _vc

        _vc._ADOPTS_ORPHANS = bool(previous)


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
