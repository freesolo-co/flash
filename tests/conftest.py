"""Offline-by-default test harness.

Autouse fixtures stub production network boundaries; later test patches still win. Freesolo auth
is not stubbed globally because client tests use a real loopback server.
"""

from __future__ import annotations

import contextlib
import os
import shlex

import pytest

PLACEHOLDER_HF_REVISION = "0" * 40


def _positive_finite_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise pytest.UsageError(
            f"--conformance-ready-timeout must be a number, got {raw!r}"
        ) from None
    if value != value or value in (float("inf"), float("-inf")) or value <= 0:
        raise pytest.UsageError(
            f"--conformance-ready-timeout must be a positive, finite number of seconds, got {raw!r}"
        )
    return value


def pytest_addoption(parser):
    group = parser.getgroup("serving conformance")
    group.addoption(
        "--serving-url",
        default=None,
        help="serving backend control root to run the contract against; skips the suite when unset",
    )
    group.addoption(
        "--conformance-repo",
        default=None,
        help="HuggingFace repo holding the adapter to register (default: $FLASH_CONFORMANCE_REPO)",
    )
    group.addoption(
        "--conformance-subfolder",
        default=None,
        help="path to the adapter inside the repo",
    )
    group.addoption(
        "--conformance-base-model",
        default=None,
        help="base model the adapter was trained on; must match what the backend serves",
    )
    group.addoption(
        "--conformance-hf-revision",
        default=None,
        help="40-char commit sha the adapter is pinned to; required, and part of the revision id",
    )
    group.addoption(
        "--conformance-repo-type",
        default=None,
        help="repo_type sent at registration (default: dataset, matching flash)",
    )
    group.addoption(
        "--conformance-ready-timeout",
        type=_positive_finite_seconds,
        default=None,
        help=(
            "seconds to wait for a revision to reach ready "
            "(default: flash's own deploy budget for the base model under test)"
        ),
    )


def _temp_root_lever(config) -> str:
    """How to move pytest's temp root, in the same precedence pytest itself resolves it.

    ``TempPathFactory.getbasetemp`` takes ``--basetemp`` first, then ``PYTEST_DEBUG_TEMPROOT``, and
    only then falls through to ``tempfile.gettempdir()`` (which is what reads ``TMPDIR``). Telling a
    developer running with ``--basetemp`` to change ``TMPDIR`` sends them to a knob that has no
    effect on the path in front of them -- exactly the misdirection this hook exists to end.
    """
    if getattr(config.option, "basetemp", None):
        return "pass a different directory to `--basetemp`"
    if os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        return "point `PYTEST_DEBUG_TEMPROOT` elsewhere"
    return "point `TMPDIR` elsewhere"


def _failing_cache_root(text: str):
    """The env cache root the exception refused, or None if the message does not name one.

    Every branch of ``validate_cache_root_ancestors`` ends with "env cache root <path>" (with an
    optional "use"/"load environment code from" in between), so the root is always recoverable from
    the message the test actually failed with -- no re-deriving where the cache should have been.

    The path is terminated by the message's own trailing text, not by whitespace: a temp root under
    a directory with a space in its name is a perfectly ordinary ``TMPDIR``, and a ``\\S+`` match
    would stop at that space, hand back a prefix that exists nowhere, and drop the diagnostic on a
    setup that needs it.
    """
    import re
    from pathlib import Path

    tail = text[text.find("refusing") :] or text
    match = re.search(r"env cache root (.+?)(?: -- chmod \+t it or move the cache root|\s*$)", tail)
    return Path(match.group(1)) if match else None


def _named_ancestor(text: str):
    """The ancestor the exception blamed, or None. Pairs with the cache root as the causality test.

    Containment alone does not establish cause: two tests here ``chmod`` a directory UNDER
    ``tmp_path`` on purpose, so the refused root is inside the temp tree while the ancestor at fault
    is one the test created. Set ``TMPDIR`` somewhere untrusted as well and the temp root's own bad
    ancestor exists too -- so a containment-only check blames it and sends the reader to fix a
    directory that had nothing to do with the failure.
    """
    import re
    from pathlib import Path

    match = re.search(r"env cache root ancestor (.+?) is group/other-writable", text)
    return Path(match.group(1)) if match else None


def _is_within(path, base) -> bool:
    """True when ``path`` is ``base`` or lives under it, comparing resolved paths."""
    try:
        resolved, root = path.resolve(), base.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _is_same_path(left, right) -> bool:
    """True when both name the same directory. Compares raw AND resolved, since the exception and
    the walk can each have found it by a different route through a symlinked temp root."""
    try:
        return left == right or left.resolve() == right.resolve()
    except OSError:
        return left == right


def _trust_failure_texts(exc) -> list[str]:
    """Every trust-check message inside ``exc``, outermost first.

    A fixture finalizer that raises alongside another one does not reach the hook as itself:
    pytest bundles teardown errors into a ``BaseExceptionGroup`` whose ``str()`` is only
    ``errors while tearing down <Function ...> (2 sub-exceptions)`` -- the trust message is in a
    member, not the summary. Matching on the top-level text alone therefore drops the diagnostic
    for exactly the multi-finalizer teardown this hook claims to cover. Recurse; groups nest.

    ALL of them, not the first: two finalizers can each hit the trust check for different reasons,
    one on a directory the test made unsafe on purpose and one on the temp root's own ancestor.
    Stopping at the first would let the deliberate failure hide the actionable one, which is the
    same silence this hook exists to end -- so the caller tests each message and uses any that
    passes the causality checks.
    """
    if isinstance(exc, BaseExceptionGroup):
        return [text for member in exc.exceptions for text in _trust_failure_texts(member)]
    text = str(exc)
    return [text] if "env cache root ancestor" in text and "sticky bit" in text else []


def _blames_the_temp_root(text: str, base, ancestor) -> bool:
    """True when THIS refusal is the temp root's fault: refused root inside it, and same ancestor."""
    failing_root, blamed = _failing_cache_root(text), _named_ancestor(text)
    if failing_root is None or not _is_within(failing_root, base):
        return False
    return blamed is not None and _is_same_path(blamed, ancestor)


def _untrusted_tmpdir_ancestor(base):
    """The first ancestor of ``base`` that the env cache's trust checks will refuse, if any.

    Mirrors the group/other-writable-without-sticky-bit rule in
    ``flash.envs.meta.cache_security.validate_cache_root_ancestors``, including the chain it walks:
    raw parents AND resolved ones. A symlinked temp root makes those two different sets, and the
    exception names whichever one it found -- so walking only the raw chain returns a path that
    never appears in the message, and the caller's causality check silently drops the diagnostic
    on exactly the setup that most needs it.
    """
    import stat

    chain = (base, *base.parents, base.resolve(), *base.resolve().parents)
    for ancestor in dict.fromkeys(chain):
        try:
            info = ancestor.stat()
        except OSError:
            continue
        if (info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)) and not (info.st_mode & stat.S_ISVTX):
            return ancestor, info.st_mode & 0o777
    return None, None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Name the temp root on env-cache trust failures, instead of only the code under test.

    ``validate_cache_root_ancestors`` refuses a cache root with a group/other-writable ancestor
    that has no sticky bit. That check is right, and the tests that trip it are not: they point
    ``_CACHE_ROOT`` at ``tmp_path``, whose ancestors belong to whatever chose pytest's temp root.
    Set ``TMPDIR`` to a shared 0775 directory and 13 tests across ``test_env_pull.py`` and
    ``test_verifiers.py`` go red naming ``cache_security.py`` and a generated cache root -- the
    code under test and a path the developer never chose -- while never naming the temp root,
    which is the thing that is actually wrong. Default ``/tmp`` is 1777 and sticky, so CI never
    sees it.

    Attached to the failing report rather than printed: ``addopts = -q`` suppresses
    ``pytest_report_header`` and capture swallows a fixture's ``print``, so both go missing in
    exactly the default run that needs them. A section on the failure is the one place pytest
    always shows it, and unlike raising in teardown it does not add a second error per test.

    Fires only when the trust check is what rejected the path AND the ancestor it named is the
    temp root's own, so it explains that failure and never editorializes over an unrelated one.
    """
    outcome = yield
    report = outcome.get_result()
    # setup and teardown too: an environment resolved in a fixture raises the same exception and
    # pytest reports it as an ERROR in that phase. those need the explanation exactly as much, and
    # gating on the call phase silently dropped it for every one of them.
    if not report.failed:
        return
    excinfo = getattr(call, "excinfo", None)
    if excinfo is None:
        return
    texts = _trust_failure_texts(excinfo.value)
    if not texts:
        return
    # a diagnostic must never be able to replace the failure it is explaining. `getbasetemp()`
    # CREATES the root on first call, so a failure that happened before any `tmp_path` fixture ran
    # -- a cache under HOME, say -- would create it here during reporting, and an unwritable
    # `--basetemp` parent would then raise inside the hook and surface as an INTERNALERROR that
    # buries the original traceback.
    try:
        base = item.config._tmp_path_factory.getbasetemp()
    except Exception:
        return
    ancestor, mode = _untrusted_tmpdir_ancestor(base)
    if ancestor is None:
        return
    # the temp root having an unsafe ancestor does not make it the CAUSE. two tests here chmod a
    # directory under `tmp_path` on purpose to exercise the refusal, and a cache root under HOME can
    # fail while the temp tree is spotless -- in both cases blaming the temp dir sends the reader to
    # a knob that changes nothing, which is the same misdirection as saying nothing.
    #
    # BOTH halves are required, and neither implies the other:
    #
    #   ancestor alone -- with HOME and the temp root as siblings under one 0775 directory, both
    #   name that directory, so a cache under HOME reads as a temp-dir problem.
    #   containment alone -- the two tests that `chmod` a directory under `tmp_path` on purpose put
    #   the refused root inside the temp tree while the ancestor at fault is one they created; if
    #   TMPDIR is also untrusted, this blames the temp root for a refusal it did not cause.
    #
    # requiring the refused root to be under the temp tree AND the blamed ancestor to be the temp
    # root's own leaves only the case where the temp root is genuinely why the check refused.
    if not any(_blames_the_temp_root(text, base, ancestor) for text in texts):
        return
    # `chmod +t` on the offending ancestor is the only fix that always works. moving the temp root
    # to a private directory does NOT, if that directory is still under the same bad ancestor --
    # every parent is checked, so 0700 below a 0775 parent fails identically. say "outside", not
    # "private".
    #
    # shell-quoted: a temp root under a directory with a space is ordinary, and `chmod +t /tmp/team
    # temp` silently addresses two paths that do not exist. this line is meant to be pasted.
    quoted = shlex.quote(str(ancestor))
    detail = (
        f"pytest's temp root is {base}, and its ancestor {ancestor} is group/other-writable "
        f"without the sticky bit (mode {mode:04o}), which the env cache trust checks refuse. "
        f"fix with `chmod +t {quoted}`, or {_temp_root_lever(item.config)} -- one whose whole "
        f"parent chain is trusted, so not another directory under {ancestor}."
    )
    report.sections.append(("the temp dir, not the code under test, caused this", detail))


@pytest.fixture(scope="session", autouse=True)
def _close_status_reporter_after_suite():
    yield
    from flash.runner.lifecycle import reporting

    reporting._shutdown_status_reporter(close=True)


@pytest.fixture(autouse=True)
def _reset_status_reporter_before_test():
    from flash.runner.lifecycle import reporting

    reporting._shutdown_status_reporter(close=True)
    reporting._open_status_reporter()


@pytest.fixture(autouse=True)
def _rebind_worker_submodule_attributes():
    """Repair the ``flash.engine.worker`` package object after a test re-imports it.

    Several tests read module-scope state captured at import (RUN_MODE, JOB_SPEC), so they drop the
    package from ``sys.modules`` and import it again to re-run that scope. Re-importing a PACKAGE
    does not rebind its submodule attributes: the submodules stay cached under their own names, so
    the `import` statements inside the re-executed parent are no-ops that never re-run the
    attribute assignment the import system does on first load. The replacement package object
    therefore has no ``.train``/``.io``/``.model`` attributes, and it is the object every later
    test sees.

    That breaks any later test patching a dotted string target such as
    ``flash.engine.worker.train.opd.orchestration.validation._resolve_structured_model_metadata``, because
    monkeypatch resolves a dotted target by walking getattr from the parent package. Alphabetical
    file order hides it today -- the re-importing file sorts after its victims -- so the suite is
    green by accident, and any reordering (``-n`` sharding, ``-p randomly``, a renamed file) turns
    it into an AttributeError attributed to the innocent test.

    Rebinding is the narrow repair: it fixes the attribute graph while LEAVING the freshly imported
    object in place, so the re-import those tests want still takes effect. Restoring the original
    object instead would undo their re-import. Reads only what is already cached -- no import is
    triggered, and it no-ops when nothing was re-imported.
    """
    import sys

    yield
    package = sys.modules.get("flash.engine.worker")
    if package is None:
        return
    prefix = "flash.engine.worker."
    for name, module in list(sys.modules.items()):
        if not name.startswith(prefix):
            continue
        child = name[len(prefix) :]
        if "." in child or module is None:
            continue
        if getattr(package, child, None) is not module:
            setattr(package, child, module)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # RunPod endpoint listing -> offline: the idle-endpoint sweep (deploy-time quota reclaim
    # and the startup/post-run orphan sweep) lists account endpoints. Default to "no endpoints"
    # so a sweep never reaches the real API; sweep tests monkeypatch it after this fixture.
    import flash.providers.runpod.client.api as runpod_api

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

    # scrub automatically forwarded secrets from the production key set so operator credentials
    # never enter fixtures. lambda/vast keys enable live providers, FREESOLO_API_KEY changes org
    # identity, and FLASH_STANDALONE/FLASH_HF_NAMESPACE change global mode. tests opt in after this
    # fixture; see tests/test_server_standalone.py.
    from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS

    for _key in {
        "FREESOLO_API_KEY",
        "LAMBDA_API_KEY",
        "VAST_API_KEY",
        "PARASAIL_API_KEY",
        "FLASH_PUBLIC_URL",
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
    import flash.providers.runpod.client.auth as rp_keys

    rp_keys.reset()

    # Provider singletons are @cache'd per name, and is_configured() now reads the env, so a test
    # that sets or clears a provider key would otherwise see the PREVIOUS test's answer. Clear
    # around every test: the cache exists to avoid re-importing, not to freeze configuration.
    from flash.providers.core import registry as _providers

    _providers._get_provider.cache_clear()

    # Pinned-revision geometry and size are memoized for the life of the PROCESS: a pinned commit is
    # immutable, so production reads it once and shares it. Tests stub the hub UNDER those memos
    # (huggingface_hub.HfApi / hf_hub_download), so a file that resolves a pin for real leaves the
    # true geometry cached under a key a later test then stubs differently -- and the later test
    # silently reads the earlier answer. Clear around every test: the memo exists to avoid a round
    # trip, not to freeze one test's stub into the next.
    import flash.cost.facts as _facts
    import flash.engine.plan.model_config_probe as _probe

    _probe._CONFIG_PROBE_MEMO.clear()
    _facts._PINNED_SIZE_MEMO.clear()

    # Always-on artifact GC: the control-plane lifespan sweeps ONCE on startup (when an operator
    # HF_TOKEN is set). Stub it to a no-op so offline TestClient startups never reach HF/serving;
    # tests/test_repo_cleanup.py restores the real function to exercise the genuine sweep.
    import flash.server.domain.ops.repo_cleanup as _rc

    monkeypatch.setattr(_rc, "run_scheduled_cleanup", lambda *a, **k: 0, raising=False)

    yield
    rp_keys.reset()
    _providers._get_provider.cache_clear()
    _probe._CONFIG_PROBE_MEMO.clear()
    _facts._PINNED_SIZE_MEMO.clear()


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
        import flash.engine.worker.train.entry.backend_common as _vc

        _vc._ADOPTS_ORPHANS = bool(previous)


@pytest.fixture(autouse=True)
def _fast_serving_readback(monkeypatch):
    """Zero the deploy read-back backoff so verification polls don't slow the suite."""
    import flash.serve.deployment.deploy as _deploy

    monkeypatch.setattr(_deploy, "READBACK_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(_deploy, "SMOKE_RETRY_FALLBACK_DELAY_SECONDS", 0.0)


@pytest.fixture
def stub_serving_registry(monkeypatch):
    """Patch GET /adapters to return the given records (deploy read-back verification)."""

    def _stub(*records: dict):
        import httpx

        class _RegistryResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"ok": True, "adapters": list(records)}

        monkeypatch.setattr(httpx, "get", lambda *a, **k: _RegistryResp())

    return _stub
