"""Repository-wide invariants for `.github/workflows/*.yml`.

CI configuration is the one part of the repo whose defects are invisible by construction: a
workflow that silently does nothing reports the same green check as one that did its job, and
nobody reads a passing run. Every invariant here was added after a real failure of exactly that
shape, so each test names the failure it prevents rather than asserting a style rule.

These parse the YAML rather than grepping it. A regex over workflow text cannot tell a real key
from the same characters inside a `run:` script or a comment, which is how a "the guard is
present" check passes while the guard is commented out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

UPSTREAM_REPOSITORY = "freesolo-co/flash"


def _workflow_paths() -> list[Path]:
    paths = sorted(WORKFLOW_DIR.glob("*.yml"))
    assert paths, f"no workflows found under {WORKFLOW_DIR}"
    return paths


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _jobs(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return document.get("jobs") or {}


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return job.get("steps") or []


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: p.name)
def test_every_action_reference_is_pinned_to_a_full_sha(path: Path):
    """The org sets `sha_pinning_required`, so a tag reference is rejected at "Set up job".

    A tag-pinned action is not merely a supply-chain risk here, it is an outright broken run: the
    job dies before its first real step with `steps=1`, and the red check says nothing about the
    change that triggered it. `main` still carries tag references, which is why every dependabot
    security PR cut from `main` fails at setup -- the failure is entirely about the stale workflow
    file it inherited.

    A version comment (`# v6.1.0`) is required alongside the sha so the pin stays legible and
    dependabot can bump it; the sha alone is unreadable. That half is checked against the raw text
    rather than the parsed document, because a YAML comment is discarded during parsing.
    """
    document = _load(path)
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith(("- uses:", "uses:")):
            continue
        if "@" not in stripped or "./.github/workflows/" in stripped:
            continue
        assert "#" in stripped, (
            f"{path.name}: {stripped!r} pins a sha with no version comment. Dependabot needs the "
            "comment to know what it is bumping, and a reviewer cannot read a bare sha."
        )

    for job_name, job in _jobs(document).items():
        # A reusable-workflow call (`jobs.<id>.uses`) is a local path, not a marketplace action.
        local_call = job.get("uses")
        if local_call is not None:
            assert local_call.startswith("./.github/workflows/"), (
                f"{path.name}:{job_name} calls a non-local reusable workflow {local_call!r}; "
                "a remote one would need a sha pin too"
            )
            continue
        for index, step in enumerate(_steps(job)):
            reference = step.get("uses")
            if reference is None:
                continue
            _, _, revision = reference.partition("@")
            unpinned = (
                f"{path.name}:{job_name} step {index} uses {reference!r}, which is not pinned to a "
                "full-length commit sha. The org requires sha pinning, so this fails the run at "
                '"Set up job" before any step executes.'
            )
            assert len(revision) == 40, unpinned
            assert all(c in "0123456789abcdef" for c in revision), unpinned


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: p.name)
def test_every_job_bounds_its_runtime(path: Path):
    """Without `timeout-minutes` a wedged job holds its runner for the six-hour default.

    That is a real cost on the org's larger runners and a real bill on `bake-kernel-cache`, which
    rents a GPU: there, the cap is the only thing bounding spend on a hung bake.
    """
    document = _load(path)
    for job_name, job in _jobs(document).items():
        if "uses" in job:
            # The callee declares its own cap; asserting here would demand it in two places and
            # let them drift.
            continue
        timeout = job.get("timeout-minutes")
        uncapped = (
            f"{path.name}:{job_name} declares no positive timeout-minutes, so a hang holds the "
            "runner for GitHub's 6-hour default"
        )
        assert isinstance(timeout, int), uncapped
        assert timeout > 0, uncapped


def test_the_tests_repo_dispatch_cannot_silently_no_op():
    """A notifier that skips when its credential is missing reports success while notifying nobody.

    This is the bug this test exists for, and it ran undetected for weeks. The step used to
    `exit 0` when `FREESOLO_GITHUB_PAT` was empty, on the theory that only a fork could reach that
    branch. But the secret has never existed on this repository, so *every* push to dev/main took
    the skip branch: the tests repo recorded zero `flash-merged` events against 100+
    `freesolo-merged` ones, and every one of those runs was green.

    The fix separates the two cases that shared one branch. A fork is excluded at the job level,
    which leaves "no PAT" meaning exactly one thing -- upstream misconfiguration -- so it can fail
    loudly. The invariant is therefore a conjunction, and asserting only half of it would pass on
    a workflow that is still fail-open.
    """
    document = _load(WORKFLOW_DIR / "notify-tests-repo.yml")
    job = _jobs(document)["notify"]

    condition = job.get("if")
    ungated = (
        "the notify job must be gated to the upstream repository, so that a missing PAT can be "
        "treated as an error rather than as the fork case"
    )
    assert condition is not None, ungated
    assert UPSTREAM_REPOSITORY in condition, ungated

    scripts = [step["run"] for step in _steps(job) if "run" in step]
    assert scripts, "the notify job must still run the dispatch"
    raw = "\n".join(scripts)
    # Strip shell comments before asserting on control flow. The first version of this test read
    # the raw script and failed on the words "exit 0" inside the comment EXPLAINING the old bug --
    # a substring assertion cannot tell code from prose, so it would equally have passed on a
    # workflow whose real `exit 0` had merely been commented out.
    code = "\n".join(line.split("#", 1)[0] for line in raw.splitlines())

    # `exit 0` on the empty-PAT branch is the precise defect; it is what made the absent secret
    # indistinguishable from a fork.
    assert "exit 0" not in code, (
        "the dispatch step must not exit 0 on a missing credential -- that is the fail-open branch "
        "that reported success while dispatching nothing"
    )
    assert "exit 1" in code, "a missing PAT must fail the job"
    assert "::error::" in code, "the failure must be annotated so it is visible on the run"
    # The dispatch itself must survive: a test pinning only the guard would pass on a workflow
    # whose curl had been deleted.
    still_dispatches = (
        "the step must still POST a flash-merged repository_dispatch to freesolo-co/tests"
    )
    assert "flash-merged" in code, still_dispatches
    assert "repos/freesolo-co/tests/dispatches" in code, still_dispatches


def test_workflows_that_touch_shared_resources_are_upstream_only():
    """A fork must not run the jobs that publish, push images, or spend GPU money.

    Beyond the wasted red X, two of these fail in ways worse than failing: `worker-image` requests
    an org-provisioned runner label, and a fork without that label leaves the job QUEUED rather
    than failing, so the PR never settles.
    """
    guarded = {
        "publish.yml": "publish-pypi",
        "publish-dev.yml": "publish-pypi-dev",
        "publish-image.yml": "publish-flash-image",
        "worker-image.yml": "build",
        "auto-rebake.yml": "gate",
        "notify-tests-repo.yml": "notify",
    }
    for filename, job_name in guarded.items():
        job = _jobs(_load(WORKFLOW_DIR / filename))[job_name]
        condition = job.get("if")
        ungated = (
            f"{filename}:{job_name} must be gated on github.repository == '{UPSTREAM_REPOSITORY}'"
        )
        assert condition is not None, ungated
        assert UPSTREAM_REPOSITORY in condition, ungated
