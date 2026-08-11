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

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

UPSTREAM_REPOSITORY = "freesolo-co/flash"


def _workflow_paths() -> list[Path]:
    # Both extensions: Actions runs `.yaml` exactly as it runs `.yml`, so globbing only `.yml`
    # would let a `.yaml` workflow run uncapped and unpinned while these contracts still reported
    # that every workflow complied. The scan has to cover what GitHub covers, not what the repo
    # happens to use today.
    paths = sorted(p for ext in ("*.yml", "*.yaml") for p in WORKFLOW_DIR.glob(ext))
    assert paths, f"no workflows found under {WORKFLOW_DIR}"
    return paths


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _jobs(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return document.get("jobs") or {}


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return job.get("steps") or []


def _assert_gated_upstream(condition: str | None, where: str) -> None:
    """Assert `condition` contains a POSITIVE equality against the upstream repository.

    Substring-matching the repository name is not enough, and this is not hypothetical: flipping
    `==` to `!=` (which inverts the guard, so the job would run ONLY on forks -- publishing,
    larger-runner and GPU-spending jobs exactly where they must never run) left every assertion in
    this file passing. The name is present in both spellings. Match the operator too.

    A compound condition is allowed (`auto-rebake` also requires the triggering run to have
    succeeded); the requirement is that one conjunct is the positive equality, in either argument
    order, under either quote style.
    """
    assert condition is not None, f"{where} has no `if:` guard at all"
    normalised = " ".join(condition.split()).replace('"', "'")
    accepted = (
        f"github.repository == '{UPSTREAM_REPOSITORY}'",
        f"'{UPSTREAM_REPOSITORY}' == github.repository",
    )
    assert any(form in normalised for form in accepted), (
        f"{where} must be gated on `github.repository == '{UPSTREAM_REPOSITORY}'`; its condition is "
        f"{condition!r}. Note a NEGATED comparison still contains the repository name while "
        "inverting the meaning of the guard."
    )


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
        # A version-SHAPED comment, not merely any comment: `# TODO` would satisfy "has a comment"
        # while telling a reviewer nothing about which release the sha is. Dependabot writes
        # `# v6.1.0`; the digits are the part that carries the information.
        #
        # Anchored at the START of the comment, not searched anywhere in it. A bare `re.search` for
        # "# then digits" is satisfied by an issue reference -- `# fixes #194` contains it -- which
        # recreates the vacuous pass this assertion exists to close. An action reference cannot
        # itself contain `#`, so the first one begins the comment.
        _, hash_marker, comment = stripped.partition("#")
        missing_version = (
            f"{path.name}: {stripped!r} pins a sha with no version comment. Dependabot needs the "
            "comment to know what it is bumping, and a reviewer cannot read a bare sha."
        )
        assert hash_marker, missing_version
        assert re.match(r"\s*v?\d+(\.\d+)*(\s|$)", comment), missing_version

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

    # The fork case must be handled HERE, at the job level, because that is what lets the step
    # below treat a missing PAT as the error it is.
    _assert_gated_upstream(job.get("if"), "notify-tests-repo.yml:notify")

    scripts = [step["run"] for step in _steps(job) if "run" in step]
    assert scripts, "the notify job must still run the dispatch"
    script = "\n".join(scripts)

    # RUN the script rather than grepping it. Searching for `exit 1` / `::error::` / the dispatch
    # URL as independent substrings proves only that those characters exist SOMEWHERE -- not that
    # they are on the missing-PAT path. Both weaknesses were demonstrated, not theorised: a
    # rewritten step that skipped the curl and returned success on an empty PAT, with the tokens
    # parked in an `if false` block, passed the substring version of this test. So did a real
    # `printf '#'; exit 0`, because stripping at the first `#` truncated the line before the
    # `exit 0` -- a shell comment cannot be found by splitting on a character that also occurs
    # inside quotes. Executing the script is what makes the two cases distinguishable.
    #
    # `curl` is replaced with a recorder on PATH, so nothing leaves the machine and the assertion
    # is about the OBSERVABLE effect: did a dispatch actually happen.
    #
    # The recorder's log is returned SEPARATELY from the process's stdout/stderr, and every
    # "it posted" assertion reads only the log. Blending them would undo the point of running the
    # script: a step that merely echoed the dispatch URL and the event type would satisfy the
    # blended text while never invoking curl, which is the same substring-matching weakness one
    # layer further out.
    def run(pat: str) -> tuple[int, str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = root / "curl-calls.txt"
            fake_curl = root / "bin" / "curl"
            fake_curl.parent.mkdir()
            fake_curl.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$CURL_CALLS"\nexit 0\n')
            fake_curl.chmod(0o755)
            completed = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                cwd=tmp,
                env={
                    "PATH": f"{fake_curl.parent}:{os.environ.get('PATH', '')}",
                    "CURL_CALLS": str(calls),
                    "PAT": pat,
                    "SHA": "0123456789abcdef0123456789abcdef01234567",
                    "REF": "dev",
                    "GITHUB_REPOSITORY": UPSTREAM_REPOSITORY,
                },
            )
            recorded = calls.read_text() if calls.exists() else ""
            return completed.returncode, recorded, completed.stdout + completed.stderr

    # 1. No credential must FAIL, and must not pretend to have dispatched.
    code, dispatched, logged = run("")
    assert code != 0, (
        "the dispatch step exited 0 with no credential -- that is the fail-open branch that "
        "reported success while dispatching nothing"
    )
    assert "::error::" in logged, "the failure must be annotated so it is visible on the run"
    assert dispatched == "", f"curl was invoked without a credential: {dispatched!r}"

    # 2. With a credential it must actually POST the dispatch, exactly once. These read the
    # recorder's log alone, so only a real curl invocation can satisfy them.
    code, dispatched, logged = run("fake-token")
    assert code == 0, f"the dispatch failed with a credential present: {logged}"
    assert dispatched.count("repos/freesolo-co/tests/dispatches") == 1, (
        f"expected exactly one dispatch to freesolo-co/tests, got: {dispatched!r}"
    )
    assert "flash-merged" in dispatched, (
        f"the dispatch must carry the flash-merged event type, got: {dispatched!r}"
    )


def _triggers(document: dict[str, Any]) -> dict[str, Any]:
    """Return the `on:` block.

    Bare `on` is a YAML 1.1 boolean, so `yaml.safe_load` yields the key `True`, not `"on"`. A
    workflow quoted as `"on":` yields the string. Looking up only one spelling silently returns
    nothing for the other, and a trigger contract that finds no triggers passes vacuously.
    """
    for key in (True, "on"):
        if key in document:
            return document[key] or {}
    raise AssertionError("workflow declares no `on:` block")


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: p.name)
def test_no_workflow_is_triggered_by_a_tag_push(path: Path):
    """Tags in this repo are published BY a workflow; none may be an input to one.

    `publish.yml` used to fire on `on: push: tags: - "flash-v*"` under the pre-1.0 scheme. It now
    fires on a push to `main` that bumps the version, and creates `v<version>` itself afterwards.
    A tag trigger reintroduced alongside that is a second route to PyPI which skips the
    version-bump check, the "already on PyPI" check and the tag-conflict preflight -- every gate
    the current design relies on.

    The failure that motivated this: pushing the abandoned `flash-v0.2.18` tag on 2026-08-09 ran
    the June workflow file stored at that tagged commit and failed in `uv publish` (run
    31330288101). Actions resolves the workflow from the pushed ref, so no edit to the file on
    `dev` could have stopped it; what this test buys is that the pattern cannot come back on any
    commit reachable from here.
    """
    push = _triggers(_load(path)).get("push")
    if not isinstance(push, dict):
        return
    assert "tags" not in push, (
        f"{path.name} declares an `on.push.tags` trigger ({push['tags']!r}). Tags are produced by "
        "publish.yml, not consumed: a tag-triggered publish bypasses the version-bump and "
        "already-on-PyPI gates."
    )


def test_main_source_guard_checks_provenance_not_just_the_branch_name():
    """A branch NAME check is spoofable by anyone with a fork the moment this repo is public.

    The guard used to compare `github.head_ref` to the literal string `dev` and nothing else. For a
    fork PR that value is the branch name *inside the fork*, so forking the repo, naming a branch
    `dev` and opening a PR straight into `main` printed "Source branch 'dev' is allowed" and went
    green -- under a check whose displayed name ("Source branch is dev") is what a reviewer reads
    in the checks list. The repository of the head ref is the part that cannot be renamed into
    compliance, so it is what must be compared.

    The script is EXECUTED rather than grepped, for the same reason as the dispatch test above:
    substring-matching `head.repo.full_name` proves the characters exist, not that they gate
    anything. The three outcomes are asserted separately, and the two rejection paths must be
    distinguishable in their messages -- "you targeted main from the wrong branch" and "fork heads
    may never target main" call for different actions from the person reading the log.
    """
    document = _load(WORKFLOW_DIR / "main-source-guard.yml")
    job = _jobs(document)["source-is-dev"]

    # The guard is only meaningful on PRs into `main`; a widened trigger would run it (and pass it)
    # where it means nothing.
    assert _triggers(document)["pull_request"]["branches"] == ["main"], (
        "main-source-guard must stay scoped to pull requests targeting main"
    )

    steps = [step for step in _steps(job) if "run" in step]
    assert len(steps) == 1, "the guard should stay a single step"
    step = steps[0]
    script = step["run"]

    # Read the head repository from the `pull_request` payload, not from `github.repository`
    # (which is the BASE repo on a fork PR and would compare a value to itself) and not from
    # `github.event.pull_request.head.label` (a display string a fork owner influences).
    env = step.get("env") or {}
    assert any(
        "github.event.pull_request.head.repo.full_name" in str(value) for value in env.values()
    ), (
        "the guard must read github.event.pull_request.head.repo.full_name; that is the only value "
        f"a fork cannot set. Its env is {env!r}"
    )

    def run(head_repo: str, head_ref: str) -> tuple[int, str]:
        completed = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            env={
                "PATH": os.environ.get("PATH", ""),
                "HEAD_REPO": head_repo,
                "HEAD_REF": head_ref,
                "UPSTREAM_REPO": UPSTREAM_REPOSITORY,
            },
        )
        return completed.returncode, completed.stdout + completed.stderr

    # 1. The one legitimate promotion.
    code, logged = run(UPSTREAM_REPOSITORY, "dev")
    assert code == 0, f"upstream dev -> main must be allowed, got exit {code}: {logged}"

    # 2. The spoof this test exists for: right branch name, wrong repository.
    code, logged = run("attacker/flash", "dev")
    assert code != 0, (
        "a fork branch named 'dev' was accepted -- the guard is still comparing names only"
    )
    assert "fork" in logged.lower(), (
        f"the fork rejection must say so, so it is not read as a branch-naming mistake: {logged}"
    )

    # 3. Wrong branch, upstream repository. Distinct message from case 2.
    code, logged = run(UPSTREAM_REPOSITORY, "feature/thing")
    assert code != 0, "a non-dev upstream branch was accepted"
    assert "fork" not in logged.lower(), (
        f"an upstream branch must not be reported as a fork problem: {logged}"
    )
    assert "dev" in logged, f"the branch rejection must name the required branch: {logged}"

    # 4. Fails closed when the field is absent (deleted fork, or a re-trigger where the payload
    # carries no head repo) rather than treating "" as "not a fork".
    code, logged = run("", "dev")
    assert code != 0, (
        f"an empty head repository must be rejected, not defaulted to upstream: {logged}"
    )


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
        # Reached two ways, and the gate on `auto-rebake` only covers one of them: this workflow is
        # also directly dispatchable, so it needs its own guard. It is the most expensive job in the
        # repo to leave ungated -- it rents a GPU.
        "bake-kernel-cache.yml": "bake",
    }
    for filename, job_name in guarded.items():
        job = _jobs(_load(WORKFLOW_DIR / filename))[job_name]
        _assert_gated_upstream(job.get("if"), f"{filename}:{job_name}")
