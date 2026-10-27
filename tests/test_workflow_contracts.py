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

import json
import os
import re
import shlex
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

UPSTREAM_REPOSITORY = "freesolo-co/flash"


def test_deepinfer_rollout_cannot_report_success_before_warm_replacement():
    workflow = yaml.safe_load((WORKFLOW_DIR / "deploy-deepinfer.yml").read_text())
    assert workflow["concurrency"]["cancel-in-progress"] is False
    job = workflow["jobs"]["deploy"]
    assert job["timeout-minutes"] >= 210
    assert "github.run_attempt" in job["env"]["DEEPINFER_ROLLOUT_ID"]
    steps = [step for step in job["steps"] if "run" in step]
    production = next(step for step in steps if "app.fleet_deploy" in step["run"])
    assert production["if"] == "github.ref == 'refs/heads/main'"
    assert "--stage-migration" not in production["run"]
    # A monolithic redeploy would terminate two-hour streams after one hour.
    native = [step for step in steps if "modal deploy" in step["run"]]
    assert native
    assert all(step["if"] == "github.ref == 'refs/heads/dev'" for step in native)


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
    """return the `on:` block.

    bare `on` is a YAML 1.1 boolean, so `yaml.safe_load` yields the key `True`, not `"on"`. a
    workflow quoted as `"on":` yields the string. looking up only one spelling silently returns
    nothing for the other, and a trigger contract that finds no triggers passes vacuously.
    """
    for key in (True, "on"):
        if key in document:
            return document[key] or {}
    raise AssertionError("workflow declares no `on:` block")


@pytest.mark.parametrize("path", _workflow_paths(), ids=lambda p: p.name)
def test_no_workflow_is_triggered_by_a_tag_push(path: Path):
    """tags in this repo are published BY a workflow; none may be an input to one.

    `publish.yml` fires on a push to `main` that bumps the version, and creates `v<version>` itself
    afterwards. a tag trigger alongside that is a second route to PyPI which skips the version-bump
    check, the "already on PyPI" check and the tag-conflict preflight, every gate the design relies
    on.

    the failure this pins: pushing an abandoned `flash-v*` tag runs whatever publish file is stored
    at that tagged commit, because Actions resolves the workflow from the pushed ref. no edit on
    `dev` can stop that for a tag that already exists; what this test buys is that the pattern
    cannot come back on any commit reachable from here.

    a literal `tags:` key is not the only way in: per actions semantics, `on.push` with NEITHER
    `branches` NOR `tags` runs on every pushed ref, tags included. `on: push` (the bare shorthand)
    loads as `push is None`, and `push: {paths: [...]}` with no `branches` key loads as a dict with
    no `tags` key either -- both used to return early or pass the old `"tags" not in push` check
    while still triggering on a tag push, so deleting the `branches:` filter from `publish.yml`
    would have restored the tag-triggered release path this test claims to prevent, and stayed
    green. the fix is to require a POSITIVE branch restriction (`branches` or `branches-ignore`)
    on every push trigger rather than merely forbidding the `tags` key. `tags-ignore` covering every
    tag would also close the hole, but is not accepted here to keep the rule a single simple check
    rather than one that has to reason about ignore-glob coverage.
    """
    push = _triggers(_load(path)).get("push")
    if push is None:
        return
    assert isinstance(push, dict), (
        f"{path.name} declares `on: push` (the bare shorthand, with no `branches` filter at all). "
        "that runs on every pushed ref, tags included, which is exactly the tag-triggered publish "
        "path this contract exists to prevent."
    )
    assert "branches" in push or "branches-ignore" in push, (
        f"{path.name} declares an `on.push` trigger with no `branches` or `branches-ignore` key "
        f"({push!r}). without a positive branch restriction the trigger also fires on tag pushes, "
        "which bypasses the version-bump and already-on-PyPI gates that publish.yml relies on."
    )
    assert "tags" not in push, (
        f"{path.name} declares an `on.push.tags` trigger ({push['tags']!r}). Tags are produced by "
        "publish.yml, not consumed: a tag-triggered publish bypasses the version-bump and "
        "already-on-PyPI gates."
    )


def test_main_source_guard_checks_provenance_not_just_the_branch_name():
    """a branch NAME check is spoofable by anyone with a fork the moment this repo is public.

    for a fork PR, `github.head_ref` is the branch name *inside the fork*, so a guard comparing
    only that name lets someone fork the repo, name a branch `dev`, open a PR straight into `main`,
    and go green, under a check whose displayed name ("Source branch is dev") is what a reviewer
    reads in the checks list. the repository of the head ref is the part that cannot be renamed
    into compliance, so it is what must be compared.

    the script is EXECUTED rather than grepped, for the same reason as the dispatch test above:
    substring-matching `head.repo.full_name` proves the characters exist, not that they gate
    anything. each outcome is asserted separately, and the two rejection paths must be
    distinguishable in their messages: "you targeted main from the wrong branch" and "fork heads
    may never target main" call for different actions from the person reading the log.
    """
    document = _load(WORKFLOW_DIR / "main-source-guard.yml")
    job = _jobs(document)["source-is-dev"]

    # the guard is only meaningful on PRs into `main`; a widened trigger would run it (and pass it)
    # where it means nothing.
    assert _triggers(document)["pull_request"]["branches"] == ["main"], (
        "main-source-guard must stay scoped to pull requests targeting main"
    )

    steps = [step for step in _steps(job) if "run" in step]
    assert len(steps) == 1, "the guard should stay a single step"
    step = steps[0]
    script = step["run"]

    # read the head repository from the `pull_request` payload, not from `github.repository`
    # (which is the BASE repo on a fork PR and would compare a value to itself) and not from
    # `github.event.pull_request.head.label` (a display string a fork owner influences).
    #
    # this checks the exact env MAPPING, not merely that the right-hand expression appears
    # somewhere in the step -- and it runs before the script is ever executed. the run() helper
    # below feeds HEAD_REPO/HEAD_REF/UPSTREAM_REPO to the script directly, so every assertion past
    # this point exercises the script's logic in isolation from the workflow file. that gap used to
    # be the whole test: an `any(... in str(value) ...)` over env.values() proves the expression is
    # present SOMEWHERE, not that it is wired to the name the script reads. `HEAD_REPO` rewired to
    # `github.repository` -- with the correct expression parked under an unused env key, or with
    # both HEAD_REPO and UPSTREAM_REPO bound to the head-repo expression so the script compares a
    # value to itself -- passed every assertion below while the live guard would accept a fork
    # branch named `dev`. asserting the mapping itself is what closes that.
    env = step.get("env") or {}
    assert env.get("HEAD_REPO") == "${{ github.event.pull_request.head.repo.full_name }}", (
        "HEAD_REPO must be wired to github.event.pull_request.head.repo.full_name -- that is the "
        f"only value a fork owner cannot set. its env is {env!r}"
    )
    assert env.get("UPSTREAM_REPO") == "${{ github.repository }}", (
        f"UPSTREAM_REPO must be wired to github.repository (the base repo). its env is {env!r}"
    )
    assert env.get("UPSTREAM_REPO") != env.get("HEAD_REPO"), (
        "UPSTREAM_REPO and HEAD_REPO must not resolve to the same expression -- that would compare "
        f"the base repo to itself and accept any head. its env is {env!r}"
    )
    assert env.get("HEAD_REF") == "${{ github.head_ref }}", (
        f"HEAD_REF must be wired to github.head_ref. its env is {env!r}"
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

    # 1. the one legitimate promotion.
    code, logged = run(UPSTREAM_REPOSITORY, "dev")
    assert code == 0, f"upstream dev -> main must be allowed, got exit {code}: {logged}"

    # 2. the spoof this test exists for: right branch name, wrong repository.
    code, logged = run("attacker/flash", "dev")
    assert code != 0, (
        "a fork branch named 'dev' was accepted -- the guard is still comparing names only"
    )
    assert "fork" in logged.lower(), (
        f"the fork rejection must say so, so it is not read as a branch-naming mistake: {logged}"
    )

    # 3. wrong branch, upstream repository. distinct message from case 2.
    code, logged = run(UPSTREAM_REPOSITORY, "feature/thing")
    assert code != 0, "a non-dev upstream branch was accepted"
    assert "fork" not in logged.lower(), (
        f"an upstream branch must not be reported as a fork problem: {logged}"
    )
    assert "dev" in logged, f"the branch rejection must name the required branch: {logged}"

    # 4. fails closed when the field is absent (deleted fork, or a re-trigger where the payload
    # carries no head repo) rather than treating "" as "not a fork".
    code, logged = run("", "dev")
    assert code != 0, (
        f"an empty head repository must be rejected, not defaulted to upstream: {logged}"
    )


def test_main_source_guard_cannot_be_skipped_into_a_pass():
    """`Source branch is dev` is a REQUIRED check, and a skipped required check counts as SUCCESS.

    That makes a job-level `if:` on this job a supersede primitive rather than an exemption. Let
    the guard run and fail on a human head, then fire an event whose payload takes the `if:` false
    branch: the run is SKIPPED, the skip supersedes the failure on the same SHA, and the required
    check reads green with the human commits still in place. Close-and-reopen is enough to do it --
    this workflow declares no `types:`, so `reopened` is in its trigger set.

    Every event-derived signal has this shape, so no condition on the job can be safe:
    `pull_request.user.login` and `head_ref` are immutable for the PR's lifetime (they describe who
    OPENED it, not what it now CONTAINS), and `github.actor` is the event trigger, which differs
    between two runs of the same commits. The invariant is structural -- no `if:` at all.
    """
    job = _jobs(_load(WORKFLOW_DIR / "main-source-guard.yml"))["source-is-dev"]
    assert "if" not in job, (
        "source-is-dev must carry no job-level `if:`. it is a required check, so a false condition "
        f"yields a SKIPPED run that counts as success and can supersede a real failure. got: {job['if']!r}"
    )


def test_main_source_guard_has_no_dependabot_carve_out():
    """No exemption may be built from commit metadata, because none of it proves bot authorship.

    An earlier revision of this guard carried a dependabot carve-out on the theory that security
    updates ignore `target-branch: dev` and open against `main`. That premise was wrong here: all
    11 dependabot PRs in this repo's history target `dev`, including under alert-driven security
    updates (which are enabled and unpaused), and this job -- which only triggers on PRs into
    `main` -- has never once run on a dependabot PR. The carve-out fixed nothing.

    It could not have been written safely either. GitHub signs whatever a caller hands it, so the
    signed payload of a real dependabot push carries no bot-specific fact:

        author    dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>
        committer GitHub <noreply@github.com>

    The author line is a caller-supplied header, and the committer is the generic identity behind
    every web-editor and contents-api commit. A write-capable maintainer calling the contents API
    with dependabot's noreply address in the `author` object reproduces that tuple exactly. Three
    successive versions of the carve-out (author+verified, committer+verified, author+committer+
    verified) were each defeated by that, so this test pins its ABSENCE: if dependabot is ever
    pointed at `main`, exempt it outside the commit -- a separate workflow keyed on the API's
    PR-level bot identity, or a ruleset bypass actor -- not by re-reading forgeable fields here.
    """
    job = _jobs(_load(WORKFLOW_DIR / "main-source-guard.yml"))["source-is-dev"]
    step = next(s for s in _steps(job) if "run" in s)
    script = step["run"]
    env = step.get("env") or {}

    # the commit lookup is the shape of every defeated attempt: resolve the head sha, read identity
    # fields off it, exit 0 on a match. no token, no sha, no lookup.
    for token in ("GH_TOKEN", "GITHUB_TOKEN"):
        assert token not in env, (
            f"the guard needs no API token; {token} implies a commit lookup was reintroduced. "
            f"env: {env!r}"
        )
    assert "HEAD_SHA" not in env, (
        f"the guard must not resolve the head commit -- its metadata is forgeable. env: {env!r}"
    )
    for forgeable in ("author", "committer", "verification", "verified"):
        assert forgeable not in script, (
            f"the guard script references {forgeable!r}, which reads commit metadata a contents-api "
            "caller controls. bot provenance cannot be established from the commit."
        )
    assert "dependabot" not in script, (
        "the guard script names dependabot, so a carve-out was reintroduced. dependabot targets "
        "`dev` here and this job never runs on its PRs; the exemption is unnecessary and unsafe."
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
        "dev-kernel-cache.yml": "detect",
        "notify-tests-repo.yml": "notify",
        # Reached two ways, and the gate on `auto-rebake` only covers one of them: this workflow is
        # also directly dispatchable, so it needs its own guard. It is the most expensive job in the
        # repo to leave ungated -- it rents a GPU.
        "bake-kernel-cache.yml": "bake",
    }
    for filename, job_name in guarded.items():
        job = _jobs(_load(WORKFLOW_DIR / filename))[job_name]
        _assert_gated_upstream(job.get("if"), f"{filename}:{job_name}")


def test_worker_image_can_build_an_exact_reusable_source():
    document = _load(WORKFLOW_DIR / "worker-image.yml")
    call = _triggers(document)["workflow_call"]
    assert call["inputs"]["tag"]["required"] is True
    assert call["inputs"]["source_ref"]["required"] is True
    assert set(call["outputs"]) == {
        "base_ref",
        "fp_cache",
        "fp_base",
        "base_revision",
    }

    job = _jobs(document)["build"]
    checkout = next(
        step for step in _steps(job) if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert "inputs.source_ref" in checkout["with"]["ref"]
    source = next(step for step in _steps(job) if step.get("id") == "source")
    assert "git rev-parse HEAD" in source["run"]
    image = next(step for step in _steps(job) if step.get("id") == "image")
    assert "steps.source.outputs.tag" in image["with"]["tags"]
    assert "steps.source.outputs.revision" in image["with"]["labels"]
    assert "steps.image.outputs.digest" in job["outputs"]["base_ref"]


def test_dev_cache_pushes_only_detect_and_off_peak_runs_build():
    document = _load(WORKFLOW_DIR / "dev-kernel-cache.yml")
    triggers = _triggers(document)
    assert triggers["push"]["branches"] == ["dev"]
    assert triggers["schedule"] == [{"cron": "0 9 * * *"}]
    assert "workflow_dispatch" in triggers

    jobs = _jobs(document)
    dispatch = jobs["dispatch-dev"]
    dispatch_script = next(step for step in _steps(dispatch) if "run" in step)["run"]
    _assert_gated_upstream(dispatch.get("if"), "dev-kernel-cache.yml:dispatch-dev")
    assert "github.event_name == 'schedule'" in dispatch["if"]
    assert "gh workflow run dev-kernel-cache.yml" in dispatch_script
    assert '--repo "$GITHUB_REPOSITORY"' in dispatch_script
    assert "--ref dev" in dispatch_script
    assert "github.event_name != 'schedule'" in jobs["detect"]["if"]
    detect_checkout = next(
        step
        for step in _steps(jobs["detect"])
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert detect_checkout["with"]["ref"] == "${{ github.sha }}"

    build = jobs["build-base"]
    bake = jobs["bake"]
    assert "github.event_name == 'workflow_dispatch'" in build["if"]
    assert "needs.detect.outputs.needs_bake == 'true'" in build["if"]
    assert build["uses"] == "./.github/workflows/worker-image.yml"
    assert "needs.detect.outputs.revision" in build["with"]["source_ref"]
    assert "github.event_name == 'workflow_dispatch'" in bake["if"]
    assert bake["uses"] == "./.github/workflows/bake-kernel-cache.yml"
    assert "cu128-dev-" in bake["with"]["target_tag_prefix"]
    assert triggers["workflow_dispatch"]["inputs"]["deadline_minutes"]["default"] == 90
    assert "fromJSON(format(" in bake["with"]["deadline_minutes"]
    assert "inputs.deadline_minutes" in bake["with"]["deadline_minutes"]
    assert "||" not in bake["with"]["deadline_minutes"]  # zero must fail validation, not become 90
    bake_workflow = _load(WORKFLOW_DIR / "bake-kernel-cache.yml")
    assert "deadline_minutes" in _triggers(bake_workflow)["workflow_call"]["inputs"]
    warm = next(
        step
        for step in _steps(_jobs(bake_workflow)["bake"])
        if str(step.get("name", "")).startswith("Warm kernels")
    )
    assert warm["env"]["BAKE_DEADLINE_MINUTES"] == "${{ inputs.deadline_minutes }}"
    assert '--deadline-min "$BAKE_DEADLINE_MINUTES"' in warm["run"]

    publish = jobs["publish-ready"]
    script = next(step for step in _steps(publish) if step.get("name") == "Publish ready aliases")[
        "run"
    ]
    assert "cu128-dev-$REVISION-$sm" in script
    assert "cu128-dev-ready-$sm" in script
    assert script.index("candidate validation failed") < script.index("crane copy")
    rerun = next(
        step
        for step in _steps(publish)
        if step.get("name") == "Re-run production readiness for dev"
    )["run"]
    assert "production-kernel-cache-ready.yml" in rerun
    assert '--repo "$GITHUB_REPOSITORY"' in rerun
    assert "--ref dev" in rerun

    classify = next(step for step in _steps(jobs["detect"]) if step.get("id") == "classify")["run"]
    assert '[ "$prod_pc" != "$FP_CACHE" ] && [ "$candidate_pc" != "$FP_CACHE" ]' in classify
    assert 'candidate_rev" != "$REVISION' not in classify


def test_main_pr_check_accepts_each_fingerprint_compatible_sm_cache():
    document = _load(WORKFLOW_DIR / "production-kernel-cache-ready.yml")
    triggers = _triggers(document)
    assert triggers["pull_request"]["branches"] == ["main"]
    assert "workflow_dispatch" in triggers
    job = _jobs(document)["ready"]
    assert job["name"] == "production kernel cache ready"

    checkout = next(
        step for step in _steps(job) if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha || github.sha }}"
    verify = next(
        step for step in _steps(job) if step.get("name") == "Verify every sm cache is ready"
    )
    script = verify["run"]
    assert "--print-baked-arches" in script
    assert "cu128-dev-ready-$sm" in script
    assert 'candidate_pc" = "$FP_CACHE' in script
    assert 'candidate_rev" = "$HEAD_SHA' not in script
    assert "exit 1" in script


def test_production_relayer_promotes_matching_dev_cache_before_gpu_fallback():
    document = _load(WORKFLOW_DIR / "auto-rebake.yml")
    jobs = _jobs(document)
    classify = next(step for step in _steps(jobs["gate"]) if step.get("id") == "classify")["run"]
    assert "cu128-dev-ready-$sm" in classify
    assert "crane digest" in classify
    assert 'crane config "$candidate_ref"' in classify
    assert 'candidate_pc" = "$FP_CACHE' in classify
    assert '--arg source "$candidate_ref"' in classify
    assert "gpu re-warm (candidate missing)" in classify
    assert "relayer_matrix=" in classify

    relayer = jobs["relayer"]
    assert "relayer_matrix" in relayer["strategy"]["matrix"]["include"]
    resolve = next(step for step in _steps(relayer) if step.get("id") == "old")["run"]
    assert 'img="${{ matrix.source }}"' in resolve
    assert '"$img" == *@sha256:*' in resolve


def test_bake_workflow_can_publish_versioned_candidate_tags():
    document = _load(WORKFLOW_DIR / "bake-kernel-cache.yml")
    call_inputs = _triggers(document)["workflow_call"]["inputs"]
    assert call_inputs["target_tag_prefix"]["default"] == "cu128"
    image = next(
        step
        for step in _steps(_jobs(document)["bake"])
        if "docker/build-push-action@" in step.get("uses", "")
    )
    assert "inputs.target_tag_prefix" in image["with"]["tags"]


def _deploy_steps() -> list[dict[str, Any]]:
    return _steps(_jobs(_load(WORKFLOW_DIR / "deploy-modal.yml"))["deploy"])


def _step_index(name_fragment: str) -> int:
    steps = _deploy_steps()
    for index, step in enumerate(steps):
        if name_fragment in (step.get("name") or ""):
            return index
    raise AssertionError(f"no deploy step named like {name_fragment!r}")


def test_promotion_is_gated_on_a_real_stream_after_readiness():
    """A healthy `/healthz` proves a ROUTER booted, not that the release can serve.

    The readiness poll only reads back the identity the deploy step just injected, so a release
    whose GPU engines never start, whose streaming path is broken, or whose accounting never
    settles passes it unchanged. The canary has to run AFTER readiness (it needs the new router
    live) and BEFORE the job can finish, or a broken release is promoted with a green check.
    """
    canary = _deploy_steps()[_step_index("real streaming canary")]
    assert "flash.serving.promotion.gate" in canary["run"]
    assert _step_index("real streaming canary") > _step_index("serving readiness")


def test_the_job_cap_leaves_room_for_a_rollback_after_every_wait_is_exhausted():
    """The job cap has to be derived from the waits, not chosen and left to rot.

    `timeout-minutes` kills the job wherever it is, and the worst case ends INSIDE the rollback:
    both readiness polls burn their full deadlines, the gate burns its canary and accounting
    budgets, and only then does the restore start its own deploy and 300s verification. A cap that
    fits the forward path but not the restore converts a recoverable failed promotion into a broken
    release left live, which is the one outcome this whole step exists to prevent.

    Summing the declared budgets rather than restating a number keeps the cap honest: lengthening
    any poll, or the gate's own deadlines, moves this bound automatically.
    """
    job = _load(WORKFLOW_DIR / "deploy-modal.yml")["jobs"]["deploy"]
    polls = sum(
        int(seconds)
        for step in _steps(job)
        for seconds in re.findall(r"SECONDS \+ (\d+)", step.get("run") or "")
    )
    assert polls > 0, "no bounded polls found -- this test is measuring nothing"

    from flash.serving.promotion import gate as gate_module

    gate_budget = (
        gate_module.CANARY_TIMEOUT_SECONDS
        + gate_module._DEFAULT_ACCOUNTING_DEADLINE_SECONDS
        + gate_module._HEALTH_ROLLOUT_DEADLINE_SECONDS
        + 15  # rollback schema probe
    )
    # two `modal deploy` invocations (forward and restore), image build included, plus checkout and
    # dependency installation. deliberately generous: the assertion should fail while there is still
    # headroom to fix, not at the moment a real run gets killed.
    deploy_allowance = 15 * 60

    required_minutes = (polls + gate_budget + deploy_allowance) / 60
    assert job["timeout-minutes"] >= required_minutes, (
        f"the deploy job caps at {job['timeout-minutes']}m, but its own declared waits "
        f"({polls}s of polls + {gate_budget:.0f}s of gate budget) plus a "
        f"{deploy_allowance // 60}m deploy allowance need {required_minutes:.1f}m. A cap below that "
        "can kill the job mid-rollback and leave a broken release serving."
    )


def test_the_promotion_canary_receives_its_credentials_only_through_the_environment():
    """A credential passed as an argument is readable in the process table and the step echo.

    `run:` lines are echoed into the public build log, so a key interpolated into argv leaks on
    every run, including successful ones. The gate reads them from `env:` instead.
    """
    canary = _deploy_steps()[_step_index("real streaming canary")]
    env = canary.get("env") or {}
    assert "FREESOLO_INTERNAL_KEY" in env
    assert "SUPABASE_SERVICE_ROLE_KEY" in env
    for secret in ("FREESOLO_INTERNAL_KEY", "SUPABASE_SERVICE_ROLE_KEY", "secrets."):
        assert secret not in canary["run"]


def test_the_previous_release_is_captured_before_the_deploy_overwrites_it():
    """`modal deploy` replaces the app in place, so the predecessor is unrecoverable afterwards.

    The gate step reads the live app's sha for its own diff bound; publishing it as an output is
    what makes rollback possible at all. Reading it only on the non-dispatch path would leave a
    manually dispatched deploy one-way.
    """
    gate = _deploy_steps()[_step_index("Decide whether to deploy")]
    assert 'previous_sha=$last_deployed_sha" >> "$GITHUB_OUTPUT' in gate["run"]
    assert _step_index("Decide whether to deploy") < _step_index("Deploy serving")


PREVIOUS = "1111111111111111111111111111111111111111"
CURRENT = "2222222222222222222222222222222222222222"


@pytest.mark.parametrize(
    ("event", "attempt", "expected"),
    [
        ("push", "1", "false"),
        ("push", "2", "true"),
        ("workflow_dispatch", "1", "true"),
    ],
)
def test_explicit_reruns_reverify_even_when_the_current_sha_is_already_live(
    event, attempt, expected
):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").write_text('#!/bin/sh\nprintf "%s\\n" "$CURRENT_SHA"\n')
        (bin_dir / "git").write_text("#!/bin/sh\nexit 0\n")
        for stub in bin_dir.iterdir():
            stub.chmod(0o755)
        output = root / "outputs"
        gate = _deploy_steps()[_step_index("Decide whether to deploy serving")]
        completed = subprocess.run(
            ["bash", "-e", "-c", gate["run"]],
            capture_output=True,
            text=True,
            cwd=tmp,
            env={
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                "EVENT": event,
                "GITHUB_RUN_ATTEMPT": attempt,
                "CURRENT_SHA": CURRENT,
                "GITHUB_OUTPUT": str(output),
            },
        )
        assert completed.returncode == 0, completed.stderr
        assert f"deploy={expected}" in output.read_text().splitlines()


def _run_rollback(
    *,
    previous_sha: str,
    is_ancestor: bool,
    health: str,
    entrypoint: str | None = "flash/serving/app/modal_app.py",
    schema_status: str = "200",
    has_schema_contract: bool = True,
) -> tuple[int, str, str]:
    """Execute the rollback script with `modal`, `curl`, `git`, and `sleep` stubbed on PATH.

    RUN it rather than grep it, for the same reason the dispatch and fork guards above are run:
    a substring is present in a step that never reaches the branch containing it. Every assertion
    below reads an OBSERVABLE effect -- did `modal deploy` get invoked, with which identity, and
    did the step exit nonzero -- so a rewritten step that echoed the right strings without
    redeploying would be red here and green under a grep.

    `modal`'s recorded invocation is returned separately from the step's stdout/stderr; blending
    them would reintroduce the same weakness one layer out, since the script also echoes the sha
    it intends to restore.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        # the working tree starts at the new layout; checkout installs the predecessor's layout.
        app = root / "flash/serving/app/modal_app.py"
        app.parent.mkdir(parents=True, exist_ok=True)
        app.touch()
        columns_path = root / (
            "flash/serving/src/persisted_columns.py"
            if entrypoint == "flash/serving/modal_app.py"
            else "flash/serving/src/store/persisted_columns.py"
        )
        if has_schema_contract:
            columns_path.parent.mkdir(parents=True, exist_ok=True)
            columns_path.write_text('PERSISTED_COLUMNS = "adapter_id,base_model"\n')
        calls = root / "modal-calls.txt"
        (bin_dir / "modal").write_text(
            '#!/bin/sh\n[ -f "$2" ] || exit 2\nprintf "%s sha=%s id=%s\\n" "$*" "$FREESOLO_DEPLOYMENT_SHA" '
            '"$FREESOLO_DEPLOYMENT_ID" >> "$MODAL_CALLS"\nexit 0\n'
        )
        # the health body is served from a file; `--output` is where the script wants it.
        (bin_dir / "curl").write_text(
            '#!/bin/sh\ncase "$*" in\n'
            '  *hosted_lora_adapters*) printf "%s" "$SCHEMA_STATUS"; exit 0;;\n'
            'esac\nfor arg in "$@"; do\n'
            '  if [ "$prev" = "--output" ]; then out="$arg"; fi\n  prev="$arg"\ndone\n'
            'printf "%s" "$HEALTH_BODY" > "$out"\nexit 0\n'
        )
        # ancestry comes from the fixture, while checkout changes the files visible to rollback.
        (bin_dir / "git").write_text(
            '#!/bin/sh\nif [ "$1" = "merge-base" ]; then exit "$ANCESTOR_EXIT"; fi\n'
            'if [ "$1" = "checkout" ]; then\n'
            "  rm -f flash/serving/app/modal_app.py flash/serving/modal_app.py\n"
            '  if [ -n "$RESTORED_ENTRYPOINT" ]; then\n'
            '    mkdir -p "$(dirname "$RESTORED_ENTRYPOINT")"\n'
            '    touch "$RESTORED_ENTRYPOINT"\n'
            "  fi\nfi\nexit 0\n"
        )
        for stub in bin_dir.iterdir():
            stub.chmod(0o755)

        # `sleep` is shadowed by a shell FUNCTION, not a PATH stub. The verification loop is bounded
        # by `$SECONDS`, which is wall-clock, so a stub that merely returns fast spins for the full
        # real 300s -- the deadline advances whether anything sleeps or not. A function runs in the
        # script's own shell, where `SECONDS` is assignable, so the loop's structure is preserved
        # (same number of naps, same deadline arithmetic) while virtual time moves in place of real
        # time.
        script = (
            "sleep() { SECONDS=$((SECONDS + 60)); }\n"
            + (_deploy_steps()[_step_index("Restore the previous release")]["run"])
        )
        completed = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            cwd=tmp,
            env={
                "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
                "MODAL_CALLS": str(calls),
                "RESTORED_ENTRYPOINT": entrypoint or "",
                "ANCESTOR_EXIT": "0" if is_ancestor else "1",
                "HEALTH_BODY": health,
                "SCHEMA_STATUS": schema_status,
                "SUPABASE_URL": "https://supabase.example",
                "SUPABASE_SERVICE_ROLE_KEY": "sb_secret_test",
                "PREVIOUS_SHA": previous_sha,
                "CURRENT_SHA": CURRENT,
                "ROLLBACK_DEPLOYMENT_ID": "9-1-rollback",
            },
        )
        recorded = calls.read_text() if calls.exists() else ""
        return completed.returncode, recorded, completed.stdout + completed.stderr


def _health(sha: str, deployment_id: str, ok: bool = True) -> str:
    return json.dumps({"ok": ok, "deployment_sha": sha, "deployment_id": deployment_id})


def test_the_rollback_step_fires_on_failure_with_a_distinct_attempt_id():
    """Wiring the script cannot prove: `if:` and `env:` are read by Actions, not by the shell.

    The attempt id must differ from the forward deploy's, or the readiness identity already
    published by the failed release would satisfy the restore check without anything being
    restored.
    """
    rollback = _deploy_steps()[_step_index("Restore the previous release")]
    assert rollback["if"].startswith("failure()")
    env = rollback.get("env") or {}
    assert env.get("PREVIOUS_SHA") == "${{ steps.gate.outputs.previous_sha }}"
    assert env.get("ROLLBACK_DEPLOYMENT_ID", "").endswith("-rollback")
    assert env["ROLLBACK_DEPLOYMENT_ID"] != "${{ github.run_id }}-${{ github.run_attempt }}"


def test_a_failed_promotion_restores_the_previous_release_and_still_fails_the_job():
    """The redeploy must carry the PREVIOUS sha, and a restore must never turn the run green.

    A green run would mark an unpromotable commit as deployed: the next push diffs against a sha
    that never passed its gate and skips the deploy entirely, so the failure compounds silently.
    """
    code, deployed, logged = _run_rollback(
        previous_sha=PREVIOUS,
        is_ancestor=True,
        health=_health(PREVIOUS, "9-1-rollback"),
    )

    assert f"sha={PREVIOUS}" in deployed, f"the redeploy did not carry the restored sha: {deployed}"
    assert "id=9-1-rollback" in deployed
    assert "flash/serving/app/modal_app.py" in deployed
    assert code != 0, "a completed rollback exited 0 -- this commit would be recorded as deployed"
    assert "is NOT deployed" in logged


@pytest.mark.parametrize(
    ("health", "why"),
    [
        (_health(CURRENT, "9-1"), "the broken release is still live"),
        (_health(CURRENT, "9-1-rollback"), "stale sha under the rollback's own attempt id"),
        (_health(PREVIOUS, "9-1"), "restored sha under the FORWARD deploy's attempt id"),
        (_health(PREVIOUS, "9-1-rollback", ok=False), "right identity, but not serving"),
    ],
)
def test_a_rollback_that_never_reports_the_restored_release_is_not_a_rollback(health, why):
    """`modal deploy` exiting zero says the app was REPLACED, not that it came up serving.

    This is the case a substring test cannot see: the identity comparison is present in the script
    either way, so what matters is that a live app failing it is rejected rather than read as a
    completed restore.

    Each field is failed ALONE. A single fixture with everything wrong lets any one surviving check
    mask the others -- disabling the sha comparison outright left such a test green, because the
    attempt id was stale in the same body and failed first. Three of these four cases exist only to
    make each comparison independently load-bearing.
    """
    code, deployed, logged = _run_rollback(previous_sha=PREVIOUS, is_ancestor=True, health=health)

    assert f"sha={PREVIOUS}" in deployed, "the restore was never attempted"
    assert code != 0, f"a rollback was accepted with {why}"
    assert "never reported the restored release" in logged
    assert "intervene manually" in logged


@pytest.mark.parametrize(
    ("previous_sha", "is_ancestor", "expected"),
    [
        ("", True, "NO previous release was recorded"),
        (PREVIOUS, False, "is not an ancestor"),
    ],
)
def test_rollback_refuses_a_previous_release_it_cannot_verify(previous_sha, is_ancestor, expected):
    """An empty or non-ancestor sha must stop WITHOUT deploying, not redeploy an unknown tree.

    Asserting the guard's text alone would pass on a script that printed the refusal and redeployed
    anyway. The load-bearing assertion is that `modal` was never invoked.
    """
    code, deployed, logged = _run_rollback(
        previous_sha=previous_sha, is_ancestor=is_ancestor, health=_health(PREVIOUS, "9-1-rollback")
    )

    assert deployed == "", f"an unverifiable predecessor was redeployed anyway: {deployed}"
    assert code != 0
    assert expected in logged
    assert "Restore production manually" in logged


def _rollback_fires(*, deploy_outcome: str, job_failed: bool = True) -> bool:
    """Evaluate the rollback step's `if:` the way Actions would, for one step-outcome scenario.

    Asserting the condition's text (`startswith("failure()")`) passes whether or not the guard that
    keeps a pre-deploy failure from restoring is present at all. The expression is the thing under
    test, so it has to be evaluated, not matched.
    """
    expression = _deploy_steps()[_step_index("Restore the previous release")]["if"]
    python = (
        expression.replace("&&", "and")
        .replace("||", "or")
        .replace("failure()", repr(job_failed))
        .replace("steps.gate.outputs.deploy", repr("true"))
        .replace("steps.deploy_serving.outcome", repr(deploy_outcome))
    )
    assert "steps." not in python, f"unresolved step reference in {expression!r}"
    # the expression comes from our own workflow file, with every step reference substituted out.
    return bool(eval(python))


def test_a_failure_before_the_deploy_never_restores_over_an_untouched_production():
    """The steps that verify secrets and auth run BEFORE any deploy, under the same gate output.

    If those satisfy the rollback condition, a run that never replaced production would `modal
    deploy` the previous sha over a live healthy app -- using the very secret set whose verification
    just failed. Restoring is only meaningful when this run actually deployed something.
    """
    steps = _deploy_steps()
    deploy_index = _step_index("Deploy serving")
    assert steps[deploy_index].get("id") == "deploy_serving", (
        "the rollback guard keys off the deploy step's id; without it the guard silently "
        "evaluates an empty outcome"
    )
    for step in steps[:deploy_index]:
        assert step.get("name") != "Restore the previous release"

    # skipped and empty are the two outcomes a step that never ran reports.
    assert not _rollback_fires(deploy_outcome="skipped")
    assert not _rollback_fires(deploy_outcome="")

    # a deploy that ran -- whether it succeeded and failed readiness later, or failed partway --
    # leaves the new release live, so both must still restore.
    assert _rollback_fires(deploy_outcome="success")
    assert _rollback_fires(deploy_outcome="failure")

    # and a green run never restores, whatever the deploy did.
    assert not _rollback_fires(deploy_outcome="success", job_failed=False)


def test_rollback_can_restore_a_release_before_the_serving_source_move():
    code, deployed, logged = _run_rollback(
        previous_sha=PREVIOUS,
        is_ancestor=True,
        health=_health(PREVIOUS, "9-1-rollback"),
        entrypoint="flash/serving/modal_app.py",
    )
    assert "deploy flash/serving/modal_app.py" in deployed
    assert f"sha={PREVIOUS}" in deployed
    assert code != 0
    assert "is NOT deployed" in logged


def test_rollback_refuses_a_release_without_a_serving_entrypoint():
    code, deployed, logged = _run_rollback(
        previous_sha=PREVIOUS,
        is_ancestor=True,
        health=_health(PREVIOUS, "9-1-rollback"),
        entrypoint=None,
    )
    assert code != 0
    assert deployed == ""
    assert "no supported serving entrypoint" in logged


@pytest.mark.parametrize("schema_status", ["400", "401", "500", "000"])
def test_rollback_does_not_deploy_an_unverified_database_contract(schema_status):
    code, deployed, logged = _run_rollback(
        previous_sha=PREVIOUS,
        is_ancestor=True,
        health=_health(PREVIOUS, "9-1-rollback"),
        schema_status=schema_status,
    )
    assert code != 0
    assert deployed == ""
    assert "refusing rollback" in logged


def test_rollback_does_not_deploy_without_a_schema_contract():
    code, deployed, logged = _run_rollback(
        previous_sha=PREVIOUS,
        is_ancestor=True,
        health=_health(PREVIOUS, "9-1-rollback"),
        has_schema_contract=False,
    )
    assert code != 0
    assert deployed == ""
    assert "Cannot verify" in logged


@pytest.mark.parametrize("org_id", ["", "   ", "invalid", "11111111-2222-4333-8444-555555555555"])
def test_canary_attribution_is_validated_before_production_deployment(org_id):
    step = _deploy_steps()[_step_index("Verify required serving runtime secrets")]
    env = dict.fromkeys(
        (
            "HF_TOKEN",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
            "FREESOLO_INTERNAL_KEY",
            "PLATFORM_BACKEND_URL",
        ),
        "configured",
    )
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        capture_output=True,
        text=True,
        env={**env, "PATH": os.environ["PATH"], "SERVING_CANARY_ORG_ID": org_id},
    )
    assert (result.returncode == 0) == org_id.startswith("11111111-")
    assert _step_index("Verify required serving runtime secrets") < _step_index("Deploy serving")


@pytest.mark.parametrize("workflow", ["deploy-modal.yml", "deploy-modal-dev.yml"])
def test_modal_deploy_client_matches_the_runtime_dependency_pin(workflow):
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    expected = next(
        requirement
        for requirement in project["project"]["optional-dependencies"]["serving"]
        if requirement.startswith("modal==")
    )
    installs = [
        step
        for job in _jobs(_load(WORKFLOW_DIR / workflow)).values()
        for step in _steps(job)
        if step.get("name") == "Install Modal client"
    ]
    assert len(installs) == 1
    requirements = [
        argument
        for line in installs[0]["run"].splitlines()
        for argument in shlex.split(line, comments=True)
        if argument.startswith("modal==")
    ]
    assert requirements == [expected]


@pytest.mark.parametrize("branch", ["dev", "main", "codex/modal-migration", ".feature"])
def test_serving_publish_tags_support_feature_branches(branch, tmp_path):
    workflow = _load(WORKFLOW_DIR / "publish-serving-image.yml")
    job = _jobs(workflow)["publish-serving-image"]
    script = next(step["run"] for step in _steps(job) if step.get("name") == "Compute tags")
    output = tmp_path / "outputs"
    env = {
        **os.environ,
        "GITHUB_REF_NAME": branch,
        "IMAGE": "ghcr.io/example/serve",
        "GITHUB_OUTPUT": str(output),
    }
    subprocess.run(["bash", "-eu", "-c", script], cwd=REPO_ROOT, env=env, check=True)
    refs = [line for line in output.read_text().splitlines() if line.startswith(env["IMAGE"])]
    assert refs
    assert all(
        re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", ref.rsplit(":", 1)[1]) for ref in refs
    )
    assert (env["IMAGE"] + ":latest" in refs) is (branch == "main")
    if branch == "dev":
        assert env["IMAGE"] + ":dev" in refs
