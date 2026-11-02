"""Promotion-triggered bakes must be trusted, bounded, and refresh the original PR check."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/dev-kernel-cache.yml").read_text())
JOBS = WORKFLOW["jobs"]


def _evaluate(expression, **context):
    # these workflow conditions use only boolean operators, comparisons, and field access.
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression)
    return eval(
        expression, {"__builtins__": {}, "fromJSON": json.loads, "format": str.format}, context
    )


def _github(event, *, repo="freesolo-co/flash", head="dev", draft=False):
    return SimpleNamespace(
        repository="freesolo-co/flash",
        event_name=event,
        event=SimpleNamespace(
            pull_request=SimpleNamespace(
                head=SimpleNamespace(repo=SimpleNamespace(full_name=repo), ref=head), draft=draft
            )
        ),
    )


@pytest.mark.parametrize(
    ("event", "head_repo", "head", "draft", "allowed"),
    [
        ("push", "freesolo-co/flash", "dev", False, True),
        ("workflow_dispatch", "freesolo-co/flash", "dev", False, True),
        ("schedule", "freesolo-co/flash", "dev", False, False),
        ("pull_request", "freesolo-co/flash", "dev", False, True),
        ("pull_request", "someone/flash", "dev", False, False),
        ("pull_request", "freesolo-co/flash", "feature", False, False),
        ("pull_request", "freesolo-co/flash", "dev", True, False),
    ],
)
def test_only_trusted_nondraft_promotions_enter_candidate_workflow(
    event, head_repo, head, draft, allowed
):
    triggers = WORKFLOW.get("on", WORKFLOW.get(True))
    assert triggers["pull_request"]["branches"] == ["main"]
    assert "ready_for_review" in triggers["pull_request"]["types"]
    assert (
        _evaluate(
            JOBS["detect"]["if"], github=_github(event, repo=head_repo, head=head, draft=draft)
        )
        is allowed
    )


@pytest.mark.parametrize("event", ["push", "workflow_dispatch", "pull_request"])
@pytest.mark.parametrize("missing", [False, True])
def test_gpu_jobs_run_only_for_missing_manual_or_promotion_caches(event, missing):
    needs = SimpleNamespace(
        detect=SimpleNamespace(outputs=SimpleNamespace(needs_bake=str(missing).lower()))
    )
    for name in ("build-base", "bake", "publish-ready"):
        assert "detect" in JOBS[name]["needs"]
        assert _evaluate(JOBS[name]["if"], github=_github(event), needs=needs) is (
            missing and event != "push"
        )


def test_metadata_push_cannot_displace_a_queued_bake():
    group = WORKFLOW["concurrency"]["group"].removeprefix("dev-kernel-cache-")
    assert WORKFLOW["concurrency"]["cancel-in-progress"] is False
    assert _evaluate(group, github=_github("push")) == "detect"
    for event in ("workflow_dispatch", "pull_request", "schedule"):
        assert _evaluate(group, github=_github(event)) == "build"


@pytest.mark.parametrize(
    ("event", "deadline", "expected"),
    [("pull_request", "", 90), ("workflow_dispatch", 30, 30), ("workflow_dispatch", 0, 0)],
)
def test_promotion_has_bounded_deadline_without_masking_invalid_manual_zero(
    event, deadline, expected
):
    assert (
        _evaluate(
            JOBS["bake"]["with"]["deadline_minutes"],
            github=_github(event),
            inputs=SimpleNamespace(deadline_minutes=deadline),
        )
        == expected
    )


@pytest.mark.parametrize("run_ids", ["", "123\n456\n"])
def test_ready_publication_retries_failed_pr_checks_for_exact_baked_revision(tmp_path, run_ids):
    step = next(
        step for step in JOBS["publish-ready"]["steps"] if step.get("name", "").startswith("Re-run")
    )
    assert step["env"]["REVISION"] == "${{ needs.detect.outputs.revision }}"
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['run', 'list']:\n"
        "    print(os.environ['RUN_IDS'])\n"
    )
    gh.chmod(0o755)
    log = tmp_path / "calls"
    revision = "a" * 40
    subprocess.run(
        ["bash", "-eu", "-c", step["run"]],
        check=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "CALLS": str(log),
            "RUN_IDS": run_ids,
            "REVISION": revision,
            "GITHUB_REPOSITORY": "freesolo-co/flash",
        },
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == [
        "run",
        "list",
        "--repo",
        "freesolo-co/flash",
        "--workflow",
        "production-kernel-cache-ready.yml",
        "--commit",
        revision,
        "--event",
        "pull_request",
        "--status",
        "failure",
        "--json",
        "databaseId",
        "--jq",
        ".[].databaseId",
    ]
    assert calls[1:] == [
        ["run", "rerun", run_id, "--repo", "freesolo-co/flash", "--failed"]
        for run_id in run_ids.split()
    ]
