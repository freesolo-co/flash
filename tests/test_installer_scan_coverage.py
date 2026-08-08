"""The installer scan discovers its own inputs, and these tests prove it still does.

Separate from `test_no_pipe_to_shell` because they exercise the SCAN surface -- which files the
guard reads -- rather than the shell grammar it applies to them. A guard that quietly narrows its
own scope reports exactly the same green as one that found nothing, so each test here re-runs the
REAL discovery against a planted file rather than asserting on a copy of its glob patterns.
"""

from __future__ import annotations

from tests.pipe_to_shell_scan import INSTALLER_FILES, REPO_ROOT, _installer_files


def test_the_installer_scan_covers_every_dockerfile_in_the_repo():
    """The scan discovers its own inputs, so a new Dockerfile cannot silently escape it.

    An enumerated list looks identical to a complete one right up until someone adds a file:
    `docker/Dockerfile.kernelcache` and its relayer were both outside a list whose own docstring
    called the rule repo-wide. Deriving the set from the tree is what makes the claim true.
    """
    on_disk = {p.resolve() for p in REPO_ROOT.rglob("Dockerfile*") if ".git" not in p.parts}
    scanned = {p.resolve() for p in INSTALLER_FILES}
    missed = on_disk - scanned
    assert not missed, f"Dockerfiles not covered by the installer scan: {sorted(missed)}"


def test_the_installer_scan_covers_workflows_under_both_yaml_extensions():
    """GitHub Actions runs `.yaml` as readily as `.yml`, so the scan must accept both.

    There is no `.yaml` workflow today, which is exactly why this is worth pinning: a scan keyed
    to one extension goes green forever, and the day a workflow is added or renamed under the
    other one it silently leaves coverage. So plant one and re-run the REAL discovery. An earlier
    version of this test rebuilt the glob patterns inline and asserted against that copy, which
    passed unchanged when the production glob was narrowed back to `*.yml` -- it was testing its
    own literal, not the scan.
    """
    workflows = REPO_ROOT / ".github" / "workflows"
    on_disk = {p.resolve() for p in workflows.iterdir() if p.suffix in (".yml", ".yaml")}
    assert not on_disk - {p.resolve() for p in INSTALLER_FILES}, "a workflow is outside the scan"

    planted = workflows / "zz-extension-probe.yaml"
    planted.write_text("name: probe\non: workflow_dispatch\njobs: {}\n")
    try:
        assert planted.resolve() in {p.resolve() for p in _installer_files()}, (
            "a .yaml workflow would not be discovered"
        )
    finally:
        planted.unlink()
