#!/usr/bin/env python3
"""fail if the pinned verl commit does not exist on the fork.

every offline pin test compares the pin against a hardcoded copy of the same sha, so all of them
pass when the sha is mis-transcribed into a commit that was never pushed. that shipped: dev carried
`32d6200de81dcc9e97e3aa2ae4a2b3ba30d33e93`, a 40-hex string sharing the real commit's first 13
characters, and every worker install would have died on `upload-pack: not our ref`. the offline
suite cannot see this -- only the fork can answer it -- so this runs as its own CI step.

`git ls-remote <url> <sha>` is the wrong probe: it matches ref NAMES, not object ids, and answers
empty for a perfectly good commit. fetching the object is what actually proves reachability.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from flash.engine.worker.train.entry.backend_common import VERL_REQUIREMENT  # noqa: E402

DOCKERFILE = os.path.join(REPO_ROOT, "Dockerfile.worker")


def pinned_url_and_commit() -> tuple[str, str]:
    _, _, ref = VERL_REQUIREMENT.partition("git+")
    url, _, commit = ref.rpartition("@")
    return url, commit


def dockerfile_commit() -> str:
    with open(DOCKERFILE) as f:
        source = f.read()
    match = re.search(r"^ARG VERL_SPEC=.*?@([0-9a-f]{40})\s*$", source, re.MULTILINE)
    if not match:
        raise SystemExit(f"could not find a 40-hex VERL_SPEC pin in {DOCKERFILE}")
    return match.group(1)


def main() -> int:
    url, commit = pinned_url_and_commit()

    # the lockstep test already asserts these agree, but it does so offline against the same two
    # strings this script is about to prove reachable. re-check here so a green result means the
    # commit the IMAGE installs was proven, not just the one the python constant names.
    baked = dockerfile_commit()
    if baked != commit:
        print(f"FAIL: Dockerfile.worker pins {baked}, backend_common pins {commit}")
        return 1

    with tempfile.TemporaryDirectory() as workdir:
        subprocess.run(["git", "init", "--quiet"], cwd=workdir, check=True)
        probe = subprocess.run(
            ["git", "fetch", "--depth", "1", "--no-tags", url, commit],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )

    if probe.returncode != 0:
        print(f"FAIL: {url}@{commit} is not fetchable")
        print(probe.stderr.strip())
        print()
        print("every worker install and image build resolves this exact commit, so a pin that")
        print("does not exist fails provisioning on every run rather than at build time.")
        return 1

    print(f"ok: {url}@{commit} exists and is fetchable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
