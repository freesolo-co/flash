"""Verify the installed upstream source identity before constructing a trainer."""

import importlib.metadata
import json

from flash.engine.worker.train.entry.backend_common import VERL_REQUIREMENT_URL


def validate_upstream():
    expected_url, expected_commit = VERL_REQUIREMENT_URL.removeprefix("git+").rsplit("@", 1)
    distribution = importlib.metadata.distribution("verl")
    source = json.loads(distribution.read_text("direct_url.json") or "{}")
    vcs = source.get("vcs_info", {})
    if (
        source.get("url", "").removesuffix(".git") != expected_url.removesuffix(".git")
        or vcs.get("vcs") != "git"
        or vcs.get("commit_id") != expected_commit
    ):
        raise RuntimeError(
            "Flash requires the pinned unmodified upstream Verl checkout; rebuild the trainer "
            f"environment with {VERL_REQUIREMENT_URL}"
        )
