"""print attempt-scoped worker text artifacts with explicit provenance."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.cli.ui import lifecycle as lifecycle_ui
from flash.cli.ui import render
from flash.client import ClientError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flash.client import ApiClient


# stands in for a terminal run whose missing remote proves no worker remains. distinct from
# ``None``, which means the live attempt is unknown.
_NO_LIVE_WORKER = "no-live-worker"
_TERMINAL_RUN_STATES = frozenset({"done", "failed", "cancelled", "dry_run"})


def _artifact_attempt(name: str) -> int | None:
    """Return the bounded attempt encoded by a worker artifact name."""
    match = re.search(r"_attempt(\d+)\.txt$", name)
    if match is None:
        return None
    attempt = int(match.group(1))
    return attempt if attempt <= MAX_ATTEMPT_ID else None


def _worker_section_name(name: str, current_attempt: int | str | None) -> str:
    """Label an attempt-scoped artifact and identify superseded output."""
    attempt = _artifact_attempt(name)
    if attempt is None:
        return name
    if current_attempt == _NO_LIVE_WORKER:
        return f"{name} (attempt={attempt}, worker torn down; no live attempt)"
    if current_attempt is None:
        return f"{name} (attempt={attempt})"
    if attempt != current_attempt:
        return f"{name} (attempt={attempt}, previous attempt; current attempt={current_attempt})"
    return f"{name} (attempt={attempt}, current attempt)"


def _worker_sections(client: ApiClient, run_id: str) -> dict[str, str]:
    """Fetch only worker artifacts with printable text."""
    return {name: text for name, text in (client.get_worker_output(run_id) or {}).items() if text}


def live_attempt_of(run: Mapping[str, object]) -> int | str | None:
    """return the live attempt, confirmed teardown, or none when identity is unknown."""
    payload = dict(run)
    identity = lifecycle_ui.live_attempt(payload)
    if run.get("remote") is not None:
        return identity[0] if identity is not None else None
    if run.get("state") in _TERMINAL_RUN_STATES:
        return _NO_LIVE_WORKER
    return identity[0] if identity is not None else None


def _snapshot_live_attempt(client: ApiClient, run_id: str) -> int | str | None:
    """``live_attempt_of`` for the non-follow path, which has to fetch the status itself.

    A lookup failure stays ``None``: not knowing the live attempt must not make this claim there
    is no live worker.
    """
    try:
        run = client.get_run(run_id) or {}
    except ClientError:
        return None
    return live_attempt_of(run)


def _print_worker_output(
    sections: Mapping[str, str],
    *,
    printed_any: bool = False,
    current_attempt: int | str | None = None,
) -> bool:
    """Print worker artifacts under headings naming the attempt that produced them."""
    for name, text in sections.items():
        # label provenance from the filename rather than letting the final section, which is merely
        # the last one appended, read as the live attempt's output.
        section_name = _worker_section_name(name, current_attempt)
        sep = "\n" if printed_any else ""
        if render.styled():
            print(f"{sep}{render.log_section(section_name)}")
        else:
            print(f"{sep}----- {section_name} -----")
        print(text, end="" if text.endswith("\n") else "\n")
        printed_any = True
    return printed_any
