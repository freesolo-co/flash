"""print attempt-scoped worker text artifacts with explicit provenance."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from flash.adapters.artifacts import MAX_ATTEMPT_ID
from flash.cli.ui import lifecycle as lifecycle_ui
from flash.cli.ui import render
from flash.client import ClientError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flash.client import ApiClient


# stands in for the live attempt during the teardown window, when the plane has PROVEN there is no
# live worker (an explicitly null ``remote``). distinct from ``None``, which means UNKNOWN.
_NO_LIVE_WORKER = "no-live-worker"


def _artifact_attempt(name: str) -> int | None:
    """Return the bounded attempt encoded by a worker artifact name."""
    match = re.search(r"_attempt(\d+)\.txt$", name)
    if match is None:
        return None
    attempt = int(match.group(1))
    return attempt if attempt <= MAX_ATTEMPT_ID else None


def _worker_section_name(name: str, current_attempt: tuple[int, int] | str | None) -> str:
    """Label an attempt-scoped artifact and identify superseded output."""
    attempt = _artifact_attempt(name)
    if attempt is None:
        return name
    if current_attempt == _NO_LIVE_WORKER:
        return f"{name} (attempt={attempt}, worker torn down; no live attempt)"
    if current_attempt is None:
        return f"{name} (attempt={attempt})"
    current_attempt_id, _current_fence = current_attempt
    if attempt != current_attempt_id:
        return f"{name} (attempt={attempt}, previous attempt; current attempt={current_attempt_id})"
    return f"{name} (attempt={attempt}, current attempt)"


def _progress_line_identity(line: str) -> tuple[int, int] | None:
    if not line.startswith("PROGRESS "):
        return None
    try:
        progress = json.loads(line.removeprefix("PROGRESS "))
    except (TypeError, ValueError):
        return None
    if not isinstance(progress, dict):
        return None
    attempt_id = progress.get("attempt_id")
    fence = progress.get("fence")
    if (
        isinstance(attempt_id, bool)
        or not isinstance(attempt_id, int)
        or attempt_id < 0
        or isinstance(fence, bool)
        or not isinstance(fence, int)
        or fence < 1
    ):
        return None
    return attempt_id, fence


def _mark_superseded_lines(
    name: str, text: str, current_attempt: tuple[int, int] | str | None
) -> str:
    """Mark appended lines proven to come from a superseded attempt or fence."""
    if not isinstance(current_attempt, tuple):
        return text
    current_attempt_id, current_fence = current_attempt
    artifact_attempt = _artifact_attempt(name)
    artifact_is_previous = artifact_attempt is not None and artifact_attempt != current_attempt_id
    marked: list[str] = []
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        ending = line[len(content) :]
        identity = _progress_line_identity(content)
        line_is_previous = (
            identity != current_attempt if identity is not None else artifact_is_previous
        )
        if not line_is_previous:
            marked.append(line)
            continue
        attempt_id = identity[0] if identity is not None else artifact_attempt
        fence = identity[1] if identity is not None else "unknown"
        provenance = (
            f"attempt_id={attempt_id} fence={fence} "
            f"current_attempt_id={current_attempt_id} current_fence={current_fence}"
        )
        if content.startswith("PROGRESS "):
            marked.append(
                f"superseded_progress {provenance} | {content.removeprefix('PROGRESS ')}{ending}"
            )
        elif content:
            marked.append(f"SUPERSEDED {provenance} | {content}{ending}")
        else:
            marked.append(line)
    return "".join(marked)


def _worker_sections(client: ApiClient, run_id: str) -> dict[str, str]:
    """Fetch only worker artifacts with printable text."""
    return {name: text for name, text in (client.get_worker_output(run_id) or {}).items() if text}


def live_attempt_of(run: Mapping[str, object]) -> tuple[int, int] | str | None:
    """return the live fenced attempt, the teardown sentinel, or none when identity is unknown."""
    if run.get("remote", False) is None:
        return _NO_LIVE_WORKER
    return lifecycle_ui.live_attempt(dict(run))


def _snapshot_live_attempt(client: ApiClient, run_id: str) -> tuple[int, int] | str | None:
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
    current_attempt: tuple[int, int] | str | None = None,
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
        output = _mark_superseded_lines(name, text, current_attempt)
        print(output, end="" if output.endswith("\n") else "\n")
        printed_any = True
    return printed_any
