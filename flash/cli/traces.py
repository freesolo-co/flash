"""`flash traces export`: turn a project's recorded traces into training rows.

Traces logged through the freesolo SDK export as freesolo environment records
(``{"input", "output"}``), the same shape `flash env setup` scaffolds into
dataset/train.jsonl, so an exported file is a drop-in dataset. The conversion
runs on the freesolo backend, which is also where the web app's trace export
gets its rows, so both produce the same file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flash.client import ApiError, ClientError, export_trace_records, list_trace_projects
from flash.client.config import load_credentials

from . import render

DEFAULT_EXPORT_PATH = Path("dataset/train.jsonl")


def _require_api_key() -> str:
    _api_url, api_key = load_credentials()
    if not api_key:
        raise ClientError(
            "not logged in. Run `flash login` with your freesolo API key (or set FREESOLO_API_KEY)"
        )
    return api_key


def project_options(projects: list[dict]) -> list[tuple[str, str, str]]:
    """`render.select` options for a project list: (id, name, id hint)."""
    return [
        (str(project.get("id")), str(project.get("name") or project.get("id")), "")
        for project in projects
        if project.get("id")
    ]


def records_to_jsonl(records: list[dict]) -> str:
    """One JSON object per line, matching the platform's downloaded export."""
    if not records:
        return ""
    return "".join(f"{json.dumps(record)}\n" for record in records)


def fetch_records(project_id: str, api_key: str | None = None) -> dict:
    """Environment records for a project's traces, converted server-side."""
    return export_trace_records(project_id, api_key or _require_api_key())


def fetch_projects(api_key: str | None = None) -> list[dict]:
    """Projects in the caller's org that traces can be exported from."""
    return list_trace_projects(api_key or _require_api_key())


def offer_projects() -> list[dict]:
    """Projects to offer as a starting dataset, or [] when we can't or shouldn't ask.

    Used by `flash env setup`, where traces are an optional head start rather than the point of
    the command: not being logged in, being offline, or having no traces yet all mean "just
    scaffold the starter dataset", so every such failure returns [] instead of raising."""
    _api_url, api_key = load_credentials()
    if not api_key:
        return []
    try:
        return fetch_projects(api_key)
    except (ApiError, ClientError, OSError, ValueError):
        return []


def _resolve_project_id(args, api_key: str) -> str:
    project_id = getattr(args, "project", None)
    if project_id:
        return str(project_id)

    projects = fetch_projects(api_key)
    if not projects:
        raise ClientError(
            "no projects with traces found for this account. Record traces with the freesolo "
            "SDK first, then export them here"
        )
    options = project_options(projects)
    if len(options) == 1:
        return options[0][0]
    if not _interactive():
        listing = ", ".join(f"{name} ({value})" for value, name, _hint in options)
        raise ClientError(f"pass --project <id> to choose a project; available: {listing}")
    return render.select("Which project's traces?", options)


def _interactive() -> bool:
    """Whether a project picker can be answered. Mirrors env setup's guard."""
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        return False
    return render.styled()


def cmd_traces_export(args) -> int:
    api_key = _require_api_key()
    project_id = _resolve_project_id(args, api_key)

    output = Path(getattr(args, "output", None) or DEFAULT_EXPORT_PATH)
    if output.exists() and not getattr(args, "force", False):
        raise ClientError(f"{output} already exists; pass --force to overwrite it")

    exported = fetch_records(project_id, api_key)
    records = exported.get("records") or []
    if not records:
        raise ClientError(
            f"no exportable traces in project {project_id}. Traces need a recorded request and "
            "response to become training rows"
        )

    parent = output.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    output.write_text(records_to_jsonl(records), encoding="utf-8")

    skipped = int(exported.get("skipped") or 0)
    summary = f"exported {len(records)} training rows to {output}"
    if skipped:
        summary += f" ({skipped} traces skipped: no usable request/response pair)"
    if render.styled():
        print(render.ok(summary))
        print(
            render.arrow(
                "train on it: flash env push --name my-env . && flash train configs/sft.toml"
            )
        )
        return 0
    print(summary)
    return 0
