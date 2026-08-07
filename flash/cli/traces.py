"""`flash traces export`: turn a project's recorded traces into training rows.

Three shapes, because these are the ones that actually differ:

- ``records`` (default): ``{"input", "output"}`` freesolo environment records,
  the same shape `flash env setup` scaffolds into dataset/train.jsonl, so an
  exported file is a drop-in dataset. Traces with no usable reply are skipped.
- ``prompts``: ``{"input"}`` only. GRPO samples its own completions and scores
  them with the environment, so no gold reply is needed -- and a call whose
  reply never arrived still exports, which ``records`` drops.
- ``raw``: the stored trace rows with their spans, converted by nothing.

There is deliberately no per-algorithm format: SFT and GRPO both read the same
environment records, so an "export for SFT" and an "export for GRPO" would be
byte-identical files.

The conversion runs on the freesolo backend, which is also where the web app's
trace export gets its rows, so both produce the same file.
"""

from __future__ import annotations

import json
from pathlib import Path

from flash.client import ClientError, export_trace_records, list_trace_projects
from flash.client.config import load_credentials

from . import render

DEFAULT_EXPORT_PATH = Path("dataset/train.jsonl")
# raw rows are not a dataset. load_freesolo_environment auto-selects
# dataset/train.jsonl as a run's dataset_path, so a raw dump written there could
# be picked up by a later `env push` + `train` and fed to a run that cannot read
# it. prompts stay on the default path: they are training input for GRPO.
RAW_EXPORT_PATH = Path("traces.raw.jsonl")

RECORDS_FORMAT = "records"
PROMPTS_FORMAT = "prompts"
RAW_FORMAT = "raw"
EXPORT_FORMATS = (RECORDS_FORMAT, PROMPTS_FORMAT, RAW_FORMAT)


def default_output_path(export_format: str) -> Path:
    """Where an export lands when no --output is given."""
    return RAW_EXPORT_PATH if export_format == RAW_FORMAT else DEFAULT_EXPORT_PATH


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
        (
            str(project.get("id")),
            str(project.get("name") or "unnamed project"),
            str(project.get("id")),
        )
        for project in projects
        if project.get("id")
    ]


def records_to_jsonl(records: list[dict]) -> str:
    """One JSON object per line, matching the platform's downloaded export."""
    if not records:
        return ""
    return "".join(f"{json.dumps(record)}\n" for record in records)


def fetch_records(
    project_id: str,
    api_key: str | None = None,
    export_format: str = RECORDS_FORMAT,
) -> dict:
    """A project's traces in the requested shape, converted server-side."""
    return export_trace_records(
        project_id, api_key or _require_api_key(), export_format=export_format
    )


def fetch_projects(api_key: str | None = None) -> list[dict]:
    """Projects in the caller's org that traces can be exported from."""
    return list_trace_projects(api_key or _require_api_key())


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
    if not render.can_prompt():
        listing = ", ".join(f"{name} ({value})" for value, name, _hint in options)
        raise ClientError(f"pass --project <id> to choose a project; available: {listing}")
    return render.select_required("Which project's traces?", options)


# what each shape is called in the summary line. `records` and `prompts` are
# training rows; `raw` is not, so calling them all "training rows" would lie.
_ROW_NOUN = {
    RECORDS_FORMAT: "training rows",
    PROMPTS_FORMAT: "prompts",
    RAW_FORMAT: "traces",
}

# why a trace produced no row, per shape. `records` needs both halves of the
# exchange; `prompts` only needs the request, so blaming a missing response
# there would contradict the reason that shape exists. `raw` converts nothing
# and so can never skip.
_SKIP_REASON = {
    RECORDS_FORMAT: "no usable request/response pair",
    PROMPTS_FORMAT: "no usable request",
}


def _empty_export_error(project_id: str, export_format: str) -> ClientError:
    """Why an export came back empty, in terms of the shape that was asked for."""

    if export_format == RECORDS_FORMAT:
        return ClientError(
            f"no exportable traces in project {project_id}. Traces need a recorded request and "
            f"response to become training rows -- try --format {PROMPTS_FORMAT} if you only "
            "need the prompts"
        )
    if export_format == PROMPTS_FORMAT:
        return ClientError(
            f"no exportable traces in project {project_id}. Traces need a recorded request to "
            "become prompts"
        )
    return ClientError(f"no traces in project {project_id}")


def cmd_traces_export(args) -> int:
    api_key = _require_api_key()
    project_id = _resolve_project_id(args, api_key)
    export_format = getattr(args, "format", None) or RECORDS_FORMAT

    output = Path(getattr(args, "output", None) or default_output_path(export_format))
    if output.exists() and not getattr(args, "force", False):
        raise ClientError(f"{output} already exists; pass --force to overwrite it")

    exported = fetch_records(project_id, api_key, export_format)

    # a backend predating format support ignores the query param and reports no
    # format at all, so a missing one is read as the records it must have sent.
    # an explicit label is the backend telling us what it actually converted, and
    # any value other than the one asked for means the rows are not what the
    # caller requested -- writing them out would label {input, output} rows as
    # prompts or raw, or land raw traces in dataset/train.jsonl as if they were
    # environment records, where a later `env push` + `train` would feed them to a
    # run that cannot read them. that mislabelling is possible in either
    # direction, so the check cannot be skipped for the default format.
    returned_format = exported.get("format") or RECORDS_FORMAT
    if returned_format != export_format:
        raise ClientError(
            f"the freesolo backend did not honour --format {export_format} "
            f"(it returned {returned_format}). It may predate format "
            "support -- check FREESOLO_BASE_URL or upgrade the backend"
        )

    records = exported.get("records") or []
    if not records:
        raise _empty_export_error(project_id, export_format)

    parent = output.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    output.write_text(records_to_jsonl(records), encoding="utf-8")

    skipped = int(exported.get("skipped") or 0)
    noun = _ROW_NOUN.get(export_format, "rows")
    summary = f"exported {len(records)} {noun} to {output}"
    if skipped:
        reason = _SKIP_REASON.get(export_format)
        summary += f" ({skipped} traces skipped: {reason})" if reason else f" ({skipped} skipped)"
    if render.styled():
        print(render.ok(summary))
        # raw rows are for taking away, not for training: pointing them at
        # `flash train` would send someone to a run that cannot read them.
        if export_format == RECORDS_FORMAT:
            print(
                render.arrow(
                    "train on it: flash env push --project <project-uuid> --name my-env . "
                    "&& flash train configs/sft.toml"
                )
            )
        elif export_format == PROMPTS_FORMAT:
            print(render.arrow("prompts train under GRPO; the environment scores its samples"))
        return 0
    print(summary)
    return 0
