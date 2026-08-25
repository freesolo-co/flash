"""export project traces as records, prompts, or raw rows.

``records`` writes environment ``input``/``output`` rows and skips missing replies. ``prompts`` writes
``input`` only, including traces without replies for grpo sampling. ``raw`` preserves stored trace rows.
records serve both sft and grpo, and backend conversion matches the web export.
"""

from __future__ import annotations

import json
from pathlib import Path

from flash._internal.channel import CLI_NAME
from flash.cli.ui import render
from flash.client import ClientError, export_trace_records, list_trace_projects
from flash.client.config import load_credentials
from flash.client.http import has_freesolo_backend

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


def _require_api_key(api_key: str | None) -> str:
    """The key from a credential snapshot the caller already read, or the logged-out refusal.

    Takes the key instead of reading the config itself so that a caller which decided something
    from a snapshot -- `cmd_traces_export` deciding on the url -- sends the key from that SAME
    read. Re-reading here would let a `flash login` landing in between pair a hosted-url decision
    with a newly stored self-hosted plane credential and send that credential to the hosted
    backend, including for a caller whose snapshot had already established there was no key.
    """
    if not api_key:
        raise ClientError(
            f"not logged in. Run `{CLI_NAME} login` with your freesolo API key (or set FREESOLO_API_KEY)"
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
    """A project's traces in the requested shape, converted server-side.

    Omitting `api_key` reads the stored credential. Callers that already resolved configuration
    should pass the key from that snapshot so a concurrent login cannot change the request identity.
    """
    key = _require_api_key(api_key or load_credentials()[1])
    return export_trace_records(project_id, key, export_format=export_format)


def fetch_projects(api_key: str) -> list[dict]:
    """Projects in the caller's org that traces can be exported from."""
    return list_trace_projects(api_key)


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
    from flash.cli.commands.ops.account import unavailable_without_a_freesolo_backend

    # one snapshot for both: the url decides whether to refuse, and the key from that same read is
    # what gets sent, so a concurrent `flash login` cannot pair this url with a later credential.
    api_url, snapshot_key = load_credentials()
    if not has_freesolo_backend(api_url):
        # unlike `projects create`, there is nothing local to substitute: traces are written by the
        # freesolo SDK into the hosted backend, and flash itself never records one. so a
        # self-hosted plane has no trace store to read, not merely no route to it.
        raise unavailable_without_a_freesolo_backend(
            "exporting traces",
            because=(
                "traces are recorded by the freesolo SDK into the hosted backend, which a "
                "self-hosted plane does not have"
            ),
            instead=(
                'write the same {"input", "output"} JSONL rows yourself and train on them '
                f"directly (`{CLI_NAME} env setup` scaffolds the format)"
            ),
        )
    api_key = _require_api_key(snapshot_key)
    project_id = _resolve_project_id(args, api_key)
    export_format = getattr(args, "format", None) or RECORDS_FORMAT

    output = Path(getattr(args, "output", None) or default_output_path(export_format))
    if output.exists() and not getattr(args, "force", False):
        raise ClientError(f"{output} already exists; pass --force to overwrite it")

    exported = fetch_records(project_id, api_key, export_format)

    # old backends omit the format label and can only mean records. when a label is present it must
    # match the request, or rows could be mislabeled and later fed to an incompatible env/train path.
    # enforce this for the default format too because mismatch can occur in either direction.
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
                    f"train on it: {CLI_NAME} env push --project <project-uuid> --name my-env . "
                    f"&& {CLI_NAME} train configs/sft.toml"
                )
            )
        elif export_format == PROMPTS_FORMAT:
            print(render.arrow("prompts train under GRPO; the environment scores its samples"))
        return 0
    print(summary)
    return 0
