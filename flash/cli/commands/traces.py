"""export project traces as records, prompts, or raw rows.

``records`` writes environment ``input``/``output`` rows and skips missing replies. ``prompts`` writes
``input`` only, including traces without replies for grpo sampling. ``raw`` preserves stored trace rows.
records serve both sft and grpo, and backend conversion matches the web export.
"""

from __future__ import annotations

import json
from pathlib import Path

from flash._internal.channel import CLI_NAME
from flash.cli.commands import render
from flash.client import ClientError, export_trace_records, list_trace_projects
from flash.client.config import load_credentials
from flash.client.http import DEFAULT_FREESOLO_BASE_URL
from flash.serve.urls import is_freesolo_hosted_url

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


def trace_store_url(api_url: str) -> str:
    """Which plane holds this account's traces, independent of project-directory overrides.

    Deliberately NOT `has_freesolo_backend`. That answers a question about the project DIRECTORY --
    "is there a Freesolo-compatible backend to resolve projects against" -- which `FREESOLO_BASE_URL`
    can satisfy on behalf of a self-hosted plane. A trace store cannot be delegated that way: the
    recording proxy writes into the SQLite database of the plane that served it, so the traces exist
    on the plane the operator is logged into and nowhere else.

    Passing an explicit URL in both directions prevents `FREESOLO_BASE_URL` from diverting either a
    self-hosted operator key or a hosted account key to a service that never stored those traces.
    """
    return DEFAULT_FREESOLO_BASE_URL if is_freesolo_hosted_url(api_url) else api_url


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
    base_url: str | None = None,
) -> dict:
    """A project's traces in the requested shape, converted server-side.

    Omitting `api_key` reads the stored credential, which is right for `env_setup`'s call: it holds
    no snapshot this key has to agree with.
    """
    if api_key is None:
        api_url, snapshot_key = load_credentials()
        key = _require_api_key(snapshot_key)
        if base_url is None:
            base_url = trace_store_url(api_url)
    else:
        key = _require_api_key(api_key)
    return export_trace_records(project_id, key, base_url, export_format=export_format)


def fetch_projects(api_key: str, base_url: str | None = None) -> list[dict]:
    """Projects in the caller's trace store that can be exported from."""
    return list_trace_projects(api_key, base_url)


def _resolve_project_id(args, api_key: str, base_url: str | None = None) -> str:
    project_id = getattr(args, "project", None)
    if project_id:
        return str(project_id)

    projects = fetch_projects(api_key, base_url)
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
    # one snapshot for both: the url decides which trace store to call, and the key from that same
    # read is what gets sent, so a concurrent `flash login` cannot pair this url with a later
    # credential and disclose either credential to the wrong service.
    api_url, snapshot_key = load_credentials()
    api_key = _require_api_key(snapshot_key)
    trace_base_url = trace_store_url(api_url)
    project_id = _resolve_project_id(args, api_key, trace_base_url)
    export_format = getattr(args, "format", None) or RECORDS_FORMAT

    output = Path(getattr(args, "output", None) or default_output_path(export_format))
    if output.exists() and not getattr(args, "force", False):
        raise ClientError(f"{output} already exists; pass --force to overwrite it")

    exported = fetch_records(project_id, api_key, export_format, trace_base_url)

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
    # the export reads a capped window of the newest traces. say so, or a project past the cap looks
    # like it was exported whole and the older traces are silently missing from the dataset.
    if exported.get("truncated"):
        summary += " -- newest traces only; the project holds more than this export can read"
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
