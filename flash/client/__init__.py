"""HTTP client for the managed Flash control plane (used by the CLI)."""

from flash.client.config import load_credentials, save_credentials
from flash.client.http import (
    ApiClient,
    ApiError,
    ClientError,
    RequestTimeoutError,
    client_from_config,
    create_project,
    export_trace_records,
    get_project,
    list_projects,
    list_trace_projects,
    upload_eval_run,
    verify_freesolo_key,
)
from flash.core.spec import require_project_id


def resolve_project_id(project_id: str, api_key: str, api_url: str | None = None) -> str:
    """Resolve one project id, turning "not yours / not there" into the actionable CLI refusal.

    403 and 404 are the same answer to a CLI user -- this project is not one you can record against
    -- and neither is distinguishable from a typo, so they collapse into one message naming the way
    out. Every other status stays an ``ApiError``: a 500 is an outage, not a bad id.

    Against a self-hosted plane there is no org directory to resolve through, so the id is only
    validated for SHAPE -- exactly what ``flash/server/domain/projects.py`` does under ``standalone()``
    when the same run is submitted. Without this the ownership lookup went to
    ``api.freesolo.co``, which has no relationship with the operator's key and answers 401, so
    ``flash env setup`` died before writing a file on the very quickstart SELF_HOSTING.md
    documents. The plane exposes no project routes at all, so there is nothing else to ask.

    ``api_url`` is the standalone signal, passed in rather than re-read from config because the
    caller has already resolved it (and, like ``_verifies_against_freesolo``, because
    ``FLASH_STANDALONE`` lives on the SERVER and the client cannot see it). Omitting it keeps the
    hosted behaviour, so this stays a widening of the contract rather than a change to it.

    Defined here rather than in ``http`` so ``get_project`` resolves through THIS module's attribute
    at call time, which is the binding the CLI tests patch.
    """
    from flash.serve.urls import is_freesolo_hosted_url

    if api_url is not None and not is_freesolo_hosted_url(api_url):
        return require_project_id(project_id)
    try:
        return str(get_project(project_id, api_key)["id"])
    except ApiError as exc:
        if exc.status not in {403, 404}:
            raise
        raise ClientError(
            f"project {project_id!r} is not accessible; run `flash projects list` "
            "and pass a project UUID from the current organization"
        ) from exc


__all__ = [
    "ApiClient",
    "ApiError",
    "ClientError",
    "RequestTimeoutError",
    "client_from_config",
    "create_project",
    "export_trace_records",
    "get_project",
    "list_projects",
    "list_trace_projects",
    "load_credentials",
    "resolve_project_id",
    "save_credentials",
    "upload_eval_run",
    "verify_freesolo_key",
]
