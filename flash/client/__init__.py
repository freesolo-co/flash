"""HTTP client for the managed Flash control plane (used by the CLI)."""

from .config import load_credentials, save_credentials
from .http import (
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


def resolve_project_id(project_id: str, api_key: str) -> str:
    """Resolve one project id, turning "not yours / not there" into the actionable CLI refusal.

    403 and 404 are the same answer to a CLI user -- this project is not one you can record against
    -- and neither is distinguishable from a typo, so they collapse into one message naming the way
    out. Every other status stays an ``ApiError``: a 500 is an outage, not a bad id.

    Defined here rather than in ``http`` so ``get_project`` resolves through THIS module's attribute
    at call time, which is the binding the CLI tests patch.
    """
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
