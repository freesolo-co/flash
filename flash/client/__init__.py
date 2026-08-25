"""HTTP client for the managed Flash control plane (used by the CLI)."""

from collections.abc import Mapping

from flash._internal.channel import CLI_NAME
from flash.client.config import load_credentials, save_credentials
from flash.client.http import (
    ApiClient,
    ApiError,
    ClientError,
    RequestTimeoutError,
    ServiceUnreachableError,
    client_from_config,
    create_project,
    export_trace_records,
    get_project,
    has_freesolo_backend,
    list_projects,
    list_trace_projects,
    upload_eval_run,
    verify_freesolo_key,
)
from flash.core.spec import require_project_id


def resolve_project(
    project_id: str, api_key: str, api_url: str | None = None
) -> Mapping[str, object]:
    """Resolve one project, turning "not yours / not there" into the actionable CLI refusal.

    403 and 404 are the same answer to a CLI user -- this project is not one you can record against
    -- and neither is distinguishable from a typo, so they collapse into one message naming the way
    out. Every other status stays an ``ApiError``: a 500 is an outage, not a bad id.

    With no Freesolo backend to reach there is no org directory to resolve through, so the id is
    only validated for SHAPE -- exactly what ``flash/server/domain/registry/projects.py`` does under
    ``standalone()`` when the same run is submitted. Without this the ownership lookup went to
    ``api.freesolo.co``, which has no relationship with the operator's key and answers 401, so
    ``flash env setup`` died before writing a file on the very quickstart SELF_HOSTING.md
    documents. A backend-less plane exposes no project routes at all, so there is nothing else to
    ask.

    ``api_url`` is passed in rather than re-read from config because the caller has already
    resolved it (and because ``FLASH_STANDALONE`` lives on the SERVER, so the client cannot see
    it). It is only one of the two signals though: an operator running their own plane against a
    Freesolo-compatible backend DOES have a directory, and deciding on the url alone skipped the
    ownership check for them, accepting any well-formed uuid. ``has_freesolo_backend`` is the
    classifier the project and trace commands already use, so route through it here too rather
    than keep a second, narrower answer to the same question. Omitting ``api_url`` keeps the
    hosted behaviour.

    Defined here rather than in ``http`` so ``get_project`` resolves through THIS module's attribute
    at call time, which is the binding the CLI tests patch.
    """
    if api_url is not None and not has_freesolo_backend(api_url):
        return {"id": require_project_id(project_id)}
    try:
        return get_project(project_id, api_key)
    except ApiError as exc:
        if exc.status not in {403, 404}:
            raise
        raise ClientError(
            f"project {project_id!r} is not accessible; run `{CLI_NAME} projects list` "
            "and pass a project UUID from the current organization"
        ) from exc


def resolve_project_id(project_id: str, api_key: str, api_url: str | None = None) -> str:
    """Resolve and return one canonical project id."""
    return str(resolve_project(project_id, api_key, api_url)["id"])


__all__ = [
    "ApiClient",
    "ApiError",
    "ClientError",
    "RequestTimeoutError",
    "ServiceUnreachableError",
    "client_from_config",
    "create_project",
    "export_trace_records",
    "get_project",
    "list_projects",
    "list_trace_projects",
    "load_credentials",
    "resolve_project",
    "resolve_project_id",
    "save_credentials",
    "upload_eval_run",
    "verify_freesolo_key",
]
