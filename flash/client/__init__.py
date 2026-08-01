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
    "save_credentials",
    "upload_eval_run",
    "verify_freesolo_key",
]
