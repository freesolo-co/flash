"""W&B metadata helpers for environment scaffolds."""

from __future__ import annotations

import json

_WANDB_PROJECT_MAX_LENGTH = 128
_WANDB_PROJECT_FORBIDDEN = frozenset("/\\#?%:")


def _require_setup_project_name(project_id: str) -> str:
    """Fetch the canonical project name used by the generated W&B metadata."""
    from flash.client import ClientError, get_project
    from flash.client.config import load_credentials
    from flash.client.http import has_freesolo_backend

    api_url, api_key = load_credentials()
    if api_url is not None and not has_freesolo_backend(api_url):
        return project_id
    project = get_project(project_id, api_key)
    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ClientError(f"project {project_id!r} has no nonblank canonical name")
    if len(name) > _WANDB_PROJECT_MAX_LENGTH or any(
        character in _WANDB_PROJECT_FORBIDDEN for character in name
    ):
        return project_id
    return name


def _wandb_block(project_name: str, run_name: str) -> str:
    return (
        "[wandb]\n"
        f"project = {json.dumps(project_name, ensure_ascii=False)}\n"
        f"run_name = {json.dumps(run_name, ensure_ascii=False)}\n\n"
    )
