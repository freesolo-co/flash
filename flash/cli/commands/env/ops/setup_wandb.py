"""W&B metadata helpers for environment scaffolds."""

from __future__ import annotations

import json
from collections.abc import Mapping

_WANDB_PROJECT_MAX_LENGTH = 128
_WANDB_PROJECT_FORBIDDEN = frozenset("/\\#?%:")


def _wandb_project(project: Mapping[str, object]) -> str:
    """Adapt a resolved canonical project to a W&B-safe project name."""
    project_id = str(project["id"])
    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        return project_id
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
