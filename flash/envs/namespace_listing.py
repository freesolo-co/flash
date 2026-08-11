"""List environments published under one managed-hub namespace.

GitHub helpers are resolved through ``loader`` lazily so the
``loader._download_github_json`` patch seam stays live. The listing uses GitHub tree API reads only
and never clones the environment hub.
"""

from __future__ import annotations

_LIST_READ_BUDGET = {"timeout": 20.0, "max_rate_limit_retries": 1}


def list_managed_namespace_slugs(namespace: str) -> list[str]:
    """Return sorted ``namespace/name`` slugs with an ``environment.py`` marker."""
    from flash.envs import loader

    if not loader._is_safe_github_path_parts((namespace,)):
        raise RuntimeError(f"unsafe managed environment namespace: {namespace!r}")
    ref = loader._parse_github_environment_ref(
        f"github:{loader._DEFAULT_MANAGED_ENV_REPO}@{loader._DEFAULT_GITHUB_REF}:"
        f"{namespace}/{loader._DEFAULT_ENVIRONMENT_PATH}"
    )
    if ref is None:  # pragma: no cover
        raise RuntimeError("could not build a managed environment reference")

    root = next(
        (
            entry
            for entry in loader._github_tree_entries(ref, ref.ref, ref.ref, **_LIST_READ_BUDGET)
            if entry.get("path") == namespace
        ),
        None,
    )
    if root is None:
        return []
    root_type = root.get("type")
    if not isinstance(root_type, str) or not root_type:
        raise RuntimeError(
            f"GitHub tree entry for environment namespace {namespace!r} has an unusable type"
        )
    if root_type != "tree":
        return []
    root_sha = root.get("sha")
    if not isinstance(root_sha, str) or not root_sha:
        raise RuntimeError(
            f"GitHub tree entry for environment namespace {namespace!r} has no usable sha"
        )

    entries = loader._github_tree_entries(
        ref, root_sha, namespace, recursive=True, **_LIST_READ_BUDGET
    )
    slugs: set[str] = set()
    for entry in entries:
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        parts = path.split("/")
        if len(parts) != 2 or parts[1] != loader._DEFAULT_ENVIRONMENT_PATH:
            continue
        kind = entry.get("type")
        if not isinstance(kind, str) or not kind:
            raise RuntimeError(
                f"GitHub tree entry {path!r} in environment namespace {namespace!r} "
                "has an unusable type"
            )
        if kind == "blob" and loader._is_safe_github_path_parts((parts[0],)):
            slugs.add(f"{namespace}/{parts[0]}")
    return sorted(slugs)
