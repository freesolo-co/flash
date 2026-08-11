"""Listing the environments published under one managed-hub namespace.

Split out of ``flash.envs.loader`` to keep that module under the file-size gate. This is the read
side of ``_github_publish``: it answers "what has this org published" from GitHub tree reads alone,
with no clone, and it is the one hub read a person waits on interactively -- which is why its budget
and its truncation handling differ from the batch download paths. ``loader`` re-exports
``list_managed_namespace_slugs``, since that is the name callers and tests already use.

Every GitHub helper is resolved through ``loader`` on each call rather than imported here. Tests
drive this whole module by patching ``loader._download_github_json``, and an import-time binding in
this namespace would keep pointing at the original function and silently ignore that patch.
"""

from __future__ import annotations

# how many per-package reads the truncation recovery may add. Sized so the worst case stays within
# the client's derived wait rather than by how many envs an org "should" have: the recursive read
# only truncates once a namespace holds roughly twenty full packages, and a prefix that settles none
# of them is the pathological shape this bound exists for.
_MAX_TRUNCATION_RECOVERY_READS = 8


def list_managed_namespace_slugs(namespace: str) -> list[str]:
    """List ``namespace/name`` slugs published under one managed-hub namespace, sorted.

    Tree reads and no clone: find the namespace directory in the hub root, then read it one level
    deep and keep the subdirectories that actually contain ``environment.py`` -- which is what
    ``_github_publish`` writes, so a stray file at the namespace root is not an environment.
    Authentication is the ambient ``GITHUB_TOKEN`` that every other GitHub read here uses.

    An absent namespace directory means the org has published nothing yet and returns an empty
    list. Every other failure propagates, because "no environments" must never be how a broken hub
    read looks: that is indistinguishable from a publish that silently did nothing.

    Both reads are deliberately bounded well below the publish path's budget. This serves a person
    waiting at a terminal, and inheriting the batch default (6 attempts x 120s socket + up to 180s
    backoff, twice over) would let a hard-down GitHub hold the request ~30 minutes before returning
    the controlled 502 -- long enough that every caller times out first and sees a generic failure
    instead of the upstream status. One retry at a 20s socket timeout answers within ~1 minute
    worst case, which is what makes the 429/502 actually reachable by a client.
    """
    from flash.envs import loader

    if not loader._is_safe_github_path_parts((namespace,)):
        raise RuntimeError(f"unsafe managed environment namespace: {namespace!r}")
    ref = loader._parse_github_environment_ref(
        f"github:{loader._DEFAULT_MANAGED_ENV_REPO}@{loader._DEFAULT_GITHUB_REF}:"
        f"{namespace}/{loader._DEFAULT_ENVIRONMENT_PATH}"
    )
    if ref is None:  # pragma: no cover - the literal above always parses
        raise RuntimeError("could not build a managed environment reference")

    root_sha = _namespace_root_sha(ref, namespace)
    if root_sha is None:
        return []

    # ONE recursive fetch of the namespace subtree, not a tree request per environment: an org with
    # 100 envs would otherwise spend 102 calls of the shared GITHUB_TOKEN's budget on a list. Paths
    # in a recursive response are relative to the namespace, so `<name>/environment.py` identifies an
    # environment directly and no per-directory read is needed.
    #
    # A recursive tree can come back TRUNCATED, though, and that is reachable on valid input: each
    # package may hold up to `_MAX_ARCHIVE_MEMBERS` files, so roughly twenty full ones in a single
    # namespace exceed GitHub's response limit. Raising there would make every environment in that
    # namespace unlistable, so a truncated tree keeps what it did prove and settles the rest below.
    entries, truncated = loader._github_tree_entries_allowing_truncation(
        ref, root_sha, namespace, recursive=True, **loader._LIST_READ_BUDGET
    )
    slugs = _slugs_from_recursive_entries(entries, namespace)
    if truncated:
        slugs.update(_recover_truncated_namespace_slugs(ref, root_sha, namespace, slugs))
    return sorted(slugs)


def _namespace_root_sha(ref: object, namespace: str) -> str | None:
    """The tree sha of the namespace directory, or ``None`` when the org published nothing yet."""
    from flash.envs import loader

    # match on the PATH only, then validate type and sha below. folding either check into this
    # predicate would make a malformed entry look like a MISSING one, so a broken hub response would
    # report "nothing published" -- the exact failure this endpoint exists to remove.
    root = next(
        (
            entry
            for entry in loader._github_tree_entries(
                ref, ref.ref, ref.ref, **loader._LIST_READ_BUDGET
            )
            if entry.get("path") == namespace
        ),
        None,
    )
    if root is None:
        return None
    root_type = root.get("type")
    if not isinstance(root_type, str) or not root_type:
        raise RuntimeError(
            f"GitHub tree entry for environment namespace {namespace!r} has an unusable type; "
            "the hub listing could not be read"
        )
    # a well-formed non-tree is a readable answer, not a broken one: `_github_publish` writes a
    # directory, so a stray file at this path is unambiguously not an environment namespace.
    if root_type != "tree":
        return None
    root_sha = root.get("sha")
    if not isinstance(root_sha, str) or not root_sha:
        raise RuntimeError(
            f"GitHub tree entry for environment namespace {namespace!r} has no usable sha; "
            "the hub listing could not be read"
        )
    return root_sha


def _slugs_from_recursive_entries(entries: list[dict], namespace: str) -> set[str]:
    """The slugs a recursive namespace tree proves, by their ``<name>/environment.py`` markers."""
    from flash.envs import loader

    slugs: set[str] = set()
    for entry in entries:
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        parts = path.split("/")
        if len(parts) != 2 or parts[1] != loader._DEFAULT_ENVIRONMENT_PATH:
            continue
        # the path says this IS an environment marker, so decide on its type only once we know the
        # type is readable. skipping an entry whose type is missing or malformed would shorten the
        # list -- to "nothing published" for a single-env org -- which is the silent-empty failure
        # this endpoint exists to remove. a well-formed non-blob (a directory that happens to be
        # named environment.py) is a legitimate skip and stays quiet.
        kind = entry.get("type")
        if not isinstance(kind, str) or not kind:
            raise RuntimeError(
                f"GitHub tree entry {path!r} in environment namespace {namespace!r} has an unusable "
                "type; the hub listing could not be read"
            )
        if kind != "blob":
            continue
        if not loader._is_safe_github_path_parts((parts[0],)):
            continue
        slugs.add(f"{namespace}/{parts[0]}")
    return slugs


def _recover_truncated_namespace_slugs(
    ref: object, root_sha: str, namespace: str, found: set[str]
) -> set[str]:
    """Settle the packages a truncated recursive tree left undecided.

    A truncated tree is a PREFIX: what it listed is true, and only the tail is missing. So the
    recovery reads the namespace one level deep -- which cannot truncate, being bounded by the number
    of packages rather than by their contents -- and for each package the prefix did not already
    settle, reads that package's own subtree to look for the marker.

    The first-level read is what keeps the common case cheap: a namespace whose tail is one large
    package pays one extra read, not one per package. A namespace where the prefix settled nothing
    still degrades to a read per package, which is why the marker lookup stays non-recursive -- a
    package's immediate children are all the marker check needs.

    Bounded to `_MAX_TRUNCATION_RECOVERY_READS` packages, because the alternative is worse than the
    truncation it repairs: the caller's timeout is derived from a fixed number of reads, so an
    unbounded fan-out here would blow it and the whole list would fail rather than one namespace's
    tail. Exceeding the bound raises, which reaches the user as the controlled 502 -- a hub too large
    to list this way is a real answer, and it must never be reported as "nothing published".
    """
    from flash.envs import loader

    recovered: set[str] = set()
    reads = 0
    for entry in loader._github_tree_entries(ref, root_sha, namespace, **loader._LIST_READ_BUDGET):
        name = entry.get("path")
        if not isinstance(name, str) or entry.get("type") != "tree":
            continue
        if not loader._is_safe_github_path_parts((name,)):
            continue
        if f"{namespace}/{name}" in found:
            continue
        sha = entry.get("sha")
        if not isinstance(sha, str) or not sha:
            raise RuntimeError(
                f"GitHub tree entry for environment {namespace}/{name!r} has no usable sha; "
                "the hub listing could not be read"
            )
        reads += 1
        if reads > _MAX_TRUNCATION_RECOVERY_READS:
            raise RuntimeError(
                f"environment namespace {namespace!r} needs more than "
                f"{_MAX_TRUNCATION_RECOVERY_READS} additional hub reads to list; "
                "the hub listing could not be read"
            )
        members = loader._github_tree_entries(
            ref, sha, f"{namespace}/{name}", **loader._LIST_READ_BUDGET
        )
        if any(
            member.get("path") == loader._DEFAULT_ENVIRONMENT_PATH and member.get("type") == "blob"
            for member in members
        ):
            recovered.add(f"{namespace}/{name}")
    return recovered
