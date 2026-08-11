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

import time

# how many per-package reads the truncation recovery may add. A count bound as well as the wall-clock
# one below, because the two limit different things: the deadline stops a slow hub, this stops a hub
# large enough that even fast reads cannot enumerate it -- which must be reported, not answered short.
_MAX_TRUNCATION_RECOVERY_READS = 8

# the whole listing's wall-clock budget, shared across every read it makes. This is what lets the
# truncation recovery issue more than two reads without the client giving up first: each read is
# capped at the remainder rather than at a fixed per-read timeout, so a fan-out makes the individual
# reads shorter instead of making the request longer.
#
# Deliberately equal to the client's derived ceiling MINUS its safety margin
# (`http.ENV_LIST_SERVER_BUDGET_SECONDS`), duplicated as arithmetic rather than imported because
# `flash.client` must stay importable without the server extra and this module is the server side. The
# invariant that matters is one-directional and stated here so a future edit can check it: the server
# must give up FIRST, so that a real upstream failure reaches the user as the controlled 429/502
# rather than as a local timeout that says nothing about why.
_LIST_READS = 2
_LIST_ATTEMPTS_PER_READ = 2
_LIST_SOCKET_TIMEOUT_SECONDS = 20.0
_LIST_BODY_DEADLINE_SECONDS = 20.0
_LIST_MAX_BACKOFF_PER_READ_SECONDS = 45.0
_LISTING_DEADLINE_SECONDS = _LIST_READS * (
    _LIST_ATTEMPTS_PER_READ * (_LIST_SOCKET_TIMEOUT_SECONDS + _LIST_BODY_DEADLINE_SECONDS)
    + _LIST_MAX_BACKOFF_PER_READ_SECONDS
)


def _remaining_budget(deadline: float, namespace: str) -> dict:
    """The read budget for the next read, narrowed to what is left of the listing's deadline.

    Raises rather than issuing a doomed read once the budget is spent: a read started with no time
    left would either block for its own full timeout (defeating the deadline) or fail confusingly.
    """
    from flash.envs import loader

    left = deadline - time.monotonic()
    if left <= 0:
        raise RuntimeError(
            f"listing environment namespace {namespace!r} exceeded its overall deadline; "
            "the hub listing could not be read"
        )
    budget = dict(loader._LIST_READ_BUDGET)
    budget["timeout"] = min(budget["timeout"], left)
    budget["body_deadline"] = min(budget["body_deadline"], left)
    return budget


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

    deadline = time.monotonic() + _LISTING_DEADLINE_SECONDS
    root_sha = _namespace_root_sha(ref, namespace, deadline)
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
        ref, root_sha, namespace, recursive=True, **_remaining_budget(deadline, namespace)
    )
    slugs = _slugs_from_recursive_entries(entries, namespace)
    if truncated:
        slugs.update(_recover_truncated_namespace_slugs(ref, root_sha, namespace, slugs, deadline))
    return sorted(slugs)


def _namespace_root_sha(ref: object, namespace: str, deadline: float) -> str | None:
    """The tree sha of the namespace directory, or ``None`` when the org published nothing yet."""
    from flash.envs import loader

    # match on the PATH only, then validate type and sha below. folding either check into this
    # predicate would make a malformed entry look like a MISSING one, so a broken hub response would
    # report "nothing published" -- the exact failure this endpoint exists to remove.
    root = next(
        (
            entry
            for entry in loader._github_tree_entries(
                ref, ref.ref, ref.ref, **_remaining_budget(deadline, namespace)
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
    ref: object, root_sha: str, namespace: str, found: set[str], deadline: float
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
    for entry in loader._github_tree_entries(
        ref, root_sha, namespace, **_remaining_budget(deadline, namespace)
    ):
        name = entry.get("path")
        if not isinstance(name, str):
            continue
        if not loader._is_safe_github_path_parts((name,)):
            continue
        # validate the type only AFTER the path matched, exactly as the prefix path does. Folding the
        # two into one predicate made a malformed type look like an absent one, so an unreadable hub
        # response silently omitted a published environment instead of reaching the controlled 502.
        kind = entry.get("type")
        if not isinstance(kind, str) or not kind:
            raise RuntimeError(
                f"GitHub tree entry for environment {namespace}/{name!r} has an unusable type; "
                "the hub listing could not be read"
            )
        # a well-formed non-tree is a readable answer: publish writes a directory, so a stray file at
        # this path is unambiguously not an environment.
        if kind != "tree":
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
            ref, sha, f"{namespace}/{name}", **_remaining_budget(deadline, namespace)
        )
        if _package_has_marker(members, namespace, name):
            recovered.add(f"{namespace}/{name}")
    return recovered


def _package_has_marker(members: list[dict], namespace: str, name: str) -> bool:
    """Whether one package's immediate children hold the ``environment.py`` blob publish writes.

    Matches on the path first and only then validates the type, for the same reason the prefix path
    does: a combined predicate makes an unusable type indistinguishable from an absent marker, so a
    broken hub response omits a published environment rather than reaching the controlled 502.
    """
    from flash.envs import loader

    for member in members:
        if member.get("path") != loader._DEFAULT_ENVIRONMENT_PATH:
            continue
        kind = member.get("type")
        if not isinstance(kind, str) or not kind:
            raise RuntimeError(
                f"GitHub tree entry {loader._DEFAULT_ENVIRONMENT_PATH!r} in environment "
                f"{namespace}/{name!r} has an unusable type; the hub listing could not be read"
            )
        # a directory that happens to be named environment.py is a legitimate skip, not a fault, so
        # keep scanning rather than deciding on the first path match.
        if kind == "blob":
            return True
    return False
