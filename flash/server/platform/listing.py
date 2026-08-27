"""Bounded keyset paging for the owner-scoped run listings.

`/v1/runs` and `/v1/deployments` both answer "every run this key owns", and both pay one status
file read per row. Without a bound, an owner's request cost grows with their run history: the
server walks every row and opens every status file before writing a byte of response. That is the
whole reason this module exists, so the page bound lives here rather than in either route.

Paging is keyset, not offset: the cursor names the exact `(created_at, run_id)` the previous page
ended on, so runs recorded or deleted between requests cannot make the walk repeat or skip rows the
way a row offset would. Ordering stays ascending by creation, which is what `runs_for_key` has
always returned.

The listings themselves stay complete. Callers match a run id against the whole listing, so a page
is a transport bound, not a truncation: the route reports the next cursor and the client follows it
until there is none left.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from fastapi import HTTPException

from flash.server.platform import db

# One page is one bounded unit of work: at most this many rows walked and status files opened per
# request. Large enough that the overwhelming majority of owners are served in a single round trip,
# small enough that no single request can be made arbitrarily expensive by run history.
PAGE_SIZE = 200

_CURSOR_SEPARATOR = ":"


def encode_cursor(row: dict) -> str:
    """Name a row as the resumption point for the page after it.

    Run ids cannot contain `:` (`_RUN_ID_RE` admits alphanumerics, `.`, `_`, and `-`), so splitting
    on the first separator recovers both halves unambiguously.
    """
    return f"{float(row['created_at'])!r}{_CURSOR_SEPARATOR}{row['run_id']}"


def decode_cursor(cursor: str | None) -> tuple[float, str] | None:
    """Parse a caller-supplied cursor, rejecting anything this server did not produce.

    The cursor arrives from the network, so a malformed value is a client error (400) rather than an
    unhandled exception or a silent restart from the beginning of the listing, which would make a
    paging client loop over the first page forever.
    """
    if cursor is None:
        return None
    created_at, separator, run_id = cursor.partition(_CURSOR_SEPARATOR)
    if not separator or not run_id:
        raise HTTPException(status_code=400, detail="malformed cursor")
    try:
        parsed = float(created_at)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="malformed cursor") from exc
    return parsed, run_id


def page_rows(key_id: int, cursor: str | None) -> tuple[list[dict], str | None]:
    """One bounded page of an owner's runs, plus the cursor for the page after it.

    Asking for one row beyond the page is what distinguishes "the page is full and more rows exist"
    from "the page happens to end exactly at the last row". Without it a listing whose length is an
    exact multiple of the page size would report a next cursor that resolves to nothing, costing
    every such caller an extra empty round trip.
    """
    rows = db.runs_for_key(key_id, after=decode_cursor(cursor), limit=PAGE_SIZE + 1)
    if len(rows) <= PAGE_SIZE:
        return rows, None
    page = rows[:PAGE_SIZE]
    return page, encode_cursor(page[-1])


def statuses(rows: list[dict], load: Callable[[str], Any]) -> Iterator[Any]:
    """Load each row's status, skipping runs whose status file is gone.

    A record whose file has been removed is not an error for a listing: it is a run that no longer
    has anything to report, and one of them must not fail the whole page.
    """
    for row in rows:
        try:
            yield load(row["run_id"])
        except FileNotFoundError:
            continue
