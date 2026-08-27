"""The owner-scoped run listings bound their per-request work without truncating the answer.

`/v1/runs` and `/v1/deployments` both answer "every run this key owns" and both open one status
file per row, so before this bound an owner's request cost grew with their whole run history: the
server walked every row and read every status file before writing a byte. The invariant proved here
is the pair of halves that makes bounding safe -- the server reads at most one page per request, and
the client follows the cursors so no caller ever mistakes a page for the whole listing.

That second half is load-bearing, not decorative: `cmd_deployments`' rollback lookup and
`_live_deployment` both scan the listing for a run id, and a run sitting on page two would read as
absent if a page were taken as the complete answer.
"""

from __future__ import annotations

import contextlib
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi import HTTPException

from flash.client.http import ApiClient, ClientError
from flash.server.platform import db, listing
from tests._helpers.server_api_plugin import api  # noqa: F401
from tests.test_server_api import _bearer, _login

# ---------------------------------------------------------------------------
# the database keyset
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "state" / "server.db"))
    return db


def _owner(store, token: str = "fslo_pager") -> int:
    row = store.ensure_external_key(token)
    assert row is not None
    return int(row["id"])


def _record_runs(store, key_id: int, count: int) -> list[str]:
    """Record `count` runs and return their ids in creation order.

    Ids ascend with creation, so they pin the expected order whether or not the wall clock ticks
    between inserts: distinct timestamps order by time, and ties order by run id, which agrees.
    """
    ids = [f"run-{i:04d}" for i in range(count)]
    for run_id in ids:
        store.record_run(run_id, key_id)
    return ids


def test_a_page_of_runs_sharing_one_timestamp_neither_repeats_nor_skips(isolated_db, monkeypatch):
    """`created_at` is not unique, so it cannot be the cursor on its own.

    Runs recorded inside the same clock tick share a timestamp exactly. Paging on that column alone
    forces a choice between `>` (skips every tied row after the first) and `>=` (returns the same
    row again forever). The cursor is the `(created_at, run_id)` pair, which the primary key makes
    unique, so a tie resolves by run id and the walk advances exactly one row at a time.
    """
    key_id = _owner(isolated_db)
    monkeypatch.setattr(isolated_db.time, "time", lambda: 1000.0)
    expected = _record_runs(isolated_db, key_id, 7)

    assert len({r["created_at"] for r in isolated_db.runs_for_key(key_id)}) == 1

    walked: list[str] = []
    after: tuple[float, str] | None = None
    for _ in range(len(expected) + 2):
        page = isolated_db.runs_for_key(key_id, after=after, limit=2)
        if not page:
            break
        walked.extend(row["run_id"] for row in page)
        after = (float(page[-1]["created_at"]), page[-1]["run_id"])

    assert walked == expected, "tied timestamps must page exactly once each, in run-id order"


def test_the_unpaged_call_is_unchanged_for_every_existing_caller(isolated_db, monkeypatch):
    """The keyset is opt-in. A caller that passes neither argument still gets the whole listing in
    creation order, which is what `runs_for_key` has always returned and what its ownership test
    pins."""
    key_id = _owner(isolated_db)
    stamps = iter([300.0, 100.0, 200.0])
    monkeypatch.setattr(isolated_db.time, "time", lambda: next(stamps))
    isolated_db.record_run("run-late", key_id)
    isolated_db.record_run("run-early", key_id)
    isolated_db.record_run("run-mid", key_id)

    assert [r["run_id"] for r in isolated_db.runs_for_key(key_id)] == [
        "run-early",
        "run-mid",
        "run-late",
    ]


def test_the_keyset_never_crosses_owners(isolated_db, monkeypatch):
    """The bound is a page of ONE owner's runs. A cursor is not a capability: resuming another
    owner's key with it must still return only rows that key owns."""
    mine = _owner(isolated_db, "fslo_mine")
    theirs = _owner(isolated_db, "fslo_theirs")
    monkeypatch.setattr(isolated_db.time, "time", lambda: 500.0)
    isolated_db.record_run("run-a", mine)
    isolated_db.record_run("run-b", theirs)
    isolated_db.record_run("run-c", mine)

    page = isolated_db.runs_for_key(mine, after=(500.0, "run-a"), limit=10)
    assert [r["run_id"] for r in page] == ["run-c"]


# ---------------------------------------------------------------------------
# the page bound itself
# ---------------------------------------------------------------------------


def test_a_full_page_reports_a_cursor_and_a_short_page_does_not(isolated_db):
    key_id = _owner(isolated_db)
    ids = _record_runs(isolated_db, key_id, listing.PAGE_SIZE + 5)

    first, cursor = listing.page_rows(key_id, None)
    assert [r["run_id"] for r in first] == ids[: listing.PAGE_SIZE]
    assert cursor is not None

    second, done = listing.page_rows(key_id, cursor)
    assert [r["run_id"] for r in second] == ids[listing.PAGE_SIZE :]
    assert done is None


def test_a_listing_that_ends_exactly_on_the_page_boundary_reports_no_cursor(isolated_db):
    """Asking for one row beyond the page is what separates "full page, more to come" from "the
    page happens to end at the last row". Without that probe every owner whose run count is an exact
    multiple of the page size would be handed a cursor resolving to nothing, and pay one empty round
    trip to discover it."""
    key_id = _owner(isolated_db)
    ids = _record_runs(isolated_db, key_id, listing.PAGE_SIZE)

    rows, cursor = listing.page_rows(key_id, None)
    assert [r["run_id"] for r in rows] == ids
    assert cursor is None


def test_a_malformed_cursor_is_a_client_error_not_a_silent_restart():
    """A cursor arrives from the network. Ignoring an unparseable one would restart the listing at
    the beginning, and a paging client would loop over page one forever instead of failing."""
    for bad in ["", "not-a-float:run-1", "1000.0", "1000.0:", ":run-1"]:
        with pytest.raises(HTTPException) as caught:
            listing.decode_cursor(bad)
        assert caught.value.status_code == 400

    assert listing.decode_cursor(None) is None
    assert listing.decode_cursor("1000.5:run-1") == (1000.5, "run-1")


def test_the_cursor_round_trips_a_timestamp_without_losing_precision(isolated_db, monkeypatch):
    """`created_at` is a REAL. A cursor that rounded it would land between two rows: either
    re-serving the row it names or stepping over the one after it."""
    key_id = _owner(isolated_db)
    monkeypatch.setattr(isolated_db.time, "time", lambda: 1739827200.1234567)
    isolated_db.record_run("run-precise", key_id)

    row = isolated_db.runs_for_key(key_id)[0]
    assert listing.decode_cursor(listing.encode_cursor(row)) == (row["created_at"], "run-precise")


def test_a_missing_status_file_skips_its_row_instead_of_failing_the_page():
    """A run whose status file is gone has nothing to report. One of them must not fail the whole
    listing for every other run the owner has."""

    def load(run_id: str) -> str:
        if run_id == "gone":
            raise FileNotFoundError(run_id)
        return run_id

    rows = [{"run_id": "here"}, {"run_id": "gone"}, {"run_id": "also-here"}]
    assert list(listing.statuses(rows, load)) == ["here", "also-here"]


# ---------------------------------------------------------------------------
# the routes
# ---------------------------------------------------------------------------


def _key_id(token: str) -> int:
    row = db.ensure_external_key(token)
    assert row is not None
    return int(row["id"])


@contextlib.contextmanager
def _counted_statuses(monkeypatch):
    """Count how many status files a single request opens.

    This is the invariant itself: the cost of one request is the page, not the history. The status
    objects are irrelevant here, so every load raises `FileNotFoundError` and the route yields an
    empty page -- the count is what is being asserted.
    """
    import flash.server.asgi.app as app_mod

    opened: list[str] = []

    def counting_get_status(run_id: str):
        opened.append(run_id)
        raise FileNotFoundError(run_id)

    monkeypatch.setattr(app_mod, "get_status", counting_get_status)
    yield opened


@contextlib.contextmanager
def _counted_rows(monkeypatch):
    """Count the rows the database actually hands back per call.

    Status opens alone do not prove the request is bounded: a route that reads every row and then
    slices to a page still opens only a page of status files, while the query behind it walks the
    owner's whole history. The row count is the half of the cost that only the query can bound.
    """
    real = db.runs_for_key
    counts: list[int] = []

    def counting_runs_for_key(*args, **kwargs):
        rows = real(*args, **kwargs)
        counts.append(len(rows))
        return rows

    monkeypatch.setattr(listing.db, "runs_for_key", counting_runs_for_key)
    yield counts


@pytest.mark.parametrize("path", ["/v1/runs", "/v1/deployments"])
def test_one_request_opens_at_most_one_page_of_status_files(api, monkeypatch, path):  # noqa: F811
    """Both listings walked every row and read every status file before answering. An owner with a
    long history therefore made every one of their own listing requests progressively more
    expensive, with no bound the server could enforce."""
    token = _login()
    _record_runs(db, _key_id(token), listing.PAGE_SIZE + 25)

    with _counted_rows(monkeypatch) as rows, _counted_statuses(monkeypatch) as opened:
        response = api.get(path, headers=_bearer(token))

    assert response.status_code == 200
    assert len(opened) == listing.PAGE_SIZE, "a request must not read past its page"
    # the query itself is what bounds the work. slicing an unbounded result down to a page would
    # satisfy the status count above while still walking the owner's entire history in sqlite.
    assert rows == [listing.PAGE_SIZE + 1], "the query must ask for one page, plus the probe row"
    assert response.json()["next_cursor"] is not None


@pytest.mark.parametrize(
    ("path", "field"), [("/v1/runs", "runs"), ("/v1/deployments", "deployments")]
)
def test_following_the_cursor_reaches_every_run_exactly_once(api, monkeypatch, path, field):  # noqa: F811
    """The bound is on work per request, not on the answer. Walking the cursors must visit every
    owned run once: `cmd_deployments`' rollback lookup and `_live_deployment` match a run id against
    the listing, and a missed run reads to them as a run that does not exist."""
    token = _login()
    expected = _record_runs(db, _key_id(token), listing.PAGE_SIZE * 2 + 3)

    with _counted_statuses(monkeypatch) as opened:
        cursor = None
        pages = 0
        while True:
            query = f"?cursor={urllib.parse.quote(cursor, safe='')}" if cursor else ""
            body = api.get(f"{path}{query}", headers=_bearer(token)).json()
            assert body[field] == []
            pages += 1
            cursor = body["next_cursor"]
            if not cursor:
                break
            assert pages < 10, "the walk must terminate"

    assert opened == expected, "every owned run is visited exactly once, in creation order"
    assert pages == 3


def test_a_malformed_cursor_is_rejected_by_the_route(api):  # noqa: F811
    token = _login()
    assert api.get("/v1/runs?cursor=garbage", headers=_bearer(token)).status_code == 400


def test_an_owner_with_no_runs_still_gets_a_complete_first_page(api):  # noqa: F811
    body = api.get("/v1/runs", headers=_bearer(_login())).json()
    assert body["runs"] == []
    assert body["next_cursor"] is None


# ---------------------------------------------------------------------------
# the client walk
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _server(handler_body):
    """A local HTTP server answering every GET with `handler_body(path) -> bytes`."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = handler_body(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _cursor_of(path: str) -> str | None:
    return urllib.parse.parse_qs(urllib.parse.urlsplit(path).query).get("cursor", [None])[0]


def _pager(field: str, pages: list[tuple[list[dict], str | None]]):
    """Answer each cursor with the page that follows it."""
    by_cursor = {None: pages[0]}
    for index, (_items, cursor) in enumerate(pages[:-1]):
        by_cursor[cursor] = pages[index + 1]

    def body(path: str) -> bytes:
        items, next_cursor = by_cursor[_cursor_of(path)]
        return json.dumps({field: items, "next_cursor": next_cursor}).encode()

    return body


@pytest.mark.parametrize(
    ("field", "call"),
    [("runs", lambda c: c.list_runs()), ("deployments", lambda c: c.deployments())],
)
def test_the_client_reassembles_every_page_in_server_order(field, call):
    """Pagination is a transport detail the CLI never sees. `cmd_runs` and `cmd_deployments` render
    whatever the client returns, so the client owes them the complete listing."""
    pages = [
        ([{"run_id": "a"}, {"run_id": "b"}], "1.0:b"),
        ([{"run_id": "c"}], "2.0:c"),
        ([{"run_id": "d"}], None),
    ]
    with _server(_pager(field, pages)) as url:
        got = call(ApiClient(url, "fslo-user-test", timeout=5))
    assert [item["run_id"] for item in got] == ["a", "b", "c", "d"]


def test_a_single_page_listing_makes_one_request():
    """The bound must not cost an extra round trip in the common case, which is one page."""
    seen: list[str] = []

    def body(path: str) -> bytes:
        seen.append(path)
        return json.dumps({"runs": [{"run_id": "only"}], "next_cursor": None}).encode()

    with _server(body) as url:
        assert ApiClient(url, "fslo-user-test", timeout=5).list_runs() == [{"run_id": "only"}]

    assert seen == ["/v1/runs"], "no cursor query and no second request"


def test_a_repeating_cursor_fails_instead_of_paging_forever():
    """A server that keeps handing back the same cursor cannot produce new items. Following it would
    hang the CLI with no output and no error, which is strictly worse than failing."""
    pages = [([{"run_id": "a"}], "1.0:a"), ([{"run_id": "a"}], "1.0:a")]
    with _server(_pager("runs", pages)) as url, pytest.raises(ClientError) as caught:
        ApiClient(url, "fslo-user-test", timeout=5).list_runs()
    assert "repeating cursor" in str(caught.value)


def test_a_cursor_with_url_significant_characters_survives_the_round_trip():
    """The cursor is a server-chosen opaque string carried in a query. Quoting it is what keeps a
    `+` from arriving as a space, or an `&` from splitting into a second parameter."""
    seen: list[str | None] = []
    cursor = "1.0e+09:run-a&b"

    def body(path: str) -> bytes:
        got = _cursor_of(path)
        seen.append(got)
        return json.dumps({"runs": [], "next_cursor": None if got else cursor}).encode()

    with _server(body) as url:
        ApiClient(url, "fslo-user-test", timeout=5).list_runs()

    assert seen == [None, cursor], "the server must read back the exact cursor it issued"
