"""Log offsets count bytes and tolerate worker output that is not valid UTF-8."""

from pathlib import Path

import pytest

from flash.runner.lifecycle.state import runs_file_path
from tests.test_server_api import _bearer, _make_run, _new_user_token
from tests.test_server_api import api as api


@pytest.mark.parametrize(
    ("content", "offset", "expected"),
    [("é\n".encode(), 1, "�\n"), (b"worker: \xff\n", 0, "worker: �\n")],
)
def test_log_reads_return_byte_offsets_for_non_utf8_slices(api, content, offset, expected):
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    Path(runs_file_path(run_id, ".log")).write_bytes(content)
    response = api.get(f"/v1/runs/{run_id}/logs?offset={offset}", headers=_bearer(key))
    assert response.status_code == 200
    assert response.json()["logs"] == expected
    assert response.json()["offset"] == len(content)


def test_unrepresentable_log_offset_is_a_request_error(api):
    key = _new_user_token()
    run_id = _make_run(api, key, "done")
    Path(runs_file_path(run_id, ".log")).write_text("worker output")
    response = api.get(f"/v1/runs/{run_id}/logs?offset={10**100}", headers=_bearer(key))
    assert response.status_code == 400
    assert "invalid log offset" in response.json()["detail"]
