"""The control plane lifts its open-file soft limit before it serves.

On 2026-11-19 the plane ran under the container default of 1024 descriptors, hit EMFILE, and failed
14 runs in seven minutes (12.4 GPU-hours of healthy GRPO among them).
"""

from __future__ import annotations

import types

import pytest

from flash.server.asgi import cli

INFINITY = -1


def _fake_resource(soft, hard, *, refuse=False):
    calls = []

    def setrlimit(which, limits):
        if refuse:
            raise ValueError("not allowed")
        calls.append((which, limits))

    fake = types.SimpleNamespace(
        RLIMIT_NOFILE=7,
        RLIM_INFINITY=INFINITY,
        getrlimit=lambda which: (soft, hard),
        setrlimit=setrlimit,
    )
    return fake, calls


def test_server_main_raises_the_soft_open_file_limit_to_the_hard_limit(monkeypatch):
    fake, calls = _fake_resource(1024, 524288)
    monkeypatch.setattr(cli, "resource", fake, raising=False)
    monkeypatch.setattr(cli, "run_server", lambda host, port: None)

    assert cli.main(["--port", "8080"]) == 0

    assert calls == [(7, (524288, 524288))]


@pytest.mark.parametrize(
    ("soft", "hard", "expected"),
    [
        (524288, 524288, []),
        (INFINITY, INFINITY, []),
        (256, INFINITY, [(7, (1 << 20, INFINITY))]),
    ],
)
def test_open_file_limit_is_left_alone_when_already_high(monkeypatch, soft, hard, expected):
    fake, calls = _fake_resource(soft, hard)
    monkeypatch.setattr(cli, "resource", fake, raising=False)

    cli._raise_open_file_limit()

    assert calls == expected


def test_a_refused_open_file_limit_does_not_stop_the_server(monkeypatch):
    fake, _ = _fake_resource(1024, 4096, refuse=True)
    monkeypatch.setattr(cli, "resource", fake, raising=False)
    served = []
    monkeypatch.setattr(cli, "run_server", lambda host, port: served.append(port))

    assert cli.main(["--port", "8080"]) == 0
    assert served == [8080]
