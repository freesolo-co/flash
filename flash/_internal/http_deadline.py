"""Apply an absolute deadline to every socket read, including HTTP status and headers."""

from __future__ import annotations

import http.client
import io
import time
import urllib.request
from functools import partial


class _DeadlineReader(io.RawIOBase):
    def __init__(self, raw, sock, deadline: float):
        self._raw = raw
        self._sock = sock
        self._deadline = deadline

    def readable(self):
        return True

    def readinto(self, buffer):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("HTTP response exceeded its overall deadline")
        timeout = self._sock.gettimeout()
        self._sock.settimeout(remaining if timeout is None else min(timeout, remaining))
        return self._raw.readinto(buffer)

    def close(self):
        try:
            self._raw.close()
        finally:
            super().close()


class _DeadlineResponse(http.client.HTTPResponse):
    def __init__(self, sock, *args, deadline: float, **kwargs):
        super().__init__(sock, *args, **kwargs)
        # Buffered readline can receive indefinitely while a peer trickles headers. Put
        # the deadline below buffering so every receive rechecks the remaining budget.
        self.fp = io.BufferedReader(_DeadlineReader(self.fp.detach(), sock, deadline))


def _connection(connection_class, deadline: float, *args, **kwargs):
    connection = connection_class(*args, **kwargs)
    connection.response_class = partial(_DeadlineResponse, deadline=deadline)
    return connection


class _DeadlineHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, deadline: float):
        super().__init__()
        self._connection = partial(_connection, http.client.HTTPConnection, deadline)

    def http_open(self, req):
        return self.do_open(self._connection, req)


class _DeadlineHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, deadline: float):
        super().__init__()
        self._connection = partial(_connection, http.client.HTTPSConnection, deadline)

    def https_open(self, req):
        return self.do_open(self._connection, req, context=self._context)


def deadline_handlers(deadline: float) -> tuple:
    return _DeadlineHTTPHandler(deadline), _DeadlineHTTPSHandler(deadline)
